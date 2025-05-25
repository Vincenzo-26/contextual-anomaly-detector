import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import ruptures as rpt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from src.utils import *
from sklearn.mixture import GaussianMixture
import warnings
from scipy.stats import ConstantInputWarning

def scale_data(X, method="zscore"):
    method = method.lower()
    if method == "zscore":
        return StandardScaler().fit_transform(X)
    elif method == "minmax":
        return MinMaxScaler().fit_transform(X)
    elif method == "robust":
        return RobustScaler().fit_transform(X)
    elif method == "maxabs":
        return MaxAbsScaler().fit_transform(X)
    elif method == "none":
        return X
    else:
        raise ValueError(f"Metodo di normalizzazione non supportato: '{method}'")


def check_thermal_sensitivity(df_segment, normalize="Zscore", corr_thresh=0.5, r2_thresh=0.5, slope_thresh=0.2):
    """
    Verifica se un segmento è termicamente sensibile usando correlazione, pendenza e R².

    Args:
        df_segment (pd.DataFrame): Segmento con colonne 'Temperature' e 'Energy'.
        normalize (str or bool): Metodo di normalizzazione ('Zscore', 'minmax', 'robust', 'maxabs', 'none' o False).
        corr_thresh (float): Soglia di correlazione.
        r2_thresh (float): Soglia R².
        slope_thresh (float): Soglia della pendenza.

    Returns:
        dict: Metriche calcolate e flag 'is_thermal_sensitive'.
    """
    X_raw = df_segment[["Temperature"]]
    y_raw = df_segment["Energy"].values.reshape(-1, 1)

    if len(X_raw) < 2:
        return {
            "correlation": np.nan,
            "slope": np.nan,
            "r2_score": np.nan,
            "is_thermal_sensitive": False,
            "scaling_used": normalize,
            "corr_thresh": corr_thresh,
            "slope_thresh": slope_thresh,
            "r2_thresh": r2_thresh
        }

    # Correlazione di Spearman
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        try:
            corr, _ = spearmanr(X_raw.values.flatten(), y_raw.flatten())
        except Exception:
            print("⚠️ Segmento con input costante: impossibile calcolare la correlazione.")
            corr = np.nan

    # Determina tipo di normalizzazione
    method = str(normalize).lower() if normalize not in [False, None] else "none"

    # Applica normalizzazione a X e y
    X = scale_data(X_raw, method)
    y = scale_data(y_raw, method).flatten()

    # Regressione
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    r2 = r2_score(y, model.predict(X))

    is_sensitive = (abs(corr) > corr_thresh) and (abs(slope) > slope_thresh) and (r2 > r2_thresh)

    return {
        "correlation": round(corr, 3),
        "slope": round(slope, 3),
        "r2_score": round(r2, 3),
        "is_thermal_sensitive": is_sensitive,
        "scaling_used": method,
        "corr_thresh": corr_thresh,
        "slope_thresh": slope_thresh,
        "r2_thresh": r2_thresh
    }


def operational_modes_detection(df: pd.DataFrame, bic_threshold: float):
    """
    Identifica le modalità operative di un sistema basandosi sui dati di Temperatura ed Energia,
    utilizzando Gaussian Mixture Models (GMM) con fino a 3 componenti.

    La funzione seleziona il miglior numero di componenti GMM sulla base del miglioramento relativo del BIC,
    **a condizione che tutti i cluster individuati abbiano almeno 5 punti** per non rompere il processo di
    change point detection successivo. Se un GMM produce anche solo un cluster
    con meno di 5 punti, viene scartato, indipendentemente dal miglioramento del BIC, in favore del modello con
    numero di componenti inferiore.

    Args:
        df (pd.DataFrame): DataFrame contenente almeno le colonne 'Temperature' e 'Energy', riferite a dati normal.
        bic_threshold (float): Soglia minima per il miglioramento relativo del BIC per accettare un numero di componenti maggiore.

    Returns:
        Tuple:
            - df (pd.DataFrame): DataFrame con una nuova colonna 'Mode' che assegna a ogni punto la modalità operativa identificata.
            - best_gmm (GaussianMixture or None): Il modello GMM selezionato, oppure None se non valido.
            - scaler (StandardScaler): Lo scaler usato per normalizzare i dati, utile per trasformare nuovi input.
            - (Δ1→2, Δ2→3) (Tuple[float or None, float or None]): Miglioramenti relativi di BIC tra 1→2 e 2→3 componenti.
            - rollback_flag (bool): True se almeno un modello con più componenti è stato scartato per via di cluster troppo piccoli.
    """
    df = df.dropna(subset=["Temperature", "Energy"])
    if df.empty or len(df) < 5:
        df["Mode"] = 0
        return df, None, None, (None, None), False

    x = df[["Temperature", "Energy"]].values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    def fit_gmm(x_scaled, n_components):
        """
        - Fitta un GMM con il numero 'n_components' di componenti.
        - Controlla che tutti i cluster abbiano almeno 5 punti.
            - Se sì → restituisce il modello, i label di appartenenza dei dati, BIC, e False (nessun cluster piccolo).
            - Se no → scarta il modello, ritorna None e True (c’erano cluster piccoli).
        """
        gmm = GaussianMixture(n_components=n_components, random_state=0).fit(x_scaled)
        labels = gmm.predict(x_scaled)
        counts = pd.Series(labels).value_counts().sort_index()

        if (counts >= 5).all():
            return gmm, labels, gmm.bic(x_scaled), False
        else:
            return None, None, None, True

    gmm1, labels1, bic1, flag1 = fit_gmm(x_scaled, 1)
    gmm2, labels2, bic2, flag2 = fit_gmm(x_scaled, 2)
    gmm3, labels3, bic3, flag3 = fit_gmm(x_scaled, 3)

    if gmm3 and not flag3 and bic2 and ((bic2 - bic3) / abs(bic2)) > bic_threshold:
        df["Mode"] = labels3
        return df, gmm3, scaler, ((bic1 - bic2) / abs(bic1), (bic2 - bic3) / abs(bic2)), False
    elif gmm2 and not flag2 and bic1 and ((bic1 - bic2) / abs(bic1)) > bic_threshold:
        df["Mode"] = labels2
        return df, gmm2, scaler, ((bic1 - bic2) / abs(bic1), None), False
    elif gmm1:
        df["Mode"] = labels1
        return df, gmm1, scaler, (None, None), flag2 or flag3
    else:
        df["Mode"] = 0
        return df, None, scaler, (None, None), True



def run_change_point(case_study: str, penalty: int, bic_threshold:float = 0.2 ,norm_for_check_term_sens: any = (True, "zscore"), scaling_method_for_change_point: str = "zscore"):
    """
    Esegue l'analisi di segmentazione per ogni foglia del caso studio. Per ciascun segmento rilevato tramite
    change point detection, verifica se è termicamente sensibile e, se sì, calcola i residui e la probabilità
    di anomalia per i punti (normali e anomali) che ricadono in tali segmenti.

    Args:
        case_study (str): Nome del caso studio da analizzare.
        penalty (int): Penalità da usare nell'algoritmo di change point.

    Returns:
        None
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)


    foglie = find_leaf_nodes(config["Load Tree"])
    print_boxed_title(f"Thermal sensitivity analysis for '{case_study}'🌡️")

    print(f"Normalization method for change point detection: {scaling_method_for_change_point}")

    if norm_for_check_term_sens is False:
        thermal_scaling = "none"
        print(f"Data not normalized for checking thermal sensitivity")
    elif norm_for_check_term_sens is True:
        thermal_scaling = "Zscore"
        print(f"Normalized data for checking thermal sensitivity    -> Method: {thermal_scaling} (default)\n")
    elif isinstance(norm_for_check_term_sens, (tuple, list)) and norm_for_check_term_sens[0] is True:
        thermal_scaling = norm_for_check_term_sens[1]
        print(f"Normalized data for checking thermal sensitivity    -> Method: {thermal_scaling}\n")
    else:
        raise ValueError(f"Parametro non valido per norm_for_check_term_sens: {norm_for_check_term_sens}")

    for foglia in foglie:
        print(f"\033[91m{foglia}\033[0m")
        segment_results = []
        models_info = []
        segment_map = {}

        groups_path = os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv")
        groups = pd.read_csv(groups_path, parse_dates=["timestamp"])
        groups["date"] = groups["timestamp"].dt.date
        context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
        cluster_cols = [col for col in groups.columns if col.startswith("Cluster_")]
        all_rows = []

        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                df_normals, df_anomalies = run_energy_temp(case_study, foglia, context, cluster)

                if df_normals is None or df_normals.empty:
                    continue

                df_normals_sorted = df_normals.sort_values("Temperature").dropna(subset=["Temperature", "Energy"])
                # df_normals_sorted = df_normals_sorted[df_normals_sorted["Energy"] > 0]

                df_modes, best_gmm, scaler, (imp12, imp23), flag = operational_modes_detection(df_normals_sorted, bic_threshold)
                for mode in sorted(df_modes["Mode"].unique()):
                    df_mode = df_modes[df_modes["Mode"] == mode]

                    X_check = df_mode[["Temperature", "Energy"]]
                    if X_check.nunique().min() < 2:
                        # Valori costanti → un solo segmento
                        segmenti = [(0, len(df_mode))]
                        manual_override = True
                    else:
                        signal = X_check.values
                        signal_scaled = scale_data(signal, method=scaling_method_for_change_point)
                        algo = rpt.Pelt(model="rank").fit(signal_scaled)
                        change_points = algo.predict(pen=penalty)
                        segmenti = [(start, end) for start, end in zip([0] + change_points[:-1], change_points)]
                        manual_override = False

                    segments_in_mode = []
                    for i, (start, end) in enumerate(segmenti):
                        segment = df_mode.iloc[start:end]
                        t_min = segment["Temperature"].min()
                        t_max = segment["Temperature"].max()
                        metrics = check_thermal_sensitivity(segment, normalize=thermal_scaling)

                        segment_results.append({
                            "Context": context,
                            "Cluster": cluster,
                            "Mode": mode,
                            "Segmento": i + 1,
                            "t_min": t_min,
                            "t_max": t_max,
                            "Thermal Sensitive": metrics["is_thermal_sensitive"],
                            "correlation": metrics["correlation"],
                            "slope": metrics["slope"],
                            "r2_score": metrics["r2_score"],
                            "corr_thresh": metrics["corr_thresh"],
                            "slope_thresh": metrics["slope_thresh"],
                            "r2_thresh": metrics["r2_thresh"]
                        })

                        segments_in_mode.append({
                            "Segmento": i + 1, "t_min": t_min, "t_max": t_max,
                            "Thermal Sensitive": metrics["is_thermal_sensitive"]
                        })

                        if metrics["is_thermal_sensitive"]:
                            model = LinearRegression().fit(segment[["Temperature"]], segment["Energy"])
                            sigma = mean_squared_error(segment["Energy"], model.predict(segment[["Temperature"]])) ** 0.5
                            models_info.append({
                                "Context": context, "Cluster": cluster, "Mode": mode, "Segmento": i + 1,
                                "model": model, "t_min": t_min, "t_max": t_max, "sigma": sigma
                            })
                        start = end

                    segment_map[(context, cluster, mode)] = segments_in_mode

                fallback = "↩️ " if flag else ""

                if imp12 is not None and imp23 is not None:
                    # GMM(3) selezionato
                    color_12 = "\033[92m"  # verde
                    color_23 = "\033[92m"  # verde
                elif imp12 is not None and imp23 is None:
                    # GMM(2) selezionato
                    color_12 = "\033[92m"  # verde
                    color_23 = "\033[91m"  # rosso
                else:
                    # GMM(1) selezionato
                    color_12 = "\033[91m"
                    color_23 = "\033[91m"

                imp12_str = f"{color_12}{imp12:.1%}\033[0m" if imp12 is not None else "\033[91mN/A\033[0m"
                imp23_str = f"{color_23}{imp23:.1%}\033[0m" if imp23 is not None else "\033[91mN/A\033[0m"

                summary = (
                    f"[Ctx {context} | Clst {cluster}]      "
                    f"({fallback}Δ1→2: {imp12_str} → Δ2→3: {imp23_str})  -> "
                )
                mode_summaries = []

                for m in sorted(df_modes["Mode"].unique()):
                    total_segments = sum(1 for seg in segment_results
                                         if seg["Context"] == context and seg["Cluster"] == cluster and seg["Mode"] == m)
                    thermal_segments = sum(1 for seg in segment_results
                                           if seg["Context"] == context and seg["Cluster"] == cluster and
                                           seg["Mode"] == m and seg["Thermal Sensitive"])
                    if thermal_segments > 0:
                        ts_text = f"\033[92m({thermal_segments}/{total_segments} Thermal sensitive)\033[0m"
                    else:
                        ts_text = f"({thermal_segments}/{total_segments} Thermal sensitive)"

                    mode_summaries.append(f"{' ⚠️' if manual_override else ''}Mode {m+1} - {total_segments} segments {ts_text}")

                summary += f"{len(mode_summaries)} operational modes: " + " | ".join(mode_summaries)

                print(summary)

                # Predict mode for all points (normals + anomalies)
                df_all = pd.concat([df_normals_sorted, df_anomalies]) if df_anomalies is not None else df_normals_sorted.copy()
                df_all = df_all.reset_index().rename(columns={"index": "Date"})
                df_all["Context"] = context
                df_all["Cluster"] = cluster
                df_all["Mode"] = best_gmm.predict(scaler.transform(df_all[["Temperature", "Energy"]].values))
                df_all["is_real_anomaly"] = df_all["Date"].isin(df_anomalies.index) if df_anomalies is not None else False
                all_rows.append(df_all)

        output_segments = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "segments")
        os.makedirs(output_segments, exist_ok=True)
        pd.DataFrame(segment_results).to_csv(os.path.join(output_segments, f"segment_{foglia}.csv"), index=False)

        if all_rows and models_info:
            df_final = pd.concat(all_rows)
            residuals, probs, modes, segments = [], [], [], []

            for row in df_final.itertuples():
                key = (row.Context, row.Cluster, row.Mode)
                segments_in_mode = segment_map.get(key, [])
                if not segments_in_mode:
                    residuals.append(np.nan)
                    probs.append(np.nan)
                    modes.append(row.Mode)
                    segments.append(np.nan)
                    continue

                temps = [(s["t_min"], s["t_max"]) for s in segments_in_mode]
                idx = next((i for i, (tmin, tmax) in enumerate(temps) if tmin <= row.Temperature <= tmax), None)
                if idx is None:
                    idx = 0 if row.Temperature < temps[0][0] else len(temps) - 1

                assigned_seg = segments_in_mode[idx]
                assigned_seg_id = assigned_seg["Segmento"]

                if not assigned_seg["Thermal Sensitive"]:
                    residuals.append(np.nan)
                    probs.append(np.nan)
                    modes.append(row.Mode)
                    segments.append(assigned_seg_id)
                    continue

                model_info = next(
                    (m for m in models_info if m["Context"] == row.Context and m["Cluster"] == row.Cluster and
                     m["Mode"] == row.Mode and m["Segmento"] == assigned_seg_id),
                    None
                )
                if model_info is None:
                    residuals.append(np.nan)
                    probs.append(np.nan)
                    modes.append(row.Mode)
                    segments.append(assigned_seg_id)
                    continue

                y_pred = model_info["model"].predict(pd.DataFrame({"Temperature": [row.Temperature]}))[0]
                residual = row.Energy - y_pred
                sigma = model_info["sigma"]
                theta = 9 / (2 * np.log(2))

                if residual > 0:
                    prob = 1 - np.exp(-(residual ** 2) / (2 * theta * sigma ** 2))
                else:
                    prob = 0

                residuals.append(residual)
                probs.append(prob)
                modes.append(row.Mode)
                segments.append(assigned_seg_id)

            df_final["residual"] = residuals
            df_final["prob_anomaly"] = probs
            df_final["assigned_mode"] = modes
            df_final["assigned_segment"] = segments

            output_residuals = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "residuals")
            os.makedirs(output_residuals, exist_ok=True)
            df_final.to_csv(os.path.join(output_residuals, f"residuals_{foglia}.csv"), index=False)
            any_anomalies = True

        n_segments = len(segment_results)
        n_sensitive = sum(1 for s in segment_results if s["Thermal Sensitive"])
        print(f" {n_segments} segments, {n_sensitive} thermal sensitive" +
              ("    -> residuals computed\n" if n_sensitive > 0 else ""))

    if any_anomalies:
        print("\nCalculated residuals and thermal anomaly probability ✅\n")



if __name__ == "__main__":
    run_change_point(case_study="Cabina", penalty=100, norm_for_check_term_sens = False)
