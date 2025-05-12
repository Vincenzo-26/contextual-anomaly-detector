import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import ruptures as rpt
from sklearn.preprocessing import StandardScaler
from src.utils import *
from sklearn.mixture import GaussianMixture
import warnings
from scipy.stats import ConstantInputWarning

def check_thermal_sensitivity(df_segment, normalize:bool, corr_thresh=0.5, r2_thresh=0.5, slope_thresh=0.2):
    """
    Verifica se un segmento è termicamente sensibile usando correlazione, pendenza e R².

    Args:
        df_segment (pd.DataFrame): Segmento con colonne 'Temperature' e 'Energy'.
        corr_thresh (float): Soglia di correlazione.
        r2_thresh (float): Soglia R².
        slope_thresh (float): Soglia della pendenza.

    Returns:
        dict: Metriche calcolate e flag 'is_thermal_sensitive'.
    """
    X_raw = df_segment[["Temperature"]]
    y_raw = df_segment["Energy"].values

    if len(X_raw) < 2:
        return {"correlation": np.nan, "slope": np.nan, "r2_score": np.nan, "is_thermal_sensitive": False}

    # Calcola la correlazione di Spearman (invariante alla scala)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        try:
            corr, _ = spearmanr(X_raw.values.flatten(), y_raw)
        except Exception:
            print("⚠️ Segmento con input costante: impossibile calcolare la correlazione.")
            corr = np.nan

    # Normalizzazione opzionale
    if normalize:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X_raw)
        y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
    else:
        X = X_raw
        y = y_raw

    # Modello lineare
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    r2 = r2_score(y, model.predict(X))

    is_sensitive = (abs(corr) > corr_thresh) and (abs(slope) > slope_thresh) and (r2 > r2_thresh)

    return {
        "correlation": round(corr, 3),
        "slope": round(slope, 3),
        "r2_score": round(r2, 3),
        "is_thermal_sensitive": is_sensitive,
        "corr_thresh": corr_thresh,
        "slope_thresh": slope_thresh,
        "r2_thresh": r2_thresh
    }

def identify_operational_modes(df: pd.DataFrame, n_components=2, bic_threshold=100):
    """
    Applica GMM per rilevare massimo 2 modalità operative e restituisce un dataframe con una colonna "Mode".
    Se la separazione non è giustificata dal BIC, assegna a tutti la stessa modalità (0).
    """
    df = df.dropna(subset=["Temperature", "Energy"])
    if df.empty or len(df) < 5:
        df["Mode"] = 0
        return df

    X = df[["Temperature", "Energy"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X_scaled)
    gmm2 = GaussianMixture(n_components=2, random_state=0).fit(X_scaled)

    bic1 = gmm1.bic(X_scaled)
    bic2 = gmm2.bic(X_scaled)

    if bic1 - bic2 > bic_threshold:
        df["Mode"] = gmm2.predict(X_scaled)
        return df, gmm2, scaler
    else:
        df["Mode"] = 0
        return df, gmm1, scaler

def run_change_point(case_study: str, penalty: int, normalize:bool=True):
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
    any_anomalies = False

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

                df_modes, best_gmm, scaler = identify_operational_modes(df_normals_sorted, bic_threshold=100)

                for mode in sorted(df_modes["Mode"].unique()):
                    df_mode = df_modes[df_modes["Mode"] == mode]
                    if len(df_mode) < 5:
                        continue

                    signal = df_mode[["Temperature", "Energy"]].values
                    signal_scaled = StandardScaler().fit_transform(signal)
                    algo = rpt.Pelt(model="rank").fit(signal_scaled)
                    change_points = algo.predict(pen=penalty)

                    start = 0
                    segments_in_mode = []
                    for i, end in enumerate(change_points):
                        segment = df_mode.iloc[start:end]
                        t_min = segment["Temperature"].min()
                        t_max = segment["Temperature"].max()
                        metrics = check_thermal_sensitivity(segment, normalize)

                        segment_results.append({
                            "Context": context, "Cluster": cluster, "Mode": mode,
                            "Segmento": i + 1, "t_min": t_min, "t_max": t_max,
                            "Thermal Sensitive": metrics["is_thermal_sensitive"]
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

                summary = f"[Ctx {context} | Clst {cluster}] -> "
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

                    mode_summaries.append(f"Mode {m+1} - {total_segments} segments {ts_text}")

                summary += f"{len(mode_summaries)} operational modes : " + " | ".join(mode_summaries)
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
                prob = 1 - np.exp(-(residual ** 2) / (2 * theta * sigma ** 2))

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
    run_change_point(case_study="Cabina", penalty=10)
