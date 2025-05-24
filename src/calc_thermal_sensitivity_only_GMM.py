import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
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
def check_thermal_sensitivity(df_segment, normalize="zscore", corr_thresh=0.5, r2_thresh=0.5, slope_thresh=0.2):
    """
    Verifica se un segmento è termicamente sensibile usando correlazione, pendenza e R².

    Args:
        df_segment (pd.DataFrame): Segmento con colonne 'Temperature' e 'Energy'.
        normalize (str or bool): Metodo di normalizzazione ('zscore', 'minmax', 'robust', 'maxabs', 'none' o False).
        corr_thresh (float): Soglia di correlazione Spearman.
        r2_thresh (float): Soglia R².
        slope_thresh (float): Soglia della pendenza della retta di regressione.

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
            "is_thermal_sensitive": False
        }

    # Correlazione di Spearman (su dati grezzi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        try:
            corr, _ = spearmanr(X_raw.values.flatten(), y_raw.flatten())
        except Exception:
            print("⚠️ Segmento con input costante: impossibile calcolare la correlazione.")
            corr = np.nan

    # Normalizzazione per la regressione
    method = str(normalize).lower() if normalize not in [False, None] else "none"
    X = scale_data(X_raw, method)
    y = scale_data(y_raw, method).flatten()

    # Regressione lineare su dati normalizzati
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    r2 = r2_score(y, model.predict(X))

    # Verifica criteri
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
def identify_operational_modes(df: pd.DataFrame, max_components=6, covariance_type="tied", alpha=25):
    df = df.dropna(subset=["Temperature", "Energy"])
    if df.empty or len(df) < 5:
        df["Mode"] = 0
        return df, None, None

    X = df[["Temperature", "Energy"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lowest_score = np.inf
    best_gmm = None
    best_n = 1

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=0)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        penalized_bic = bic + alpha * n  # penalizzazione personalizzata

        if penalized_bic < lowest_score:
            lowest_score = penalized_bic
            best_gmm = gmm
            best_n = n

    df["Mode"] = best_gmm.predict(X_scaled)
    return df, best_gmm, scaler


def run_change_point(case_study: str, norm_for_check_term_sens: any = (True, "zscore")):
    """
    Analizza la thermal sensitivity delle modalità operative di ciascuna foglia dell’albero di carico
    utilizzando GMM, e calcola i residui e la probabilità di anomalia solo per le modalità termicamente sensibili.

    1. Per ogni foglia del sistema energetico (leaf node):
        - Si itera su ogni combinazione context-cluster

    2. Per ciascuna combinazione:
        - Si caricano i dati "normali" e "anomali" di temperatura ed energia.
        - Si applica una Gaussian Mixture Model (GMM) ai dati normali per identificare automaticamente
          le modalità operative, selezionando, tra un range, il numero ottimale che minimizza la metrica BIC.

    3. Per ciascuna modalità individuata:
        - Si verifica la sensibilità termica applicando una regressione lineare sui dati **normalizzati**
          (z-score, min-max, ecc. a scelta) e valutando:
            • la correlazione di Spearman (temperatura vs energia),
            • la pendenza della retta interpolante,
            • il coefficiente R².
        - Se la modalità è termicamente sensibile:
            • Si costruisce un nuovo modello lineare **sui dati reali (non normalizzati)**.
            • Si calcola l’errore standard del modello (sigma).
            • Il modello viene salvato per l’analisi dei residui.

    4. Tutti i dati (normali e anomali) vengono:
        - Rietichettati con la modalità assegnata dal GMM (predict).
        - Annotati come anomali o normali a seconda dell’origine.

    5. Per ogni punto associato a una modalità termicamente sensibile:
        - Si calcola il **residuo** rispetto alla previsione del modello.
        - Si calcola la **probabilità di anomalia** usando un modello gaussiano controllato da sigma.

    6. Output:
        - `segment_{foglia}.csv`: riepilogo delle modalità (con flag di thermal sensitivity).
        - `residuals_{foglia}.csv`: dataset esteso con residui e probabilità di anomalia.

    Args:
        case_study (str): Nome del caso studio, utilizzato per caricare configurazioni e dati.
        norm_for_check_term_sens (any): Controlla se e come normalizzare i dati nella verifica
            di sensibilità termica (passato come argomento alla funzione check_thermal_sensitivity).
             Può essere:
            - False: nessuna normalizzazione;
            - True: normalizzazione standard z-score;
            - (True, "minmax" / "robust" / ecc.): metodo personalizzato.

    Returns:
        None. I risultati vengono salvati nei file CSV nelle cartelle di output.
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    foglie = find_leaf_nodes(config["Load Tree"])
    print_boxed_title(f"Thermal sensitivity analysis for '{case_study}'🌡️")

    if norm_for_check_term_sens is False:
        thermal_scaling = "none"
        print("Data not normalized for checking thermal sensitivity")
    elif norm_for_check_term_sens is True:
        thermal_scaling = "zscore"
        print(f"Normalized data for checking thermal sensitivity -> Method: {thermal_scaling} (default)\n")
    elif isinstance(norm_for_check_term_sens, (tuple, list)) and norm_for_check_term_sens[0] is True:
        thermal_scaling = norm_for_check_term_sens[1]
        print(f"Normalized data for checking thermal sensitivity -> Method: {thermal_scaling}\n")
    else:
        raise ValueError(f"Parametro non valido per norm_for_check_term_sens: {norm_for_check_term_sens}")

    for foglia in foglie:
        print(f"\033[91m{foglia}\033[0m")
        segment_results = []
        models_info = []
        segment_map = {}
        all_rows = []

        groups_path = os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv")
        groups = pd.read_csv(groups_path, parse_dates=["timestamp"])
        groups["date"] = groups["timestamp"].dt.date
        context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
        cluster_cols = [col for col in groups.columns if col.startswith("Cluster_")]

        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                df_normals, df_anomalies = run_energy_temp(case_study, foglia, context, cluster)
                if df_normals is None or df_normals.empty:
                    continue

                df_normals_sorted = df_normals.sort_values("Temperature").dropna(subset=["Temperature", "Energy"])
                df_modes, best_gmm, scaler = identify_operational_modes(df_normals_sorted, max_components=6)

                for mode in sorted(df_modes["Mode"].unique()):
                    df_mode = df_modes[df_modes["Mode"] == mode]
                    if len(df_mode) < 5:
                        continue

                    t_min = df_mode["Temperature"].min()
                    t_max = df_mode["Temperature"].max()

                    metrics = check_thermal_sensitivity(df_mode, normalize=thermal_scaling)

                    segment_results.append({
                        "Context": context,
                        "Cluster": cluster,
                        "Mode": mode,
                        "t_min": t_min,
                        "t_max": t_max,
                        "Thermal Sensitive": metrics["is_thermal_sensitive"]
                    })

                    # Se sensibile, calcolo del modello lineare sui dati originali non normalizzati per il calcolo dei residui
                    if metrics["is_thermal_sensitive"]:
                        model = LinearRegression().fit(df_mode[["Temperature"]], df_mode["Energy"])
                        sigma = mean_squared_error(df_mode["Energy"], model.predict(df_mode[["Temperature"]])) ** 0.5
                        models_info.append({
                            "Context": context,
                            "Cluster": cluster,
                            "Mode": mode,
                            "is_thermal_sensitive": True,
                            "model": model,
                            "sigma": sigma
                        })
                    else:
                        models_info.append({
                            "Context": context,
                            "Cluster": cluster,
                            "Mode": mode,
                            "is_thermal_sensitive": False,
                            "model": None,
                            "sigma": None
                        })

                # Output riepilogo per context-cluster
                n_modes = len(df_modes["Mode"].unique())
                n_ts = sum(
                    m["is_thermal_sensitive"]
                    for m in models_info
                    if m["Context"] == context and m["Cluster"] == cluster
                )
                summary = f"[Ctx {context} | Clst {cluster}] -> {n_modes} operational modes: "
                ts_info = f"{n_ts}/{n_modes} thermal sensitive"
                if n_ts > 0:
                    ts_info = f"\033[92m{ts_info}\033[0m"
                print(summary + ts_info)

                df_all = pd.concat([df_normals_sorted, df_anomalies]) if df_anomalies is not None else df_normals_sorted.copy()
                df_all = df_all.reset_index().rename(columns={"index": "Date"})
                df_all["Context"] = context
                df_all["Cluster"] = cluster
                df_all["Mode"] = best_gmm.predict(scaler.transform(df_all[["Temperature", "Energy"]].values))
                df_all["is_real_anomaly"] = df_all["Date"].isin(df_anomalies.index) if df_anomalies is not None else False
                all_rows.append(df_all)

        output_segments = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity_only_GMM", "segments")
        os.makedirs(output_segments, exist_ok=True)
        pd.DataFrame(segment_results).to_csv(os.path.join(output_segments, f"segment_{foglia}.csv"), index=False)

        if all_rows and models_info:
            df_final = pd.concat(all_rows)
            residuals, probs = [], []

            for row in df_final.itertuples():
                model_info = next(
                    (m for m in models_info
                     if m["Context"] == row.Context and m["Cluster"] == row.Cluster and m["Mode"] == row.Mode),
                    None
                )

                if model_info is None or not model_info["is_thermal_sensitive"]:
                    residuals.append(np.nan)
                    probs.append(np.nan)
                    continue

                model = model_info["model"]
                sigma = model_info["sigma"]
                y_pred = model.predict(pd.DataFrame({"Temperature": [row.Temperature]}))[0]
                residual = row.Energy - y_pred
                theta = 9 / (2 * np.log(2))
                prob = 1 - np.exp(-(residual ** 2) / (2 * theta * sigma ** 2))

                residuals.append(residual)
                probs.append(prob)

            df_final["residual"] = residuals
            df_final["prob_anomaly"] = probs

            output_residuals = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity_only_GMM", "residuals")
            os.makedirs(output_residuals, exist_ok=True)
            df_final.to_csv(os.path.join(output_residuals, f"residuals_{foglia}.csv"), index=False)
            print("Calculated residuals and thermal anomaly probability ✅\n")



if __name__ == "__main__":
    run_change_point(case_study="Cabina")
