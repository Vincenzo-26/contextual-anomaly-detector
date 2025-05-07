import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import ruptures as rpt
from src.utils import *

def check_thermal_sensitivity(df_segment, corr_thresh=0.5, r2_thresh=0.5, slope_thresh=0.2):
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
    X = df_segment[["Temperature"]]
    y = df_segment["Energy"].values

    if len(X) < 2:
        return {"correlation": np.nan, "slope": np.nan, "r2_score": np.nan, "is_thermal_sensitive": False}

    corr = np.corrcoef(X.values.flatten(), y)[0, 1]
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

def run_change_point(case_study: str, penalty: int):
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

    titolo = f"Thermal sensitivity analysis for '{case_study}'🌡️"
    print_boxed_title(titolo)

    any_anomalies = False

    for foglia in foglie:
        print(f"[{foglia}] Processing...", end="")
        segment_results = []
        models_info = []

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
                df_normals_sorted = df_normals_sorted[df_normals_sorted["Energy"] > 0]

                if len(df_normals_sorted) < 5:
                    continue

                signal = df_normals_sorted[["Temperature", "Energy"]].values
                algo = rpt.Pelt(model="rbf").fit(signal)
                change_points = algo.predict(pen=penalty)

                start = 0
                for i, end in enumerate(change_points):
                    segment = df_normals_sorted.iloc[start:end]
                    metrics = check_thermal_sensitivity(segment)

                    segment_results.append({
                        "Context": context,
                        "Cluster": cluster,
                        "Segmento": i + 1,
                        "Thermal Sensitive": metrics["is_thermal_sensitive"]
                    })

                    if metrics["is_thermal_sensitive"]:
                        model = LinearRegression().fit(segment[["Temperature"]], segment["Energy"])
                        t_min = segment["Temperature"].min()
                        t_max = segment["Temperature"].max()
                        sigma = mean_squared_error(segment["Energy"], model.predict(segment[["Temperature"]])) ** 0.5

                        models_info.append({
                            "Context": context,
                            "Cluster": cluster,
                            "Segmento": i + 1,
                            "model": model,
                            "t_min": t_min,
                            "t_max": t_max,
                            "sigma": sigma
                        })

                    start = end

                if df_anomalies is not None and not df_anomalies.empty:
                    combined = pd.concat([df_normals, df_anomalies])
                else:
                    combined = df_normals.copy()

                combined = combined.reset_index().rename(columns={"index": "Date"})
                combined["is_real_anomaly"] = combined["Date"].isin(df_anomalies.index) if df_anomalies is not None else False
                combined["Context"] = context
                combined["Cluster"] = cluster
                all_rows.append(combined)

        output_segments = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "segments")
        os.makedirs(output_segments, exist_ok=True)
        pd.DataFrame(segment_results).to_csv(os.path.join(output_segments, f"segment_{foglia}.csv"), index=False)

        if all_rows and models_info:
            df_final = pd.concat(all_rows)
            anomaly_data = []

            for row in df_final.itertuples():
                temp = row.Temperature
                energy = row.Energy
                date_str = str(row.Date)
                matched = False

                for m in models_info:
                    if m["Context"] == row.Context and m["Cluster"] == row.Cluster and m["t_min"] <= temp <= m["t_max"]:
                        y_pred = m["model"].predict(pd.DataFrame([[round(temp, 2)]], columns=["Temperature"]))[0]
                        residual = energy - y_pred
                        theta = 9 / (2 * np.log(2))
                        prob_anomaly = 1 - np.exp(-(residual ** 2) / (2 * theta * m["sigma"] ** 2))

                        anomaly_data.append({
                            "Date": date_str,
                            "Context": row.Context,
                            "Cluster": row.Cluster,
                            "Segmento": m["Segmento"],
                            "Temperature": temp,
                            "Energy": energy,
                            "Residual": residual,
                            "Anomaly_Prob": prob_anomaly,
                            "is_real_anomaly": row.is_real_anomaly,
                        })
                        matched = True
                        break
            df_anomaly = pd.DataFrame(anomaly_data)
            output_residuals = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "residuals")
            os.makedirs(output_residuals, exist_ok=True)
            df_anomaly.to_csv(os.path.join(output_residuals, f"residuals_{foglia}.csv"), index=False)
            any_anomalies = True

        n_segments = len(segment_results)
        n_sensitive = sum(s["Thermal Sensitive"] for s in segment_results)

        if n_sensitive > 0:
            print(f" {n_segments} segments, {n_sensitive} thermal sensitive    -> residuals computed")
        else:
            print(f" {n_segments} segments, 0 thermal sensitive")

    if any_anomalies:
        print("\nCalculated residuals and thermal anomaly probability ✅\n\n")


if __name__ == "__main__":
    run_change_point(case_study="Cabina", penalty=10)
