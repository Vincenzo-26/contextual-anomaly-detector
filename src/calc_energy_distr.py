import os
os.environ["OMP_NUM_THREADS"] = "1"
from src.utils import *
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from tabulate import tabulate

def sigmoid_iqr(X_ref, X_target, k):
    q1 = np.percentile(X_ref, 25)
    q3 = np.percentile(X_ref, 75)
    iqr = q3 - q1
    threshold = X_ref.max() + 1.5 * iqr  # sposta il centro della sigmoide
    z = k * (X_target - threshold)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
def sigmoid_single_point(x_anom, X_target, k):
    offset = np.log(1 / 0.95 - 1) / -k
    threshold = x_anom - offset
    z = k * (X_target - threshold)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z)), threshold
def extend_probability_tail(X_apply, anomaly_prob, max_x):
    last_valid_idx = np.where(X_apply <= max_x)[0][-1]
    if last_valid_idx >= 1:
        dx = X_apply[last_valid_idx] - X_apply[last_valid_idx - 1]
        dy = anomaly_prob[last_valid_idx] - anomaly_prob[last_valid_idx - 1]
        slope = dy / dx if dx != 0 else 0.01
    else:
        slope = 0.01

    base_prob = anomaly_prob[last_valid_idx]

    for i in range(last_valid_idx + 1, len(X_apply)):
        delta = X_apply[i] - max_x
        anomaly_prob[i] = 1 - (1 - base_prob) * np.exp(-slope * delta / (1 - base_prob))
        anomaly_prob[i] = min(anomaly_prob[i], 1.0)
    return anomaly_prob

def run_soft_evd_EM(case_study: str, k_sigmoide: float = 6, threshold_metrics: float = 0.8, one_anm_peak=True):
    """
    Calcola le probabilità di anomalia per ogni foglia dell'albero dei carichi
    utilizzando modelli GMM o curve sigmoidi, e salva i risultati su file CSV.

    Args:
        case_study (str): Nome del caso studio.
        k_sigmoide (float): Coefficiente di pendenza per la sigmoide (default = 6 per pendenza a 45°).
        threshold_metrics (float): Soglia per determinare se un punto è anomalo per le metriche (default = 0.8).
        one_anm_peak (bool): Se True, usa una sola componente GMM per le anomalie, se False calcola in automatico
        il numero di componenti GMM (default = True).

    Returns:
        None
    """

    print_boxed_title("Energy evidences calculation 📈")

    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_EM")
    os.makedirs(evidence_path, exist_ok=True)

    TP_total = 0
    FP_total = 0
    FN_total = 0

    all_data = []

    foglie = find_leaf_nodes(config["Load Tree"])

    for foglia in foglie:
        print(f"\033[91m{foglia}\033[0m")

        energy_data_full = run_energy_in_tw(case_study, foglia)
        anm_table = pd.read_csv(os.path.join(anomaly_path, f"anomaly_table_{foglia}.csv"))

        anm_table["Date"] = pd.to_datetime(anm_table["Date"]).dt.date
        energy_data_full["Date"] = pd.to_datetime(energy_data_full["Date"]).dt.date

        merged = energy_data_full.merge(
            anm_table[["Date", "Context", "Cluster"]],
            on=["Date", "Context", "Cluster"],
            how="left", indicator=True
        )
        energy_data_clean = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        energy_data_anm = merged[merged["_merge"] == "both"].drop(columns=["_merge"])

        energy_data_full["anomaly_prob"] = np.nan

        anm_keys = set((str(row.Date), row.Context, row.Cluster) for row in anm_table.itertuples(index=False))
        energy_data_full["is_real_anomaly"] = energy_data_full.apply(
            lambda row: (str(row.Date), row.Context, row.Cluster) in anm_keys, axis=1
        )

        combinations = energy_data_full[["Context", "Cluster"]].drop_duplicates()

        skipped_cases = 0

        for _, row in combinations.iterrows():
            context = row["Context"]
            cluster = row["Cluster"]

            clean_subset = energy_data_clean[
                (energy_data_clean["Context"] == context) &
                (energy_data_clean["Cluster"] == cluster)
            ]
            anm_subset = energy_data_anm[
                (energy_data_anm["Context"] == context) &
                (energy_data_anm["Cluster"] == cluster)
            ]

            if clean_subset.empty:
                print(f"No normal data for Context {context} Cluster {cluster}. Skipping.")
                skipped_cases += 1
                continue

            mask = (energy_data_full["Context"] == context) & (energy_data_full["Cluster"] == cluster)
            X_apply = energy_data_full.loc[mask, "Energy"].values
            X_normal = clean_subset["Energy"].values.reshape(-1, 1)

            counts, _ = np.histogram(X_normal.flatten(), bins=30)
            peaks, _ = find_peaks(counts, height=0.05 * np.max(counts))
            k_estimated = min(max(1, len(peaks)), 3)

            gmm_normal = GaussianMixture(n_components=k_estimated, covariance_type='full', random_state=0)
            gmm_normal.fit(X_normal)
            p_x_normal = np.exp(gmm_normal.score_samples(X_apply.reshape(-1, 1)))

            if anm_subset.empty or anm_subset["Energy"].isna().all():
                print(f"Ctx {context} Clst {cluster} no anomaly data -> Using sigmoidal anomaly probability.")
                anomaly_prob = sigmoid_iqr(X_normal, X_apply, k_sigmoide)
                energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob
                continue

            X_anomaly = anm_subset["Energy"].values.reshape(-1, 1)

            try:
                if one_anm_peak:
                    k_anm_estimated = 1
                else:
                    counts_anm, _ = np.histogram(X_anomaly.flatten(), bins=30)
                    peaks_anm, _ = find_peaks(counts_anm, height=0.05 * np.max(counts_anm))
                    k_anm_estimated = min(max(1, len(peaks_anm)), 3)

                gmm_anomaly = GaussianMixture(n_components=k_anm_estimated, covariance_type='full', random_state=0)
                gmm_anomaly.fit(X_anomaly)
                print(f"Ctx {context} Clst {cluster} -> GMM applied via EM with {k_anm_estimated} peak(s).")
                p_x_anomaly = np.exp(gmm_anomaly.score_samples(X_apply.reshape(-1, 1)))
            except:
                x_anom = X_anomaly.flatten()[0]
                anomaly_prob, threshold = sigmoid_single_point(x_anom, X_apply, k_sigmoide)
                energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob
                print(f"Ctx {context} Clst {cluster} -> Only 1 anomaly point. Using sigmoid centered at {threshold:.2f} for P=0.95 at x={x_anom:.2f}.")
                continue

            n_normal = len(X_normal)
            n_anomaly = len(X_anomaly)
            prior_normal = n_normal / (n_normal + n_anomaly)
            prior_anomaly = n_anomaly / (n_normal + n_anomaly)

            anomaly_prob = (p_x_anomaly * prior_anomaly) / (p_x_anomaly * prior_anomaly + p_x_normal * prior_normal + 1e-10)

            # per impedire di avere falsi positivi quando la curva anm ha code molto lunghe
            min_safe = np.percentile(X_normal, 1)
            anomaly_prob[X_apply < min_safe] = 0

            max_anomaly_energy = X_anomaly.max()
            anomaly_prob = extend_probability_tail(X_apply, anomaly_prob, max_anomaly_energy)

            energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob

        # Metrics for this foglia
        predicted_anomalies = energy_data_full["anomaly_prob"] > threshold_metrics
        true_anomalies = energy_data_full["is_real_anomaly"]

        TP_total += ((predicted_anomalies == True) & (true_anomalies == True)).sum()
        FP_total += ((predicted_anomalies == True) & (true_anomalies == False)).sum()
        FN_total += ((predicted_anomalies == False) & (true_anomalies == True)).sum()

        all_data.append(energy_data_full)

        output_file = os.path.join(evidence_path, f"evd_{foglia}.csv")
        energy_data_full.to_csv(output_file, index=False)
        print(f"\nSaved - Skipped {skipped_cases} Ctx-Clst combinations due to insufficient data.\n")

    # Unione finale
    df_all = pd.concat(all_data, ignore_index=True)

    TP = TP_total
    FP = FP_total
    FN = FN_total

    precision = TP / (TP + FP) if TP + FP > 0 else None
    recall = TP / (TP + FN) if TP + FN > 0 else None

    try:
        roc_auc = roc_auc_score(df_all["is_real_anomaly"], df_all["anomaly_prob"])
    except ValueError:
        roc_auc = None

    # Output finale
    metrics = [
        [f"Precisione (threshold={threshold_metrics})", f"{precision:.3f} ({TP}/{TP + FP})" if precision is not None else "n.d."],
        [f"Recall     (threshold={threshold_metrics})", f"{recall:.3f} ({TP}/{TP + FN})" if recall is not None else "n.d."],
        ["ROC AUC", f"{roc_auc:.3f}" if roc_auc is not None else "n.d."]
    ]
    print(tabulate(metrics, headers=["Metriche 📊", ""], tablefmt="grid"))
    print("\n\n")



if __name__ == "__main__":
    run_soft_evd_EM("Cabina")