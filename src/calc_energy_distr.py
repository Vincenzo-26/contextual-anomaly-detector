import os
os.environ["OMP_NUM_THREADS"] = "1"
from utils import *
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from tabulate import tabulate


def run_soft_evd_EM(case_study: str, k_sigmoide: float = 6, threshold: float = 0.8):
    """
     Calcola la probabilità di anomalia per ciascun dato,
     utilizzando Gaussian Mixture Models (GMM) o una funzione sigmoide di fallback,
     e valuta la qualità della classificazione con Precision, Recall e ROC AUC.

     Per ogni sottocarico:
        - Separa i dati normal da quelli anomali.
        - Per ogni combinazione di Context e Cluster:
            - Se esistono dati normal:
                - Stima la distribuzione dei dati normal tramite GMM (numero di distribuzioni stimato dai picchi di frequenza).
                - Se esistono dati anomali validi:
                    - Stima anche la distribuzione anomala tramite GMM.
                    - Calcola la probabilità di anomalia via formula bayesiana.
                - Se i dati anomali non sono disponibili o insufficienti:
                    - Usa una funzione sigmoide basata sul massimo valore di energia normal.
            - Se non esistono dati normal:
                - Skippa la combinazione context-cluster.
        - Salva il dataframe aggiornato con le anomaly_prob in `Evidences_EM`.

     Al termine:
     - Applica la soglia definita su `anomaly_prob` per predire anomalie.
     - Calcola e stampa le metriche:
         - Precision
         - Recall
         - ROC AUC

     Args:
         case_study (str): Nome del caso studio da analizzare.
         k_sigmoide (float, optional): Fattore di pendenza per la funzione sigmoide. Default 6.
         threshold (float, optional): Soglia di probabilità per precisione e richiamo. Default 0.8.

     """
    titolo = "Energy evidences calculation 📈"
    print_boxed_title(titolo)

    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_EM")
    os.makedirs(evidence_path, exist_ok=True)

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

        anm_keys = set((row.Date, row.Context, row.Cluster) for row in anm_table.itertuples(index=False))
        energy_data_full["is_real_anomaly"] = energy_data_full.apply(
            lambda row: (row.Date, row.Context, row.Cluster) in anm_keys, axis=1
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

            # Stima il numero di picchi
            counts, _ = np.histogram(X_normal.flatten(), bins=30)
            peaks, _ = find_peaks(counts, height=0.05 * np.max(counts))  # 5% soglia per ignorare rumore

            k_estimated = len(peaks)
            k_estimated = min(max(1, k_estimated), 3)  # almeno 1, massimo 3

            # Fit GMM sui dati normal
            gmm_normal = GaussianMixture(n_components=k_estimated, covariance_type='full', random_state=0)
            gmm_normal.fit(X_normal)

            # Calcola p(x | normal)
            p_clean = np.exp(gmm_normal.score_samples(X_apply.reshape(-1, 1)))

            # Se non ho anomalie o sono NaN, uso la sigmoide
            if anm_subset.empty or anm_subset["Energy"].isna().all():
                print(f"Ctx {context} Clst {cluster} no anomaly data    -> Using sigmoidal anomaly probability.")

                max_normal = X_normal.max()
                z = k_sigmoide * (X_apply - max_normal)
                z = np.clip(z, -500, 500)
                anomaly_prob = 1 / (1 + np.exp(-z))

                energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob
                continue

            X_anomaly = anm_subset["Energy"].values.reshape(-1, 1)

            if np.isnan(X_anomaly).any() or len(X_anomaly) < 2:
                print(
                    f"Ctx {context} Clst {cluster} dev.std invalid for anomaly data (only 1 record)    -> Using sigmoidal anomaly probability.")

                max_normal = X_normal.max()
                z = k_sigmoide * (X_apply - max_normal)
                z = np.clip(z, -500, 500)
                anomaly_prob = 1 / (1 + np.exp(-z))

                energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob
                continue

            # Fit GMM anche sugli anomaly se possibile
            try:
                gmm_anomaly = GaussianMixture(n_components=1, covariance_type='full', random_state=0)
                gmm_anomaly.fit(X_anomaly)
                p_anm = np.exp(gmm_anomaly.score_samples(X_apply.reshape(-1, 1)))
            except:
                print(f"Failed fitting GMM anomaly model for context {context} cluster {cluster}. Using sigmoide.")
                max_normal = X_normal.max()
                z = k_sigmoide * (X_apply - max_normal)
                z = np.clip(z, -500, 500)
                anomaly_prob = 1 / (1 + np.exp(-z))
                energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob
                continue

            n_clean = len(X_normal)
            n_anm = len(X_anomaly)
            prior_clean = n_clean / (n_clean + n_anm)
            prior_anm = n_anm / (n_clean + n_anm)

            anomaly_prob = (p_anm * prior_anm) / (p_anm * prior_anm + p_clean * prior_clean + 1e-10)

            energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob

        output_file = os.path.join(evidence_path, f"evd_{foglia}.csv")
        energy_data_full.to_csv(output_file, index=False)

        print(f"\nSaved - Skipped {skipped_cases} Ctx-Clst combinations due to insufficient data.\n")

    predicted_anomalies = energy_data_full["anomaly_prob"] > threshold
    true_anomalies = energy_data_full["is_real_anomaly"]

    TP = ((predicted_anomalies == True) & (true_anomalies == True)).sum()
    FP = ((predicted_anomalies == True) & (true_anomalies == False)).sum()
    FN = ((predicted_anomalies == False) & (true_anomalies == True)).sum()

    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = None

    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = None

    # Calcolo ROC-AUC sempre, usando le probabilità continue
    try:
        roc_auc = roc_auc_score(true_anomalies, energy_data_full["anomaly_prob"])
    except ValueError:
        roc_auc = None

    metrics = [
        [f"Precisione (threshold={threshold})", f"{precision:.3f}" if precision is not None else "n.d."],
        [f"Recall     (threshold={threshold})", f"{recall:.3f}" if recall is not None else "n.d."],
        ["ROC AUC", f"{roc_auc:.3f}" if roc_auc is not None else "n.d."]
    ]

    print(tabulate(metrics, headers=["Metriche 📊", ""], tablefmt="grid"))
    print("\n\n")

if __name__ == "__main__":
    run_soft_evd_EM("Cabina")