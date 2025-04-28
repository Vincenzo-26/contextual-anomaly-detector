import os
os.environ["OMP_NUM_THREADS"] = "1"
from utils import *
import json
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from scipy.stats import percentileofscore
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture



def run_soft_evd_KDE(case_study: str):

    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_KDE_PDF")
    os.makedirs(evidence_path, exist_ok=True)

    foglie = find_leaf_nodes(config["Load Tree"])

    for foglia in foglie:
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

        # Calcola KDE per ogni gruppo (context, cluster)
        energy_data_clean["Context"] = energy_data_clean["Context"].astype(str)
        energy_data_clean["Cluster"] = energy_data_clean["Cluster"].astype(str)
        grouped = energy_data_clean.groupby(["Context", "Cluster"])
        density_stats = {}

        for (context, cluster), group in grouped:
            X = group["Energy"].values.reshape(-1, 1)
            if len(X) < 2:
                continue
            kde = KernelDensity(kernel="gaussian", bandwidth=0.3).fit(X)
            max_density = np.exp(kde.score_samples(X)).max()
            density_stats[(context, cluster)] = (kde, max_density)

        # Calcolo della probabilità di anomalia su tutti i punti originali
        anomaly_probs = []
        energy_data_full["Context"] = energy_data_full["Context"].astype(str)
        energy_data_full["Cluster"] = energy_data_full["Cluster"].astype(str)
        for _, row in energy_data_full.iterrows():
            key = (str(row["Context"]), str(row["Cluster"]))
            x = row["Energy"]

            if key in density_stats:
                kde, max_d = density_stats[key]
                density = np.exp(kde.score_samples([[x]]))[0]
                score = 1 - (density / max_d)
                score = np.clip(score, 0, 1)
            else:
                score = np.nan

            anomaly_probs.append(score)

        energy_data_full["anomaly_prob"] = anomaly_probs

        # Salvataggio CSV finale
        out_file = os.path.join(evidence_path, f"evd_{foglia}.csv")
        energy_data_full.to_csv(out_file, index=False)
    print(f"✅ evidences avaiable for '{case_study}'")

def run_soft_evd_HDBSCAN_KNN(case_study: str, alpha: float = 0.5, threshold: float = 0.8):
    """
    Calcola la probabilità di anomalia per ciascuna date-context-cluster combinando la distanza media dai vicini (k-NN)
    e la probabilità di appartenenza HDBSCAN. è stato usato questo metodo perchè le distribuzioni dei punti 'normal'
    identificati dalla CMP per ogni sottocarico risultavano molto ampie e non normali.

    Per ogni cluster di ogni sottocarico:
    1. Si selezionano i soli punti 'normal'.
    2. Si applica HDBSCAN per creare dei sotto-cluster, ottenendo anche lo score di membership.
    3. Si addestra un modello k-NN solo sui 'normal' etichettati da HDBSCAN (escludendo il rumore).
    4. Ogni punto, normale o anomalo, viene assegnato a un sotto-cluster tramite k-NN.
    5. Si calcola la distanza media dai k vicini più vicini e si normalizza rispetto alla distribuzione osservata.
    6. La probabilità finale di anomalia (anomaly_prob) è calcolata come combinazione pesata tra:
       - il punteggio geometrico (1 - exp(-z-score)) derivato dal k-NN
       - lo score topologico (1 - Membership Probability) derivato da HDBSCAN

    Salva un file CSV per ogni foglia con le probabilità di anomalia e stampa il ROC AUC globale come metrica.

    Args:
        case_study (str): Nome del caso di studio (es. "Cabina").
        alpha (float): Peso del contributo k-NN (0 = solo HDBSCAN, 1 = solo k-NN).
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_HDBSCAN_KNN")
    os.makedirs(evidence_path, exist_ok=True)

    all_results = []
    foglie = find_leaf_nodes(config["Load Tree"])

    for foglia in foglie:
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
        energy_data_full["anomaly_prob"] = np.nan

        for main_cluster in sorted(energy_data_clean["Cluster"].unique()):
            cluster_data = energy_data_clean[energy_data_clean["Cluster"] == main_cluster].copy()
            X_normal = cluster_data[["Energy"]].values

            result = optimize_hdbscan_cluster_size(X_normal)
            labels = result["labels"]
            probs = result["probs"]

            cluster_data["HDBSCAN_Label"] = labels
            cluster_data["Membership_Prob"] = probs
            cluster_data["key"] = (
                cluster_data["Date"].astype(str) + "_" +
                cluster_data["Context"].astype(str) + "_" +
                cluster_data["Cluster"].astype(str)
            )
            temp_membership = cluster_data.set_index("key")["Membership_Prob"]

            energy_data_clean.loc[cluster_data.index, "HDBSCAN_Label"] = labels
            energy_data_clean.loc[cluster_data.index, "Membership_Prob"] = probs

            valid_mask = labels != -1
            if not np.any(valid_mask):
                continue

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_normal[valid_mask], labels[valid_mask])

            X_all = energy_data_full[energy_data_full["Cluster"] == main_cluster][["Energy"]].values
            distances, _ = knn.kneighbors(X_all)

            dist_mean = distances.mean(axis=1)
            ref_dist_mean = dist_mean[:len(X_normal)]  # solo i punti normali usati nel kNN
            knn_prob = np.array([
                percentileofscore(ref_dist_mean, d, kind="mean") / 100
                for d in dist_mean
            ])
            idx = energy_data_full[energy_data_full["Cluster"] == main_cluster].index
            energy_data_full["key"] = (
                energy_data_full["Date"].astype(str) + "_" +
                energy_data_full["Context"].astype(str) + "_" +
                energy_data_full["Cluster"].astype(str)
            )
            membership_probs = energy_data_full.loc[idx, "key"].map(temp_membership).fillna(0.0).values
            hdbscan_score = 1 - membership_probs

            anomaly_prob = alpha * knn_prob + (1 - alpha) * hdbscan_score
            energy_data_full.loc[idx, "anomaly_prob"] = anomaly_prob

        energy_data_full["is_anomaly"] = 0
        anm_table["key"] = (
            anm_table["Date"].astype(str) + "_" +
            anm_table["Context"].astype(str) + "_" +
            anm_table["Cluster"].astype(str)
        )
        energy_data_full.loc[energy_data_full["key"].isin(anm_table["key"]), "is_anomaly"] = 1

        all_results.append(energy_data_full[["anomaly_prob", "is_anomaly"]].dropna())

        out_file = os.path.join(evidence_path, f"evd_{foglia}.csv")
        energy_data_full[["Date", "Context", "Cluster", "Energy", "anomaly_prob"]].to_csv(out_file, index=False)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)

        y_true = df_all["is_anomaly"].values
        y_score = df_all["anomaly_prob"].values
        roc_auc = roc_auc_score(y_true, y_score)

        y_pred = (y_score > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n📊 Metriche globali (soglia = {threshold})")
        print(f" - ROC AUC         : {roc_auc:.4f}")
        print(f" - Specificità (TNR): {specificity:.4f}")
        print(f" - Precision       : {precision:.4f}")
        print(f" - Recall          : {recall:.4f}")
        print(f" - F1-score        : {f1:.4f}")

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
    print("\n📈 Calculating soft evidences...\n")
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
        print(f"Precisione (threshold={threshold}): {precision:.3f}")
    else:
        print(f"Precision non definita.")

    if TP + FN > 0:
        recall = TP / (TP + FN)
        print(f"Recall (threshold={threshold}): {recall:.3f}")
    else:
        print(f"Recall non definito.")

    # Calcolo ROC-AUC sempre, usando le probabilità continue
    try:
        roc_auc = roc_auc_score(true_anomalies, energy_data_full["anomaly_prob"])
        print(f"ROC AUC: {roc_auc:.3f}")
    except ValueError:
        print("ROC AUC non calcolabile.")


if __name__ == "__main__":
    # run_soft_evd_KDE("Cabina")

    run_soft_evd_EM("Cabina")