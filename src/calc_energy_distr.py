# Salverà un csv con media e deviazione standard per ogni sottocarico sulla potenza non anomala. QUeste cose vanno fatte solo sul fondo, quindi
# bisogna riutilizzare la ricorsione per prendere i nodi finali. Salva nella cartella result/distribution il nome del nodo con mean e stdv
from utils import *
import json
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.metrics import roc_auc_score


def run_soft_evidence(case_study: str):

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

def run_soft_evidence_HDBSCAN_KNN(case_study: str, alpha: float = 0.1):
    """
    Calcola una probabilità di anomalia per ciascuna date-context-cluster combinando la distanza media dai vicini (k-NN)
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
            d_mean = dist_mean.mean()
            d_std = dist_mean.std() if dist_mean.std() > 0 else 1e-6
            knn_score = np.clip((dist_mean - d_mean) / d_std, 0, None)
            knn_prob = 1 - np.exp(-knn_score)

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
        roc_auc = roc_auc_score(df_all["is_anomaly"], df_all["anomaly_prob"])
        print(f"\n📈 ROC AUC GLOBALE (alpha={alpha}): {roc_auc:.4f}")



if __name__ == "__main__":
    # run_soft_evidence("Cabina")
    run_soft_evidence_HDBSCAN_KNN("Cabina")