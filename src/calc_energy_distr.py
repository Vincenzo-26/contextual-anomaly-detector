# Salverà un csv con media e deviazione standard per ogni sottocarico sulla potenza non anomala. QUeste cose vanno fatte solo sul fondo, quindi
# bisogna riutilizzare la ricorsione per prendere i nodi finali. Salva nella cartella result/distribution il nome del nodo con mean e stdv
from utils import *
import json
from sklearn.neighbors import KernelDensity
import numpy as np

def run_soft_evidence(case_study: str):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")

    foglie = find_leaf_nodes(config["Load Tree"])
    for foglia in foglie:
        energy_data_full = run_energy_in_tw(case_study, foglia)
        anm_table = pd.read_csv(os.path.join(anomaly_path, f"anomaly_table_{foglia}.csv"))
        merged = energy_data_full.merge(anm_table[["Date", "Context", "Cluster"]], on=["Date", "Context", "Cluster"],
                                 how="left", indicator=True)
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

            # Trova il picco massimo della distribuzione stimata
            x_dens = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
            scores = np.exp(kde.score_samples(x_dens))
            x_peak = x_dens[scores.argmax()][0]  # valore della moda stimata

            density_stats[(context, cluster)] = (kde, max_density, x_peak)

        # Calcolo della probabilità di anomalia su tutti i punti originali
        anomaly_probs = []
        energy_data_full["Context"] = energy_data_full["Context"].astype(str)
        energy_data_full["Cluster"] = energy_data_full["Cluster"].astype(str)
        for _, row in energy_data_full.iterrows():
            key = (str(row["Context"]), str(row["Cluster"]))
            x = row["Energy"]

            if key in density_stats:
                kde, max_d, x_peak = density_stats[key]
                if x <= x_peak:
                    score = 0.0
                else:
                    density = np.exp(kde.score_samples([[x]]))[0]
                    score = 1 - (density / max_d)
                    score = np.clip(score, 0, 1)
            else:
                score = np.nan

            anomaly_probs.append(score)

        energy_data_full["anomaly_prob"] = anomaly_probs

        evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences")
        os.makedirs(evidence_path, exist_ok=True)
        energy_data_full.to_csv(os.path.join(evidence_path, f"evd_{foglia}.csv"), index=False)
    print(f"✅ Evidences avaiable for '{case_study}'")

if __name__ == "__main__":
    run_soft_evidence("Cabina")