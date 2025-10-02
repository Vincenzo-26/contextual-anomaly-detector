import json
import os
import pandas as pd

from utils import print_boxed_title, find_leaf_nodes
from settings import PROJECT_ROOT

def combine_soft_evidence(case_study: str):
    """
    Combina le probabilità di anomalia derivate dall'analisi energetica e da quella termica
    per ciascuna foglia del load tree, producendo le soft evidences necessarie
    per l'inferenza bayesiana.

    Per ogni nodo foglia:
      - carica le evidenze energetiche stimate con logistic regression (file CSV);
      - verifica la presenza dei risultati di sensibilità termica;
      - se non disponibili, mantiene solo le anomalie energetiche e marca il nodo come non termicamente sensibile;
      - se disponibili, unisce le due fonti (energia e temperatura) sulle chiavi [Date, Context, Cluster];
      - calcola la colonna "anomaly_prob" scegliendo tra probabilità termica e probabilità energetica
        a seconda della sensibilità termica del nodo;
      - salva il file CSV finale nella cartella `soft_evidences`.

    Args:
        case_study (str): Nome del caso di studio (cartella contenente dati, configurazioni e risultati).

    Returns:
        None: I risultati vengono salvati su disco come file CSV, uno per ciascun nodo foglia.
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    titolo = f"Combination of Energy and Temperature results for '{case_study}'🔌🌡️"
    print_boxed_title(titolo)

    energy_folder_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_LR")
    temp_folder_path = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity")
    output_folder = os.path.join(PROJECT_ROOT, "results", case_study, "soft_evidences")
    os.makedirs(output_folder, exist_ok=True)

    foglie = find_leaf_nodes(config["Load Tree"])

    for foglia in foglie:
        print(f"[{foglia}] Processing...    ", end="")

        df_energy = pd.read_csv(os.path.join(energy_folder_path, f"evd_{foglia}.csv"))
        temp_path = os.path.join(temp_folder_path, "ctx_thermal_sens", f"{foglia}.csv")

        if not os.path.exists(temp_path):
            df_energy["thermal_sensitive"] = False
            output_path = os.path.join(output_folder, f"soft_evidence_{foglia}.csv")
            df_energy.to_csv(output_path, index=False)
            print(f" Anomaly prob. avaiable (only energy)")
            continue


        df_energy = df_energy.rename(columns={"anomaly_prob": "energy_anomaly_prob"})

        df_temp = pd.read_csv(temp_path)
        df_temp = df_temp[["Date", "Context", "Cluster", "Mean_Temp", "anm_prob"]]
        df_temp = df_temp.rename(columns={"anm_prob": "temp_anomaly_prob"})

        df_merged = df_energy.merge(df_temp, on=["Date", "Context", "Cluster"], how="left")
        df_merged["thermal_sensitive"] = df_merged["temp_anomaly_prob"].notna()

        df_merged["anomaly_prob"] = df_merged.apply(
            lambda row: row["temp_anomaly_prob"] if row["thermal_sensitive"] else row["energy_anomaly_prob"],
            axis=1
        )

        ordered_cols = ["Date", "Context", "Cluster", "Energy", "Temperature", "energy_anomaly_prob", "temp_anomaly_prob",
                         "anomaly_prob", "is_real_anomaly", "thermal_sensitive"]
        df_merged = df_merged[[col for col in ordered_cols if col in df_merged.columns]]
        output_path = os.path.join(output_folder, f"soft_evidence_{foglia}.csv")
        df_merged.to_csv(output_path, index=False)
        print(f"Anomaly probabilities avaiable (energy and temperature)")

    print("\nAnomaly probabilities calculated ✅     -> ready for bayesian inference\n\n")

if __name__ == "__main__":
    combine_soft_evidence("Total")






