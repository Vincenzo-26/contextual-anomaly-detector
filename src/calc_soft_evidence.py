import os
from utils import *
import json

def combine_soft_evidence(case_study: str, alpha: float = 0.7):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    titolo = f"Combination of Energy and Temperature results for '{case_study}'🔌🌡️"
    print_boxed_title(titolo)

    energy_folder_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_EM")
    temp_folder_path = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "residuals")
    output_folder = os.path.join(PROJECT_ROOT, "results", case_study, "soft_evidences")
    os.makedirs(output_folder, exist_ok=True)

    foglie = find_leaf_nodes(config["Load Tree"])

    for foglia in foglie:
        print(f"[{foglia}] Processing...", end="")

        df_energy = pd.read_csv(os.path.join(energy_folder_path, f"evd_{foglia}.csv"))
        temp_path = os.path.join(temp_folder_path, f"residuals_{foglia}.csv")

        if not os.path.exists(temp_path):
            df_energy["thermal_sensitive"] = False
            output_path = os.path.join(output_folder, f"soft_evidence_{foglia}.csv")
            df_energy.to_csv(output_path, index=False)
            print(f" Anomaly prob. avaiable (only energy)")
            continue


        df_energy = df_energy.rename(columns={"anomaly_prob": "energy_anomaly_prob"})

        df_temp = pd.read_csv(temp_path)
        df_temp = df_temp[["Date", "Context", "Cluster", "Temperature", "Anomaly_Prob"]]
        df_temp = df_temp.rename(columns={"Anomaly_Prob": "temp_anomaly_prob"})

        df_merged = df_energy.merge(df_temp, on=["Date", "Context", "Cluster"], how="left")
        df_merged["thermal_sensitive"] = df_merged["temp_anomaly_prob"].notna()

        df_merged["anomaly_prob"] = df_merged.apply(
            lambda row: row["energy_anomaly_prob"] * (1 - alpha * row["temp_anomaly_prob"])
            if row["thermal_sensitive"] else row["energy_anomaly_prob"],
            axis=1
        )
        ordered_cols = ["Date", "Context", "Cluster", "Energy", "Temperature", "energy_anomaly_prob", "temp_anomaly_prob",
                         "anomaly_prob", "is_real_anomaly", "thermal_sensitive"]
        df_merged = df_merged[[col for col in ordered_cols if col in df_merged.columns]]
        output_path = os.path.join(output_folder, f"soft_evidence_{foglia}.csv")
        df_merged.to_csv(output_path, index=False)
        print(f" Anomaly probabilities avaiable (energy and temperature mixed)")

    print("\nAnomaly probabilities calculated ✅     -> ready for bayesian inference\n\n")

if __name__ == "__main__":
    combine_soft_evidence("Cabina")






