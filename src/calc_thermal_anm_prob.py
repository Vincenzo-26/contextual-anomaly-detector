import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import numpy as np
import pandas as pd
from utils import run_energy_temp, scale_data
from settings import PROJECT_ROOT
from src.utils import get_nodes_by_level
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.gridspec as gridspec

def calc_anm_prob(case_study: str, norm_method):
    """
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
        config = json.load(f)

    output_folder_viz = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "ctx_thermal_sens")
    os.makedirs(output_folder_viz, exist_ok=True)

    groups_path = os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv")
    groups = pd.read_csv(groups_path, parse_dates=["timestamp"])
    groups["date"] = groups["timestamp"].dt.date
    context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
    cluster_cols = [col for col in groups.columns if col.startswith("Cluster_")]

    levels = get_nodes_by_level(config["Load Tree"])
    first_level = levels[0]
    for leaf in first_level:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{leaf}.csv"), index_col=0, parse_dates=True)
        temp_file = config["Outside Temperature"]
        df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{temp_file}.csv"), index_col=0, parse_dates=True)
        df_sens = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "prova", "daily_thermal_sensitivity", f"segs_{leaf}.csv"), index_col=0)

        if not df_sens["Thermal Sensitive"].any():
            continue

        print(f"\033[91m{leaf}\033[0m")
        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                print(f"[Ctx {context} | Clst {cluster}]")
                df_normals, df_anomalies = run_energy_temp(case_study, leaf, context, cluster)

                df_normals = df_normals.dropna(subset=["Temperature", "Energy"]).copy()

                # Filtro dei valori nulli o zero
                df_normals = df_normals[df_normals["Energy"] > 0].copy()
                if df_normals.empty:
                    continue

                # Applica GMM per distinguere "on" e "off"
                energy_raw = df_normals[["Energy"]].values
                energy_scaled = scale_data(energy_raw, norm_method)

                gmm = GaussianMixture(n_components=2, random_state=0)
                labels = gmm.fit_predict(energy_scaled)

                # Determina quale componente è "on" (quella con media maggiore)
                means = gmm.means_.flatten()
                on_label = np.argmax(means)
                df_normals["State"] = ["on" if lbl == on_label else "off" for lbl in labels]
                df_on = df_normals[df_normals["State"] == "on"].sort_values("Temperature")

                temp_on = df_on["Temperature"].values
                energy_on = df_on["Energy"].values

                df_off = df_normals[df_normals["State"] == "off"]
                temp_off = df_off["Temperature"].values
                energy_off = df_off["Energy"].values

                df_anomalies_clean = df_anomalies.dropna(subset=["Temperature", "Energy"])
                temp_anomalies = df_anomalies_clean["Temperature"].values
                energy_anomalies = df_anomalies_clean["Energy"].values

                # === PLOT ===
                fig = plt.figure(figsize=(12, 5))
                gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.3)

                # Scatterplot a sinistra
                ax0 = fig.add_subplot(gs[0])
                ax0.scatter(temp_off, energy_off, color="gray", alpha=0.3, label="Off (GMM)")
                ax0.scatter(temp_on, energy_on, color="blue", alpha=0.5, label="On (fit)")
                # ax0.scatter(temp_anomalies, energy_anomalies, color="red", alpha=0.8, label="Anomalie")
                ax0.set_xlabel("Temperatura Esterna [°C]")
                ax0.set_ylabel("Energia [kWh]")
                ax0.set_title(f"{leaf} | Context {context} | Cluster {cluster} - norm_method: {norm_method}")
                ax0.legend()
                ax0.grid(True)

                # Istogramma della frequenza a destra
                ax1 = fig.add_subplot(gs[1])
                sns.histplot(y=df_normals["Energy"], ax=ax1)
                ax1.set_xlabel("Frequenza")
                ax1.set_title("Distribuzione Energia")
                ax1.set_ylabel("")  # rimuove duplicazione asse y
                ax1.set_xlabel("Frequenza")
                ax1.set_title("Distribuzione Energia")

                # Salva la figura
                fig_name = f"{leaf}_ctx{context}_cl{cluster}.png".replace(" ", "_")
                output_folder_viz_leaf = os.path.join(output_folder_viz, f"{leaf}")
                os.makedirs(output_folder_viz_leaf, exist_ok=True)
                plt.savefig(os.path.join(output_folder_viz_leaf, fig_name), dpi=300, bbox_inches="tight")
                plt.close()
    return

if __name__ == "__main__":
    calc_anm_prob("Cabina", "zscore")