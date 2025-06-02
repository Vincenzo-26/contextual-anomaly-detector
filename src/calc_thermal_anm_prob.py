import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import numpy as np
import pandas as pd
from utils import run_energy_temp
from settings import PROJECT_ROOT
from src.utils import get_nodes_by_level
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def calc_anm_prob(case_study: str, soglia_en_max: float):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json")) as f:
        config = json.load(f)

    output_folder_viz = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "ctx_thermal_sens")
    os.makedirs(output_folder_viz, exist_ok=True)

    groups = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv"), parse_dates=["timestamp"])
    groups["date"] = groups["timestamp"].dt.date
    context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
    cluster_cols = [col for col in groups.columns if col.startswith("Cluster_")]

    levels = get_nodes_by_level(config["Load Tree"])
    for leaf in levels[0]:
        df_sens_path = os.path.join(PROJECT_ROOT, "results", case_study, "prova", "daily_thermal_sensitivity", f"segs_{leaf}.csv")
        df_sens = pd.read_csv(df_sens_path, index_col=0)
        if not df_sens["Thermal Sensitive"].any():
            continue

        print(f"\033[91m{leaf}\033[0m")

        E_max_leaf = 0
        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                df_normals, _ = run_energy_temp(case_study, leaf, context, cluster)
                if df_normals is not None and "Energy" in df_normals.columns:
                    E_max_leaf = max(E_max_leaf, df_normals["Energy"].max(skipna=True))
        energy_cutoff = soglia_en_max * E_max_leaf

        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                print(f"[Ctx {context} | Clst {cluster}]")
                df_normals, _ = run_energy_temp(case_study, leaf, context, cluster)
                df_normals = df_normals.dropna(subset=["Temperature", "Energy"])
                df_normals = df_normals[df_normals["Energy"] > 0].copy()
                if df_normals.empty:
                    continue

                df_normals["State"] = np.where(df_normals["Energy"] >= energy_cutoff, "on", "off")

                df_on = df_normals[df_normals["State"] == "on"]
                df_off = df_normals[df_normals["State"] == "off"]

                fig = plt.figure(figsize=(12, 5))
                ax = fig.add_subplot(gridspec.GridSpec(1, 1)[0])
                ax.scatter(df_off["Temperature"], df_off["Energy"], color="gray", alpha=0.3, label="Off (<10% max)")
                ax.scatter(df_on["Temperature"], df_on["Energy"], color="blue", alpha=0.5, label="On (≥10% max)")
                ax.axhline(energy_cutoff, color="blue", linestyle=":", linewidth=2, label="Soglia 10% max")
                ax.set_xlabel("Temperatura Esterna [°C]")
                ax.set_ylabel("Energia [kWh]")
                ax.set_title(f"{leaf} | Context {context} | Cluster {cluster}")
                ax.legend()
                ax.grid(True)

                output_leaf_folder = os.path.join(output_folder_viz, leaf)
                os.makedirs(output_leaf_folder, exist_ok=True)
                fig_path = os.path.join(output_leaf_folder, f"{leaf}_ctx{context}_cl{cluster}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close()

if __name__ == "__main__":
    calc_anm_prob("Cabina", 0.05)