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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
        E_max_leaf = 0
        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                df_normals, _ = run_energy_temp(case_study, leaf, context, cluster)
                if df_normals is not None and "Energy" in df_normals.columns:
                    max_energy = df_normals["Energy"].max()
                    if pd.notna(max_energy):
                        E_max_leaf = max(E_max_leaf, max_energy)

        energy_cutoff = 0.1 * E_max_leaf  # 10% del massimo

        # 2. Ora prosegui con la logica principale
        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                print(f"[Ctx {context} | Clst {cluster}]")
                df_normals, df_anomalies = run_energy_temp(case_study, leaf, context, cluster)

                df_normals = df_normals.dropna(subset=["Temperature", "Energy"]).copy()
                df_normals = df_normals[df_normals["Energy"] > 0].copy()
                if df_normals.empty:
                    continue

                X_raw = df_normals[["Temperature", "Energy"]].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_raw)

                gmm_1 = GaussianMixture(n_components=1, random_state=0).fit(X_scaled)
                gmm_2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit(X_scaled)

                bic_1 = gmm_1.bic(X_scaled)
                bic_2 = gmm_2.bic(X_scaled)

                try:
                    labels_2 = gmm_2.predict(X_scaled)
                    sil_score = silhouette_score(X_scaled, labels_2)
                except:
                    sil_score = -1

                use_2_components = (bic_2 < bic_1) and (sil_score > 0.3)

                if use_2_components:
                    gmm = gmm_2
                    labels = gmm.predict(X_scaled)
                    means = scaler.inverse_transform(gmm.means_)
                    on_label = np.argmax(means[:, 1])
                    df_normals["State"] = np.where(labels == on_label, "on", "off")

                    quantile = 0.4
                    threshold_energy = 2 * df_normals.loc[df_normals["State"] == "off", "Energy"].quantile(quantile)

                    # Prima regola: riassegna secondo la soglia 2x quantile
                    df_normals["State"] = np.where(
                        df_normals["Energy"] >= threshold_energy, "on", "off"
                    )

                    # Seconda regola: i punti on con energia < 10% max → off
                    df_normals.loc[
                        (df_normals["State"] == "on") & (df_normals["Energy"] < energy_cutoff),
                        "State"
                    ] = "off"

                else:
                    gmm = gmm_1
                    df_normals["State"] = "on"

                # Prepara i dati per il plot
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
                gs = gridspec.GridSpec(1, 1)
                ax0 = fig.add_subplot(gs[0])
                ax0.scatter(temp_off, energy_off, color="gray", alpha=0.3, label="Off (GMM + Soglie)")
                ax0.scatter(temp_on, energy_on, color="blue", alpha=0.5, label="On (fit)")
                ax0.set_xlabel("Temperatura Esterna [°C]")
                ax0.set_ylabel("Energia [kWh]")
                ax0.set_title(f"{leaf} | Context {context} | Cluster {cluster} - norm_method: {norm_method}")

                if use_2_components:
                    ax0.axhline(threshold_energy, color="orange", linestyle="--", linewidth=1,
                                label="Soglia 2x 40° percentile")
                    ax0.axhline(energy_cutoff, color="blue", linestyle=":", linewidth=2, label="Soglia 10% max leaf")

                ax0.legend()
                ax0.grid(True)

                fig_name = f"{leaf}_ctx{context}_cl{cluster}.png".replace(" ", "_")
                output_folder_viz_leaf = os.path.join(output_folder_viz, f"{leaf}")
                os.makedirs(output_folder_viz_leaf, exist_ok=True)
                plt.savefig(os.path.join(output_folder_viz_leaf, fig_name), dpi=300, bbox_inches="tight")
                plt.close()
    return

if __name__ == "__main__":
    calc_anm_prob("Cabina", "zscore")