import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from utils import run_energy_temp
from settings import PROJECT_ROOT
from src.utils import get_nodes_by_level
from scipy.stats import norm, shapiro

def calc_anm_prob(case_study: str, soglia_en_max: float):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json")) as f:
        config = json.load(f)

    output_folder_viz = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "ctx_thermal_sens")
    os.makedirs(output_folder_viz, exist_ok=True)
    output_df = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity")

    context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
    groups = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv"), parse_dates=["timestamp"])
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

        all_residuals = []

        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                print(f"[Ctx {context} | Clst {cluster}]...     ", end="")
                df_normals, df_anomalous = run_energy_temp(case_study, leaf, context, cluster)

                if df_normals is None or df_normals.empty:
                    print("Nessun punto normal.")
                    continue

                df_normals = df_normals.dropna(subset=["Temperature", "Energy"])
                df_normals["Status"] = np.where(df_normals["Energy"] >= energy_cutoff, "ON", "OFF")
                df_normals["anomalous"] = False

                if df_anomalous is not None and not df_anomalous.empty:
                    df_anomalous = df_anomalous.dropna(subset=["Temperature", "Energy"])
                    df_anomalous["Status"] = np.where(df_anomalous["Energy"] >= energy_cutoff, "ON", "OFF")
                    df_anomalous["anomalous"] = True
                else:
                    df_anomalous = pd.DataFrame(columns=df_normals.columns)

                df_on = df_normals[df_normals["Status"] == "ON"].sort_values("Temperature")
                df_off = df_normals[df_normals["Status"] == "OFF"]
                if len(df_on) < 5:
                    print("OFF")
                    continue

                # Change point detection
                signal = df_on[["Energy"]].values
                algo = rpt.Binseg(model="l2").fit(signal)
                for n_bkps in [3, 2, 1]:
                    try:
                        change_points = algo.predict(n_bkps=n_bkps)
                        if change_points == [len(signal)]:
                            print("Num cp: 0")
                        else:
                            print(f"Num cp: {len(change_points)}")
                        break
                    except rpt.exceptions.BadSegmentationParameters:
                        continue

                # Segmenti e interpolazione
                segments = []
                for i in range(len(change_points)):
                    start = 0 if i == 0 else change_points[i - 1]
                    end = change_points[i]
                    seg_x = df_on["Temperature"].values[start:end]
                    seg_y = df_on["Energy"].values[start:end]
                    coeffs = np.polyfit(seg_x, seg_y, deg=1)
                    Tmin, Tmax = seg_x.min(), seg_x.max()
                    segments.append((np.poly1d(coeffs), Tmin, Tmax))

                # Calcolo residui solo per normal ON
                residui_normal = []
                for model, Tmin, Tmax in segments:
                    mask = (df_on["Temperature"] >= Tmin) & (df_on["Temperature"] <= Tmax)
                    temps = df_on.loc[mask, "Temperature"]
                    true_vals = df_on.loc[mask, "Energy"]
                    y_pred = model(temps)
                    residui_normal.extend(true_vals - y_pred)

                residui_normal = np.array(residui_normal)
                sigma2 = np.var(residui_normal)
                theta = 6.5

                def compute_residual_prob(row):
                    if row["Status"] == "OFF":
                        return np.nan, np.nan
                    T = row["Temperature"]
                    E = row["Energy"]
                    # Trova segmento
                    for model, Tmin, Tmax in segments:
                        if Tmin <= T <= Tmax:
                            y_pred = model(T)
                            break
                    else:
                        if T < segments[0][1]:
                            model = segments[0][0]
                        elif T > segments[-1][2]:
                            model = segments[-1][0]
                        y_pred = model(T)
                    residuo = E - y_pred
                    if residuo < 0:
                        return residuo, 0
                    prob = 1 - np.exp(- (residuo ** 2) / (2 * theta * sigma2)) if sigma2 > 0 else 0
                    return residuo, prob

                frames_to_concat = [df_on, df_off]
                if not df_anomalous.empty:
                    frames_to_concat.append(df_anomalous)
                df_all = pd.concat(frames_to_concat, ignore_index=True)
                df_all = df_all.sort_values("Temperature")
                df_all[["Residuo", "Probabilità"]] = df_all.apply(compute_residual_prob, axis=1, result_type="expand")
                df_all["leaf"] = leaf
                df_all["context"] = context
                df_all["cluster"] = cluster
                all_residuals.append(df_all)

                # Plot
                output_leaf_folder = os.path.join(output_folder_viz, leaf)
                os.makedirs(output_leaf_folder, exist_ok=True)
                fig_path = os.path.join(output_leaf_folder, f"{leaf}_ctx{context}_cl{cluster}.png")

                fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})
                ax = axs[0]

                ax.scatter(df_off["Temperature"], df_off["Energy"], color="gray", alpha=0.3, label="Normal OFF")
                ax.scatter(df_on["Temperature"], df_on["Energy"], color="blue", alpha=0.4, label="Normal ON")

                df_an_on = df_anomalous[df_anomalous["Status"] == "ON"]
                df_an_off = df_anomalous[df_anomalous["Status"] == "OFF"]
                ax.scatter(df_an_on["Temperature"], df_an_on["Energy"], color="red", alpha=0.6, label="Anomalous ON")
                ax.scatter(df_an_off["Temperature"], df_an_off["Energy"], color="orange", alpha=0.6, label="Anomalous OFF")

                for model, Tmin, Tmax in segments:
                    T_plot = np.linspace(Tmin, Tmax, 100)
                    ax.plot(T_plot, model(T_plot), color="black", linewidth=1)

                ax.axhline(energy_cutoff, color="orange", linewidth=1, label="Soglia ON/OFF")
                ax.set_title(f"{leaf}_ctx{context}_cl{cluster}")
                ax.set_xlabel("Temperatura [°C]")
                ax.set_ylabel("Energia [kWh]")
                ax.legend()
                ax.grid(True)

                ax2 = axs[1]
                if len(residui_normal) > 0:
                    count, bins, ignored = ax2.hist(residui_normal, bins=20, density=True, alpha=0.6, color='skyblue',
                                                    edgecolor='black', label='Residui')
                    mu, std = np.mean(residui_normal), np.std(residui_normal)
                    x = np.linspace(min(bins), max(bins), 100)
                    ax2.plot(x, norm.pdf(x, mu, std), 'r--', label='Normale teorica')
                    p_value = shapiro(residui_normal)[1]
                    ax2.set_title(f"Residui - p = {p_value:.3f}")
                else:
                    ax2.set_title("Residui non disponibili")

                ax2.set_xlabel("Residuo [kWh]")
                ax2.set_ylabel("Densità")
                ax2.legend()
                ax2.grid(True)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close()

        df_concat = pd.concat(all_residuals, ignore_index=True)
        df_concat.to_csv(os.path.join(output_df, f"{leaf}.csv"), index=False)
if __name__ == "__main__":
    calc_anm_prob("Cabina", 0.05)