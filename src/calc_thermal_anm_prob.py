import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from utils import run_energy_temp
from settings import PROJECT_ROOT
from src.utils import get_nodes_by_level
from scipy.stats import norm
import matplotlib.colors as mcolors


def calc_anm_prob(case_study: str, soglia_en_max: float):
    """
    Calcola la probabilità di anomalia energetica per ciascun subload, in base ai residui rispetto a segmenti interpolati.

    L'analisi prevede:
        - Lettura della segmentazione termica giornaliera per ciascun sottocarico (se presente).
        - Estrazione dei dati normali e anomali da `run_energy_temp`.
        - Filtraggio rispetto a una soglia (percentuale del massimo nei dati normali).
        - Interpolazione per segmenti su profili ON e weekday normal.
        - Calcolo dei residui e della probabilità di anomalia (funzione gaussiana se il residuo è positivo).

    Args:
        - case_study (str): Nome del caso studio.
        - soglia_en_max (float): Percentuale (es. 0.10) del consuno massimo per quella combinazione di
          context-cluster-subload usata come soglia ON/OFF.

    Returns:
        None.
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json")) as f:
        config = json.load(f)

    output_folder_viz = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "thermal_sensitivity", "ctx_thermal_sens")
    os.makedirs(output_folder_viz, exist_ok=True)
    output_df = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "ctx_thermal_sens")
    os.makedirs(output_df, exist_ok=True)

    context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
    groups = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv"), parse_dates=["timestamp"])
    cluster_cols = [col for col in groups.columns if col.startswith("Cluster_")]
    levels = get_nodes_by_level(config["Load Tree"])

    for leaf in levels[0]:
        df_sens_path = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "daily_thermal_sens", f"segs_{leaf}.csv")
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
                if df_normals is not None and not df_normals.empty:
                    df_normals = df_normals.reset_index().rename(columns={"index": "Date"})
                else:
                    print("Nessun punto normal.")
                    continue

                df_normals = df_normals.dropna(subset=["Temperature", "Energy"])
                df_normals["Status"] = np.where(df_normals["Energy"] >= energy_cutoff, "ON", "OFF")
                df_normals["anomalous"] = False

                if df_anomalous is not None and not df_anomalous.empty:
                    df_anomalous = df_anomalous.reset_index().rename(columns={"index": "Date"})
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

                # === CHANGE POINT DETECTION ===
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

                # === SEGMENTI E INTERPOLAZIONE ===
                segments = []
                for i in range(len(change_points)):
                    start = 0 if i == 0 else change_points[i - 1]
                    end = change_points[i]
                    seg_x = df_on["Temperature"].values[start:end]
                    seg_y = df_on["Energy"].values[start:end]
                    coeffs = np.polyfit(seg_x, seg_y, deg=1)
                    Tmin, Tmax = seg_x.min(), seg_x.max()
                    segments.append((np.poly1d(coeffs), Tmin, Tmax))

                # === CALCOLO RESIDUI (solo per i dati ON) ===
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
                df_all[["Residual", "prob_anomaly"]] = df_all.apply(compute_residual_prob, axis=1, result_type="expand")
                df_all["Subload"] = leaf
                df_all["Context"] = context
                df_all["Cluster"] = cluster
                all_residuals.append(df_all)

                # === PLOT ===
                output_leaf_folder = os.path.join(output_folder_viz, leaf)
                os.makedirs(output_leaf_folder, exist_ok=True)
                fig_path = os.path.join(output_leaf_folder, f"{leaf}_ctx{context}_cl{cluster}.png")

                fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})
                ax = axs[0]

                cmap = plt.get_cmap("coolwarm")
                norm_color = mcolors.Normalize(vmin=0, vmax=1)

                df_off = df_all[df_all["Status"] == "OFF"]
                ax.scatter(df_off["Temperature"], df_off["Energy"], color="gray", alpha=0.3, label="OFF", marker='o')

                df_on_norm = df_all[(df_all["Status"] == "ON") & (df_all["anomalous"] == False)]
                sc1 = ax.scatter(
                    df_on_norm["Temperature"], df_on_norm["Energy"],
                    c=df_on_norm["prob_anomaly"], cmap=cmap, norm=norm_color,
                    alpha=0.8, label="ON", marker='o'
                )

                df_on_anm = df_all[(df_all["Status"] == "ON") & (df_all["anomalous"] == True)]
                sc2 = ax.scatter(
                    df_on_anm["Temperature"], df_on_anm["Energy"],
                    c=df_on_anm["prob_anomaly"], cmap=cmap, norm=norm_color,
                    alpha=0.8, label="ON Anomalous", marker='^'
                )

                ax.axhline(energy_cutoff, color="orange", linewidth=1, linestyle=":", label="Soglia ON/OFF")

                cb = fig.colorbar(sc2, ax=ax)
                cb.set_label("Prob. Anomalia")

                ax.set_title(f"{leaf} | ctx {context} - cl {cluster}")
                ax.set_xlabel("Temperatura")
                ax.set_ylabel("Energia")
                ax.legend()
                ax.grid(True)

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
                    ax2.set_title(f"Residui")
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