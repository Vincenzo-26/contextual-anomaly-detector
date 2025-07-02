import os

os.environ["OMP_NUM_THREADS"] = "1"
import json
import numpy as np
import pandas as pd
import ruptures as rpt
from settings import PROJECT_ROOT
from src.utils import get_nodes_by_level, scale_data, find_leaf_nodes
import warnings
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, ConstantInputWarning
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps


def check_thermal_sens(df_segment, x_min: float, x_max: float, y_min: float, y_max: float,
                       corr_thresh=0.3, r2_thresh=0.2, slope_thresh=0.03):
    """
    Verifica se un segmento è termicamente sensibile usando correlazione, pendenza e R².

    Args:
        - df_segment (pd.DataFrame): Segmento con colonne 'Temperature' e 'Energy'.
        - norm_method (str or bool): Metodo di normalizzazione ('Zscore', 'minmax', 'robust', 'maxabs') o False se non
          si normalizza
        - corr_thresh (float): Soglia di correlazione.
        - r2_thresh (float): Soglia R².
        - slope_thresh (float): Soglia della pendenza.

    Returns:
        dict: Metriche calcolate e flag 'is_thermal_sensitive'.
    """
    X_raw = df_segment[["Temperature"]]
    y_raw = df_segment["Energy"].values.reshape(-1, 1)

    # Correlazione di Spearman
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        try:
            corr, _ = spearmanr(X_raw.values.flatten(), y_raw.flatten())
        except Exception:
            print("⚠️ Segmento con input costante: impossibile calcolare la correlazione.")
            corr = np.nan

    x = X_raw
    y = ((y_raw - y_min) / (y_max - y_min)).flatten()

    model = LinearRegression().fit(x, y)
    slope = model.coef_[0]
    r2 = r2_score(y, model.predict(x))

    # is_sensitive = (abs(corr) > corr_thresh) and (abs(slope) > slope_thresh) and (r2 > r2_thresh)
    is_sensitive = (abs(corr) > corr_thresh) and (abs(slope) > slope_thresh)

    return {
        "correlation": round(corr, 3),
        "slope": round(slope, 3),
        "r2_score": round(r2, 3),
        "is_thermal_sensitive": is_sensitive,
        "corr_thresh": corr_thresh,
        "slope_thresh": slope_thresh,
        "r2_thresh": r2_thresh
    }


def find_thermal_sens(case_study: str, ruptures_penalty: float, norm_method_for_check: any):
    """
    Analizza la sensibilità termica giornaliera per ciascun subload dell'albero dei carichi.

    L'analisi prevede:
        - Aggregazione giornaliera dei dati di potenza e temperatura.
        - Filtraggio dei giorni feriali e dei giorni con consumo sotsopra la soglia di 10% del massimo.
        - Clustering di mode differenti tramite GMM (1 vs 2 componenti, selezione via BIC).
        - Segmentazione per ciascun mode tramite ruptures.
        - Verifica della sensibilità termica per ciascun segmento di ciascun mode.
        - Plot dei risultati con segmenti (verde = sensibile, rosso = non sensibile).

    Args:
        - case_study (str): Nome del caso studio.
        - ruptures_penalty (int): Penalità da passare al metodo PELT di ruptures.
        - norm_method_for_check (str): Metodo di normalizzazione da usare nel controllo di sensibilità termica
          ('Zscore', 'minmax', 'robust', 'maxabs') oppure False per non normalizzare.

    Returns:
        None.

    Note: I risultati salvati come file .csv sono presenti solo per i subload con almeno un segmento thermal sensitive
          mentre i plot .png nelle cartelle in tutti i casi.
    """

    def color_text(text, color):
        color_codes = {"red": "\033[91m", "green": "\033[92m", "reset": "\033[0m"}
        return f"{color_codes[color]}{text}{color_codes['reset']}"

    print(
        f"\nRuptures penalty: {ruptures_penalty}\nNorm method form checking thermal sensitivity: {norm_method_for_check}\n")
    with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
        config = json.load(f)

    output_folder = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "daily_thermal_sens")
    output_folder_viz = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "thermal_sensitivity",
                                     "daily_thermal_sens")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_viz, exist_ok=True)

    # levels = get_nodes_by_level(config["Load Tree"])
    # first_level = levels[0]
    leaves = find_leaf_nodes(config["Load Tree"])
    for leaf in leaves:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{leaf}.csv"), index_col=0, parse_dates=True)
        temp_file = config["Outside Temperature"]
        df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{temp_file}.csv"), index_col=0,
                              parse_dates=True)

        df = df.merge(df_temp, left_index=True, right_index=True)
        df.columns = ["Power", "Temperature"]
        df["Energy"] = df["Power"] * 0.25
        df = df.resample("1D").agg({"Power": "mean", "Temperature": "mean", "Energy": "sum"})

        # === FILTRAGGIO ===
        df = df[df.index.weekday < 5]
        energia_max = df["Energy"].max()
        threshold = energia_max * 0.10

        df["Mode"] = np.where(df["Energy"] > threshold, "on", "off")
        df = df[df["Mode"] == "on"]

        # === GMM per rilevare mode operativi ===
        X_energy = df["Energy"].values.reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X_energy)
        gmm2 = GaussianMixture(n_components=2, random_state=0).fit(X_energy)

        bic_diff = gmm1.bic(X_energy) - gmm2.bic(X_energy)
        bic_threshold = 50

        if bic_diff > bic_threshold:
            labels = gmm2.predict(X_energy)
            counts = pd.Series(labels).value_counts()
            if (counts >= 10).any():  # almeno un cluster con >= 10 punti
                df["ModeID"] = labels
            else:
                df["ModeID"] = 0
        else:
            df["ModeID"] = 0

        segs_results = []
        algo_model = "linear"

        fig, ax = plt.subplots(figsize=(10, 6))
        df["Temp_norm"] = scale_data(df[["Temperature"]], "minmax").flatten()
        df["Energy_norm"] = scale_data(df[["Energy"]], "minmax").flatten()
        for mode_id in sorted(df["ModeID"].unique()):
            df_mode = df[df["ModeID"] == mode_id].sort_values(by="Temperature")
            signal = df_mode[["Temp_norm", "Energy_norm"]].values
            algo = rpt.Pelt(model=algo_model).fit(signal)
            change_points = algo.predict(pen=ruptures_penalty)
            change_points = [cp for cp in change_points if cp < len(df_mode)]
            segs = [(start, end) for start, end in zip([0] + change_points, change_points + [len(df_mode)])]

            if mode_id == sorted(df["ModeID"].unique())[0]:
                print(f"[{leaf}]")

            for i, (start, end) in enumerate(segs):
                if end - start < 2:
                    continue
                seg = df_mode.iloc[start:end]
                metrics = check_thermal_sens(seg, x_min=df["Temperature"].min(), x_max=df["Energy"].max(),
                                             y_min=df["Energy"].min(), y_max=df["Energy"].max())
                result = {
                    "ModeID": mode_id,
                    "Segmento": i + 1,
                    "t_min": round(seg["Temperature"].min(), 2),
                    "t_max": round(seg["Temperature"].max(), 2),
                    "Thermal Sensitive": metrics["is_thermal_sensitive"],
                    "spearmann": metrics["correlation"],
                    "slope": metrics["slope"],
                    "r2": metrics["r2_score"]
                }
                color = "green" if result["Thermal Sensitive"] else "red"
                temp_range_str = (
                    f"Mode {mode_id} - Seg {i + 1}  ->  ΔT: [{result['t_min']}°C, {result['t_max']}°C]  "
                    f"- slope: {result['slope']:.3f} | spearman: {result['spearmann']:.3f}"
                )
                print(color_text(temp_range_str, color))
                segs_results.append(result)

                # Plot interpolazione
                x = seg["Temperature"].values.reshape(-1, 1)
                y = seg["Energy"].values
                model = LinearRegression().fit(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
                y_fit = model.predict(x_fit)
                ax.plot(x_fit, y_fit, color=color, linestyle="-", linewidth=2.0, alpha=0.9)
        print("\n")
        pd.DataFrame(segs_results).to_csv(os.path.join(output_folder, f"segs_{leaf}.csv"), index=False)

        # === PLOT ===
        df_all = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{leaf}.csv"), index_col=0,
                             parse_dates=True)
        df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{temp_file}.csv"), index_col=0,
                              parse_dates=True)
        df_all = df_all.merge(df_temp, left_index=True, right_index=True)
        df_all.columns = ["Power", "Temperature"]
        df_all = df_all.resample("1D").agg({"Power": "sum", "Temperature": "mean"})

        df_all["Energy"] = df_all["Power"] * 0.25
        on_indices = df[df["Mode"] == "on"].index
        df_all["Mode"] = "off"
        df_all.loc[on_indices, "Mode"] = "on"
        df_all["ModeID"] = np.nan
        df_all.loc[df.index, "ModeID"] = df["ModeID"]

        df_all = df_all.sort_values(by="Temperature")
        weekday_df = df_all[df_all.index.to_series().dt.weekday < 5]

        weekday_off_df = weekday_df[weekday_df["Mode"] == "off"]
        ax.scatter(weekday_off_df["Temperature"], weekday_off_df["Energy"],
                   color="gray", label="Off", s=30, alpha=0.6)

        weekday_on_df = weekday_df[weekday_df["Mode"] == "on"]
        unique_modes = sorted(df["ModeID"].dropna().unique())
        cmap = colormaps["tab10"].resampled(len(unique_modes))
        colors = {mode_id: mcolors.to_hex(cmap(i)) for i, mode_id in enumerate(unique_modes)}

        for mode_id in unique_modes:
            df_mode = weekday_on_df[weekday_on_df["ModeID"] == mode_id]
            ax.scatter(df_mode["Temperature"], df_mode["Energy"],
                       label=f"Mode {mode_id+1} ",
                       s=30, alpha=0.9, color=colors[mode_id])
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.5, label="On-Off theshold")
        ax.set_xlabel("Daily mean temperature [°C]", fontsize=16)
        ax.set_ylabel("Daily energy [kWh]", fontsize=16)
        ax.set_title(f"{leaf}", fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True)
        ax.legend(fontsize=14)
        fig.tight_layout()
        plt.savefig(os.path.join(output_folder_viz, f"{leaf}.png"), dpi=300)
        plt.close()
    return


if __name__ == "__main__":
    find_thermal_sens("Total_cut", 2.56, "minmax")
