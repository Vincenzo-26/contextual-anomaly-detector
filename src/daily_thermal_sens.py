import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import numpy as np
import pandas as pd
import ruptures as rpt
from settings import PROJECT_ROOT
from src.utils import get_nodes_by_level, scale_data
import warnings
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, ConstantInputWarning
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt



def check_thermal_sensitivity(df_segment, norm_method: str or bool, corr_thresh=0.5, r2_thresh=0.2, slope_thresh=20):
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

    x = scale_data(X_raw, norm_method)
    y = scale_data(y_raw, norm_method).flatten()

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
        "scaling_used": norm_method,
        "corr_thresh": corr_thresh,
        "slope_thresh": slope_thresh,
        "r2_thresh": r2_thresh
    }


def percentile_post_primo_picco(df, bins):
    values = df["Energy"].dropna().values

    counts, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(counts)
    if len(peaks) == 0:
        return 0.25

    first_peak_idx = peaks[0]
    for i in range(first_peak_idx + 1, len(counts)):
        if counts[i] < 0.5 * counts[first_peak_idx]:  # soglia decrescita configurabile
            threshold_value = bin_centers[i]
            break
    else:
        threshold_value = bin_centers[first_peak_idx + 1]
    percentile = (values < threshold_value).mean()

    return percentile


def find_thermal_sensitivity(case_study: str, ruptures_penalty: int, filter_method: any, norm_method_for_check: str):
    """
    method (any) 'GMM' oppure 'peak, ' oppure '(quantile, 0.25)'
    """
    def color_text(text, color):
        color_codes = {"red": "\033[91m", "green": "\033[92m", "reset": "\033[0m"}
        return f"{color_codes[color]}{text}{color_codes['reset']}"
    print(f"\nMethod: {filter_method[0]}\nPenalty: {ruptures_penalty}\nNorm method form thermal sensitivity: {norm_method_for_check}\n")
    with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
        config = json.load(f)

    output_folder = os.path.join(PROJECT_ROOT, "results", case_study, "prova", "daily_thermal_sensitivity")
    output_folder_viz = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "daily_thermal_sensitivity",
                                     f"{filter_method[0]}", f"pen{ruptures_penalty}")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_viz, exist_ok=True)

    levels = get_nodes_by_level(config["Load Tree"])
    first_level = levels[0]
    for leaf in first_level:
        segs_results = []

        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{leaf}.csv"), index_col=0, parse_dates=True)
        temp_file = config["Outside Temperature"]
        df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{temp_file}.csv"), index_col=0, parse_dates=True)

        df = df.merge(df_temp, left_index=True, right_index=True)
        df.columns = ["Power", "Temperature"]
        df = df.resample("1D").agg({"Power": "sum", "Temperature": "mean"})
        df["Energy"] = df["Power"] * 0.25 / 1000

        # === FILTRAGGIO ===
        df = df[df.index.weekday < 5]

        if filter_method == "peak":
            percentile_dynamic = percentile_post_primo_picco(df, 100)
            threshold = df["Energy"].quantile(percentile_dynamic)
            df["Mode"] = np.where(df["Energy"] > threshold, "on", "off")
            df = df[df["Mode"] == "on"]

        elif filter_method == "GMM":
            values = df["Energy"].dropna().values.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=0).fit(values)
            labels = gmm.predict(values)
            low_component = np.argmin(gmm.means_.flatten())
            df["Mode"] = "off"
            df.loc[df["Energy"].dropna().index[labels != low_component], "Mode"] = "on"
            df = df[df["Mode"] == "on"]

        elif isinstance(filter_method, tuple) and filter_method[0] == "quantile":
            q = filter_method[1]
            threshold = df["Energy"].quantile(q)
            df["Mode"] = np.where(df["Energy"] > threshold, "on", "off")
            df = df[df["Mode"] == "on"]
        else:
            raise ValueError("Metodo di filtraggio non riconosciuto.")


        # === CHANGE POINT E THERMAL SENSITIVITY ===
        df = df.sort_values(by="Temperature")
        signal = df[["Temperature", "Energy"]].values
        algo_model = "linear"
        algo = rpt.Pelt(model=algo_model).fit(signal)
        change_points = algo.predict(pen=ruptures_penalty)
        change_points = [cp for cp in change_points if cp < len(df)]
        segs = [(start, end) for start, end in zip([0] + change_points, change_points + [len(df)])]

        print(f"\n[Leaf: {leaf}] -> {len(segs)} segmenti trovati con {len(change_points)} change point")

        for i, (start, end) in enumerate(segs):
            if end - start < 2:
                continue
            seg = df.iloc[start:end]
            metrics = check_thermal_sensitivity(seg, norm_method_for_check)
            result = {
                "Segmento": i + 1,
                "start_idx": start,
                "end_idx": end,
                "t_min": round(seg["Temperature"].min(), 2),
                "t_max": round(seg["Temperature"].max(), 2),
                "Thermal Sensitive": metrics["is_thermal_sensitive"],
                "spearmann": metrics["correlation"],
                "slope": metrics["slope"],
                "r2": metrics["r2_score"]
            }
            color = "green" if result["Thermal Sensitive"] else "red"
            temp_range_str = (
                f"    Segmento {i + 1}  ->  T: [{result['t_min']}°C, {result['t_max']}°C]  "
                f"| slope: {result['slope']:.3f} | r2: {result['r2']:.3f} | spearman: {result['spearmann']:.3f}"
            )
            print(color_text(temp_range_str, color))
            segs_results.append(result)

        pd.DataFrame(segs_results).to_csv(os.path.join(output_folder, f"segs_{leaf}.csv"), index=False)

        # === PLOT ===
        df_all = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{leaf}.csv"), index_col=0,
                             parse_dates=True)
        df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{temp_file}.csv"), index_col=0,
                              parse_dates=True)
        df_all = df_all.merge(df_temp, left_index=True, right_index=True)
        df_all.columns = ["Power", "Temperature"]
        df_all = df_all.resample("1D").agg({"Power": "sum", "Temperature": "mean"})

        df_all["Energy"] = df_all["Power"] * 0.25 / 1000
        on_indices = df[df["Mode"] == "on"].index
        df_all["Mode"] = "off"
        df_all.loc[on_indices, "Mode"] = "on"
        df_all = df_all.sort_values(by="Temperature")
        weekend_dates = df_all[df_all.index.to_series().dt.weekday >= 5]
        weekday_df = df_all[df_all.index.to_series().dt.weekday < 5]
        fig, ax = plt.subplots(figsize=(10, 6))
        # Scatter dei punti
        ax.scatter(weekend_dates["Temperature"], weekend_dates["Energy"], color="orange", label="weekend")
        ax.scatter(weekday_df[weekday_df["Mode"] == "off"]["Temperature"],
                   weekday_df[weekday_df["Mode"] == "off"]["Energy"],
                   color="lightgray", label="off (weekday)")

        ax.scatter(weekday_df[weekday_df["Mode"] == "on"]["Temperature"],
                   weekday_df[weekday_df["Mode"] == "on"]["Energy"],
                   color="#4a90e2", label="on (weekday)")
        # Change points
        for i, cp in enumerate(change_points):
            t_cp = df.iloc[cp]["Temperature"]
            ax.axvline(t_cp, color="black", linewidth=1, label="Change point" if i == 0 else None)
        # === FILL DINAMICO ===
        # Estremi dei segmenti
        temp_bounds = [df.iloc[0]["Temperature"]]  # inizio
        temp_bounds += [df.iloc[cp]["Temperature"] for cp in change_points]  # change points
        temp_bounds += [df.iloc[-1]["Temperature"]]  # fine
        for i in range(len(temp_bounds) - 1):
            t_start = temp_bounds[i]
            t_end = temp_bounds[i + 1]
            is_sensitive = segs_results[i]["Thermal Sensitive"]
            color_fill = "green" if is_sensitive else "red"
            ax.axvspan(t_start, t_end, color=color_fill, alpha=0.15, zorder=0)
        # === INTERPOLAZIONE LINEARE PER SEGMENTO ===
        for i, (start, end) in enumerate(segs):
            if end - start < 2:
                continue
            seg = df.iloc[start:end]
            x = seg["Temperature"].values.reshape(-1, 1)
            y = seg["Energy"].values
            model = LinearRegression().fit(x, y)
            x_fit = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            y_fit = model.predict(x_fit)
            ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        # Etichette e salvataggio
        ax.set_xlabel("Mean temperature [°C]")
        ax.set_ylabel("Daily energy [kWh]")
        ax.set_title(f"{leaf} - algo:{algo_model} - {filter_method} - penalty: {ruptures_penalty}")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(output_folder_viz, f"{leaf}.png"), dpi=300)
        plt.close()

    return


if __name__ == "__main__":
    penalty = 1000
    find_thermal_sensitivity("Cabina", penalty, 'peak', False)
    # find_thermal_sensitivity("Cabina", penalty, ('quantile',0.25), False)