from src.utils import *
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from src.utils import PROJECT_ROOT


def plot_residuals(case_study: str, sottocarico: str, context: int, cluster: int, save_plot: bool):
    base_dir = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity")
    path_segments = os.path.join(base_dir, "segments", f"segment_{sottocarico}.csv")
    path_residuals = os.path.join(base_dir, "residuals", f"residuals_{sottocarico}.csv")

    if not os.path.exists(path_segments) or not os.path.exists(path_residuals):
        print("❌ File non trovati.")
        return

    df_segments = pd.read_csv(path_segments)
    df_residuals = pd.read_csv(path_residuals)

    # Filtra per combinazione specifica
    df_segments = df_segments[(df_segments["Context"] == context) & (df_segments["Cluster"] == cluster)]
    df_residuals = df_residuals[(df_residuals["Context"] == context) & (df_residuals["Cluster"] == cluster)]

    if df_segments.empty or df_residuals.empty:
        print("⚠️ Nessun dato per la combinazione specificata.")
        return

    modes = sorted(df_segments["Mode"].unique())
    mode_colors = {0: "#1f77b4", 1: "#d62728"}

    fig = go.Figure()

    for mode in modes:
        df_mode = df_residuals[df_residuals["Mode"] == mode]
        df_normal = df_mode[df_mode["is_real_anomaly"] == False]
        df_anomalous = df_mode[df_mode["is_real_anomaly"] == True]

        # Punti normali
        n_normal = len(df_normal)
        fig.add_trace(go.Scatter(
            x=df_normal["Temperature"],
            y=df_normal["Energy"],
            mode="markers",
            marker=dict(color=mode_colors.get(mode, "gray"), size=5, symbol="circle"),
            name=f"Mode {mode} - Normali ({n_normal})",
            hovertemplate=(
                "Temp: %{x:.2f}<br>"
                "Energia: %{y:.2f}<br>"
                f"Mode: {mode}<br>"
                "Segmento: %{customdata}"
            ),
            customdata=df_normal["assigned_segment"]
        ))

        # Punti anomali
        n_anomalous = len(df_anomalous)
        fig.add_trace(go.Scatter(
            x=df_anomalous["Temperature"],
            y=df_anomalous["Energy"],
            mode="markers",
            marker=dict(color=mode_colors.get(mode, "gray"), size=7, symbol="x"),
            name=f"Mode {mode} - Anomali ({n_anomalous})",
            hovertemplate=(
                "Temp: %{x:.2f}<br>"
                "Energia: %{y:.2f}<br>"
                f"Mode: {mode}<br>"
                "Segmento: %{customdata}"
            ),
            customdata=df_anomalous["assigned_segment"]
        ))

        df_mode_seg = df_segments[df_segments["Mode"] == mode]
        for seg_id in df_mode_seg["Segmento"].unique():
            seg_info = df_mode_seg[df_mode_seg["Segmento"] == seg_id].iloc[0]
            tmin, tmax = seg_info["t_min"], seg_info["t_max"]
            is_sensitive = seg_info["Thermal Sensitive"]

            df_seg = df_mode[(df_mode["assigned_segment"] == seg_id) & (~df_mode["is_real_anomaly"])]
            if df_seg.empty:
                continue

            X = df_seg["Temperature"].values.reshape(-1, 1)
            y = df_seg["Energy"].values
            model = LinearRegression().fit(X, y)

            x_line = np.linspace(df_seg["Temperature"].min(), df_seg["Temperature"].max(), 100)
            y_line = model.predict(x_line.reshape(-1, 1))
            sensitive_status = "✅ Thermal Sensitive" if is_sensitive else "❌ Non Sensitive"
            corr_val = seg_info.get("correlation", "NA")
            slope_val = seg_info.get("slope", "NA")
            r2_val = seg_info.get("r2_score", "NA")

            corr_thresh = seg_info.get("corr_thresh", "NA")
            slope_thresh = seg_info.get("slope_thresh", "NA")
            r2_thresh = seg_info.get("r2_thresh", "NA")

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=f"rgba{(0, 128, 0, 0.4)}" if is_sensitive else f"rgba{(255, 0, 0, 0.4)}",width=2),
                name=f"Segment {seg_id} (Mode {mode})",
                hovertemplate=(
                    f"{sensitive_status}<br>"
                    f"Mode {mode} - Segmento: {seg_id}<br>"
                    f"Corr: {corr_val} (>{corr_thresh})<br>"
                    f"Slope: {slope_val} (>{slope_thresh})<br>"
                    f"R²: {r2_val} (>{r2_thresh})<br>"
                    "Temp: %{x:.2f}<br>"
                    "Energia: %{y:.2f}<extra></extra>"
                )
            ))

    fig.update_layout(
        title=f"{sottocarico} | Context {context} – Cluster {cluster}",
        xaxis_title="Temperature",
        yaxis_title="Energy",
        template="plotly_white"
    )

    if save_plot:
        output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "plot_thermal_sens")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{sottocarico}_ctx{context}_cls{cluster}.html")
        fig.write_html(output_file, include_plotlyjs="cdn")
        print(f"Saved {sottocarico}_ctx{context}_cls{cluster}.html")
    else:
        fig.show()




if __name__ == "__main__":
    case_study = "Cabina"
    save_plot = True

    if save_plot:
        with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
            config = json.load(f)
        foglie = find_leaf_nodes(config["Load Tree"])
        for foglia in foglie:
            for cls in range(1, 6):
                for ctx in range(1, 6):
                    plot_residuals(case_study, foglia, ctx, cls, save_plot)

    else:
        plot_residuals(
            case_study="Cabina",
            sottocarico="QE UTA 1_1B_5",
            context=4,
            cluster=3,
            save_plot=save_plot
        )