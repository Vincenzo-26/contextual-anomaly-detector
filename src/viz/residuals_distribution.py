from src.utils import *
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from src.utils import PROJECT_ROOT
from scipy.stats import norm


def plot_residuals(case_study:str, sottocarico:str, context:int, cluster:int, save_plot:bool):
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
    has_multiple_modes = len(modes) > 1

    # Conta quanti segmenti sensibili ci sono per ogni modalità (max per le colonne)
    max_ts_segments = df_segments[df_segments["Thermal Sensitive"] == True].groupby("Mode")["Segmento"].nunique().max()
    max_ts_segments = max_ts_segments if pd.notna(max_ts_segments) else 0

    n_rows = len(modes) + (1 if has_multiple_modes else 0)
    n_cols = 1 + max_ts_segments

    # Titoli subplot
    subplot_titles = []
    if has_multiple_modes:
        subplot_titles += ["Firma Generale"] + [""] * (n_cols - 1)
    for mode in modes:
        subplot_titles.append(f"Mode {mode} – Firma Energetica")
        ts_segments = df_segments[(df_segments["Mode"] == mode) & (df_segments["Thermal Sensitive"] == True)]["Segmento"].unique()
        for seg_id in ts_segments:
            subplot_titles.append(f"Residui – Segmento {seg_id}")
        subplot_titles += [""] * (n_cols - 1 - len(ts_segments))

    if n_cols == 1:
        column_widths = [1.0]
    else:
        column_widths = [0.5] + [0.5 / (n_cols - 1)] * (n_cols - 1)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=False,
        horizontal_spacing=0.05,
        vertical_spacing=0.12,
        subplot_titles=subplot_titles,
        column_widths=column_widths
    )
    current_row = 1

    if has_multiple_modes:
        colors = ["IndianRed", "LightSeaGreen", "purple", "cyan", "magenta"]

        for i, mode in enumerate(modes):
            df_mode = df_residuals[df_residuals["Mode"] == mode]
            df_normal = df_mode[df_mode["is_real_anomaly"] == False]
            df_anomalous = df_mode[df_mode["is_real_anomaly"] == True]

            # Normali (cerchi)
            fig.add_trace(
                go.Scatter(
                    x=df_normal["Temperature"],
                    y=df_normal["Energy"],
                    mode="markers",
                    marker=dict(color=colors[i % len(colors)], size=4),
                    name=f"Mode {mode} - Normal",
                    showlegend=True
                ),
                row=current_row,
                col=1
            )

            # Anomali (X)
            fig.add_trace(
                go.Scatter(
                    x=df_anomalous["Temperature"],
                    y=df_anomalous["Energy"],
                    mode="markers",
                    marker=dict(color=colors[i % len(colors)], size=6, symbol='x'),
                    name=f"Mode {mode} - Anomalous",
                    showlegend=True
                ),
                row=current_row,
                col=1
            )
        current_row += 1

    for mode in modes:
        df_mode = df_residuals[df_residuals["Mode"] == mode]
        df_mode_seg = df_segments[df_segments["Mode"] == mode]

        df_normal = df_mode[df_mode["is_real_anomaly"] == False]
        fig.add_trace(
            go.Scatter(
                x=df_normal["Temperature"],
                y=df_normal["Energy"],
                mode="markers",
                marker=dict(color="blue", size=4),
                name=f"Mode {mode} - Normal",
                showlegend=False
            ),
            row=current_row,
            col=1
        )

        df_anomalous = df_mode[df_mode["is_real_anomaly"] == True]
        fig.add_trace(
            go.Scatter(
                x=df_anomalous["Temperature"],
                y=df_anomalous["Energy"],
                mode="markers",
                marker=dict(color="red", size=4),
                name=f"Mode {mode} - Anomalous",
                showlegend=False
            ),
            row=current_row,
            col=1
        )

        segments = df_mode_seg["Segmento"].unique()
        ts_col_idx = 2  # da colonna 2 in poi per i residui
        for seg_id in segments:
            seg_info = df_mode_seg[df_mode_seg["Segmento"] == seg_id].iloc[0]
            tmin, tmax = seg_info["t_min"], seg_info["t_max"]
            is_sensitive = seg_info["Thermal Sensitive"]

            # Tutti i punti del segmento (per stabilire l'intervallo)
            df_seg_all = df_mode[(df_mode["assigned_segment"] == seg_id) |
                                 ((df_mode["Temperature"] >= tmin) & (df_mode["Temperature"] <= tmax))]

            # Solo punti normali per il fit
            df_seg_fit = df_seg_all[df_seg_all["is_real_anomaly"] == False]

            if df_seg_fit.empty:
                continue

            fill_color = "rgba(0,200,0,0.1)" if is_sensitive else "rgba(200,0,0,0.1)"

            fig.add_vrect(
                x0=tmin,
                x1=tmax,
                fillcolor=fill_color,
                layer="below",
                line_width=0,
                row=current_row,
                col=1
            )

            X = df_seg_fit["Temperature"].values.reshape(-1, 1)
            y = df_seg_fit["Energy"].values
            model = LinearRegression().fit(X, y)

            x_line = np.linspace(df_seg_all["Temperature"].min(), df_seg_all["Temperature"].max(), 100)
            y_line = model.predict(x_line.reshape(-1, 1))

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(
                        color="black",
                        width=1,
                        dash="solid"
                    ),
                    showlegend=False
                ),
                row=current_row,
                col=1
            )

            if is_sensitive:
                df_seg_points_resid = df_seg_fit.dropna(subset=["residual"])
                if not df_seg_points_resid.empty:
                    residuals = df_seg_points_resid["residual"].values
                    mean_r = residuals.mean()
                    std_r = residuals.std()

                    fig.add_trace(
                        go.Histogram(
                            x=residuals,
                            nbinsx=30,
                            marker_color="green",
                            opacity=0.6,
                            showlegend=False
                        ),
                        row=current_row,
                        col=ts_col_idx
                    )

                    x_vals = np.linspace(residuals.min(), residuals.max(), 200)
                    y_vals = norm.pdf(x_vals, mean_r, std_r)
                    y_scaled = y_vals * len(residuals) * (residuals.max() - residuals.min()) / 30

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_scaled,
                            mode="lines",
                            line=dict(color="black"),
                            showlegend=False
                        ),
                        row=current_row,
                        col=ts_col_idx
                    )
                    ts_col_idx += 1

        current_row += 1

    fig.update_layout(
        height=600 * n_rows,
        title_text=f"{sottocarico} | Context {context} – Cluster {cluster}",
        template="plotly_white"
    )

    if save_plot:
        output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "plot_thermal_sens")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{foglia}_ctx{context}_cls{cluster}.html")
        fig.write_html(output_file, include_plotlyjs="cdn")
        print(f"Saved {foglia}_ctx{context}_cls{cluster}.html")
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
            sottocarico="Rooftop 3",
            context=3,
            cluster=3,
            save_plot=save_plot
        )