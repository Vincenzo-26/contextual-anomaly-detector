from src.utils import *
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from src.utils import PROJECT_ROOT
from scipy.stats import norm


def plot_residuals(case_study: str, sottocarico: str, context: int, cluster: int, save_plot: bool, alpha = 25):
    import plotly.express as px

    base_dir = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity_only_GMM")
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

    specs = [[{"colspan": 2}, None]]  # Prima riga: scatter generale
    subplot_titles = ["Scatter generale – tutte le modalità", ""]

    layout_domains = {}  # Per centrare le xaxis dinamicamente

    for idx, mode in enumerate(modes):
        is_sensitive = df_segments[df_segments["Mode"] == mode]["Thermal Sensitive"].iloc[0]
        row_number = idx + 2  # +2 perché la prima riga è il grafico generale

        if is_sensitive:
            specs.append([{}, {}])
            subplot_titles += [f"Mode {mode} – Firma Energetica", f"Mode {mode} – Residui"]
        else:
            specs.append([{}, None])
            subplot_titles += [f"Mode {mode} – Firma Energetica", ""]
            layout_domains[f"xaxis{2 * row_number - 1}"] = dict(domain=[0.25, 0.75])

    fig = make_subplots(
        rows=len(specs),
        cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=0.05,
        column_widths=[0.5, 0.5]
    )

    segment_colors = px.colors.qualitative.Plotly

    # RIGA 1 – Scatter generale
    for mode in modes:
        df_mode = df_residuals[df_residuals["Mode"] == mode]
        df_normal = df_mode[df_mode["is_real_anomaly"] == False]
        df_anomalous = df_mode[df_mode["is_real_anomaly"] == True]
        color = segment_colors[mode % len(segment_colors)]

        fig.add_trace(go.Scatter(
            x=df_normal["Temperature"],
            y=df_normal["Energy"],
            mode="markers",
            marker=dict(color=color, size=6),
            name=f"Mode {mode} - Normal",
            legendgroup=f"mode_{mode}",
            showlegend=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_anomalous["Temperature"],
            y=df_anomalous["Energy"],
            mode="markers",
            marker=dict(color=color, size=6, symbol="x"),
            name=f"Mode {mode} - Anomalous",
            legendgroup=f"mode_{mode}",
            showlegend=True
        ), row=1, col=1)

    # MODI INDIVIDUALI
    for i, mode in enumerate(modes):
        row = i + 2
        is_sensitive = df_segments[df_segments["Mode"] == mode]["Thermal Sensitive"].iloc[0]
        color = segment_colors[mode % len(segment_colors)]
        df_mode = df_residuals[df_residuals["Mode"] == mode]
        df_normal = df_mode[df_mode["is_real_anomaly"] == False]
        df_anomalous = df_mode[df_mode["is_real_anomaly"] == True]

        # Scatter
        fig.add_trace(go.Scatter(
            x=df_normal["Temperature"],
            y=df_normal["Energy"],
            mode="markers",
            marker=dict(color=color, size=4),
            name=f"Mode {mode} - Normal",
            showlegend=False
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=df_anomalous["Temperature"],
            y=df_anomalous["Energy"],
            mode="markers",
            marker=dict(color=color, size=6, symbol="x"),
            name=f"Mode {mode} - Anomalous",
            showlegend=False
        ), row=row, col=1)

        if is_sensitive:
            # Regressione
            X = df_normal["Temperature"].values.reshape(-1, 1)
            y = df_normal["Energy"].values
            model = LinearRegression().fit(X, y)
            x_line = np.linspace(X.min(), X.max(), 100)
            y_line = model.predict(x_line.reshape(-1, 1))

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False
            ), row=row, col=1)

            # Istogramma dei residui
            residuals = df_normal["residual"].dropna().values
            if len(residuals) > 0:
                fig.add_trace(go.Histogram(
                    x=residuals,
                    nbinsx=25,
                    marker_color="green",
                    opacity=0.7,
                    marker_line_color="black",
                    marker_line_width=1,
                    showlegend=False
                ), row=row, col=2)

                # PDF teorica
                mean_r = residuals.mean()
                std_r = residuals.std()
                x_vals = np.linspace(residuals.min(), residuals.max(), 200)
                y_vals = norm.pdf(x_vals, mean_r, std_r)
                y_scaled = y_vals * len(residuals) * (residuals.max() - residuals.min()) / 25

                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_scaled,
                    mode="lines",
                    line=dict(color="black"),
                    showlegend=False
                ), row=row, col=2)

    fig.update_layout(
        height=700 * len(specs),
        width=1200,
        title=f"{sottocarico} | Context {context} – Cluster {cluster}",
        template="plotly_white",
        **layout_domains  # ← Qui viene applicato il centramento dinamico
    )

    if save_plot:
        output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "viz", f"plot_thermal_sens_only_GMM_alpha{alpha}")
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
            sottocarico="Rooftop 3",
            context=3,
            cluster=4,
            save_plot=save_plot
        )