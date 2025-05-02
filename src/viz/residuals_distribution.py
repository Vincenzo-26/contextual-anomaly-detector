import os
import numpy as np
import ruptures as rpt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, normaltest
from sklearn.metrics import r2_score
from src.utils import PROJECT_ROOT, run_energy_temp


def plot_residuals(case_study: str, sottocarico: str, context: int, cluster: int, penalty: int = 10):
    df_normals, df_anomalies = run_energy_temp(case_study, sottocarico, context, cluster)
    if df_normals is None or df_normals.empty:
        print("⚠️ Dati insufficienti.")
        return

    df_sorted = df_normals.sort_values(by="Temperature").dropna(subset=["Temperature", "Energy"])
    df_sorted = df_sorted[df_sorted["Energy"] > 0]

    signal = df_sorted[["Temperature", "Energy"]].values
    algo = rpt.Pelt(model="rbf").fit(signal)
    change_points = algo.predict(pen=penalty)

    residuals_by_segment = []
    segment_labels = []
    thermal_sensitive_indices = []

    start = 0
    for i, end in enumerate(change_points):
        segment = df_sorted.iloc[start:end]
        X = segment["Temperature"].values.reshape(-1, 1)
        y = segment["Energy"].values
        if len(segment) < 2:
            start = end
            continue

        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        corr = np.corrcoef(X.flatten(), y)[0, 1]
        slope = model.coef_[0]
        r2 = r2_score(y, y_pred)
        is_sensitive = abs(corr) > 0.5 and abs(slope) > 0.2 and r2 > 0.5

        if is_sensitive:
            residuals = y - y_pred
            residuals_by_segment.append((residuals, i))
            segment_labels.append(f"Segmento {i+1}")
            thermal_sensitive_indices.append((start, end, model))

        start = end

    n_sens = len(residuals_by_segment)
    fig = make_subplots(
        rows=1, cols=1 + n_sens,
        column_widths=[0.5] + [0.5 / n_sens] * n_sens if n_sens > 0 else [1],
        subplot_titles=["Firma Energetica"] + [f"Residui {label}" for label in segment_labels]
    )

    # Reset start per il ciclo corretto
    start = 0
    for i, end in enumerate(change_points):
        segment = df_sorted.iloc[start:end]
        X = segment["Temperature"].values.reshape(-1, 1)
        y = segment["Energy"].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        corr = np.corrcoef(X.flatten(), y)[0, 1]
        slope = model.coef_[0]
        r2 = r2_score(y, y_pred)
        is_sensitive = abs(corr) > 0.5 and abs(slope) > 0.2 and r2 > 0.5

        # Calcolo dei limiti orizzontali del segmento
        x0 = df_sorted.iloc[start]["Temperature"]
        x1 = df_sorted.iloc[end - 1]["Temperature"]
        fillcolor = "rgba(0,200,0,1)" if is_sensitive else "rgba(200,0,0,1)"

        # Riempimento verticale per il segmento
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=0, y1=1,
            xref="x1", yref="paper",
            fillcolor=fillcolor,
            opacity=0.2,
            line=dict(width=0),
            layer="below"
        )

        start = end  # aggiorno lo start alla fine del ciclo

    # Punti reali
    fig.add_trace(go.Scatter(
        x=df_sorted["Temperature"], y=df_sorted["Energy"],
        mode='markers', name="Dati reali", marker=dict(color='skyblue'),
        customdata=df_sorted.index,
        hovertemplate='Data: %{customdata}<br>Temperatura: %{x:.2f}°C<br>Energia: %{y:.2f} kWh<extra></extra>'
    ), row=1, col=1)

    # Aggiunta scatter delle anomalie
    if df_anomalies is not None and not df_anomalies.empty:
        df_anomalies_sorted = df_anomalies.sort_values(by="Temperature").dropna(subset=["Temperature", "Energy"])
        fig.add_trace(go.Scatter(
            x=df_anomalies_sorted["Temperature"],
            y=df_anomalies_sorted["Energy"],
            mode='markers',
            name="Anomalie (CMP)",
            marker=dict(color='red', size=6),
            customdata=df_anomalies_sorted.index,
            hovertemplate='Data: %{customdata}<br>Temperatura: %{x:.2f}°C<br>Energia: %{y:.2f} kWh<extra></extra>'
        ), row=1, col=1)

    # Linee di regressione e change point
    start = 0
    for i, end in enumerate(change_points):
        segment = df_sorted.iloc[start:end]
        X = segment["Temperature"].values.reshape(-1, 1)
        y = segment["Energy"].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        # Regressione
        fig.add_trace(go.Scatter(
            x=X.flatten(), y=y_pred, mode='lines', name=f"Segmento {i+1}",
            hovertemplate='Temperatura: %{x:.2f}°C<br>Predizione: %{y:.2f} kWh'
        ), row=1, col=1)

        # Linea tratteggiata verticale solo se non è l'ultimo change point
        if i < len(change_points) - 1:
            x_cp = df_sorted.iloc[end - 1]["Temperature"]
            fig.add_trace(go.Scatter(
                x=[x_cp, x_cp],
                y=[df_sorted["Energy"].min(), df_sorted["Energy"].max()],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ), row=1, col=1)

        start = end

    # Plot dei residui
    for idx, (residuals, seg_index) in enumerate(residuals_by_segment):
        hist, edges = np.histogram(residuals, bins=30)
        bin_width = edges[1] - edges[0]
        mu, sigma = np.mean(residuals), np.std(residuals)
        x_norm = np.linspace(min(residuals), max(residuals), 100)
        y_norm = norm.pdf(x_norm, mu, sigma) * len(residuals) * bin_width

        fig.add_trace(go.Bar(
            x=edges[:-1], y=hist, opacity=0.6,
            name=f"Residui Segmento {seg_index+1}",
            marker_color='steelblue',
            hovertemplate=f"[%{{x:.2f}} ± {bin_width/2:.2f}]<br>Frequenza: %{{y}}<extra></extra>"
        ), row=1, col=2 + idx)

        fig.add_trace(go.Scatter(
            x=x_norm, y=y_norm,
            mode='lines', line=dict(dash='dot', color='black'),
            name=f"Normale {seg_index+1}", showlegend=False
        ), row=1, col=2 + idx)

    fig.update_layout(
        title=f"{sottocarico} | Context {context} | Cluster {cluster}",
        template="plotly_white",
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    fig.update_xaxes(title_text="Temperatura [°C]", row=1, col=1)
    fig.update_yaxes(title_text="Energia [kWh]", row=1, col=1)
    for i in range(n_sens):
        fig.update_xaxes(title_text="Residui", row=1, col=2+i)
        fig.update_yaxes(title_text="Frequenza", row=1, col=2+i)

    fig.show()



if __name__ == "__main__":
    plot_residuals(
        case_study="Cabina",
        sottocarico="Rooftop 5",
        context=1,
        cluster=3,
        penalty=10
    )
