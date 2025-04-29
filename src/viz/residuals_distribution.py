from src.utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from src.change_point_detection import run_change_point

def plot_residuals(case_study: str, sottocarico: str, context: int, cluster: int, residuals, models, change_points, p_value):
    df_normals, df_anm = run_energy_temp(case_study, sottocarico, context, cluster)
    df_normals_sorted = df_normals.sort_values(by="Temperature")

    threshold = 0.05
    residual_title = f" Residui - p_value {round(p_value,2)}>{threshold} -> Normal" if p_value > threshold else \
        (f"p_value {round(p_value,2)}<{threshold} -> Not normal")
    firma_title = f" Firma - {len(change_points)-1} change points"

    fig = make_subplots(rows=1, cols=2, subplot_titles=(firma_title, residual_title))
    fig.add_trace(go.Scatter(
        x=df_normals_sorted["Temperature"],
        y=df_normals_sorted["Energy"],
        mode='markers',
        name="Normal",
        marker=dict(color='skyblue'),
        customdata=np.expand_dims(df_normals_sorted.index.astype(str), axis=1),
        hovertemplate="Data: %{customdata[0]}<br>Temperatura: %{x:.2f}<br>Energia: %{y:.2f}<extra></extra>"
    ), row=1, col=1)

    start = 0
    for i, end in enumerate(change_points):
        X_seg = df_normals_sorted.iloc[start:end]["Temperature"].values
        y_pred = models[i].predict(X_seg.reshape(-1, 1))
        fig.add_trace(go.Scatter(
            x=X_seg,
            y=y_pred,
            mode='lines',
            name=f"Segmento {i+1}",
            hovertemplate="Temperatura: %{x:.2f}<br>Energia: %{y:.2f}<extra></extra>"
        ), row=1, col=1)
        start = end

    for cp in change_points[:-1]:
        x_cp = df_normals_sorted.iloc[cp]["Temperature"]
        fig.add_trace(go.Scatter(
            x=[x_cp, x_cp],
            y=[df_normals_sorted["Energy"].min(), df_normals_sorted["Energy"].max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name="Change point",
            showlegend=False,
            hoverinfo="skip"
        ), row=1, col=1)

    hist = np.histogram(residuals, bins=30)
    bin_edges = hist[1]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    bin_halfwidth = bin_width / 2

    hover_templates = [f"Intervallo: [{x - bin_halfwidth:.2f}, {x + bin_halfwidth:.2f}]<br>Frequenza: {y}"
                       for x, y in zip(bin_centers, hist[0])]

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist[0],
        width=bin_width,
        name="Residui",
        marker_color='skyblue',
        opacity=0.7,
        hovertext=hover_templates,
        hoverinfo="text"
    ), row=1, col=2)

    x_vals = np.linspace(residuals.min(), residuals.max(), 200)
    pdf_vals = norm.pdf(x_vals, loc=np.mean(residuals), scale=np.std(residuals))
    pdf_scaled = pdf_vals * len(residuals) * bin_width
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=pdf_scaled,
        mode='lines',
        name='Normale teorica',
        line=dict(color='black'),
        hoverinfo='skip'
    ), row=1, col=2)

    fig.update_layout(template="plotly_white", title_text=f"{sottocarico}, Context {context}, Cluster {cluster}", title_x=0.5)
    fig.update_xaxes(title_text="Temperatura [°C]", row=1, col=1)
    fig.update_yaxes(title_text="Energia [kWh]", row=1, col=1)
    fig.update_xaxes(title_text="Residui", row=1, col=2)
    fig.update_yaxes(title_text="Frequenza", row=1, col=2)

    fig.show()

if __name__ == "__main__":
    case_study = "Cabina"
    sottocarico = "Rooftop 1"
    context = 3
    cluster = 4

    models, residuals, change_points, p_value = run_change_point(
                                                    case_study=case_study,
                                                    sottocarico=sottocarico,
                                                    context=context,
                                                    cluster=cluster,
                                                    penalty=10
                                                )
    plot_residuals(case_study, sottocarico, context, cluster, residuals, models, change_points, p_value)