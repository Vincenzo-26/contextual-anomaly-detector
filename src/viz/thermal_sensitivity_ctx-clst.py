import plotly.graph_objs as go

from src.utils import *


def plot_thermal_sensitivity_ctx_clst(case_study: str, sottocarico: str, context: int, cluster: int):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    df_normals, df_anomalies = run_energy_temp(case_study, sottocarico, context, cluster)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_normals["Temperature"],
        y=df_normals["Energy"],
        mode='markers',
        text=df_normals.index.astype(str),
        marker=dict(size=6, color="Blue"),
        name="Normal",
        hovertemplate="Date: %{text}<br>Temp: %{x:.2f} °C<br>Energy: %{y:.2f} kWh<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_anomalies["Temperature"],
        y=df_anomalies["Energy"],
        mode='markers',
        text=df_anomalies.index.astype(str),
        marker=dict(size=6, color="red"),
        name="Anomalies (CMP)",
        hovertemplate="Date: %{text}<br>Temp: %{x:.2f} °C<br>Energy: %{y:.2f} kWh<extra></extra>",
    ))

    fig.update_layout(
        title=f"{sottocarico}, Context {context}, Cluster {cluster}",
        xaxis_title="Temperature [°C]",
        yaxis_title="Energy [kWh]",
        template="plotly_white",
        title_x=0.5
    )

    fig.show()


if __name__ == "__main__":
    plot_thermal_sensitivity_ctx_clst(
        case_study="Cabina",
        sottocarico="QE UTA 1_1B_5",
        context=3,
        cluster=4
    )
