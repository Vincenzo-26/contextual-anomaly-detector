from src.utils import *
import plotly.graph_objects as go
from src.calc_energy_anm import sigmoid_iqr, logistic_regression


def plot_logistic_regression(case_study: str, foglia: str, context: int, cluster: int, save_plot: bool):
    """
    Genera un grafico Plotly che mostra la distribuzione dell'energia per una specifica foglia,
    combinazione di context e cluster, sovrapponendo i dati normali e anomali e la curva di
    probabilità di anomalia stimata.

    Il comportamento si adatta dinamicamente ai dati:
      - Se non ci sono anomalie reali, viene usata una curva sigmoide centrata su max + 1.5*IQR.
      - Se ci sono una o più anomalie, viene usata una regressione logistica per stimare la curva.

    Args:
        case_study (str): Nome del caso di studio (es. 'Cabina').
        foglia (str): Nome del sottocarico da analizzare.
        context (int): Valore del contesto.
        cluster (int): Numero del cluster.
        save_plot (bool): Se True, salva il grafico in HTML; se False, lo mostra a schermo.

    Returns:
        None. Il grafico viene visualizzato o salvato su disco a seconda del flag `save_plot`.
    """
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_LR")
    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")

    df = pd.read_csv(os.path.join(evidence_path, f"evd_{foglia}.csv"))
    anm_table = pd.read_csv(os.path.join(anomaly_path, f"anomaly_table_{foglia}.csv"))

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    anm_table["Date"] = pd.to_datetime(anm_table["Date"]).dt.date

    df_sub = df[(df["Context"] == context) & (df["Cluster"] == cluster)]

    if df_sub.empty:
        print(f"No data for context {context}, cluster {cluster}")
        return

    x_target = df_sub["Energy"].values
    y_real = df_sub["is_real_anomaly"]
    x_normal = x_target[~y_real]
    x_norm_flat = x_normal.flatten()
    x_anomaly = x_target[y_real]

    fig = go.Figure()

    if len(x_anomaly) == 0:
        # Caso 1: nessuna anomalia
        q1, q3 = np.percentile(x_norm_flat, [25, 75])
        iqr = q3 - q1
        threshold = x_norm_flat.max() + 1.5 * iqr
        x_min = min(x_target.min(), threshold) - 2
        x_max = threshold + 5
        x_plot = np.linspace(x_min, x_max, 600)

        anomaly_prob = sigmoid_iqr(x_norm_flat, x_plot, k=6)
        fig.add_vline(x=threshold, line=dict(color='orange', width=1), name='Max + 1.5*IQR')
        title_suffix = f"| Sigmoid (K=6) centered in {threshold:.2f}"

    else:
        x_anomaly = x_anomaly.reshape(-1, 1)
        x_normal_2d = x_normal.reshape(-1, 1)

        x_min = min(x_target.min(), x_anomaly.min(), x_normal.min()) - 5
        x_max = max(x_target.max(), x_anomaly.max(), x_normal.max()) + 5
        x_plot = np.linspace(x_min, x_max, 600)

        anomaly_prob = logistic_regression(x_normal_2d, x_anomaly, x_plot)
        title_suffix = ""

    fig.add_trace(go.Scatter(x=x_plot, y=anomaly_prob, mode='lines', name='Anomaly Probability',
                             line=dict(color='black', width=1),
                             hovertemplate='Energy: %{x:.2f} kWh<br>Anomaly Prob: %{y:.3f}<extra></extra>'))

    def scatter_points_binary(x_vals, y_val, color, name):
        probs_interp = np.interp(x_vals, x_plot, anomaly_prob) * 100
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=[y_val] * len(x_vals),
            mode='markers',
            name=f'{name} ({len(x_vals)})',
            marker=dict(color=color, size=7, symbol='circle'),
            hovertemplate=[
                f'Energy: {float(x):.2f} kWh<br>Anomaly Prob: {float(p):.1f}%<extra></extra>'
                for x, p in zip(x_vals, probs_interp)
            ]
        ))
    scatter_points_binary(x_normal, 0, 'blue', 'Normal points')

    if len(x_anomaly) > 0:
        x_anomaly_1d = x_anomaly.flatten()
        scatter_points_binary(x_anomaly_1d, 1, 'red', 'Anomaly points')

    fig.update_layout(
        title=f"{foglia} | Context {context} - Cluster {cluster}  {title_suffix}",
        xaxis_title="Energy [kWh]",
        yaxis_title="Anomalia [0/1] / Probability [-]",
        title_x=0.5,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if save_plot:
        output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "plot_LR")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{foglia}_ctx{context}_cls{cluster}.html")
        fig.write_html(output_file, include_plotlyjs="cdn")
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
                    plot_logistic_regression(case_study, foglia, ctx, cls, save_plot)

    else:
        plot_logistic_regression(case_study=case_study,
                                 foglia="QE UTA 3_3B_7",
                                 context=3,
                                 cluster=3,
                                 save_plot=save_plot)
