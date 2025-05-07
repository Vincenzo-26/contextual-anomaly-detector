from src.utils import *
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
from src.calc_energy_distr import sigmoid_iqr, sigmoid_single_point, extend_probability_tail

def plot_EM(case_study: str, foglia: str, context: int, cluster: int, one_anm_peak=True):
    """
    Genera un grafico che mostra la distribuzione dell'energia per il contesto e cluster specificato,
    sovrapponendo le curve GMM per dati normali e anomali, e la probabilità di anomalia stimata.

    Args:
        case_study (str): Nome del caso studio.
        foglia (str): Nome della foglia dell'albero dei carichi da analizzare.
        context (int): Contesto.
        cluster (int): Cluster.
        one_anm_peak (bool): Se True, usa una sola componente GMM per le anomalie, se False calcola in automatico
        il numero di componenti GMM (default = True).

    Returns:
        None
    """
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_EM")
    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")

    df = pd.read_csv(os.path.join(evidence_path, f"evd_{foglia}.csv"))
    anm_table = pd.read_csv(os.path.join(anomaly_path, f"anomaly_table_{foglia}.csv"))

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    anm_table["Date"] = pd.to_datetime(anm_table["Date"]).dt.date

    # df_sub = df[(df["Context"] == context) & (df["Cluster"] == cluster) & (df["Energy"] != 0)]
    df_sub = df[(df["Context"] == context) & (df["Cluster"] == cluster)]

    if df_sub.empty:
        print(f"No data for context {context}, cluster {cluster}")
        return

    X = df_sub["Energy"].values
    y_real = df_sub["is_real_anomaly"]
    X_normal = X[~y_real]
    X_anomaly = X[y_real]

    q1, q3 = np.percentile(X_normal, [25, 75])
    iqr = q3 - q1
    shifted_threshold = X_normal.max() + 1.5 * iqr
    x_min, x_max = min(X) - 2, max(shifted_threshold, max(X)) + 2
    x_plot = np.linspace(x_min, x_max, 500)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=X_normal, nbinsx=30, name='Normal histogram',
                               marker_color='rgba(0,0,255,0.3)', opacity=0.5, histnorm='probability density'))
    if len(X_anomaly) > 0:
        fig.add_trace(go.Histogram(x=X_anomaly, nbinsx=30, name='Anomaly histogram',
                                   marker_color='rgba(255,0,0,0.3)', opacity=0.5, histnorm='probability density'))

    def estimate_gmm(X_data, one_peak=False):
        if one_peak:
            return GaussianMixture(n_components=1, covariance_type='full', random_state=0)
        counts, _ = np.histogram(X_data.flatten(), bins=30)
        peaks, _ = find_peaks(counts, height=0.05 * np.max(counts))
        k_est = min(max(1, len(peaks)), 3)
        return GaussianMixture(n_components=k_est, covariance_type='full', random_state=0)

    X_normal_reshape = X_normal.reshape(-1, 1)
    gmm_normal = estimate_gmm(X_normal)
    gmm_normal.fit(X_normal_reshape)
    p_x_normal = np.exp(gmm_normal.score_samples(x_plot.reshape(-1, 1)))
    fig.add_trace(go.Scatter(x=x_plot, y=p_x_normal, mode='lines', name='Normal GMM',
                             line=dict(color='blue', width=1), fill='tozeroy', opacity=0.5,
                             hovertemplate='Energy: %{x:.2f} kWh<br>PDF: %{y:.5f}<extra></extra>'))

    if len(X_anomaly) >= 2:
        gmm_anomaly = estimate_gmm(X_anomaly, not one_anm_peak)
        gmm_anomaly.fit(X_anomaly.reshape(-1, 1))
        p_x_anomaly = np.exp(gmm_anomaly.score_samples(x_plot.reshape(-1, 1)))
        fig.add_trace(go.Scatter(x=x_plot, y=p_x_anomaly, mode='lines', name='Anomaly GMM',
                                 line=dict(color='red', width=1), fill='tozeroy', opacity=0.5,
                                 hovertemplate='Energy: %{x:.2f} kWh<br>PDF: %{y:.5f}<extra></extra>'))

        n_normal, n_anomaly = len(X_normal), len(X_anomaly)
        prior_normal = n_normal / (n_normal + n_anomaly)
        prior_anomaly = n_anomaly / (n_normal + n_anomaly)
        anomaly_prob = (p_x_anomaly * prior_anomaly) / (p_x_anomaly * prior_anomaly + p_x_normal * prior_normal + 1e-10)
        anomaly_prob = extend_probability_tail(x_plot, anomaly_prob, X_anomaly.max())
        min_threshold = np.percentile(X_normal, 1)
        anomaly_prob[x_plot < min_threshold] = 0
        fig.add_vline(x=min_threshold, line=dict(color='purple', width=1, dash='dash'), name='1° percentile Normal')

    elif len(X_anomaly) == 1:
        anomaly_prob, threshold = sigmoid_single_point(X_anomaly[0], x_plot, 6)
        fig.add_vline(x=threshold, line=dict(color='orange', width=1), name='Sigmoid Center')

    else:
        anomaly_prob = sigmoid_iqr(X_normal, x_plot, 6)
        fig.add_vline(x=shifted_threshold, line=dict(color='orange', width=1), name='Max + 1.5*IQR')

    fig.add_trace(go.Scatter(x=x_plot, y=anomaly_prob, mode='lines', name='Anomaly Probability',
                             line=dict(color='black', width=1),
                             hovertemplate='Energy: %{x:.2f} kWh<br>Anomaly Prob: %{y:.3f}<extra></extra>'))

    def scatter_points(X_vals, color, name, probs):
        fig.add_trace(go.Scatter(
            x=X_vals,
            y=[max(p_x_normal.max(), 1) * 1.05] * len(X_vals),
            mode='markers',
            name=f'{name} ({len(X_vals)})',
            marker=dict(color=color, size=6, symbol='circle'),
            hovertemplate=[
                f'Energy: {x:.2f} kWh<br>Anomaly Prob: {p:.1f}%<extra></extra>'
                for x, p in zip(X_vals, probs)
            ]
        ))

    y_prob_normal = np.interp(X_normal, x_plot, anomaly_prob) * 100
    scatter_points(X_normal, 'blue', 'Normal points', y_prob_normal)

    if len(X_anomaly) > 0:
        y_prob_anomaly = np.interp(X_anomaly, x_plot, anomaly_prob) * 100
        scatter_points(X_anomaly, 'red', 'Anomaly points', y_prob_anomaly)

    fig.update_layout(
        title=f"{foglia} | Context {context} - Cluster {cluster}",
        xaxis_title="Energy [kWh]",
        yaxis_title="PDF / Probability [-]",
        title_x=0.5,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "plot_GMM_EM")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{foglia}_ctx{context}_cls{cluster}.html")
    fig.write_html(output_file, include_plotlyjs="cdn")

if __name__ == "__main__":

    case_study = "Cabina"

    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)
    foglie = find_leaf_nodes(config["Load Tree"])
    for foglia in foglie:
        for cls in range(1, 6):
            for ctx in range(1, 6):
                plot_EM(case_study=case_study, foglia=foglia, context=ctx, cluster=cls)