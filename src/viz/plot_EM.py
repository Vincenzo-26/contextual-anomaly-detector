from src.utils import *
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
from src.calc_energy_distr import sigmoid_iqr, sigmoid_single_point, compute_gmm_anomaly_probability

def plot_EM(case_study: str, foglia: str, context: int, cluster: int, save_plot:bool, one_anm_peak=False, height: float = 0.6):
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

    df_sub = df[(df["Context"] == context) & (df["Cluster"] == cluster)]

    if df_sub.empty:
        print(f"No data for context {context}, cluster {cluster}")
        return

    X_target = df_sub["Energy"].values
    y_real = df_sub["is_real_anomaly"]
    X_normal = X_target[~y_real]
    X_norm_flat = X_normal.flatten()
    X_anomaly = X_target[y_real]

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
        peaks, _ = find_peaks(counts, height=height * np.max(counts))
        k_est = min(max(1, len(peaks)), 3)
        return GaussianMixture(n_components=k_est, covariance_type='full', random_state=0)

    if len(X_anomaly) == 0:
        # Caso 1: nessuna anomalia
        q1, q3 = np.percentile(X_norm_flat, [25, 75])
        iqr = q3 - q1
        threshold = X_norm_flat.max() + 1.5 * iqr
        x_min = min(X_target.min(), threshold) - 2
        x_max = threshold + 5
        x_plot = np.linspace(x_min, x_max, 600)

        anomaly_prob = sigmoid_iqr(X_norm_flat, x_plot, k=6)
        fig.add_vline(x=threshold, line=dict(color='orange', width=1), name='Max + 1.5*IQR')
        title_suffix = f"Sigmoid (K=6) centered in {threshold:.2f}"

    elif len(X_anomaly) == 1:
        # Caso 2: un solo punto anomalo
        x_anom = float(X_anomaly[0])
        x0_est = x_anom - np.log(0.95 / (1 - 0.95)) / 6
        x_max_extended = x_anom + (x_anom - x0_est) * 2
        x_min = min(X_target.min(), x0_est) - 2
        x_max = max(X_target.max(), x_max_extended)
        x_plot = np.linspace(x_min, x_max, 600)

        anomaly_prob, x0, k = sigmoid_single_point(x_anom, x_plot, X_norm_flat, base_k=6, target_prob=0.95)
        fig.add_vline(x=x0, line=dict(color='orange', width=1, dash='dot'), name=f'Sigmoid Center (k={k:.2f})')
        title_suffix = f"Sigmoid (K={k:.2f}) - Single anomaly"

    else:
        # Caso 3
        if len(np.unique(X_anomaly)) == 1:
            # Caso 3.1: Tutti i punti anomali hanno lo stesso valore → sigmoide come caso 2
            x_anom = float(X_anomaly[0])
            x0_est = x_anom - np.log(0.95 / (1 - 0.95)) / 6
            x_max_extended = x_anom + (x_anom - x0_est) * 2
            x_min = min(X_target.min(), x0_est) - 2
            x_max = max(X_target.max(), x_max_extended)
            x_plot = np.linspace(x_min, x_max, 600)

            anomaly_prob, x0, k = sigmoid_single_point(x_anom, x_plot, X_norm_flat, base_k=6, target_prob=0.95)
            fig.add_vline(x=x0, line=dict(color='orange', width=1, dash='dot'), name=f'Sigmoid Center (k={k:.2f})')
            title_suffix = f"Sigmoid (K={k:.2f}) - {len(X_anomaly)} identical anomalies"

        else:
            # Caso 3.2: Almeno 2 anomalie diverse → GMM + formula di Bayes + regolazione bordi
            X_anomaly = X_anomaly.reshape(-1, 1)
            X_normal_2d = X_normal.reshape(-1, 1)

            x_min = min(X_target.min(), X_anomaly.min(), X_normal.min()) - 5
            x_max = max(X_target.max(), X_anomaly.max(), X_normal.max()) + 5
            x_plot = np.linspace(x_min, x_max, 600)

            anomaly_prob, p_x_normal, p_x_anomaly, k_norm, k_anom = compute_gmm_anomaly_probability(
                x_plot, X_normal_2d, X_anomaly, height, one_anm_peak
            )

            fig.add_trace(go.Scatter(x=x_plot, y=p_x_normal, mode='lines', name='Normal GMM',
                                     line=dict(color='blue', width=1), fill='tozeroy', opacity=0.5,
                                     hovertemplate='Energy: %{x:.2f} kWh<br>PDF: %{y:.5f}<extra></extra>'))

            fig.add_trace(go.Scatter(x=x_plot, y=p_x_anomaly, mode='lines', name='Anomaly GMM',
                                     line=dict(color='red', width=1), fill='tozeroy', opacity=0.5,
                                     hovertemplate='Energy: %{x:.2f} kWh<br>PDF: %{y:.5f}<extra></extra>'))

            title_suffix = f"GMM with {k_norm} norm peak(s) - {k_anom} anm peak(s)"


    fig.add_trace(go.Scatter(x=x_plot, y=anomaly_prob, mode='lines', name='Anomaly Probability',
                             line=dict(color='black', width=1),
                             hovertemplate='Energy: %{x:.2f} kWh<br>Anomaly Prob: %{y:.3f}<extra></extra>'))

    def scatter_points(X_vals, color, name, probs):
        fig.add_trace(go.Scatter(
            x=X_vals,
            y=[max(anomaly_prob.max(), 1) * 1.05] * len(X_vals),
            mode='markers',
            name=f'{name} ({len(X_vals)})',
            marker=dict(color=color, size=6, symbol='circle'),
            hovertemplate=[
                f'Energy: {float(x):.2f} kWh<br>Anomaly Prob: {float(p):.1f}%<extra></extra>'
                for x, p in zip(X_vals, probs)
            ]
        ))

    y_prob_normal = np.interp(X_normal, x_plot, anomaly_prob) * 100
    scatter_points(X_normal, 'blue', 'Normal points', y_prob_normal)

    if len(X_anomaly) > 0:
        X_anomaly_1d = X_anomaly.flatten()
        y_prob_anomaly = np.interp(X_anomaly_1d, x_plot, anomaly_prob) * 100
        scatter_points(X_anomaly_1d, 'red', 'Anomaly points', y_prob_anomaly)

    fig.update_layout(
        title=f"{foglia} | Context {context} - Cluster {cluster} | {title_suffix}",
        xaxis_title="Energy [kWh]",
        yaxis_title="PDF / Probability [-]",
        title_x=0.5,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if save_plot:
        output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "plot_GMM_EM")
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
                    plot_EM(case_study, foglia, ctx, cls, save_plot)

    else:
        plot_EM(case_study=case_study, foglia="QE UTA 3_3B_7", context=3, cluster=3, save_plot=save_plot)