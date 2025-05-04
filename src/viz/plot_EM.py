from src.utils import *
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks


def plot_EM(case_study: str, foglia: str, context: int, cluster: int):
    # Paths
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_EM")
    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")

    # Carica i dati
    df = pd.read_csv(os.path.join(evidence_path, f"evd_{foglia}.csv"))
    anm_table = pd.read_csv(os.path.join(anomaly_path, f"anomaly_table_{foglia}.csv"))

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    anm_table["Date"] = pd.to_datetime(anm_table["Date"]).dt.date

    # Filtro context e cluster
    df_sub = df[(df["Context"] == context) & (df["Cluster"] == cluster)]
    df_sub = df_sub[df_sub["Energy"] != 0]

    if df_sub.empty:
        print(f"No data for context {context}, cluster {cluster}")
        return

    # Dati energy
    X = df_sub["Energy"].values
    y_real = df_sub["is_real_anomaly"]

    # Split
    X_normal = X[~y_real]
    X_anomaly = X[y_real]

    # Range x per plot
    x_plot = np.linspace(min(X) - 2, max(X) + 2, 500)

    fig = go.Figure()

    # === Istogrammi ===
    fig.add_trace(go.Histogram(
        x=X_normal,
        nbinsx=30,
        name='Normal histogram',
        marker_color='rgba(0,0,255,0.3)',
        opacity=0.5,
        histnorm='probability density',
        showlegend=True
    ))

    if len(X_anomaly) > 0:
        fig.add_trace(go.Histogram(
            x=X_anomaly,
            nbinsx=30,
            name='Anomaly histogram',
            marker_color='rgba(255,0,0,0.3)',
            opacity=0.5,
            histnorm='probability density',
            showlegend=True
        ))

    # Fit GMM ai normal
    X_normal_reshape = X_normal.reshape(-1, 1)
    counts, _ = np.histogram(X_normal.flatten(), bins=30)
    peaks, _ = find_peaks(counts, height=0.05 * np.max(counts))
    k_estimated = min(max(1, len(peaks)), 3)

    gmm_normal = GaussianMixture(n_components=k_estimated, covariance_type='full', random_state=0)
    gmm_normal.fit(X_normal_reshape)
    p_x_normal = np.exp(gmm_normal.score_samples(x_plot.reshape(-1, 1)))

    fig.add_trace(go.Scatter(
        x=x_plot, y=p_x_normal,
        mode='lines',
        name='Normal GMM',
        line=dict(color='blue', width=1),
        fill='tozeroy',
        opacity=0.5,
        hovertemplate='Energy: %{x:.2f}<br>PDF: %{y:.5f}<extra></extra>'
    ))

    if len(X_anomaly) >= 2:
        X_anomaly_reshape = X_anomaly.reshape(-1, 1)
        gmm_anomaly = GaussianMixture(n_components=1, covariance_type='full', random_state=0)
        gmm_anomaly.fit(X_anomaly_reshape)
        p_x_anomaly = np.exp(gmm_anomaly.score_samples(x_plot.reshape(-1, 1)))

        fig.add_trace(go.Scatter(
            x=x_plot, y=p_x_anomaly,
            mode='lines',
            name='Anomaly GMM',
            line=dict(color='red', width=1),
            fill='tozeroy',
            opacity=0.5,
            hovertemplate='Energy: %{x:.2f}<br>PDF: %{y:.5f}<extra></extra>'
        ))

        n_normal = len(X_normal)
        n_anomaly = len(X_anomaly)
        prior_normal = n_normal / (n_normal + n_anomaly)
        prior_anomaly = n_anomaly / (n_normal + n_anomaly)

        anomaly_prob = (p_x_anomaly * prior_anomaly) / (p_x_anomaly * prior_anomaly + p_x_normal * prior_normal + 1e-10)

        # Saturazione continua asintotica per x > max_anomaly_energy
        max_anomaly_energy = X_anomaly.max()
        alpha = 1.5
        idx_start = np.argmax(x_plot > max_anomaly_energy)
        if idx_start < len(x_plot):
            base_prob = anomaly_prob[idx_start - 1] if idx_start > 0 else anomaly_prob[idx_start]
            for i in range(idx_start, len(x_plot)):
                delta = x_plot[i] - max_anomaly_energy
                anomaly_prob[i] = base_prob + (1 - base_prob) * (1 - np.exp(-alpha * delta))
                anomaly_prob[i] = min(anomaly_prob[i], 1.0)

    else:
        print(f"Warning: insufficient anomaly data for context {context}, cluster {cluster}. Using sigmoid anomaly probability.")
        max_normal = X_normal.max()
        k = 6
        z = k * (x_plot - max_normal)
        z = np.clip(z, -500, 500)
        anomaly_prob = 1 / (1 + np.exp(-z))

    fig.add_trace(go.Scatter(
        x=x_plot, y=anomaly_prob,
        mode='lines',
        name='Anomaly Probability',
        line=dict(color='black', width=1),
        hovertemplate='Energy: %{x:.2f}<br>Anomaly Prob: %{y:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=X_normal, y=[max(p_x_normal.max(), 1) * 1.05] * len(X_normal),
        mode='markers',
        name=f'Normal points ({len(X_normal)})',
        marker=dict(color='blue', size=6, symbol='circle'),
        hovertemplate='Energy: %{x:.2f}<extra></extra>'
    ))

    if len(X_anomaly) > 0:
        fig.add_trace(go.Scatter(
            x=X_anomaly, y=[max(p_x_normal.max(), 1) * 1.05] * len(X_anomaly),
            mode='markers',
            name=f'Anomaly points ({len(X_anomaly)})',
            marker=dict(color='red', size=6, symbol='circle'),
            hovertemplate='Energy: %{x:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=f"{foglia} - Context {context} - Cluster {cluster}",
        xaxis_title="Energy",
        yaxis_title="PDF / Probability",
        title_x=0.5,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[
            dict(
                text=f"Normal points: {len(X_normal)} - Anomaly points: {len(X_anomaly)}",
                xref="paper", yref="paper",
                x=0, y=-0.25,
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )

    fig.show()


if __name__ == "__main__":
    plot_EM(case_study="Cabina", foglia="QE UTA 3_3B_7", context=1, cluster=3)