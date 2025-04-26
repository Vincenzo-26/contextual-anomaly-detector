import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
from src.utils import *
import plotly.express as px
import plotly.graph_objects as go

def plot_energy_distribution(df: pd.DataFrame, context: str, cluster: str, anomaly_df: pd.DataFrame = None):
    """
    Visualizza la distribuzione dei valori di energia per uno specifico (context, cluster),
    mostrando la KDE dei dati normali e sovrapponendo i valori anomali.
    """
    df = df.astype({"Context": int, "Cluster": int})
    if anomaly_df is not None:
        anomaly_df = anomaly_df.astype({"Context": int, "Cluster": int})

    # Dati non anomali = df esclusi quelli presenti in anomaly_df per Date, Context, Cluster
    if anomaly_df is not None:
        df_clean = df.merge(anomaly_df[["Date", "Context", "Cluster"]],
                            on=["Date", "Context", "Cluster"],
                            how="left", indicator=True)
        df_clean = df_clean[df_clean["_merge"] == "left_only"].drop(columns=["_merge"])
    else:
        df_clean = df.copy()

    subset = df_clean[(df_clean["Context"] == int(context)) & (df_clean["Cluster"] == int(cluster))]
    X = subset["Energy"].values.reshape(-1, 1)
    if len(X) < 2:
        print("⚠️ Troppi pochi dati per visualizzare la distribuzione.")
        return

    kde = KernelDensity(kernel="gaussian", bandwidth=0.3).fit(X)
    x_plot = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    log_dens = kde.score_samples(x_plot)

    plt.figure(figsize=(10, 5))
    plt.plot(x_plot[:, 0], np.exp(log_dens), label="KDE")
    plt.scatter(X[:, 0], np.full_like(X[:, 0], -0.0005), alpha=0.6, marker="|", label="normali")

    if anomaly_df is not None:
        anomalous = df.merge(anomaly_df[["Date", "Context", "Cluster"]],
                             on=["Date", "Context", "Cluster"], how="inner")
        anomalous = anomalous[(anomalous["Context"] == int(context)) & (anomalous["Cluster"] == int(cluster))]
        if not anomalous.empty:
            plt.scatter(anomalous["Energy"], np.full_like(anomalous["Energy"], -0.0008),
                        color="red", alpha=1.0, marker="x", s=100, linewidths=2, label="anomali")

    plt.title(f"Distribuzione energia - Context {context}, Cluster {cluster}")
    plt.xlabel("Energy [Wh]")
    plt.ylabel("Densità stimata")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_energy_scatter_by_cluster(df: pd.DataFrame, df_anomalie: pd.DataFrame = None, title: str = "Distribuzione energia per cluster"):
    """
    Crea un grafico a dispersione con Plotly per visualizzare i valori di energia,
    colorati in base al cluster di appartenenza. Se fornito un secondo DataFrame con anomalie,
    evidenzia i punti anomali con una 'X' sovrapposta, associata alla legenda del cluster.

    Args:
        df (pd.DataFrame): DataFrame con almeno le colonne 'Energy', 'Cluster', 'Date', 'Context'.
        df_anomalie (pd.DataFrame, optional): DataFrame con anomalie, con le colonne 'Date', 'Context', 'Cluster'.
        title (str): Titolo del grafico.
    """
    required_cols = {"Energy", "Cluster", "Date", "Context"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Il DataFrame deve contenere le colonne: {required_cols}")

    df_plot = df.copy()
    df_plot["Index"] = range(len(df_plot))

    # grafico base
    fig = px.scatter(
        df_plot,
        x="Index",
        y="Energy",
        color=df_plot["Cluster"].astype(str),
        labels={"color": "Cluster", "Index": "Indice", "Energy": "Energia [Wh]"},
        title=title
    )

    # aggiunta X per anomalie per ogni cluster
    if df_anomalie is not None:
        df_anom = pd.merge(df_plot, df_anomalie, on=["Date", "Context", "Cluster"], how="inner")

        for cluster in sorted(df_anom["Cluster"].unique()):
            cluster_anom = df_anom[df_anom["Cluster"] == cluster]
            fig.add_trace(go.Scatter(
                x=cluster_anom["Index"],
                y=cluster_anom["Energy"],
                mode="markers",
                marker=dict(symbol="x", size=10),
                name=f"Anomalia Cluster {cluster}",
                legendgroup=str(cluster),
                showlegend=True
            ))

    fig.update_layout(template="plotly_white")
    fig.show()

def plot_hdbscan_clusters(df, foglia, main_cluster_id, save_path=None):
    """
    Crea un grafico scatter con i punti ordinati per energia e colorati in base al sottocluster HDBSCAN.

    Args:
        df (pd.DataFrame): DataFrame con colonne "Energy" e "HDBSCAN_Label".
        foglia (str): Nome del nodo foglia (per titolo e salvataggio).
        main_cluster_id (int): ID del cluster principale.
        save_path (str): Se specificato, salva l'immagine in questa path.
    """
    # Ordina per energia
    df_plot = df.sort_values("Energy").reset_index(drop=True)
    df_plot["Rank"] = df_plot.index

    fig = px.scatter(
        df_plot,
        x="Rank",
        y="Energy",
        color=df_plot["HDBSCAN_Label"].astype(str),
        title=f"{foglia} - Cluster {main_cluster_id}: Sottocluster HDBSCAN (ordinati per energia)",
        labels={"color": "Sotto-cluster", "Rank": "Posizione ordinata"}
    )

    fig.update_layout(height=500)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
    else:
        fig.show()


if __name__ == "__main__":

    case_study = "Cabina"
    sottocarico = "QE UTA 4_4B_8"

    evd_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_KDE_PDF")
    anm_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")

    df = pd.read_csv(os.path.join(evd_path, f"evd_{sottocarico}.csv"))
    df_anm = pd.read_csv(os.path.join(anm_path, f"anomaly_table_{sottocarico}.csv"))

    plot_energy_distribution(df, 3, 3, df_anm)
    # plot_energy_scatter_by_cluster(df, df_anm)