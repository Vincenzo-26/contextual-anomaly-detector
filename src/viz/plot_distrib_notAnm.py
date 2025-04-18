import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
from src.utils import *

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


if __name__ == "__main__":
    case_study = "Cabina"
    sottocarico = "QE UTA 3_3B_7"
    evd_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences")
    anm_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")
    df = pd.read_csv(os.path.join(evd_path, f"evd_{sottocarico}.csv"))
    df_anm = pd.read_csv(os.path.join(anm_path, f"anomaly_table_{sottocarico}.csv"))
    plot_energy_distribution(df, 3, 3, df_anm)