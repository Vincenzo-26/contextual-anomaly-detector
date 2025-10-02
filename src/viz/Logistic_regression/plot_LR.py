from src.utils import *
import matplotlib.pyplot as plt
from src.non_thermal_sens_node_anm import sigmoid_iqr, logistic_regression


def plot_logistic_regression(case_study: str, foglia: str, context: int, cluster: int, save_plot: bool):
    """
    Genera un grafico matplotlib che mostra la distribuzione dell'energia per una specifica foglia,
    combinazione di context e cluster, sovrapponendo i dati normali e anomali e la curva di
    probabilità di anomalia stimata.

    Args:
        case_study (str): Nome del caso di studio.
        foglia (str): Nome del sottocarico.
        context (int): Valore del contesto.
        cluster (int): Numero del cluster.
        save_plot (bool): Se True salva il grafico, altrimenti lo mostra a schermo.
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

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(x_anomaly) == 0:
        # Caso 1: nessuna anomalia
        q1, q3 = np.percentile(x_norm_flat, [25, 75])
        iqr = q3 - q1
        threshold = x_norm_flat.max() + 1.5 * iqr
        x_min = min(x_target.min(), threshold) - 2
        x_max = threshold + 5
        x_plot = np.linspace(x_min, x_max, 600)

        anomaly_prob = sigmoid_iqr(x_norm_flat, x_plot, k=6)
        ax.axvline(x=threshold, color='orange', label=f'Sigmoid center = {threshold:.2f} kWh')

    else:
        x_anomaly_2d = x_anomaly.reshape(-1, 1)
        x_normal_2d = x_normal.reshape(-1, 1)

        x_min = min(x_target.min(), x_anomaly.min(), x_normal.min()) - 5
        x_max = max(x_target.max(), x_anomaly.max(), x_normal.max()) + 5
        x_plot = np.linspace(x_min, x_max, 600)

        anomaly_prob = logistic_regression(x_normal_2d, x_anomaly_2d, x_plot)

    # Curva di probabilità
    ax.plot(x_plot, anomaly_prob, color='black', label='Anomaly Probability')

    # Scatter dei punti normali
    ax.scatter(x_normal, [0]*len(x_normal), color='steelblue', alpha=1, label=f'Normal ({len(x_normal)} items)')

    # Scatter dei punti anomali (se ci sono)
    if len(x_anomaly) > 0:
        ax.scatter(x_anomaly, [1]*len(x_anomaly), color='#FF9999', alpha=1, label=f'Anomaly ({len(x_anomaly)} items)')

    ax.set_title(f"{foglia} - Context {context} - Cluster {cluster}", fontsize=16)
    ax.set_xlabel("Energy [kWh]", fontsize=14)
    ax.set_ylabel("Anomaly probability [-]", fontsize=14)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(loc='best')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='best', fontsize=12)

    if save_plot:
        output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "plot_LR")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{foglia}_ctx{context}_cls{cluster}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":

    case_study = "Total"

    thermal_sensitive_load_path = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "ctx_thermal_sens")
    thermal_sensitive_loads = [os.path.splitext(f)[0] for f in os.listdir(thermal_sensitive_load_path) if f.endswith(".csv")]

    save_plot = True

    if save_plot:
        with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
            config = json.load(f)
        foglie = find_leaf_nodes(config["Load Tree"])
        for foglia in foglie:
            if foglia in thermal_sensitive_loads:
                continue
            for cls in range(1, 6):
                for ctx in range(1, 5):
                    plot_logistic_regression(case_study, foglia, ctx, cls, save_plot)

    else:
        plot_logistic_regression(case_study=case_study,
                                 foglia="Rooftop 4",
                                 context=3,
                                 cluster=3,
                                 save_plot=save_plot)
