import os
os.environ["OMP_NUM_THREADS"] = "1"
from src.utils import *
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from tabulate import tabulate
import matplotlib.pyplot as plt

def sigmoid_iqr(X_norm, X_target, k):
    """
    Calcola la probabilità di anomalia utilizzando una curva sigmoide centrata sul valore massimo dei dati normali più
    1.5 volte l'intervallo interquartile (IQR).

    Args:
        X_norm (np.ndarray): array dei punti 'normal' da cui viene calcolato il centro della sigmoide.
        X_target (np.ndarray): array dei punti su cui calcolare la probabilità di anomalia.
        k (float): Coefficiente di pendenza della curva sigmoide (valori maggiori -> più ripida).

    Returns:
        np.ndarray: Array di probabilità di anomalia corrispondente a ciascun elemento di X_target.

    Nota: in assenza di dati anomali, X_target e X_norm coincidono, ma sono tenuti distinti se si
    desidera calcolare la probabilità su un dominio più esteso o per motivi di visualizzazione della curva.

    """
    q1 = np.percentile(X_norm, 25)
    q3 = np.percentile(X_norm, 75)
    iqr = q3 - q1
    threshold = X_norm.max() + 1.5 * iqr
    z = k * (X_target - threshold)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
def sigmoid_single_point(x_anom, X_target, X_normal, base_k=6, target_prob=0.95):
    """
    Calcola una curva sigmoide adattativa considerando un solo punto anomalo osservato.

    La sigmoide è costruita in modo che:
    - La probabilità in corrispondenza del punto anomalo `x_anom` sia pari a `target_prob` (es. 0.95).
    - Se `x_anom` è troppo vicino ai dati normali, tale per cui il centro della sigmoide x0 è all'interno della
     distribuzione 'normal', allora viene fissato x0 = x_max_norm e la pendenza `k` viene aumentata per mantenere la
     condizione `P(x_anom) = target_prob`.

    Args:
        x_anom (float): energia dell'unico punto anomalo disponibile.
        X_target (np.ndarray): Array dei punti su cui calcolare la probabilità di anomalia.
        X_normal (np.ndarray): Array dei dati 'normal', utilizzato per posizionare il centro della sigmoide.
        base_k (float): Pendenza di default della sigmoide (default = 6).
        target_prob (float): Valore di probabilità per il punto `x_anom` (default = 0.95).

    Returns:
        tuple:
            - np.ndarray: Array di probabilità di anomalia su X_target.
            - float: Centro della sigmoide `x0`.
            - float: Pendenza effettiva `k` della sigmoide calcolata.
    """
    log_ratio = np.log(target_prob / (1 - target_prob))
    x_max_normal = np.max(X_normal)

    x0_try = x_anom - log_ratio / base_k

    if x0_try >= x_max_normal:
        k = base_k
        x0 = x0_try
    else:
        x0 = x_max_normal
        delta = x_anom - x0
        if delta <= 0:
            delta = 1e-6
        k = log_ratio / delta

    z = k * (X_target - x0)
    z = np.clip(z, -500, 500)
    prob = 1 / (1 + np.exp(-z))

    return prob, x0, k
def adjust_probability_tails(X_target, anomaly_prob, X_normal, X_anomaly_max):
    """
    Regola i bordi della curva di probabilità:
    - Imposta a zero la probabilità per i valori inferiori al primo percentile dei dati normali.
    - Mantiene costante la probabilità per i valori superiori al massimo anomalo.

    Args:
        - X_target (np.ndarray): array dell'energia su cui è calcolata la curva via Bayes. (non è necessariamente l'unione
        dei punti normal e anomali ma è l'insieme delle ascisse su cui poi è calcolata la curva.
        - X_normal (np.ndarray): Valori 'normal' usati per calcolare il 50esimo percentile.
        - anomaly_prob (np.ndarray): curva di probabilità calcolata via Bayes.
        - X_anomaly_max (float): valore massimo di energia dei punti anomali.

    Returns:
        np.ndarray: curva di probabilità modificata.
    """
    # Coda sinistra
    min_safe = np.percentile(X_normal, 50)
    anomaly_prob[X_target < min_safe] = 0

    # Coda destra
    idx = np.argmin(np.abs(X_target - X_anomaly_max))
    plateau_value = anomaly_prob[idx]
    anomaly_prob[X_target > X_anomaly_max] = plateau_value

    return np.clip(anomaly_prob, 0, 1)
def compute_gmm_anomaly_probability(X_target: np.ndarray,X_normal: np.ndarray,X_anomaly: np.ndarray,height: float,one_anm_peak: bool) -> np.ndarray:
    """
    Calcola la probabilità di anomalia tramite modelli GMM e formula di Bayes, nel caso in cui siano presenti almeno
    due punti anomali. Dopo il calcolo, la curva viene modificata forzando:
    - valore nullo al di sotto del 1° percentile dei dati normali,
    - valore costante oltre il massimo dei dati anomali.

    Args:
        X_target (np.ndarray): Valori di energia su cui calcolare la probabilità di anomalia.
        X_normal (np.ndarray): Valori normali (shape N x 1).
        X_anomaly (np.ndarray): Valori anomali (shape M x 1).
        height (float): Soglia relativa per il rilevamento dei picchi (usata per stimare i componenti GMM).
        one_anm_peak (bool): Se True, usa 1 sola componente per le anomalie. Altrimenti stima il numero di componenti.

Returns:
        tuple:
            - anomaly_prob (np.ndarray): Curva di probabilità finale (Bayes).
            - p_x_normal (np.ndarray): PDF GMM stimata sui normali.
            - p_x_anomaly (np.ndarray): PDF GMM stimata sugli anomali.
            - k_normal (int): numero di picchi della distribuzione normal
            - k_anom (int): numero di picchi della distribuzione anomala
    """
    # GMM sui normali
    counts, _ = np.histogram(X_normal.flatten(), bins=30)
    peaks, _ = find_peaks(counts, height=height * np.max(counts))
    k_normal = min(max(1, len(peaks)), 3)

    gmm_normal = GaussianMixture(n_components=k_normal, covariance_type='full', random_state=0)
    gmm_normal.fit(X_normal)
    p_x_normal = np.exp(gmm_normal.score_samples(X_target.reshape(-1, 1)))

    # GMM sugli anomali
    if one_anm_peak:
        k_anom = 1
    else:
        counts_anm, _ = np.histogram(X_anomaly.flatten(), bins=30)
        peaks_anm, _ = find_peaks(counts_anm, height=height * np.max(counts_anm))
        k_anom = min(max(1, len(peaks_anm)), 3)

    gmm_anomaly = GaussianMixture(n_components=k_anom, covariance_type='full', random_state=0)
    gmm_anomaly.fit(X_anomaly)
    p_x_anomaly = np.exp(gmm_anomaly.score_samples(X_target.reshape(-1, 1)))

    # Probabilità tramite Bayes
    n_normal = len(X_normal)
    n_anomaly = len(X_anomaly)
    prior_normal = n_normal / (n_normal + n_anomaly)
    prior_anomaly = n_anomaly / (n_normal + n_anomaly)

    anomaly_prob = (p_x_anomaly * prior_anomaly) / (
        p_x_anomaly * prior_anomaly + p_x_normal * prior_normal + 1e-10
    )

    # Forzature alle code
    min_safe = np.percentile(X_normal, 1)
    anomaly_prob[X_target < min_safe] = 0
    max_anom = X_anomaly.max()
    anomaly_prob = adjust_probability_tails(X_target, anomaly_prob, X_normal, max_anom)

    return anomaly_prob, p_x_normal, p_x_anomaly, k_normal, k_anom


def run_soft_evd_EM(case_study: str, height: float, one_anm_peak: bool, k_sigmoide: float = 6, threshold_metrics: float = 0.8, ):
    """
    Calcola le probabilità di anomalia per ogni foglia del load tree
    utilizzando modelli GMM (Gaussian Mixture Model) o curve sigmoidi, e salva i risultati su file CSV.

    Per ogni combinazione foglia-contesto-cluster:
    - Se sono presenti dati anomali:
        - Se ≥ 2 punti anomali → stima due GMM (normali e anomali) e calcola la probabilità via formula di Bayes.
            Dopo il calcolo:
            - La probabilità è forzata a zero sotto il primo percentile dei dati normali per ridurre i falsi positivi a
              sinistra.
            - La probabilità è mantenuta costante oltre il massimo valore anomalo per evitare falsi negativi a destra.
        - Se 1 solo punto anomalo → costruisce una curva sigmoide adattativa centrata in modo da assegnare al punto
          una probabilità target (default = 0.95).
    - Se non ci sono dati anomali → usa una curva sigmoide centrata su max(normal) + 1.5*IQR.

    Args:
        case_study (str): Nome del caso studio (usato per caricare dati e salvare risultati).
        height (float): Altezza relativa al massimo per il rilevamento dei picchi nei GMM (usata per stimare il numero
        di componenti).
        one_anm_peak (bool): Se True, forza l’uso di una sola componente GMM per gli anomali. Se False, il numero viene
        calcolato automaticamente come per la distribuzione 'normal'.
        k_sigmoide (float): Pendenza di base per la costruzione della sigmoide (default = 6).
        threshold_metrics (float): Soglia sopra cui una probabilità è considerata anomala per il calcolo delle metriche
        (default = 0.8).

    Returns:
        None
    """

    print_boxed_title("Energy evidences calculation 📈")

    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_EM")
    os.makedirs(evidence_path, exist_ok=True)

    TP_total = 0
    FP_total = 0
    FN_total = 0

    all_data = []

    foglie = find_leaf_nodes(config["Load Tree"])

    for foglia in foglie:
        print(f"\033[91m{foglia}\033[0m")

        energy_data_full = run_energy_in_tw(case_study, foglia)
        anm_table = pd.read_csv(os.path.join(anomaly_path, f"anomaly_table_{foglia}.csv"))

        anm_table["Date"] = pd.to_datetime(anm_table["Date"]).dt.date
        energy_data_full["Date"] = pd.to_datetime(energy_data_full["Date"]).dt.date

        merged = energy_data_full.merge(
            anm_table[["Date", "Context", "Cluster"]],
            on=["Date", "Context", "Cluster"],
            how="left", indicator=True
        )
        energy_data_clean = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        energy_data_anm = merged[merged["_merge"] == "both"].drop(columns=["_merge"])

        energy_data_full["anomaly_prob"] = np.nan

        anm_keys = set((str(row.Date), row.Context, row.Cluster) for row in anm_table.itertuples(index=False))
        energy_data_full["is_real_anomaly"] = energy_data_full.apply(
            lambda row: (str(row.Date), row.Context, row.Cluster) in anm_keys, axis=1
        )

        combinations = energy_data_full[["Context", "Cluster"]].drop_duplicates()

        skipped_cases = 0

        for _, row in combinations.iterrows():
            context = row["Context"]
            cluster = row["Cluster"]

            clean_subset = energy_data_clean[
                (energy_data_clean["Context"] == context) &
                (energy_data_clean["Cluster"] == cluster)
            ]
            anm_subset = energy_data_anm[
                (energy_data_anm["Context"] == context) &
                (energy_data_anm["Cluster"] == cluster)
            ]

            if clean_subset.empty:
                print(f"No normal data for Context {context} Cluster {cluster}. Skipping.")
                skipped_cases += 1
                continue

            mask = (energy_data_full["Context"] == context) & (energy_data_full["Cluster"] == cluster)
            X_target = energy_data_full.loc[mask, "Energy"].values
            X_normal = clean_subset["Energy"].values.reshape(-1, 1)
            X_norm_flat = X_normal.flatten()

            if anm_subset.empty or anm_subset["Energy"].isna().all():
                # Caso 1: Nessun punto anomalo → uso sigmoide basata su IQR
                print(f"[Ctx {context} | Clst {cluster}] no anomaly data -> Using sigmoidal anomaly probability.")
                anomaly_prob = sigmoid_iqr(X_normal, X_target, k_sigmoide)
                energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob
                continue

            elif len(anm_subset) == 1:
                # Caso 2: Un solo punto anomalo → uso sigmoide adattata su singolo punto
                x_anom = anm_subset["Energy"].values[0]
                anomaly_prob, x0, k = sigmoid_single_point(
                    x_anom, X_target, X_norm_flat, base_k=k_sigmoide, target_prob=0.95
                )
                energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob
                print(
                    f"[Ctx {context} | Clst {cluster}] 1 anomaly -> Using sigmoidal anomaly probability (K = {round(k, 2)}).")
                continue

            else:
                # Caso 3
                if anm_subset["Energy"].nunique() == 1:
                    # Caso 3.1: Tutti i punti anomali hanno lo stesso valore → sigmoide come caso 2
                    x_anom = anm_subset["Energy"].values[0]
                    anomaly_prob, x0, k = sigmoid_single_point(
                        x_anom, X_target, X_norm_flat, base_k=k_sigmoide, target_prob=0.95
                    )
                    energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob
                    print(
                        f"[Ctx {context} | Clst {cluster}] {len(anm_subset)} identical anomalies -> Using sigmoidal anomaly probability (K = {round(k, 2)}).")
                    continue

                # Caso 3.2: Almeno 2 anomalie diverse → GMM + formula di Bayes + regolazione bordi
                X_anomaly = anm_subset["Energy"].values.reshape(-1, 1)
                anomaly_prob, _, _, k_norm, k_anom = compute_gmm_anomaly_probability(
                    X_target, X_normal, X_anomaly, height, one_anm_peak
                )
                print(
                    f"[Ctx {context} | Clst {cluster}] -> GMM applied via EM with {k_norm} normal peak(s) - {k_anom} anomalous peak(s).")
                energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob

        # Metriche parziali per la foglia
        predicted_anomalies = energy_data_full["anomaly_prob"] > threshold_metrics
        true_anomalies = energy_data_full["is_real_anomaly"]

        TP_total += ((predicted_anomalies == True) & (true_anomalies == True)).sum()
        FP_total += ((predicted_anomalies == True) & (true_anomalies == False)).sum()
        FN_total += ((predicted_anomalies == False) & (true_anomalies == True)).sum()

        all_data.append(energy_data_full)

        output_file = os.path.join(evidence_path, f"evd_{foglia}.csv")
        energy_data_full.to_csv(output_file, index=False)
        print(f"\nSaved - Skipped {skipped_cases} Ctx-Clst combinations due to insufficient data.\n")

    # Metriche totali
    df_all = pd.concat(all_data, ignore_index=True)

    TP = TP_total
    FP = FP_total
    FN = FN_total

    precision = TP / (TP + FP) if TP + FP > 0 else None
    recall = TP / (TP + FN) if TP + FN > 0 else None

    try:
        roc_auc = roc_auc_score(df_all["is_real_anomaly"], df_all["anomaly_prob"])
    except ValueError:
        roc_auc = None

    metrics = [
        [f"Precisione (threshold={threshold_metrics})", f"{precision:.3f} ({TP}/{TP + FP})" if precision is not None else "n.d."],
        [f"Recall     (threshold={threshold_metrics})", f"{recall:.3f} ({TP}/{TP + FN})" if recall is not None else "n.d."],
        ["ROC AUC", f"{roc_auc:.3f}" if roc_auc is not None else "n.d."]
    ]
    print(tabulate(metrics, headers=["Metriche 📊", ""], tablefmt="grid"))
    print("\n\n")



if __name__ == "__main__":

    sensitivity_analisys = False

    if sensitivity_analisys:
        heights = np.arange(0.1, 1.01, 0.1)
        results = {}
        for one_anm_peak in [True, False]:
            precisions, recalls, aucs = [], [], []

            for height in heights:
                print(f"\n===> Running for height={height:.1f}, one_anm_peak={one_anm_peak}")
                run_soft_evd_EM("Cabina", height, one_anm_peak=one_anm_peak)

                result_path = os.path.join(PROJECT_ROOT, "results", "Cabina", "Evidences_EM")
                all_dfs = []
                for file in os.listdir(result_path):
                    if file.endswith(".csv"):
                        df = pd.read_csv(os.path.join(result_path, file))
                        all_dfs.append(df)
                df_all = pd.concat(all_dfs, ignore_index=True)

                predicted = df_all["anomaly_prob"] > 0.8
                actual = df_all["is_real_anomaly"]

                TP = ((predicted == True) & (actual == True)).sum()
                FP = ((predicted == True) & (actual == False)).sum()
                FN = ((predicted == False) & (actual == True)).sum()

                precision = TP / (TP + FP) if TP + FP > 0 else 0
                recall = TP / (TP + FN) if TP + FN > 0 else 0
                try:
                    auc = roc_auc_score(actual, df_all["anomaly_prob"])
                except:
                    auc = 0

                precisions.append(precision)
                recalls.append(recall)
                aucs.append(auc)

            results[one_anm_peak] = {
                "precisions": precisions,
                "recalls": recalls,
                "aucs": aucs
            }

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        for idx, one_anm_peak in enumerate([True, False]):
            ax = axes[idx]
            precisions = results[one_anm_peak]["precisions"]
            recalls = results[one_anm_peak]["recalls"]
            aucs = results[one_anm_peak]["aucs"]

            max_prec_idx = np.argmax(precisions)
            max_recall_idx = np.argmax(recalls)
            max_auc_idx = np.argmax(aucs)

            ax.plot(heights, precisions, 'o-', label="Precision")
            ax.plot(heights, recalls, 's--', label="Recall")
            ax.plot(heights, aucs, 'd-.', label="ROC AUC")

            ax.scatter([heights[max_prec_idx]], [precisions[max_prec_idx]], color='red', zorder=5)
            ax.scatter([heights[max_recall_idx]], [recalls[max_recall_idx]], color='red', zorder=5)
            ax.scatter([heights[max_auc_idx]], [aucs[max_auc_idx]], color='red', zorder=5)

            ax.set_xlabel("Soglia (in % del picco più alto) per rilevare picchi")
            ax.set_title(f"Metriche - one_anm_peak = {one_anm_peak}")
            ax.grid(True)
            if idx == 0:
                ax.set_ylabel("Score")
            ax.legend()

        plt.tight_layout()
        plt.show()

        print("\n=== RISULTATI MASSIMI ===")
        metric_keys = {
            "Precision": "precisions",
            "Recall": "recalls",
            "ROC AUC": "aucs"
        }
        for metric_name, key in metric_keys.items():
            best_config = None
            best_value = -1
            for one_anm_peak in [True, False]:
                values = results[one_anm_peak][key]
                max_idx = np.argmax(values)
                value = values[max_idx]
                if value > best_value:
                    best_value = value
                    best_config = (one_anm_peak, heights[max_idx])
            print(f"{metric_name} massima: {best_value:.3f} con height = {best_config[1]:.1f}, one_anm_peak = {best_config[0]}")
    else:
        run_soft_evd_EM("Cabina", 0.6, False)