from src.utils import *
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def sigmoid_iqr(x_norm: np.ndarray, x_target: np.ndarray, k: float) -> np.ndarray:
    """
        Calcola la probabilità di anomalia utilizzando una curva sigmoide centrata su
        max(X_norm) + 1.5 * IQR. È usata nei casi in cui non ci sono dati anomali disponibili.

        Args:
            x_norm (np.ndarray): Dati normali da cui calcolare soglia e pendenza (shape: N x 1 o N).
            x_target (np.ndarray): Ascisse su cui calcolare la probabilità (shape: T,).
            k (float): Coefficiente di pendenza della sigmoide (valori maggiori = transizione più rapida).

        Returns:
            np.ndarray: Probabilità di anomalia associata a ciascun punto in X_target.

        Nota: in assenza di dati anomali, X_target e X_norm coincidono, ma sono tenuti distinti se si
        desidera calcolare la probabilità su un dominio più esteso o per motivi di visualizzazione della curva.
    """
    q1 = np.percentile(x_norm, 25)
    q3 = np.percentile(x_norm, 75)
    iqr = q3 - q1
    threshold = x_norm.max() + 1.5 * iqr
    z = k * (x_target - threshold)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def logistic_regression(x_normal, x_anomaly, x_target, c: int = 50) -> np.ndarray:
    """
    Calcola la probabilità di anomalia utilizzando una regressione logistica binaria
    addestrata su dati normali e anomali.

    Args:
        x_normal (np.ndarray): Dati etichettati come normali (shape: N x 1).
        x_anomaly (np.ndarray): Dati etichettati come anomali (shape: M x 1).
        x_target (np.ndarray): Valori su cui stimare la probabilità di anomalia (shape: T,).
        c (float): Inverso della regolarizzazione. Valori più alti rendono il modello più flessibile.

    Returns:
        np.ndarray: Probabilità di anomalia per ciascun punto in X_target.
    """
    X = np.concatenate([x_normal, x_anomaly]).reshape(-1, 1)
    y = np.array([0] * len(x_normal) + [1] * len(x_anomaly))

    model = LogisticRegression(C = c, class_weight='balanced', max_iter=1000)
    model.fit(X, y)

    return model.predict_proba(x_target.reshape(-1, 1))[:, 1]


def run_soft_evd_LR(case_study: str, c: int, k_sigmoide: float = 4, threshold_metrics: float = 0.8):
    """
    Calcola la probabilità di anomalia per ciascun cluster di ciascun contesto e foglia del load tree,
    utilizzando:
      - una curva sigmoide se non ci sono anomalie;
      - una regressione logistica se presenti una o più anomalie.

    Le probabilità sono salvate in CSV, e vengono calcolate precision, recall e ROC AUC
    sulla base della soglia impostata.

    Args:
        case_study (str): Nome del caso studio.
        c (float): Parametro di regolarizzazione della regressione logistica (c=50 da analisi di sensibilità).
        k_sigmoide (float): Pendenza della sigmoide in caso di assenza di anomalie (default 4).
        threshold_metrics (float): Soglia sulla probabilità per considerare un punto anomalo (default 0.8).

    Returns:
        None. I risultati vengono stampati e salvati.
    """

    print_boxed_title("Energy evidences calculation 📈")

    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "Evidences_LR")
    os.makedirs(evidence_path, exist_ok=True)

    true_positive_total = 0
    false_positive_total = 0
    false_negative_total = 0

    thermal_sensitive_load_path = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "ctx_thermal_sens")
    thermal_sensitive_loads = [os.path.splitext(f)[0] for f in os.listdir(thermal_sensitive_load_path) if f.endswith(".csv")]
    
    all_data = []
    leaves = find_leaf_nodes(config["Load Tree"])
    for foglia in leaves:
        # if foglia in thermal_sensitive_loads:
        #     continue
        print(f"\033[91m{foglia}\033[0m")

        energy_data_full = run_energy_in_tw(case_study, foglia)
        anm_table = pd.read_csv(os.path.join(anomaly_path, f"anomaly_table_{foglia}.csv"))

        anm_table["Date"] = pd.to_datetime(anm_table["Date"], errors="coerce").dt.date
        energy_data_full["Date"] = pd.to_datetime(energy_data_full["Date"], errors="coerce").dt.date

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
            x_target = energy_data_full.loc[mask, "Energy"].values
            x_normal = clean_subset["Energy"].values.reshape(-1, 1)
            x_norm_flat = x_normal.flatten()

            if anm_subset.empty or anm_subset["Energy"].isna().all():
                # Caso 1: Nessun punto anomalo → sigmoide basata su IQR
                print(f"[Ctx {context} | Clst {cluster}] no anomaly data -> Using sigmoidal anomaly probability.")
                anomaly_prob = sigmoid_iqr(x_normal, x_target, k_sigmoide)
                energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob
                continue
                # Caso 2: Presenti anomalie → logistic regression
            elif len(anm_subset) >= 1:
                x_anomaly = anm_subset["Energy"].values
                anomaly_prob = logistic_regression(x_norm_flat, x_anomaly, x_target, c)
                energy_data_full.loc[mask, "anomaly_prob"] = anomaly_prob

            print(f"[Ctx {context} | Clst {cluster}] {len(anm_subset)} anomalies -> Logistic Regression applied.")
        # Metriche parziali per la foglia
        predicted_anomalies = energy_data_full["anomaly_prob"] > threshold_metrics
        true_anomalies = energy_data_full["is_real_anomaly"]

        true_positive_total += ((predicted_anomalies == True) & (true_anomalies == True)).sum()
        false_positive_total += ((predicted_anomalies == True) & (true_anomalies == False)).sum()
        false_negative_total += ((predicted_anomalies == False) & (true_anomalies == True)).sum()

        all_data.append(energy_data_full)

        output_file = os.path.join(evidence_path, f"evd_{foglia}.csv")
        energy_data_full.to_csv(output_file, index=False)
        print(f"\nSaved - Skipped {skipped_cases} Ctx-Clst combinations due to insufficient data.\n")

    # Metriche totali
    df_all = pd.concat(all_data, ignore_index=True)

    true_positive = true_positive_total
    false_positive = false_positive_total
    false_negative = false_negative_total

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else None

    try:
        roc_auc = roc_auc_score(df_all["is_real_anomaly"], df_all["anomaly_prob"])
    except ValueError:
        roc_auc = None

    metrics = [
        [f"Precisione (threshold={threshold_metrics})", f"{precision:.3f} ({true_positive}/{true_positive + false_positive})" if precision is not None else "n.d."],
        [f"Recall     (threshold={threshold_metrics})", f"{recall:.3f} ({true_positive}/{true_positive + false_negative})" if recall is not None else "n.d."],
        ["ROC AUC", f"{roc_auc:.3f}" if roc_auc is not None else "n.d."]
    ]
    print(tabulate(metrics, headers=["Metriche 📊", ""], tablefmt="grid"))
    print("\n\n")
    return


if __name__ == "__main__":

    sensitivity_analisys = False
    case_study = "Total_cut"

    if sensitivity_analisys:
        c_list = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        precisions, recalls, aucs = [], [], []

        for C in c_list:
            print(f"\n===> Running for C={C}")
            run_soft_evd_LR(case_study, C)

            result_path = os.path.join(PROJECT_ROOT, "results", f"{case_study}", "Evidences_LR")
            all_dfs = []
            for file in os.listdir(result_path):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(result_path, file))
                    all_dfs.append(df)
            df_all = pd.concat(all_dfs, ignore_index=True)

            predicted = df_all["anomaly_prob"] > 0.8
            actual = df_all["is_real_anomaly"]

            true_positive = ((predicted == True) & (actual == True)).sum()
            false_positive = ((predicted == True) & (actual == False)).sum()
            false_negative = ((predicted == False) & (actual == True)).sum()

            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
            try:
                auc = roc_auc_score(actual, df_all["anomaly_prob"])
            except:
                auc = 0

            precisions.append(precision)
            recalls.append(recall)
            aucs.append(auc)

        plt.figure(figsize=(8, 6))
        plt.plot(c_list, precisions, 'o-', label='Precision', color='#1f77b4')
        plt.plot(c_list, recalls, 's-', label='Recall', color='#ff7f0e')
        plt.plot(c_list, aucs, 'd-', label='ROC AUC', color='#2ca02c')

        max_prec_idx = np.argmax(precisions)
        max_recall_idx = np.argmax(recalls)
        max_auc_idx = np.argmax(aucs)

        plt.xscale('log')
        plt.xlabel("C", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.tick_params(axis='both', labelsize=10)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("\n📊 MIGLIORI CONFIGURAZIONI TROVATE:")
        print(f"→ Precisione massima: {precisions[max_prec_idx]:.3f} con C = {c_list[max_prec_idx]}")
        print(f"→ Recall massimo:     {recalls[max_recall_idx]:.3f} con C = {c_list[max_recall_idx]}")
        print(f"→ ROC AUC massimo:    {aucs[max_auc_idx]:.3f} con C = {c_list[max_auc_idx]}")
    else:
        run_soft_evd_LR(case_study, 50, 0.4, 0.8)

