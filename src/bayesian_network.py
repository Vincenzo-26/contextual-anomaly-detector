import numexpr
numexpr.set_num_threads(1)
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import copy
import seaborn as sns

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination
from itertools import product
from settings import PROJECT_ROOT

from src.utils import print_boxed_title, find_leaf_nodes, merge_anomaly_tables, get_children_of_node, get_nodes_by_level


def build_BN_structural_model(case_study: str):
    """
    Costruisce la struttura e i CPD di una rete bayesiana basata sulla struttura del load tree.
    Per ogni nodo foglia viene assegnata una CPD a priori uniforme [0.9, 0.1] (sostituita in fase di inferenza).
    Per i nodi interni (padri), i CPD condizionati vengono calcolati a partire dalle frequenze
    nei dati storici. Alle combinazioni mai osservate viene assegnata 50% - 50%.

    Args:
        case_study (str): Nome del case study.

    Returns:
        BayesianNetwork: Oggetto pgmpy BayesianNetwork con struttura e CPD definiti.
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

        def extract_edges_and_nodes(tree):
            edges = []

            def recurse(subtree):
                for parent, children in subtree.items():
                    for child in children:
                        edges.append((child, parent))
                        recurse({child: children[child]})

            recurse(tree)
            return edges

        edges = extract_edges_and_nodes(config["Load Tree"])
        model = BayesianNetwork(edges)

        # Aggiunta CPD a priori uniformi per le foglie
        foglie = find_leaf_nodes(config["Load Tree"])
        cpds_foglia = [TabularCPD(variable=f, variable_card=2, values=[[0.9], [0.1]]) for f in foglie]
        model.add_cpds(*cpds_foglia)

        df = merge_anomaly_tables(case_study)

        levels = get_nodes_by_level(config["Load Tree"])
        all_nodes = [node for level in levels for node in level]
        # Calcolo CPD condizionate per ogni nodo interno
        for nodo in all_nodes:
            if nodo in foglie:
                continue
            print(f"[{nodo}] Processing...", end="")
            figli = get_children_of_node(config["Load Tree"], nodo)

            # Frequenze condizionate
            group = df.groupby(figli)[nodo].value_counts(normalize=True).unstack().fillna(0)
            group = group[[0, 1]]
            group = group.reset_index()

            combinazioni = pd.DataFrame(list(product([0, 1], repeat=len(figli))), columns=figli)
            group_completo = pd.merge(combinazioni, group, how="left", on=figli).fillna(0)
            group_sorted = group_completo.sort_values(by=figli).drop(columns=figli)
            values = group_sorted.T.values

            zero_cols = (values.sum(axis=0) == 0)
            values[0, zero_cols] = 0.95
            values[1, zero_cols] = 0.05

            cpd = TabularCPD(
                variable=nodo,
                variable_card=2,
                values=values,
                evidence=figli,
                evidence_card=[2 for _ in figli]
            )
            model.add_cpds(cpd)

            evidence_combinations = list(product([0, 1], repeat=len(figli)))
            df_cpd = pd.DataFrame(evidence_combinations, columns=figli)
            for i in range(cpd.variable_card):
                df_cpd[f"P({nodo}={i})"] = values[i]

            output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "CPDs")
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"cpd_{nodo}.csv")
            df_cpd.to_csv(csv_path, index=False)
    return model

def run_BN(case_study: str):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)


    levels = get_nodes_by_level(config["Load Tree"])
    foglie = find_leaf_nodes(config["Load Tree"])

    nodi_interni = []
    all_nodes = [node for level in levels for node in level]
    for nodo in all_nodes:
        if nodo not in foglie:
            nodi_interni.append(nodo)

    print_boxed_title("Running Bayesian Network 🔄")
    print("Creation of BN structural model...\n")
    model = build_BN_structural_model(case_study)

    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "soft_evidences")

    dfs = []
    for foglia in foglie:
        path_csv = os.path.join(evidence_path, f"soft_evidence_{foglia}.csv")
        df = pd.read_csv(path_csv)
        df["foglia"] = foglia
        dfs.append(df[["Date", "Context", "Cluster", "anomaly_prob", "foglia", "thermal_sensitive"]])
    df_all = pd.concat(dfs)

    df_pivot = df_all.pivot_table(index=["Date", "Context", "Cluster"], columns="foglia", values="anomaly_prob").reset_index()
    # se almeno una foglia è thermal sensitive
    df_thermal = df_all.groupby(["Date", "Context", "Cluster"])['thermal_sensitive'].any().reset_index()

    # Probabilità P(foglia=1) e P(foglia=0)
    df_probs = df_pivot.copy()
    df_probs_0 = df_probs.copy()
    df_probs_0.iloc[:, 3:] = 1 - df_probs_0.iloc[:, 3:]
    df_probs_0.columns = df_probs_0.columns[:3].tolist() + [f"P({c}=0)" for c in df_probs_0.columns[3:]]
    df_probs_1 = df_probs.copy()
    df_probs_1.columns = df_probs_1.columns[:3].tolist() + [f"P({c}=1)" for c in df_probs_1.columns[3:]]
    df_probs_all = df_probs_0.merge(df_probs_1, on=["Date", "Context", "Cluster"])

    results = []
    print(f"\nInference on {nodi_interni}...\n")

    for _, row in df_pivot.iterrows():
        model_copy = copy.deepcopy(model)
        inference = VariableElimination(model_copy)

        virtual_evidence = []

        for foglia in foglie:
            prob = row.get(foglia)
            if pd.notna(prob):
                factor = DiscreteFactor(
                    variables=[foglia],
                    cardinality=[2],
                    values=[1 - prob, prob]
                )
                virtual_evidence.append(factor)

        result = inference.query(variables=nodi_interni, virtual_evidence=virtual_evidence)

        row_result = {
            "Date": row["Date"],
            "Context": row["Context"],
            "Cluster": row["Cluster"]
        }
        for nodo in nodi_interni:
            marginal = result.marginalize(
                [n for n in nodi_interni if n != nodo],
                inplace=False
            )
            row_result[f"P({nodo}=1)"] = marginal.values[1]
            row_result[f"P({nodo}=0)"] = marginal.values[0]

        results.append(row_result)

    df_result = pd.DataFrame(results)
    df_result = df_result.merge(df_thermal, on=["Date", "Context", "Cluster"], how="left")
    df_result = df_result.merge(df_probs_all, on=["Date", "Context", "Cluster"], how="left")

    # TODO: OCCHIO AL anomaly_table_Total CHE VA CON IL anomaly_table_{case_study}
    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table", f"anomaly_table_Total.csv")
    if os.path.exists(anomaly_path):
        df_anomaly = pd.read_csv(anomaly_path)
        df_result = df_result.merge(
            df_anomaly.assign(Anomaly=True)[["Date", "Context", "Cluster", "Anomaly"]],
            on=["Date", "Context", "Cluster"],
            how="left"
        )
        df_result["Anomaly"] = df_result["Anomaly"].fillna(False).astype(bool)

    cols = list(df_result.columns)
    for base_col in ["Date", "Context", "Cluster"]:
        cols.remove(base_col)
    cols.remove("Anomaly")
    cols.remove("thermal_sensitive")

    # TODO: OCCHIO AL P(Total=1) e P(Total}=0) CHE VA CON IL {case_study}
    case_study_cols = [f"P(Total=1)", f"P(Total=0)"]
    foglia_cols = sorted([col for col in cols if col.startswith("P(") and any(f in col for f in foglie)])
    other_cols = [col for col in cols if col not in case_study_cols + foglia_cols]
    ordered_cols = ["Date", "Context", "Cluster", "Anomaly", "thermal_sensitive"] + case_study_cols + other_cols + foglia_cols
    df_result = df_result[ordered_cols]

    output_path = os.path.join(PROJECT_ROOT, "results", case_study, "inference_results.csv")
    df_result.to_csv(output_path, index=False, float_format="%.5f")
    print(f"\033[92mCompleted analysis for '{case_study}' 🎉\033[0m\n")

    return df_result

def generate_barplots_from_cpds(case_study: str):
    """
    Genera un barplot per ogni CSV CPD nella cartella results/<case_study>/CPDs.
    Ogni barra rappresenta la probabilità P(nodo=1) in corrispondenza della riga in cui
    solo quella variabile è 1 e tutte le altre sono 0.
    """
    input_dir = os.path.join(PROJECT_ROOT, "results", case_study, "CPDs")
    output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "CPDs")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath)

        # Estrai il nome del nodo target (es. 'Total' da 'cpd_Total.csv')
        node_name = filename.replace("cpd_", "").replace(".csv", "")
        prob_col = f"P({node_name}=1)"
        input_cols = df.columns[:-2]  # tutte tranne le ultime due

        bar_labels = []
        bar_values = []

        for col in input_cols:
            # cerca la riga dove col == 1 e tutte le altre == 0
            mask = (df[col] == 1)
            for other_col in input_cols:
                if other_col != col:
                    mask &= (df[other_col] == 0)
            if mask.any():
                value = df.loc[mask, prob_col].values[0] * 100
            else:
                value = 0
            bar_labels.append(col)
            bar_values.append(value)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.bar(bar_labels, bar_values, color='lightcoral')
        plt.ylim(0, 100)
        plt.ylabel(f"P({node_name}=1) [%]")
        plt.title(f"Conditional Activation for '{node_name}'")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Salvataggio
        save_path = os.path.join(output_dir, f"barplot_{node_name}.png")
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    case_study = "Total_cut"
    df = run_BN(case_study)
    generate_barplots_from_cpds(case_study)