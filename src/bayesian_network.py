import numexpr
numexpr.set_num_threads(1)
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import copy
import numpy as np

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination
from itertools import product, combinations
from settings import PROJECT_ROOT


from src.utils import print_boxed_title, find_leaf_nodes, merge_anomaly_tables, get_children_of_node, get_nodes_by_level, assign_values_by_depth

a_priori_0 = 0.9
a_priori_1 = 0.1


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
        cpds_foglia = [TabularCPD(variable=f, variable_card=2, values=[[a_priori_0], [a_priori_1]]) for f in foglie]
        model.add_cpds(*cpds_foglia)

        df = merge_anomaly_tables(case_study)

        levels = get_nodes_by_level(config["Load Tree"])
        all_nodes = [node for level in levels for node in level]
        # Calcolo CPD condizionate per ogni nodo interno
        for nodo in all_nodes:
            if nodo in foglie:
                continue
            print(f"[{nodo}] Processing...")
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
            for idx, is_zero in enumerate(zero_cols):
                if is_zero:
                    comb = combinazioni.iloc[idx].values
                    pos_sottosequenze = []
                    sottocomb_max = 0.0
                    sottosequenza_osservata = False

                    for i in range(1, len(figli) + 1):
                        for sottoindici in combinations(np.where(comb == 1)[0], i):
                            sotto = np.zeros_like(comb)
                            sotto[list(sottoindici)] = 1

                            try:
                                pos = np.where((combinazioni.values == sotto).all(axis=1))[0][0]
                                pos_sottosequenze.append(pos)
                                if not zero_cols[pos]:  # sottosequenza osservata
                                    sottosequenza_osservata = True
                                    prob_1 = values[1, pos]
                                    sottocomb_max = max(sottocomb_max, prob_1)
                            except IndexError:
                                continue

                    if sottosequenza_osservata:
                        values[1, idx] = sottocomb_max
                        values[0, idx] = 1 - sottocomb_max
                    else:
                        values[0, idx] = 0.95
                        values[1, idx] = 0.05

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
            df_cpd["Observed"] = ~(zero_cols)
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
    print(f"\nInference on internal nodes: {nodi_interni}...\n")

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
    print(f"\033[92mEsportato CSV: {output_path}\033[0m")


    return df_result

def generate_barplots_from_cpds(case_study: str):
    """
    Genera barplot con larghezza uniforme e colori pastello per ogni CPD.
    Tutti i grafici hanno lo stesso numero di barre (paddate con 0 se necessario),
    così da mantenere una larghezza costante delle colonne.
    """
    input_dir = os.path.join(PROJECT_ROOT, "results", case_study, "CPDs")
    output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "CPDs")
    os.makedirs(output_dir, exist_ok=True)

    pastel_colors = plt.get_cmap('Pastel1')
    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    n_files = len(file_list)

    # numero massimo di colonne per averle della stessa larghezza tra i plot
    n_max = 0
    for filename in file_list:
        df_tmp = pd.read_csv(os.path.join(input_dir, filename))
        n_cols = len(df_tmp.columns) - 3  # esclusiuone di P=1 e P=0 e observed dalle colonne
        n_max = max(n_max, n_cols)

    # Step 2: genera i barplot
    for idx, filename in enumerate(file_list):
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath)

        node_name = filename.replace("cpd_", "").replace(".csv", "")
        prob_col = f"P({node_name}=1)"
        input_cols = df.columns[:-3]

        bar_labels = []
        bar_values = []

        for col in input_cols:
            mask = (df[col] == 1)
            for other_col in input_cols:
                if other_col != col:
                    mask &= (df[other_col] == 0)
            value = df.loc[mask, prob_col].values[0] * 100 if mask.any() else 0
            bar_labels.append(col)
            bar_values.append(value)

        bar_labels += [''] * (n_max - len(bar_labels))
        bar_values += [0] * (n_max - len(bar_values))
        bar_pos = np.arange(n_max)

        plt.figure(figsize=(8, 8))
        plt.bar(bar_pos, bar_values, color=pastel_colors(idx % 8))
        plt.xticks(bar_pos, bar_labels, rotation=45, ha='right', fontsize=18)
        plt.ylim(0, 100)
        plt.ylabel(f"P({node_name}=1) [%]", fontsize=18)
        plt.title(f"{node_name}", fontsize=20)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.yticks(fontsize=18)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"barplot_{node_name}.png")
        plt.savefig(save_path)
        plt.close()

def export_probabilities_for_date(case_study: str, date: str, context: int):
    results_path = os.path.join(PROJECT_ROOT, "results", case_study, "inference_results.csv")
    soft_evidence_dir = os.path.join(PROJECT_ROOT, "results", case_study, "soft_evidences")
    output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "inference_on_date&ctx")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"prob_{date}_ctx{context}.csv")

    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)
    tree = config["Load Tree"]
    foglie = find_leaf_nodes(tree)

    df = pd.read_csv(results_path)
    row = df[(df["Date"] == date) & (df["Context"] == context)]
    if row.empty:
        raise ValueError(f"Nessun risultato per Date={date}, Context={context}")
    row = row.iloc[0]

    data = []
    # solo colonne con =0 per evitare duplicati
    prob_cols_0 = [c for c in df.columns if c.startswith("P(") and c.endswith("=0)")]
    for col0 in prob_cols_0:
        node = col0.split("(")[1].split("=")[0]
        col1 = f"P({node}=1)"

        if col1 not in df.columns:
            continue  # ignora se manca la colonna =1

        p0 = row[col0]
        p1 = row[col1]
        if pd.isna(p0) or pd.isna(p1):
            p0 = a_priori_0
            p1 = a_priori_1

        # determina tipo
        if node in foglie:
            soft_path = os.path.join(soft_evidence_dir, f"soft_evidence_{node}.csv")
            tipo = "Prior"
            if os.path.exists(soft_path):
                df_soft = pd.read_csv(soft_path)
                match = df_soft[(df_soft["Date"] == date) & (df_soft["Context"] == context)]
                if not match.empty:
                    tipo = "Soft evidence"
        else:
            tipo = "Marginal"

        data.append({
            "Load": node,
            "Type": tipo,
            "P(0)": round(p0, 5),
            "P(1)": round(p1, 5)
        })


    df_out = pd.DataFrame(data)
    df_out.to_csv(output_path, index=False)
    print(f"\033[92mEsportato CSV: {output_path}\033[0m")

def cmp_vs_tool(case_study: str):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)
    inference_path = os.path.join(PROJECT_ROOT, "results", case_study, "inference_results.csv")
    anomaly_table_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")
    summary_path =  os.path.join(PROJECT_ROOT, "results", case_study)

    inference_df = pd.read_csv(inference_path)
    threshold_dict = assign_values_by_depth(config['Load Tree'])

    results = []

    levels = get_nodes_by_level(config["Load Tree"])
    all_nodes = [node for level in levels for node in level]
    for node in all_nodes:
        anomaly_table_node = pd.read_csv(os.path.join(anomaly_table_path, f"anomaly_table_{node}.csv"))
        threshold = threshold_dict[node]

        n_cmp = len(anomaly_table_node)

        inference_high = inference_df[inference_df[f'P({node}=1)'] >= threshold]
        n_inference_high = len(inference_high)

        anomaly_pairs = set(zip(anomaly_table_node["Date"], anomaly_table_node["Context"]))
        inference_high_pairs = set(zip(inference_high["Date"], inference_high["Context"]))
        n_common = len(anomaly_pairs & inference_high_pairs)

        # Calcolo variazione percentuale
        if n_cmp == 0:
            change = float('inf') if n_inference_high > 0 else 0
        else:
            change = round((n_inference_high - n_cmp) / n_cmp * 100, 2)

        results.append({
            "Load": node,
            "Threshold": threshold * 100,
            "CMP": n_cmp,
            "Proposed Approach": n_inference_high,
            "Common Anomalies": n_common,
            "Detection Shift": f"{change:.2f}%" if change != float('inf') else "-"
        })

    df_summary = pd.DataFrame(results)
    df_summary["Threshold"] = df_summary["Threshold"].map(lambda x: f"{x:.0f}%")
    df_summary.to_csv(os.path.join(summary_path, "cmp_vs_tool_summary.csv"), index=False)
    return df_summary



if __name__ == "__main__":
    case_study = "Total_cut"
    # df = run_BN(case_study)
    # generate_barplots_from_cpds(case_study)
    # export_probabilities_for_date(case_study, "2024-07-21", 3)
    cmp_vs_tool(case_study)