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
    Builds a structural Bayesian network model based on the load tree of a case study, including the CPDs (Conditional Probability Distributions) calculated from anomaly data.

    The function:
    - Loads the case study configuration and extracts nodes and edges from the load tree.
    - Initializes the Bayesian network model with the obtained edges.
    - Adds uniform a priori CPDs for the leaf nodes.
    - Merges the anomaly tables related to the case study.
    - For each internal node, calculates conditional CPDs based on observed anomaly frequencies relative to its children.
      Handles cases of unobserved combinations by defining default probabilities.
    - Saves the calculated CPDs as CSV files for each node.
    - Returns the complete Bayesian network model with all CPDs added.

    Input:
    case_study : str
        Name of the case study.

    Output:
    model : BayesianNetwork
        Structural Bayesian network model with associated CPDs built from the case study.
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
    """
    Performs inference on a structural Bayesian network for a case study,
    calculating the marginal probabilities of internal nodes based on soft evidence from leaf nodes.

    The function:
    - Loads the configuration and identifies leaf and internal nodes of the load tree.
    - Builds the structural model of the Bayesian network.
    - Loads the soft evidence for each leaf node.
    - For each combination of date, context, and cluster, performs inference with virtual evidence
      to calculate the marginal probabilities of internal nodes.
    - Merges the results with information on thermal sensitivity and anomalies.
    - Sorts and saves the results in a CSV file within the case study results folder.
    - Returns a DataFrame containing all inferred probabilities and associated states.

    Input:
    case_study : str
        Name of the case study.

    Output:
    df_result : pandas.DataFrame
        DataFrame containing the marginal probabilities of each internal node, soft evidence for leaf nodes,
        anomalies, and thermal sensitivity for each combination of date, context, and cluster.
    """

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

def export_probabilities_for_date(case_study: str, date: str, context: int):
    """
    Exports the marginal probabilities and soft evidence of each node in the Bayesian network
    for a specific date and context, saving them in a CSV file.

    The function:
    - Loads inference results for the case study, the configuration, and the leaf nodes.
    - Filters data based on the specified date and context.
    - For each node, extracts the probabilities of state 0 and state 1, handling missing values
      with prior probabilities.
    - Determines the type of probability (Prior, Soft evidence, or Marginal) based on the presence
      of soft evidence and whether the node is a leaf or internal.
    - Prepares a DataFrame with loads, type, and rounded probabilities, then saves it as a CSV.

    Input:
    case_study : str
        Name of the case study from which to extract data.
    date : str
        The date for which to export the probabilities.
    context : int
        Identifier of the context for the probability selection.

    Output:
    Saves a CSV file containing the probabilities for each node in the path
    results/<case_study>/inference_on_date&ctx.
    """

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

if __name__ == "__main__":
    case_study = "Total"
    df = run_BN(case_study)
    # export_probabilities_for_date(case_study, "2024-07-21", 3)