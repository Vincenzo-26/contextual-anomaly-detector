import numexpr
numexpr.set_num_threads(1)
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination
from itertools import product
from src.utils import *
import copy

def build_BN_structural_model(case_study: str):
    """
    Costruisce la struttura e i CPD di una rete bayesiana basata sulla struttura del load tree.
    Per ogni nodo foglia viene assegnata una CPD a priori uniforme [0.5, 0.5] (sostituita in fase di inferenza).
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
        cpds_foglia = [TabularCPD(variable=f, variable_card=2, values=[[0.5], [0.5]]) for f in foglie]
        model.add_cpds(*cpds_foglia)


        levels = get_nodes_by_level(config["Load Tree"])

        # Calcolo CPD condizionate per ogni nodo interno
        for livello in levels[1:]:
            for nodo in livello:
                figli = get_children_of_node(config["Load Tree"], nodo)
                if not figli:
                    continue

                print(f"[{nodo}] Processing...", end="")

                df = merge_anomaly_tables(nodo)

                # Frequenze condizionate
                group = df.groupby(figli)[nodo].value_counts(normalize=True).unstack().fillna(0)
                group = group[[0, 1]]
                group = group.reset_index()

                combinazioni = pd.DataFrame(list(product([0, 1], repeat=len(figli))), columns=figli)
                group_completo = pd.merge(combinazioni, group, how="left", on=figli).fillna(0)
                group_sorted = group_completo.sort_values(by=figli).drop(columns=figli)
                values = group_sorted.T.values

                zero_cols = (values.sum(axis=0) == 0)
                values[0, zero_cols] = 0.5
                values[1, zero_cols] = 0.5

                cpd = TabularCPD(
                    variable=nodo,
                    variable_card=2,
                    values=values,
                    evidence=figli,
                    evidence_card=[2 for _ in figli]
                )

                model.add_cpds(cpd)

                # Salva la CPD in CSV
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
    nodi_interni = [n for lvl in levels[1:] for n in lvl]

    print_boxed_title("Running Bayesian Network 🔄")
    print("Creation of BN structural model...\n")
    model = build_BN_structural_model(case_study)

    foglie = find_leaf_nodes(config["Load Tree"])
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "soft_evidences")

    dfs = []
    for foglia in foglie:
        path_csv = os.path.join(evidence_path, f"soft_evidence_{foglia}.csv")
        df = pd.read_csv(path_csv)
        df["foglia"] = foglia
        dfs.append(df[["Date", "Context", "Cluster", "anomaly_prob", "foglia", "thermal_sensitive"]])

    df_all = pd.concat(dfs)

    df_pivot = df_all.pivot_table(index=["Date", "Context", "Cluster"], columns="foglia", values="anomaly_prob").reset_index()
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

    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table", f"anomaly_table_{case_study}.csv")
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
    cabina_cols = [f"P({case_study}=1)", f"P({case_study}=0)"]
    foglia_cols = sorted([col for col in cols if col.startswith("P(") and any(f in col for f in foglie)])
    other_cols = [col for col in cols if col not in cabina_cols + foglia_cols]
    ordered_cols = ["Date", "Context", "Cluster", "Anomaly", "thermal_sensitive"] + cabina_cols + other_cols + foglia_cols
    df_result = df_result[ordered_cols]

    output_path = os.path.join(PROJECT_ROOT, "results", case_study, "inference_results.csv")
    df_result.to_csv(output_path, index=False, float_format="%.5f")
    print(f"\033[92mCompleted analysis for '{case_study}' 🎉\033[0m\n")

    return df_result

def run_BN_non_thermal(case_study: str):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    levels = get_nodes_by_level(config["Load Tree"])
    nodi_interni = [n for lvl in levels[1:] for n in lvl]

    print_boxed_title("Running NON-THERMAL Bayesian Network 🔄")
    print("Creation of BN structural model...\n")
    model = build_BN_structural_model(case_study)

    foglie = find_leaf_nodes(config["Load Tree"])
    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "soft_evidences")

    dfs = []
    for foglia in foglie:
        path_csv = os.path.join(evidence_path, f"soft_evidence_{foglia}.csv")
        df = pd.read_csv(path_csv)
        df["foglia"] = foglia
        prob_col = "energy_anomaly_prob" if "energy_anomaly_prob" in df.columns else "anomaly_prob"
        df["used_prob"] = df[prob_col]
        dfs.append(df[["Date", "Context", "Cluster", "used_prob", "foglia", "thermal_sensitive"]])

    df_all = pd.concat(dfs)

    df_pivot = df_all.pivot_table(index=["Date", "Context", "Cluster"], columns="foglia", values="used_prob").reset_index()
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
    print(f"\nInference on {nodi_interni} (non-thermal)...\n")

    for _, row in df_pivot.iterrows():
        model_copy = copy.deepcopy(model)
        inference = VariableElimination(model_copy)
        virtual_evidence = []

        for foglia in foglie:
            prob = row.get(foglia)
            if pd.notna(prob):
                factor = DiscreteFactor([foglia], [2], [1 - prob, prob])
                virtual_evidence.append(factor)

        result = inference.query(variables=nodi_interni, virtual_evidence=virtual_evidence)

        row_result = {
            "Date": row["Date"],
            "Context": row["Context"],
            "Cluster": row["Cluster"]
        }
        for nodo in nodi_interni:
            marginal = result.marginalize([n for n in nodi_interni if n != nodo], inplace=False)
            row_result[f"P({nodo}=1)"] = marginal.values[1]
            row_result[f"P({nodo}=0)"] = marginal.values[0]

        results.append(row_result)

    df_result = pd.DataFrame(results)
    df_result = df_result.merge(df_thermal, on=["Date", "Context", "Cluster"], how="left")
    df_result = df_result.merge(df_probs_all, on=["Date", "Context", "Cluster"], how="left")

    anomaly_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table", f"anomaly_table_{case_study}.csv")
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
    cabina_cols = [f"P({case_study}=1)", f"P({case_study}=0)"]
    foglia_cols = sorted([col for col in cols if col.startswith("P(") and any(f in col for f in foglie)])
    other_cols = [col for col in cols if col not in cabina_cols + foglia_cols]
    ordered_cols = ["Date", "Context", "Cluster", "Anomaly", "thermal_sensitive"] + cabina_cols + other_cols + foglia_cols
    df_result = df_result[ordered_cols]

    output_path = os.path.join(PROJECT_ROOT, "results", case_study, "inference_results_non_thermal.csv")
    df_result.to_csv(output_path, index=False, float_format="%.5f")
    print(f"\033[92mCompleted NON-THERMAL analysis for '{case_study}' 🎉\033[0m\n")

    return df_result


if __name__ == "__main__":
     df = run_BN("Cabina")
     df = run_BN_non_thermal("Cabina")