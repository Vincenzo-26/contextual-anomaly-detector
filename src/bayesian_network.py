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

        def extract_edges_and_nodes(load_tree):
            edges = []
            nodes = set()

            def recurse(tree):
                for parent, children in tree.items():
                    nodes.add(parent)
                    for child in children:
                        nodes.add(child)
                        edges.append((child, parent))  # direzione bottom-up
                        recurse({child: children[child]})

            recurse(load_tree)
            return edges, nodes

        # struttura della BN proveniente dalla struttura del load tree nel config
        edges, nodes = extract_edges_and_nodes(config["Load Tree"])
        model = BayesianNetwork(edges)

        # Probabilità a priori dei nodi foglia (sono rimpiazzate dalle soft evidence)
        foglie = find_leaf_nodes(config["Load Tree"])
        cpds_foglia = []

        for foglia in foglie:
            cpd = TabularCPD(variable=foglia, variable_card=2, values=[[0.5], [0.5]])
            cpds_foglia.append(cpd)

        model.add_cpds(*cpds_foglia)

        # probabilità condizionate
        parents = find_parents_of_leaves(config["Load Tree"])

        combinazioni = list(product([0, 1], repeat=len(parents)))
        cpd_case_study = TabularCPD(
            variable=f"{case_study}",
            variable_card=2,
            values=[[0.0] * len(combinazioni),  [1.0] * len(combinazioni)],
            evidence=parents,
            evidence_card=[2] * len(parents)
        )

        cpds_condizionate = []

        for sottocarico in parents:
            figli = get_children_of_node(config["Load Tree"], sottocarico)

            df = merge_anomaly_tables(sottocarico)

            # Frequenze condizionate
            group = df.groupby(figli)[sottocarico].value_counts(normalize=True).unstack().fillna(0)
            group = group[[0, 1]]
            group = group.reset_index()
            combinazioni = pd.DataFrame(list(product([0, 1], repeat=len(figli))), columns=figli)
            group_completo = pd.merge(combinazioni, group, how="left", on=figli).fillna(0)
            group_sorted = group_completo.sort_values(by=figli).drop(columns=figli)
            values = group_sorted.T.values

            # Per combinazioni mai viste: assegna [0.5, 0.5]
            col_sums = values.sum(axis=0)
            zero_cols = (col_sums == 0)

            values[0, zero_cols] = 0.5
            values[1, zero_cols] = 0.5

            cpd = TabularCPD(
                variable=sottocarico,
                variable_card=2,
                values=values,
                evidence=figli,
                evidence_card=[2 for _ in figli]
            )
            cpds_condizionate.append(cpd)
            print(f"✅ CPD condizionata creata per '{sottocarico}'\n")

            # salvataggio cpd condizionate
            evidence_names = cpd.get_evidence()
            evidence_combinations = list(product([0, 1], repeat=len(evidence_names)))

            prob_values = cpd.get_values().T

            df_cpd = pd.DataFrame(evidence_combinations, columns=evidence_names)
            for i in range(cpd.variable_card):
                df_cpd[f"P({cpd.variable}={i})"] = prob_values[:, i]

            output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "CPDs")
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"cpd_{sottocarico}.csv")
            df_cpd.to_csv(csv_path, index=False)

        model.add_cpds(*cpds_condizionate)
        model.add_cpds(cpd_case_study)
    return model

def run_BN(case_study: str):
    """
    Esegue l'inferenza bayesiana per un dato case study, utilizzando le probabilità di anomalia
    dei nodi foglia come soft evidence nella rete bayesiana.

    Per ogni combinazione univoca di Date, Context e Cluster, calcola la probabilità che ciascun
    nodo padre (interno) del load tree sia anomalo (stato=1), sulla base delle evidenze fornite
    dalle foglie.

    Args:
        case_study (str): Nome del case study da analizzare.

    Returns:
        pd.DataFrame: DataFrame con le probabilità di anomalia dei nodi interni per ciascuna riga.
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    model = build_BN_structural_model(case_study)

    foglie = find_leaf_nodes(config["Load Tree"])

    evidence_path = os.path.join(PROJECT_ROOT, "results", case_study, "evidences")

    dfs = []
    for foglia in foglie:
        path_csv = os.path.join(evidence_path, f"evd_{foglia}.csv")
        df = pd.read_csv(path_csv)
        df["foglia"] = foglia
        dfs.append(df[["Date", "Context", "Cluster", "anomaly_prob", "foglia"]])

    df_all = pd.concat(dfs)

    df_pivot = df_all.pivot_table(index=["Date", "Context", "Cluster"],
                                  columns="foglia", values="anomaly_prob").reset_index()

    results = []
    nodi_padri = find_parents_of_leaves(config["Load Tree"])

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
                    values=[1 - prob, prob]  # P(0), P(1)
                )
                virtual_evidence.append(factor)

        result = inference.query(variables=nodi_padri, virtual_evidence=virtual_evidence)

        row_result = {
            "Date": row["Date"],
            "Context": row["Context"],
            "Cluster": row["Cluster"]
        }
        for nodo in nodi_padri:
            marginal = result.marginalize(
                [n for n in nodi_padri if n != nodo],
                inplace=False
            )
            row_result[f"P({nodo}=1)"] = marginal.values[1]
            row_result[f"P({nodo}=0)"] = marginal.values[0]

        results.append(row_result)

    df_result = pd.DataFrame(results)

    output_path = os.path.join(PROJECT_ROOT, "results", case_study, "inference_results.csv")
    df_result.to_csv(output_path, index=False)
    print(f"✅ Inferenza completata per '{case_study}'\n")

    return df_result


if __name__ == "__main__":
     df = run_BN("Cabina")


















