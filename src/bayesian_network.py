from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from itertools import product
from src.utils import *


def build_BN_structural_model(case_study: str):
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

        # Probabilità condizionate
        parents = find_parents_of_leaves(config["Load Tree"])
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

            cpd = TabularCPD(
                variable=sottocarico,
                variable_card=2,
                values=values,
                evidence=figli,
                evidence_card=[2 for _ in figli]
            )
            cpds_condizionate.append(cpd)
            print(f"✅ CPD condizionata creata per '{sottocarico}'")

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
    return model
if __name__ == "__main__":
    build_BN_structural_model("Cabina")


















