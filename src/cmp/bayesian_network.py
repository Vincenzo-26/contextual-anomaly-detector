import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from utils_hard_rules import *
import matplotlib.pyplot as plt
import networkx as nx

import os
import numpy as np
############ 0 = NORMALE     1 = FAULT ###########

# per la DEFINIIONE DELLE PROBABILITà CONDIZIONALI E A PRIORI
"""
Nel TabularCPD si specificano le probabilità condizionali, ovvero la probabilità che "variable" assuma un certo stato 
dato lo stato delle sue "evidence" (es. la probabilità di int_fault_1_1B_5 dato lo stato di el_UTA_1_1B_5).

Nel parametro "values" si passa una matrice in forma di lista di liste (ogni lista è una riga) in cui:
- Ogni riga corrisponde a uno stato della "variable" (es. int_fault_1_1B_5  0 = NORMALE e 1 = FAULT).
- Ogni colonna rappresenta una particolare configurazione degli stati delle "evidence" (es. el_UTA_1_1B_5 che ha due stati).

Per ogni colonna la somma delle probabilità deve essere uguale a 1.

INPUT:
-   t_ext_score_var_full da "main_var.py"
-   evidence%_el_&_var_full da "main_var.py"
-   evidence_el_&_var_ditrib da "power_evidence.py"
-   r2_df_firma.csv da "firme.py"

OUTPUT:
-   inference_results: ha tante righe quante sono le anomalie ad alto livello e in corrispondenza di 
    ciascun sottocarico c'è la probabilità che sia stato lui a causare l'anomalia a livello superiore. Le probabilità
    a posteriori della temperatura sono fisse, quelle dei sottocarichi provengono dagli score CMP
-   inference_results_distrib: uguale a inference_results ma le probabilitàa  posteriori provengono dalle soglie
"""

r2_df_firma = pd.read_csv("data/diagnosis/r2_df_firma.csv")
df_evidence_t_ext_full = pd.read_csv("data/diagnosis/Evidence_tables/T_ext/t_ext_score_var_full.csv")
df_evidence = pd.read_csv("data/diagnosis/Evidence_tables/Power/CMP/evidence%_el_&_var_full.csv")
df_evidence_distrib = pd.read_csv("data/diagnosis/Evidence_tables/Power/Soglie/evidence_el_&_var_ditrib.csv")

soglia_r2 = 0.8

def run_bayesian_inference(df_evidence, output_path):
    models = {}
    results_by_key = {}

    for idx, row in df_evidence.iterrows():
        for var in df_evidence.drop(columns=["date", "Context", "Cluster"]).columns:
            date = row['date']
            cluster = row['Cluster']
            context = row['Context']
            t_ext_score = df_evidence_t_ext_full[
                (df_evidence_t_ext_full['date'] == date) &
                (df_evidence_t_ext_full['Cluster'] == cluster) &
                (df_evidence_t_ext_full['Context'] == context)
            ]['t_ext_score_tanh'].values[0]

            r2_value = r2_df_firma.loc[r2_df_firma["cluster"] == cluster, var].values[0]

            archi = [(f"{var}", f"power_KPI_{var}")]
            if r2_value >= soglia_r2:
                archi.append((f"{var}", f"t_ext_KPI_{var}"))

            model = BayesianNetwork(archi)

            #----- plot BN--------#
            # Percorso di salvataggio dinamico
            graph_folder = "data/diagnosis/Inference/graphs_BN"
            os.makedirs(graph_folder, exist_ok=True)
            graph_filename = f"{date}_{context}_{cluster}_{var}.png"
            graph_path = os.path.join(graph_folder, graph_filename)

            G = nx.DiGraph()
            G.add_edges_from(model.edges())
            plt.figure(figsize=(4, 4))
            pos = {}
            nodes = list(G.nodes)

            if len(nodes) == 2:
                # Layout verticale
                variable_node = [n for n in nodes if not n.startswith('power_KPI')][0]
                kpi_node = [n for n in nodes if n.startswith('power_KPI')][0]
                pos[variable_node] = (0, 1)
                pos[kpi_node] = (0, 0)
            else:
                # Layout ad albero
                variable_node = [n for n in nodes if not n.startswith(('power_KPI', 't_ext_KPI'))][0]
                power_node = [n for n in nodes if n.startswith('power_KPI')][0]
                text_node = [n for n in nodes if n.startswith('t_ext_KPI')][0]
                pos[variable_node] = (0, 1)
                pos[power_node] = (-0.7, 0)
                pos[text_node] = (0.7, 0)

            node_size = 1200
            node_colors = ['#5dade2' if not n.startswith(('power_KPI', 't_ext_KPI')) else '#58d68d' for n in nodes]

            # Disegno nodi
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=node_size,
                edgecolors='black',
                linewidths=0.8
            )

            # Disegno archi senza freccia (arrowsize = 0) e colore migliorato
            nx.draw_networkx_edges(
                G, pos,
                edge_color='gray',
                arrows=False,  # disattiva la freccia
                width=1
            )

            # Etichette distanziate di più dai cerchi
            labels = {}
            label_positions = {}
            for node in nodes:
                if node.startswith('power_KPI'):
                    labels[node] = "Power KPI"
                    label_positions[node] = (pos[node][0], pos[node][1] - 0.25)  # aumentata la distanza sotto
                elif node.startswith('t_ext_KPI'):
                    labels[node] = "T ext KPI"
                    label_positions[node] = (pos[node][0], pos[node][1] - 0.25)  # aumentata la distanza sotto
                else:
                    labels[node] = node
                    label_positions[node] = (pos[node][0], pos[node][1] + 0.25)  # aumentata la distanza sopra

            for node, (x, y) in label_positions.items():
                plt.text(x, y, labels[node],
                         fontsize=9,
                         fontweight='bold',
                         ha='center',
                         va='center')

            plt.axis('off')
            plt.margins(x=0.4, y=0.4)
            plt.savefig(graph_path, bbox_inches='tight', dpi=150)
            plt.close()


            cpd_priori = TabularCPD(variable=f'{var}', variable_card=2, values=[[0.9], [0.1]])
            model.add_cpds(cpd_priori)
            cpd_power_KPI = TabularCPD(
                variable=f"power_KPI_{var}",
                variable_card=2,
                values=[[0.9, 0.1], [0.1, 0.9]],
                evidence=[f"{var}"],
                evidence_card=[2]
            )
            model.add_cpds(cpd_power_KPI)

            if r2_value >= soglia_r2:
                cpd_t_ext_KPI = TabularCPD(
                    variable=f"t_ext_KPI_{var}",
                    variable_card=2,
                    values=[[0.9, 0.1], [0.1, 0.9]],
                    evidence=[f"{var}"],
                    evidence_card=[2]
                )
                model.add_cpds(cpd_t_ext_KPI)

            model_key = f"model_{var}"
            models[model_key] = model

            if model_key not in models:
                continue

            power_kpi = f"power_KPI_{var}"
            virtual_evidence = [
                TabularCPD(power_kpi, 2, [[1 - row[var]], [row[var]]])
            ]

            if r2_value >= soglia_r2:
                t_ext_kpi = f"t_ext_KPI_{var}"
                virtual_evidence.append(
                    TabularCPD(t_ext_kpi, 2, [[1 - t_ext_score], [t_ext_score]])
                )

            inference = VariableElimination(models[model_key])
            phi_query = inference.query([var], virtual_evidence=virtual_evidence)

            key = (date, cluster, context)
            if key not in results_by_key:
                results_by_key[key] = {
                    'date': date,
                    'Cluster': cluster,
                    'Context': context
                }

            results_by_key[key][var] = phi_query.values[1]

    inference_results_df = pd.DataFrame(results_by_key.values())
    for var in inference_results_df.drop(columns=["date", "Context", "Cluster"]).columns:
        inference_results_df[var] = inference_results_df[var].multiply(100).round(3)
    inference_results_df.to_csv(output_path, index=False)

run_bayesian_inference(df_evidence, "data/diagnosis/Inference/CMP/inference_results.csv")
run_bayesian_inference(df_evidence_distrib, "data/diagnosis/Inference/Soglie/inference_results_distrib.csv")










