import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
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
"""


var_list = ['el_UTA_1_1B_5', 'el_UTA_2_2B_6', 'el_UTA_3_3B_7', 'el_UTA_4_4B_8']
var_aule_list = ['T_amb', 'T_setpoint']

models = {}
# modelli strutturali
for uta in var_list:
    archi = [(f"{uta}", f"int_fault_{uta}"), (f"{uta}", "t_ext"), (f"{uta}", "Condizionamento")]
    aule = uta.split("_")[2:]
    for var in var_aule_list:
        for aula in aule:
            archi.append((f"{uta}", f"{var}_{aula}"))
    model = BayesianNetwork(archi)

    # probabilità a priori dei nodi senza padre
    cpd_priori = TabularCPD(variable=f'{uta}', variable_card=2, values=[[0.9], [0.1]])
    model.add_cpds(cpd_priori)

    # probabilità condizionate
    cpd_cond = TabularCPD(variable="Condizionamento",
                               variable_card=2,
                               values=[[0.9, 0.1],
                                       [0.1, 0.9]],
                               evidence=[f"{uta}"],
                               evidence_card=[2])
    model.add_cpds(cpd_cond)

    # fault interno
    cpd_int_fault = TabularCPD(variable=f"int_fault_{uta}",
                               variable_card=2,
                               values=[[0.9, 0.1],
                                       [0.1, 0.9]],
                               evidence=[f"{uta}"],
                               evidence_card=[2])
    model.add_cpds(cpd_int_fault)

    # fault temperatura esterna
    cpd_t_ext = TabularCPD(variable=f"t_ext",
                           variable_card=2,
                           values=[[0.8, 0.2],
                                   [0.2, 0.8]],
                           evidence=[f"{uta}"],
                           evidence_card=[2])
    model.add_cpds(cpd_t_ext)


    cpd_var_dict = {}
    for var in var_aule_list:
        for aula in aule:
            key = f"cpd_{var}_{aula}"
            cpd = TabularCPD(
                variable=f"{var}_{aula}",
                variable_card=2,
                values=[[0.85, 0.15],
                        [0.15, 0.85]],
                evidence=[f"{uta}"],
                evidence_card=[2]
            )
            cpd_var_dict[key] = cpd
    model.add_cpds(*cpd_var_dict.values())

    # mettere i vari modelli nel dizionario, uno per ogni uta
    models[f"model_{uta}"] = model

df_evidence = pd.read_csv("data/diagnosis/anomalies_table_var/probability_table.csv")

evidence = {}
virtual_evidence = []

var_inference = 'el_UTA_1_1B_5'
aule_inference = var_inference.split("_")[2:]
for index, row in df_evidence.iterrows():
    evidence['Condizionamento'] = 1 # 1 = fault
    virtual_evidence = []
    for var_int in df_evidence.drop(columns=['Date', 'Cluster', 'Context']).columns:
        parts = var_int.split("_")
        if len(parts) > 2 and parts[2] in aule_inference:
            virtual_evidence.append(
                TabularCPD(var_int, 2, [[row[var_int]], [1 - row[var_int]]])
            )

    inference = VariableElimination(models[f"model_{var_inference}"])
    phi_query = inference.query([f'{var_inference}'], evidence=evidence, virtual_evidence=virtual_evidence)
    # prob_df = prob_df.append({'el_UTA_1_1B_5_fault': phi_query.values[0], 'el_UTA_1_1B_5_normal': phi_query.values[1]}, ignore_index=True)
    print(phi_query)








