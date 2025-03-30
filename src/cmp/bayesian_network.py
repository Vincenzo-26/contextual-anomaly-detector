import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from utils_hard_rules import *
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

r2_df_firma = pd.read_csv("data/diagnosis/anomalies_table_var/r2_df_firma.csv")
soglia_r2 = 0.6
df_evidence = pd.read_csv("data/diagnosis/anomalies_table_var/evidence%_el_&_var_full.csv")

models = {}
results_by_key = {}

# modelli strutturali
for idx, row in df_evidence.iterrows():
    for var in df_evidence.drop(columns=["date", "Context", "Cluster", "t_ext_score"]).columns:
        date = row['date']
        cluster = row['Cluster']
        context = row['Context']
        t_ext_score = row['t_ext_score']

        # Prendi il valore R²
        r2_value = r2_df_firma.loc[r2_df_firma["cluster"] == cluster, var].values[0]

        # Crea il grafo del modello in base al valore di R²
        archi = [(f"{var}", f"power_KPI_{var}")]
        if r2_value >= soglia_r2:
            archi.append((f"{var}", f"t_ext_KPI_{var}"))

        model = BayesianNetwork(archi)

        # CPD a priori
        cpd_priori = TabularCPD(variable=f'{var}', variable_card=2, values=[[0.9], [0.1]])
        model.add_cpds(cpd_priori)

        # CPD power KPI
        cpd_power_KPI = TabularCPD(
            variable=f"power_KPI_{var}",
            variable_card=2,
            values=[[0.9, 0.1], [0.1, 0.9]],
            evidence=[f"{var}"],
            evidence_card=[2]
        )
        model.add_cpds(cpd_power_KPI)

        # CPD temperatura esterna, solo se R² alto
        if r2_value >= soglia_r2:
            cpd_t_ext_KPI = TabularCPD(
                variable=f"t_ext_KPI_{var}",
                variable_card=2,
                values=[[0.9, 0.1], [0.1, 0.9]],
                evidence=[f"{var}"],
                evidence_card=[2]
            )
            model.add_cpds(cpd_t_ext_KPI)

        # Salva il modello
        model_key = f"model_{var}"
        models[model_key] = model

        # Inference
        result_row = {
            'date': date,
            'Cluster': cluster,
            'Context': context
        }

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
inference_results_df.to_csv("data/diagnosis/anomalies_table_var/inference_results_prova.csv", index=False)










