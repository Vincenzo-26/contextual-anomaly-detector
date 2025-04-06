import numpy as np
import pandas as pd
from statistics import mean
from scipy.stats import zscore
from datetime import datetime, time
from src.cmp.anomaly_detection_functions import (
    anomaly_detection,
    extract_vector_ad_temperature,
    extract_vector_ad_energy,
    extract_vector_ad_cmp,
)
from src.cmp.utils import *
from src.distancematrix.calculator import AnytimeCalculator
from src.distancematrix.consumer.contextmanager import GeneralStaticManager
from src.distancematrix.consumer.contextual_matrix_profile import ContextualMatrixProfile
from src.distancematrix.generator.euclidean import Euclidean
from utils_hard_rules import *

"""
OUTPUT:
-   anomalies_var_table_overall: tante righe quante sono le combinazioni di data - contesto - cluster anomale
    per almeno un sottocarico 
-   evidence_var_full: tante righe quante sono le combinazioni totali di date - context - cluster, sia anomale sia
    non anomale e per ogni sottocarico è associato il suo anomaly score nell'intervallo [0 - 8]
-   evidence%_var_full: evidence_var_full dove gli anomaly score di ogni date-context-cluster, sia anomali che normal
    sono trasformati in probabilità. (questo per dare delle evidenze anche alle anomalie a livello superiore non 
    diagnosticate da almeno un sottocarico a livello inferiore)
-   anomalies_el_&_var_table_overall: ha tante righe quanti sono le combinazioni date-context-cluster anomale
    comuni a livello superiore e in almeno uno dei sottocarichi (potrebbero non essere tutte quelle a livello
    superiore
-   evidence_el_&_var_full: è anomalies_el_&_var_table_overall con le varie date-context-cluster anomali associati
    al relativo anomaly score
-   evidence%_el_&_var_full: evidence_el_&_var_full con anomaly score trasformato in percenutale (evidenze per rete
    bayesiana)
-   t_ext_score_var_full: tante righe quante sono le combinazioni totali di date - context - cluster, sia anomale 
    sia non anomale ed è associata la temperatura media della time window relativa al contesto analizzato
"""

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s](%(name)s) %(message)s')

# Initialize final dataframe
df_tot = pd.DataFrame(columns=["date", "Context", "Cluster"])
evidence_var_full = pd.DataFrame(columns=["date", "Context", "Cluster"])
t_ext_score_var_full = pd.DataFrame(columns=["date", "Context", "Cluster"])

# Load temperature and electric data
df_temp = pd.read_csv("data/Aule_R/raw_data/T_ext_aule_R.csv")
df_el = pd.read_csv("data/Aule_R/raw_data/electric_data_raw/data_el_aule_R.csv")
df_el = preprocess(df_el, df_temp)
df_el.set_index('timestamp', inplace=True)
if 'temp' in df_el.columns:
    df_el.drop(columns='temp', inplace=True)

for variable_name in df_el.columns:
    print(f'\n\033[91m{variable_name}\033[0m')

    raw_data = dataformat2("data/Aule_R/raw_data/electric_data_raw/data_el_aule_R.csv",variable_name,"data/Aule_R/raw_data/T_ext_aule_R.csv")

    begin_time = datetime.now()

    data, obs_per_day, obs_per_hour = process_data(raw_data, variable_name)

    df_time_window = pd.read_csv("data/time_windows.csv")
    context_windows = {
        row["id"]: (hour_to_dec(row["from"]), hour_to_dec(row["to"]))
        for _, row in df_time_window.iterrows()
    }
    group_df = pd.read_csv("data/group_cluster_new.csv")
    group_df['timestamp'] = pd.to_datetime(group_df['timestamp'])
    group_df.set_index('timestamp', inplace=True)
    n_group = group_df.shape[1]

    m_context = 1

    anomalies_table_var = pd.DataFrame()

    for id_tw in range(len(df_time_window)):
    # for id_tw in range(2):
        if id_tw == 0:
            context_start = 0
            context_end = context_start + m_context
            m = int((hour_to_dec(df_time_window["to"][id_tw]) - 0.25 - m_context) * obs_per_hour)
        else:
            m = df_time_window["observations"][id_tw]
            context_end = hour_to_dec(df_time_window["from"][id_tw]) + 0.25
            context_start = context_end - m_context

        context_string_small = (f'ctx_from{dec_to_hour(context_start)}_'
                                f'to{dec_to_hour(context_end)}_m{dec_to_hour(m / obs_per_hour)}'
                                ).replace(":", "_")
        print(f'\n*********************\nCONTEXT {str(id_tw + 1)} : {context_string_small}')

        # creazione delle timeserie per ogni contesto


        contexts = GeneralStaticManager([
            range(((x * obs_per_day) + dec_to_obs(context_start, obs_per_hour)),
                  ((x * obs_per_day) + dec_to_obs(context_end, obs_per_hour)))
            for x in range(len(data) // obs_per_day)
        ])

        calc = AnytimeCalculator(m, data['value'].values)
        calc.add_generator(0, Euclidean())
        cmp = calc.add_consumer([0], ContextualMatrixProfile(contexts))
        calc.calculate_columns(print_progress=True)
        print("\n")
        date_labels = data.index[::obs_per_day].strftime('%Y-%m-%d')

        for id_cluster in range(n_group):
        # for id_cluster in range(2):
            begin_time_group = datetime.now()
            group_name = group_df.columns[id_cluster]
            group = np.array(group_df.T)[id_cluster]
            group_cmp = cmp.distance_matrix[:, group][group, :]
            group_cmp[group_cmp == np.inf] = 0
            group_dates = data.index[::obs_per_day].values[group]

            vector_ad_cmp = extract_vector_ad_cmp(group_cmp=group_cmp)
            vector_ad_energy = extract_vector_ad_energy(group, data, df_time_window, id_tw)
            vector_ad_temperature = extract_vector_ad_temperature(group, data, df_time_window, id_tw)

            cmp_ad_score = anomaly_detection(group, vector_ad_cmp)
            energy_ad_score = anomaly_detection(group, vector_ad_energy)
            temperature_ad_score = anomaly_detection(group, vector_ad_temperature)

            total_score = np.array(cmp_ad_score + energy_ad_score)
            cmp_ad_score = np.array(cmp_ad_score + energy_ad_score)
            cmp_ad_score = np.where(cmp_ad_score < 6, np.nan, cmp_ad_score)
            cmp_ad_score_index = np.where(~np.isnan(cmp_ad_score))[0].tolist()
            cmp_ad_score_dates = date_labels[cmp_ad_score_index]

            anomalies_table_partial = pd.DataFrame()
            anomalies_table_partial["Date"] = cmp_ad_score_dates
            anomalies_table_partial["Context"] = id_tw + 1
            anomalies_table_partial["Cluster"] = id_cluster + 1
            anomalies_table_var = pd.concat([anomalies_table_var, anomalies_table_partial], ignore_index=True)

            # dataframe con l'anomaly-score associato a ciascuna configurazione data-context-cluster (per ogni variabile)
            evidence_table_full_partial = pd.DataFrame({
                "date": date_labels,
                "Context": id_tw + 1,
                "Cluster": id_cluster + 1,
                variable_name: total_score
            })

            temp_data = data.copy()
            temp_data["date"] = temp_data.index.date
            temp_data["time_dec"] = temp_data.index.hour + temp_data.index.minute / 60
            from_dec, to_dec = context_windows[id_tw + 1]
            t_ext_score = temp_data[(temp_data["time_dec"] >= from_dec) &
                                     (temp_data["time_dec"] < to_dec)]
            t_ext_score = t_ext_score.groupby("date")["temp"].mean().reindex(pd.to_datetime(date_labels).date, fill_value=0)
            t_ext_score_full_partial = pd.DataFrame({
                "date": date_labels,
                "Context": id_tw + 1,
                "Cluster": id_cluster + 1,
                "t_ext_score": t_ext_score
            })
            # evidence_table_full_partial["t_ext_score"] = t_ext_score.values
            evidence_var_full = pd.concat([evidence_var_full, evidence_table_full_partial], ignore_index=True)
            t_ext_score_var_full = pd.concat([t_ext_score_var_full, t_ext_score_full_partial], ignore_index=True)

            num_anomalies_to_show = np.count_nonzero(~np.isnan(cmp_ad_score))
            if num_anomalies_to_show > 0:
                if num_anomalies_to_show > 10:
                    num_anomalies_to_show = 10
                time_interval_group = datetime.now() - begin_time_group
                seconds = time_interval_group.total_seconds() % 60
                string_anomaly_print = '- %s (%.3f s) \t\033[91m-> %d anomalies\033[0m' % (
                    group_name.replace('_', ' '), seconds, num_anomalies_to_show)
                print(string_anomaly_print)
            else:
                string_anomaly_print = "- " + group_name.replace('_', ' ') + ' (-)\t\033[92m-> no anomalies\033[0m'
                print(string_anomaly_print, "\033[92mgreen\033[0m")

    if not anomalies_table_var.empty:
        anomalies_table_var["date"] = anomalies_table_var["Date"]
        anomalies_table_var = anomalies_table_var[["date", "Context", "Cluster"]].copy()
        anomalies_table_var[variable_name] = 1

        df_tot = pd.merge(df_tot, anomalies_table_var, on=["date", "Context", "Cluster"], how="outer")
        for col in df_tot.columns:
            if col not in ["date", "Context", "Cluster"]:
                df_tot[col] = df_tot[col].fillna(0).astype(int)

    total_time = datetime.now() - begin_time
    seconds = total_time.total_seconds() % 60
    minutes = (total_time.total_seconds() // 60) % 60
    logger.info(f"{variable_name} {str(int(minutes))} min {str(int(seconds))} s")

fixed = ["date", "Context", "Cluster"]
cols = df_tot.columns.tolist()
variable_cols = sorted([col for col in cols if col not in fixed])
df_tot = df_tot[fixed + variable_cols]
df_tot.to_csv("data/diagnosis/Anomalie_tables/CMP/anomalies_var_table_overall.csv", index=False)

# filtrare evidence_var_full mantenendo solo le righe che corrispondono al cluster di appartenenza per ciascuna data
cluster_data = pd.read_csv("data/cluster_data.csv")
cluster_data['date'] = pd.to_datetime(cluster_data['date'])
cluster_map = (
    cluster_data.drop_duplicates(subset='date')
    .assign(Cluster=lambda df: df['cluster'].str.extract(r'Cluster_(\d+)').astype(int))
    [['date', 'Cluster']])
evidence_var_full['date'] = pd.to_datetime(evidence_var_full['date'])
evidence_var_full = pd.merge(
    evidence_var_full,
    cluster_map,
    on=['date', 'Cluster'],
    how='inner'
)
# fare il merge per non avere righe duplicate per le varie variabili
evidence_var_full = (
    evidence_var_full
    .groupby(["date", "Context", "Cluster"], as_index=False)
    .agg("max")
)
cols_ev = evidence_var_full.columns.tolist()
variable_cols_ev = sorted([col for col in cols_ev if col not in fixed])
evidence_var_full = evidence_var_full[fixed + variable_cols_ev]
evidence_var_full.to_csv("data/diagnosis/Evidence_tables/Power/CMP/evidence_var_full.csv", index=False)


####################
anm_table_el = pd.read_csv(f'data/anomalies_table_overall.csv') # dataframe con le anomalie elettiche
anm_table_var = df_tot.copy()

set_el = set(anm_table_el[['Date', 'Context', 'Cluster']].apply(tuple, axis=1))
anm_el_and_var = anm_table_var[anm_table_var[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_el)].reset_index(drop=True)
anm_el_and_var.to_csv(f"data/diagnosis/Anomalie_tables/CMP/anomalies_el_&_var_table_overall.csv", index=False)

evidence_var_full['date'] = evidence_var_full['date'].dt.strftime('%Y-%m-%d')
evidence_var_full['Context'] = evidence_var_full['Context'].astype(int)
evidence_var_full['Cluster'] = evidence_var_full['Cluster'].astype(int)
evidence_el_and_var_full = evidence_var_full[evidence_var_full[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_el)].reset_index(drop=True)
evidence_el_and_var_full.to_csv(f"data/diagnosis/Evidence_tables/Power/CMP/evidence_el_&_var_full.csv", index=False)

set_var = set(anm_table_var[['date', 'Context', 'Cluster']].apply(tuple, axis=1))
anm_table_el['anm_var'] = anm_table_el[['Date', 'Context', 'Cluster']].apply(lambda row: tuple(row) in set_var, axis=1)

num_anm_el_and_var = len(anm_el_and_var)
num_anm_tot = len(anm_table_el)
anm_string = (f"{num_anm_el_and_var} anomalie interne su {num_anm_tot} anomalie a livello superiore "
              f"({round(num_anm_el_and_var/num_anm_tot*100, 1)}%)")

evidence_var_full.set_index(['date', 'Context', 'Cluster'], inplace=True)
evidence_var_full = evidence_var_full.map(lambda x: x / 8)
evidence_var_full.reset_index(inplace=True)
evidence_var_full.to_csv(f"data/diagnosis/Evidence_tables/Power/CMP/evidence%_var_full.csv", index=False)

evidence_el_and_var_full = evidence_var_full[evidence_var_full[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_el)].reset_index(drop=True)
evidence_el_and_var_full.to_csv(f"data/diagnosis/Evidence_tables/Power/CMP/evidence%_el_&_var_full.csv", index=False)

# ---------------------------------
#             t-ext
# ---------------------------------
t_ext_score_var_full['date'] = pd.to_datetime(t_ext_score_var_full['date'])
t_ext_score_var_full = pd.merge(
    t_ext_score_var_full,
    cluster_map,
    on=['date', 'Cluster'],
    how='inner'
)
t_ext_score_var_full = (
    t_ext_score_var_full
    .groupby(["date", "Context", "Cluster"], as_index=False)
    .agg("max")
)

keys_anom = [tuple(x) for x in anm_el_and_var[['date', 'Context', 'Cluster']].values.tolist()]
keys_all = [tuple(x) for x in evidence_var_full[['date', 'Context', 'Cluster']].values.tolist()]
filtered_idx = [i for i, k in enumerate(keys_all) if k not in keys_anom]

# media e dev.std per trasformazione in z-score solo delle temperatura dei dati normal
t_ext_scores_filtered = t_ext_score_var_full.iloc[filtered_idx]['t_ext_score'].tolist()
mean_filtered = np.mean(t_ext_scores_filtered)
std_filtered = np.std(t_ext_scores_filtered, ddof=0)

# Calcolo z-score per tutti i punti
z_scores_all = ((t_ext_score_var_full['t_ext_score'] - mean_filtered) / std_filtered).tolist()
t_ext_score_var_full['t_ext_Z_score'] = z_scores_all
# trasformazione z-score in probabilità.
# Caso 1: tangente iperbolica
z_scores_array = np.array(z_scores_all)
probabilities_tanh = np.tanh(np.abs(z_scores_array)) * 100
t_ext_score_var_full['t_ext_score_tanh'] = probabilities_tanh.tolist()
t_ext_score_var_full.to_csv("data/diagnosis/Evidence_tables/T_ext/t_ext_score_var_full.csv", index=False)

total_time = datetime.now() - begin_time
seconds = total_time.total_seconds() % 60
minutes = (total_time.total_seconds() // 60) % 60
logger.info(f"TOTAL {str(int(minutes))} min {str(int(seconds))} s")