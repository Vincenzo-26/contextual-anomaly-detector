import datetime
import numpy as np
import pandas as pd
from statistics import mean
from scipy.stats import zscore

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

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s](%(name)s) %(message)s')

# Initialize final dataframe
df_tot = pd.DataFrame(columns=["date", "Context", "Cluster"])

# Load temperature and electric data
df_temp = pd.read_csv("data/Aule_R/raw_data/T_ext_aule_R.csv")
df_el = pd.read_csv("data/Aule_R/raw_data/electric_data_raw/data_el_aule_R.csv")
df_el = preprocess(df_el, df_temp)
df_el.set_index('timestamp', inplace=True)
df_el.rename(columns={'temp': 't_ext'}, inplace=True)

# Loop over each variable
for variable_name in df_el.columns[:-1]:
    print(f'\n\033[91m{variable_name}\033[0m')

    raw_data = dataformat2("data/Aule_R/raw_data/electric_data_raw/data_el_aule_R.csv",
                           variable_name,
                           "data/Aule_R/raw_data/T_ext_aule_R.csv")

    begin_time = datetime.datetime.now()

    data, obs_per_day, obs_per_hour = process_data(raw_data, variable_name)

    df_time_window = pd.read_csv("data/diagnosis/time_windows.csv")
    group_df = pd.read_csv("data/group_cluster_new.csv")
    group_df['timestamp'] = pd.to_datetime(group_df['timestamp'])
    group_df.set_index('timestamp', inplace=True)
    n_group = group_df.shape[1]

    m_context = 1

    anomalies_table_var = pd.DataFrame()

    for id_tw in range(len(df_time_window)):
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
            begin_time_group = datetime.datetime.now()
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

            cmp_ad_score = np.array(cmp_ad_score + energy_ad_score)
            cmp_ad_score = np.where(cmp_ad_score < 6, np.nan, cmp_ad_score)
            cmp_ad_score_index = np.where(~np.isnan(cmp_ad_score))[0].tolist()
            cmp_ad_score_dates = date_labels[cmp_ad_score_index]

            anomalies_table_partial = pd.DataFrame()
            anomalies_table_partial["Date"] = cmp_ad_score_dates
            anomalies_table_partial["Context"] = id_tw + 1
            anomalies_table_partial["Cluster"] = id_cluster + 1
            anomalies_table_var = pd.concat([anomalies_table_var, anomalies_table_partial], ignore_index=True)

            num_anomalies_to_show = np.count_nonzero(~np.isnan(cmp_ad_score))
            if num_anomalies_to_show > 0:
                if num_anomalies_to_show > 10:
                    num_anomalies_to_show = 10
                time_interval_group = datetime.datetime.now() - begin_time_group
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

    total_time = datetime.datetime.now() - begin_time
    seconds = total_time.total_seconds() % 60
    minutes = (total_time.total_seconds() // 60) % 60
    logger.info(f"TOTAL {str(int(minutes))} min {str(int(seconds))} s")

# Ordina colonne e salva
cols = df_tot.columns.tolist()
fixed = ["date", "Context", "Cluster"]
variable_cols = sorted([col for col in cols if col not in fixed])
df_tot = df_tot[fixed + variable_cols]

df_tot.to_csv("data/diagnosis/anomalies_table_var/anomalies_var_table_overall.csv", index=False)
