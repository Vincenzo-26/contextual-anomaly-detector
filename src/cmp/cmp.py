import datetime
import pandas as pd
import numpy as np

from src.cmp.utils import hour_to_dec, dec_to_obs, dec_to_hour
from src.distancematrix.calculator import AnytimeCalculator
from src.distancematrix.consumer.contextmanager import GeneralStaticManager
from src.distancematrix.consumer.contextual_matrix_profile import ContextualMatrixProfile
from src.distancematrix.generator.euclidean import Euclidean
from src.cmp.anomaly_detection_functions import anomaly_detection, extract_vector_ad_cmp, extract_vector_ad_energy


def cmp_calculation(data: pd.DataFrame, groups: pd.DataFrame, time_windows: pd.DataFrame, m_context: float = 1):
    """
    Execute the CMP calculation. Return the anomaly detection results.
    Args:
        data (pd.DataFrame): Dataframe containing a datetime index and the column 'value', which contains the
            values to analyze.
        groups (pd.DataFrame): Dataframe containing the groups.
        time_windows (pd.DataFrame): Dataframe containing the time windows.
        m_context (float): Context window size. Default is 1.
    Returns:
        pd.DataFrame: Dataframe containing the anomaly detection results.
    """

    obs_per_hour = int(np.median(data.resample('1h').count()))
    obs_per_day = int(np.median(data.resample('1d').count()))
    n_group = groups.shape[1]
    anomalies_table_overall = pd.DataFrame()

    for id_tw in range(len(time_windows)):
        if id_tw == 0:
            context_start = 0
            context_end = context_start + m_context
            m = int((hour_to_dec(time_windows["to"][id_tw]) - 0.25 - m_context) * obs_per_hour)
        else:
            m = time_windows["observations"][id_tw]
            context_end = hour_to_dec(time_windows["from"][id_tw]) + 0.25
            context_start = context_end - m_context

        context_string = (f'Subsequences of {dec_to_hour(m / obs_per_hour)} h (m = {m}) that '
                          f'start in [{dec_to_hour(context_start)},{dec_to_hour(context_end)})')

        context_string_small = (f'ctx_from{dec_to_hour(context_start)}_'
                                f'to{dec_to_hour(context_end)}_m{dec_to_hour(m / obs_per_hour)}'
                                ).replace(":", "_")

        print(f'\n*********************\nCONTEXT {str(id_tw + 1)} : {context_string} ({context_string_small})')

        contexts = GeneralStaticManager([
            range(
                # FROM  [observations]  = x * 96 [observations] + 0 [hour] * 4 [observation/hour]
                ((x * obs_per_day) + dec_to_obs(context_start, obs_per_hour)),
                # TO    [observations]  = x * 96 [observations] + (0 [hour] + 2 [hour]) * 4 [observation/hour]
                ((x * obs_per_day) + dec_to_obs(context_end, obs_per_hour)))
            for x in range(len(data) // obs_per_day)
        ])

        calc = AnytimeCalculator(m, data['value'].values)

        calc.add_generator(0, Euclidean())

        # We want to calculate CMP initialize element
        cmp = calc.add_consumer([0], ContextualMatrixProfile(contexts))

        # Calculate Contextual Matrix Profile (CMP)
        calc.calculate_columns(print_progress=True)
        print("")

        date_labels = data.index[::obs_per_day].strftime('%Y-%m-%d')

        for id_cluster in range(n_group):

            begin_time_group = datetime.datetime.now()
            # create this dataframe where dates cluster and anomalies scores will be saved
            df_result_context_cluster = pd.DataFrame()

            # get group name from dataframe
            group_name = groups.columns[id_cluster]

            # create empty group vector
            group = np.array(groups.T)[id_cluster]
            # get cmp from previously computed cmp
            group_cmp = cmp.distance_matrix[:, group][group, :]
            # substitute inf with zeros
            group_cmp[group_cmp == np.inf] = 0
            # get dates
            group_dates = data.index[::obs_per_day].values[group]

            df_result_context_cluster["Date"] = groups.index
            df_result_context_cluster["cluster"] = group

            vector_ad_cmp = extract_vector_ad_cmp(group_cmp=group_cmp)

            vector_ad_energy = extract_vector_ad_energy(
                group=group,
                data_full=data,
                tw=time_windows,
                tw_id=id_tw)

            # calculate anomaly score though majority voting
            cmp_ad_score = anomaly_detection(
                group=group,
                vector_ad=vector_ad_cmp)

            energy_ad_score = anomaly_detection(
                group=group,
                vector_ad=vector_ad_energy)

            df_result_context_cluster["cmp_score"] = cmp_ad_score
            df_result_context_cluster["energy_score"] = energy_ad_score

            cmp_ad_score = np.array(df_result_context_cluster["cmp_score"] + df_result_context_cluster["energy_score"])

            cmp_ad_score = np.where(cmp_ad_score < 6, np.nan, cmp_ad_score)
            # get date to plot
            cmp_ad_score_index = np.where(~np.isnan(cmp_ad_score))[0].tolist()
            cmp_ad_score_dates = date_labels[cmp_ad_score_index]

            anomalies_table = pd.DataFrame()
            anomalies_table["Date"] = cmp_ad_score_dates
            anomalies_table["Anomaly Score"] = cmp_ad_score[cmp_ad_score_index]
            anomalies_table["Rank"] = anomalies_table.index + 1
            anomalies_table["Cluster"] = id_cluster + 1
            anomalies_table["Context"] = id_tw + 1
            anomalies_table_overall = pd.concat([anomalies_table_overall, anomalies_table])

            num_anomalies_to_show = np.count_nonzero(~np.isnan(cmp_ad_score))

            time_interval_group = datetime.datetime.now() - begin_time_group
            hours, remainder = divmod(time_interval_group.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            if num_anomalies_to_show > 0:
                string_anomaly_print = '- %s (%.3f s) \t-> %.d anomalies' % (
                    group_name.replace('_', ' '), seconds, num_anomalies_to_show)
                print(string_anomaly_print)
            else:
                string_anomaly_print = '- %s (%.3f s) \t-> No anomalies' % (group_name.replace('_', ' '), seconds)
                print(string_anomaly_print)

    return anomalies_table_overall
