import datetime
import plotly.express as px
from src.cmp.anomaly_detection_functions import anomaly_detection, extract_vector_ad_temperature, \
    extract_vector_ad_energy, extract_vector_ad_cmp
from src.cmp.utils import *
from src.cmp.utils_hard_rules import *
from src.distancematrix.calculator import AnytimeCalculator
from src.distancematrix.consumer.contextmanager import GeneralStaticManager
from src.distancematrix.consumer.contextual_matrix_profile import ContextualMatrixProfile
from src.distancematrix.generator.euclidean import Euclidean

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s](%(name)s) %(message)s')
begin_time = datetime.datetime.now()

df_anm = pd.read_csv("data/diagnosis/anomalies_table_overall.csv")
df_anm.rename(columns={'Date': 'timestamp'}, inplace=True)
df_tw = pd.read_csv("data/diagnosis/time_windows.csv")
df_cluster = pd.read_csv("data/diagnosis/cluster_data.csv")

df_temp = pd.read_csv("data/Aule_R/raw_data/T_ext_aule_R.csv")

df_tot = pd.DataFrame(columns=["date", "Context", "Cluster"])

df_el = pd.read_csv("data/Aule_R/raw_data/electric_data_raw/data_el_aule_R.csv")
df_el = preprocess(df_el, df_temp)
df_el.set_index('timestamp', inplace=True)

for var in df_el.columns:
    df_var = pd.DataFrame()

    print(f'\n\033[91m{var}\033[0m')

    for context in range(1, len(df_tw) + 1):
        print("\n*********************")

        from_tw = df_tw.iloc[context - 1]["from"]
        from_tw = pd.to_datetime(from_tw).time()
        from_tw_float = time_to_float(from_tw)

        to_tw = df_tw.iloc[context - 1]["to"]
        m_context = 1
        obs_per_hour = 4

        if context == 1:
            context_start = 0
            context_end = context_start + m_context
            m = int((hour_to_dec(df_tw["to"][context - 1]) - 0.25 - m_context) * obs_per_hour)
        else:
            m = df_tw["observations"][context-1]
            context_end = hour_to_dec(df_tw["from"][context - 1]) + 0.25
            context_start = context_end - m_context

        if to_tw == "24:00":
            to_tw_float = 24
            observations = [int((to_tw_float - from_tw_float) * 4)]
            to_tw = "23:45"
            to_tw = pd.to_datetime(to_tw).time()
            df_time_window = pd.DataFrame({
                'observations': observations,
                'from': from_tw.strftime("%H:%M"),
                'to': "24:00",
            })
            obs_per_day = int(df_time_window["observations"][0])
        else:
            to_tw_float = time_to_float(to_tw)
            to_tw = pd.to_datetime(to_tw).time()
            df_time_window = pd.DataFrame({
                'observations': [int((to_tw_float - from_tw_float) * 4)],
                'from': from_tw.strftime("%H:%M"),
                'to': to_tw.strftime("%H:%M"),
            })
            obs_per_day = int(df_time_window["observations"][0]) + 1
        # obs_per_day = m

        tw_string = (f'Subsequences of {dec_to_hour(m / obs_per_hour)} h (m = {m}) that '
                     f'start in [{dec_to_hour(context_start)},{dec_to_hour(context_end)})')
        context_string_small = (f'ctx_from{dec_to_hour(context_start)}_'
                                f'to{dec_to_hour(context_end)}_m{dec_to_hour(m / obs_per_hour)}'
                                ).replace(":", "_")

        max_cluster = df_cluster['cluster'].nunique()
        for num_cluster in range(1, max_cluster + 1):

            print(f'\nCONTEXT {context}, CLUSTER {num_cluster}, {tw_string} ({context_string_small})')
            anm_table_overall = pd.DataFrame(columns=["date", "Context", "Cluster"])

            # Date uniche del cluster
            unique_dates = df_cluster[df_cluster['cluster'] == f"Cluster_{num_cluster}"]['date'].unique().tolist()
            unique_dates = [pd.to_datetime(d).date() for d in unique_dates]

            data = df_el.copy()
            current_var = var
            if current_var == "temp":
                data['t_ext'] = data['temp']
                current_var = "t_ext"


            data['timestamp'] = pd.to_datetime(data.index)
            data['date'] = data['timestamp'].dt.date
            data['time'] = data['timestamp'].dt.time

            data = data[
                (data['time'] >= from_tw) &
                (data['time'] <= to_tw) &
                (data['date'].isin(unique_dates))
                ]

            data = data[['timestamp', current_var, 'temp']]
            data = data.rename(columns={current_var: "value"})
            unique_days = data['timestamp'].dt.date.unique()

            group_df = pd.DataFrame({
                'timestamp': unique_days,
                'Cluster_1': True
            })
            group_df = group_df.sort_values(by='timestamp').reset_index(drop=True)
            group_df.set_index('timestamp', inplace=True)
            data.set_index('timestamp', inplace=True)

            contexts = GeneralStaticManager([
                range(
                    ((x * obs_per_day) + dec_to_obs(context_start, obs_per_hour)),
                    ((x * obs_per_day) + dec_to_obs(context_end, obs_per_hour))
                )
                for x in range(len(data) // obs_per_day)
            ])
            ########## NEL DF ORIGINALE C'è M AL POSTO DI OBS_PER_DAY IN CALC
            calc = AnytimeCalculator(obs_per_day, data['value'].values)
            ##########
            calc.add_generator(0, Euclidean())
            cmp = calc.add_consumer([0], ContextualMatrixProfile(contexts))
            calc.calculate_columns(print_progress=True)
            print("\n")
            ensure_dir(os.path.join(path_to_data, context_string_small))
            np.savetxt(os.path.join(path_to_data, context_string_small, 'plot_cmp_full.csv'),
                       nan_diag(cmp.distance_matrix),
                       delimiter=",")
            np.savetxt(os.path.join(path_to_data, context_string_small, 'match_index_query.csv'),
                       cmp.match_index_query,
                       delimiter=",")
            np.savetxt(os.path.join(path_to_data, context_string_small, 'match_index_series.csv'),
                       cmp.match_index_series,
                       delimiter=",")

            date_labels = data.index[::obs_per_day].strftime('%Y-%m-%d')
            df_anomaly_context = group_df.astype(int)
            df_result_context_cluster = pd.DataFrame()
            begin_time_group = datetime.datetime.now()
            group_name = group_df.columns[0]
            df_anomaly_context[f'{group_name}.{context_string_small}'] = [0 for _ in range(len(group_df))]
            group = np.array(group_df.T)[0]
            group_cmp = cmp.distance_matrix[:, group][group, :]
            group_cmp[group_cmp == np.inf] = 0
            group_dates = data.index[::obs_per_day].values[group]
            np.savetxt(os.path.join(path_to_data, context_string_small, f'plot_cmp_{group_name}.csv'),
                       nan_diag(group_cmp), delimiter=",")
            np.savetxt(os.path.join(path_to_data, context_string_small, f'match_index_query_{group_name}.csv'),
                       cmp.match_index_query[:, group][group, :], delimiter=",")
            np.savetxt(os.path.join(path_to_data, context_string_small, f'match_index_series_{group_name}.csv'),
                       cmp.match_index_series[:, group][group, :], delimiter=",")

            df_result_context_cluster["Date"] = group_df.index
            df_result_context_cluster["cluster"] = group
            vector_ad_cmp = extract_vector_ad_cmp(group_cmp=group_cmp)
            vector_ad_energy = extract_vector_ad_energy(
                group=group,
                data_full=data,
                tw=df_time_window,
                tw_id=0)
            vector_ad_temperature = extract_vector_ad_temperature(
                group=group,
                data_full=data,
                tw=df_time_window,
                tw_id=0)
            cmp_ad_score = anomaly_detection(
                group=group,
                vector_ad=vector_ad_cmp)
            energy_ad_score = anomaly_detection(
                group=group,
                vector_ad=vector_ad_energy)
            temperature_ad_score = anomaly_detection(
                group=group,
                vector_ad=vector_ad_temperature)
            df_result_context_cluster["cmp_score"] = cmp_ad_score
            df_result_context_cluster["energy_score"] = energy_ad_score
            df_result_context_cluster["temperature_score"] = temperature_ad_score

            cmp_ad_score = np.array(
                df_result_context_cluster["cmp_score"] + df_result_context_cluster["energy_score"])
            cmp_ad_score = np.where(cmp_ad_score < 6, np.nan, cmp_ad_score)
            cmp_ad_score_index = np.where(~np.isnan(cmp_ad_score))[0].tolist()
            cmp_ad_score_dates = date_labels[cmp_ad_score_index]

            anomalies_table_var = pd.DataFrame()
            anomalies_table_var["Date"] = cmp_ad_score_dates
            anomalies_table_var["Anomaly Score"] = cmp_ad_score[cmp_ad_score_index]
            anomalies_table_var["Context"] = context
            anomalies_table_var["Cluster"] = num_cluster

            for anomaly_date in anomalies_table_var["Date"]:
                if anomaly_date not in anm_table_overall["date"].values:
                    new_row = {"date": anomaly_date, "Context": context, "Cluster": num_cluster}
                    for col in anm_table_overall.columns:
                        if col not in ["date", "Context", "Cluster"]:
                            new_row[col] = 0
                    anm_table_overall = pd.concat([anm_table_overall, pd.DataFrame([new_row])],ignore_index=True)

                anm_table_overall.loc[anm_table_overall["date"] == anomaly_date, var] = 1

            num_anomalies_to_show = np.count_nonzero(~np.isnan(cmp_ad_score))
            if num_anomalies_to_show > 0:
                if num_anomalies_to_show > 10:
                    num_anomalies_to_show = 10

                time_interval_group = datetime.datetime.now() - begin_time_group
                hours, remainder = divmod(time_interval_group.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                string_anomaly_print = '- %s (%.3f s) \t-> %.d anomalies' % (
                    var, seconds, num_anomalies_to_show)
                print(string_anomaly_print)
            else:
                string_anomaly_print = "- " + var + " (-) \t\t-> no anomalies "
                print(string_anomaly_print + "\033[92mgreen\033[0m")

            df_var = pd.concat([df_var, anm_table_overall], ignore_index=True)

    df_tot = merge_anomaly_tables(df_tot, df_var)

cols = df_tot.columns.tolist()
fixed_order = ["date", "Context", "Cluster"]
other_cols = [c for c in cols if c not in fixed_order]
df_tot = df_tot[fixed_order + other_cols]

df_tot.to_csv("data/diagnosis/anomalies_table_var/anomalies_var_table_overall.csv", index=False)

total_time = datetime.datetime.now() - begin_time
hours, remainder = divmod(total_time.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)
logger.info(f"TOTAL {str(int(minutes))} min {str(int(seconds))} s")


