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

# Setup del logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s](%(name)s) %(message)s')
begin_time = datetime.datetime.now()

# Caricamento dei dataframe necessari
df_anm = pd.read_csv("data/diagnosis/anomalies_table_overall.csv")
df_anm.rename(columns={'Date': 'timestamp'}, inplace=True)
df_tw = pd.read_csv("data/diagnosis/time_windows.csv")
df_cluster = pd.read_csv("data/diagnosis/cluster_data.csv")


# var = "el_UTA_1_1B_5"
# var = "el_UTA_2_2B_6"
# var = "el_UTA_3_3B_7"
# var = "el_UTA_4_4B_8"

var_list = ['el_UTA_1_1B_5', 'el_UTA_2_2B_6', 'el_UTA_3_3B_7', 'el_UTA_4_4B_8']


df_tot = pd.DataFrame(columns=["date", "Context", "Cluster"])


for var in var_list:

    # Dataframe temporaneo per le anomalie relative alla variabile corrente
    df_var = pd.DataFrame()

    print(f'\n\033[91m{var}\033[0m')

    for context in range(1, len(df_tw) + 1):

        from_tw = df_tw.iloc[context - 1]["from"]
        from_tw = pd.to_datetime(from_tw).time()
        from_tw_float = time_to_float(from_tw)

        to_tw = df_tw.iloc[context - 1]["to"]
        # if else per evitare errori di trasformazione dell'ora per le 24:00 e per non avere problemi per il numero di osservazioni
        # se si convertisse 24 in 00:00

        m_context = 1
        obs_per_hour = 4

        if context == 1:
            # manually define context if it is the beginning
            context_start = 0  # [hours] i.e., 00:00
            context_end = context_start + m_context  # [hours] i.e., 01:00
            # [observations] = ([hour]-[hour])*[observations/hour]
            m = int((hour_to_dec(df_tw["to"][context - 1]) - 0.25 - m_context) * obs_per_hour)
            # m = 23
        else:
            m = df_tw["observations"][context-1]  # [observations]
            context_end = hour_to_dec(df_tw["from"][context - 1]) + 0.25  # [hours]
            context_start = context_end - m_context  # [hours]


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
            # m_context = 24 - time_to_float(df_time_window["from"][0])
            obs_per_day = int(df_time_window["observations"][0])
        else:
            to_tw_float = time_to_float(to_tw)
            to_tw = pd.to_datetime(to_tw).time()
            df_time_window = pd.DataFrame({
                'observations': [int((to_tw_float - from_tw_float) * 4)],
                'from': from_tw.strftime("%H:%M"),
                'to': to_tw.strftime("%H:%M"),
            })
            # m_context = time_to_float(df_time_window["to"][0]) - time_to_float(df_time_window["from"][0])
            obs_per_day = int(df_time_window["observations"][0]) + 1

        context_start = time_to_float(df_time_window["from"][0])
        context_end = context_start + m_context
        # m = int(df_time_window["observations"][0])
        # obs_per_hour = 4


        # obs_per_day = int(df_time_window["observations"][0]) + 1
        tw_string = (f'Subsequences of {dec_to_hour(m / obs_per_hour)} h (m = {m}) that '
                     f'start in [{dec_to_hour(context_start)},{dec_to_hour(context_end)})')
        context_string_small = (f'ctx_from{dec_to_hour(context_start)}_'
                                f'to{dec_to_hour(context_end)}_m{dec_to_hour(m / obs_per_hour)}'
                                ).replace(":", "_")





        max_cluster = df_cluster['cluster'].nunique()
        for num_cluster in range(1, max_cluster + 1):

            print(f'\n*********************\nCONTEXT {context}, CLUSTER {num_cluster}, {tw_string} ({context_string_small})')
            anm_table_overall = pd.DataFrame(columns=["date", "Context", "Cluster"])

            # Date uniche del cluster
            unique_dates = df_cluster[df_cluster['cluster'] == f"Cluster_{num_cluster}"]['date'].unique().tolist()
            unique_dates = [pd.to_datetime(d).date() for d in unique_dates]

            # Dizionario con i df delle aule processate
            var_dict_df = preprocess_variabili_interne(var)

            for key, df in var_dict_df.items():
                var_dict_df[key] = df[
                    (df['time'] >= from_tw) &
                    (df['time'] <= to_tw) &
                    (df['date'].isin(unique_dates))
                    ]

            t_ext_done = False

            aule_list = var.split("_")[2:]
            for aula in aule_list:

                df_aula = var_dict_df[f"df_{aula}"].copy()
                df_aula.rename(columns={'t_ext': 'temp'}, inplace=True)

                # Lista delle variabili interne da analizzare
                var_int_list = [f"T_amb_{aula}", f"T_setpoint_{aula}", "t_ext"]

                for var_int in var_int_list:
                    if var_int == "t_ext":
                        df_aula = df_aula.copy()
                        df_aula["t_ext"] = df_aula["temp"]
                        if t_ext_done:
                            continue
                    data = df_aula.copy()
                    data = data[['timestamp', f'{var_int}', 'temp']]
                    data = data.rename(columns={f'{var_int}': "value"})

                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data['date'] = data['timestamp'].dt.date
                    unique_days = data['timestamp'].dt.date.unique()

                    group_df = pd.DataFrame({
                        'timestamp': unique_days,
                        'Cluster_1': True
                    })
                    group_df = group_df.sort_values(by='timestamp').reset_index(drop=True)
                    group_df.set_index('timestamp', inplace=True)
                    data.set_index('timestamp', inplace=True)
                    a = len(data) // obs_per_day

                    contexts = GeneralStaticManager([
                        range(
                            ((x * obs_per_day) + dec_to_obs(context_start, obs_per_hour)),
                            ((x * obs_per_day) + dec_to_obs(context_end, obs_per_hour))
                        )
                        for x in range(len(data) // obs_per_day)
                    ])

                    calc = AnytimeCalculator(m, data['value'].values)
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

                    if var_int not in anm_table_overall.columns:
                        anm_table_overall[var_int] = 0

                    for anomaly_date in anomalies_table_var["Date"]:
                        if anomaly_date not in anm_table_overall["date"].values:
                            new_row = {"date": anomaly_date, "Context": context, "Cluster": num_cluster}
                            for col in anm_table_overall.columns:
                                if col not in ["date", "Context", "Cluster"]:
                                    new_row[col] = 0
                            anm_table_overall = pd.concat([anm_table_overall, pd.DataFrame([new_row])],ignore_index=True)

                        anm_table_overall.loc[anm_table_overall["date"] == anomaly_date, var_int] = 1

                    num_anomalies_to_show = np.count_nonzero(~np.isnan(cmp_ad_score))
                    if num_anomalies_to_show > 0:
                        if num_anomalies_to_show > 10:
                            num_anomalies_to_show = 10
                        data_plot = data['value'].values.reshape((-1, obs_per_day))[group].T
                        data_plot = pd.DataFrame(data_plot, columns=group_dates)
                        data_plot.columns = pd.to_datetime(data_plot.columns).strftime('%Y-%m-%d')
                        fig = px.line(data_plot, line_shape="spline")
                        fig.update_traces(line=dict(color='rgba(128, 128, 128, 0.2)'))
                        for date in cmp_ad_score_dates:
                            cmp_ad_score_dates_fmt = pd.to_datetime(date).strftime('%Y-%m-%d')
                            index_anom_plot = data_plot.columns.get_loc(cmp_ad_score_dates_fmt)
                            fig.data[index_anom_plot].update(line=dict(color='red', width=2))
                        fig.update(layout=dict(
                            title=f"{var_int}",
                            xaxis_title=None,
                            yaxis_title="Power [kW]",
                            paper_bgcolor='rgba(0,0,0,0)',
                            showlegend=False
                        ))
                        time_interval_group = datetime.datetime.now() - begin_time_group
                        hours, remainder = divmod(time_interval_group.total_seconds(), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        string_anomaly_print = '- %s (%.3f s) \t-> %.d anomalies' % (
                            var_int, seconds, num_anomalies_to_show)
                        print(string_anomaly_print)
                    else:
                        string_anomaly_print = "- " + var_int + ' (-) \t\t-> no anomalies'
                        print(string_anomaly_print, "green")

                    if var_int == "t_ext":
                        t_ext_done = True

            df_var = pd.concat([df_var, anm_table_overall], ignore_index=True)

    df_tot = merge_anomaly_tables(df_tot, df_var)

cols = df_tot.columns.tolist()
fixed_order = ["date", "Context", "Cluster"]
if "t_ext" in cols:
    fixed_order.append("t_ext")
other_cols = [c for c in cols if c not in fixed_order]
df_tot = df_tot[fixed_order + other_cols]

for col in df_tot.columns:
    if col not in fixed_order:
        df_tot[col] = df_tot[col].fillna(0).astype(int)

df_tot.to_csv("data/diagnosis/anomalies_table_var/anomalies_var_table_overall.csv", index=False)

total_time = datetime.datetime.now() - begin_time
hours, remainder = divmod(total_time.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)
logger.info(f"TOTAL {str(int(minutes))} min {str(int(seconds))} s")


