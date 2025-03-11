import pandas as pd
pd.set_option('display.max_columns', None)
from typing import Union
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeRegressor, _tree
import os
import numpy as np
import logging
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s](%(name)s) %(message)s')

def preprocess(df, df_temp):
    df.rename(columns={'Time': 'timestamp'}, inplace=True)
    df.rename(columns={'QE Pompe': 'el_pompe'}, inplace=True)
    df.rename(columns={'QE UTA 1/1B/5': 'el_UTA_1_1B_5'}, inplace=True)
    df.rename(columns={'QE UTA 2/2B/6': 'el_UTA_2_2B_6'}, inplace=True)
    df.rename(columns={'QE UTA 3/3B/7': 'el_UTA_3_3B_7'}, inplace=True)
    df.rename(columns={'QE UTA 4/4B/8': 'el_UTA_4_4B_8'}, inplace=True)
    df['total_power'] = df.drop(columns=['timestamp']).sum(axis=1)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_temp.rename(columns={'Time': 'timestamp'}, inplace=True)
    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])

    df = pd.merge(df, df_temp[['timestamp', 'Temperatura Esterna']], on='timestamp', how='left')
    df.rename(columns={'Temperatura Esterna': 't_ext'}, inplace=True)

    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    df = df[df['timestamp'].dt.minute.isin([0, 15, 30, 45])]
    df = df.drop_duplicates(subset='timestamp')

    first_ts = pd.to_datetime(df.iloc[0]['timestamp'])
    if first_ts.time() != pd.Timestamp("00:00:00").time():
        first_day = first_ts.date()
        df = df[df['timestamp'].dt.date != first_day]

    last_ts = pd.to_datetime(df.iloc[-1]['timestamp'])
    if last_ts.time() != pd.Timestamp("23:45:00").time():
        last_day = last_ts.date()
        df = df[df['timestamp'].dt.date != last_day]

    df.set_index('timestamp', inplace=True)

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min')
    df = df.reindex(full_index)
    df.index.name = 'timestamp'

    df = df.infer_objects(copy=False)
    df.interpolate(method='linear', inplace=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "..", "Aule_R", "data", "preprocess_data", f"el_aule_R.csv")
    df.to_csv(save_path)
    return df

def preprocess_variabili_interne(var):
    dfs_dict = {}
    aule = var.split("_") [2:]
    for aula in aule:
        file_name = f"data/Aule_R/raw_data/raw_aula_R{aula}.csv"
        df = pd.read_csv(file_name)
        df.rename(columns={'Time': 'timestamp'}, inplace=True)
        df.rename(columns={'Setpoint Effettivo': f'T_setpoint_{aula}'}, inplace=True)
        if ('Temperatura Ambiente Z1-Basso' in df.columns) and ('Temperatura Ambiente Z2-Alto' in df.columns):
            df[f'T_amb_{aula}'] = df[['Temperatura Ambiente Z1-Basso', 'Temperatura Ambiente Z2-Alto']].mean(axis=1)
            df.drop(columns=['Temperatura Ambiente Z1-Basso', 'Temperatura Ambiente Z2-Alto'], inplace=True)
        else:
            df.rename(columns={'Temperatura Ambiente': f'T_amb_{aula}'}, inplace=True)
        df.rename(columns={'Temperatura Esterna': 't_ext'}, inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # mantenere solo valori quartiorari
        df = df[df['timestamp'].dt.minute.isin([0, 15, 30, 45])]
        # eliminare righe duplicate
        df = df.drop_duplicates(subset='timestamp')
        # eliminare il primo giorno se non parte dalle 00:00 e quindi non è completo
        first_ts = pd.to_datetime(df.iloc[0]['timestamp'])
        if first_ts.time() != pd.Timestamp("00:00:00").time():
            first_day = first_ts.date()
            df = df[df['timestamp'].dt.date != first_day]
        # eliminare l'ultimo giorno se non finisce a 23:45 e quindi non è completo
        last_ts = pd.to_datetime(df.iloc[-1]['timestamp'])
        if last_ts.time() != pd.Timestamp("23:45:00").time():
            last_day = last_ts.date()
            df = df[df['timestamp'].dt.date != last_day]

        df.set_index('timestamp', inplace=True)

        # df con tutte le date univoche corrette nell'intervallo
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min')
        df = df.reindex(full_index)
        df.index.name = 'timestamp'

        df = df.infer_objects()
        df.interpolate(method='linear', inplace=True)
        # numeric_cols = df.select_dtypes(include=['number']).columns
        # df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        df = df.reset_index()
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time



        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, "data", "Aule_R", "preprocess_data", f"aula_R{aula}.csv")
        df.to_csv(save_path)
        dfs_dict[f"df_{aula}"] = df
    return dfs_dict

def run_clustering(data: pd.DataFrame, df_holidays: Union[None, pd.DataFrame], var) -> pd.DataFrame:
    """
    Run hierarchical clustering algorithm with ward linkage method. The algorithm will cluster the data into 2 fixed clusters (sundays and saturdays) and a variable number of clusters for the working days.
    In particular, the number of clusters for the working days is determined by the silhouette score between 3 and 6 clusters.
    In this way are returned between 5 and 8 clusters in total.

    :param data: DataFrame with the data to cluster. It must have a datetime index and a column 'value' with the values to cluster.
    :param df_holidays: DataFrame with the holidays. It must have a datetime.date index.

    :return: DataFrame with the clusters. The rows are the dates and the columns are the clusters. The columns are named as 'Cluster_1', 'Cluster_2', ..., 'Cluster_n'.
    """
    data['date'] = data.index.date
    data['time'] = data.index.time
    if df_holidays is not None:
        # of sundays and holidays
        sunday_dates = [
            date for date in data['date'].unique()
            if pd.Timestamp(date).weekday() == 6 or date in df_holidays.index
        ]
    else:
        # Only sundays
        sunday_dates = [date for date in data['date'].unique() if pd.Timestamp(date).weekday() == 6]
    Cluster1 = pd.DataFrame({'date': sunday_dates})

    # Cluster of saturdays
    saturdays = [
        date for date in data['date'].unique()
        if pd.Timestamp(date).weekday() == 5
    ]
    Cluster2 = pd.DataFrame({'date': saturdays})

    # Hierarchical clustering
    df_working_days = data[~data['date'].isin(set(Cluster1['date']).union(set(Cluster2['date'])))][
        [f'{var}', 'date', 'time']]
    wd_daily_matrix = df_working_days.pivot(index='date', columns='time', values=f'{var}')
    range_clusters = range(3, 5)
    silhouette_scores = []
    for n_clusters in range_clusters:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clustering.fit_predict(wd_daily_matrix)
        score = silhouette_score(wd_daily_matrix, cluster_labels)
        silhouette_scores.append(score)
    optimal_clusters = range_clusters[silhouette_scores.index(max(silhouette_scores))]
    final_clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
    final_labels = final_clustering.fit_predict(wd_daily_matrix) + 3
    wd_daily_matrix['Cluster'] = final_labels

    # Grouping clusters
    group_cluster_df = pd.DataFrame({'timestamp': pd.to_datetime(data['date'].unique())})
    group_cluster_df['Cluster_1'] = group_cluster_df['timestamp'].dt.date.isin(Cluster1['date'])
    group_cluster_df['Cluster_2'] = group_cluster_df['timestamp'].dt.date.isin(Cluster2['date'])
    for i in range(3, optimal_clusters + 3):
        group_cluster_df[f'Cluster_{i}'] = group_cluster_df['timestamp'].dt.date.isin(
            wd_daily_matrix[wd_daily_matrix['Cluster'] == i].index
        )

    # Creating csv
    file_path = os.path.join(os.path.dirname(__file__), 'data', f'cluster_df_{var}.csv')
    group_cluster_df.to_csv(file_path, index=False)

    logging.info(f"📊 Clustering algorithm completed successfully. Final number of cluster: {optimal_clusters + 2}")
    return group_cluster_df

def run_cart(data: pd.DataFrame, var) -> pd.DataFrame:
    """
    Fits a Decision Tree Regressor (CART) model on time-series data (excluding weekends) to identify time intervals
    based on power consumption. The model iteratively adjusts the minimum number of samples per leaf to ensure that all
    intervals have a duration of at least 2.5 hours. The final time intervals and related information are returned in a
    DataFrame.

    :param data: time-series data with a datetime index and a 'value' column
    :return: DataFrame with time intervals and related information. Columns include 'id', 'description', 'observations',
    'from', 'to', 'duration', and 'node'.
    """

    MIN_INTERVAL_LENGTH = 2.5
    min_samples_leaf = int(len(data) * 0.05)

    # Data preparation
    data['date'] = data.index.date
    data['time'] = data.index.time
    working_days_df = data[~data.index.weekday.isin([5, 6])]
    # working_days_df = working_days_df.drop(columns=['temp'])
    working_days_df = working_days_df.copy()
    working_days_df['time_numeric'] = working_days_df['time'].apply(lambda x: x.hour + x.minute / 60)

    # Defining X and y for the CART
    X = working_days_df['time_numeric'].to_numpy().reshape(-1, 1)
    y = working_days_df[f'{var}'].to_numpy().reshape(-1, 1)

    def extract_intervals(tree):
        """Extract the time intervals from the CART model

        :param tree: CART model

        :return: intervals, nodes
        """
        cart = tree.tree_
        intervals = []
        nodes = []
        node_counter = [1]

        def recurse(node, lower=0, upper=24):
            current_node = f"Node {node_counter[0]}"
            node_counter[0] += 1
            if cart.feature[node] != _tree.TREE_UNDEFINED:
                threshold = cart.threshold[node]
                left_child = cart.children_left[node]
                right_child = cart.children_right[node]
                recurse(left_child, lower, min(upper, threshold))
                recurse(right_child, max(lower, threshold), upper)
            else:
                intervals.append((lower, upper))
                nodes.append(current_node)

        recurse(0)
        return intervals, nodes

    n_iterations = 0
    iter_max = 10
    while n_iterations < iter_max:
        tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, max_depth=4, random_state=42)
        tree.fit(X, y)

        intervals, nodes = extract_intervals(tree)
        intervals = [(float(start), float(end)) for start, end in intervals]
        rounded_intervals = np.round(np.array(intervals) * 4) / 4
        intervals_duration = [(end - start) for start, end in rounded_intervals]

        if all(duration >= MIN_INTERVAL_LENGTH for duration in intervals_duration):
            time_windows = pd.DataFrame({
                'id': range(1, len(rounded_intervals) + 1),
                'description': [
                    f"[{int(start):02d}:{int((start % 1) * 60):02d} - {int(end):02d}:{int((end % 1) * 60):02d})"
                    for start, end in rounded_intervals
                ],
                'observations': [
                    int((end - start) * 4)
                    for start, end in rounded_intervals
                ],
                'from': [
                    f"{int(start):02d}:{int((start % 1) * 60):02d}"
                    for start, end in rounded_intervals
                ],
                'to': [
                    f"{int(end):02d}:{int((end % 1) * 60):02d}"
                    for start, end in rounded_intervals
                ],
                'duration': [
                    f"{int((end - start) * 3600)}s (~{(end - start):.2f} hours)"
                    for start, end in rounded_intervals
                ],
                'node': nodes
            })
            break
        else:
            min_samples_leaf += 500
    n_iterations += 1

    # Creating csv
    file_path = os.path.join(os.path.dirname(__file__), 'data', f'time_windows_{var}.csv')
    time_windows.to_csv(file_path, index=False)

    logging.info(f"📊 Cart algorithm completed successfully. Final number of time windows: {len(time_windows)}")
    return time_windows

def save_report(context, filepath: str) -> None:
    """Save the report to a file

    :param context: context of the report
    :param filepath: path to save the report

    """
    # Set up the Jinja2 environment for report
    path_to_templates = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(path_to_templates))
    template = env.get_template('template.html')

    # Render the template with the data
    html_content = template.render(context)

    # Save the rendered HTML to a file (optional, for inspection)
    with open(filepath, 'w', encoding='utf-8') as file:  # Specifica UTF-8
        file.write(html_content)
        logger.info(f'🎉 Report generated successfully on {filepath}')


def time_to_float(t):
    # Se t è una stringa, prova a convertirla in datetime
    if isinstance(t, str):
        try:
            t = pd.to_datetime(t)
        except Exception as e:
            raise ValueError(f"Impossibile convertire la stringa '{t}' in datetime: {e}")

    # Se t non ha gli attributi 'hour' e 'minute', solleva un errore
    if not hasattr(t, 'hour') or not hasattr(t, 'minute'):
        raise ValueError("L'input non ha un formato riconosciuto (manca 'hour' o 'minute').")

    return t.hour + t.minute / 60