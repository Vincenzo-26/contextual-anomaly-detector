from typing import Union
# from loguru import logger
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import numpy as np


def run_clustering(data: pd.DataFrame, df_holidays: Union[None, pd.DataFrame]) -> pd.DataFrame:
    """
    Run hierarchical clustering algorithm with ward linkage method. The algorithm will cluster the data into 2 fixed clusters (sundays and saturdays) and a variable number of clusters for the working days.
    In particular, the number of clusters for the working days is determined by the silhouette score between 3 and 6 clusters.
    In this way are returned between 5 and 8 clusters in total.

    :param data: DataFrame with the data to cluster. It must have a datetime index and a column 'value' with the values to cluster.
    :param df_holidays: DataFrame with the holidays. It must have a datetime.date index.

    :return: DataFrame with the clusters. The rows are the dates and the columns are the clusters. The columns are named as 'Cluster_1', 'Cluster_2', ..., 'Cluster_n'.
    """

    # logger.info("🌲 Running Hierarchical clustering algorithm with ward linkage method.")
    print("🌲 Running Hierarchical clustering algorithm with ward linkage method.")
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
        ['value', 'date', 'time']]
    wd_daily_matrix = df_working_days.pivot(index='date', columns='time', values='value').dropna(axis=1)
    range_clusters = range(3, 6)
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

    def reassign_small_clusters(df, min_days=7):
        while True:
            cluster_counts = df['Cluster'].value_counts()
            small_clusters = cluster_counts[cluster_counts < min_days].index.tolist()
            valid_clusters = cluster_counts[cluster_counts >= min_days].index.tolist()
            if not small_clusters:
                break
            print(f"🔄 Too small clusters detected: " +
                  ", ".join([f"Cluster_{i}" for i in small_clusters]) +
                  ". Reassigning... ", end="")
            centroids = df[df['Cluster'].isin(valid_clusters)].drop(columns='Cluster').groupby(df['Cluster']).mean()
            for cl in small_clusters:
                rows = df[df['Cluster'] == cl].drop(columns='Cluster')
                if rows.empty:
                    continue
                # Calcola distanza da tutti i centroidi
                distances = cdist(rows, centroids, metric='euclidean')
                new_labels = centroids.index[np.argmin(distances, axis=1)]
                df.loc[rows.index, 'Cluster'] = new_labels
            # logger.info("Done ✅")
            print("Done ✅")
        return df

    # check that all clusters have at least 'min_days' elements and recursively reassign
    wd_daily_matrix = reassign_small_clusters(wd_daily_matrix, min_days=7)

    # Grouping clusters
    group_cluster_df = pd.DataFrame({'timestamp': pd.to_datetime(data['date'].unique())})
    group_cluster_df['Cluster_1'] = group_cluster_df['timestamp'].dt.date.isin(Cluster1['date'])
    group_cluster_df['Cluster_2'] = group_cluster_df['timestamp'].dt.date.isin(Cluster2['date'])
    final_cluster_labels = sorted(wd_daily_matrix['Cluster'].unique())
    for i in final_cluster_labels:
        group_cluster_df[f'Cluster_{i}'] = group_cluster_df['timestamp'].dt.date.isin(
            wd_daily_matrix[wd_daily_matrix['Cluster'] == i].index
        )

    # logger.info(f"📊 Clustering algorithm completed successfully. Final number of cluster: {optimal_clusters + 2}")
    final_cluster_labels = sorted(wd_daily_matrix['Cluster'].unique())
    n_clusters_total = 2 + len(final_cluster_labels)  # 2 fissi (domenica + sabato) + quelli feriali riassegnati
    print(f"📊 Clustering algorithm completed successfully. Final number of clusters: {n_clusters_total}")
    for col in group_cluster_df.columns[1:]:
        count = group_cluster_df[col].sum()
        # logger.info(f"{col:<12} -> {int(count)} days")
        print(f"{col:<12} -> {int(count)} days")
    return group_cluster_df
