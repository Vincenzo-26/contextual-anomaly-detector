import os
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import hdbscan
from scipy.spatial.distance import cdist
from datetime import datetime
import os
import pandas as pd
from datetime import datetime
from settings import PROJECT_ROOT
import json

def print_boxed_title(title: str, side_padding: int = 15):
    total_length = len(title) + side_padding * 2
    border = "=" * total_length
    spaces = " " * side_padding
    print(f"\n{border}")
    print(f"{spaces}{title}{spaces}")
    print(f"{border}\n")

def clean_time_series(df: pd.DataFrame, unit: str = "W") -> pd.DataFrame:
    """
    Pulisce e riallinea un DataFrame temporale con indice datetime.
    Converte in watt se i dati sono in kWh o Wh.
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    start = df.index.min()
    end = df.index.max()

    full_index = pd.date_range(start=start, end=end, freq="15min")
    df = df.reindex(full_index)
    df = df.interpolate(method="time")
    df = df[df.index.minute.isin([0, 15, 30, 45])]

    first_day = df.index[0].normalize()
    if df[df.index.normalize() == first_day].index.min().time() != pd.Timestamp("00:00").time():
        df = df[df.index.normalize() > first_day]

    last_day = df.index[-1].normalize()
    if df[df.index.normalize() == last_day].index.max().time() != pd.Timestamp("23:45").time():
        df = df[df.index.normalize() < last_day]

    # 🔁 Conversione in watt (se serve)
    if unit.lower() == "kwh":
        df = df * 4000
    elif unit.lower() == "wh":
        df = df * (1000 / 0.25)  # = 4000
    elif unit.lower() == "w":
        pass  # nessuna conversione necessaria
    else:
        print(f"⚠️ Unità sconosciuta: {unit} - nessuna conversione applicata.")

    return df

def find_parents_of_leaves(subtree: dict) -> list:
    """
    Ricorsivamente restituisce i nomi dei nodi che hanno solo figli foglia (cioè figli che sono dict vuoti).
    """
    parents_of_leaves = []

    for key, value in subtree.items():
        if isinstance(value, dict):
            # Se tutti i figli di questo nodo sono foglie, aggiungilo alla lista
            if all(isinstance(v, dict) and not v for v in value.values()):
                parents_of_leaves.append(key)
            else:
                # Altrimenti continua a cercare in profondità
                parents_of_leaves.extend(find_parents_of_leaves(value))

    return parents_of_leaves

def get_children_of_node(load_tree: dict, node: str) -> list:
    """
    Cerca i figli diretti di un nodo all'interno del Load Tree.
    """
    for parent, children in load_tree.items():
        if parent == node:
            return list(children.keys())
        # Ricorsione: cerca nei figli
        found = get_children_of_node(children, node)
        if found:
            return found
    return []

def find_leaf_nodes(subtree: dict) -> list:
    """
    Ricorsivamente restituisce i nomi dei nodi foglia (cioè nodi che hanno un dict vuoto).
    """
    leaf_nodes = []

    for key, value in subtree.items():
        if isinstance(value, dict):
            if not value:
                # Se il valore è un dizionario vuoto, è una foglia
                leaf_nodes.append(key)
            else:
                # Ricorsione nei figli
                leaf_nodes.extend(find_leaf_nodes(value))

    return leaf_nodes

def merge_anomaly_tables(sottocarico: str):
    anomaly_folder = os.path.join(PROJECT_ROOT, "results", sottocarico, "anomaly_table")
    merged = None

    for file in os.listdir(anomaly_folder):
        if file.endswith(".csv") and file.startswith("anomaly_table_"):
            file_path = os.path.join(anomaly_folder, file)
            df = pd.read_csv(file_path)
            sub_name = file.replace("anomaly_table_", "").replace(".csv", "")

            df = df[["Date", "Context", "Cluster"]].copy()
            df[sub_name] = 1

            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on=["Date", "Context", "Cluster"], how="outer")

    if merged is not None:
        cols = ["Date", "Context", "Cluster"] + [c for c in merged.columns if c not in ["Date", "Context", "Cluster"]]
        merged = merged[cols]
        merged = merged.fillna(0).astype({col: int for col in merged.columns if col not in ["Date", "Context", "Cluster"]})

        output_path = os.path.join(anomaly_folder, "anomaly_table_overall.csv")
        merged.to_csv(output_path, index=False)
    else:
        print(f"⚠️ No files for {sottocarico}\n")
    return merged

def run_energy_in_tw(case_study: str, sottocarico: str):
    """
        Calcolo dell'energia e della temperatura media per ciascuna finestra temporale per un dato sottocarico.

        Returns:
            pd.DataFrame: colonne [Date, Context, Cluster, Energy, Temperature]
    """
    data_path = os.path.join(PROJECT_ROOT, "data", case_study)
    results_path = os.path.join(PROJECT_ROOT, "results", case_study)

    # Carica file
    with open(os.path.join(data_path, "config.json"), "r") as f:
        config = json.load(f)

    df_tw = pd.read_csv(os.path.join(results_path, "time_windows.csv"))
    df_groups = pd.read_csv(os.path.join(results_path, "groups.csv"), parse_dates=["timestamp"])
    data = pd.read_csv(os.path.join(data_path, f"{sottocarico}.csv"), parse_dates=["timestamp"])

    temp_file = config["Outside Temperature"]
    df_temp = pd.read_csv(os.path.join(data_path, f"{temp_file}.csv"), parse_dates=["timestamp"])
    df_temp.columns = ["timestamp", "Temperature"]

    # Preprocess
    data["date"] = data["timestamp"].dt.date
    data["time"] = data["timestamp"].dt.time
    data["energy_Wh"] = data["value"] * 0.25 / 1000
    df_groups["date"] = df_groups["timestamp"].dt.date

    # Join temperatura
    data = data.merge(df_temp, on="timestamp", how="left")

    results = []

    for day in data["date"].unique():
        day_data = data[data["date"] == day]

        row_group = df_groups[df_groups["date"] == day]
        if row_group.empty:
            continue

        cluster_raw = row_group.iloc[0].drop(["timestamp", "date"])
        cluster = cluster_raw[cluster_raw].index[0].split("_")[-1] if cluster_raw.any() else "Unknown"

        for _, row in df_tw.iterrows():
            context = row["id"]

            from_time = datetime.strptime(row["from"], "%H:%M").time()
            to_time = datetime.strptime("23:59", "%H:%M").time() if row["to"] == "24:00" else datetime.strptime(row["to"], "%H:%M").time()

            tw_data = day_data[(day_data["time"] >= from_time) & (day_data["time"] < to_time)]

            energy = tw_data["energy_Wh"].sum()
            temp_mean = tw_data["Temperature"].mean()

            results.append({
                "Date": str(day),
                "Context": int(context),
                "Cluster": int(cluster),
                "Energy": energy,
                "Temperature": temp_mean
            })

    return pd.DataFrame(results)

def optimize_hdbscan_cluster_size(X, min_size_start=3, max_size_ratio=0.5, step=2):
    """
    Trova il valore ottimale di min_cluster_size che minimizza il rumore (label -1),
    aumentando progressivamente il valore fino a che i punti sono tutti assegnati.

    Args:
        X (np.ndarray): array (n_samples, n_features) con i dati.
        min_size_start (int): valore iniziale per min_cluster_size.
        max_size_ratio (float): massimo rapporto rispetto al numero di punti.
        step (int): incremento per ogni iterazione.

    Returns:
        dict: con chiavi {"best_size", "labels", "probs", "n_noise"}
    """
    N = len(X)
    max_size = max(3, int(N * max_size_ratio))
    best_result = None

    for size in range(min_size_start, max_size + 1, step):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
        labels = clusterer.fit_predict(X)
        probs = clusterer.probabilities_
        n_noise = np.sum(labels == -1)

        if n_noise == 0:
            return {"best_size": size, "labels": labels, "probs": probs, "n_noise": 0}

        if best_result is None or n_noise < best_result["n_noise"]:
            best_result = {"best_size": size, "labels": labels, "probs": probs, "n_noise": n_noise}

    return best_result

def run_energy_temp(case_study: str, sottocarico: str, context: int, cluster: int):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    results_path = os.path.join(PROJECT_ROOT, "results", case_study)
    anomaly_path = os.path.join(results_path, "anomaly_table")

    df_leaf = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{sottocarico}.csv"), index_col=0, parse_dates=True)
    temp_file = config["Outside Temperature"]
    df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{temp_file}.csv"), index_col=0, parse_dates=True)
    df_temp.columns = ["Temperatura Esterna"]
    df_temp = df_temp.interpolate(method="time").bfill().ffill()
    df_leaf.columns = ["Power"]
    df_anm = pd.read_csv(os.path.join(anomaly_path, f"anomaly_table_{sottocarico}.csv"), index_col=0, parse_dates=True)

    groups = pd.read_csv(os.path.join(results_path, "groups.csv"), index_col=0, parse_dates=True)
    time_windows = pd.read_csv(os.path.join(results_path, "time_windows.csv"))
    time_windows['to'] = time_windows['to'].replace('24:00', '23:59')

    cluster_col = f"Cluster_{cluster}"
    cluster_series = groups[cluster_col]

    if cluster_series.dtype == bool or set(cluster_series.unique()) <= {0, 1}:
        selected_dates = cluster_series[cluster_series == True].index.date
    else:
        selected_dates = cluster_series[cluster_series == cluster].index.date

    df_leaf = df_leaf[pd.Series(df_leaf.index.date).isin(selected_dates).values]

    selected_window = time_windows[time_windows["id"] == context].iloc[0]
    from_hour = pd.to_datetime(selected_window['from'], format='%H:%M').time()
    to_hour = pd.to_datetime(selected_window['to'], format='%H:%M').time()

    def is_in_time_window(ts):
        t = ts.time()
        return (from_hour <= t < to_hour)

    df_leaf = df_leaf[df_leaf.index.map(is_in_time_window)]
    df_merged = df_leaf.join(df_temp, how="inner")

    if df_merged.empty:
        print("Nessun dato trovato per il cluster e la fascia oraria selezionati.")
        return

    df_grouped = df_merged.groupby(df_merged.index.date).agg(
        Energy=("Power", lambda x: x.sum() * 0.25 / 1000),
        Temperature=("Temperatura Esterna", "mean")
    )

    anomalous_dates = df_anm.index.date
    df_anomalies = df_grouped[df_grouped.index.isin(anomalous_dates)]
    df_normals = df_grouped[~df_grouped.index.isin(anomalous_dates)]

    return df_normals, df_anomalies



if __name__ == "__main__":
    df = run_energy_in_tw("Cabina", "QE Pompe")
