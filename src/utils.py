import os
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import hdbscan
from scipy.spatial.distance import cdist
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
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

def clean_time_series(df: pd.DataFrame, unit: str = None) -> pd.DataFrame:
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

    if unit:
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


def run_energy_temp(case_study: str, sottocarico: str, context: int, cluster: int):
    """
    per ogni giorno appartenente al cluster estrae i profili di potenza del sottocarico e di temperatura nella
    time window relativa al context.
    Restituisce due DataFrame: uno relativo ai giorni normali e uno
    relativo a quelli anomali, etichettati sulla base della tabella delle anomalie.

    Per ciascun giorno vengono calcolati:
      - il profilo di potenza nella finestra oraria;
      - il profilo di temperatura esterna corrispondente;
      - la media della temperatura;
      - il giorno della settimana (0=lunedì);
      - il sottocarico identificato da un numero (per input alla temp_XGboost);
      - il contesto e il cluster di appartenenza.

    Args:
        case_study (str): Nome del caso studio.
        sottocarico (str): Nome del file CSV del sottocarico (es. 'Rooftop 1').
        context (int): Numero del contesto.
        cluster (int): Numero del cluster.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_normals: DataFrame con i giorni normali e le seguenti colonne:
                ['power_profile', 'temp_profile', 'Mean_Temp', 'weekday', 'Subload', 'Context', 'Cluster']
            - df_anomalies: DataFrame con i giorni anomali e le stesse colonne sopra elencate.
    """
    results_path = os.path.join(PROJECT_ROOT, "results", case_study)
    anomaly_path = os.path.join(results_path, "anomaly_table")

    df_leaf = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{sottocarico}.csv"), index_col=0, parse_dates=True)
    df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{json.load(open(os.path.join(PROJECT_ROOT, 'data', case_study, 'config.json')))['Outside Temperature']}.csv"), index_col=0, parse_dates=True)
    df_temp.columns = ["Temperatura Esterna"]
    df_temp = df_temp.interpolate(method="time").bfill().ffill()
    df_leaf.columns = ["Power"]

    df_anm = pd.read_csv(os.path.join(anomaly_path, f"anomaly_table_{sottocarico}.csv"), index_col=0, parse_dates=True)
    groups = pd.read_csv(os.path.join(results_path, "groups.csv"), index_col=0, parse_dates=True)
    time_windows = pd.read_csv(os.path.join(results_path, "time_windows.csv"))
    time_windows['to'] = time_windows['to'].replace('24:00', '23:59')

    cluster_col = f"Cluster_{cluster}"
    cluster_series = groups[cluster_col]
    selected_dates = cluster_series[cluster_series == cluster].index.date if cluster_series.dtype != bool else cluster_series[cluster_series].index.date

    df_leaf = df_leaf[pd.Series(df_leaf.index.date).isin(selected_dates).values]
    selected_window = time_windows[time_windows["id"] == context].iloc[0]
    from_hour = pd.to_datetime(selected_window['from'], format='%H:%M').time()
    to_hour = pd.to_datetime(selected_window['to'], format='%H:%M').time()

    def is_in_window(ts):
        t = ts.time()
        return from_hour <= t < to_hour

    df_merged = df_leaf.join(df_temp, how="inner")
    df_merged = df_merged[df_merged.index.map(is_in_window)]

    if df_merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    days = df_merged.index.date
    daily_profiles = []
    for date in np.unique(days):
        day_mask = pd.Series(df_merged.index.date, index=df_merged.index) == date
        day_df = df_merged[day_mask]
        if day_df.empty:
            continue
        daily_profiles.append({
            "Date": date,
            "Energy": day_df["Power"].sum() * 0.25 / 1000,
            "Mean_Temp": np.mean(day_df["Temperatura Esterna"]),
            "Weekday": pd.Timestamp(date).weekday(),
            "Month": pd.Timestamp(date).month,
            "Subload": sottocarico,
            "Context": context,
            "Cluster": cluster
        })

    df_profiles = pd.DataFrame(daily_profiles).set_index("Date")
    anomalous_dates = df_anm.index.date
    df_anomalies = df_profiles[df_profiles.index.isin(anomalous_dates)].copy()
    df_normals = df_profiles[~df_profiles.index.isin(anomalous_dates)].copy()

    return df_normals, df_anomalies

def run_energy_temp_profile(case_study: str, sottocarico: str, context: int, cluster: int):
    """
    Per ogni giorno appartenente al cluster, estrae i dati quartorari di potenza e temperatura
    nella finestra oraria del contesto e costruisce un DataFrame codificato con:

      - data, mese, weekday;
      - ora e numero del quarto d’ora (0-3);
      - temperatura media giornaliera nella finestra (temp_mean);
      - temperatura e potenza per ogni timestep;
      - numero del contesto e del cluster.

    I giorni vengono poi separati tra normali e anomali sulla base della tabella delle anomalie.

    Args:
        case_study (str): Nome del caso studio.
        sottocarico (str): Nome del file CSV del sottocarico (es. 'Rooftop 1').
        context (int): Numero del contesto.
        cluster (int): Numero del cluster.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_normals: DataFrame quartorario dei giorni normali.
            - df_anomalies: DataFrame quartorario dei giorni anomali.
    """
    results_path = os.path.join(PROJECT_ROOT, "results", case_study)
    anomaly_path = os.path.join(results_path, "anomaly_table")

    df_leaf = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{sottocarico}.csv"), index_col=0, parse_dates=True)
    df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{json.load(open(os.path.join(PROJECT_ROOT, 'data', case_study, 'config.json')))['Outside Temperature']}.csv"), index_col=0, parse_dates=True)
    df_temp.columns = ["Temperatura Esterna"]
    df_temp = df_temp.interpolate(method="time").bfill().ffill()
    df_leaf.columns = ["Power"]

    df_anm = pd.read_csv(os.path.join(anomaly_path, f"anomaly_table_{sottocarico}.csv"), index_col=0, parse_dates=True)
    groups = pd.read_csv(os.path.join(results_path, "groups.csv"), index_col=0, parse_dates=True)
    time_windows = pd.read_csv(os.path.join(results_path, "time_windows.csv"))
    time_windows['to'] = time_windows['to'].replace('24:00', '23:59')

    cluster_col = f"Cluster_{cluster}"
    cluster_series = groups[cluster_col]
    selected_dates = cluster_series[cluster_series == cluster].index.date if cluster_series.dtype != bool else cluster_series[cluster_series].index.date

    df_leaf = df_leaf[pd.Series(df_leaf.index.date).isin(selected_dates).values]
    df_temp = df_temp[df_temp.index.normalize().isin(pd.to_datetime(selected_dates))]

    selected_window = time_windows[time_windows["id"] == context].iloc[0]
    from_hour = pd.to_datetime(selected_window['from'], format='%H:%M').time()
    to_hour = pd.to_datetime(selected_window['to'], format='%H:%M').time()

    def is_in_window(ts):
        t = ts.time()
        return from_hour <= t < to_hour

    df_merged = df_leaf.join(df_temp, how="inner")
    df_merged = df_merged[df_merged.index.map(is_in_window)]

    if df_merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    records = []
    for date in np.unique(df_merged.index.date):
        day_mask = pd.Series(df_merged.index.date, index=df_merged.index) == date
        day_df = df_merged[day_mask].copy()
        if day_df.empty:
            continue

        temp_mean = day_df["Temperatura Esterna"].mean()
        weekday = pd.Timestamp(date).weekday()
        month = pd.Timestamp(date).month

        for timestamp, row in day_df.iterrows():
            records.append({
                "Date": timestamp.date(),
                "mese": month,
                "weekday": weekday,
                "ora": timestamp.hour,
                "quartodora": timestamp.minute // 15,
                "Context": context,
                "Cluster": cluster,
                "temp": row["Temperatura Esterna"],
                "temp_mean": round(temp_mean, 3),
                "Energy": row["Power"] * 0.25 / 1000
            })

    df_full = pd.DataFrame(records)
    df_full.set_index(pd.to_datetime(df_full["Date"]), inplace=True)
    df_full.drop(columns=["Date"], inplace=True)

    anomalous_dates = df_anm.index.date
    mask = pd.Series(df_full.index.date, index=df_full.index)

    df_anomalies = df_full[mask.isin(anomalous_dates)].copy()
    df_normals = df_full[~mask.isin(anomalous_dates)].copy()

    return df_normals, df_anomalies

def get_nodes_by_level(load_tree: dict) -> list[list[str]]:
    from collections import defaultdict, deque

    levels_dict = defaultdict(list)
    queue = deque([(load_tree, 0)])

    while queue:
        subtree, level = queue.popleft()
        for parent, children in subtree.items():
            levels_dict[level].append(parent)
            if isinstance(children, dict):
                queue.append((children, level + 1))

    max_level = max(levels_dict.keys())
    return [levels_dict[i] for i in reversed(range(max_level + 1))]  # bottom-up

def scale_data(X, method):
    if method is False:
        return X
    method = method.lower()
    if method == "zscore":
        return StandardScaler().fit_transform(X)
    elif method == "minmax":
        return MinMaxScaler().fit_transform(X)
    elif method == "robust":
        return RobustScaler().fit_transform(X)
    elif method == "maxabs":
        return MaxAbsScaler().fit_transform(X)
    else:
        raise ValueError(f"Metodo di normalizzazione non supportato: '{method}'")


if __name__ == "__main__":
    df = run_energy_in_tw("Cabina", "QE Pompe")
