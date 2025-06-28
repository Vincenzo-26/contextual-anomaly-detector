import os
import pandas as pd
from settings import PROJECT_ROOT
from utils import clean_time_series, get_nodes_by_level, print_boxed_title
import json
from src.cmp.groups_definition import run_clustering
from src.cmp.time_windows_definition import run_cart
from src.cmp.utils import extract_holidays

def run_data(case_study: str, date_range: tuple[str, str] = None):
    """
    Per ogni nodo del load tree del caso studio:
      - carica i dati grezzi dal file CSV corrispondente;
      - esegue la pulizia della serie temporale (vedi funzione clean_time_series in utils.py);
      - identifica l’intervallo temporale comune a tutti i nodi validi;
      - riallinea le serie temporali su una griglia di tempo comune (frequenza 15 minuti);
      - salva i dati puliti e allineati in file CSV nella directory di output.

    Args:
        case_study (str): Nome del caso studio (corrispondente alla sottocartella all’interno di `raw_data/`).

    Returns:
        None
    """

    def filter_date_range(df, date_range):
        if date_range is None:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        data_min = df["timestamp"].min()
        data_max = df["timestamp"].max()
        if end < data_min or start > data_max:
            print(
                f"⚠️ Warning: specified date range ({start.date()} to {end.date()}) is outside the dataset range ({data_min.date()} to {data_max.date()}). Returning empty dataframe.")
            return df.iloc[0:0]
        if start > data_min or end < data_max:
            print(
                f"ℹ️ Info: trimming dataset to range {start.date()} to {end.date()} (original range was {data_min.date()} to {data_max.date()})")
        return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    print_boxed_title("Preprocessing & Alignment 🧹📊")

    output_dir = os.path.join(PROJECT_ROOT, "data", case_study)
    os.makedirs(output_dir, exist_ok=True)

    # Load config & Load Tree
    with open(os.path.join(PROJECT_ROOT, "raw_data", case_study, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(output_dir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=2)

    load_tree = config["Load Tree"]
    levels = get_nodes_by_level(load_tree)
    all_nodes = [node for level in levels for node in level]
    root_node = levels[-1][0]

    cleaned_data = {}

    # Step 1: Clean root node
    print(f"\n🔍 Cleaning root node: {root_node}")
    df_root_raw = pd.read_csv(os.path.join(PROJECT_ROOT, "raw_data", case_study, f"{root_node}.csv"))
    df_root_raw = filter_date_range(df_root_raw, date_range)

    df_root_clean, root_stats  = clean_time_series(df_root_raw)
    df_root_clean.to_csv(os.path.join(output_dir, f"{root_node}.csv"), index=False)
    cleaned_data[root_node] = df_root_clean.copy()
    valid_days_root = set(pd.to_datetime(df_root_clean["timestamp"]).dt.date)

    # Step 2: Clustering and Cart on root node for groups and time windows
    df_root_clean.set_index("timestamp", inplace=True)
    holidays = config.get("holidays", None)
    df_holidays = extract_holidays(df_root_clean, holidays) if holidays else None
    groups = run_clustering(df_root_clean, df_holidays, case_study)
    cluster_map = dict(zip(groups["timestamp"].dt.date, groups.drop("timestamp", axis=1).idxmax(axis=1)))
    time_windows = run_cart(df_root_clean)

    output_cls_ctx_dir = os.path.join(PROJECT_ROOT, "results", case_study)
    os.makedirs(output_cls_ctx_dir, exist_ok=True)

    groups.to_csv(os.path.join(output_cls_ctx_dir, "groups.csv"), index=False)
    time_windows.to_csv(os.path.join(output_cls_ctx_dir, "time_windows.csv"), index=False)

    # Step 3: Clean other nodes
    day_stats = [{
        "Node": root_node,
        "Complete": len(set(root_stats["dates"]["complete"])),
        "Interpolated": len(set(root_stats["dates"]["interpolated"])),
        "KNN": len(set(root_stats["dates"]["knn"])),
        "Removed": root_stats["removed"],
        "Rebuilt": 0
    }]
    for node in all_nodes:
        if node == root_node:
            continue

        print(f"\n🔧 Processing {node}")
        df_node_raw = pd.read_csv(os.path.join(PROJECT_ROOT, "raw_data", case_study, f"{node}.csv"))
        df_node_raw = filter_date_range(df_node_raw, date_range)

        df_node_clean, node_stats = clean_time_series(df_node_raw)
        df_node_clean["date"] = pd.to_datetime(df_node_clean["timestamp"]).dt.date

        days_node = set(df_node_clean["date"])
        extra_days = days_node - valid_days_root
        missing_days = valid_days_root - days_node

        # Remove extra days
        df_node_clean = df_node_clean[~df_node_clean["date"].isin(extra_days)]

        # Built leaked days
        df_missing_rebuilt = []
        for missing_day in missing_days:
            cluster = cluster_map.get(missing_day)
            if not cluster:
                continue

            # Trova giorni presenti con stesso cluster
            similar_days = [d for d in df_node_clean["date"].unique() if cluster_map.get(d) == cluster]
            if not similar_days:
                continue

            # Calcola centroide
            df_node_clean["time"] = pd.to_datetime(df_node_clean["timestamp"]).dt.time
            profiles = df_node_clean[df_node_clean["date"].isin(similar_days)]
            pivot = profiles.pivot(index="date", columns="time", values="value")
            centroid = pivot.mean().reset_index()
            centroid["date"] = missing_day
            centroid["timestamp"] = pd.to_datetime(centroid["date"].astype(str) + " " + centroid["time"].astype(str))
            df_missing_rebuilt.append(centroid[["timestamp", 0]].rename(columns={0: "value"}))

        if df_missing_rebuilt:
            df_node_clean = pd.concat([df_node_clean[["timestamp", "value"]]] + df_missing_rebuilt)

        df_node_clean = df_node_clean.sort_values("timestamp")

        df_node_clean.to_csv(os.path.join(output_dir, f"{node}.csv"), index=False)
        cleaned_data[node] = df_node_clean
        rebuilt_dates = set(pd.to_datetime(df_node_clean["timestamp"]).dt.date)
        known_dates = set(node_stats["dates"]["complete"]) | set(node_stats["dates"]["interpolated"]) | set(
            node_stats["dates"]["knn"])
        rebuilt_count = len(rebuilt_dates - known_dates)

        day_stats.append({
            "Node": node,
            "Complete": len(set(node_stats["dates"]["complete"])),
            "Interpolated": len(set(node_stats["dates"]["interpolated"])),
            "KNN": len(set(node_stats["dates"]["knn"])),
            "Removed": node_stats["removed"],
            "Rebuilt": rebuilt_count
        })
    df_summary = pd.DataFrame(day_stats)
    df_summary.to_csv(os.path.join(PROJECT_ROOT, "results", case_study, "day_reconstruction_summary.csv"), index=False)

    print("\n✅ All nodes cleaned and aligned.\n")

if __name__ == "__main__":
    run_data("Total", date_range=("2024-03-01", "2025-02-28"))