import os
import pandas as pd
from settings import PROJECT_ROOT
from utils import clean_time_series, get_nodes_by_level
import json

def run_data(case_study: str):
    cleaned_data = {}
    common_start, common_end = None, None
    print(f"🔍 {case_study}...\n")

    with open(os.path.join(PROJECT_ROOT, "raw_data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    output_dir = os.path.join(PROJECT_ROOT, "data", case_study)
    os.makedirs(output_dir, exist_ok=True)

    levels = get_nodes_by_level(config["Load Tree"])
    all_nodes = [node for level in levels for node in level]

    max_len = max(len(node) for node in all_nodes)

    for node in all_nodes:
        raw_path = os.path.join(PROJECT_ROOT, "raw_data", f"{case_study}", f"{node}.csv")

        print(f"🔍 {node.ljust(max_len)}", end="")

        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        df_clean = clean_time_series(df, unit=None)

        if df_clean.empty:
            print(" ⚠️ df_clean è vuoto")
            continue

        cleaned_data[node] = df_clean

        start, end = df_clean.index.min(), df_clean.index.max()
        if common_start is None or start > common_start:
            common_start = start
        if common_end is None or end < common_end:
            common_end = end

        print(f"   from: {start} - to: {end}")
    if common_start is None or common_end is None:
        print("❌ Nessun dato valido trovato.")
        return

    aligned_index = pd.date_range(common_start, common_end, freq="15min")
    print(f"📅 Intervallo comune: {common_start} ➔ {common_end} ({len(aligned_index)} valori)\n")

    for node, df_clean in cleaned_data.items():
        df_aligned = df_clean.loc[common_start:common_end]
        df_aligned = df_aligned.reindex(aligned_index)

        for col in df_aligned.columns:
            df_col = df_aligned[[col]].copy()
            df_col.columns = ["value"]
            df_col["timestamp"] = aligned_index
            df_col = df_col[["timestamp", "value"]]
            out_path = os.path.join(output_dir, f"{node}.csv")
            df_col.to_csv(out_path, index=False)

    print(f"✅ Dati allineati salvati in {output_dir}")

if __name__ == "__main__":
    run_data("Total")