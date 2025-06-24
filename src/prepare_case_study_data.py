import os
import pandas as pd
from settings import PROJECT_ROOT
from utils import clean_time_series, get_nodes_by_level, print_boxed_title
import json

def run_data(case_study: str):
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
    print_boxed_title("Preprocessing 📊")
    cleaned_data = {}

    with open(os.path.join(PROJECT_ROOT, "raw_data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    output_dir = os.path.join(PROJECT_ROOT, "data", case_study)
    os.makedirs(output_dir, exist_ok=True)

    levels = get_nodes_by_level(config["Load Tree"])
    all_nodes = [node for level in levels for node in level]

    max_len = max(len(node) for node in all_nodes)

    for node in all_nodes:
        raw_path = os.path.join(PROJECT_ROOT, "raw_data", f"{case_study}", f"{node}.csv")

        print(f"\n🔍 {node.ljust(max_len)}")

        df = pd.read_csv(raw_path, parse_dates=True)

        # preprocessing
        df_clean = clean_time_series(df, unit=None)

        if df_clean.empty:
            print(" ⚠️ df_clean empty")
            continue

        cleaned_data[node] = df_clean

        start, end = df_clean['timestamp'].min(), df_clean['timestamp'].max()
        print(f"from: {start} - to: {end}")
        out_path = os.path.join(output_dir, f"{node}.csv")
        df_clean.to_csv(out_path, index=False)

    print(f"\n🔍 External temperature")
    temp_raw_path = os.path.join(PROJECT_ROOT, "raw_data", f"{case_study}", "Temperatura Esterna.csv")
    df_temp = pd.read_csv(temp_raw_path, parse_dates=True)
    df_temp_clean = clean_time_series(df_temp, unit=None)
    start, end = df_temp_clean['timestamp'].min(), df_temp_clean['timestamp'].max()
    print(f"from: {start} - to: {end}")
    out_path = os.path.join(output_dir, f"Temperatura Esterna.csv")
    df_temp_clean.to_csv(out_path, index=False)

    print(f"✅📊 Preprocessed data saved in {output_dir}")

if __name__ == "__main__":
    run_data("Total")