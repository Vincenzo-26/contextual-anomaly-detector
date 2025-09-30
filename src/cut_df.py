import os
import pandas as pd
from settings import PROJECT_ROOT
from utils import clean_time_series, get_nodes_by_level, print_boxed_title
import json
from src.cmp.groups_definition import run_clustering
from src.cmp.time_windows_definition import run_cart
from src.cmp.utils import extract_holidays
import datetime


start_date = datetime.date(2024, 3, 1)
end_date = datetime.date(2025, 2, 28)

case_study = "Total"
input_dir = os.path.join(PROJECT_ROOT, "raw_data", case_study)
output_dir = os.path.join(PROJECT_ROOT, "raw_data", "Total_cut")

os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)

        # Individua la colonna timestamp (modifica se ha nome diverso)
        if 'timestamp' not in df.columns:
            print(f"[!] Colonna 'timestamp' mancante in {filename}")
            continue

        # Converte la colonna timestamp in datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Trova i limiti interni del file
        file_min_date = df['timestamp'].dt.date.min()
        file_max_date = df['timestamp'].dt.date.max()

        # Calcola l'intervallo da usare: intersezione tra (start_date, end_date) e (file_min_date, file_max_date)
        actual_start = max(start_date, file_min_date)
        actual_end = min(end_date, file_max_date)

        # Filtro del dataframe
        mask = (df['timestamp'].dt.date >= actual_start) & (df['timestamp'].dt.date <= actual_end)
        df_filtered = df[mask]

        # Salva nel nuovo file
        output_path = os.path.join(output_dir, filename)
        df_filtered.to_csv(output_path, index=False)
        print(f"[✓] Salvato {filename} con {len(df_filtered)} righe nel range {actual_start} - {actual_end}")

