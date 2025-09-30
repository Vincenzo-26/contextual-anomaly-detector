import os
import pandas as pd
from utils import *
from settings import PROJECT_ROOT

# raw_path = os.path.join(PROJECT_ROOT, "raw_data", "AuleR.csv")
#
# output_dir = os.path.join(PROJECT_ROOT, "raw_data")
#
# df = pd.read_csv(raw_path)
#
# for col in df.columns:
#     if col == 'timestamp':
#         continue
#     df_out = df[['timestamp', col]].rename(columns={col: 'value'})
#     df_out['value'] = df_out['value']/1000
#     output_path = os.path.join(output_dir, f"{col}.csv")
#     df_out.to_csv(output_path, index=False)

def aggregate_selected_csv_sum(file_list, input_dir, output_file):
    dataframes = []

    for file in file_list:
        path = os.path.join(input_dir, file)
        df = pd.read_csv(path)

        # Verifica che ci sia 'timestamp' e solo un'altra colonna
        if df.shape[1] != 2 or 'timestamp' not in df.columns:
            raise ValueError(f"Il file {file} deve contenere una colonna 'timestamp' e una colonna dati.")

        # Rinomina la colonna dati con il nome del file (senza estensione)
        data_col = [col for col in df.columns if col != 'timestamp'][0]
        df = df.rename(columns={data_col: os.path.splitext(file)[0]})
        dataframes.append(df)

    # Merge esterno su 'timestamp'
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on='timestamp', how='outer')

    value_cols = merged_df.columns.drop('timestamp')

    # Crea flag: True se tutta la riga è NaN
    merged_df['all_nan_flag'] = merged_df[value_cols].isna().all(axis=1)

    # Somma i valori validi
    merged_df['Total'] = merged_df[value_cols].sum(axis=1, skipna=True)

    # Dove il flag è True → forziamo NaN nel totale
    merged_df.loc[merged_df['all_nan_flag'], 'Total'] = float('nan')

    # Drop del flag temporaneo
    merged_df.drop(columns='all_nan_flag', inplace=True)

    # Salva solo timestamp e Total
    result_df = merged_df[['timestamp', 'Total']]
    result_df.to_csv(output_file, index=False)
    print(f"Aggregated CSV salvato in: {output_file}")

def compute_difference_from_first(file_list, input_dir, output_file):
    if len(file_list) < 2:
        raise ValueError("Servono almeno due file CSV per calcolare la differenza.")

    dataframes = []

    # Caricamento e rinomina colonne
    for file in file_list:
        path = os.path.join(input_dir, file)
        df = pd.read_csv(path)

        if df.shape[1] != 2 or 'timestamp' not in df.columns:
            raise ValueError(f"Il file {file} deve contenere una colonna 'timestamp' e una colonna dati.")

        data_col = [col for col in df.columns if col != 'timestamp'][0]
        df = df.rename(columns={data_col: os.path.splitext(file)[0]})
        dataframes.append(df)

    # Merge esterno su 'timestamp'
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on='timestamp', how='outer')

    first_col = os.path.splitext(file_list[0])[0]
    other_cols = [col for col in merged_df.columns if col not in ['timestamp', first_col]]

    # Flag: True se tutti i valori da sottrarre sono NaN
    all_nan_others = merged_df[other_cols].isna().all(axis=1)

    # Somma degli altri, ignorando i NaN
    merged_df['Others_Sum'] = merged_df[other_cols].sum(axis=1, skipna=True)

    # Differenza: primo - somma degli altri
    merged_df['Difference'] = merged_df[first_col] - merged_df['Others_Sum']

    # Dove tutti gli altri sono NaN → forziamo NaN nella differenza
    merged_df.loc[all_nan_others, 'Difference'] = float('nan')

    # Se la differenza è negativa → forziamo NaN
    merged_df.loc[merged_df['Difference'] < 0, 'Difference'] = float('nan')

    # Salva solo timestamp e Difference
    result_df = merged_df[['timestamp', 'Difference']]
    result_df.to_csv(output_file, index=False)
    print(f"CSV di differenza salvato in: {output_file}")

if __name__ == "__main__":
    input_dir = os.path.join(PROJECT_ROOT, "raw_data")
    file_list = ["Cabina X.csv", "Lab Avio.csv", "Chillers.csv", "Aule R.csv", "I3P.csv"]
    output_file = os.path.join(PROJECT_ROOT, "raw_data", "Unlabelled R.csv")

    # aggregate_selected_csv_sum(file_list, input_dir, output_file)
    compute_difference_from_first(file_list, input_dir, output_file)