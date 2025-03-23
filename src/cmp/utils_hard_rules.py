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
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df_temp.rename(columns={'Time': 'timestamp'}, inplace=True)
    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])

    df = pd.merge(df, df_temp[['timestamp', 'Temperatura Esterna']], on='timestamp', how='left')
    df.rename(columns={'Temperatura Esterna': 'temp'}, inplace=True)

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

    df.reset_index(inplace=True)
    df.to_csv("data/Aule_R/preprocess_data/electric_data/data_el_aule_R_pre.csv", index = False)
    return df

def preprocess_variabili_interne(var):
    """
    Fa un dizionario avente i dataframe delle variabili associate alle varie aule.
    Le chiavi sono df_{aula}
    """
    dfs_dict = {}
    aule = var.split("_") [2:]
    for aula in aule:
        file_name = f"data/Aule_R/raw_data/var_int_data_raw/raw_aula_R{aula}.csv"
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
        save_path = os.path.join(current_dir, "data", "Aule_R", "preprocess_data", "var_int_data", f"aula_R{aula}.csv")
        df.to_csv(save_path, index = False)
        dfs_dict[f"df_{aula}"] = df
    return dfs_dict

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

def merge_anomaly_tables(df_tot, df_var):
    key_cols = ["date", "Context", "Cluster"]
    if df_tot.empty:
        return df_var.copy()
    df_merged = pd.merge(df_tot, df_var, on=key_cols, how='outer')
    df_merged.fillna(0, inplace=True)

    return df_merged


def extract_date_time(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    other_cols = [col for col in df.columns if col not in ['timestamp', 'date', 'time']]
    df = df[['timestamp', 'date', 'time'] + other_cols]

    return df