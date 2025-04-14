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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

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

def extract_date_time(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        other_cols = [col for col in df.columns if col not in ['timestamp', 'date', 'time']]
        df = df[['timestamp', 'date', 'time'] + other_cols]
    else:
        df.index = pd.to_datetime(df.index)
        df['date'] = df.index.date
        df['time'] = df.index.time
        other_cols = [col for col in df.columns if col not in ['date', 'time']]
        df = df[['date', 'time'] + other_cols]
    return df
