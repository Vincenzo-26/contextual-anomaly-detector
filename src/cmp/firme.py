from src.cmp.utils_hard_rules import *
from src.cmp.utils import *
import plotly.graph_objects as go
import datetime

"""
INPUT:
-   data_el_aule_R_pre.csv da "main.py"
-   time_windows.csv da "main.py"
-   cluster_data.csv da "main.py"

-   anomalies_var_table_overall.csv da "main_var.py"

OUTPUT:
-   r2_df_firma.csv
"""

df_el = pd.read_csv("data/Aule_R/preprocess_data/electric_data/data_el_aule_R_pre.csv")
df_tw = pd.read_csv("data/time_windows.csv") # dataframe con le tw
df_cluster = pd.read_csv("data/cluster_data.csv") # dataframe con i cluster associati ai timestep
anm_table_var = pd.read_csv(f'data/diagnosis/Anomalie_tables/CMP/anomalies_var_table_overall.csv') # dataframe con le anomalie delle variabili interne

# ---------------------------------
#               Firme
# ---------------------------------
df_el['timestamp'] = pd.to_datetime(df_el['timestamp'])
df_el.set_index('timestamp', inplace=True)
t_ext_list = df_el['temp'].copy()
df_el = df_el.drop(columns=['temp'])
df_cluster['timestamp'] = pd.to_datetime(df_cluster['timestamp'])
anm_table_var['date'] = pd.to_datetime(anm_table_var['date'])

firma_by_var = {}
r2_dict = {var: {} for var in df_el.columns}
firma_comment = ("È stata creata una firma per ogni cluster di ogni variabile. Questo serve per associare oppure no, "
                 "il nodo relativo alla temperatura esterna al nodo del sottosistema nella rete bayesiana.<br>"
                 "La divisione è stata fatta per ogni cluster poichè tramite il clustering già vengono divise le varie funzioni "
                 "operative (es. cooling, heating ecc.). La gestione dei context invece è stata la seguente: non sono "
                 "stati analizzati separatamente (in quanto si suppone che nell'orario lavorativo se un context è "
                 "thermal sensitive lo sia anche il successivo) ma sono stati "
                 "considerati solo i dati nell'intervallo [7:30 - 19:30) (orario lavorativo), tranne "
                 "per i cluster 1 e 2 (domeniche e sabati) dove non è stato applicato questo filtro.<br>"
                 "Inoltre i punti sono relativi solo alle condizioni 'normal': in presenza di un contesto anomalo i dati "
                 "sono filtrati dai valori relativi a quella sottosequenza, per non falsare la dipendenza.<br>"
                 "Infine i dati sono raggruppati all'ora per ragioni di visualizzazione e di caricamento del file html.")
for var in df_el.columns:
    firma_by_var[var] = {}
    # Filtra solo i set (date, context, cluster) in cui var è anomala
    set_anomali = anm_table_var[anm_table_var[var] == 1][['date', 'Context', 'Cluster']].drop_duplicates()
    cluster_ids = df_cluster['cluster'].str.extract(r'Cluster_(\d+)')[0].dropna().astype(int).unique()
    for cluster_id in sorted(cluster_ids):
        cluster_name = f"Cluster_{cluster_id}"
        cluster_data = df_cluster[df_cluster['cluster'] == cluster_name].copy()
        cluster_data['time'] = cluster_data['timestamp'].dt.time
        # Filtro orario 7:00 - 19:30 per cluster diversi da 1 e 2
        if cluster_id not in [1, 2]:
            cluster_data = cluster_data[
                (cluster_data['time'] >= datetime.time(7, 30)) &
                (cluster_data['time'] < datetime.time(19, 30))
            ]
        timestamp_filtrati = cluster_data.copy()
        timestamp_filtrati['date'] = timestamp_filtrati['timestamp'].dt.date
        for _, row in set_anomali[set_anomali['Cluster'] == cluster_id].iterrows():
            ctx = int(row['Context'])
            data = row['date'].date()
            from_str = df_tw.iloc[ctx - 1]['from']
            to_str = df_tw.iloc[ctx - 1]['to']
            from_time = datetime.datetime.strptime(from_str, "%H:%M").time()
            if to_str == "24:00":
                to_time = datetime.time(23, 59, 59)
            else:
                to_time = datetime.datetime.strptime(to_str, "%H:%M").time()
            mask = ~((timestamp_filtrati['date'] == data) &
                     (timestamp_filtrati['time'] >= from_time) &
                     (timestamp_filtrati['time'] < to_time))
            timestamp_filtrati = timestamp_filtrati[mask]
        if timestamp_filtrati.empty:
            continue
        timestamps = timestamp_filtrati['timestamp']
        # Filtro i dati originali su quei timestamp e raggruppo per ora
        df_temp = df_el[df_el.index.isin(timestamps)].copy()
        df_temp['t_ext'] = t_ext_list[df_temp.index]
        df_temp = df_temp.resample('h').mean().dropna(subset=[var, 't_ext'])

        x = pd.to_numeric(df_temp['t_ext'], errors='coerce').values.reshape(-1, 1)
        y = pd.to_numeric(df_temp[var], errors='coerce').values

        mask = ~np.isnan(x).flatten() & ~np.isnan(y)
        if sum(mask) < 2:
            continue

        # Regressione lineare
        reg = LinearRegression().fit(x[mask], y[mask])
        y_pred = reg.predict(x[mask])
        r2 = r2_score(y[mask], y_pred)

        r2_dict[var][cluster_id] = round(r2, 4)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x[mask].flatten(), y=y[mask], mode='markers'))
        fig.add_trace(go.Scatter(x=x[mask].flatten(), y=y_pred, mode='lines'))
        fig.update_layout(
            title=f"{var} - {cluster_name} (R²={r2:.2f})",
            xaxis_title="Temperatura esterna (°C)",
            yaxis_title=f"{var} (kW)",
            title_x=0.5,
            template="plotly_white",
            legend=None
        )
        firma_by_var[var][cluster_name] = fig.to_html(full_html=False, include_plotlyjs='cdn')
r2_df_firma = pd.DataFrame(r2_dict).reset_index().rename(columns={'index': 'cluster'})
r2_df_firma.to_csv(f"data/diagnosis/r2_df_firma.csv", index=False)
r2_html = r2_df_firma.to_html(index=False)