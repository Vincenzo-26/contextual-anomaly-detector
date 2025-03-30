import webbrowser
import pandas as pd
from src.cmp.utils_hard_rules import *
from src.cmp.utils import *
import plotly.graph_objects as go
import datetime
from scipy.stats import norm
from scipy.stats import zscore
from collections import defaultdict

# read dataframe
df_el = pd.read_csv("data/Aule_R/preprocess_data/electric_data/data_el_aule_R_pre.csv")
df_tw = pd.read_csv("data/diagnosis/time_windows.csv") # dataframe con le tw
df_ctx = pd.read_csv("data/contexts.csv") # dataframe con i contesti
df_cluster = pd.read_csv("data/diagnosis/cluster_data.csv") # dataframe con i cluster associati ai timestep
anm_table_var = pd.read_csv(f'data/diagnosis/anomalies_table_var/anomalies_var_table_overall.csv') # dataframe con le anomalie delle variabili interne
anm_table_el = pd.read_csv(f'data/diagnosis/anomalies_table_overall.csv') # dataframe con le anomalie elettiche
evidence_var_full = pd.read_csv(f'data/diagnosis/anomalies_table_var/evidence_var_full.csv')

set_el = set(anm_table_el[['Date', 'Context', 'Cluster']].apply(tuple, axis=1))
anm_el_and_var = anm_table_var[anm_table_var[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_el)].reset_index(drop=True)
anm_el_and_var.to_csv(f"data/diagnosis/anomalies_table_var/anomalies_el_&_var_table_overall.csv", index=False)

evidence_el_and_var_full = evidence_var_full[evidence_var_full[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_el)].reset_index(drop=True)
evidence_el_and_var_full.to_csv(f"data/diagnosis/anomalies_table_var/evidence_el_&_var_full.csv", index=False)

set_var = set(anm_table_var[['date', 'Context', 'Cluster']].apply(tuple, axis=1))
anm_table_el['anm_var'] = anm_table_el[['Date', 'Context', 'Cluster']].apply(lambda row: tuple(row) in set_var, axis=1)

num_anm_el_and_var = len(anm_el_and_var)
num_anm_tot = len(anm_table_el)
anm_string = (f"{num_anm_el_and_var} anomalie interne su {num_anm_tot} anomalie a livello superiore "
              f"({round(num_anm_el_and_var/num_anm_tot*100, 1)}%)")

# Analisi z-score t-ext
keys_anom = [tuple(x) for x in anm_el_and_var[['date', 'Context', 'Cluster']].values.tolist()]
keys_all = [tuple(x) for x in evidence_var_full[['date', 'Context', 'Cluster']].values.tolist()]

filtered_idx = [i for i, k in enumerate(keys_all) if k not in keys_anom]

t_ext_scores_filtered = evidence_var_full.iloc[filtered_idx]['t_ext_score'].tolist()
mean_filtered = np.mean(t_ext_scores_filtered)
std_filtered = np.std(t_ext_scores_filtered, ddof=0)  # ddof=0 per compatibilità con .std()

# Calcolo z-score per tutti i punti
z_scores_all = ((evidence_var_full['t_ext_score'] - mean_filtered) / std_filtered).tolist()

# Calcolo z-score per i punti anomali (usando gli indici corrispondenti)
z_scores_anomalies = [z_scores_all[i] for i, k in enumerate(keys_all) if k in keys_anom]

pdf_max = norm.pdf(0)  # Valore massimo della PDF della normale standard
prob_normalizzate_all = [norm.pdf(z) / pdf_max for z in z_scores_all]

hover_texts = [
    f"{row['date']} - Ctx {row['Context']} - Cl {row['Cluster']}<br>t_ext_score: {row['t_ext_score']:.2f}<br>Zscore: {z_scores_all[i]:.2f}"
    for i, (idx, row) in enumerate(evidence_var_full.iterrows())
    if keys_all[i] in keys_anom
]

fig_zscore = go.Figure()
fig_zscore.add_trace(go.Histogram(
    x=[z_scores_all[i] for i in filtered_idx],
    nbinsx=50,
    name='t_ext_score set normali<br>livello alto&basso',
    marker=dict(color='lightblue', line=dict(width=0.5, color='grey')),
    opacity=0.7
))
fig_zscore.add_trace(go.Scatter(
    x=z_scores_anomalies,
    y=[0.5] * len(z_scores_anomalies),
    mode='markers',
    marker=dict(color='red', size=10, symbol='x'),
    name='t_ext_score set anomali<br>livello alto&basso',
    text=hover_texts,
    hovertemplate="%{text}<extra></extra>"
))

fig_zscore.update_layout(
    title="z-score dello t_ext_score sui set data-context-cluster",
    title_x= 0.5,
    xaxis_title="Z-score",
    yaxis_title="Frequenza",
    barmode='overlay',
    template='plotly_white',
    shapes=[
        dict(type="rect", x0=-1, x1=1, y0=0, y1=1, xref='x', yref='paper', fillcolor="lightgreen", opacity=0.3, layer="below", line_width=0),
        dict(type="rect", x0=-2, x1=-1, y0=0, y1=1, xref='x', yref='paper', fillcolor="yellow", opacity=0.3, layer="below", line_width=0),
        dict(type="rect", x0=1, x1=2, y0=0, y1=1, xref='x', yref='paper', fillcolor="yellow", opacity=0.3, layer="below", line_width=0),
        dict(type="rect", x0=-3, x1=-2, y0=0, y1=1, xref='x', yref='paper', fillcolor="red", opacity=0.3, layer="below", line_width=0),
        dict(type="rect", x0=2, x1=3, y0=0, y1=1, xref='x', yref='paper', fillcolor="red", opacity=0.3, layer="below", line_width=0)
    ]
)
evidence_var_full['t_ext_score'] = 1 - np.array(prob_normalizzate_all)
evidence_var_full.to_csv(f"data/diagnosis/anomalies_table_var/evidence%_var_full.csv", index=False)
evidence_el_and_var_full = evidence_var_full[evidence_var_full[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_el)].reset_index(drop=True)
evidence_el_and_var_full.to_csv(f"data/diagnosis/anomalies_table_var/evidence%_el_&_var_full.csv", index=False)

# Firme
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
r2_df_firma.to_csv(f"data/diagnosis/anomalies_table_var/r2_df_firma.csv", index=False)
r2_html = r2_df_firma.to_html(index=False)

# Plot delle anomalie elettriche e relativi plot delle anomalie nel sottolivello
context_table_data = []
for i, row in df_tw.iterrows():
    context_id = row['id']
    # Filtra righe del context corrente
    context_rows = anm_table_el[anm_table_el['Context'] == context_id]
    total = len(context_rows)
    with_var = context_rows['anm_var'].sum()  # True conta come 1
    percentage = round((with_var / total * 100), 1) if total > 0 else 0.0

    context_table_data.append({
        'context_name': f"Context {context_id}",
        'num_anomalies': f"{int(with_var)}/{int(total)}",
        'percentage': percentage
    })

report_content = {
    'subtitle': anm_string,
    'context_table': context_table_data,
    'plot_zscore': fig_zscore.to_html(full_html=False, include_plotlyjs='cdn'),
    'plots': {
        'plot_el': {},
        'plot_var': {},
    },
}
report_content['plot_firme_cluster'] = firma_by_var
report_content['firma_comment'] = firma_comment
report_content['r2_df_firma'] = r2_df_firma

# plot anomalie elettriche e delle variabili
grouped = anm_el_and_var.sort_values(by=['Context', 'Cluster']).groupby(['Context', 'Cluster'])
for (context, cluster), group in grouped:
    data_anomala_list = pd.to_datetime(group['date']).tolist()
    first_date = data_anomala_list[0]  # per l'anomaly score
    data_anomala_str = first_date.strftime('%Y-%m-%d')

    df_cluster_filtr = df_cluster[df_cluster['cluster'] == f"Cluster_{cluster}"].copy()

    pivot_el = df_cluster_filtr.pivot_table(
        index='date',
        columns='time',
        values='value'
    )
    pivot_el.columns = pd.to_datetime(pivot_el.columns, format="%H:%M:%S")
    pivot_el.index = pd.to_datetime(pivot_el.index)

    match_row = anm_table_el[
        (anm_table_el['Date'] == data_anomala_str) &
        (anm_table_el['Context'] == context) &
        (anm_table_el['Cluster'] == cluster)
    ]

    from_tw = datetime.datetime.strptime(df_tw.iloc[context - 1]["from"], "%H:%M")
    to_tw = datetime.datetime.strptime("23:45", "%H:%M") if context == len(df_tw) else datetime.datetime.strptime(df_tw.iloc[context - 1]["to"], "%H:%M")

    from_ctx = datetime.datetime.strptime(df_ctx.iloc[context - 1]["from"], "%H:%M")
    to_ctx = datetime.datetime.strptime(df_ctx.iloc[context - 1]["to"], "%H:%M") - datetime.timedelta(minutes=15)

    fig_el = go.Figure()
    for date in pivot_el.index:
        if date not in data_anomala_list:
            fig_el.add_trace(go.Scatter(
                x=pivot_el.columns,
                y=pivot_el.loc[date],
                mode='lines',
                name=str(date.date()),
                line=dict(color='grey'),
                hovertemplate=f"{date.date()}<br>%{{x|%H:%M}}<br>%{{y}}kW<extra></extra>"
            ))
    for date in pivot_el.index:
        if date in data_anomala_list:
            fig_el.add_trace(go.Scatter(
                x=pivot_el.columns,
                y=pivot_el.loc[date],
                mode='lines',
                name=str(date.date()),
                line=dict(color='red'),
                hovertemplate=f"{date.date()}<br>%{{x|%H:%M}}<br>%{{y}}kW<extra></extra>"
            ))

    fig_el.add_vrect(x0=from_ctx, x1=to_ctx, fillcolor="darkred", opacity=0.5, layer="below", line_width=0)
    fig_el.add_vrect(x0=from_tw, x1=to_tw, fillcolor="lightcoral", opacity=0.5, layer="below", line_width=0)

    fig_el.update_layout(
        title=f'Electric anomaly',
        title_x=0.5,
        xaxis_title='Time',
        yaxis_title='Electric Power [kW]'
    )

    key_el = f"fig_el_C{context}_K{cluster}"
    report_content['plots']['plot_el'][key_el] = {
        'title': f"Context {context} Cluster {cluster}",
        'fig': fig_el.to_html(full_html=False, include_plotlyjs='cdn'),
        'date': str(data_anomala_str)  # aggiungiamo anche la data!
    }
    report_content['plots']['plot_var'][key_el] = {}

    var_columns = anm_el_and_var.columns[3:-1]  # Salta date, Context, Cluster, anm_el

    for var in var_columns:
        group_var = group[group[var] == 1]
        if group_var.empty:
            continue

        data_anomala_var_list = pd.to_datetime(group_var['date']).tolist()

        df_var = pd.read_csv("data/Aule_R/preprocess_data/electric_data/data_el_aule_R_pre.csv")
        df_var['timestamp'] = pd.to_datetime(df_var['timestamp'])
        df_var['date'] = df_var['timestamp'].dt.date
        df_var['time'] = df_var['timestamp'].dt.time
        df_var['date'] = pd.to_datetime(df_var['date'])

        df_cluster_filtr['date'] = pd.to_datetime(df_cluster_filtr['date'])
        df_cluster_var_filtr = df_var[df_var['date'].isin(df_cluster_filtr['date'])]

        pivot_var = df_cluster_var_filtr.pivot_table(
            index='date',
            columns='time',
            values=var
        )
        pivot_var.columns = pd.to_datetime(pivot_var.columns, format="%H:%M:%S")
        pivot_var.index = pd.to_datetime(pivot_var.index)

        fig_var = go.Figure()
        for date in pivot_var.index:
            if date not in data_anomala_var_list:
                fig_var.add_trace(go.Scatter(
                    x=pivot_var.columns,
                    y=pivot_var.loc[date],
                    mode='lines',
                    name=str(date.date()),
                    line=dict(color='grey'),
                    hovertemplate=f"{date.date()}<br>%{{x|%H:%M}}<br>%{{y}}<extra></extra>"
                ))
        for date in pivot_var.index:
            if date in data_anomala_var_list:
                fig_var.add_trace(go.Scatter(
                    x=pivot_var.columns,
                    y=pivot_var.loc[date],
                    mode='lines',
                    name=str(date.date()),
                    line=dict(color='red'),
                    hovertemplate=f"{date.date()}<br>%{{x|%H:%M}}<br>%{{y}}<extra></extra>"
                ))

        fig_var.add_vrect(x0=from_ctx, x1=to_ctx, fillcolor="darkred", opacity=0.5, layer="below", line_width=0)
        fig_var.add_vrect(x0=from_tw, x1=to_tw, fillcolor="lightcoral", opacity=0.5, layer="below", line_width=0)

        fig_var.update_layout(
            title=f'{var} Context {context} Cluster {cluster}',
            title_x=0.5,
            xaxis_title='Time',
            yaxis_title=var
        )

        # Salva il grafico della variabile
        report_content['plots']['plot_var'][key_el][var] = fig_var.to_html(full_html=False, include_plotlyjs='cdn')

# barplot inferenza
# Barplot inferenza
inference_results = pd.read_csv(f'data/diagnosis/anomalies_table_var/inference_results_prova.csv')
columns_to_plot = [col for col in inference_results.columns if col not in ['date', 'Context', 'Cluster']]

# Set per match con anm_el_and_var
set_anom = set(anm_el_and_var[['date', 'Context', 'Cluster']].apply(tuple, axis=1))

barplots_group_a = {}  # Per inserimento nel loop di fig_el
barplots_group_b = []  # Per visualizzazione dopo il plot z-score

# Analisi inferenza/anomalie
summary_counts = defaultdict(lambda: {'count': 0, 'match': 0})
variables = columns_to_plot.copy()

# Mappatura rapida anomalie
anm_dict = {
    (str(pd.to_datetime(row['date']).date()), row['Context'], row['Cluster']): row
    for _, row in anm_el_and_var.iterrows()
}

for _, row in inference_results.iterrows():
    date = pd.to_datetime(row['date']).date()
    context = int(row['Context'])
    cluster = int(row['Cluster'])
    key = (str(date), context, cluster)

    # Costruzione barplot
    probs = [row[var] for var in columns_to_plot]
    max_val = max(probs) if probs else 1
    y_max = max_val * 1.1  # +10% margine

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=columns_to_plot,
        y=probs,
        marker_color='lightblue',
        hovertemplate="<b>%{x}</b><br>Prob: %{y:.2f}%<extra></extra>"
    ))
    bar_fig.update_layout(
        title=f"{date} - ctx {context} cls {cluster}",
        title_x=0.5,
        yaxis_range=[0, y_max],
        yaxis_title="Probabilità (%)",
        xaxis_title="Sottosistemi",
        template="plotly_white"
    )

    barplot_html = bar_fig.to_html(full_html=False, include_plotlyjs='cdn')
    plot_key = f"fig_el_C{context}_K{cluster}"

    if key in set_anom:
        if plot_key not in barplots_group_a:
            barplots_group_a[plot_key] = []
        barplots_group_a[plot_key].append(barplot_html)

        # Analisi incrociata con anomalie
        anm_row = anm_dict.get(key)
        if anm_row is not None:
            var_anomale = [var for var in variables if anm_row.get(var, 0) == 1]
            n_anomale = len(var_anomale)
            if n_anomale > 0:
                prob_dict = {var: row[var] for var in variables}
                top_n_vars = sorted(prob_dict, key=prob_dict.get, reverse=True)[:n_anomale]
                match_count = sum(1 for var in var_anomale if var in top_n_vars)
                summary_counts[n_anomale]['count'] += 1
                summary_counts[n_anomale]['match'] += match_count
    else:
        barplots_group_b.append(barplot_html)

# Inserimento nel report
report_content["plot_inference_group_a"] = barplots_group_a
report_content["plot_inference_group_b"] = barplots_group_b

# Generazione elenco puntato HTML
html_lines = ["<ul>"]
for n in sorted(summary_counts):
    count = summary_counts[n]['count']
    match = summary_counts[n]['match']
    html_lines.append(
        f"<li>{count}/{num_anm_el_and_var} casi con {n} variabili anomala: in {match}/{count * n} casi la variabile anomala è "
        f"risultata tra le top-{n} per probabilità.</li>"
    )
html_lines.append("</ul>")
report_content["analisi_inferenza_summary"] = "\n".join(html_lines)


output_file = f"results/reports/report_var.html"
save_report(report_content, output_file, "template_var.html")
absolute_path = os.path.abspath(output_file)
webbrowser.open_new_tab(f'file://{absolute_path}')







