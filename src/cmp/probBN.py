import webbrowser
import pandas as pd
from src.cmp.utils_hard_rules import *
from src.cmp.utils import *
import plotly.graph_objects as go
import datetime
from scipy.stats import zscore

# read dataframe
data_el_prep = pd.read_csv(f'data/Aule_R/preprocess_data/electric_data/el_data_prep.csv') # df con la timeserie del carico elettrico analizzato
data_el_prep = extract_date_time(data_el_prep)
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

evidence_var_full.set_index(['date', 'Context', 'Cluster', 't_ext_score'], inplace=True)
evidence_var_full = evidence_var_full.map(lambda x: x / 8)
evidence_var_full.reset_index(inplace=True)
evidence_var_full.to_csv(f"data/diagnosis/anomalies_table_var/evidence_var_full.csv", index=False)

# Analisi z-score
anm_el_and_var['tuple_key'] = list(anm_el_and_var[['date', 'Context', 'Cluster']].apply(tuple, axis=1))
evidence_var_full['tuple_key'] = list(evidence_var_full[['date', 'Context', 'Cluster']].apply(tuple, axis=1))

filtered = evidence_var_full[~evidence_var_full['tuple_key'].isin(anm_el_and_var['tuple_key'])].copy()
mean_filtered = filtered['t_ext_score'].mean()
std_filtered = filtered['t_ext_score'].std()
evidence_var_full['z_score'] = (evidence_var_full['t_ext_score'] - mean_filtered) / std_filtered
filtered['z_score'] = (filtered['t_ext_score'] - mean_filtered) / std_filtered

marker_red = evidence_var_full[evidence_var_full['tuple_key'].isin(anm_el_and_var['tuple_key'])]
marker_red_z = marker_red['z_score'].tolist()

marker_style_red = dict(color='red', size=10, symbol='x')
marker_style_hist = dict(color='lightblue', line=dict(width=0.5, color='grey'))

fig_zscore = go.Figure()
fig_zscore.add_trace(go.Histogram(
    x=filtered['z_score'],
    nbinsx=50,
    name='t_ext_score set normali<br>livello alto&basso',
    marker=marker_style_hist,
    opacity=0.7
))
fig_zscore.add_trace(go.Scatter(
    x=marker_red_z,
    y=[0.5] * len(marker_red_z),
    mode='markers',
    marker=marker_style_red,
    name='t_ext_score set anomali<br>livello alto&basso',
    text=[f"{row['date']} - Ctx {row['Context']} - Cl {row['Cluster']}<br>t_ext_score: {row['t_ext_score']:.2f}<br>Zscore: {row['z_score']:.2f}" for _, row in marker_red.iterrows()],
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
        'fig': fig_el.to_html(full_html=False, include_plotlyjs='cdn')
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


output_file = f"results/reports/report_var.html"
save_report(report_content, output_file, "template_var.html")
absolute_path = os.path.abspath(output_file)
webbrowser.open_new_tab(f'file://{absolute_path}')







