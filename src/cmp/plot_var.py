import webbrowser
import pandas as pd
from src.cmp.utils_hard_rules import *
from src.cmp.utils import *
import plotly.graph_objects as go
import datetime
from functools import reduce
import os

anm_table_el = pd.read_csv(f'data/diagnosis/anomalies_table_overall.csv')  # dataframe con le anomalie elettriche
anm_table_el.rename(columns={"Date": "date"}, inplace=True)
anomalous_days = set(anm_table_el[['date', 'Cluster', 'Context']].itertuples(index=False, name=None))
anomalous_days = {(pd.to_datetime(d, format="%Y-%m-%d").date(), cl, ct) for (d, cl, ct) in anomalous_days}

anm_table_var = pd.read_csv(f'data/diagnosis/anomalies_table_var/anomalies_var_table_overall.csv')

# dataframe con la time serie elettrica preprocessata
data = pd.read_csv(f'data/Aule_R/preprocess_data/electric_data/el_data_prep.csv')

df_tw = pd.read_csv("data/diagnosis/time_windows.csv")  # dataframe con le tw
df_ctx = pd.read_csv("data/contexts.csv")  # dataframe con i contesti
df_cluster = pd.read_csv("data/diagnosis/cluster_data.csv")  # dataframe con i cluster associati ai timestep

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['date'] = data['timestamp'].dt.date
data['time'] = data['timestamp'].dt.time

# Ristrutturiamo il report come lista di blocchi
report_content = {
    'plots': []  # ogni elemento sarà un blocco contenente fig_el e le fig_var associate
}

var_list = ['el_UTA_1_1B_5', 'el_UTA_2_2B_6', 'el_UTA_3_3B_7', 'el_UTA_4_4B_8']

var_dict = {}
for var in var_list:
    var_dict.update(preprocess_variabili_interne(var))

max_cluster = df_cluster['cluster'].nunique()
for context in range(1, len(df_tw) + 1):

    # per fill della sottosequenza
    from_tw = datetime.datetime.strptime(df_tw.iloc[context - 1]["from"], "%H:%M").time()
    if context == (len(df_tw)):
        to_tw = datetime.datetime.strptime("23:45", "%H:%M").time()
    else:
        to_tw = datetime.datetime.strptime(df_tw.iloc[context - 1]["to"], "%H:%M").time()

    # per fill del contesto
    from_ctx = datetime.datetime.strptime(df_ctx.iloc[context - 1]["from"], "%H:%M").time()
    to_ctx = (datetime.datetime.strptime(df_ctx.iloc[context - 1]["to"], "%H:%M") - datetime.timedelta(
        minutes=15)).time()

    for cluster in range(1, max_cluster + 1):
        df_cluster_filtr = df_cluster[
            (df_cluster['cluster'] == f"Cluster_{cluster}")
        ]

        pivot_el = df_cluster_filtr.pivot_table(
            index='date',
            columns='time',
            values='value'
        )

        fig_el = go.Figure()
        for date in pivot_el.index:
            if (date, cluster, context) not in set(
                    anm_table_el[['date', 'Cluster', 'Context']].itertuples(index=False, name=None)):
                fig_el.add_trace(go.Scatter(
                    x=pivot_el.columns,
                    y=pivot_el.loc[date],
                    mode='lines',
                    name=str(date),
                    line=dict(color='grey'),
                    hovertemplate=f"{date}<br>%{{x|%H:%M}}<br>%{{y}}kW<extra></extra>"
                ))
        for date in pivot_el.index:
            if (date, cluster, context) in set(
                    anm_table_el[['date', 'Cluster', 'Context']].itertuples(index=False, name=None)):
                fig_el.add_trace(go.Scatter(
                    x=pivot_el.columns,
                    y=pivot_el.loc[date],
                    mode='lines',
                    name=str(date),
                    line=dict(color='red'),
                    hovertemplate=f"{date}<br>%{{x}}<br>%{{y}}kW<extra></extra>"
                ))
        fig_el.add_vrect(
            x0=from_ctx,
            x1=to_ctx,
            fillcolor="darkred",
            opacity=0.5,
            layer="below",
            line_width=0
        )
        fig_el.add_vrect(
            x0=from_tw,
            x1=to_tw,
            fillcolor="lightcoral",
            opacity=0.5,
            layer="below",
            line_width=0
        )
        fig_el.update_layout(
            title=f'Context {context} Cluster {cluster}',
            title_x=0.5,
            xaxis_title='Time',
            yaxis_title='Electric Power [kW]'
        )

        # Creiamo un blocco con fig_el e una struttura per le fig_var
        block = {
            'title': f"Context {context} Cluster {cluster}",
            'fig_el': fig_el.to_html(full_html=False, include_plotlyjs='cdn'),
            'fig_vars': {}  # qui verranno aggiunte le figure delle variabili
        }

        date_uniche_cluster = df_cluster_filtr["date"].unique().tolist()
        date_uniche_cluster = [pd.to_datetime(d).date() for d in date_uniche_cluster]

        # Cicliamo sulle colonne (variabili) che non sono colonne di time stamp
        for key, df in var_dict.items():
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df_filtered = df[df["date"].isin(date_uniche_cluster)]
            for col in [col for col in df_filtered.columns if col not in {"timestamp", "date", "time"}]:
                pivot_var = df_filtered.pivot_table(
                    index='date',
                    columns='time',
                    values=col
                )
                fig_var = go.Figure()
                for date in pivot_var.index:
                    if (date, cluster, context) in anomalous_days:
                        color = "red"
                    else:
                        color = "gray"
                    fig_var.add_trace(go.Scatter(
                        x=pivot_var.columns,
                        y=pivot_var.loc[date],
                        mode='lines',
                        name=str(date),
                        line=dict(color=color),
                        hovertemplate=f"{date}<br>%{{x}}<br>%{{y}}<extra></extra>"
                    ))
                    # fig_var.add_vrect(
                    #     x0=from_ctx,
                    #     x1=to_ctx,
                    #     fillcolor="darkred",
                    #     opacity=0.5,
                    #     layer="below",
                    #     line_width=0
                    # )
                    # fig_var.add_vrect(
                    #     x0=from_tw,
                    #     x1=to_tw,
                    #     fillcolor="lightcoral",
                    #     opacity=0.5,
                    #     layer="below",
                    #     line_width=0
                    # )
                    # fig_var.update_layout(
                    #     title=f'{col} Context {context} Cluster {cluster}',
                    #     title_x=0.5,
                    #     xaxis_title='Time',
                    #     yaxis_title=f'{col}'
                    # )
                block['fig_vars'][col] = fig_var.to_html(full_html=False, include_plotlyjs=False)

        report_content['plots'].append(block)

output_file = f"results/reports/report_anm_el.html"
save_report(report_content, output_file, "template_anm_el.html")
absolute_path = os.path.abspath(output_file)
webbrowser.open_new_tab(f'file://{absolute_path}')
