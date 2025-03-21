import webbrowser

import pandas as pd

from src.cmp.utils_hard_rules import *
from src.cmp.utils import *
import plotly.graph_objects as go
import datetime
# dataframe con i carici elettrici

# df con la timeserie del carico elettrico analizzato
data_el_prep = pd.read_csv(f'data/Aule_R/preprocess_data/electric_data/el_data_prep.csv')
data_el_prep['timestamp'] = pd.to_datetime(data_el_prep['timestamp'])
data_el_prep['date'] = data_el_prep['timestamp'].dt.date
data_el_prep['time'] = data_el_prep['timestamp'].dt.time

df_tw = pd.read_csv("data/diagnosis/time_windows.csv") # dataframe con le tw
df_ctx = pd.read_csv("data/contexts.csv") # dataframe con i contesti
df_cluster = pd.read_csv("data/diagnosis/cluster_data.csv") # dataframe con i cluster associati ai timestep

anm_table_var = pd.read_csv(f'data/diagnosis/anomalies_table_var/anomalies_var_table_overall.csv') # dataframe con le anomalie delle variabili interne
anm_table_el = pd.read_csv(f'data/diagnosis/anomalies_table_overall.csv') # dataframe con le anomalie elettiche

set_el = set(anm_table_el[['Date', 'Context', 'Cluster']].apply(tuple, axis=1))

# dataframe dove vengono evidenziate le date anomale contemporaneamente del carico elettrico e di almeno una variabile per
# datra, contesto e cluster
anm_table_var['anm_el'] = anm_table_var[['date', 'Context', 'Cluster']].apply(
    lambda row: tuple(row) in set_el, axis=1
)

anm_el_and_var = anm_table_var[anm_table_var['anm_el']]
anm_el_and_var.to_csv(f"data/diagnosis/anomalies_table_var/anomalies_el_&_var_table_overall.csv", index=False)


report_content = {
    'plots': {
        'plot_el': {},
        'plot_var': {}
    },
}

for index, row in anm_el_and_var.iterrows():
    context = row['Context']
    cluster = row['Cluster']

    df_cluster_filtr = df_cluster[
        (df_cluster['cluster'] == f"Cluster_{cluster}")
    ]

    pivot_el = df_cluster_filtr.pivot_table(
        index='date',
        columns='time',
        values='value'
    )
    pivot_el.columns = pd.to_datetime(pivot_el.columns, format="%H:%M:%S")
    data_anomala = row['date']

    from_tw = df_tw.iloc[context - 1]["from"]
    from_tw = datetime.datetime.strptime(from_tw, "%H:%M")

    if context == (len(df_tw)):
        to_tw = datetime.datetime.strptime("23:45", "%H:%M").time()
    else:
        to_tw = datetime.datetime.strptime(df_tw.iloc[context - 1]["to"], "%H:%M").time()


    from_ctx =df_ctx.iloc[context - 1]["from"]
    from_ctx = datetime.datetime.strptime(from_ctx, "%H:%M")

    to_ctx =df_ctx.iloc[context - 1]["to"]
    to_ctx = datetime.datetime.strptime(to_ctx, "%H:%M") - datetime.timedelta(minutes=15) # perchè l'ultimo non è compreso

    fig_el = go.Figure()
    for date in pivot_el.index:
        if date != data_anomala:
            fig_el.add_trace(go.Scatter(
                x=pivot_el.columns,
                y=pivot_el.loc[date],
                mode='lines',
                name=str(date),
                line=dict(color='grey'),
                hovertemplate=f"{date}<br>%{{x|%H:%M}}<br>%{{y}}kW<extra></extra>"
            ))
    for date in pivot_el.index:
        if date == data_anomala:
            fig_el.add_trace(go.Scatter(
                x=pivot_el.columns,
                y=pivot_el.loc[date],
                mode='lines',
                name=str(date),
                line=dict(color='red'),
                hovertemplate=f"{date}<br>%{{x|%H:%M}}<br>%{{y}}kW<extra></extra>"
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
        title=f'Electric anomaly for context {context} and cluster {cluster}',
        title_x=0.5,
        xaxis_title='Time',
        yaxis_title='Electric Power [kW]'
    )
    key_el = f"fig_el_{index}"
    report_content['plots']['plot_el'][key_el] = {
        'title': f"Context {context} Cluster {cluster}",
        'fig': fig_el.to_html(full_html=False, include_plotlyjs='cdn')
    }
    report_content['plots']['plot_var'][key_el] = {}

    for col in range(3, len(anm_el_and_var.columns) - 1): # il -1 perchè l'ultima colonne è di vero o falso
        if row.iloc[col] == 1:
            var = anm_el_and_var.columns[col]

            df_var = pd.read_csv("data/Aule_R/preprocess_data/electric_data/data_el_aule_R_pre.csv")
            df_var['timestamp'] = pd.to_datetime(df_var['timestamp'])
            df_var['date'] = df_var['timestamp'].dt.date
            df_var['time'] = df_var['timestamp'].dt.time

            df_var["date"] = pd.to_datetime(df_var["date"]).dt.date
            df_cluster_filtr["date"] = pd.to_datetime(df_cluster_filtr["date"]).dt.date

            df_cluster_var_filtr = df_var[df_var["date"].isin(df_cluster_filtr["date"])]

            pivot_var = df_cluster_var_filtr.pivot_table(
                index='date',
                columns='time',
                values=var
            )
            pivot_var.columns = pd.to_datetime(pivot_el.columns, format="%H:%M:%S")

            fig_var = go.Figure()
            for date in pivot_var.index:
                if date != data_anomala:
                    fig_var.add_trace(go.Scatter(
                        x=pivot_var.columns,
                        y=pivot_var.loc[date],
                        mode='lines',
                        name=str(date),
                        line=dict(color='grey'),
                        hovertemplate=f"{date}<br>%{{x|%H:%M}}<br>%{{y}}kW<extra></extra>"
                    ))
            for date in pivot_var.index:
                if date == data_anomala:
                    fig_var.add_trace(go.Scatter(
                        x=pivot_var.columns,
                        y=pivot_var.loc[date],
                        mode='lines',
                        name=str(date),
                        line=dict(color='red'),
                        hovertemplate=f"{date}<br>%{{x|%H:%M}}<br>%{{y}}kW<extra></extra>"
                    ))
            fig_var.add_vrect(
                x0=from_ctx,
                x1=to_ctx,
                fillcolor="darkred",
                opacity=0.5,
                layer="below",
                line_width=0
            )
            fig_var.add_vrect(
                x0=from_tw,
                x1=to_tw,
                fillcolor="lightcoral",
                opacity=0.5,
                layer="below",
                line_width=0
            )
            fig_var.update_layout(
                title=f'{var} Context {context} Cluster {cluster}',
                title_x=0.5,
                xaxis_title='Time',
                yaxis_title=var
            )
            key_var = var
            report_content['plots']['plot_var'][key_el][key_var] = fig_var.to_html(full_html=False,
                                                                                   include_plotlyjs='cdn')

        else:
            continue

output_file = f"results/reports/report_var.html"
save_report(report_content, output_file, "template_var.html")
absolute_path = os.path.abspath(output_file)
webbrowser.open_new_tab(f'file://{absolute_path}')

# # tabella delle probabilità
# prob_fault = 0.9
# prob_no_fault = 0.1
#
# anm_el_and_var = anm_el_and_var.drop(columns=["anm_el"])
#
# anm_el_and_var = anm_el_and_var.rename(columns={"date": "Date"})
# var_columns = list(anm_el_and_var.columns)[3:]
#
# probability_table = anm_table_el[["Date", "Context", "Cluster"]].copy()
# merged_table = pd.merge(probability_table, anm_el_and_var, on=["Date", "Context", "Cluster"], how="left")
#
# #    Se per la combinazione (Date, Context, Cluster) non è presente (NaN) =>  prob_no_fault (0.1)
# #    - Se il valore è 1 =>  prob_fault (0.9)
# for col in var_columns:
#     probability_table[col] = merged_table[col].apply(lambda x: prob_fault if x == 1 else prob_no_fault)
# probability_table.to_csv("data/diagnosis/anomalies_table_var/probability_table.csv", index=False)




