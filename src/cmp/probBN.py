import webbrowser
from src.cmp.utils_hard_rules import *
from src.cmp.utils import *
import plotly.graph_objects as go
import datetime
# dataframe con i carici elettrici

var = "el_UTA_3_3B_7"

data_el_prep = pd.read_csv(f'data/Aule_R/preprocess_data/electric_data/el_data_prep.csv')
data_el_prep['timestamp'] = pd.to_datetime(data_el_prep['timestamp'])
data_el_prep['date'] = data_el_prep['timestamp'].dt.date
data_el_prep['time'] = data_el_prep['timestamp'].dt.time

df_tw = pd.read_csv("data/diagnosis/time_windows.csv") # dataframe con le tw
df_ctx = pd.read_csv("data/contexts.csv") # dataframe con i contesti
df_cluster = pd.read_csv("data/diagnosis/cluster_data.csv") # dataframe con i cluster associati ai timestep

anm_table_var = pd.read_csv(f'data/diagnosis/anomalies_table_var/final_anomalies_{var}.csv') # dataframe con le anomalie delle variabili interne
anm_table_el = pd.read_csv(f'data/diagnosis/anomalies_table_overall.csv') # dataframe con le anomalie elettiche
date_anomale_el = set(anm_table_el['Date']) # date anomale elettriche identificate da CMP
anm_table_var['anm_el'] = anm_table_var['date'].isin(date_anomale_el)
anm_table_var.to_csv("data/diagnosis/anomalies_table_var/anm_table_var.csv", index=False) # dataframe dove vengono evidenziate le date anomale contemporaneamente del carico elettrico e di almeno una variabile
anm_el_and_var = anm_table_var[anm_table_var['anm_el']]
anm_el_and_var.to_csv("data/diagnosis/anomalies_table_var/anm_el_and_var.csv", index=False)


report_content = {
    'title' : var,
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

    to_tw = df_tw.iloc[context - 1]["to"]
    to_tw = datetime.datetime.strptime(to_tw, "%H:%M")

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
        if anm_el_and_var.iloc[0, col] == 1:
            var = anm_el_and_var.columns[col]
            aula = var.split("_")[2]

            df_var = pd.read_csv(f"data/Aule_R/preprocess_data/var_int_data/aula_R{aula}.csv")

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

output_file = f"results/reports/report_anm_var.html"
save_report(report_content, output_file, "template_var.html")
absolute_path = os.path.abspath(output_file)
webbrowser.open_new_tab(f'file://{absolute_path}')





