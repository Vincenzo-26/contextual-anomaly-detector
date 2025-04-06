import webbrowser
import pandas as pd
from src.cmp.utils_hard_rules import *
from src.cmp.utils import *
import plotly.graph_objects as go
import datetime
from scipy.stats import norm
from scipy.stats import zscore
from collections import defaultdict

"""
INPUT:
-   data_el_aule_R_pre.csv da "main.py"
-   time_windows.csv da "main.py"
-   contexts.csv da "main.py"
-   cluster_data.csv da "main.py"
-   anomalies_table_overall.csv da "main.py"

-   t_ext_score_var_full.csv da "main_var.py"
-   anomalies_var_table_overall.csv da "main_var.py"
-   evidence_var_full.csv da "main_var.py"

-   inference_results_prova_distrib.csv da "bayesian_network.py"

-   r2_df_firma.csv da "firme.py"

OUTPUT:
-   report_var.html
"""

"""
Il report creato contiene
-   i plot delle sole anomalie ad alto livello che hanno contemporaneamente anomalia a livello più basso.
    i plot delle anomalie dei sottosistemi è relativo ai sottocarichi che risultano anomali da anomaly score di CMP
    i batplot sono relativi invece ai sottocarichi individuati anomali tramite rete bayesiana con evidenze da soglie,
    per analizzare le differenze tra i 2 metodi.
"""

# read dataframe
df_el = pd.read_csv("data/Aule_R/preprocess_data/electric_data/data_el_aule_R_pre.csv")
df_tw = pd.read_csv("data/time_windows.csv") # dataframe con le tw
df_ctx = pd.read_csv("data/contexts.csv") # dataframe con i contesti
df_cluster = pd.read_csv("data/cluster_data.csv") # dataframe con i cluster associati ai timestep
anm_table_var = pd.read_csv(f'data/diagnosis/Anomalie_tables/CMP/anomalies_var_table_overall.csv') # dataframe con le anomalie delle variabili interne
anm_table_el = pd.read_csv(f'data/anomalies_table_overall.csv') # dataframe con le anomalie elettiche
anm_el_and_var = pd.read_csv(f'data/diagnosis/Anomalie_tables/CMP/anomalies_el_&_var_table_overall.csv')
evidence_var_full = pd.read_csv(f'data/diagnosis/Evidence_tables/Power/CMP/evidence_var_full.csv')
t_ext_score_var_full = pd.read_csv(f'data/diagnosis/Evidence_tables/T_ext/t_ext_score_var_full.csv')
inference_results = pd.read_csv(f'data/diagnosis/Inference/CMP/inference_results.csv')
inference_results_distrib = pd.read_csv(f'data/diagnosis/Inference/Soglie/inference_results_distrib.csv')
r2_df_firma = pd.read_csv("data/diagnosis/r2_df_firma.csv")



num_anm_el_and_var = len(anm_el_and_var)
num_anm_tot = len(anm_table_el)
anm_string = (f"{num_anm_el_and_var} anomalie interne su {num_anm_tot} anomalie a livello superiore "
              f"({round(num_anm_el_and_var/num_anm_tot*100, 1)}%)")

# ----------------------------------------------
#            Plot distribuzione z-score t_ext
# -----------------------------------------------
keys_anom = [tuple(x) for x in anm_el_and_var[['date', 'Context', 'Cluster']].values.tolist()]
keys_all = [tuple(x) for x in evidence_var_full[['date', 'Context', 'Cluster']].values.tolist()]

filtered_idx = [i for i, k in enumerate(keys_all) if k not in keys_anom]

z_scores_all = t_ext_score_var_full['t_ext_Z_score']
z_scores_anomalies = [z_scores_all[i] for i, k in enumerate(keys_all) if k in keys_anom]

hover_texts = [
    f"{row['date']} - Ctx {row['Context']} - Cl {row['Cluster']}<br>t_ext_score: {row['t_ext_score']:.2f}<br>Zscore: {z_scores_all[i]:.2f}"
    for i, (idx, row) in enumerate(t_ext_score_var_full.iterrows())
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

report_content = {
    'subtitle': anm_string,
    'plot_zscore': fig_zscore.to_html(full_html=False, include_plotlyjs='cdn'),
    'plots': {
        'plot_el': {},
        'plot_var': {},
    },
}

# -----------------------------------------
#              Barplot inferenze
# -----------------------------------------
def barplot_inference(inference_results, metodo):
    columns_to_plot = [col for col in inference_results.columns if col not in ['date', 'Context', 'Cluster']]

    # Mappa barplot semplice: {(plot_key, date_str): barplot_html}
    barplots = {}

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
        y_max = max_val * 1.1

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=columns_to_plot,
            y=probs,
            marker_color='lightblue',
            hovertemplate="<b>%{x}</b><br>Prob: %{y:.2f}%<extra></extra>"
        ))
        bar_fig.update_layout(
            title=f"{date} - ctx {context} cls {cluster}<br><b>{metodo}</b></br> ",
            title_x=0.5,
            yaxis_range=[0, y_max],
            yaxis_title="Probabilità (%)",
            xaxis_title="Sottosistemi",
            template="plotly_white"
        )

        barplot_html = bar_fig.to_html(full_html=False, include_plotlyjs='cdn')
        plot_key = f"fig_el_C{context}_K{cluster}"
        date_str = str(date)

        # Assegna sempre nel dizionario unico
        barplots[(plot_key, date_str)] = barplot_html

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

    # Genera elenco puntato HTML
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

    return {
        "barplots": barplots,
        "analisi_inferenza_summary": "\n".join(html_lines)
    }


results_cmp = barplot_inference(inference_results, "CMP")
results_soglie = barplot_inference(inference_results_distrib, "Soglie")


# ---------------------------------------------------
#            Anomalie aggregate e sottosistemi
# ---------------------------------------------------
set_var = set(anm_table_var[['date', 'Context', 'Cluster']].apply(tuple, axis=1))
anm_table_el['anm_var'] = anm_table_el[['Date', 'Context', 'Cluster']].apply(lambda row: tuple(row) in set_var, axis=1)
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

report_content["context_table"] = context_table_data

# plot anomalie elettriche e delle variabili
grouped = anm_el_and_var.sort_values(by=['Context', 'Cluster']).groupby(['Context', 'Cluster'])
for (context, cluster), group in grouped:
    data_anomala_list = pd.to_datetime(group['date']).tolist()
    first_date = data_anomala_list[0]  # per l'anomaly score
    data_anomala_list_str = [d.strftime('%Y-%m-%d') for d in data_anomala_list]

    df_cluster_filtr = df_cluster[df_cluster['cluster'] == f"Cluster_{cluster}"].copy()

    pivot_el = df_cluster_filtr.pivot_table(
        index='date',
        columns='time',
        values='value'
    )
    pivot_el.columns = pd.to_datetime(pivot_el.columns, format="%H:%M:%S")
    pivot_el.index = pd.to_datetime(pivot_el.index)

    match_rows = []

    for data_anomala_str in data_anomala_list_str:
        match = anm_table_el[
            (anm_table_el['Date'] == data_anomala_str) &
            (anm_table_el['Context'] == context) &
            (anm_table_el['Cluster'] == cluster)
            ]
        if not match.empty:
            match_rows.append(match)
    if match_rows:
        match_row = pd.concat(match_rows, ignore_index=True)
    else:
        match_row = pd.DataFrame()

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
        'dates': data_anomala_list_str  # aggiungiamo la lista completa di date!
    }
    report_content['plots']['plot_var'][key_el] = {}


    for date_str in data_anomala_list_str:
        report_content['plots']['plot_var'][key_el][date_str] = {
            'figures': {},
            'barplot_cmp': results_cmp['barplots'].get((key_el, date_str)),
            'barplot_soglie': results_soglie['barplots'].get((key_el, date_str)),
            'barplot_b': results_cmp['barplots'].get((key_el, date_str))
        }

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

        for date_str in data_anomala_list_str:
            if pd.to_datetime(date_str) in data_anomala_var_list:
                fig_var = go.Figure()

                # Solo profili normali (grigi)
                for date in pivot_var.index:
                    if date != pd.to_datetime(date_str):
                        fig_var.add_trace(go.Scatter(
                            x=pivot_var.columns,
                            y=pivot_var.loc[date],
                            mode='lines',
                            name=str(date.date()),
                            line=dict(color='grey'),
                            opacity=0.5,
                            hovertemplate=f"{date.date()}<br>%{{x|%H:%M}}<br>%{{y}}<extra></extra>"
                        ))

                # Solo la data anomala selezionata (rosso)
                fig_var.add_trace(go.Scatter(
                    x=pivot_var.columns,
                    y=pivot_var.loc[pd.to_datetime(date_str)],
                    mode='lines',
                    name=str(pd.to_datetime(date_str).date()),
                    line=dict(color='red'),
                    hovertemplate=f"{date_str}<br>%{{x|%H:%M}}<br>%{{y}}<extra></extra>"
                ))

                fig_var.add_vrect(x0=from_ctx, x1=to_ctx, fillcolor="darkred", opacity=0.5, layer="below", line_width=0)
                fig_var.add_vrect(x0=from_tw, x1=to_tw, fillcolor="lightcoral", opacity=0.5, layer="below",
                                  line_width=0)

                fig_var.update_layout(
                    title=f'<b>{var}</b><br>Context {context} Cluster {cluster} - {date_str}',
                    title_x=0.5,
                    xaxis_title='Time',
                    yaxis_title=var
                )

                report_content['plots']['plot_var'][key_el][date_str]['figures'][var] = fig_var.to_html(full_html=False,
                                                                                                        include_plotlyjs='cdn')

report_content["cmp"] = results_cmp["barplots"]
report_content["soglie"] = results_soglie["barplots"]
report_content["analisi_inferenza_summary"] = results_cmp["analisi_inferenza_summary"]


output_file = f"results/reports/report_var.html"
save_report(report_content, output_file, "template_var.html")
absolute_path = os.path.abspath(output_file)
webbrowser.open_new_tab(f'file://{absolute_path}')







