import webbrowser
import pandas as pd
import plotly.graph_objects as go
import datetime
import os
from src.cmp.utils import *
from src.cmp.utils_hard_rules import *

# Load data
anm_table_el = pd.read_csv('data/diagnosis/anomalies_table_overall.csv')
anm_table_el.rename(columns={"Date": "date"}, inplace=True)
anm_table_el['date'] = pd.to_datetime(anm_table_el['date']).dt.date

anomalous_days = set(anm_table_el[['date', 'Cluster', 'Context']].itertuples(index=False, name=None))

# Load other datasets
df_tw = pd.read_csv("data/diagnosis/time_windows.csv")
df_ctx = pd.read_csv("data/contexts.csv")
df_cluster = pd.read_csv("data/diagnosis/cluster_data.csv")

# Load and process preprocessed electrical data
df_el_preprocessed = pd.read_csv("data/Aule_R/preprocess_data/electric_data/data_el_aule_R_pre.csv")
df_el_preprocessed = extract_date_time(df_el_preprocessed)
df_el_preprocessed.set_index('timestamp', inplace=True)

report_content = {'plots': []}

# Group by context and cluster for sorted iteration
sorted_context_cluster = sorted(set((cl, ct) for (_, cl, ct) in anomalous_days), key=lambda x: (x[1], x[0]))

for (cluster, context) in sorted_context_cluster:
    cluster_name = f"Cluster_{cluster}"

    from_tw = datetime.datetime.strptime(df_tw.iloc[context - 1]["from"], "%H:%M").time()
    to_tw = datetime.datetime.strptime("23:45" if context == len(df_tw) else df_tw.iloc[context - 1]["to"], "%H:%M").time()

    from_ctx = datetime.datetime.strptime(df_ctx.iloc[context - 1]["from"], "%H:%M").time()
    to_ctx = (datetime.datetime.strptime(df_ctx.iloc[context - 1]["to"], "%H:%M") - datetime.timedelta(minutes=15)).time()

    df_cluster_filtr = df_cluster[df_cluster['cluster'] == cluster_name]
    pivot_el = df_cluster_filtr.pivot_table(index='date', columns='time', values='value')

    fig_el = go.Figure()

    for date in pivot_el.index:
        if (pd.to_datetime(date).date(), cluster, context) not in anomalous_days:
            fig_el.add_trace(go.Scatter(
                x=pivot_el.columns,
                y=pivot_el.loc[date],
                mode='lines',
                name=str(date),
                line=dict(color='gray'),
                hovertemplate=f"{date}<br>%{{x}}<br>%{{y}}kW<extra></extra>"
            ))

    for date in pivot_el.index:
        if (pd.to_datetime(date).date(), cluster, context) in anomalous_days:
            fig_el.add_trace(go.Scatter(
                x=pivot_el.columns,
                y=pivot_el.loc[date],
                mode='lines',
                name=str(date),
                line=dict(color='red'),
                hovertemplate=f"{date}<br>%{{x}}<br>%{{y}}kW<extra></extra>"
            ))

    fig_el.add_vrect(x0=from_ctx, x1=to_ctx, fillcolor="darkred", opacity=0.5, layer="below", line_width=0)
    fig_el.add_vrect(x0=from_tw, x1=to_tw, fillcolor="lightcoral", opacity=0.5, layer="below", line_width=0)

    fig_el.update_layout(
        title=f'Context {context} Cluster {cluster}',
        title_x=0.5,
        xaxis_title='Time',
        yaxis_title='Electric Power [kW]'
    )

    block = {
        'title': f"Context {context} Cluster {cluster}",
        'fig_el': fig_el.to_html(full_html=False, include_plotlyjs='cdn'),
        'fig_vars': {}
    }

    date_uniche_cluster = df_cluster_filtr["date"].unique().tolist()
    date_uniche_cluster = [pd.to_datetime(d).date() for d in date_uniche_cluster]

    for var in df_el_preprocessed.columns:
        if var in ["date", "time"]:
            continue

        df_filtered = df_el_preprocessed[df_el_preprocessed["date"].isin(date_uniche_cluster)]
        pivot_var = df_filtered.pivot_table(index='date', columns='time', values=var)

        fig_var = go.Figure()
        for date in pivot_var.index:
            if (date, cluster, context) not in anomalous_days:
                fig_var.add_trace(go.Scatter(
                    x=pivot_var.columns,
                    y=pivot_var.loc[date],
                    mode='lines',
                    name=str(date),
                    line=dict(color='gray'),
                    hovertemplate=f"{date}<br>%{{x}}<br>%{{y}}<extra></extra>"
                ))
        for date in pivot_var.index:
            if (date, cluster, context) in anomalous_days:
                fig_var.add_trace(go.Scatter(
                    x=pivot_var.columns,
                    y=pivot_var.loc[date],
                    mode='lines',
                    name=str(date),
                    line=dict(color='red'),
                    hovertemplate=f"{date}<br>%{{x}}<br>%{{y}}<extra></extra>"
                ))

        fig_var.add_vrect(x0=from_ctx, x1=to_ctx, fillcolor="darkred", opacity=0.5, layer="below", line_width=0)
        fig_var.add_vrect(x0=from_tw, x1=to_tw, fillcolor="lightcoral", opacity=0.5, layer="below", line_width=0)

        fig_var.update_layout(
            title=f'{var} Context {context} Cluster {cluster}',
            title_x=0.5,
            xaxis_title='Time',
            yaxis_title=var
        )

        block['fig_vars'][var] = fig_var.to_html(full_html=False, include_plotlyjs=False)

    report_content['plots'].append(block)

output_file = "results/reports/report_anm_el_filtered.html"
save_report(report_content, output_file, "template_anm_el.html")
absolute_path = os.path.abspath(output_file)
webbrowser.open_new_tab(f'file://{absolute_path}')
