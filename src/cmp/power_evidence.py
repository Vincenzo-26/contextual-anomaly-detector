import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
"""
INPUT:
-   time_windows.csv da "main.py" (run_CART)
-   anomalies_table_overall.csv da "main.py"
-   data_el_aule_R_pre.csv da "main.py"
-   anomalies_var_table_overall.csv da "main_var.py"
-   anomalies_el_&_var_table_overall da "main_var.py"
-   evidence_var_full.csv da "main_var.py"
-   evidence_el_&_var_full.csv da "main_var.py"

OUTPUT:
evidence_var_full_ditrib → Tutte le combinazioni date-context-cluster.
Per ciascuna variabile, 0 se l'energia nella TW è sotto soglia, 1 se è sopra, soglia calcolata dai punti labelled CMP.

anomalies_var_table_overall_distrib → È evidence_var_full_ditrib filtrato, solo le combinazioni date-context-cluster 
risultate anomale rispetto alle soglie, quindi dove c'è almeno un 1 tra i sottocarichi.
 
evidence_el_&_var_ditrib → È evidence_var_full_ditrib filtrato solo con le date-context-cluster anomali elettrici (alto 
livello).

anomalies_el_&_var_table_distrib → È anomalies_var_table_overall_distrib filtrato ulteriormente solo per le 
date-context-cluster che sono anomali anche a livello elettrico (quindi anomali con le soglie e ad alto livello).
"""

anm_table_var = pd.read_csv('data/diagnosis/Anomalie_tables/CMP/anomalies_var_table_overall.csv')
anm_table_el_and_var = pd.read_csv('data/diagnosis/Anomalie_tables/CMP/anomalies_el_&_var_table_overall.csv')
df_tw = pd.read_csv("data/time_windows.csv")
df_el = pd.read_csv("data/Aule_R/preprocess_data/electric_data/data_el_aule_R_pre.csv")
evidence_var_full = pd.read_csv("data/diagnosis/Evidence_tables/Power/CMP/evidence_var_full.csv")
evidence_var_and_el = pd.read_csv("data/diagnosis/Evidence_tables/Power/CMP/evidence_el_&_var_full.csv")
anm_el = pd.read_csv('data/anomalies_table_overall.csv')

df_el['timestamp'] = pd.to_datetime(df_el['timestamp'])
df_el.set_index('timestamp', inplace=True)
df_el.drop(columns=['temp'], inplace=True)

energy_var_full = evidence_var_full.copy()

for idx, row in energy_var_full.iterrows():
    date = row['date']
    context = row['Context']
    cluster = row['Cluster']

    tw_row = df_tw[df_tw['id'] == context].iloc[0]

    start_time = datetime.datetime.strptime(tw_row['from'], "%H:%M").time()
    if tw_row['to'] == "24:00":
        end_time = datetime.time(23, 59)
    else:
        end_time = datetime.datetime.strptime(tw_row['to'], "%H:%M").time()

    date_obj = pd.to_datetime(date).date()
    start_dt = pd.Timestamp.combine(date_obj, start_time)
    end_dt = pd.Timestamp.combine(date_obj, end_time)

    df_slice = df_el[(df_el.index >= start_dt) & (df_el.index < end_dt)]
    #---- Calcolo energia nella sottosequenza ----#
    for var in energy_var_full.columns:
        if (var == "date") or (var == "Cluster") or (var =="Context"):
            continue
        energy = df_slice[var].sum() * 0.25
        energy_var_full.at[idx, var] = energy

    #---- Costruzione distribuzioni ----#
variabili = [col for col in energy_var_full.columns if col not in ['date', 'Context', 'Cluster']]

set_anomali = set(tuple(x) for x in anm_table_var[['date', 'Context', 'Cluster']].values)
energy_var_anm = energy_var_full[energy_var_full[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_anomali)]
energy_var_clean = energy_var_full[~energy_var_full[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_anomali)]

soglie_records = []

for (context, cluster), gruppo in energy_var_clean.groupby(['Context', 'Cluster']):
    dev_std = gruppo[variabili].std()
    mean = gruppo[variabili].mean()

    record = {'Context': context, 'Cluster': cluster}
    for var in variabili:
        soglia_dx = (mean[var] + 3 * dev_std[var]) * 1.1
        record[f'soglia_{var}'] = soglia_dx
    soglie_records.append(record)

df_soglie = pd.DataFrame(soglie_records)

####### evidenze ########
stats_by_group = {}
for (context, cluster), group in energy_var_clean.groupby(['Context', 'Cluster']):
    stats_by_group[(context, cluster)] = {
        var: (group[var].mean(), group[var].std()) for var in variabili
    }

# --- Calcolo scores ---
energy_scores = energy_var_full.copy()

for idx, row in energy_var_full.iterrows():
    context = row['Context']
    cluster = row['Cluster']
    key_stats = stats_by_group.get((context, cluster))

    for var in variabili:
        val = row[var]

        # Recupera la soglia dal df_soglie
        soglia_dx = df_soglie.loc[
            (df_soglie['Context'] == context) & (df_soglie['Cluster'] == cluster), f'soglia_{var}'
        ].values[0]
        if val > soglia_dx:
            score = int(1)
        else:
            score = int(0)
        energy_scores.at[idx, var] = score

energy_scores.to_csv('data/diagnosis/Evidence_tables/Power/Soglie/evidence_var_full_ditrib.csv', index = False)
mask = (energy_scores.drop(columns=['date', 'Context', 'Cluster']) == 1).any(axis=1)
anm_table_var_distrib = energy_scores[mask]
anm_table_var_distrib.to_csv('data/diagnosis/Anomalie_tables/Soglie/anomalies_var_table_overall_distrib.csv', index = False)

set_el = set(anm_el[['Date', 'Context', 'Cluster']].apply(tuple, axis=1))

anm_el_and_var_distrib = anm_table_var_distrib[anm_table_var_distrib[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_el)].reset_index(drop=True)
anm_el_and_var_distrib.to_csv(f"data/diagnosis/Anomalie_tables/Soglie/anomalies_el_&_var_table_distrib.csv", index=False)

#####------ evidenze per anomalie el & var ------#######
energy_scores_filtered = energy_scores[
    energy_scores[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_el)].copy()
energy_scores_filtered.to_csv('data/diagnosis/Evidence_tables/Power/Soglie/evidence_el_&_var_ditrib.csv', index = False)


# ---------------------------------------------------
#            Calcolo metriche performance
# ---------------------------------------------------
keys_cmp = set(tuple(x) for x in anm_table_el_and_var[['date', 'Context', 'Cluster']].values)
keys_distrib = set(tuple(x) for x in anm_el_and_var_distrib[['date', 'Context', 'Cluster']].values)

diff_keys = keys_cmp.symmetric_difference(keys_distrib)

anm_table_el_and_var_diff = anm_table_el_and_var[
    anm_table_el_and_var[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(diff_keys)
]
anm_el_and_var_distrib_diff = anm_el_and_var_distrib[
    anm_el_and_var_distrib[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(diff_keys)
]

variabili_cmp_soglie = [col for col in anm_table_el_and_var.columns if col not in ['date', 'Context', 'Cluster', 'count_cmp']]

metrics_cmp_soglie = {
    'anm_el': len(set_el),
    'anm_el_and_var_cmp': len(keys_cmp),
    'anm_el_and_var_soglie': len(keys_distrib),
    'diagnosi_differenti': len(diff_keys),
}

all_keys = keys_cmp.union(keys_distrib)

cmp_all = anm_table_el_and_var[
    anm_table_el_and_var[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(all_keys)
].set_index(['date', 'Context', 'Cluster'])

distrib_all = anm_el_and_var_distrib[
    anm_el_and_var_distrib[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(all_keys)
].set_index(['date', 'Context', 'Cluster'])

cmp_all = cmp_all[variabili_cmp_soglie]
distrib_all = distrib_all[variabili_cmp_soglie]

cmp_all = cmp_all.fillna(0)
distrib_all = distrib_all.fillna(0)

# --- Calcolo matrice di confusione ---
tp = int(((cmp_all == 1) & (distrib_all == 1)).sum().sum())
tn = int(((cmp_all == 0) & (distrib_all == 0)).sum().sum())
fp = int(((cmp_all == 0) & (distrib_all == 1)).sum().sum())
fn = int(((cmp_all == 1) & (distrib_all == 0)).sum().sum())

# --- Calcolo metriche ---
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

metrics_cmp_soglie.update({
    'true_positive': tp,
    'true_negative': tn,
    'false_positive': fp,
    'false_negative': fn,
    'precision': round(precision, 3),
    'recall': round(recall, 3),
    'accuracy': round(accuracy, 3),
    'f1_score': round(f1_score, 3)
})

# -------------------------------------------
#            Figura metriche per report
# -------------------------------------------

variabili_cmp = [col for col in anm_table_el_and_var.columns if col not in ['date', 'Context', 'Cluster']]
variabili_distrib = [col for col in anm_table_var_distrib.columns if col not in ['date', 'Context', 'Cluster']]

count_cmp_dict = dict(zip(
    anm_table_el_and_var[['date', 'Context', 'Cluster']].apply(tuple, axis=1),
    anm_table_el_and_var[variabili_cmp].sum(axis=1)
))
count_distrib_dict = dict(zip(
    anm_table_var_distrib[['date', 'Context', 'Cluster']].apply(tuple, axis=1),
    anm_table_var_distrib[variabili_distrib].sum(axis=1)
))
confronto_records = []

for key in set_el:
    count_cmp = count_cmp_dict.get(key, 0)
    count_distrib = count_distrib_dict.get(key, 0)
    confronto_records.append({
        'key': key,
        'count_cmp': count_cmp,
        'count_distrib': count_distrib
    })

confronto_merge = pd.DataFrame(confronto_records)

grouped_confronto = confronto_merge.groupby(['count_cmp', 'count_distrib']).size().reset_index(name='counts')
cmp_vars_dict = {
    tuple(row[['date', 'Context', 'Cluster']]): [
        var for var in variabili_cmp if row[var] == 1
    ]
    for _, row in anm_table_el_and_var.iterrows()
}
distrib_vars_dict = {
    tuple(row[['date', 'Context', 'Cluster']]): [
        var for var in variabili_distrib if row[var] == 1
    ]
    for _, row in anm_table_var_distrib.iterrows()
}
def create_hover(row):
    if row['count_cmp'] == row['count_distrib']:
        return f"Occorrenze: {int(row['counts'])}"
    else:
        details = confronto_merge[
            (confronto_merge['count_cmp'] == row['count_cmp']) &
            (confronto_merge['count_distrib'] == row['count_distrib'])
        ]
        txt = f"Occorrenze: {int(row['counts'])}<br><br>"
        for _, det in details.iterrows():
            key = det['key']
            date_str = f"{key[0]}"
            context_str = key[1]
            cluster_str = key[2]

            var_cmp = cmp_vars_dict.get(key, ['Nessuna informazione'])
            var_distrib = distrib_vars_dict.get(key, ['Nessuna diagnosi'])

            txt += f"{date_str} ctx {context_str} clst {cluster_str}<br>"
            txt += f"<b>CMP:</b> {', '.join(var_cmp) if var_cmp else 'Nessuna'}<br>"
            txt += f"<b>Soglie:</b> {', '.join(var_distrib) if var_distrib else 'Nessuna'}<br><br>"

        return txt

grouped_confronto['custom_hover'] = grouped_confronto.apply(create_hover, axis=1)
max_var_count = max(len(variabili_cmp), len(variabili_distrib))

fig_cmp_vs_distrib = px.scatter(
    grouped_confronto,
    x='count_cmp',
    y='count_distrib',
    size='counts',
    color='counts',
    size_max=30,
    labels={
        'count_cmp': 'CMP Anomaly Score',
        'count_distrib': 'Soglie',
        'counts': 'Occorrenze'
    },
    color_continuous_scale=['red', 'orange', 'yellow', 'green', 'cyan'],
    custom_data=['custom_hover']
)
fig_cmp_vs_distrib.update_traces(
    hovertemplate='%{customdata[0]}<extra></extra>'
)
fig_cmp_vs_distrib.add_shape(
    type='line',
    x0=0,
    y0=0,
    x1=max_var_count,
    y1=max_var_count,
    line=dict(color='gray'),
    opacity=0.7,
    layer='below'
)
fig_cmp_vs_distrib.update_layout(
    template='plotly_white',
    xaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=1,
        range=[-0.2, max_var_count + 0.2],
    ),
    yaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=1,
        range=[-0.2, max_var_count + 0.2],
    ),
    title=dict(
        text='CMP vs Soglie',
        x=0.5,
        xanchor='center'
    ),
)
# fig_cmp_vs_distrib.show()


def plot_cmp_vs_soglia(sottocarico, context, cluster):
    df_normali = energy_var_full[
        (energy_var_full['Context'] == context) &
        (energy_var_full['Cluster'] == cluster)
    ]
    set_anomali_cmp = set(tuple(x) for x in anm_table_var[['date', 'Context', 'Cluster']].values)
    df_normali = df_normali[
        ~df_normali[['date', 'Context', 'Cluster']].apply(tuple, axis=1).isin(set_anomali_cmp)
    ]
    valori_normali = df_normali[sottocarico]
    media_normali = valori_normali.mean()
    std_normali = valori_normali.std()
    soglia_dinamica = (media_normali + 3 * std_normali) * 1.1
    media_plus_std = media_normali + std_normali
    df_anomali_cmp = energy_var_full[
        (energy_var_full['Context'] == context) &
        (energy_var_full['Cluster'] == cluster)
    ]
    key_anm_cmp = anm_table_var.loc[
        (anm_table_var[sottocarico] == 1) &
        (anm_table_var['Context'] == context) &
        (anm_table_var['Cluster'] == cluster),
        ['date', 'Context', 'Cluster']
    ]
    df_anomali_cmp = df_anomali_cmp.merge(key_anm_cmp, on=['date', 'Context', 'Cluster'], how='inner')
    valori_anomali_cmp = df_anomali_cmp[sottocarico]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valori_normali,
        y=[0] * len(valori_normali),
        mode='markers',
        marker=dict(color='skyblue', size=8),
        text=df_normali['date'],
        hovertemplate="%{text}<br>Energia: %{x:.2f}kWh<extra></extra>",
        name='Normali CMP'
    ))
    fig.add_trace(go.Scatter(
        x=valori_anomali_cmp,
        y=[0] * len(valori_anomali_cmp),
        mode='markers',
        marker=dict(color='red', size=8),
        text=df_anomali_cmp['date'],
        hovertemplate="%{text}<br>Energia: %{x:.2f}kWh<extra></extra>",
        name='Anomali CMP'
    ))
    fig.add_vline(
        x=soglia_dinamica,
        line_dash='dash',
        line_color='red',
        annotation_text=f"Soglia: {soglia_dinamica:.2f}",
        annotation_position="top right"
    )
    fig.add_vline(
        x=media_normali,
        line_dash='dash',
        line_color='blue',
        annotation_text=f"Media: {media_normali:.2f}",
        annotation_position="top right"
    )
    fig.add_vline(
        x=media_plus_std,
        line_dash='dash',
        line_color='lightblue',
        annotation_text=f"Media + σ: {media_plus_std:.2f}",
        annotation_position="top right"
    )
    fig.update_layout(
        title=f"Energia {sottocarico} (ctx {context}, clst {cluster}) - μ = {media_normali:.2f} kWh, σ = {std_normali:.2f} kWh",
        xaxis_title="Energia nella sottosequenza",
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
        template='plotly_white'
    )
    fig.show()

plot_cmp_vs_soglia("el_UTA_3_3B_7", 2, 5)


