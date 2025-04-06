import pandas as pd
import datetime
from scipy.stats import norm
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
-   evidence_var_full_ditrib: tante righe quante sono le combinazioni di date-context-cluster. In corrispondenza
    di ciascuna variabile è presente 0 se l'energia [kWh] nella tw è minore della soglia di quel context-cluster
    (calcolata a partire dai punti labelled grazie alla CMP) oppure 1 se è maggiore.
-   anomalies_var_table_overall_distrib: è evidence_var_full_ditrib filtrato solo con le date-context-cluster risultati
    anomali con le soglie (quindi dove c'è almeno un 1 tra i sotocarichi)
-   evidence_el_&_var_ditrib: è evidence_var_full_ditrib filtrato solo con le date-context-cluster contemporaneamente 
    anomali ad alto livello e a basso livello con le soglie
"""

anm_table_var = pd.read_csv('data/diagnosis/Anomalie_tables/CMP/anomalies_var_table_overall.csv')
anm_table_el = pd.read_csv('data/diagnosis/Anomalie_tables/CMP/anomalies_el_&_var_table_overall.csv')
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

set_anomali = anm_table_var[['date', 'Context', 'Cluster']].astype(str).agg('-'.join, axis=1)
energy_var_full['key'] = energy_var_full[['date', 'Context', 'Cluster']].astype(str).agg('-'.join, axis=1)
energy_var_anm = energy_var_full[energy_var_full['key'].isin(set_anomali)]
energy_var_clean = energy_var_full[~energy_var_full['key'].isin(set_anomali)]

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

#####------ evidenze per anomalie el & var ------#######
set_el = set(anm_el[['Date', 'Context', 'Cluster']].apply(tuple, axis=1))
energy_scores['key_tuple'] = energy_scores[['date', 'Context', 'Cluster']].apply(tuple, axis=1)
energy_scores_filtered = energy_scores[energy_scores['key_tuple'].isin(set_el)].copy()
energy_scores_filtered.drop(columns='key_tuple', inplace=True)
energy_scores_filtered.drop(columns='key', inplace=True)
energy_scores_filtered.to_csv('data/diagnosis/Evidence_tables/Power/Soglie/evidence_el_&_var_ditrib.csv', index = False)


# energy_raw = pd.read_csv("data/diagnosis/anomalies_table_var/evidence_el_&_var_full.csv")
#
# # Colonne da sogliare
# id_cols = ["date", "Context", "Cluster", "t_ext_score"]
# var_cols = [col for col in energy_raw.columns if col not in id_cols]
#
# # Applica soglia 0.75 a energy_raw
# energy_raw_binarized = energy_raw.copy()
# energy_raw_binarized[var_cols] = (energy_raw_binarized[var_cols] >= 0.75).astype(int)
#
# # Aggiungiamo chiave per merge
# energy_raw_binarized["key"] = energy_raw_binarized[["date", "Context", "Cluster"]].astype(str).agg("-".join, axis=1)
# energy_scores_filtered["key"] = energy_scores_filtered[["date", "Context", "Cluster"]].astype(str).agg("-".join, axis=1)
#
# # Merge sulle chiavi
# merged = energy_scores_filtered.merge(
#     energy_raw_binarized,
#     on="key",
#     suffixes=("_score", "_binarized"),
#     how="left"
# )
#
# # Estrai differenze
# diff_details = []
# for _, row in merged.iterrows():
#     for col in var_cols:
#         val_score = row.get(f"{col}_score")
#         val_binarized = row.get(f"{col}_binarized")
#         if pd.notna(val_binarized) and val_score != val_binarized:
#             diff_details.append({
#                 "date": row["date_score"],
#                 "Context": row["Context_score"],
#                 "Cluster": row["Cluster_score"],
#                 "Variable": col,
#                 "Thresholded": val_score,
#                 "Original (>=0.75)": val_binarized
#             })
#
# # Stampa risultati
# if diff_details:
#     print("\n--- DIFFERENZE RISCONTRATE (filtrate sulle anomalie elettriche) ---")
#     for d in diff_details:
#         print(f"Data: {d['date']}, Context: {d['Context']}, Cluster: {d['Cluster']}, "
#               f"Var: {d['Variable']} -> Distrib: {d['Thresholded']}, CMP: {d['Original (>=0.75)']}")
# else:
#     print("\nNessuna differenza riscontrata tra energy_scores_filtered e energy_raw sogliato a 0.75.")
# # plot di un esempio
# sottocarico_plot = "el_UTA_4_4B_8"
# context_plot = 2
# cluster_plot = 2
#
# df_plot = energy_var_full[
#     (energy_var_full['Context'] == context_plot) &
#     (energy_var_full['Cluster'] == cluster_plot)
# ]
# valori = df_plot[f'{sottocarico_plot}']
#
# key_anm_var = anm_table_var.loc[
#     (anm_table_var[f'{sottocarico_plot}'] == 1) &
#     (anm_table_var['Context'] == context_plot) &
#     (anm_table_var['Cluster'] == cluster_plot),
#     ['date', 'Context', 'Cluster']
# ]
# df_plot_anm = energy_var_anm.merge(key_anm_var, on=['date', 'Context', 'Cluster'], how='inner')
# valori_anm = df_plot_anm[f'{sottocarico_plot}']
#
# key_anm_el = anm_table_el.loc[anm_table_el[f'{sottocarico_plot}'] == 1, ['date', 'Context', 'Cluster']]
# df_plot_anm_and_el = df_plot_anm.merge(key_anm_el, on=['date', 'Context', 'Cluster'], how='inner')
# valori_anm_and_el = df_plot_anm_and_el[f'{sottocarico_plot}']
#
# soglia = df_soglie.loc[
#     (df_soglie['Context'] == context_plot) & (df_soglie['Cluster'] == cluster_plot), f'soglia_{sottocarico_plot}'
# ].values[0]
# media_plot, std_plot = stats_by_group[(context_plot, cluster_plot)][sottocarico_plot]
# media_plus_std = media_plot + std_plot
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=valori,
#     y=[0]*len(valori),
#     mode='markers',
#     marker=dict(color='skyblue', size=8),
#     text=df_plot['date'],
#     hovertemplate="%{text}<br>Energia: %{x:.2f}kWh<extra></extra>",
#     name='Normal'
# ))
# fig.add_trace(go.Scatter(
#     x=valori_anm,
#     y=[0]*len(valori_anm),
#     mode='markers',
#     marker=dict(color='red', size=8),
#     text=df_plot_anm['date'],
#     hovertemplate="%{text}<br>Energia: %{x:.2f}kWh<extra></extra>",
#     name='Anm var'
# ))
# fig.add_trace(go.Scatter(
#     x=valori_anm,
#     y=[0]*len(valori_anm_and_el),
#     mode='markers',
#     marker=dict(color='darkgoldenrod', size=8),
#     text=df_plot_anm_and_el['date'],
#     hovertemplate="%{text}<br>Energia: %{x:.2f}kWh<extra></extra>",
#     name='Anm el&var'
# ))
# fig.add_vline(
#     x=soglia,
#     line_dash='dash',
#     line_color='red',
#     annotation_text=f"Soglia: {soglia:.2f}",
#     annotation_position="top right",
#     name='Soglia'
# )
# fig.add_vline(
#     x=media_plot,
#     line_dash='dash',
#     line_color='blue',
#     name = 'Media'
# )
# fig.add_vline(
#     x=media_plus_std,
#     line_dash='dash',
#     line_color='lightblue',
#     name='dev.std'
# )
# fig.update_layout(
#     title=f"Energia {sottocarico_plot} (ctx {context_plot} , Cls {cluster_plot}) - μ = {media_plot:.2f} kWh, σ = {std_plot:.2f} kWh",
#     xaxis_title="Energia nella sottosequenza",
#     yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
#     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#     showlegend=True
# )
# # fig.show()