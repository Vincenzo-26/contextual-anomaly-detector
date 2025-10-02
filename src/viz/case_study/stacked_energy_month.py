import os
import pandas as pd
import matplotlib.pyplot as plt
from settings import PROJECT_ROOT

case_study = "Total"

df = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, f"Pred_XGboost", "wasted_energy_by_leaf_month_BN.csv"))
df.set_index(df.columns[0], inplace=True)
df_t = df.T
df_t.index = pd.to_datetime(df_t.index, format="%Y-%m")
df_t = df_t.sort_index()
df_t.index = df_t.index.strftime("%Y-%m")
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.Pastel2.colors
num_colors = len(df_t.columns)
pastel_colors = [colors[i % len(colors)] for i in range(num_colors)]

# Grafico stacked bar
bottom = pd.Series([0]*len(df_t), index=df_t.index)
for i, column in enumerate(df_t.columns):
    ax.bar(df_t.index, df_t[column], bottom=bottom, label=column, color=pastel_colors[i])
    bottom += df_t[column]

ax.set_ylabel("Energy [kWh]", fontsize=14)
plt.grid(True, which='major', axis='y')
plt.xticks(rotation=45)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()