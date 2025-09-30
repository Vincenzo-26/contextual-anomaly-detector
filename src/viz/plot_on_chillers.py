import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from settings import PROJECT_ROOT

case_study = "Total_cut"
output_path = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "ON_hour")
os.makedirs(output_path, exist_ok=True)

chillers = ['GF1', 'GF2', 'GF3', 'GF4', 'GF5']

for leaf in chillers:
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{leaf}.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    df_daily = df.groupby('date')['value'].mean().reset_index()

    # Classificazione: 0 -> "off", >0 -> "on"
    df_daily["status"] = df_daily["value"].apply(lambda v: "off" if v == 0 else "on")
    df_daily["weekday"] = df_daily["date"].dt.weekday
    df_daily["week"] = df_daily["date"].apply(lambda d: d - pd.Timedelta(days=d.weekday()))

    pivot = df_daily.pivot(index="weekday", columns="week", values="status")

    color_map = {
        "on": "steelblue",
        "off": "indianred"
    }
    color_matrix = pivot.applymap(lambda x: color_map.get(x, "#ffffff"))

    fig, ax = plt.subplots(figsize=(len(pivot.columns) * 0.2, 2.2))
    for i, weekday in enumerate(pivot.index):
        for j, week in enumerate(pivot.columns):
            color = color_matrix.loc[weekday, week]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))

    for i in range(8):
        ax.axhline(i, color='white', linewidth=0.8)
    for j in range(len(pivot.columns) + 1):
        ax.axvline(j, color='white', linewidth=0.8)

    xticks = np.arange(len(pivot.columns))
    step = max(1, len(xticks) // 15)
    xtick_labels = [w.strftime('%Y-%m-%d') if i % step == 0 else '' for i, w in enumerate(pivot.columns)]
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels(xtick_labels, fontsize=7, rotation=30)

    ax.set_yticks(np.arange(7) + 0.5)
    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    ax.set_xlim(0, len(pivot.columns))
    ax.set_ylim(0, 7)
    ax.invert_yaxis()
    ax.set_title(f"{leaf}", fontsize=18)
    # ax.tick_params(axis='both', labelsize=10)
    legend_elements = [
        Patch(facecolor='steelblue', label='On'),
        Patch(facecolor='indianred', label='Off')
    ]
    fig.subplots_adjust(right=0.82)
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), title="Status", fontsize=14)

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, f"{leaf}_calendarplot.png"), dpi=300)
    plt.close(fig)
