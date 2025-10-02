import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from settings import PROJECT_ROOT
from src.utils import clean_time_series
from matplotlib.patches import Patch

def plot_cleaning_carpetplot(case_study: str, node_name: str, start_date: str = None, end_date: str = None):
    """
    Visualizza un carpetplot con codifica colore in base alla tecnica di pulizia
    per ciascun giorno del nodo specificato.

    Args:
        node_name (str): Nome del nodo (es. "QE Pompe")
        case_study (str): Nome del caso studio (es. "Cabina")
        start_date (str): Data di inizio in formato "YYYY-MM-DD" (opzionale)
        end_date (str): Data di fine in formato "YYYY-MM-DD" (opzionale)
    """

    ore_remove = 4
    ore_interp = 1

    path_csv = os.path.join(PROJECT_ROOT, "raw_data", case_study, f"{node_name}.csv")
    if not os.path.exists(path_csv):
        print(f"❌ File not found: {path_csv}")
        return

    df_raw = pd.read_csv(path_csv, parse_dates=["timestamp"])

    # Filtro per date se specificate
    if start_date:
        start = pd.to_datetime(start_date)
        if start < df_raw['timestamp'].min():
            print(f"❌ Start date {start_date} not in dataset.")
            return
        df_raw = df_raw[df_raw['timestamp'] >= start]

    if end_date:
        end = pd.to_datetime(end_date) + pd.Timedelta("23:45:00")
        if end > df_raw['timestamp'].max():
            print(f"❌ End date {end_date} not in dataset.")
            return
        df_raw = df_raw[df_raw['timestamp'] <= end]

    if df_raw.empty:
        print("❌ No data available in the specified date range.")
        return

    strategies = {}
    def wrapped_clean(df):
        nonlocal strategies
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        df = df[~df.index.duplicated()]
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="15min")
        df = df.reindex(full_index)

        stats = {"full day": [], "interpolated": [], "knn imputed": [], "removed": []}

        for day in df.index.normalize().unique():
            expected_index = pd.date_range(start=day, end=day + pd.Timedelta("23:45:00"), freq="15min")
            daily = df.reindex(expected_index)
            val = daily['value']
            if val.isnull().sum() == 0:
                stats["full day"].append(day.date())
            else:
                gaps = val.isnull().astype(int).groupby(val.notnull().astype(int).cumsum()).sum()
                max_gap = gaps.max() if not gaps.empty else 0
                if max_gap > ore_remove * 4:
                    stats["removed"].append(day.date())
                elif all(g <= ore_interp * 4 for g in gaps):
                    stats["interpolated"].append(day.date())
                else:
                    stats["knn imputed"].append(day.date())

        for method, days in stats.items():
            for d in days:
                strategies[d] = method

        return clean_time_series(df_raw)

    _ = wrapped_clean(df_raw)

    if not strategies:
        print("⚠️ No cleaning strategy information found.")
        return

    df_status = pd.DataFrame.from_dict(strategies, orient="index", columns=["status"])
    df_status.index = pd.to_datetime(df_status.index)
    df_status["weekday"] = df_status.index.weekday
    df_status["week"] = df_status.index.to_period("W").to_timestamp()

    pivot = df_status.pivot(index="weekday", columns="week", values="status")

    color_map = {
        "full day": "#deebf7",     # Blu chiaro
        "interpolated": "#9ecae1", # Blu medio
        "knn imputed": "#3182bd",  # Blu intenso
        "removed": "#08306b"       # Blu molto scuro
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
    ax.set_title(f"Load tree node: '{node_name}'")

    legend_elements = [Patch(facecolor=c, label=lbl.capitalize()) for lbl, c in color_map.items()]
    fig.subplots_adjust(right=0.82)
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), title="")

    plt.tight_layout()
    plt.show()

# TODO includere i missing days non presenti nel df_raw nel carpetplot perchè al momento li elimina in au
plot_cleaning_carpetplot("Total", "Total", "2024-03-01", "2025-05-31")

