import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from settings import PROJECT_ROOT
import matplotlib.ticker as ticker


def plot_time_windows(profiles_path, tw_path, output_path):
    df = pd.read_csv(profiles_path, index_col=0, parse_dates=True)
    df['date'] = df.index.date
    df['hour'] = df.index.hour + df.index.minute / 60

    tw = pd.read_csv(tw_path)
    tw = tw.sort_values("from")
    colors = sns.color_palette("pastel", len(tw))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for date in df['date'].unique():
        day_df = df[df['date'] == date]
        ax.plot(day_df['hour'], day_df.iloc[:, 0], color='grey', alpha=0.2, linewidth=0.8)

    for i, row in tw.iterrows():
        def parse_time_to_hour(t):
            if t == "24:00":
                return 24.0
            h, m = map(int, t.split(":"))
            return h + m / 60

        from_hour = parse_time_to_hour(row['from'])
        to_hour = parse_time_to_hour(row['to'])
        ax.axvspan(from_hour, to_hour, color=colors[i], alpha=0.2)

    tick_hours = []
    for _, row in tw.iterrows():
        tick_hours.append(parse_time_to_hour(row['from']))
        tick_hours.append(parse_time_to_hour(row['to']))
    tick_hours = sorted(set(tick_hours))

    ax.set_xlim(0, 24)
    ax.set_xticks(tick_hours)
    ax.set_xticklabels([f"{int(h):02d}:{int((h % 1) * 60):02d}" for h in tick_hours])
    ax.set_ylabel("Power [kW]")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    output_folder = os.path.join(PROJECT_ROOT, "results", "Total", "viz", "time_windows")
    total_path = os.path.join(PROJECT_ROOT, "data", "Total", "Total.csv")
    tw_path = os.path.join(PROJECT_ROOT, "results", "Total", "time_windows.csv")
    plot_time_windows(total_path, tw_path, output_folder)