import json
import os
import pandas as pd
from settings import PROJECT_ROOT
from src.utils import get_nodes_by_level
import matplotlib.pyplot as plt
import numpy as np

"""
Generates and saves daily load profiles for all nodes in a case study by plotting mean and percentile bands
of 15-minute aggregated power measurements.

The script:
- Loads the case study configuration to list all nodes from the load tree.
- Reads each node’s raw time series (timestamp, value) and resamples it to 15-minute means.
- Groups data by time-of-day and computes percentile bands (20–40%, 40–60%, 60–80%) and the mean (centroid).
- Plots filled bands between percentile pairs and overlays the mean curve using a rotating qualitative colormap.
- Formats x-axis ticks as HH:MM over the day and labels axes and legend.
- Saves a PNG profile figure per node to results/<case_study>/viz/case_study_profiles.

Inputs:
- case_study (str): Case study name; determines configuration and data locations.

Outputs:
- One PNG per node named "<node>_profile.png" stored in results/<case_study>/viz/case_study_profiles.
"""

case_study = "Total"
output_file = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "nodes_profiles")
os.makedirs(output_file, exist_ok=True)
raw_data_dir = os.path.join(PROJECT_ROOT, "raw_data", case_study)

with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json")) as f:
    config = json.load(f)
    levels = get_nodes_by_level(config["Load Tree"])
    all_nodes = [node for level in levels for node in level]

available_cmaps = [
    "Blues", "Oranges", "Greens", "Purples", "Reds", "Greys"
]
num_cmaps = len(available_cmaps)

for i, leaf in enumerate(all_nodes):
    df = pd.read_csv(os.path.join(raw_data_dir, f"{leaf}.csv"), parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df = df.resample("15min").mean()
    df = df.reset_index()
    df["time"] = df["timestamp"].dt.time

    grouped = df.groupby("time")["value"]
    percentiles = grouped.quantile([0.2, 0.4, 0.6, 0.8]).unstack()
    mean = grouped.mean()

    times_sorted = sorted(mean.index)
    mean = mean.loc[times_sorted]
    # p20 = percentiles.loc[times_sorted, 0.2]
    # p40 = percentiles.loc[times_sorted, 0.4]
    # p60 = percentiles.loc[times_sorted, 0.6]
    # p80 = percentiles.loc[times_sorted, 0.8]

    cmap_name = available_cmaps[i % num_cmaps]
    cmap = plt.get_cmap(cmap_name)

    bands = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]

    x = np.arange(len(times_sorted))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, alpha=0.4)
    for i, (low, high) in enumerate(bands):
        p_low = grouped.quantile(low).loc[times_sorted]
        p_high = grouped.quantile(high).loc[times_sorted]
        ax.fill_between(
            x, p_low.values, p_high.values,
            color=cmap(0.2 + i * 0.15),
            alpha=0.8,
            label=f'{int(low * 100)}–{int(high * 100)}%'
        )

    ax.plot(x, mean.values, color=cmap(0.9), linewidth=2, label='Centroid')
    xticks_to_show = x[::max(1, len(x) // 6)]
    xticklabels = [times_sorted[i].strftime("%H:%M") for i in xticks_to_show]
    ax.set_xticks(xticks_to_show)
    ax.set_xticklabels(xticklabels)

    ax.set_title(f"{leaf}", fontsize=20)
    ax.set_ylabel("Power [kW]", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=16)

    plt.tight_layout()
    fig.savefig(os.path.join(output_file, f"{leaf}_profile.png"))
    plt.close(fig)





