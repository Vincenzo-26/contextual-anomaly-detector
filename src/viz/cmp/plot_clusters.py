import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from settings import PROJECT_ROOT
from src.utils import get_nodes_by_level
import matplotlib.ticker as ticker


def plot_groups(case_study: str):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    output_folder = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "groups")
    os.makedirs(output_folder, exist_ok=True)

    levels = get_nodes_by_level(config["Load Tree"])
    all_nodes = [node for level in levels for node in level]

    for leaf in all_nodes:
        if leaf != "Total":
            continue

        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{leaf}.csv"),
                         index_col=0, parse_dates=True)
        df["date"] = df.index.date
        df = df.reset_index(drop=False)

        # Load groups
        groups = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv"), index_col=0)
        groups.index = pd.to_datetime(groups.index).date
        groups = groups.melt(ignore_index=False)
        groups = groups[groups["value"] == 1].drop(columns=["value"])
        groups = groups.rename(columns={"variable": "Cluster"}).reset_index(names="date")

        df = df.merge(groups, on=["date"], how="left")
        df["hour"] = df["timestamp"].dt.strftime("%H:%M")
        df = df.sort_values(by=["Cluster", "timestamp"])

        clusters = df["Cluster"].dropna().unique()
        num_clusters = len(clusters)

        palette = sns.color_palette("magma", num_clusters)
        cluster_color = {cluster: palette[i] for i, cluster in enumerate(clusters)}

        fig, axes = plt.subplots(1, num_clusters, figsize=(3 * num_clusters, 4), sharey=True)
        if num_clusters == 1:
            axes = [axes]

        for ax, cluster in zip(axes, clusters):
            df_cluster = df[df["Cluster"] == cluster]

            for date in df_cluster["date"].unique():
                df_day = df_cluster[df_cluster["date"] == date]
                ax.plot(df_day["hour"], df_day["value"],
                        color=cluster_color[cluster], alpha=0.2, linewidth=0.8)

            df_centroid = df_cluster.groupby("hour")["value"].mean().reset_index()
            ax.plot(df_centroid["hour"], df_centroid["value"],
                    color=cluster_color[cluster], linewidth=2.5, alpha=1.0, label="Centroid")
            ax.grid(True, color='lightgray', linestyle='--', linewidth=0.4, alpha=0.5)
            ax.set_title(f"{cluster}", fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.set_xticks(ax.get_xticks()[::3])
        axes[0].set_ylabel("Power [kW]")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig.savefig(os.path.join(output_folder, f"groups_{leaf}.png"), dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    plot_groups("Total")
