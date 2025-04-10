import os
import json

import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from settings import PROJECT_ROOT


def plot_groups(case_study: str):
    """
    Plot the groups for the given case study.
    Args:
        case_study (str): The name of the case study to process.
    Returns:
        None
    """
    # Load the configuration file

    with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
        config = json.load(f)

    first_level = list(config["Load Tree"].keys())[0]

    # Load the data
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{first_level}.csv"), index_col=0,
                     parse_dates=True)
    df["date"] = df.index.date
    df = df.reset_index(drop=False)

    # Load the groups
    groups = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv"), index_col=0)
    groups.index = pd.to_datetime(groups.index).date
    groups = groups.melt(ignore_index=False)
    groups = groups[groups["value"] == 1]
    groups = groups.drop(columns=["value"])
    groups = groups.rename(columns={"variable": "Cluster"})
    groups = groups.reset_index(names="date")

    # Merge the groups with the data
    df = df.merge(groups, on=["date"], how="left")

    df["hour"] = df["timestamp"].dt.strftime("%H:%M")

    df = df.sort_values(by=["Cluster", "timestamp"])

    fig = make_subplots(rows=1, cols=len(df["Cluster"].unique()), shared_xaxes=True, horizontal_spacing=0.02,
                        subplot_titles=df["Cluster"].unique(), shared_yaxes=True)

    # Generate a color palette and then convert into rgb string
    palette = sns.color_palette("magma", len(df["Cluster"].unique()))
    palette = [f"rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})" for color in palette]

    for i, cluster in enumerate(df["Cluster"].unique()):
        df_cluster = df[df["Cluster"] == cluster]

        for date in df_cluster["date"].unique():
            df_day = df_cluster[df_cluster["date"] == date]
            fig.add_trace(
                go.Scatter(
                    x=df_day["hour"],
                    y=df_day["value"],
                    mode='lines',
                    name=str(date),  # Optional: change to '' if you want no legend entries
                    showlegend=False,  # Set to True if you want to see dates in legend
                    line=dict(color=palette[i], width=1),
                    hovertemplate=f"{date} %{{x}}: %{{y:.2f}} W<extra></extra>",

                ),
                row=1,
                col=i + 1
            )

    fig.update_layout(
        title=f"{case_study}",
        xaxis_title='Hour',
        yaxis_title='Power [W]',
        template='plotly_white',
        title_x=0.5,
        title_font=dict(size=24),
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
    )

    fig.write_html(os.path.join(PROJECT_ROOT, "results", case_study, "viz", f"groups.html"),
                   include_plotlyjs="cdn")


if __name__ == "__main__":
    plot_groups("AuleR")
