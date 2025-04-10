import os
import json

import pandas as pd
import plotly.graph_objs as go

from settings import PROJECT_ROOT


def plot_stacked_area(case_study: str):
    """
    Plot a stacked area chart for the given case study.
    Args:
        case_study (str): The name of the case study to process.
    Returns:
        None
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
        config = json.load(f)

    first_level = list(config["Load Tree"].keys())[0]
    second_level = list(config["Load Tree"][first_level].keys())

    df_list = []
    for level in second_level:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{level}.csv"), index_col=0, parse_dates=True)
        df_list.append(df)

    df_second_level = pd.concat(df_list, axis=1)
    df_second_level.columns = second_level

    fig = go.Figure()
    for col in df_second_level.columns:
        fig.add_trace(go.Scatter(
            x=df_second_level.index,
            y=df_second_level[col],
            mode='lines',
            name=col,
            stackgroup='one',
            hovertemplate=f"{col}: %{{y:.2f}} W<extra></extra>",
        ))

    fig.update_layout(
        title=f"{case_study}",
        xaxis_title='Timestamp',
        yaxis_title='Power [W]',
        legend_title='',
        template='plotly_white',
        hovermode="x unified",
        title_x=0.5,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5
        ),
        title_font=dict(size=24),
        legend_font=dict(size=14),
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
        xaxis_title_font=dict(size=18),
        yaxis_title_font=dict(size=18),
    )

    fig.write_html(os.path.join(PROJECT_ROOT, "results", case_study, "viz", f"stacked_area.html"),
                   include_plotlyjs="cdn")


if __name__ == "__main__":
    plot_stacked_area("AuleR")
