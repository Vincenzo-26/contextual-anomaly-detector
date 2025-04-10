import os
import json

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from settings import PROJECT_ROOT


def plot_thermal_sensitivity(case_study: str):
    """
    Plot the thermal sensitivity for the given case study.
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

    temp_file = config["Outside Temperature"]
    df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{temp_file}.csv"), index_col=0,
                          parse_dates=True)

    fig = make_subplots(rows=1, cols=len(df_second_level.columns), horizontal_spacing=0.02)
    for i, col in enumerate(df_second_level.columns):
        df = df_second_level[[col]].merge(df_temp, left_index=True, right_index=True)
        df.columns = ["Power", "Temperature"]

        # Resample to 1 day and calculate the energy
        df = df.resample("1D").agg(
            {"Power": "sum", "Temperature": "mean"}
        )
        df["Energy"] = df["Power"] * 0.25 / 1000  # Convert to kWh
        df["Date"] = df.index.date
        df["DayofWeek"] = df.index.dayofweek
        # Transform day of week into string
        df["DayofWeek"] = df["DayofWeek"].apply(lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x])

        # Add to the figure: on the xaxis the temperature and on the yaxis the power. Add in the hovertemplate
        # the date and the dayofweek
        fig.add_trace(go.Scatter(
            x=df["Temperature"],
            y=df["Energy"],
            mode='markers',
            text=df["Date"].astype(str) + " " + df["DayofWeek"],
            name=col,
        ), row=1, col=i + 1)

        fig.update_xaxes(title_text="Temperature [°C]", row=1, col=i + 1)
        fig.update_traces(
            hovertemplate="%{text}<br>Temperature: <b>%{x:.2f}</b> °C<br>Energy: <b>%{y:.2f}</b> kWh<extra></extra>",
            row=1, col=i + 1)
        fig.update_layout(
            title=f"{case_study}",
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
            yaxis_tickfont=dict(size=14)
        )

    fig.update_layout(height=200*len(df_second_level.columns))

    fig.write_html(os.path.join(PROJECT_ROOT, "results", case_study, "viz", "thermal_sensitivity.html"),
                   include_plotlyjs="cdn")


if __name__ == "__main__":
    plot_thermal_sensitivity("AuleR")
