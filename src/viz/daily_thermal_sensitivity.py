import os
os.environ["OMP_NUM_THREADS"] = "1"
import json

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from settings import PROJECT_ROOT
from src.utils import get_nodes_by_level


def plot_thermal_sensitivity(case_study: str, percentile: float, type: str):
    """
    Plot the thermal sensitivity for the given case study.
    Args:
        case_study (str): The name of the case study to process.
    Returns:
        None
    """
    print(f"Percentile: {percentile*100}th")
    with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
        config = json.load(f)
    output_folder = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "daily_thermal_sensitivity")
    os.makedirs(output_folder, exist_ok=True)

    levels = get_nodes_by_level(config["Load Tree"])
    for level in levels:
        for leaf in level:
            df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{leaf}.csv"), index_col=0,
                             parse_dates=True)

            temp_file = config["Outside Temperature"]
            df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{temp_file}.csv"), index_col=0,
                                  parse_dates=True)

            df = df.merge(df_temp, left_index=True, right_index=True)
            df.columns = ["Power", "Temperature"]

            # Resample to 1 day and calculate the energy
            df = df.resample("1D").agg(
                {"Power": "sum", "Temperature": "mean"}
            )
            df["Energy"] = df["Power"] * 0.25 / 1000  # Convert to kWh
            df["Date"] = df.index.date
            df["DayofWeek"] = df.index.dayofweek
            df["DayofWeek"] = df["DayofWeek"].apply(lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x])

            # Calcolo del 15° percentile
            threshold = df["Energy"].quantile(percentile)

            # Dividi i punti sopra e sotto il 15° percentile
            df_above = df[df["Energy"] > threshold]
            df_below = df[df["Energy"] <= threshold]

            df_above = df_above.sort_values(by="Temperature").reset_index(drop=True)
            if type == "png":
                plt.figure(figsize=(10, 6))
                plt.scatter(df_below["Temperature"], df_below["Energy"], color="lightgray",
                            label=f"Below {percentile * 100}th percentile")
                plt.scatter(df_above["Temperature"], df_above["Energy"], color="C0",
                            label=f"Above {percentile * 100}th percentile")
                plt.axhline(threshold, color="indianred", linestyle="-", linewidth=2,
                            label=f"{int(percentile * 100)}th percentile")

                try:
                    df_segments = pd.read_csv(
                        os.path.join(PROJECT_ROOT, "results", case_study, "prova", "daily_thermal_sensitivity",
                                     f"segs_{leaf}.csv"))
                    change_points = df_segments["end_idx"][:-1]
                    temps_cp = [f"{df_above.iloc[int(cp)]['Temperature']:.2f}°C" for cp in change_points]
                    label = "Change points: " + ", ".join(temps_cp)
                    for i, cp in enumerate(change_points):
                        plt.axvline(df_above.iloc[int(cp)]["Temperature"], color="blue", linewidth=2,
                                    label=label if i == 0 else None)

                except FileNotFoundError:
                    print(f"⚠️  Change points non trovati per {leaf}, eseguire prima 'calc_thermal_anm_prova'")

                plt.xlabel("Mean temperature [°C]")
                plt.ylabel("Daily energy [kWh]")
                plt.title(f"{leaf}")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f"thermal_sens_{leaf}.png"), dpi=300)
                plt.close()
            elif type == "html":
                fig = make_subplots(rows=1, cols=1)

                fig.add_trace(go.Scatter(
                    x=df_below["Temperature"],
                    y=df_below["Energy"],
                    mode='markers',
                    marker=dict(color='lightgray'),
                    text=df_below["Date"].astype(str) + " " + df_below["DayofWeek"],
                    name="Below 15th percentile"
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=df_above["Temperature"],
                    y=df_above["Energy"],
                    mode='markers',
                    text=df_above["Date"].astype(str) + " " + df_above["DayofWeek"],
                    name=leaf
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=[df["Temperature"].min(), df["Temperature"].max()],
                    y=[threshold, threshold],
                    mode='lines',
                    line=dict(color='indianred', dash='solid', width=2),
                    name=f'{percentile * 100}th percentile'
                ), row=1, col=1)

                fig.update_xaxes(title_text="Mean temperature [°C]", row=1, col=1)
                fig.update_yaxes(title_text="Daily energy [kWh]", row=1, col=1)

                fig.update_traces(
                    hovertemplate="%{text}<br>Temperature: <b>%{x:.2f}</b> °C<br>Energy: <b>%{y:.2f}</b> kWh<extra></extra>",
                    selector=dict(mode='markers')
                )

                fig.update_layout(
                    title=f"{leaf}",
                    template='plotly_white',
                    hovermode="x unified",
                    title_x=0.5,
                    height=700,
                    title_font=dict(size=22),
                    xaxis_tickfont=dict(size=14),
                    yaxis_tickfont=dict(size=14)
                )
                fig.write_html(os.path.join(output_folder, f"thermal_sens_{leaf}.html"),include_plotlyjs="cdn")


if __name__ == "__main__":
    plot_thermal_sensitivity("Cabina", 0.60, "png")
