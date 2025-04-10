import os
import json
import pandas as pd

from src.cmp.groups_definition import run_clustering
from src.cmp.time_windows_definition import run_cart
from src.cmp.utils import process_data, extract_holidays
from settings import PROJECT_ROOT


def run_groups_and_tw(case_study: str):
    """
    Run the groups and time windows identification process. Save the results in the result folder of the case study.
    Args:
        case_study (str): The name of the case study to process.
    Returns:
        None
    """

    # Load the configuration file
    with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
        config = json.load(f)

    first_level = list(config["Load Tree"].keys())[0]
    holidays = config.get("holidays", None)

    # Load the data
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{first_level}.csv"), index_col=0, parse_dates=True)
    df_clean, _, _ = process_data(df, variable="value")

    if holidays is not None:
        df_holidays = extract_holidays(df_clean, holidays)
    else:
        df_holidays = None

    # Run the clustering algorithm
    groups = run_clustering(df, df_holidays)

    # Run the time windows algorithm
    time_windows = run_cart(df)

    # Save the results
    groups.to_csv(os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv"), index=False)
    time_windows.to_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv"), index=False)


if __name__ == "__main__":
    run_groups_and_tw("AuleR")
