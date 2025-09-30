import os
import json
import pandas as pd
from loguru import logger
import numpy as np

from utils import print_boxed_title
from settings import PROJECT_ROOT
from src.cmp.cmp import cmp_calculation


def run_cmp(case_study: str, groups_and_tw_case_study: str = None):
    """
    Run the CMP calculation for a given case study. Save the results in the result folder of the case study.
    Args:
        case_study (str): The name of the case study to process.
        groups_and_tw_case_study (str, optional): The name of the case study to which the time windows and groups refer.
            If None, defaults to the value of `case_study`.
    Returns:
        None
    """
    print_boxed_title("CMP calculation 🧮")
    # Load the configuration file
    with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
        config = json.load(f)

    group_tw_path = groups_and_tw_case_study if groups_and_tw_case_study else case_study

    # Load the data
    load_tree = config["Load Tree"]
    groups = pd.read_csv(os.path.join(PROJECT_ROOT, "results", group_tw_path, "groups.csv"), index_col=0, parse_dates= True)
    time_windows = pd.read_csv(os.path.join(PROJECT_ROOT, "results", group_tw_path, "time_windows.csv"))

    def traverse_tree(tree, function, groups, time_windows, results=None, level=0):
        if results is None:
            results = {}

        for key, subtree in tree.items():
            if os.path.exists(os.path.join(PROJECT_ROOT, "data", case_study, f"{key}.csv")):
                logger.info(f"Running CMP for {key}")

                df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{key}.csv"), index_col=0, parse_dates=True)

                df_dates = set(pd.to_datetime(df.index.date))
                group_dates = set(pd.to_datetime(groups.index.date))

                common_dates = df_dates & group_dates
                common_dates = set(d.date() for d in common_dates)

                df_filtered = df[[d.date() in common_dates for d in df.index]].copy()
                groups_filtered = groups[[d.date() in common_dates for d in groups.index]].copy()

                logger.info(f"{key}: filtered df → {len(df_filtered)} records ({int(len(df_filtered)/96)}/{int(len(df)/96)} days) | groups → {len(groups_filtered)}/{len(groups)}days")

                result = function(data=df_filtered, groups=groups_filtered, time_windows=time_windows)
                if isinstance(result, pd.DataFrame):
                    results[key] = result

            if isinstance(subtree, dict) and subtree:
                traverse_tree(subtree, function, groups, time_windows, results, level=level + 1)
        return results

    # Run the CMP calculation
    results_dict = traverse_tree(load_tree, cmp_calculation, groups=groups, time_windows=time_windows)

    # Save the results
    output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")
    os.makedirs(output_dir, exist_ok=True)
    for name, df in results_dict.items():
        df.to_csv(os.path.join(output_dir, f"anomaly_table_{name}.csv"),
                  index=False)


if __name__ == "__main__":
    run_cmp("Total_cut")