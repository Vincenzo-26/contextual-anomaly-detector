import os
import json
import pandas as pd
from loguru import logger

from settings import PROJECT_ROOT
from src.cmp.cmp import cmp_calculation


def run_cmp(case_study: str):
    """
    Run the CMP calculation for a given case study. Save the results in the result folder of the case study.
    Args:
        case_study (str): The name of the case study to process.
    Returns:
        None
    """

    # Load the configuration file
    with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
        config = json.load(f)

    # Load the data
    load_tree = config["Load Tree"]
    groups = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv"), index_col=0)
    time_windows = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv"))

    def traverse_tree(tree, function, results=None, **kwargs):
        if results is None:
            results = {}

        for key, subtree in tree.items():
            logger.info(f"Running CMP for {key}")
            df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{key}.csv"), index_col=0,
                             parse_dates=True)
            result = function(data=df, groups=groups, time_windows=time_windows)
            if isinstance(result, pd.DataFrame):
                results[key] = result
            if isinstance(subtree, dict) and subtree:
                traverse_tree(subtree, function, results, **kwargs)

        return results

    # Run the CMP calculation
    results_dict = traverse_tree(load_tree, cmp_calculation, groups=groups, time_windows=time_windows)

    # Save the results
    for name, df in results_dict.items():
        df.to_csv(os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_tables", f"anomaly_table_{name}.csv"),
                  index=False)


if __name__ == "__main__":
    run_cmp("AuleR")