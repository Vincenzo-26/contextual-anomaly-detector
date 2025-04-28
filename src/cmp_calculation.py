import os
import json
import pandas as pd
from loguru import logger

from settings import PROJECT_ROOT
from src.cmp.cmp import cmp_calculation


def run_cmp(case_study: str, groups_tw_case_study: str = None):
    """
    Run the CMP calculation for a given case study. Save the results in the result folder of the case study.
    Args:
        case_study (str): The name of the case study to process.
        groups_tw_case_study (str, optional): The name of the case study to which the time windows and groups refer.
            If None, defaults to the value of `case_study`.
    Returns:
        None
    """

    # Load the configuration file
    with open(os.path.join(PROJECT_ROOT, "data", case_study, f"config.json"), "r") as f:
        config = json.load(f)

    group_tw_path = groups_tw_case_study if groups_tw_case_study else case_study

    # Load the data
    load_tree = config["Load Tree"]
    groups = pd.read_csv(os.path.join(PROJECT_ROOT, "results", group_tw_path, "groups.csv"), index_col=0)
    time_windows = pd.read_csv(os.path.join(PROJECT_ROOT, "results", group_tw_path, "time_windows.csv"))

    def traverse_tree(tree, function, results=None, level=0, **kwargs):
        if results is None:
            results = {}

        for key, subtree in tree.items():
            is_leaf = isinstance(subtree, dict) and len(subtree) == 0

            # Esegui CMP solo se è il primo nodo (level 0) o è un nodo foglia
            if level == 0 or is_leaf:
                logger.info(f"Running CMP for {key}")
                df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{key}.csv"), index_col=0,
                                 parse_dates=True)
                result = function(data=df, **kwargs)
                if isinstance(result, pd.DataFrame):
                    results[key] = result

            if isinstance(subtree, dict) and subtree:
                traverse_tree(subtree, function, results, level=level + 1, **kwargs)

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
    run_cmp("AuleP", "Cabina")