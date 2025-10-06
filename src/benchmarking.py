import numexpr
numexpr.set_num_threads(1)
import os
import pandas as pd
import json
from settings import PROJECT_ROOT

from src.utils import get_nodes_by_level, assign_values_by_depth, find_leaf_nodes

def cmp_vs_tool(case_study: str):
    """
    Compares anomalies detected by the CMP method with those detected by the Bayesian network method for a case study.

    The function:
    - Loads the case study configuration and inference results.
    - Retrieves detection thresholds from the levels of the load tree.
    - For each node in the network, counts the number of anomalies detected by CMP and by the proposed method,
      and calculates the number of anomalies common to both methods.
    - Calculates the percentage change between anomalies detected by the proposed method compared to CMP.
    - Generates and saves a summary CSV file with the comparison for each node.

    Input:
    case_study : str
        Name of the case study on which to perform the method comparison.

    Output:
    df_summary : pandas.DataFrame
        DataFrame containing the comparison between CMP and the proposed method, including columns for
        threshold, number of anomalies detected by each, common anomalies, and percentage change.
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    inference_path = os.path.join(PROJECT_ROOT, "results", case_study, "inference_results.csv")
    anomaly_table_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")
    summary_path = os.path.join(PROJECT_ROOT, "results", case_study, "benchmarking")

    os.makedirs(summary_path, exist_ok=True)

    inference_df = pd.read_csv(inference_path)
    threshold_dict = assign_values_by_depth(config['Load Tree'])

    results = []

    levels = get_nodes_by_level(config["Load Tree"])
    all_nodes = [node for level in levels for node in level]
    for node in all_nodes:
        anomaly_table_node = pd.read_csv(os.path.join(anomaly_table_path, f"anomaly_table_{node}.csv"))
        threshold = threshold_dict[node]

        n_cmp = len(anomaly_table_node)

        inference_high = inference_df[inference_df[f'P({node}=1)'] >= threshold]
        n_inference_high = len(inference_high)

        anomaly_pairs = set(zip(anomaly_table_node["Date"], anomaly_table_node["Context"]))
        inference_high_pairs = set(zip(inference_high["Date"], inference_high["Context"]))
        n_common = len(anomaly_pairs & inference_high_pairs)

        # Calcolo variazione percentuale
        if n_cmp == 0:
            change = float('inf') if n_inference_high > 0 else 0
        else:
            change = round((n_inference_high - n_cmp) / n_cmp * 100, 2)

        results.append({
            "Load": node,
            "Threshold": threshold * 100,
            "CMP": n_cmp,
            "Proposed Approach": n_inference_high,
            "Common Anomalies": n_common,
            "Detection Shift": f"{change:.2f}%" if change != float('inf') else "-"
        })

    df_summary = pd.DataFrame(results)
    df_summary["Threshold"] = df_summary["Threshold"].map(lambda x: f"{x:.0f}%")
    df_summary.to_csv(os.path.join(summary_path, "cmp_vs_tool_summary.csv"), index=False)
    return df_summary

def calc_year_energy(case_study: str):
    """
    Calculates and prints the total energy consumption (in kWh) for each leaf node and the sum of all leaves
    for a given case study based on timestamped energy data.

    The function:
    - Loads the case study configuration and identifies the leaf nodes in the load tree.
    - For each leaf, reads the corresponding CSV file containing timestamped energy values.
    - Converts timestamps to datetime and calculates the total energy by summing the values multiplied by 0.25.
    - Prints the energy consumption per leaf and accumulates the total energy for all leaves.
    - Additionally, reads and filters the "Total" energy CSV for summer months (June, July, August),
      calculates the total energy during this period, and prints the result.

    Input:
    case_study : str
        Name of the case study containing the energy data files.

    Output:
    Prints energy consumption per leaf, total energy across leaves, and summer total energy from the "Total" data.
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json")) as f:
        config = json.load(f)

    total_waste = 0
    leaves = find_leaf_nodes(config['Load Tree'])
    energy_total_leaf = 0
    for leaf in leaves:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{leaf}.csv"))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # df = df[df["timestamp"].dt.month.isin([6, 7, 8])]
        energy_leaf = (df['value']*0.25).sum()
        print(f"{leaf}: {round(energy_leaf, 2)} kWh")

        energy_total_leaf += energy_leaf

    print(energy_total_leaf)

    df_total = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"Total.csv"))
    df_total["timestamp"] = pd.to_datetime(df_total["timestamp"])
    df_total = df_total[df_total["timestamp"].dt.month.isin([6, 7, 8])]

    print((df_total['value']*0.25).sum())


if __name__ == "__main__":
    case_study = "Total"

    cmp_vs_tool(case_study)
    calc_year_energy(case_study)