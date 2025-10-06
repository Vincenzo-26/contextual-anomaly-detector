import numexpr
numexpr.set_num_threads(1)
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from settings import PROJECT_ROOT

"""
Genera e salva barplot dalle Conditional Probability Distributions (CPDs) di un case_study.
Ogni barplot viene salvato con nome "barplot_<nome_nodo>.png" nella output_dir.
"""

case_study = "Total"

input_dir = os.path.join(PROJECT_ROOT, "results", case_study, "CPDs")
output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "CPDs")
os.makedirs(output_dir, exist_ok=True)

pastel_colors = plt.get_cmap('Pastel1')
file_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
n_files = len(file_list)

# numero massimo di colonne per averle della stessa larghezza tra i plot
n_max = 0
for filename in file_list:
    df_tmp = pd.read_csv(os.path.join(input_dir, filename))
    n_cols = len(df_tmp.columns) - 3  # esclusiuone di P=1 e P=0 e observed dalle colonne
    n_max = max(n_max, n_cols)

# Step 2: genera i barplot
for idx, filename in enumerate(file_list):
    filepath = os.path.join(input_dir, filename)
    df = pd.read_csv(filepath)

    node_name = filename.replace("cpd_", "").replace(".csv", "")
    prob_col = f"P({node_name}=1)"
    input_cols = df.columns[:-3]

    bar_labels = []
    bar_values = []

    for col in input_cols:
        mask = (df[col] == 1)
        for other_col in input_cols:
            if other_col != col:
                mask &= (df[other_col] == 0)
        value = df.loc[mask, prob_col].values[0] * 100 if mask.any() else 0
        bar_labels.append(col)
        bar_values.append(value)

    bar_labels += [''] * (n_max - len(bar_labels))
    bar_values += [0] * (n_max - len(bar_values))
    bar_pos = np.arange(n_max)

    plt.figure(figsize=(8, 8))
    plt.bar(bar_pos, bar_values, color=pastel_colors(idx % 8))
    plt.xticks(bar_pos, bar_labels, rotation=45, ha='right', fontsize=18)
    plt.ylim(0, 100)
    plt.ylabel(f"P({node_name}=1) [%]", fontsize=18)
    plt.title(f"{node_name}", fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"barplot_{node_name}.png")
    plt.savefig(save_path)
    plt.close()