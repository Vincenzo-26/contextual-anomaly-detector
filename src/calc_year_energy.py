import os
import json
import pandas as pd

from utils import find_leaf_nodes
from settings import PROJECT_ROOT

case_study = "Total_cut"
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
