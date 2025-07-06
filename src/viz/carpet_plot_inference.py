import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from settings import PROJECT_ROOT


case_study = "Total_cut"
load = "GF5"

results_path = os.path.join(PROJECT_ROOT, "results", case_study, "inference_results.csv")
time_window_path = os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")
df = pd.read_csv(results_path)

# heatmap_data = df.pivot(index="Context", columns="Date", values="P(Total=1)")
heatmap_data = df.pivot(index="Context", columns="Date", values=f"P({load}=1)")
heatmap_data = heatmap_data.astype(float)

plt.figure(figsize=(18, 4))
sns.heatmap(
    heatmap_data,
    cmap="vlag",
    cbar_kws={'label': f'P({load}=1)'},
)

plt.title(f"{load}", fontsize=18)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Context", fontsize=16)
xticks = plt.gca().get_xticks()
xticklabels = plt.gca().get_xticklabels()
plt.xticks(xticks[::4], [label.get_text() for label in xticklabels][::4], rotation=30, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
