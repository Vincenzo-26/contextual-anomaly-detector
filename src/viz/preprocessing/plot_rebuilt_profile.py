import os
import pandas as pd
import matplotlib.pyplot as plt
from settings import PROJECT_ROOT
import matplotlib.dates as mdates


case_study = "Total"
root_node = "Total"
day_interp = "2024-09-02"
day_knn = "2024-08-11"

df_raw = pd.read_csv(os.path.join(PROJECT_ROOT, "raw_data", case_study, f"{root_node}.csv"))
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
df_day_raw = df_raw[df_raw['timestamp'].dt.date == pd.to_datetime(day_interp).date()]

df_prep = pd.read_csv(os.path.join(PROJECT_ROOT, "data", case_study, f"{root_node}.csv"))
df_prep['timestamp'] = pd.to_datetime(df_prep['timestamp'])
df_day_prep = df_prep[df_prep['timestamp'].dt.date == pd.to_datetime(day_interp).date()]

time_range_interp = pd.date_range(start=pd.to_datetime(day_interp), periods=96, freq='15min')
time_range_knn = pd.date_range(start=pd.to_datetime(day_knn), periods=96, freq='15min')

df_day_raw_full = df_day_raw.set_index('timestamp').reindex(time_range_interp)
df_day_prep_full = df_day_prep.set_index('timestamp').reindex(time_range_interp)

df_day_knn_raw = df_raw[df_raw['timestamp'].dt.date == pd.to_datetime(day_knn).date()]
df_day_knn_prep = df_prep[df_prep['timestamp'].dt.date == pd.to_datetime(day_knn).date()]

df_day_knn_raw_full = df_day_knn_raw.set_index('timestamp').reindex(time_range_knn)
df_day_knn_prep_full = df_day_knn_prep.set_index('timestamp').reindex(time_range_knn)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": True,
    "axes.spines.right": True,
})

fig, axs = plt.subplots(2, 1, figsize=(12, 8))

axs[0].plot(time_range_interp, df_day_prep_full['value'], label="Linear interpolation", color="#e74c3c")
axs[0].plot(time_range_interp, df_day_raw_full['value'], label="Raw profile", color="#1f77b4")
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axs[0].set_title(f"{root_node} {day_interp}")
axs[0].set_ylabel("Power [kW]")
axs[0].legend()
axs[0].legend(frameon=False, fontsize=10)
axs[0].grid(True)
axs[0].grid(True, linewidth=0.5, color='lightgray')

axs[1].plot(time_range_knn, df_day_knn_prep_full['value'], label="KNN imputation", color="#e74c3c")
axs[1].plot(time_range_knn, df_day_knn_raw_full['value'], label="Raw profile", color="#1f77b4")
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axs[1].set_title(f"{root_node} {day_knn}")
axs[1].set_ylabel("Power [kW]")
axs[1].legend()
axs[1].legend(frameon=False, fontsize=10)
axs[1].grid(True)
axs[1].grid(True, linewidth=0.5, color='lightgray')

plt.tight_layout()
plt.show()
