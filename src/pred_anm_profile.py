import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

from xgboost import XGBRegressor, Booster
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

from src.utils import run_energy_temp, run_energy_temp_profile, get_nodes_by_level
from settings import PROJECT_ROOT

def run_dataset(case_study: str):
    print("Creating dataset for prediction XGboost model...\n")
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    output_dir_train = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "train_df")
    output_dir_val = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "val_df")
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_val, exist_ok=True)

    inference_path = os.path.join(PROJECT_ROOT, "results", case_study, "inference_results.csv")
    inference_df = pd.read_csv(inference_path)

    groups_path = os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv")
    groups = pd.read_csv(groups_path, parse_dates=["timestamp"])
    groups["date"] = groups["timestamp"].dt.date
    time_windows_df = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv"))

    levels = get_nodes_by_level(config["Load Tree"])
    all_nodes = [node for level in levels for node in level]
    for leaf in all_nodes:
        print(f"{leaf}...   ", end="")

        # Calcola la soglia come il 10% del massimo consumo mediato sul numero di quarti d'ora del contesto tra tutti i context e cluster.
        soglia_en_max = 0.10
        E_max_leaf = 0
        for context in time_windows_df.id.unique():
            obs = time_windows_df.loc[time_windows_df['id'] == context, 'observations'].values[0]
            for cluster_col in [col for col in groups.columns if col.startswith("Cluster_")]:
                cluster = int(cluster_col.split("_")[-1])
                df_normals, _ = run_energy_temp(case_study, leaf, context, cluster)
                max_energy = df_normals["Energy"].max(skipna=True)
                avg_energy = max_energy / obs
                E_max_leaf = max(E_max_leaf, avg_energy)
        energy_cutoff = soglia_en_max * E_max_leaf

        df_all_normal = []
        df_all_anomalous = []
        for context in time_windows_df.id.unique():
            for cluster_col in [col for col in groups.columns if col.startswith("Cluster_")]:
                cluster = int(cluster_col.split("_")[-1])
                df_normal, df_anomalous = run_energy_temp_profile(case_study, leaf, context, cluster)
                df_normal["Status"] = df_normal["Energy"] >= energy_cutoff
                df_all_normal.append(df_normal)

                if df_anomalous is not None and not df_anomalous.empty:
                    df_anomalous["Status"] = df_anomalous["Energy"] >= energy_cutoff
                    df_all_anomalous.append(df_anomalous)

        df_all_normal = pd.concat(df_all_normal, ignore_index=False)
        df_all_anomalous = pd.concat(df_all_anomalous, ignore_index=False)

        probs_df = inference_df[["Date", "Context", f'P({leaf}=1)']].copy()
        probs_df.columns = ["Date", "Context", "anm_BN"]
        probs_df["Date"] = pd.to_datetime(probs_df["Date"])

        df_train_norm = df_all_normal.merge(probs_df, how="left", on=["Date", "Context"])
        df_train_norm.set_index("Date", inplace=True)

        df_val_anm = pd.DataFrame()
        if df_all_anomalous is not None and not df_all_anomalous.empty:
            df_val_anm = df_all_anomalous.merge(probs_df, how="left", on=["Date", "Context"])
            df_val_anm.set_index("Date", inplace=True)

        df_train_norm.to_csv(os.path.join(output_dir_train, f"df_train_{leaf}.csv"))
        if not df_val_anm.empty:
            df_val_anm.to_csv(os.path.join(output_dir_val, f"df_val_{leaf}.csv"))
        print(f"ok")
    return


def min_max_scaling(array, reverse=False, min_val=None, max_val=None):
    if not reverse:
        min_val = np.min(array)
        max_val = np.max(array)
        scaled = (array - min_val) / (max_val - min_val)
        return scaled, min_val, max_val
    else:
        return array * (max_val - min_val) + min_val
def run_model(case_study: str):
    print("Creating profile prediction models...\n")
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)
    train_dir = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "train_df")
    model_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "models")
    os.makedirs(model_path, exist_ok=True)

    metrics_results = []
    scaling_info = []

    levels = get_nodes_by_level(config["Load Tree"])
    all_nodes = [node for level in levels for node in level]
    for leaf in all_nodes:
        df_train_path = os.path.join(train_dir, f"df_train_{leaf}.csv")
        df_train = pd.read_csv(df_train_path, parse_dates=["Date"])
        if 'Date' in df_train.columns:
            df_train.set_index('Date', inplace=True)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        y_raw = df_train["Energy"].values
        y, y_min, y_max = min_max_scaling(y_raw)
        scaling_info.append({"Load": leaf, "Energy_min": y_min, "Energy_max": y_max})
        X = df_train.drop(columns=["Energy", "anm", "anm_BN"])
        X["Status"] = X["Status"].astype(bool)

        model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)

        y_preds_all = np.zeros_like(y)
        mae_scores = []
        rmse_scores = []
        r2_scores = []

        # cross validation
        for train_idx, test_idx in kf.split(X):
            model.fit(X.iloc[train_idx], y[train_idx])
            preds = model.predict(X.iloc[test_idx])
            y_preds_all[test_idx] = preds

            preds_rescaled = min_max_scaling(preds, reverse=True, min_val=y_min, max_val=y_max)
            y_true_rescaled = min_max_scaling(y[test_idx], reverse=True, min_val=y_min, max_val=y_max)

            mae_scores.append(mean_absolute_error(y_true_rescaled, preds_rescaled))
            rmse_scores.append(np.sqrt(mean_squared_error(y_true_rescaled, preds_rescaled)))
            r2_scores.append(r2_score(y_true_rescaled, preds_rescaled))

        metrics_results.append({
            "Load": leaf,
            "MAE [kWh]": np.mean(mae_scores),
            "RMSE [kWh]": np.mean(rmse_scores),
            "R² [-]": np.mean(r2_scores),
            # "MAPE [%]": mape
        })

        print(f"[{leaf}]  MAE: {np.mean(mae_scores):.2f} kWh  "
              f"RMSE: {np.mean(rmse_scores):.2f} kWh  "
              f"R²: {np.mean(r2_scores):.2f}", end="")
              # f"MAPE: {mape:.2f}%   -   ", end="")

        model.fit(X, y)
        model.save_model(os.path.join(model_path, f"model_{leaf}.json"))

        print("Model saved ✅")

    df_all_results = pd.DataFrame(metrics_results)
    df_all_results.to_csv(os.path.join(model_path, "metrics.csv"), index=False)
    df_scaling = pd.DataFrame(scaling_info)
    df_scaling.to_csv(os.path.join(model_path, "scaling_minmax_info.csv"), index=False)
    print("\n📊 metrics.csv saved.")
    print("📉 energy_min_max.csv salvato correttamente.")



def run_profile(case_study: str, leaf: str, date: str, context: int, which_df: str, plot_pred: bool):
    print(f"Predicting energy profile (kWh) for:\n{case_study} - {leaf} - {date} - ctx{context}\n")

    output_folder_viz = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "Pred_XGboost")
    os.makedirs(output_folder_viz, exist_ok=True)

    df_val_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "val_df", f"df_val_{leaf}.csv")
    df_val = pd.read_csv(os.path.join(df_val_path), parse_dates=["Date"])
    df_val.set_index('Date', inplace=True)

    df_train_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "train_df", f"df_train_{leaf}.csv")
    df_train = pd.read_csv(os.path.join(df_train_path), parse_dates=["Date"])
    df_train.set_index('Date', inplace=True)

    scaling_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "models", "scaling_minmax_info.csv")
    df_scaling = pd.read_csv(scaling_path)
    row_scaling = df_scaling[df_scaling["Load"] == leaf]
    if row_scaling.empty:
        print(f"⚠️ Nessun valore di scaling trovato per {leaf}")
        return None
    y_min = row_scaling["Energy_min"].values[0]
    y_max = row_scaling["Energy_max"].values[0]

    model_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "models", f"model_{leaf}.json")
    model = Booster()
    model.load_model(model_path)

    if which_df == "test":
        df_day_ctx = df_val[(df_val.index.date == pd.to_datetime(date).date()) & (df_val["Context"] == context)].copy()
    elif which_df == "train":
        df_day_ctx = df_train[(df_train.index.date == pd.to_datetime(date).date()) & (df_train["Context"] == context)].copy()

    if df_day_ctx.empty:
        print(f"⚠️ No data for {leaf} in {date} - ctx{context}")
        return None
    df_day_ctx = df_day_ctx.sort_values(by=["ora", "quartodora"])

    pred_profile = []
    real_profile = []
    for _, row in df_day_ctx.iterrows():
        row_input = row.drop(labels=["Energy", "anm","anm_BN"], errors="ignore").to_frame().T
        row_input = row_input.apply(pd.to_numeric, errors="coerce")
        if "Status" in row_input.columns:
            row_input["Status"] = row_input["Status"].astype(bool)


        dmatrix = xgb.DMatrix(row_input)
        pred_scaled = model.predict(dmatrix)[0]
        pred_rescaled = max(pred_scaled * (y_max - y_min) + y_min, 0)
        pred_profile.append(round(pred_rescaled, 3))
        real_profile.append(round(row["Energy"], 3))

    print(f"📈 Real profile (kWh): {real_profile}")
    print(f"🔮 Importing model_{leaf}...   ", end="")
    print(f"Predicted profile (kWh): {pred_profile}")


    profile_difference = np.array(real_profile) - np.array(pred_profile)
    profile_difference = profile_difference[profile_difference > 0]
    difference = round(np.sum(profile_difference), 2) # differenza punto a punto misura quando è maggiore, ignorando quando pred>real

    real_total = round(sum(real_profile), 2)
    pred_total = round(sum(pred_profile), 2)

    print(f"\nCMP   -> Anomaly? {df_day_ctx['anm'].iloc[0]}")
    print(f"BN prob    -> {df_day_ctx['anm_BN'].iloc[0] * 100:.2f}%")
    print(f"\nActual total energy: {real_total} kWh")
    print(f"Predicted total energy: {pred_total} kWh")
    print(f"Wasted energy: {difference} kWh ♻️")

    plt.figure(figsize=(10, 6))
    clt = int(row_input["Cluster"].iloc[0])
    df_train_ctx = df_train[(df_train["Context"] == context) & (df_train["Cluster"] == clt)].copy()
    unique_dates = df_train_ctx.index.normalize().unique()

    for d in unique_dates:
        df_day = df_train_ctx[df_train_ctx.index.normalize() == d].sort_values(by=["ora", "quartodora"])
        if len(df_day) > 0:
            plt.plot(
                [f"{h:02d}:{m * 15:02d}" for h, m in zip(df_day["ora"], df_day["quartodora"])],
                df_day["Energy"],
                color="gray",
                alpha=0.3,
                linewidth=0.6
            )

    x_labels = [f"{h:02d}:{m * 15:02d}" for h, m in zip(df_day_ctx["ora"], df_day_ctx["quartodora"])]
    plt.plot(x_labels, real_profile, label="Actual", color="#4682B4", marker="o", markersize=5, linewidth=1.5)
    if plot_pred:
        plt.plot(x_labels, pred_profile, label="Predicted", color="#CD5C5C", marker="o", markersize=5, linewidth=1.5)

    plt.title(f"{leaf} - {date} - ctx{context}", fontsize=18)
    plt.ylabel("Energy [kWh]", fontsize=14)

    step = 2
    tick_positions = list(range(0, len(x_labels), step))
    plt.xticks(ticks=tick_positions, labels=[x_labels[i] for i in tick_positions], rotation=45)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return difference


def calc_wasted_energy(case_study: str, threshold: float, which_df: str):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)
    test_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "test_df")
    train_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "train_df")

    total_wasted_cmp = 0
    total_wasted_bn = 0
    count_cmp = 0
    count_bn = 0

    levels = get_nodes_by_level(config["Load Tree"])
    first_level = levels[0]

    def run_profile1(model, date: str, context: int, df: pd.DataFrame):
        df_day_ctx = df[(df.index.date == pd.to_datetime(date).date()) & (df["Context"] == context)].copy()
        if df_day_ctx.empty:
            print(f"⚠️ No data in {date} - ctx{context}")
            return None
        df_day_ctx = df_day_ctx.sort_values(by=["ora", "quartodora"])

        pred_profile = []
        real_profile = []
        for _, row in df_day_ctx.iterrows():
            row_input = row.drop(labels=["Energy", "anm", "anm_BN"], errors="ignore").to_frame().T
            row_input = row_input.apply(pd.to_numeric, errors="coerce")
            if "Status" in row_input.columns:
                row_input["Status"] = row_input["Status"].astype(bool)

            dmatrix = xgb.DMatrix(row_input)
            pred = model.predict(dmatrix)[0]

            pred_profile.append(round(pred, 3))
            real_profile.append(round(row["Energy"], 3))

        # print(f"📈 Real profile (kWh): {real_profile}")
        # print(f"Predicted profile (kWh): {pred_profile}")

        real_total = round(sum(real_profile), 2)
        pred_total = round(sum(pred_profile), 2)
        difference = round(real_total - pred_total, 2)
        print(f"CMP   -> Anomaly? {row["anm"]}")
        print(f"BN prob    -> {row["anm_BN"] * 100:.2f}%")
        print(f"Actual total energy: {real_total} kWh")
        print(f"Predicted total energy: {pred_total} kWh")
        if difference > 0:
            print(f"Wasted energy: {difference} kWh 🗑️\n")
        else:
            print(f"Saved energy: {difference} kWh ♻️\n")
        return difference

    for leaf in first_level:

        model_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "models", f"model_{leaf}.json")
        model = Booster()
        model.load_model(model_path)

        if which_df == "test":
            df = pd.read_csv(os.path.join(test_path, f"df_test_{leaf}.csv"), parse_dates=["Date"])
        elif which_df == "train":
            df = pd.read_csv(os.path.join(train_path, f"df_train_{leaf}.csv"), parse_dates=["Date"])

        # df = df[(df["mese"] <= 12) & (df["mese"] >= 10)]

        df.set_index('Date', inplace=True)
        df_cmp = df[df["anm"] == True]
        df_bn = df[df["anm_BN"] > threshold]

        # --- CMP ---
        for (date, context), _ in df_cmp.groupby(["Date", "Context"]):
            print(f"\nCMP - {leaf} - {date} - ctx{context}")
            waste = run_profile1(model, str(pd.to_datetime(date).date()), int(context), df_cmp)
            total_wasted_cmp += waste
            count_cmp += 1

        # --- BN ---
        for (date, context), _ in df_bn.groupby(["Date", "Context"]):
            print(f"\nBN - {leaf} - {date} - ctx{context}")
            waste = run_profile1(model, str(pd.to_datetime(date).date()), int(context), df_bn)
            total_wasted_bn += waste
            count_bn += 1

    results = pd.DataFrame({
        "Combinazioni Anomale": [count_cmp, count_bn],
        "Energia Persa [kWh]": [round(total_wasted_cmp, 2), round(total_wasted_bn, 2)]
    }, index=["CMP", "BN"])
    # all_dates = pd.concat([df_cmp["Date"], df_bn["Date"]], ignore_index=True)
    # min_date = pd.to_datetime(all_dates.min()).strftime("%Y-%m-%d")
    # max_date = pd.to_datetime(all_dates.max()).strftime("%Y-%m-%d")
    # print(f"\n📊 Results for [{min_date} - {max_date}]")
    print(f"\n📊 Results:")
    print(results)




if __name__ == "__main__":
    case_study = "Total_cut"
    # run_dataset(case_study)
    # run_model(case_study)
    run_profile(case_study, "GF3", "2024-07-28", 4, "test", True)
    # calc_wasted_energy(case_study, 0.8)


    # "2024-08-20", 1 buono per anomalia esagerata