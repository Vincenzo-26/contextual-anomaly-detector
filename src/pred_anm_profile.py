from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error
from src.utils import *
from settings import PROJECT_ROOT
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import Booster
import xgboost as xgb


def run_dataset(case_study: str):
    print("Creating dataset for prediction XGboost model...\n")
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    output_dir_train = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "train_df")
    output_dir_test = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "test_df")
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)

    levels = get_nodes_by_level(config["Load Tree"])
    first_level = levels[0]

    groups_path = os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv")
    groups = pd.read_csv(groups_path, parse_dates=["timestamp"])
    groups["date"] = groups["timestamp"].dt.date
    context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
    cluster_cols = [col for col in groups.columns if col.startswith("Cluster_")]
    soglia_en_max = 0.05



    for leaf in first_level:
        print(f"{leaf}...   ", end="")

        E_max_leaf = 0
        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                df_normals, _ = run_energy_temp(case_study, leaf, context, cluster)
                # escludere gli ultimi 2 mesi
                df_normals.index = pd.to_datetime(df_normals.index)
                last_two = df_normals.index.to_period("M").unique().sort_values()[-2:]
                df_normals = df_normals[~df_normals.index.to_period("M").isin(last_two)]

                E_max_leaf = max(E_max_leaf, df_normals["Energy"].max(skipna=True))
        energy_cutoff = soglia_en_max * E_max_leaf

        df_all = []
        df_all_test = []
        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                df_normal, df_anomalous = run_energy_temp_profile(case_study, leaf, context, cluster)
                df_normal.index = pd.to_datetime(df_normal.index)
                df_anomalous.index = pd.to_datetime(df_anomalous.index)

                df_normal_test = df_normal[df_normal.index.to_period("M").isin(last_two)]
                df_anomalous_test = df_anomalous[df_anomalous.index.to_period("M").isin(last_two)]

                df_normal = df_normal[~df_normal.index.to_period("M").isin(last_two)]
                df_anomalous = df_anomalous[~df_anomalous.index.to_period("M").isin(last_two)]

                df_normal["anm"] = False
                df_normal_test["anm"] = False
                if df_anomalous is not None and not df_anomalous.empty:
                    df_anomalous["anm"] = True
                    df_all_ctx_cl = pd.concat([df_normal, df_anomalous])
                else:
                    df_all_ctx_cl = df_normal
                if df_anomalous_test is not None and not df_anomalous.empty:
                    df_anomalous_test["anm"] = True
                    df_all_ctx_cl_test = pd.concat([df_normal_test, df_anomalous_test])
                else:
                    df_all_ctx_cl_test = df_normal_test

                df_all_ctx_cl["Status"] = df_all_ctx_cl["Energy"] >= energy_cutoff
                df_all_ctx_cl_test["Status"] = df_all_ctx_cl_test["Energy"] >= energy_cutoff
                df_all.append(df_all_ctx_cl)
                df_all_test.append(df_all_ctx_cl_test)
        df_all = pd.concat(df_all, ignore_index=False)
        df_all_test = pd.concat(df_all_test, ignore_index=False)
        df_all.to_csv(os.path.join(output_dir_train, f"df_train_{leaf}.csv"))
        df_all_test.to_csv(os.path.join(output_dir_test, f"df_test_{leaf}.csv"))
        print(f"ok")
    return

def run_model(case_study: str):
    print("Creating profile prediction models...\n")
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)
    output_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost")
    os.makedirs(output_path, exist_ok=True)

    levels = get_nodes_by_level(config["Load Tree"])
    first_level = levels[0]
    for leaf in first_level:
        df_train_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "train_df", f"df_train_{leaf}.csv")
        if not os.path.exists(df_train_path):
            run_dataset(case_study)
        df_train = pd.read_csv(os.path.join(df_train_path), parse_dates=["Date"])
        df_train.set_index('Date', inplace=True)
        df_train = df_train[df_train['Energy'] > 0]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        results = []

        df_train = df_train[(df_train['anm'] == False)].copy()

        y = df_train["Energy"].values
        X = df_train.drop(columns=["Energy", "anm"])
        X["Status"] = X["Status"].astype(bool)

        model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)

        mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

        y_preds_all = np.zeros_like(y)
        for train_idx, test_idx in kf.split(X):
            model.fit(X.iloc[train_idx], y[train_idx])
            y_preds_all[test_idx] = model.predict(X.iloc[test_idx])

        mape = mean_absolute_percentage_error(y, y_preds_all) * 100  # in %

        results.append({
            "MAE [kWh]": np.mean(mae_scores),
            "RMSE [kWh]": np.mean(rmse_scores),
            "R² [-]": np.mean(r2_scores),
            "MAPE [%]": mape
        })

        print(f"[{leaf}]  MAE: {np.mean(mae_scores):.2f} kWh  "
              f"RMSE: {np.mean(rmse_scores):.2f} kWh  "
              f"R²: {np.mean(r2_scores):.2f}  "
              f"MAPE: {mape:.2f}%   -   ", end="")

        model.fit(X, y)
        output_model = os.path.join(output_path, "models")
        os.makedirs(output_model, exist_ok=True)
        model.save_model(os.path.join(output_model, f"model_{leaf }.json"))

        df_results = pd.DataFrame(results)
        output_folder = os.path.join(output_model, "metrics")
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"metrics_model_{leaf}.csv")
        df_results.to_csv(output_file, index=False)
        print(f"Model saved ✅")

def run_profile(case_study: str, leaf: str, date: str, context: int):
    print(f"Predicting energy profile (kWh) for:\n{case_study} - {leaf} - {date} - ctx{context}\n")

    output_folder_viz = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "Pred_XGboost")
    os.makedirs(output_folder_viz, exist_ok=True)

    df_test_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "test_df", f"df_test_{leaf}.csv")
    df_test = pd.read_csv(os.path.join(df_test_path), parse_dates=["Date"])
    df_test.set_index('Date', inplace=True)

    df_train_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "train_df", f"df_train_{leaf}.csv")
    df_train = pd.read_csv(os.path.join(df_train_path), parse_dates=["Date"])
    df_train.set_index('Date', inplace=True)

    model_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "models", f"model_{leaf}.json")
    model = Booster()
    model.load_model(model_path)

    # df_day_ctx = df_test[(df_test.index.date == pd.to_datetime(date).date()) & (df_test["Context"] == context)].copy()
    df_day_ctx = df_train[(df_train.index.date == pd.to_datetime(date).date()) & (df_train["Context"] == context)].copy()
    if df_day_ctx.empty:
        print(f"⚠️ No data for {leaf} in {date} - ctx{context}")
        return None
    df_day_ctx = df_day_ctx.sort_values(by=["ora", "quartodora"])

    pred_profile = []
    real_profile = []
    for _, row in df_day_ctx.iterrows():
        row_input = row.drop(labels=["Energy", "anm"], errors="ignore").to_frame().T
        row_input = row_input.apply(pd.to_numeric, errors="coerce")
        if "Status" in row_input.columns:
            row_input["Status"] = row_input["Status"].astype(bool)

        dmatrix = xgb.DMatrix(row_input)
        pred = model.predict(dmatrix)[0]

        pred_profile.append(round(pred, 3))
        real_profile.append(round(row["Energy"], 3))

    print(f"📈 Real profile (kWh): {real_profile}")
    print(f"🔮 Importing model_{leaf}...   ", end="")
    print(f"Predicted profile (kWh): {pred_profile}")

    real_total = round(sum(real_profile), 2)
    pred_total = round(sum(pred_profile), 2)
    difference = round(real_total - pred_total, 2)
    print(f"\nActual total energy: {real_total} kWh")
    print(f"Predicted total energy: {pred_total} kWh")
    if difference > 0:
        print(f"Wasted energy: {difference} kWh 🗑️")
    else:
        print(f"Saved energy: {difference} kWh ♻️")

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
                alpha=0.5,
                linewidth=1
            )

    x_labels = [f"{h:02d}:{m * 15:02d}" for h, m in zip(df_day_ctx["ora"], df_day_ctx["quartodora"])]

    plt.plot(x_labels, real_profile, label="Reale", color="blue", marker="o", alpha=0.6)
    plt.plot(x_labels, pred_profile, label="Predetto", color="red", linestyle="--", marker="x")

    plt.title(f"Profilo energetico - {leaf} - {date} (cls {clt}) - ctx{context}")
    plt.xlabel("Orario")
    plt.ylabel("Energia [kWh]")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # run_dataset("Cabina")
    # run_model("Cabina")
    run_profile("Cabina", "Rooftop 4", "2025-03-01", 2)