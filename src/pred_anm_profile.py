from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error
from src.utils import *
from settings import PROJECT_ROOT
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest


def run_dataset(case_study: str):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost")
    os.makedirs(output_dir, exist_ok=True)

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

        df_all = []

        E_max_leaf = 0
        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                df_normals, _ = run_energy_temp(case_study, leaf, context, cluster)
                if df_normals is not None and "Energy" in df_normals.columns:
                    E_max_leaf = max(E_max_leaf, df_normals["Energy"].max(skipna=True))
        energy_cutoff = soglia_en_max * E_max_leaf

        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                df_normal, df_anomalous = run_energy_temp_profile(case_study, leaf, context, cluster)
                df_normal["anm"] = False
                if df_anomalous is not None and not df_anomalous.empty:
                    df_anomalous["anm"] = True
                    df_all_ctx_cl = pd.concat([df_normal, df_anomalous])
                else:
                    df_all_ctx_cl = df_normal
                df_all_ctx_cl["Status"] = df_all_ctx_cl["Energy"] >= energy_cutoff
                df_all.append(df_all_ctx_cl)
        df_all = pd.concat(df_all, ignore_index=False)
        df_all.to_csv(os.path.join(output_dir, f"df_train_{leaf}.csv"))
        print(f"ok")
    return

def run_model(case_study: str):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)
    output_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost")
    os.makedirs(output_path, exist_ok=True)

    levels = get_nodes_by_level(config["Load Tree"])
    first_level = levels[0]
    for leaf in first_level:
        df_train_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", f"df_train_{leaf}.csv")
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
              f"MAPE: {mape:.2f}%")

        model.fit(X, y)
        output_model = os.path.join(output_path, "models")
        os.makedirs(output_model, exist_ok=True)
        model.save_model(os.path.join(output_model, f"model_{leaf }.json"))

        df_results = pd.DataFrame(results)
        output_folder = os.path.join(output_model, "metrics")
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"metrics_model_{leaf}.csv")
        df_results.to_csv(output_file, index=False)
        print(f"✅ Modello salvato\n")

if __name__ == "__main__":
    # run_dataset("Cabina")
    run_model("Cabina")