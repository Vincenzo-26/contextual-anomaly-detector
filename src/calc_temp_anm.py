import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import normaltest

from src.utils import run_energy_temp, find_leaf_nodes, print_boxed_title
from settings import PROJECT_ROOT


def run_dataset(case_study: str):
    """
    Costruisce il dataset completo per l'addestramento del modello XGBoost, aggregando dati normali e anomali
    per ciascun contesto e cluster del primo livello del load tree.

    Per ogni combinazione (leaf, context, cluster):
      - calcola il massimo valore di energia tra i profili normali;
      - definisce una soglia pari al 5% dell'energia massima;
      - assegna un'etichetta booleana "Status" che indica se l'energia supera tale soglia;
      - concatena i profili normali e anomali, marcando questi ultimi con la colonna "anm".

    Infine, il dataset viene codificato con one-hot encoding sui nomi dei sottocarichi ("Subload"),
    e viene salvato su file insieme all'elenco delle feature.

    Args:
        case_study (str): Nome del caso studio (cartella contenente dati e configurazioni).

    Returns:
        pd.DataFrame: Dataset completo con etichette di anomalia e feature per il training.
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "temp_XGboost")
    os.makedirs(output_dir, exist_ok=True)



    groups_path = os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv")
    groups = pd.read_csv(groups_path, parse_dates=["timestamp"])
    groups["date"] = groups["timestamp"].dt.date
    context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
    cluster_cols = [col for col in groups.columns if col.startswith("Cluster_")]

    df_all = []
    soglia_en_max = 0.10

    # levels = get_nodes_by_level(config["Load Tree"])
    # first_level = levels[0]
    leaves = find_leaf_nodes(config["Load Tree"])
    for leaf in leaves:
        print(f"{leaf}...   ", end="")

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
                df_normal, df_anomalous = run_energy_temp(case_study, leaf, context, cluster)

                df_normal["anm"] = False
                if df_anomalous is not None and not df_anomalous.empty:
                    df_anomalous["anm"] = True
                    df_all_ctx_cl = pd.concat([df_normal, df_anomalous])
                else:
                    df_all_ctx_cl = df_normal

                df_all_ctx_cl["Status"] = df_all_ctx_cl["Energy"] >= energy_cutoff
                df_all.append(df_all_ctx_cl)

        print(f"ok")
    df_all = pd.concat(df_all, ignore_index=False)

    df_all = pd.get_dummies(df_all, columns=["Subload"], prefix="", prefix_sep="", drop_first=False)
    feature_columns = df_all.drop(columns=["Energy", "anm"]).columns.tolist()
    with open(os.path.join(output_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)

    print("\n")
    df_all.to_csv(os.path.join(output_dir, "df_train.csv"))
    return df_all


def run_model(case_study: str):
    output_path = os.path.join(PROJECT_ROOT, "results", case_study, "temp_XGboost")
    os.makedirs(output_path, exist_ok=True)

    context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
    df_train_path = os.path.join(PROJECT_ROOT, "results", case_study, "temp_XGboost", "df_train.csv")
    # if not os.path.exists(df_train_path):
    run_dataset(case_study)
    df_train_all = pd.read_csv(os.path.join(df_train_path), parse_dates=["Date"])
    df_train_all.set_index('Date', inplace=True)
    df_train_all = df_train_all[df_train_all['Energy'] > 0]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for context in context_ids:
        # train sia su normal sia su anomalie
        df_train_ctx = df_train_all[(df_train_all['Context'] == context)].copy()
        y = np.log1p(df_train_ctx["Energy"].values)
        X = df_train_ctx.drop(columns=["Energy", "anm"])
        X["Status"] = X["Status"].astype(bool)

        model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)

        mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

        y_preds_all = np.zeros_like(y)
        for train_idx, test_idx in kf.split(X):
            model.fit(X.iloc[train_idx], y[train_idx])
            y_preds_all[test_idx] = np.expm1(model.predict(X.iloc[test_idx]))

        mape = mean_absolute_percentage_error(y, y_preds_all) * 100  # in %

        results.append({
            "Context": context,
            "MAE [kWh]": np.mean(mae_scores),
            "RMSE [kWh]": np.mean(rmse_scores),
            "R² [-]": np.mean(r2_scores),
            "MAPE [%]": mape
        })

        print(f"[Context {context}]  MAE: {np.mean(mae_scores):.2f} kWh  "
              f"RMSE: {np.mean(rmse_scores):.2f} kWh  "
              f"R²: {np.mean(r2_scores):.2f}  "
              f"MAPE: {mape:.2f}%")

        model.fit(X, y)
        output_model = os.path.join(output_path, "models")
        os.makedirs(output_model, exist_ok=True)
        model.save_model(os.path.join(output_model, f"model_ctx{context}.json"))

    df_results = pd.DataFrame(results)
    output_file = os.path.join(output_path, "model_results.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\n✅ Models saved in: {output_path}\n")
    return


def calc_temp_anm_prob(case_study: str):
    print_boxed_title("Thermal evidence calculation 📈")
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json")) as f:
        config = json.load(f)

    output_folder_viz = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "thermal_sensitivity", "ctx_thermal_sens", "XGBoost_viz")
    os.makedirs(output_folder_viz, exist_ok=True)
    output_folder_df = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "ctx_thermal_sens")
    os.makedirs(output_folder_df, exist_ok=True)

    context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
    groups = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv"), parse_dates=["timestamp"])
    cluster_cols = [col for col in groups.columns if col.startswith("Cluster_")]


    df_path = os.path.join(PROJECT_ROOT, "results", case_study, "temp_XGboost", "df_train.csv")
    df_all = pd.read_csv(os.path.join(df_path), parse_dates=["Date"])
    df_all.set_index('Date', inplace=True)
    # df_all = df_all[df_all['Energy'] > 0]

    model_folder_path = os.path.join(PROJECT_ROOT, "results", case_study, "temp_XGboost", "models")
    if not os.path.exists(model_folder_path):
        print("Creating XGboost models for each context...")
        run_model(case_study)
    print("Thermal probability calculation...\n")
    # levels = get_nodes_by_level(config["Load Tree"])
    leaves = find_leaf_nodes(config["Load Tree"])
    for leaf in leaves:
        df_leaf_all = []
        df_sens_path = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "daily_thermal_sens", f"segs_{leaf}.csv")
        df_sens = pd.read_csv(df_sens_path, index_col=0)
        if not df_sens["Thermal Sensitive"].any():
            continue

        print(f"\033[91m{leaf}\033[0m")
        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                print(f"[Ctx {context} | Clst {cluster}]...    ", end="")

                df = df_all[(df_all['Context'] == context) & (df_all['Cluster'] == cluster) & (df_all[leaf])].copy()

                if df is None or df.empty:
                    print(f"    ⚠️ Dataframe vuoto")
                df = df.sort_values("Mean_Temp")

                features = df.drop(columns=["Energy", "anm"])
                with open(os.path.join(PROJECT_ROOT, "results", case_study, "temp_XGboost", "feature_columns.json")) as f:
                    training_columns = json.load(f)

                if not all(col in features.columns for col in training_columns):
                    missing = [col for col in training_columns if col not in features.columns]
                    print(f"⚠️ Skipped due to missing columns: {missing}")
                    continue

                features = features[training_columns]

                model_path = os.path.join(model_folder_path, f"model_ctx{context}.json")
                model = XGBRegressor()
                model.load_model(model_path)

                df["y_pred"] = np.expm1(model.predict(features))
                df["residual"] = df["Energy"] - df["y_pred"]

                residuals_normal = df.loc[df["anm"] == False, "residual"]
                if len(residuals_normal) >= 8:  # normaltest richiede almeno 8 dati
                    stat, p_value = normaltest(residuals_normal)
                    is_normal = p_value > 0.05
                    emoji = "✅" if is_normal else "❌"
                else:
                    emoji = "⚠️ dati non sufficienti per la valutazione"

                # calcolo della probabilità di anomalia
                sigma = df.loc[df["anm"] == False, "residual"].std()
                theta = 6.5
                df["anm_prob"] = 1 - np.exp(- (df["residual"] ** 2) / (2 * theta * sigma ** 2))
                # sovrascrivere a 0% per residui negativi
                df.loc[df["residual"] < 0, "anm_prob"] = 0
                print(f"Residual normal distribution {emoji}")

                df_leaf_all.append(df)

                plt.figure(figsize=(10, 6))
                plt.grid(True)

                for status in [False, True]:
                    mask = (df["Status"] == status)
                    if mask.sum() == 0:
                        continue

                    marker = "^" if not status else "o"
                    label = f"{'OFF' if not status else 'ON'}"
                    sc = plt.scatter(
                        df.loc[mask, "Mean_Temp"],
                        df.loc[mask, "Energy"],
                        c=df.loc[mask, "anm_prob"],
                        cmap="coolwarm",
                        vmin=0, vmax=1,
                        marker=marker,
                        edgecolor="black",
                        label=label,
                        alpha=0.8
                    )

                # Colorbar con label e tick più grandi
                cb = plt.colorbar(sc, label="Anomaly probability", shrink=0.8, aspect=20, pad=0.02)
                cb.ax.tick_params(labelsize=12)  # dimensione dei numeri della colorbar
                cb.set_label("Anomaly probability", fontsize=14)

                # Secondo scatter
                plt.scatter(
                    df["Mean_Temp"],
                    df["y_pred"],
                    c='black', marker='x', label='Prediction', alpha=0.6, zorder=1
                )

                # Etichette e titolo
                plt.xlabel("Mean temperature in time window [°C]", fontsize=14)
                plt.ylabel("Energy in time window [kWh]", fontsize=14)
                plt.title(f"{leaf} - Context {context} - Cluster {cluster}", fontsize=16)

                # Tick degli assi
                plt.tick_params(axis='both', labelsize=12)

                # Legenda
                plt.legend(fontsize=12)
                plt.tight_layout()

                filename = f"{leaf}_ctx{context}_cl{cluster}.png"
                filepath_leaf = os.path.join(output_folder_viz, f"{leaf}")
                os.makedirs(filepath_leaf, exist_ok=True)
                filepath = os.path.join(filepath_leaf, filename)
                plt.savefig(filepath)
                plt.close()
        df_leaf_concat = pd.concat(df_leaf_all)
        output_csv_path = os.path.join(output_folder_df, f"{leaf}.csv")
        df_leaf_concat.to_csv(output_csv_path, index=True)
        print("\n")

def plot_predicted_vs_actual(case_study: str):
    output_folder = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "thermal_sensitivity", "ctx_thermal_sens", "actual_vs_pred")
    os.makedirs(output_folder, exist_ok=True)

    df_path = os.path.join(PROJECT_ROOT, "results", case_study, "temp_XGboost", "df_train.csv")
    df_all = pd.read_csv(df_path, parse_dates=["Date"])
    df_all.set_index("Date", inplace=True)
    df_all = df_all[df_all["Energy"] > 0]

    feature_file = os.path.join(PROJECT_ROOT, "results", case_study, "temp_XGboost", "feature_columns.json")
    with open(feature_file) as f:
        feature_columns = json.load(f)

    model_folder = os.path.join(PROJECT_ROOT, "results", case_study, "temp_XGboost", "models")
    context_ids = df_all["Context"].unique()
    colors = sns.color_palette("pastel", n_colors=len(context_ids))

    for idx, context in enumerate(context_ids):
        df_ctx = df_all[df_all["Context"] == context].copy()
        if df_ctx.empty:
            continue

        X = df_ctx[feature_columns]
        y_true = df_ctx["Energy"].values

        model = XGBRegressor()
        model.load_model(os.path.join(model_folder, f"model_ctx{context}.json"))
        y_pred = np.expm1(model.predict(X))

        # Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(
            y_pred, y_true,
            alpha=0.7,
            color=colors[idx],
            edgecolors='white',
            linewidth=0.5
        )

        max_val = max(max(y_pred), max(y_true))
        min_val = min(min(y_pred), min(y_true))
        plt.plot([min_val, max_val], [min_val, max_val], alpha=0.6, color="black")

        plt.xlabel("Predicted energy [kWh]", fontsize=14)
        plt.ylabel("Actual energy [kWh]", fontsize=14)
        plt.title(f"Context {context} XGBoost model", fontsize=16)
        plt.tick_params(axis='both', labelsize=12)
        plt.grid(True, linewidth=0.5, alpha=0.4)
        plt.tight_layout()

        outpath = os.path.join(output_folder, f"ctx{context}.png")
        plt.savefig(outpath, dpi=150)
        plt.close()



if __name__ == "__main__":
    case_study = "Total_cut"
    # run_dataset(case_study)
    # run_model(case_study)
    calc_temp_anm_prob(case_study)
    # plot_predicted_vs_actual(case_study)
