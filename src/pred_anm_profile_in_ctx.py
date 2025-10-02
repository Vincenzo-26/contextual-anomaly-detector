import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from xgboost import XGBRegressor, Booster
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import defaultdict

from src.utils import run_energy_temp, run_energy_temp_profile, assign_values_by_depth, find_leaf_nodes
from settings import PROJECT_ROOT

def run_dataset(case_study: str):
    """
    Crea dataset di addestramento e validazione per modelli di previsione XGBoost
    basati sui dati di inferenza e profili energetici di un dato caso studio.

    - Carica la configurazione e i risultati di inferenza del caso di studio.
    - Organizza i dati relativi ai nodi foglia della rete, calcolando una soglia di split
      per dividere i dati in training e validation set basandosi sulle probabilità inferite.
    - Recupera profili energetici normali e anomali per ogni foglia, contesto e cluster,
      combinandoli e classificandoli in base a un cutoff energetico calcolato.
    - Unisce i dati con i risultati di inferenza per ciascuna foglia.
    - Genera e salva file CSV separati per il training e la validazione in directory dedicate.

    Input:
    case_study : str
        Nome del caso di studio.
    """

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

    soglie_per_leaf = assign_values_by_depth(config["Load Tree"])

    leaves = find_leaf_nodes(config["Load Tree"])
    for leaf in leaves:
        soglia_split = soglie_per_leaf.get(leaf)
        print(f"{leaf} (train-val split threshold {soglia_split*100}%)...   ", end="")

        soglia_en_max = 0.10
        E_max_leaf = 0
        for context in time_windows_df.id.unique():
            obs = time_windows_df.loc[time_windows_df['id'] == context, 'observations'].values[0]
            for cluster_col in [col for col in groups.columns if col.startswith("Cluster_")]:
                cluster = int(cluster_col.split("_")[-1])
                df_normals, _ = run_energy_temp(case_study, leaf, context, cluster)
                max_energy = df_normals["Energy"].max(skipna=True)
                avg_energy = max_energy / obs if obs > 0 else 0
                E_max_leaf = max(E_max_leaf, avg_energy)
        energy_cutoff = soglia_en_max * E_max_leaf

        df_all = []
        for context in time_windows_df.id.unique():
            for cluster_col in [col for col in groups.columns if col.startswith("Cluster_")]:
                cluster = int(cluster_col.split("_")[-1])
                df_normal, df_anomalous = run_energy_temp_profile(case_study, leaf, context, cluster)
                combined = pd.concat([df_normal, df_anomalous], ignore_index=True)
                if not combined.empty:
                    df_all.append(combined)

        df_all_combined = pd.concat(df_all, ignore_index=True)
        df_all_combined["Status"] = df_all_combined["Energy"] >= energy_cutoff
        inference_df["Date"] = pd.to_datetime(inference_df["Date"])
        df_all_combined["Date"] = pd.to_datetime(df_all_combined["Date"])

        inference_df["Date"] = pd.to_datetime(inference_df["Date"])
        merge_cols = ["Date", "Context"]
        use_cols = merge_cols + ["P(Total=1)"]

        leaf_col = f"P({leaf}=1)"
        if leaf_col in inference_df.columns:
            if not inference_df["P(Total=1)"].equals(inference_df[leaf_col]):
                use_cols.append(leaf_col)

        df_all_combined = df_all_combined.merge(
            inference_df[use_cols],
            how="left",
            on=merge_cols
        )

        df_train = df_all_combined[df_all_combined[f"P({leaf}=1)"] < soglia_split].copy()
        df_val = df_all_combined[df_all_combined[f"P({leaf}=1)"] >= soglia_split].copy()

        df_train.set_index("Date", inplace=True)
        df_val.set_index("Date", inplace=True)

        df_train.to_csv(os.path.join(output_dir_train, f"df_train_{leaf}.csv"))
        if not df_val.empty:
            df_val.to_csv(os.path.join(output_dir_val, f"df_val_{leaf}.csv"))
        print("ok")
    return


def run_model(case_study: str):
    """
    Addestra modelli di regressione XGBoost per prevedere il profilo energetico dei nodi foglia
    di una rete bayesiana relativa a un caso di studio specifico, valutandone le performance
    tramite cross-validation a 5 fold.

    La funzione:
    - Carica la configurazione e i dataset di addestramento per ciascun nodo foglia.
    - Esegue una normalizzazione min-max sull'energia misurata.
    - Definisce le feature e l'etichetta da usare per il training, escludendo colonne non rilevanti.
    - Esegue validazione incrociata per calcolare metriche di accuratezza (MAE, RMSE, R²).
    - Visualizza scatter plot con valori predetti vs valori reali per ogni nodo.
    - Salva i modelli addestrati in formato JSON e le metriche di valutazione in CSV.
    - Registra informazioni di scaling per l'energia usate durante il preprocessing.

    Input:
    case_study : str
        Nome del caso di studio.
    """

    def min_max_scaling(array, reverse=False, min_val=None, max_val=None):
        if not reverse:
            min_val = np.min(array)
            max_val = np.max(array)
            scaled = (array - min_val) / (max_val - min_val)
            return scaled, min_val, max_val
        else:
            return array * (max_val - min_val) + min_val


    print("Creating profile prediction models...\n")
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    train_dir = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "train_df")
    model_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "models")
    output_folder_viz = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "Pred_XGboost")

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_folder_viz, exist_ok=True)

    metrics_results = []
    scaling_info = []

    leaves = find_leaf_nodes(config["Load Tree"])
    for leaf in leaves:
        df_train_path = os.path.join(train_dir, f"df_train_{leaf}.csv")
        df_train = pd.read_csv(df_train_path, parse_dates=["Date"])
        if 'Date' in df_train.columns:
            df_train.set_index('Date', inplace=True)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        y_raw = df_train["Energy"].values
        y, y_min, y_max = min_max_scaling(y_raw)
        scaling_info.append({"Load": leaf, "Energy_min": y_min, "Energy_max": y_max})
        if leaf == "Total":
            X = df_train.drop(columns=["Energy", "anm", "P(Total=1)"])
        else:
            X = df_train.drop(columns=["Energy", "anm", "P(Total=1)", f"P({leaf}=1)"])
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
        })

        #PLOT ACT VS PRED
        y_pred_rescaled = min_max_scaling(y_preds_all, reverse=True, min_val=y_min, max_val=y_max)
        y_true_rescaled = min_max_scaling(y, reverse=True, min_val=y_min, max_val=y_max)
        color_map = plt.colormaps["tab20"].resampled(len(leaves))
        leaf_idx = leaves.index(leaf)
        leaf_color = color_map(leaf_idx)
        plt.figure(figsize=(6, 6))
        plt.grid(True, linewidth=0.5, alpha=0.4)
        plt.scatter(
            y_pred_rescaled,
            y_true_rescaled,
            color=leaf_color,
            edgecolor="white",
            alpha=0.7,
            linewidth=0.5
        )
        max_val = max(max(y_pred_rescaled), max(y_true_rescaled)) * 1.1
        plt.plot([0, max_val], [0, max_val], linestyle="--", color="black", alpha=0.6)
        plt.xlabel("Actual Energy [kWh]", fontsize=16)
        plt.ylabel("Predicted Energy [kWh]", fontsize=16)
        plt.title(f"{leaf}", fontsize=20)
        plt.tick_params(axis='both', labelsize=14)
        plt.tight_layout()
        save_path = os.path.join(output_folder_viz, f"actual_vs_pred_{leaf}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"[{leaf}]  MAE: {np.mean(mae_scores):.2f} kWh  "
              f"RMSE: {np.mean(rmse_scores):.2f} kWh  "
              f"R²: {np.mean(r2_scores):.2f}", end="")

        model.fit(X, y)
        model.save_model(os.path.join(model_path, f"model_{leaf}.json"))

        print("    Model saved ✅")

    df_all_results = pd.DataFrame(metrics_results)
    df_all_results.to_csv(os.path.join(model_path, "metrics.csv"), index=False)
    df_scaling = pd.DataFrame(scaling_info)
    df_scaling.to_csv(os.path.join(model_path, "scaling_minmax_info.csv"), index=False)
    print(f"\n📊 metrics.csv saved")
    print("📉 energy_min_max.csv salvato correttamente.")


def run_profile(case_study: str, leaf: str, date: str, context: int, which_df: str, plot: bool, plot_pred: bool, plot_fill: bool, plot_temp: bool):
    """
    Predice e visualizza il profilo energetico (kWh) di un nodo specifico (leaf) per una data,
    un contesto e un caso di studio, utilizzando un modello XGBoost addestrato.

    Carica i dati di addestramento o validazione, normalizza i dati di energia,
    applica il modello pre-addestrato per generare previsioni puntuali e confronta i valori
    energetici reali e predetti calcolando la differenza di energia sprecata.

    Opzionalmente, la funzione può mostrare grafici del profilo energetico reale e previsto,
    evidenziando l'energia sprecata per quel profilo, e un grafico della temperatura esterna relativa al contesto.

    input:
    case_study : str
        Nome del caso di studio per reperire dati e modelli.
    leaf : str
        Nodo della rete bayesiana per cui effettuare la previsione.
    date : str
        Data specifica per la previsione.
    context : int
        Identificatore del contesto temporale.
    which_df : str
        Determina se usare i dati di "train" o "test" per la previsione.
    plot : bool
        Se True, visualizza il grafico del profilo energetico reale (e opzionalmente previsto).
    plot_pred : bool
        Se True, visualizza anche il profilo energetico previsto nel grafico.
    plot_fill : bool
        Se True, evidenzia con colore l’energia sprecata tra reale e previsto.
    plot_temp : bool
        Se True, visualizza un grafico della temperatura esterna per il contesto e la data specificata.

    Output:
    difference : float o None
        Quantità totale di energia "sprecata" per quel singolo profilo;
        None se i dati per la previsione non sono disponibili.
    """
    print(f"Predicting energy profile (kWh) for:{case_study} - {leaf} - {date} - ctx{context}")

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
        df_val_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "val_df", f"df_val_{leaf}.csv")
        df_val = pd.read_csv(os.path.join(df_val_path), parse_dates=["Date"])
        df_val.set_index('Date', inplace=True)
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
        if leaf == "Total":
            row_input = row.drop(labels=["Energy", "anm", "P(Total=1)"], errors="ignore").to_frame().T
        else:
            row_input = row.drop(labels=["Energy", "anm", "P(Total=1)", f"P({leaf}=1)"], errors="ignore").to_frame().T
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

    print(f"CMP   -> Anomaly? {df_day_ctx['anm'].iloc[0]}")
    print(f"BN 'Total' prob    -> {df_day_ctx['P(Total=1)'].iloc[0] * 100:.2f}%")
    print(f"Actual total energy: {real_total} kWh")
    print(f"Predicted total energy: {pred_total} kWh")
    print(f"Wasted energy: {difference} kWh 🗑️")

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
    x = np.arange(len(x_labels))
    real_array = np.array(real_profile)
    pred_array = np.array(pred_profile)
    # plt.plot(x, real_array, label="Actual", color="#4682B4", marker="o", markersize=5, linewidth=1.5)
    plt.plot(x, real_array, color="#4682B4", marker="o", markersize=5, linewidth=1.5)

    if plot_pred:
        plt.plot(x, pred_array, label="Predicted", color="#CD5C5C", marker="o", markersize=5, linewidth=1.5)

        if plot_fill:
            mask = real_array > pred_array
            plt.fill_between(x, pred_array, real_array, where=mask, interpolate=True,
                             color="orange", alpha=0.4, label=f"Wasted Energy ({difference} kWh)")
    plt.title(f"{leaf} - {date} - Context {context}", fontsize=18)
    plt.ylabel("Energy [kWh]", fontsize=14)

    step = 2
    plt.xticks(ticks=x[::step], labels=[x_labels[i] for i in x[::step]], rotation=45)
    plt.tick_params(axis='both', labelsize=14)
    # plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if plot:
        plt.show()

    if plot_temp:
        temp_path = os.path.join(PROJECT_ROOT, "data", case_study, "Temperatura Esterna.csv")
        df_temp = pd.read_csv(temp_path)
        df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
        df_temp["date"] = df_temp["timestamp"].dt.date
        df_temp["time"] = df_temp["timestamp"].dt.time

        cluster_df = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv"))
        tw_df = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv"))

        cluster_df["timestamp"] = pd.to_datetime(cluster_df["timestamp"])
        cluster_df["date"] = cluster_df["timestamp"].dt.date

        cluster_cols = [col for col in cluster_df.columns if col.startswith("Cluster_")]
        cluster_df["Cluster"] = cluster_df[cluster_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
        cluster_map = cluster_df.set_index("date")["Cluster"].to_dict()
        df_temp["Cluster"] = df_temp["date"].map(cluster_map)
        df_temp["Cluster"] = df_temp["Cluster"].astype("Int64")

        tw_df["to"] = tw_df["to"].replace("24:00", "23:59")
        tw_df["from"] = pd.to_datetime(tw_df["from"], format="%H:%M").dt.time
        tw_df["to"] = pd.to_datetime(tw_df["to"], format="%H:%M").dt.time

        def assign_context(row):
            for _, tw_row in tw_df.iterrows():
                from_time = tw_row["from"]
                to_time = tw_row["to"]
                if from_time <= row["time"] < to_time or (
                        from_time > to_time and (row["time"] >= from_time or row["time"] < to_time)):
                    return tw_row["id"]
            return None

        df_temp["Context"] = df_temp.apply(assign_context, axis=1)

        clt = int(row_input["Cluster"].iloc[0])
        df_temp_ctx = df_temp[(df_temp["Context"] == context) & (df_temp["Cluster"] == clt)].copy()

        pivot_temp = df_temp_ctx.pivot(index="date", columns="time", values="value")
        pivot_temp = pivot_temp.sort_index(axis=1)
        time_labels = [t.strftime("%H:%M") for t in pivot_temp.columns]

        plt.figure(figsize=(10, 6))
        for d in pivot_temp.index:
            y_vals = pivot_temp.loc[d]
            if d == pd.to_datetime(date).date():
                plt.plot(time_labels, y_vals, color="orange", linewidth=1.5, marker="o", markersize=5)
            else:
                plt.plot(time_labels, y_vals, color="gray", alpha=0.3, linewidth=0.8)

        plt.title(f"Outdoor Temperature - {date} - Context {context}", fontsize=18)
        plt.ylabel("Temperature [°C]", fontsize=14)
        plt.xticks(ticks=np.arange(0, len(time_labels), 2), labels=time_labels[::2], rotation=45)
        plt.tick_params(axis='both', labelsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return difference


def calc_wasted_energy(case_study: str, method: str):
    """
        Calcola e riassume l'energia sprecata rispetto a quella prevista per un caso di studio specifico utilizzando due
        metodi distinti: la rete bayesiana (BN) o il metodo CMP. Aggrega i risultati per foglia e per mese.

        La funzione:
        - Legge la configurazione del caso di studio e carica i dati di validazione e le tabelle di anomalie.
        - Per ogni nodo foglia, calcola l'energia sprecata eseguendo previsioni di profilo energetico per ogni
          combinazione data-contesto rilevante, a seconda del metodo scelto.
        - Somma e aggrega i valori di energia sprecata per foglia e per mese.
        - Salva i risultati aggregati in più file CSV (sintesi per foglia, per mese e matrice foglia × mese).
        - Stampa a video lo stato dell'elaborazione e un riepilogo finale dell'energia sprecata totale.

        Input:
        case_study : str
            Nome del caso di studio per cui calcolare l'energia sprecata.
        method : str
            Metodo di calcolo da utilizzare, "BN" per Bayesian Network o "CMP" per il metodo CMP.
        """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)
    wasted_dir = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost")
    df_val_path = os.path.join(PROJECT_ROOT, "results", case_study, "Pred_XGboost", "val_df")
    anm_table_path = os.path.join(PROJECT_ROOT, "results", case_study, "anomaly_table")

    wasted_results = []
    total_waste = 0
    waste_by_month = defaultdict(float)
    waste_by_leaf_month = defaultdict(lambda: defaultdict(float))
    leaves = find_leaf_nodes(config["Load Tree"])
    for leaf in leaves:
        print(f"\n🔍 Processing leaf: {leaf}")
        val_file = os.path.join(df_val_path, f"df_val_{leaf}.csv")
        if not os.path.exists(val_file):
            print(f"⚠️ No anomalies for leaf: {leaf}")
            continue
        df_leaf_val = pd.read_csv(val_file) # già è filtrato con P(total=1)>0.3

        # TODO CAMBIARE CSV SE CONSIDERARE QUELLE DEI NODI FOGLIA
        anm_table_leaf = pd.read_csv(os.path.join(anm_table_path, f"anomaly_table_{leaf}.csv"))

        if method == "BN":
            print("Energy waste with Bayesian Network")
            df_leaf_val["Date"] = pd.to_datetime(df_leaf_val["Date"])
            df_leaf_val = df_leaf_val[df_leaf_val["Date"].dt.month.isin([6, 7, 8])] #per giugno luglio e agosto
            combs = df_leaf_val[["Date", "Context"]].drop_duplicates()
            combs = [tuple(x) for x in combs.to_numpy()]
        elif method == "CMP":
            print("Energy waste with CMP")
            combs = anm_table_leaf[["Date", "Context"]].drop_duplicates()
            combs = [tuple(x) for x in combs.to_numpy()]

        waste_leaf = 0

        for date, context in combs:
            which_df = "test"
            waste = run_profile(case_study, leaf, date, context, which_df, False, False, False, False)
            print("\n")
            if waste is not None:
                month_str = pd.to_datetime(date).strftime("%Y-%m")
                waste_by_month[month_str] += waste
                waste_by_leaf_month[leaf][month_str] += waste
                waste_leaf += waste
                total_waste += waste

        wasted_results.append((leaf, round(waste_leaf, 2)))
    df_result = pd.DataFrame(wasted_results, columns=["Load", "Wasted energy [kWh]"])
    df_result["%"] = df_result["Wasted energy [kWh]"] / total_waste * 100
    df_result["%"] = df_result["%"].round(2)

    save_path = os.path.join(wasted_dir, f"wasted_energy_summary_{method}.csv")
    df_result.to_csv(save_path, index=False)
    print(f"\n✅ Saved summary to: {save_path}")
    df_monthly = pd.DataFrame(list(waste_by_month.items()), columns=["Month", "Wasted energy [kWh]"])
    df_monthly["%"] = df_monthly["Wasted energy [kWh]"] / total_waste * 100
    df_monthly["%"] = df_monthly["%"].round(2)

    monthly_path = os.path.join(wasted_dir, f"wasted_energy_monthly_{method}.csv")
    df_monthly.to_csv(monthly_path, index=False)
    df_pivot = pd.DataFrame(waste_by_leaf_month).fillna(0).T
    pivot_path = os.path.join(wasted_dir, f"wasted_energy_by_leaf_month_{method}.csv")
    df_pivot.to_csv(pivot_path)
    print(f"📊 Saved matrix leaf × month to: {pivot_path}")
    print(f"📅 Saved monthly summary to: {monthly_path}")
    print(f"\n✅ Total wasted energy: {total_waste}")


if __name__ == "__main__":
    case_study = "Total"
    # run_dataset(case_study)
    # run_model(case_study)
    run_profile(case_study, "GF4", "2024-08-04", 1, "train",
                True,
                True,
                True,
                True)
    # calc_wasted_energy(case_study, "BN")

