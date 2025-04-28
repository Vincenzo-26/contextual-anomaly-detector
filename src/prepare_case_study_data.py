from utils import *
import json
from collections import defaultdict

def run_data(case_study: str, case_studies_to_align_on: list[str] = None):
    def load_config_and_leaves(case_study_name):
        config_path = os.path.join(PROJECT_ROOT, "data", case_study_name, "config.json")
        if not os.path.exists(config_path):
            print(f"⚠️ Config non trovato per {case_study_name}")
            return None, []
        with open(config_path, "r") as f:
            config = json.load(f)
        leaf_nodes = find_parents_of_leaves(config["Load Tree"])
        return config, leaf_nodes

    # Prima carichiamo il case_study principale
    cleaned_data = {}
    common_start, common_end = None, None

    all_case_studies = [case_study]
    if case_studies_to_align_on:
        all_case_studies.extend(case_studies_to_align_on)

    for cs in all_case_studies:
        print(f"🔍 Analizzo {cs}...")
        config, leaf_nodes = load_config_and_leaves(cs)
        if config is None:
            continue

        for leaf in leaf_nodes:
            raw_path = os.path.join(PROJECT_ROOT, "raw_data", f"{leaf}.csv")
            if not os.path.exists(raw_path):
                print(f"⚠️ File non trovato: {raw_path}")
                continue

            df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

            unit_config = config.get("Unit")
            if unit_config is None:
                unit = "W"
            elif isinstance(unit_config, dict):
                unit = unit_config.get(leaf, "W")
            else:
                unit = unit_config

            df_clean = clean_time_series(df, unit=unit)

            if df_clean.empty:
                print(f"⚠️ Dataset vuoto dopo la pulizia: {leaf}")
                continue

            cleaned_data[f"{cs}/{leaf}"] = df_clean

            start, end = df_clean.index.min(), df_clean.index.max()
            if common_start is None or start > common_start:
                common_start = start
            if common_end is None or end < common_end:
                common_end = end

    if common_start is None or common_end is None:
        print("❌ Nessun dato valido trovato.")
        return

    aligned_index = pd.date_range(common_start, common_end, freq="15min")
    if case_studies_to_align_on:
        print(f"📅 Intervallo comune: {common_start} ➔ {common_end} ({len(aligned_index)} punti)")

    # Riallineiamo e salviamo tutto
    dfs_per_case_study = defaultdict(list)

    for key, df_clean in cleaned_data.items():
        case_dir, leaf = key.split("/", 1)
        output_dir = os.path.join(PROJECT_ROOT, "data", case_dir)
        os.makedirs(output_dir, exist_ok=True)

        df_aligned = df_clean.loc[common_start:common_end]
        df_aligned = df_aligned.reindex(aligned_index)

        for col in df_aligned.columns:
            df_col = df_aligned[[col]].copy()
            df_col.columns = ["value"]
            df_col["timestamp"] = aligned_index
            df_col = df_col[["timestamp", "value"]]

            safe_col_name = col.replace("/", "_")
            out_path = os.path.join(output_dir, f"{safe_col_name}.csv")
            df_col.to_csv(out_path, index=False)

            dfs_per_case_study[case_dir].append(df_col)

    # Ora creiamo il totale sommato per ogni case study
    for case_dir, dfs in dfs_per_case_study.items():
        if dfs:
            df_all = pd.concat(dfs, axis=0)
            total_df = df_all.groupby("timestamp", as_index=False)["value"].sum()
            output_dir = os.path.join(PROJECT_ROOT, "data", case_dir)
            total_df.to_csv(os.path.join(output_dir, f"{case_dir}.csv"), index=False)

    print(f"✅ avaiable data for {case_study}")

if __name__ == "__main__":
    # run_data("AuleP")
    # run_data("AuleP", ["AuleR"])
    run_data("Cabina", ["AuleR", "AuleP"])