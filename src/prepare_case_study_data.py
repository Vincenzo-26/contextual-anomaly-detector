from utils import *
import json

def run_data(case_study: str, case_studies_to_align_on: list[str] = None):
    """
    Prepara i dati per un dato case study. Se vengono forniti altri case study nella lista `aligned_to`,
    taglia l'intervallo temporale per allinearsi al sottoinsieme più restrittivo tra tutti.

    Args:
        case_study (str): Nome del case study principale da preparare.
        case_studies_to_align_on (list[str], optional): Lista di altri case study da usare per allineare l'intervallo temporale.
                                           Se None, considera solo il case study principale.

    Returns:
        None
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    leaf_nodes = find_parents_of_leaves(config["Load Tree"])
    cleaned_data = {}
    common_start, common_end = None, None

    for leaf in leaf_nodes:
        raw_path = os.path.join(PROJECT_ROOT, "raw_data", f"{leaf}.csv")
        if not os.path.exists(raw_path):
            print(f"⚠️ File non trovato: {raw_path}")
            continue

        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

        unit_config = config.get("Unit")
        if unit_config is None:
            print(f"⚠️ Nessuna unità specificata nel config di '{case_study}'. Uso 'W' per default.")
            unit = "W"
        elif isinstance(unit_config, dict):
            unit = unit_config.get(leaf)
            if unit is None:
                print(f"⚠️ Nessuna unità specificata per il nodo '{leaf}'. Uso 'W' per default.")
                unit = "W"
        else:
            unit = unit_config

        df_clean = clean_time_series(df, unit=unit)

        if df_clean.empty:
            print(f"⚠️ Dataset vuoto dopo la pulizia: {leaf}")
            continue

        cleaned_data[leaf] = df_clean

        start, end = df_clean.index.min(), df_clean.index.max()
        if common_start is None or start > common_start:
            common_start = start
        if common_end is None or end < common_end:
            common_end = end

    if case_studies_to_align_on is None:
        case_studies_to_align_on = []

    for other in case_studies_to_align_on:
        print(f"🔍 Analizzo {other} per allineamento temporale...")
        other_path = os.path.join(PROJECT_ROOT, "data", other, f"{other}.csv")
        if not os.path.exists(other_path):
            print(f"⚠️ CSV non trovato per {other}")
            continue

        df_other = pd.read_csv(other_path, index_col=0, parse_dates=True)
        df_other = clean_time_series(df_other)

        start_o, end_o = df_other.index.min(), df_other.index.max()
        if start_o > common_start:
            common_start = start_o
        if end_o < common_end:
            common_end = end_o

    aligned_index = pd.date_range(common_start, common_end, freq="15min")

    output_dir = os.path.join(PROJECT_ROOT, "data", case_study)
    os.makedirs(output_dir, exist_ok=True)

    list_df = []

    for leaf, df_clean in cleaned_data.items():
        df_aligned = df_clean.loc[common_start:common_end]
        df_aligned = df_aligned.reindex(aligned_index)

        for col in df_aligned.columns:
            df_col = df_aligned[[col]].copy()
            df_col.columns = ["value"]
            df_col["timestamp"] = aligned_index
            df_col = df_col[["timestamp", "value"]]

            if df_col["timestamp"].is_unique and len(df_col) == len(aligned_index):
                list_df.append(df_col)
            else:
                print(f"❌ Problema con: {leaf} - {col}")

            safe_col_name = col.replace("/", "_")
            out_path = os.path.join(output_dir, f"{safe_col_name}.csv")
            df_col.to_csv(out_path, index=False)

    for i, df in enumerate(list_df):
        assert len(df) == len(aligned_index), f"❌ Lunghezza incoerente nel DataFrame #{i}"
        assert (df["timestamp"] == aligned_index).all(), f"❌ Timestamp non allineati nel DataFrame #{i}"

    df_all = pd.concat(list_df, axis=0)
    total_df = df_all.groupby("timestamp", as_index=False)["value"].sum()
    total_df.to_csv(os.path.join(output_dir, f"{case_study}.csv"), index=False)

    print(f"✅ avaiable data for {case_study}")

if __name__ == "__main__":
    run_data("AuleP", ["AuleR"])