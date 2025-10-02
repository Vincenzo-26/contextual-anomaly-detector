import os
import pandas as pd
from settings import PROJECT_ROOT
from utils import clean_time_series, get_nodes_by_level, print_boxed_title
import json
from src.cmp.groups_definition import run_clustering
from src.cmp.time_windows_definition import run_cart
from src.cmp.utils import extract_holidays

def run_data(case_study: str):
    """
    Esegue il preprocessing dei dati grezzi di un caso studio, pulendo e ricostruendo le serie temporali
    di ciascun nodo del load tree e della temperatura esterna.

    Per ogni nodo:
      - carica la serie temporale grezza dal file CSV;
      - applica la funzione `clean_time_series`;
      - salva la serie temporale pulita in `data/<case_study>/`.

    Inoltre:
      - esegue la stessa procedura per la temperatura esterna;
      - raccoglie statistiche sul preprocessing (giorni completi, interpolati, ricostruiti con KNN, rimossi);
      - salva un riepilogo globale in `results/<case_study>/summary_preprocessing.csv`.

    Args:
        case_study (str): Nome del caso studio.

    Returns:
        None: I risultati sono salvati su disco (serie pulite e riepilogo delle statistiche).
    """
    print_boxed_title("Preprocessing & Alignment 🧹📊")

    output_dir = os.path.join(PROJECT_ROOT, "data", case_study)
    os.makedirs(output_dir, exist_ok=True)

    # Load config & Load Tree
    with open(os.path.join(PROJECT_ROOT, "raw_data", case_study, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(output_dir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=2)

    load_tree = config["Load Tree"]
    levels = get_nodes_by_level(load_tree)
    all_nodes = [node for level in levels for node in level]

    summary_rows = []
    for node in all_nodes:
        print(f"\n🔧 Processing {node}")
        df_node_raw = pd.read_csv(os.path.join(PROJECT_ROOT, "raw_data", case_study, f"{node}.csv"))
        df_node_clean, node_stats = clean_time_series(df_node_raw)

        df_node_clean.to_csv(os.path.join(output_dir, f"{node}.csv"), index=False)
        summary = node_stats["summary"]
        summary_rows.append({
            "Nodo": node,
            "Intervallo": summary["range"],
            "Giorni totali": summary["total_days"],
            "Giorni interi": node_stats["complete"],
            "Interpolati": node_stats["interpolated"],
            "KNN": node_stats["knn"],
            "Rimossi": node_stats["removed"]
        })
    df_temp_raw = pd.read_csv(os.path.join(PROJECT_ROOT, "raw_data", case_study, "Temperatura Esterna.csv"))
    df_temp_clean, temp_stats = clean_time_series(df_temp_raw)


    df_temp_clean.to_csv(os.path.join(output_dir, "Temperatura Esterna.csv"), index=False)
    summary = temp_stats["summary"]
    summary_rows.append({
        "Nodo": node,
        "Intervallo": summary["range"],
        "Giorni totali": summary["total_days"],
        "Giorni interi": len(node_stats["dates"]["complete"]),
        "Interpolati": len(node_stats["dates"]["interpolated"]),
        "KNN": len(node_stats["dates"]["knn"]),
        "Rimossi": summary["total_days"] - (
                len(node_stats["dates"]["complete"]) +
                len(node_stats["dates"]["interpolated"]) +
                len(node_stats["dates"]["knn"])
        )
    })
    print("\n✅ All nodes cleaned.\n")
    df_summary = pd.DataFrame(summary_rows)
    output_dir_results = os.path.join(PROJECT_ROOT, "results", case_study)
    os.makedirs(output_dir_results, exist_ok=True)
    df_summary.to_csv(os.path.join(output_dir_results, "summary_preprocessing.csv"), index=False)
    print(f"\n✅ Summary stats saved for the whole time range.\n")
if __name__ == "__main__":
    run_data("Total")