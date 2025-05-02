from src.utils import *
import ruptures as rpt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, normaltest
from sklearn.metrics import r2_score


def check_thermal_sensitivity(df_segment, corr_thresh=0.5, r2_thresh=0.5, slope_thresh=0.2):
    X = df_segment["Temperature"].values.reshape(-1, 1)
    y = df_segment["Energy"].values

    if len(X) < 2:
        return {"correlation": np.nan, "slope": np.nan, "r2_score": np.nan, "is_thermal_sensitive": False}

    corr = np.corrcoef(X.flatten(), y)[0, 1]
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    r2 = r2_score(y, model.predict(X))

    is_sensitive = (abs(corr) > corr_thresh) and (abs(slope) > slope_thresh) and (r2 > r2_thresh)

    return {
        "correlation": round(corr, 3),
        "slope": round(slope, 3),
        "r2_score": round(r2, 3),
        "is_thermal_sensitive": is_sensitive,
        "corr_thresh": corr_thresh,
        "slope_thresh": slope_thresh,
        "r2_thresh": r2_thresh
    }


def analyze_thermal_sensitivity_per_segment(df_normals_sorted, change_points):
    print("\nThermal Sensitivity per Segmento:")
    results = []
    start = 0
    for i, end in enumerate(change_points):
        df_segment = df_normals_sorted.iloc[start:end]
        metrics = check_thermal_sensitivity(df_segment)
        status = "Thermal sensitive ✅" if metrics["is_thermal_sensitive"] else "Not thermal sensitive ❌"
        t_min = df_segment["Temperature"].min()
        t_max = df_segment["Temperature"].max()
        print(
            f"Segmento {i + 1} [{t_min:.1f}°C - {t_max:.1f}°C]:     "
            f"Corr: |{metrics['correlation']}| {'>' if abs(metrics['correlation']) > metrics['corr_thresh'] else '<'} {metrics['corr_thresh']} "
            f"{'✅' if abs(metrics['correlation']) > metrics['corr_thresh'] else '❌'}, "
            f"Slope: |{metrics['slope']}| {'>' if abs(metrics['slope']) > metrics['slope_thresh'] else '<'} {metrics['slope_thresh']} "
            f"{'✅' if abs(metrics['slope']) > metrics['slope_thresh'] else '❌'}, "
            f"R²: {metrics['r2_score']} {'>' if metrics['r2_score'] > metrics['r2_thresh'] else '<'} {metrics['r2_thresh']} "
            f"{'✅' if metrics['r2_score'] > metrics['r2_thresh'] else '❌'} |    -> {status}"
        )
        results.append(metrics)
        start = end
    return results


def run_change_point(case_study: str, penalty: int):
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    foglie = find_leaf_nodes(config["Load Tree"])

    for foglia in foglie:
        segment_results = []  # raccoglie tutto per foglia
        groups_path = os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv")
        groups = pd.read_csv(groups_path, parse_dates=["timestamp"])
        groups["date"] = groups["timestamp"].dt.date
        context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
        cluster_cols = [col for col in groups.columns if col.startswith("Cluster_")]

        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                df_normals, df_anm = run_energy_temp(case_study, foglia, context, cluster)

                if df_normals is None or df_normals.empty:
                    print(f"⚠️  Nessun dato per {foglia} - ctx {context} cl {cluster}")
                    segment_results.append({
                        "Context": context,
                        "Cluster": cluster,
                        "Segmento": 1,
                        "Thermal Sensitive": False
                    })
                    continue

                df_normals_sorted = df_normals.sort_values(by="Temperature").dropna(subset=["Temperature", "Energy"])
                df_normals_sorted = df_normals_sorted[df_normals_sorted["Energy"] > 0]

                print(
                    f"{foglia} - ctx {context} cl {cluster} -> After drop: {len(df_normals_sorted)} rows (NaN e zero esclusi)")

                if len(df_normals_sorted) < 5:
                    print(f"⚠️  Dati insufficienti dopo filtro per {foglia} - ctx {context} cl {cluster}")
                    segment_results.append({
                        "Context": context,
                        "Cluster": cluster,
                        "Segmento": 1,
                        "Thermal Sensitive": False
                    })
                    continue

                print(
                    f"{foglia} - ctx {context} cl {cluster} -> After drop: {len(df_normals_sorted)} rows (NaN e zero esclusi)"
                )

                signal = df_normals_sorted[["Temperature", "Energy"]].values
                algo = rpt.Pelt(model="rbf").fit(signal)
                change_points = algo.predict(pen=penalty)

                models = []
                start = 0
                for i, end in enumerate(change_points):
                    X_seg = df_normals_sorted.iloc[start:end]["Temperature"].values.reshape(-1, 1)
                    y_seg = df_normals_sorted.iloc[start:end]["Energy"].values
                    model = LinearRegression().fit(X_seg, y_seg)
                    models.append(model)

                    segment = df_normals_sorted.iloc[start:end]
                    metrics = check_thermal_sensitivity(segment)

                    segment_results.append({
                        "Context": context,
                        "Cluster": cluster,
                        "Segmento": i + 1,
                        "Thermal Sensitive": metrics["is_thermal_sensitive"]
                    })
                    start = end

        # Salva CSV unico per foglia
        output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_segments")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"thermal_segments_{foglia}.csv")
        pd.DataFrame(segment_results).to_csv(output_file, index=False)



if __name__ == "__main__":
    run_change_point(
        case_study="Cabina",
        penalty=10
    )