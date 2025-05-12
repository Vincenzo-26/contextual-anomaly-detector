import numpy as np
import ruptures as rpt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from src.utils import PROJECT_ROOT, run_energy_temp
from src.calc_thermal_sensitivity import check_thermal_sensitivity, identify_operational_modes
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


def plot_residuals(case_study: str, sottocarico: str, context: int, cluster: int, penalty: int = 10, normalize: bool = True):
    df_normals, df_anomalies = run_energy_temp(case_study, sottocarico, context, cluster)
    if df_normals is None or df_normals.empty:
        print("⚠️ Dati insufficienti.")
        return

    df_sorted = df_normals.sort_values("Temperature").dropna(subset=["Temperature", "Energy"])
    df_sorted = df_sorted[df_sorted["Energy"] > 0]
    df_sorted = identify_operational_modes(df_sorted)
    mode_list = sorted(df_sorted["Mode"].unique())

    if len(mode_list) == 1:
        # === CASO CON UNA SOLA MODALITÀ ===
        signal = df_sorted[["Temperature", "Energy"]].values
        scaler = StandardScaler()
        signal_scaled = scaler.fit_transform(signal)
        algo = rpt.Pelt(model="rbf").fit(signal_scaled)
        change_points = algo.predict(pen=penalty)

        residuals_by_segment = []
        segment_labels = []
        thermal_sensitive_indices = []

        start = 0
        for i, end in enumerate(change_points):
            segment = df_sorted.iloc[start:end]
            if len(segment) < 2:
                start = end
                continue

            X = segment["Temperature"].values.reshape(-1, 1)
            y = segment["Energy"].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)

            metrics = check_thermal_sensitivity(segment, normalize=normalize)
            is_sensitive = metrics["is_thermal_sensitive"]

            if is_sensitive:
                residuals = y - y_pred
                residuals_by_segment.append((residuals, i))
                segment_labels.append(f"Segmento {i + 1}")
                thermal_sensitive_indices.append((start, end, model))

            start = end

        n_sens = len(residuals_by_segment)
        fig = make_subplots(
            rows=1, cols=1 + n_sens,
            column_widths=[0.5] + [0.5 / n_sens] * n_sens if n_sens > 0 else [1],
            subplot_titles=["Firma Energetica"] + [f"Residui {label}" for label in segment_labels]
        )

        # Reset start
        start = 0
        for i, end in enumerate(change_points):
            segment = df_sorted.iloc[start:end]
            if len(segment) < 2:
                start = end
                continue

            X = segment["Temperature"].values.reshape(-1, 1)
            y = segment["Energy"].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)

            metrics = check_thermal_sensitivity(segment, normalize=normalize)
            is_sensitive = metrics["is_thermal_sensitive"]

            x0 = segment["Temperature"].iloc[0]
            x1 = segment["Temperature"].iloc[-1]
            fillcolor = "rgba(0,200,0,1)" if is_sensitive else "rgba(200,0,0,1)"

            fig.add_shape(
                type="rect", x0=x0, x1=x1, y0=0, y1=1,
                xref="x1", yref="paper",
                fillcolor=fillcolor,
                opacity=0.2,
                line=dict(width=0),
                layer="below"
            )

            fig.add_trace(go.Scatter(
                x=X.flatten(), y=y_pred,
                mode="lines", name=f"Segmento {i+1}"
            ), row=1, col=1)

            start = end

        fig.add_trace(go.Scatter(
            x=df_sorted["Temperature"], y=df_sorted["Energy"],
            mode='markers', name="Dati reali", marker=dict(color='skyblue'),
        ), row=1, col=1)

        if df_anomalies is not None and not df_anomalies.empty:
            df_anomalies_sorted = df_anomalies.sort_values(by="Temperature").dropna(subset=["Temperature", "Energy"])
            fig.add_trace(go.Scatter(
                x=df_anomalies_sorted["Temperature"],
                y=df_anomalies_sorted["Energy"],
                mode='markers',
                name="Anomalie (CMP)",
                marker=dict(color='red', size=6),
            ), row=1, col=1)

        for idx, (residuals, seg_index) in enumerate(residuals_by_segment):
            hist, edges = np.histogram(residuals, bins=30)
            bin_width = edges[1] - edges[0]
            mu, sigma = np.mean(residuals), np.std(residuals)
            x_norm = np.linspace(min(residuals), max(residuals), 100)
            y_norm = norm.pdf(x_norm, mu, sigma) * len(residuals) * bin_width

            fig.add_trace(go.Bar(
                x=edges[:-1], y=hist, opacity=0.6,
                name=f"Residui Segmento {seg_index + 1}",
                marker_color='steelblue'
            ), row=1, col=2 + idx)

            fig.add_trace(go.Scatter(
                x=x_norm, y=y_norm,
                mode='lines', line=dict(dash='dot', color='black'),
                showlegend=False
            ), row=1, col=2 + idx)

        fig.update_layout(title=f"{sottocarico} | Context {context} | Cluster {cluster}",
                          template="plotly_white", title_x=0.5)

        fig.show()
        return

    # === CASO CON PIÙ MODALITÀ ===

    # conta quanti segmenti sensibili ha ogni modalità
    residuals_by_mode = {mode: [] for mode in mode_list}
    plots_per_mode = []

    for mode in mode_list:
        df_mode = df_sorted[df_sorted["Mode"] == mode]
        signal = df_mode[["Temperature", "Energy"]].values
        scaler = StandardScaler()
        signal_scaled = scaler.fit_transform(signal)
        algo = rpt.Pelt(model="rank").fit(signal_scaled)
        change_points = algo.predict(pen=penalty)

        start = 0
        for i, end in enumerate(change_points):
            segment = df_mode.iloc[start:end]
            if len(segment) < 2:
                start = end
                continue

            X = segment["Temperature"].values.reshape(-1, 1)
            y = segment["Energy"].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)

            metrics = check_thermal_sensitivity(segment, normalize=normalize)
            if metrics["is_thermal_sensitive"]:
                residuals = y - y_pred
                residuals_by_mode[mode].append((residuals, segment))

            start = end

        plots_per_mode.append(1 + len(residuals_by_mode[mode]))

    total_rows = 2 + len(mode_list)  # 1° tutti i dati, poi una riga per ciascuna modalità
    max_cols = max(plots_per_mode)

    fig = make_subplots(
        rows=total_rows,
        cols=max_cols,
        row_heights=[0.4] + [0.3] * (total_rows - 1),  # 40% alla prima riga, 30% ciascuna alle altre
        subplot_titles=["Tutti i dati"] +
                       [f"Modalità {mode}" for mode in mode_list] +
                       [f"Residui {i + 1}" for mode in mode_list for i in range(len(residuals_by_mode[mode]))],
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )

    # RIGA 1 - TUTTI I DATI
    fig.add_trace(go.Scatter(
        x=df_sorted["Temperature"], y=df_sorted["Energy"],
        mode="markers", name="Tutti i dati", marker=dict(color="skyblue")
    ), row=1, col=1)

    if df_anomalies is not None and not df_anomalies.empty:
        fig.add_trace(go.Scatter(
            x=df_anomalies["Temperature"], y=df_anomalies["Energy"],
            mode="markers", name="Anomalie", marker=dict(color="red", size=6)
        ), row=1, col=1)

    # RIGHE 2+ - MODALITÀ + RESIDUI
    for mode_idx, mode in enumerate(mode_list):
        row_mode = 2 + mode_idx
        df_mode = df_sorted[df_sorted["Mode"] == mode]
        signal = df_mode[["Temperature", "Energy"]].values
        scaler = StandardScaler()
        signal_scaled = scaler.fit_transform(signal)
        algo = rpt.Pelt(model="rank").fit(signal_scaled)
        change_points = algo.predict(pen=penalty)

        start = 0
        for i, end in enumerate(change_points):
            segment = df_mode.iloc[start:end]
            if len(segment) < 2:
                start = end
                continue

            X = segment["Temperature"].values.reshape(-1, 1)
            y = segment["Energy"].values
            model = LinearRegression().fit(X, y)
            x0 = segment["Temperature"].iloc[0]
            x1 = segment["Temperature"].iloc[-1]
            if i == 0:
                x0 = df_mode["Temperature"].min()
            if i == len(change_points) - 2:
                x1 = df_mode["Temperature"].max()

            y_pred_range = model.predict(np.linspace(x0, x1, 100).reshape(-1, 1))

            metrics = check_thermal_sensitivity(segment, normalize=normalize)
            fillcolor = "rgba(0,200,0,0.2)" if metrics["is_thermal_sensitive"] else "rgba(200,0,0,0.2)"

            yaxis_idx = 1 + mode_idx * max_cols  # calcola indice subplot corrente
            yref = f"y{yaxis_idx}"
            xaxis = f"x{yaxis_idx}"
            ymin = df_mode["Energy"].min()
            ymax = df_mode["Energy"].max()

            fig.add_shape(
                type="rect", x0=x0, x1=x1, y0=ymin, y1=ymax,
                xref=xaxis, yref=yref,
                fillcolor=fillcolor, line=dict(width=0), layer="below"
            )

            fig.add_trace(go.Scatter(
                x=np.linspace(x0, x1, 100), y=y_pred_range,
                mode="lines", name=f"Mod {mode} – segmento {i+1}"
            ), row=row_mode, col=1)

            start = end

        # Punti reali
        fig.add_trace(go.Scatter(
            x=df_mode["Temperature"], y=df_mode["Energy"],
            mode="markers", name=f"Dati Mod {mode}",
            marker=dict(color="skyblue")
        ), row=row_mode, col=1)

        for idx, (residuals, _) in enumerate(residuals_by_mode[mode]):
            hist, edges = np.histogram(residuals, bins=30)
            bin_width = edges[1] - edges[0]
            mu, sigma = np.mean(residuals), np.std(residuals)
            x_norm = np.linspace(min(residuals), max(residuals), 100)
            y_norm = norm.pdf(x_norm, mu, sigma) * len(residuals) * bin_width

            fig.add_trace(go.Bar(
                x=edges[:-1], y=hist, opacity=0.6,
                marker_color='steelblue',
                name=f"Residui M{mode}-{idx+1}"
            ), row=row_mode, col=2 + idx)

            fig.add_trace(go.Scatter(
                x=x_norm, y=y_norm,
                mode='lines', line=dict(dash='dot', color='black'),
                showlegend=False
            ), row=row_mode, col=2 + idx)

    fig.update_layout(
        title=f"{sottocarico} | Context {context} | Cluster {cluster}",
        template="plotly_white",
        title_x=0.5,
        height=300 + 350 * total_rows,  # aumenta proporzionalmente
        showlegend=True
    )
    fig.show()



if __name__ == "__main__":
    plot_residuals(
        case_study="Cabina",
        sottocarico="Rooftop 3",
        context=3,
        cluster=4,
        penalty=100
    )
