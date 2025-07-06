import plotly.graph_objects as go
from src.utils import *

def plot_inference_result(case_study, date, context):
    y_spacing = 4
    bar_height_scale = 0.8
    label_offset = -0.2
    bar_width = 0.2
    arrow_start_offset = 0.1
    arrow_end_offset = 0.2

    config_path = os.path.join(PROJECT_ROOT, "data", case_study, "config.json")
    results_path = os.path.join(PROJECT_ROOT, "results", case_study, "inference_results.csv")
    time_window_path = os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")

    ctx_thermal_sens_path = os.path.join(PROJECT_ROOT, "results", case_study, "thermal_sensitivity", "ctx_thermal_sens")

    # Carichi thermal sensitive da nome dei file
    thermal_sensitive_nodes = {
        os.path.splitext(f)[0]
        for f in os.listdir(ctx_thermal_sens_path)
        if f.endswith(".csv")
    }

    df_descr = pd.read_csv(time_window_path)
    context_description = df_descr[df_descr["id"] == int(context)].iloc[0]["description"]

    with open(config_path, "r") as f:
        config = json.load(f)

    df = pd.read_csv(results_path)
    df_row = df[(df["Date"] == date) & (df["Context"] == int(context))]
    if df_row.empty:
        raise ValueError(f"Nessuna riga trovata per Date={date} e Context={context}")
    row = df_row.iloc[0]

    tree = config["Load Tree"]
    levels = get_nodes_by_level(tree)[::-1]
    foglie = find_leaf_nodes(tree)

    x_pos_counter = [1]
    positions = {}

    def assign_positions(node, depth=0):
        children = get_children_of_node(tree, node)
        if not children:
            x = x_pos_counter[0]
            positions[node] = (x, -depth * y_spacing + 1)
            x_pos_counter[0] += 1
        else:
            for child in children:
                assign_positions(child, depth + 1)
            child_xs = [positions[child][0] for child in children]
            x = sum(child_xs) / len(child_xs)
            positions[node] = (x, -depth * y_spacing)

    root = list(tree.keys())[0]
    assign_positions(root)

    fig = go.Figure()

    # Dummy legends
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", marker=dict(color="green", size=10),
        name="Thermal sensitive", showlegend=True, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", marker=dict(color="red", size=10),
        name="Non thermal sensitive", showlegend=True, hoverinfo="skip"
    ))

    # Bar plot + node label
    for node, (x, y) in positions.items():
        p0 = row.get(f"P({node}=0)", None)
        p1 = row.get(f"P({node}=1)", None)
        if p0 is not None and p1 is not None:
            tooltip = (
                f"P({node}=0): {p0:.2f}<br>"
                f"P({node}=1): {p1:.2f}<extra></extra>"
            )
            fig.add_trace(go.Bar(
                x=[x - 0.1], y=[p0 * bar_height_scale],
                width=bar_width, base=y,
                marker_color="steelblue", hovertemplate=tooltip, showlegend=False
            ))
            fig.add_trace(go.Bar(
                x=[x + 0.1], y=[p1 * bar_height_scale],
                width=bar_width, base=y,
                marker_color="crimson", hovertemplate=tooltip, showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[x], y=[y + label_offset], text=[node],
                mode="text", textfont=dict(size=14), showlegend=False, hoverinfo="skip"
            ))

    # Leaf node markers: thermal sensitive = green, otherwise red
    for node in foglie:
        if node in positions:
            x, y = positions[node]
            if node in thermal_sensitive_nodes:
                color = "green"
                name = "Thermal sensitive"
            else:
                color = "red"
                name = "Non thermal sensitive"

            fig.add_trace(go.Scatter(
                x=[x], y=[y + label_offset - 0.5],
                mode="markers",
                marker=dict(color=color, size=10),
                name=name,
                showlegend=False,
                hovertemplate=f"{name}<extra></extra>"
            ))

    # Add edges (lines between nodes)
    for parent, (x1, y1) in positions.items():
        children = get_children_of_node(tree, parent)
        for child in children:
            if child in positions:
                x0, y0 = positions[child]
                p0 = row.get(f"P({child}=0)", None)
                p1 = row.get(f"P({child}=1)", None)
                if p0 is not None and p1 is not None and not (np.isnan(p0) or np.isnan(p1)):
                    max_bar_height = max(p0, p1) * bar_height_scale
                    line_dash = "solid"
                else:
                    max_bar_height = 0
                    line_dash = "dash"

                y_start = y0 + max_bar_height + arrow_start_offset
                y_end = y1 + label_offset - arrow_end_offset

                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y_start, y_end],
                    mode="lines",
                    line=dict(color="gray", width=1, dash=line_dash),
                    hoverinfo='skip',
                    showlegend=False
                ))

    fig.update_layout(
        title=f"{date} | Context {context} - {context_description}",
        title_x=0.5,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        height=800,
        margin=dict(t=60, b=80, l=20, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        )
    )

    fig.show()

if __name__ == "__main__":
    plot_inference_result(case_study="Total_cut", date="2024-08-26", context=2)
