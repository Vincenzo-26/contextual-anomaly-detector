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
    soft_evidence_dir = os.path.join(PROJECT_ROOT, "results", case_study, "soft_evidences")
    time_window_path = os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")

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

    x_pos_counter = [0]
    positions = {}

    def assign_positions(node, depth=0):
        children = get_children_of_node(tree, node)
        if not children:
            x = x_pos_counter[0]
            positions[node] = (x, -depth * y_spacing)
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

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color="green", size=10),
        name="Thermal sensitive",
        showlegend=True,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Non thermal sensitive",
        showlegend=True,
        hoverinfo="skip"
    ))

    for node, (x, y) in positions.items():
        p0 = row.get(f"P({node}=0)", None)
        p1 = row.get(f"P({node}=1)", None)
        if p0 is not None and p1 is not None:
            tooltip = (
                f"<b>{node}</b><br>"
                f"P({node}=0): {p0:.3f}<br>"
                f"P({node}=1): {p1:.3f}<extra></extra>"
            )
            fig.add_trace(go.Bar(
                x=[x - 0.1], y=[p0 * bar_height_scale],
                width=bar_width, base=y,
                marker_color="steelblue",
                hovertemplate=tooltip,
                showlegend=False
            ))
            fig.add_trace(go.Bar(
                x=[x + 0.1], y=[p1 * bar_height_scale],
                width=bar_width, base=y,
                marker_color="crimson",
                hovertemplate=tooltip,
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[x], y=[y + label_offset],
                text=[node], mode="text",
                showlegend=False, hoverinfo="skip"
            ))

        if node in foglie:
            path_csv = os.path.join(soft_evidence_dir, f"soft_evidence_{node}.csv")
            if os.path.exists(path_csv):
                df_soft = pd.read_csv(path_csv)
                row_soft = df_soft[(df_soft["Date"] == date) & (df_soft["Context"] == int(context))]
                if not row_soft.empty:
                    thermal = bool(row_soft.iloc[0]["thermal_sensitive"])
                    color = "green" if thermal else "red"
                    name = "Thermal sensitive" if thermal else "Non thermal sensitive"
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y + label_offset - 0.3],
                        mode="markers",
                        marker=dict(color=color, size=10),
                        name=name,
                        showlegend=False,
                        hovertemplate=f"{name}<extra></extra>"
                    ))

    for parent, (x1, y1) in positions.items():
        children = get_children_of_node(tree, parent)
        for child in children:
            if child in positions:
                x0, y0 = positions[child]
                p0 = row.get(f"P({child}=0)", 0)
                p1 = row.get(f"P({child}=1)", 0)
                max_bar_height = max(p0, p1) * bar_height_scale
                y_start = y0 + max_bar_height + arrow_start_offset
                y_end = y1 + label_offset - arrow_end_offset
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y_start, y_end],
                    mode="lines",
                    line=dict(color="gray", width=1),
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
            x=0.5
        )
    )

    fig.show()

if __name__ == "__main__":
    plot_inference_result(case_study="Cabina", date="2024-09-03", context=2)
