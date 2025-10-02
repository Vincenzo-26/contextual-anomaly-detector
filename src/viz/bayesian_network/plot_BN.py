from pyvis.network import Network
import webbrowser
import tempfile
import os
import json

from src.utils import find_leaf_nodes
from settings import PROJECT_ROOT
from src.bayesian_network import build_BN_structural_model


def visualize_bn_interactive(model, case_study:str):
    """
    Visualizza interattivamente una rete bayesiana
    """

    net = Network(height="100vh", width="100vw", directed=True, notebook=False)
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.005,
          "springLength": 200,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 1
        },
        "minVelocity": 0.75
      },
      "layout": {
        "improvedLayout": true
      }
    }
    """)
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)
    leaves = find_leaf_nodes(config["Load Tree"])
    # nodi e archi
    for node in model.nodes():
        if node in leaves:
            net.add_node(node, label=node, color="#f4cccc", font={"size": 22})
        elif node == "Total":
            net.add_node(node, label=node, color="#d9ead3", font={"size": 22})
        else:
            net.add_node(node, label=node, color="#97C2FC", font={"size": 22})
    for edge in model.edges():
        net.add_edge(edge[0], edge[1])

    # net.show_buttons(filter_=['physics']) #per impostare i parametri a mano

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        temp_path = tmp_file.name

    net.write_html(temp_path)
    webbrowser.open("file://" + os.path.realpath(temp_path))


if __name__ == "__main__":
    case_study = "Total"
    model = build_BN_structural_model(case_study)
    visualize_bn_interactive(model, case_study)
