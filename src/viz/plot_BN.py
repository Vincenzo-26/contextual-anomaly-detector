from pyvis.network import Network
import webbrowser
import tempfile
import os
import sys

# Aggiunge il root del progetto al PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

# Import dal pacchetto src
from src.bayesian_network import build_BN_structural_model

def visualize_bn_interactive(model):
    """
    Visualizza una rete bayesiana in modo interattivo usando PyVis (senza salvare permanentemente il file).
    """
    net = Network(height="600px", width="100%", directed=True, notebook=False)
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

    # Aggiungi nodi e archi
    for node in model.nodes():
        net.add_node(node, label=node)
    for edge in model.edges():
        net.add_edge(edge[0], edge[1])

    # net.show_buttons(filter_=['physics']) #per impostare i parametri a mano

    # Genera file HTML temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        temp_path = tmp_file.name

    net.write_html(temp_path)
    webbrowser.open("file://" + os.path.realpath(temp_path))


if __name__ == "__main__":
    model = build_BN_structural_model("Cabina")
    visualize_bn_interactive(model)
