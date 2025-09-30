import json
import os

from graphviz import Digraph
from settings import PROJECT_ROOT

def draw_load_tree(tree, parent=None, dot=None):
    if dot is None:
        dot = Digraph(format='png')
        dot.attr('node', shape='circle')

    for node, children in tree.items():
        dot.node(node)
        if parent:
            dot.edge(parent, node)
        draw_load_tree(children, parent=node, dot=dot)
    return dot

case_study = "Total_cut"
with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
    config = json.load(f)
output_dir = os.path.join(PROJECT_ROOT, "results", case_study)
load_tree = config["Load Tree"]
dot = draw_load_tree(load_tree)
dot.render(filename="load_tree_diagram", directory=output_dir, format="png", view=False)