import json
import os
from settings import PROJECT_ROOT
from pyecharts import options as opts
from pyecharts.charts import Tree


case_study = "Total"
output_file = os.path.join(PROJECT_ROOT, "results", case_study, "viz", "load_tree.html")

def convert_to_tree_format(name, subtree, is_root=False):
    if not subtree:
        return {
            "name": name,
            "itemStyle": {"color": "#f4cccc"}
        }
    else:
        children = [convert_to_tree_format(k, v) for k, v in subtree.items()]
        return {
            "name": name,
            "children": children,
            "itemStyle": {"color": "forestgreen" if is_root else "#97C2FC"}
        }

with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json")) as f:
    data = json.load(f)

tree_data = convert_to_tree_format("Total", data["Load Tree"]["Total"], is_root=True)
c = (
    Tree(init_opts=opts.InitOpts(width="100%", height="100vh"))
    .add(
        series_name="",
        data=[tree_data],
        collapse_interval=2,
        orient="TB",
        label_opts=opts.LabelOpts(
            position="top",
            vertical_align="bottom",
            horizontal_align="center",
            font_size=16,
            font_weight="bold",
        ),
        leaves_opts=opts.TreeLeavesOpts(
            label_opts=opts.LabelOpts(
                position="bottom",
                horizontal_align="center",
                vertical_align="top",
                font_size=16,
                font_weight="bold",
            )
        ),
        layout="orthogonal",
        pos_left="0%",
        pos_right="0%",
        pos_top="5%",
        pos_bottom="5%",
        symbol_size=30,
    )
    .set_series_opts(roam=True, symbol_size=10)
)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
c.render(output_file)
