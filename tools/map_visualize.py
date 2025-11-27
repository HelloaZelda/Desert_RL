import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

edges = json.load(open(DATA_DIR / "map_edges.json"))

G = nx.Graph()
for n, nbrs in edges.items():
    for v in nbrs:
        G.add_edge(n, v)

plt.figure(figsize=(18,18))
nx.draw(G, with_labels=True, node_size=1200, font_size=10)
plt.show()
