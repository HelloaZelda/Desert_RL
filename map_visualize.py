import json
import networkx as nx
import matplotlib.pyplot as plt

edges = json.load(open("map_edges.json"))

G = nx.Graph()
for n, nbrs in edges.items():
    for v in nbrs:
        G.add_edge(n, v)

plt.figure(figsize=(18,18))
nx.draw(G, with_labels=True, node_size=1200, font_size=10)
plt.show()
