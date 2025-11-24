import json
import networkx as nx
import matplotlib.pyplot as plt

def plot_trajectory(edges, trajectory):
    G = nx.Graph()
    for n, nbrs in edges.items():
        for v in nbrs:
            G.add_edge(n, v)

    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(20,20))
    nx.draw(G, pos, with_labels=True, node_size=1500, font_size=10, alpha=0.3)

    # draw trajectory
    path_edges = [(trajectory[i], trajectory[i+1]) for i in range(len(trajectory)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3)

    plt.title("AI Movement Trajectory", fontsize=18)
    plt.show()
