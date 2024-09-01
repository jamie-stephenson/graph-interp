import networkx as nx
import torch
import matplotlib.pyplot as plt

def graph_to_adj(g: nx.classes.graph.Graph) -> torch.Tensor:
    return torch.tensor(nx.adjacency_matrix(g).toarray(),dtype=torch.float)

def adj_to_graph(adj: torch.Tensor) -> nx.classes.graph.Graph:
    return nx.from_numpy_array(adj.numpy())

def plot_graph(g: nx.classes.graph.Graph):
    plt.figure(figsize=(8, 6))
    nx.draw(g, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("Random Graph")
    plt.show()