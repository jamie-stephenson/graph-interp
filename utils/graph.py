from .interp import plot_attention

import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from jaxtyping import Float

class Graph:
    
    def __init__(self, graph: nx.Graph | Float[Tensor,"n_vertices n_vertices"]):
        if isinstance(graph,nx.Graph):
            self.G = graph
            self.X = torch.tensor(nx.adjacency_matrix(graph).toarray(),dtype=torch.float32).unsqueeze(0) # batch dimension for convenience
        elif isinstance(graph,Tensor):
            graph.squeeze_(0)
            self.G = nx.from_numpy_array(graph.numpy(force=True))
            self.X = graph.unsqueeze(0) # squeeze+unsqueeze guarantees correct shape for `.numpy()`:(a,b), and `self.X`:(1,a,b).
        else:
            raise TypeError(f"Unsupported type: {type(graph)}")

    def plot(
        self, 
        pos: dict | None = None,
        node_color='lightblue', 
        edge_color='gray', 
        node_size=500, 
        font_size=10
    ):
        nx.draw(
            self.G, 
            with_labels=True, 
            pos=pos, 
            node_color=node_color, 
            edge_color=edge_color, 
            node_size=node_size, 
            font_size=font_size
        )
        plt.show()

    def plot_attention(self, attention: Float[Tensor,"n_heads n_vertices n_vertices"]):
        plot_attention(self.G,attention)
    