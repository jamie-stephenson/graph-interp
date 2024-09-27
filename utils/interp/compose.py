"""Helper functions for interpreting component composition"""

import plotly.subplots
from models import Transformer

from transformer_lens import ActivationCache
import plotly
import plotly.express as px
import plotly.graph_objects as go
import torch
from torch import Tensor
import einops
from jaxtyping import Float
import numpy as np

from typing import Optional, List

def decompose_qk_input(model: Transformer, cache: ActivationCache, layer: int = 1) -> Float[Tensor, "batch n_components n_vertices d_model"]:

    assert layer in [0,1], f"Only supported layers are 0 and 1, not {layer}."

    y_embed = cache["embed"].unsqueeze(dim=1) # (batch 1 n_vertices d_model)
    y_pos = cache["pos_embed"].unsqueeze(dim=1) # (batch 1 n_vertices d_model)

    if layer == 1:
        y_heads = cache["result", 0].transpose(1, 2) # (batch n_heads n_vertices d_model)
        y_stack = torch.cat([y_embed, y_pos, y_heads], dim=1)

        mlp = model.blocks[0].mlp
        decomposed_qk_input = mlp(y_stack) + y_stack # Not very principled!
    else:
        decomposed_qk_input = torch.cat([y_embed, y_pos], dim=1)

    return decomposed_qk_input

def decompose_q(
    model: Transformer,
    decomposed_qk_input: Float[Tensor, "batch n_components n_vertices d_head"],
    head_index: int,
    layer: int = 1
) -> Float[Tensor, "batch n_components n_vertices d_head"]:

    W_Q = model.W_Q[layer, head_index]

    return einops.einsum(
        decomposed_qk_input, W_Q,
        "batch n_components n_vertices d_model, d_model d_head -> batch n_components n_vertices d_head"
    )

def decompose_k(
    model: Transformer,
    decomposed_qk_input: Float[Tensor, "batch n_components n_vertices d_head"],
    head_index: int,
    layer: int = 1
) -> Float[Tensor, "batch n_components n_vertices d_head"]:

    W_K = model.W_K[layer, head_index]

    return einops.einsum(
        decomposed_qk_input, W_K,
        "batch n_components n_vertices d_model, d_model d_head -> batch n_components n_vertices d_head"
    )

def decompose_attn_scores(
    decomposed_q: Float[Tensor, "batch n_components n_vertices d_head"],
    decomposed_k: Float[Tensor, "batch n_components n_vertices d_head"]
) -> Float[Tensor, "batch query_component key_component query_pos key_pos"]:
    
    d_head = decomposed_q.shape[-1]

    return einops.einsum(
        decomposed_q, decomposed_k,
        "batch q_comp q_pos d_model, batch k_comp k_pos d_model -> batch q_comp k_comp q_pos k_pos",
    )/d_head**0.5

def plot_decomposed_attn_scores(
    decomposed_scores: Float[Tensor, "batch query_component key_component query_pos key_pos"],
    cache: ActivationCache,
    component_labels: List[str],
    head_idx: int,
    layer: int = 1,
    batch_sample_idx: int = 0,
    zmax: float = 80.,
):
    """
    Given batch, will take one sample graph and plot the attention scores for the given head.
    Then plots the given decomposed scores for that head and sample graph.
    Then plots the standard deviation of the decomposed scores, averaged out across the whole batch.
    """
    
    assert len(cache['attn_scores',layer].shape) == 4,(
        "Cached pattern has incorrect shape, perhaps you forgot to include a batch dimension?"
    )

    n_components = len(component_labels)

    decomposed_stds = einops.reduce(
        decomposed_scores,
        "batch query_decomp key_decomp query_pos key_pos -> batch query_decomp key_decomp",
        torch.std
    )

    px.imshow(
        cache['attn_scores',layer][batch_sample_idx,head_idx].detach().cpu(),
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0,
        title="Original Sample Attention Scores",
        width=610
    ).show()

    fig = plotly.subplots.make_subplots(
        rows=n_components, 
        row_titles=component_labels,
        y_title='Query Components',
        cols=n_components,
        column_titles=component_labels,
        x_title='Key Components'
    )

    for i in range(n_components):
        for j in range(n_components):

            heatmap = go.Heatmap(
                z=decomposed_scores[batch_sample_idx, i, j].detach().cpu(),
                colorscale='RdBu',
                zmax=zmax,
                zmin=-zmax
            )
            fig.add_trace(heatmap, row=i+1, col=j+1)
            fig.update_yaxes(autorange='reversed')

    fig.update_layout(height=155*n_components+300, width=155*n_components+300)

    fig.show()

    # std dev over query and key positions, shown by component. Mean over whole batch
    px.imshow(
        decomposed_stds.mean(0).detach().cpu(),
        labels={"x": "Key Component", "y": "Query Component"},
        title="Standard deviations of attention score contributions (by key and query component)",
        x=component_labels,
        y=component_labels,
        color_continuous_scale='Blues',
        width=610
    ).show()


def get_comp_score(
    W_A: Float[Tensor, "in_A out_A"],
    W_B: Float[Tensor, "out_A out_B"]
) -> float:

    W_A_norm = W_A.pow(2).sum().sqrt()
    W_B_norm = W_B.pow(2).sum().sqrt()
    W_AB_norm = (W_A @ W_B).pow(2).sum().sqrt()

    return (W_AB_norm / (W_A_norm * W_B_norm)).item()

def plot_comp_scores(
    model, 
    comp_scores, 
    component_labels, 
    title: str = "", 
    baseline: Optional[Tensor] = None
) -> go.Figure:
    
    px.imshow(
        comp_scores.cpu(),
        y=component_labels,
        x=[f"1.{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title=title,
        color_continuous_scale="RdBu" if baseline is not None else "Blues",
        color_continuous_midpoint=baseline if baseline is not None else None,
        zmin=None if baseline is not None else 0.0,
    ).show()

def generate_single_random_comp_score(model) -> float:
    '''
    Generates a single composition score for random matrices
    '''
    W_A_left = torch.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_left = torch.empty(model.cfg.d_model, model.cfg.d_head)
    W_A_right = torch.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_right = torch.empty(model.cfg.d_model, model.cfg.d_head)

    for W in [W_A_left, W_B_left, W_A_right, W_B_right]:
        torch.nn.init.kaiming_uniform_(W, a=np.sqrt(5))

    W_A = W_A_left @ W_A_right.T
    W_B = W_B_left @ W_B_right.T

    return get_comp_score(W_A, W_B)