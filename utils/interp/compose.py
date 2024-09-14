"""Helper functions for interpreting component composition"""

from models import Transformer

from transformer_lens import ActivationCache
import plotly.express as px
import plotly.graph_objects as go
import torch
from torch import Tensor
import einops
from jaxtyping import Float
import numpy as np

from typing import Optional

def decompose_qk_input(model: Transformer, cache: ActivationCache) -> Float[Tensor, "batch n_components n_vertices d_model"]:

    y_embed = cache["embed"].unsqueeze(dim=1) # (batch 1 n_vertices d_model)
    y_pos = cache["pos_embed"].unsqueeze(dim=1) # (batch 1 n_vertices d_model)
    y_heads = cache["result", 0].transpose(1, 2) # (batch n_heads n_vertices d_model)
    y_stack = torch.cat([y_embed, y_pos, y_heads], dim=1)

    mlp = model.blocks[0].mlp

    return mlp(y_stack) + y_stack

def decompose_q(
    decomposed_qk_input: Float[Tensor, "batch n_components n_vertices d_head"],
    head_index: int,
    model: Transformer,
) -> Float[Tensor, "batch n_components n_vertices d_head"]:

    W_Q = model.W_Q[1, head_index]

    return einops.einsum(
        decomposed_qk_input, W_Q,
        "batch n_components n_vertices d_model, d_model d_head -> batch n_components n_vertices d_head"
    )

def decompose_k(
    decomposed_qk_input: Float[Tensor, "batch n_components n_vertices d_head"],
    head_index: int,
    model: Transformer,
) -> Float[Tensor, "batch n_components n_vertices d_head"]:

    W_K = model.W_K[1, head_index]

    return einops.einsum(
        decomposed_qk_input, W_K,
        "batch n_components n_vertices d_model, d_model d_head -> batch n_components n_vertices d_head"
    )

def decompose_attn_scores(
    decomposed_q: Float[Tensor, "batch n_components n_vertices d_head"],
    decomposed_k: Float[Tensor, "batch n_components n_vertices d_head"]
) -> Float[Tensor, "batch query_component key_component query_pos key_pos"]:
    
    return einops.einsum(
        decomposed_q, decomposed_k,
        "batch q_comp q_pos d_model, batch k_comp k_pos d_model -> batch q_comp k_comp q_pos k_pos",
    )

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
        comp_scores,
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