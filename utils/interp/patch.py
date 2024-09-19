"""Helper functions for patching"""

from models import Transformer

from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint 
import plotly.express as px
import torch
from torch import Tensor
from jaxtyping import Float

from typing import Optional, Callable, Union, Sequence, List
from functools import partial

def patch_head_acts(
    corrupted_head_outputs: Float[Tensor, "batch n_vertices n_heads d_head"],
    hook: HookPoint,
    head_ids: List[int],
    clean_cache: ActivationCache
) -> Float[Tensor, "batch n_vertices n_heads d_head"]:
    '''
    Patches the activations (sepcified by `hook`) of a given head at
    every vertex position, using the value from the clean cache.
    '''
    corrupted_head_outputs[:, :, head_ids] = clean_cache[hook.name][:,:, head_ids]
    
    return corrupted_head_outputs

def run_with_patched_heads(
    model: Transformer,
    corrupted_input: Float[Tensor, "batch n_vertices n_vertices"],
    clean_cache: ActivationCache,
    head_ids: List[int]
):
    hook_fn = partial(patch_head_acts, head_ids=head_ids, clean_cache=clean_cache)
    model.run_with_hooks(
        corrupted_input,
        fwd_hooks=[('blocks.0.attn.hook_z',hook_fn)]
    )


def run_with_patched_head_path(
    model: Transformer,
    corrupted_input: Float[Tensor,"batch n_vertices n_vertices"],
    corrupted_cache: ActivationCache,
    clean_cache: ActivationCache,
    layer_0_heads: List[int],
    layer_1_head: int,
    comp_types: str
 ) -> Float[Tensor, "layer head"]:
    """"
    Patch the path from a collection of heads in layer 0 to a single input ('Q', 'K' or 'V')
    or multiple inputs ('QK', 'QV' etc.) of a single head in layer 1.
    """
    allowed_types = set('qkv')
    comp_types = set(comp_types.lower())
    assert comp_types.issubset(allowed_types), "Invalid comp type, must be 'Q', 'K' or 'V' or a combination of them."
    assert comp_types!=allowed_types, "Path patching to all head inputs is the same as activation patching, please do that instead."

    # for each input type, we need to know which heads to freeze, start by assuming we are freezing them all...
    heads_to_freeze = {input_type: list(range(model.cfg.n_heads)) for input_type in allowed_types}

    # and then remove the given head from input types any that we don't want to freeze  
    for comp_type in comp_types:
        heads_to_freeze[comp_type].pop(layer_1_head)

    patch_layer_0_heads_hook_fn = partial(patch_head_acts, head_ids=layer_0_heads, clean_cache=clean_cache)   # patch activations
    freeze_layer_1_q_hook_fn = partial(patch_head_acts, head_ids=heads_to_freeze['q'], clean_cache=corrupted_cache) # freeze activations
    freeze_layer_1_k_hook_fn = partial(patch_head_acts, head_ids=heads_to_freeze['k'], clean_cache=corrupted_cache) # freeze activations
    freeze_layer_1_v_hook_fn = partial(patch_head_acts, head_ids=heads_to_freeze['v'], clean_cache=corrupted_cache) # freeze activations

    model.run_with_hooks(
        corrupted_input,
        fwd_hooks=[
            ('blocks.0.attn.hook_z',patch_layer_0_heads_hook_fn),
            ('blocks.1.attn.hook_q',freeze_layer_1_q_hook_fn),
            ('blocks.1.attn.hook_k',freeze_layer_1_k_hook_fn),
            ('blocks.1.attn.hook_v',freeze_layer_1_v_hook_fn)
        ]
    )


def rank_heads_by_patching_metric(
    model: Transformer,
    corrupted_input: Float[Tensor, "batch n_vertices n_vertices"],
    clean_cache: ActivationCache,
    patching_metric: Callable,
    patching_names: Optional[Union[Callable[[str], bool], Sequence[str], str]] 
) -> Float[Tensor, "batch n_vertices n_vertices"]:
    '''
    Patches an increasing set of heads, each time adding the head that has the best effect
    on the `patching_metric`. Displays heads in the order they were added along with the
    metric score the achieved along with all previously added heads.

    The `patching_metric` function should be called on the model's patched and clean caches. 
    The patched cache is created by caching activations specified by `patching_names`.
    '''
    model.reset_hooks()

    patched_cache = model.add_caching_hooks(patching_names)

    importance_ranking = []
    metric_list = []

    heads = list(range(model.cfg.n_heads))

    while heads:

        min_metric = 100
        min_head = -1

        for head in heads:
            ids = importance_ranking + [head]
            
            run_with_patched_heads(model,corrupted_input,clean_cache,ids)

            metric = patching_metric(patched_cache,clean_cache)
            if metric < min_metric:
                min_metric = metric
                min_head = head

        metric_list.append(min_metric)
        importance_ranking.append(heads.pop(heads.index(min_head)))
    
    px.line(x=[str(i) for i in importance_ranking], y=metric_list, labels={'x': 'Head', 'y': 'Pattern Reconstruction Metric'}).show()