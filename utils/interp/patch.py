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

def patch_head_outputs(
    corrupted_head_outputs: Float[Tensor, "batch n_vertices n_heads d_head"],
    hook: HookPoint,
    head_ids: List[int],
    clean_cache: ActivationCache
) -> Float[Tensor, "batch n_vertices n_heads d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
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
    hook_fn = partial(patch_head_outputs, head_ids=head_ids, clean_cache=clean_cache)
    model.run_with_hooks(
        corrupted_input,
        fwd_hooks=[('blocks.0.attn.hook_z',hook_fn)]
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