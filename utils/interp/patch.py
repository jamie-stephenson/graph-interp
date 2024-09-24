"""Helper functions for patching"""

from models import Transformer

from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint 
import plotly.express as px
import torch
from torch import Tensor
import einops
from jaxtyping import Float
from tqdm import tqdm

from typing import Optional, Callable, Union, Sequence, List
from functools import partial
import itertools

def patch_head_acts(
    corrupted_head_outputs: Float[Tensor, "batch n_vertices n_heads d_head"],
    hook: HookPoint,
    head_ids: List[int],
    clean_cache: ActivationCache
) -> Float[Tensor, "batch n_vertices n_heads d_head"]:
    '''
    Hook function that patches the activations (sepcified by `hook`) of a given head at
    every vertex position, using the value from the clean cache.
    '''
    corrupted_head_outputs[:, :, head_ids] = clean_cache[hook.name][:, :, head_ids]
    
    return corrupted_head_outputs

def patch_entire_act(
    old_act: Tensor,
    hook: HookPoint,
    new_act: Tensor
) -> Tensor:
    '''
    Hook function that patches the entire activation at `hook`.
    Probably a more natural way to implement this using transformer_lens but will take me longer to find it than to write this.
    '''
    return new_act

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

def get_path_patched_attn_pattern(
    model: Transformer,
    corrupted_input: Float[Tensor,"1 n_vertices n_vertices"],
    corrupted_cache: ActivationCache,
    clean_input: Float[Tensor,"batch n_vertices n_vertices"],
    clean_cache: ActivationCache,
    q_heads_to_patch: List[int],
    q_patch_embed: bool,
    k_heads_to_patch: List[int],
    k_patch_embed: bool,
    layer_1_head: int
 ) -> Float[Tensor, "n_vertices n_vertices"]:
    """
    "Simultaneously" patch paths from two sets of components to the query and key of a layer 1 head.
    Return the resulting attention pattern from that head.
    """

    corrupted_input = einops.repeat(
        corrupted_input,
        "1 n_vertices_a n_vertices_b -> batch_size n_vertices_a n_vertices_b",
        batch_size = clean_input.shape[0]
    )

    q_layer_0_heads_to_freeze = [i for i in range(model.cfg.n_heads) if i not in q_heads_to_patch]
    k_layer_0_heads_to_freeze = [i for i in range(model.cfg.n_heads) if i not in k_heads_to_patch]
    
    if q_patch_embed:
        # If we are patching the embed and freezing some heads in layer 0 this is the same as using clean input and patching the same heads in layer 0 
        q_input = clean_input
        q_layer_0_hook_fn = partial(patch_head_acts, head_ids=q_layer_0_heads_to_freeze, clean_cache=corrupted_cache)
    else:
        q_input = corrupted_input
        q_layer_0_hook_fn = partial(patch_head_acts, head_ids=q_heads_to_patch, clean_cache=clean_cache)

    if k_patch_embed:
        # If we are patching the embed and freezing some heads in layer 0 this is the same as using clean input and patching the same heads in layer 0 
        k_input = clean_input
        k_layer_0_hook_fn = partial(patch_head_acts, head_ids=k_layer_0_heads_to_freeze, clean_cache=corrupted_cache)
    else:
        k_input = corrupted_input
        k_layer_0_hook_fn = partial(patch_head_acts, head_ids=k_heads_to_patch, clean_cache=clean_cache)

    model.reset_hooks()

    q_cache = model.add_caching_hooks('blocks.1.attn.hook_q')

    model.run_with_hooks(
        q_input,
        fwd_hooks=[('blocks.0.attn.hook_z',q_layer_0_hook_fn)]
    )

    model.reset_hooks()

    q_patch_hook_fn = partial(patch_head_acts, head_ids=layer_1_head, clean_cache=q_cache)
    pattern_cache = model.add_caching_hooks('blocks.1.attn.hook_pattern')

    model.run_with_hooks(
        k_input,
        fwd_hooks=[
            ('blocks.0.attn.hook_z',k_layer_0_hook_fn),
            ('blocks.1.attn.hook_q',q_patch_hook_fn)
        ]
    )

    return pattern_cache['blocks.1.attn.hook_pattern'][:,layer_1_head].squeeze(0).detach()

def get_pos_embed_to_key_path_patched_attn_pattern(
    model: Transformer,
    input: Float[Tensor,"batch n_vertices n_vertices"],
    cache: ActivationCache,
    new_pos_embed: Float[Tensor,"1 n_vertices d_model"],
    layer_1_head: int
 ) -> Float[Tensor, "n_vertices n_vertices"]:
    
    model.reset_hooks()

    pos_embed_hook_fn = partial(patch_entire_act, new_act=new_pos_embed)
    z_hook_fn = partial(patch_head_acts, head_ids=list(range(model.cfg.n_heads)), clean_cache=cache)
    q_hook_fn = partial(patch_head_acts, head_ids=layer_1_head, clean_cache=cache)

    pattern_cache = model.add_caching_hooks('blocks.1.attn.hook_pattern')

    model.run_with_hooks(
        input,
        fwd_hooks=[
            ('hook_pos_embed',pos_embed_hook_fn),
            ('blocks.0.attn.hook_z',z_hook_fn),
            ('blocks.1.attn.hook_q',q_hook_fn)
        ]
    )

    return pattern_cache['blocks.1.attn.hook_pattern'][:,layer_1_head].squeeze(0).detach()

def get_path_patching_metric_results(
    model: Transformer,
    corrupted_input: Float[Tensor, "batch n_vertices n_vertices"],
    corrupted_cache: ActivationCache,
    clean_input: Float[Tensor, "batch n_vertices n_vertices"],
    clean_cache: ActivationCache,
    patching_metric: Callable,
    layer_1_head: int,
    disable_k_heads: bool = True
) -> dict:
    '''
    Calculates the `patching_metric` when the paths from every combination of components to
    `layer_1_head`s query and key inputs are patched using the `clean_cache`.

    Returns a dict where keys are a tuples containing the tuples of components patched into the query and key,
    and values are the patching metric when those specific paths are patched.

    By default we avoid patching the paths from layer 0 heads to layer 1 keys, you can specify `disable_k_heads = False` 
    to stop this.

    If `clean_input` and `clean_cache` are for a batch of input graphs, then all metric results will be average across the batch.
    '''

    heads = list(range(model.cfg.n_heads))
    head_combinations = []

    for r in range(0, len(heads) + 1):
        head_combinations.extend(itertools.combinations(heads, r))

    component_combinations = [
        (patch_embed,*head_ids) 
        for patch_embed in (True, False)
        for head_ids in head_combinations
    ]    

    k_combinations = [(True,),(False,)] if disable_k_heads else component_combinations

    results = {
        (q_components,k_components) : None
        for q_components in component_combinations
        for k_components in k_combinations
    }

    for q_components in tqdm(component_combinations):
        for k_components in k_combinations:
            pattern = get_path_patched_attn_pattern(
                model,
                corrupted_input,
                corrupted_cache,
                clean_input,
                clean_cache,
                q_heads_to_patch = q_components[1:],
                q_patch_embed = q_components[0],
                k_heads_to_patch = k_components[1:],
                k_patch_embed = k_components[0],
                layer_1_head = layer_1_head
            ) 

            results_key = (q_components,k_components) 

            results[results_key] = patching_metric(
                pattern,
                clean_cache['blocks.1.attn.hook_pattern'][:,layer_1_head],
                corrupted_cache['blocks.1.attn.hook_pattern'][:,layer_1_head]
            )

    return results
