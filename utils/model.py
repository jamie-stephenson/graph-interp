from transformer_lens import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.utilities.addmm import batch_addmm
from transformer_lens.components import Embed
import torch
from torch import nn, Tensor
from jaxtyping import Float

from dataclasses import dataclass
from typing import Union, Dict

@dataclass(kw_only=True)
class ModelConfig(HookedTransformerConfig):
    """
    Hacky wrapper to add: 
    - clearer contextual names (e.g.`n_vertices`)
    - more suitable defaults (e.g. "bidirectional")
    - use as a config for more generic models (e.g. MLP)
    """
    # clearer names
    n_vertices: int

    model: str = "transformer"

    n_ctx: int = None

    # mlps do not require `d_head` and `act_fn` but __post_init__ does
    d_head: int = 1 
    act_fn: str = "gelu"

    def __post_init__(self):
        self.n_ctx = self.n_vertices

        # More appropriate defaults
        self.attention_dir = "bidirectional"
        self.normalization_type = None
        super().__post_init__()

class TokenPool(nn.Module):
    def forward(
        self, 
        input:Float[Tensor, "batch n_ctx d_model"]
        )-> Float[Tensor, "batch d_model"]:
        return input[:,0]

class MeanPool(nn.Module):
    def forward(
        self, 
        input:Float[Tensor, "batch n_ctx d_model"]
        )-> Float[Tensor, "batch d_model"]:
        return input.mean(dim=1)
    
class HookedLinear(nn.Module):

    def __init__(self,size_in,size_out,dtype):
        super().__init__()
        self.W = nn.Parameter(torch.empty(size_in, size_out, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(size_out, dtype=dtype))
        self.hook_pre = HookPoint()

    def forward(
        self, x: Float[Tensor, "batch pos d_model"]
    ) -> Float[Tensor, "batch pos d_model"]:

        return batch_addmm(self.b, self.W, self.hook_pre(x))
    
class GraphEmbed(nn.Module):
    """
    Embed the adjacency matrix by "looking up" the embeddings for the adjacent vertices
    and summing those embeddings (i.e. multiply adjacency matrix by W_E) 
    """
    def __init__(self, cfg: Union[Dict, ModelConfig]):
        super().__init__()
        self.cfg = ModelConfig.unwrap(cfg)
        self.W_E: Float[torch.Tensor, "n_vertices d_model"] = nn.Parameter(
            torch.empty(self.cfg.n_vertices, self.cfg.d_model, dtype=self.cfg.dtype)
        )

    def forward(
        self, input: Float[Tensor, "batch n_vertices n_vertices"]
    ) -> Float[torch.Tensor, "batch n_vertices d_model"]:
        return input @ self.W_E
    
# for laplacian??
def deactivate_position(model):
    model.pos_embed.W_pos.data[:] = 0.0
    model.pos_embed.W_pos.requires_grad = False 