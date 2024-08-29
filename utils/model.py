from torch import nn, Tensor
from jaxtyping import Float

# for laplacian??
def deactivate_position(model):
    model.pos_embed.W_pos.data[:] = 0.0
    model.pos_embed.W_pos.requires_grad = False

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