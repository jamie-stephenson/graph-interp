from utils.model import ModelConfig, HookedLinear

from transformer_lens.hook_points import HookedRootModule
from transformer_lens.utils import init_kaiming_normal_
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from jaxtyping import Float
import einops

from typing import Optional, Union, Tuple
import logging

class MLP(HookedRootModule):
    def __init__(self,cfg: ModelConfig):
        super().__init__()#TODO initialise layer weights correctly
        self.cfg = cfg

        layer_in = HookedLinear(cfg.n_vertices**2,cfg.d_model,torch.float32)

        hidden_layers = []
        for _ in range(cfg.n_layers): # here we use n_layers to represent number of hidden layers
            hidden_layers.append(nn.GELU())
            hidden_layers.append(HookedLinear(cfg.d_model,cfg.d_model,torch.float32))

        hidden_layers.append(nn.GELU())

        layer_out = HookedLinear(cfg.d_model,cfg.d_vocab_out,torch.float32)
        
        self.layers = nn.Sequential(layer_in,*hidden_layers,layer_out).to(cfg.device)

        if self.cfg.init_weights:
            self._init_weights_kaiming()

        self.setup()

    def forward(
        self,
        input: Float[Tensor, "batch n_vertices n_vertices"],
        label: Optional[Tensor] = None, # `bool` is usecase specific may need to change
        return_type: Optional[str] = "logits",
    ) -> Union[
        None, # none
        Float[Tensor, "batch d_vocab_out"], # logits
        Float[Tensor, ""] , # loss
        Tuple[Float[Tensor, "batch d_vocab_out"], Float[Tensor, ""] ], # both
    ]:
        x = einops.rearrange(input,"a b c -> a (b c)") # flatten
        logits = self.layers(x)

        if return_type is None:
            return None
        else:
            if return_type == "logits":
                return logits
            else:
                assert (
                    label is not None
                ), "label must be passed in if return_type is 'loss' or 'both'"
                loss = F.cross_entropy(logits, label)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return logits, loss
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None
    
    def _init_weights_kaiming(self):
        gain = self.cfg.initializer_range
        for name, param in self.named_parameters():
            if "W" in name:
                init_kaiming_normal_(param, gain=gain, nonlinearity="relu", mode="fan_in")

