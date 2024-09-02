from utils.model import MeanPool, ModelConfig

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint 
from transformer_lens.components import TransformerBlock, Unembed
from transformer_lens.utilities import devices
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from jaxtyping import Float, Int

import logging
from typing import Optional, Tuple, Dict, Union

class Transformer(HookedTransformer):
    """
    Transformer designed to take a graph's adjacency matrix as input.
    """
    def __init__(
        self,
        cfg: Union[ModelConfig, Dict],
        move_to_device: bool = True,
    ):
        """Model initialization.

        Args:
            cfg: The config to use for the model.
            move_to_device: Whether to move the model to the device specified in cfg.
                device. Must be true if `n_devices` in the config is greater than 1, since the
                model's layers will be split across multiple devices.
        """
        # Init from HookedRootModule
        super(HookedTransformer, self).__init__()

        self.cfg = ModelConfig.unwrap(cfg)

        self.embed = nn.Linear(self.cfg.n_vertices,self.cfg.d_model)
        self.hook_embed = HookPoint()  

        self.pos_embed = nn.Embedding(self.cfg.n_ctx,self.cfg.d_model)
        self.hook_pos_embed = HookPoint() 

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.cfg, block_index) for block_index in range(self.cfg.n_layers)]
        )

        self.pool =  MeanPool()
        
        self.unembed = Unembed(self.cfg) 

        if self.cfg.init_weights:
            self.init_weights()

        if move_to_device:
            self.move_model_modules_to_device()

        self.dataset = None

        self.setup()

    def forward(
        self,
        input: Float[Tensor, "batch n_ctx n_vertices"],
        label: Optional[Tensor] = None, # `bool` is usecase specific may need to change
        return_type: Optional[str] = "logits",
        shortformer_pos_embed: Optional[Float[Tensor, "batch n_ctx d_model"]] = None,
        start_at_layer: Optional[int] = None,
        stop_at_layer: Optional[int] = None,
    ) -> Union[
        None, # none
        Float[Tensor, "batch d_vocab_out"], # logits
        Float[Tensor, ""] , # loss
        Tuple[Float[Tensor, "batch d_vocab_out"], Float[Tensor, ""] ], # both
    ]:
        
        if len(input.shape) == 2:
            # Add a dummy batch dimension to avoid things breaking.
            input = input[None]

        if start_at_layer is None:

            if self.cfg.pool_type == "token":
                # Add special class token to begining, our logits will be calculated from this position.
                batch_size, _, n_vertices = input.shape
                planar_token = torch.zeros((batch_size,1,n_vertices),device=self.cfg.device)
                input = torch.cat([planar_token,input], dim=1)

            embed = self.hook_embed(self.embed(input))
            pos_embed = self.hook_pos_embed(self.pos_embed(torch.arange(self.cfg.n_ctx,device=self.cfg.device)))
            residual = embed + pos_embed

        else:
            assert type(input) == Tensor
            residual = input

        if start_at_layer is None:
            start_at_layer = 0

        blocks_and_idxs = list(zip(range(self.cfg.n_layers), self.blocks))
        for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:

            residual = residual.to(devices.get_device_for_block_index(i, self.cfg))

            if shortformer_pos_embed is not None:
                shortformer_pos_embed = shortformer_pos_embed.to(
                    devices.get_device_for_block_index(i, self.cfg)
                )
                
            residual = block(residual,shortformer_pos_embed=shortformer_pos_embed) # [batch, pos, d_model]

        if stop_at_layer is not None:
            return residual
        
        if return_type is None:
            return None
        else:
            logits = self.unembed(self.pool(residual))  # [batch, d_vocab_out]

            if return_type == "logits":
                return logits
            else:
                assert (
                    label is not None
                ), "label must be passed in if return_type is 'loss' or 'both'"
                loss = self.loss_fn(logits, label)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return logits, loss
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None
                
    def loss_fn(
        self,
        logits: Float[Tensor, "batch d_vocab_out"],
        label: Int[Tensor, "batch"],
    ):
        return F.cross_entropy(logits, label)




