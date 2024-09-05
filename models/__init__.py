from .transformer import Transformer
from .mlp import MLP
from utils.model import ModelConfig

from wandb.sdk.wandb_config import Config
from transformer_lens import HookedTransformerConfig

from inspect import signature

MODEL_MAP = {
    'transformer': Transformer,
    'mlp': MLP
}

def get_model(cfg: Config | ModelConfig):

    model_class = MODEL_MAP[cfg.model]

    if isinstance(cfg, Config):
        # I can't get my wandb sweep to support nested config dict
        # so all model params have to be a top level param which
        # we filter as follows:

        # Dataclasses don't seem to inherit full signature? So we combine both signatures:
        cfg_signature = {
            **signature(ModelConfig).parameters,
            **signature(HookedTransformerConfig).parameters
        }

        cfg = ModelConfig(
            **{
                key: value
                for key, value in cfg.items()
                if key in cfg_signature
            }
        )

    return model_class(cfg)