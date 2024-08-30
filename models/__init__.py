import importlib
from wandb.sdk.wandb_config import Config

def get_model(model_format, args: Config):
    
    model_module = importlib.import_module(f".{model_format}", package="models")
    model = model_module.get_model(args)

    return model