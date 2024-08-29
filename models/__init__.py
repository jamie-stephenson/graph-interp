import importlib

def get_model(name, model_config):
    
    model_module = importlib.import_module(f".{name}", package="models")
    model = model_module.get_model(model_config)

    return model