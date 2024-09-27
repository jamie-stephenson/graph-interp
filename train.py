from models import get_model
from utils import train, get_dataloader

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import wandb

from datetime import datetime
import argparse
import os
import yaml

def main(
    wandb_mode: str = "online",
    config_file: str | None = None
):

    run = wandb.init(
        project='aisf',
        config=config_file,
        mode=wandb_mode
    )

    args = wandb.config

    run_name = f"{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if wandb_mode != "disabled": # "disabled" will be removed from wandb TODO: find alternative
        run.name = run_name

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model = get_model(args)
    train_dataloader = get_dataloader(args.train_path,args.batch_size)
    eval_dataloader = get_dataloader(args.val_path,args.batch_size)

    optimizer = AdamW(
        params=model.parameters(),
        betas=(args.beta1,args.beta2),
        weight_decay=args.weight_decay, 
        fused='cuda'==args.device
    )

    lr_scheduler = OneCycleLR(
        optimizer, 
        args.lr_max, 
        pct_start=args.lr_pct_start,
        epochs=args.epochs, 
        steps_per_epoch=len(train_dataloader)
    ) 
    
    model = train(model,train_dataloader,eval_dataloader,optimizer,lr_scheduler,args)

    os.makedirs("trained_models",exist_ok=True)
    torch.save(model.state_dict(),f"trained_models/{run_name}.pt")
    with open(f"trained_models/{run_name}.yaml",'w') as file:
        yaml.dump(dict(args.items()), file) # args: wandb_config.Config requires weird handling 

    run.finish()

def parse_clargs():

    parser = argparse.ArgumentParser()

    # args I want to set from command line:
    parser.add_argument(
        "--no-wandb",
        "--no_wandb",
        action="store_true",
        help="If set, wandb logging is disabled."
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "If set, perform hyperparameter sweep.`--config` "
            "arg must be path to yaml file containing valid wandb sweep config. "
            "This file does not need to be a valid sweep file for running "
            "sweeps from the command line, it just needs to give a valid "
            "sweep config dict when loaded using `yaml.safe_load`."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/model.yaml",
        type=str,
        help="Path to config file."
    )
    
    clargs = parser.parse_args()

    if clargs.sweep and clargs.no_wandb:
        parser.error("The --sweep argument cannot be used when --no-wandb is set.")

    return clargs

def run_main(clargs):
    """
    Either creates the `args` needed for a single run and performs it
    or creates a sweep ID needed for a wandb sweep and performs that,
    depending on the value of `clargs.sweep`.
    """

    if clargs.sweep:

        with open(clargs.config, 'r') as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)

        sweep_id = wandb.sweep(yaml_config, project="aisf")
        wandb.agent(sweep_id, main)

    else:

        main(
            wandb_mode = "disabled" if clargs.no_wandb else "online",
            config_file = clargs.config
        )

if __name__ == '__main__':

    clargs = parse_clargs()
    run_main(clargs)

 
    

