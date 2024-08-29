from models import get_model
from utils import train, get_dataloader

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import wandb

from types import SimpleNamespace
from datetime import datetime
import argparse
import os
import yaml


def main(args):

    if not args.no_wandb:
        wandb.init(project='aisf',name=args.run_name,config=args)
        wandb.define_metric("Effective Batch Number") 

    model = get_model(args.model_name,args.model_config)
    train_dataloader = get_dataloader(args.dataset_format,args.train_path,args.batch_size)
    eval_dataloader = get_dataloader(args.dataset_format,args.val_path,args.batch_size)

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
    torch.save(model.state_dict(),f"trained_models/{args.run_name}.pt")
    with open(f"trained_models/{args.run_name}.yaml",'w') as file:
        yaml.dump(vars(args), file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # args I want to set from command line:
    parser.add_argument(
        "--no-wandb",
        "--no_wandb",
        action="store_true",
        help="If set, wandb logging is disabled."
    )

    parser.add_argument(
        "--config",
        default="configs/adj_config.yaml",
        type=str,
        help="Path to config file."
    )
    
    clargs = parser.parse_args()

    # args I want to set from a file:
    with open(clargs.config, 'r') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    
    args = SimpleNamespace(**yaml_config)
    args.no_wandb = clargs.no_wandb
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.run_name = f"{args.model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    args.dataset_format = args.model_name # Each model is differentiated by its input data format

    torch.manual_seed(args.seed) 
    
    main(args)
