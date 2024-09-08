from utils.dataset.build import generate_dataset, generate_natural_split, generate_normalized_split

from types import SimpleNamespace
from pathlib import Path
import argparse
import yaml

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # args I want from the command line:
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        type=Path,
        help="Path to config file."
    )
    
    clargs = parser.parse_args()

    # args I want to set from a file:
    with open(clargs.config, 'r') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    
    args = SimpleNamespace(**yaml_config)

    # Make informative directory name for dataset: 
    normalized_train_string = "_tn" if args.normalized_train else ""
    normalized_val_string ="_vn" if args.normalized_val else ""
    path = Path(f"data/n{args.n}_{args.start}-{args.end}{normalized_train_string}{normalized_val_string}")

    generate_dataset(
        n=args.n,
        start=args.start,
        end=args.end,
        path=path,
        train_size=args.train_size,
        val_size=args.val_size,
        normalized_train=args.normalized_train,
        normalized_val=args.normalized_val,
        cutoff=args.cutoff
    )
