import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from types import SimpleNamespace
from pathlib import Path
import argparse
import yaml

def generate_dataset(
        n: int,
        start: int,
        end: int,
        path: Path,
        train_size: int,   
        val_size: int,
        normalized_train: bool = True,
        normalized_val: bool = True,
        cutoff: int = 500_000
    ):

    """
    Generate two splits of random graphs. 
    Graphs will be sampled from G(n,m) for each m in [start,end).
    Each value of m will have an equal number of corresponding graphs sampled
    (Therefore number of m values from `start` to `end` must divide both split sizes).
    Adjacency matrices and planarity labels will be saved to their respective split
    files in the directory with the given `path`.
    """
    
    split_args = list(zip(
        ["train","val"],
        [train_size,val_size],
        [normalized_train,normalized_val]
    ))

    # Check sizes before starting any generation:
    for split,size,_ in split_args:
        assert size%2==0,f"{split}_size must be even"
        assert size%(end-start)==0,(
            f"number of m values ({end-start}, from {start} to {end}) must " 
            f"divide {split}_size ({size}), maybe try {2*(end-start)*(size//(2*(end-start)))}."
        )

    for split,size,normalized in split_args:

        split_path = path/Path(f"{split}.npz")

        if normalized:
            _generate_normalized_split(n,split,start,end,split_path,size,cutoff)
        else:
            _generate_natural_split(n,split,start,end,split_path,size)


def _generate_normalized_split(
        n: int,
        split: str,
        start: int,
        end: int,
        path: Path,
        size: int,   
        cutoff: int
    ):
        """
        Generate a single split of random graphs. 
        Graphs will be sampled from G(n,m) for each m in [start,end).
        Each value of m will have an equal number of corresponding graphs sampled
        (Therefore number of m values from `start` to `end` must divide both split sizes).
        Adjacency matrices and planarity labels will be saved to a file at the given `path`.

        The split will be normalized, meaning that for each m, half the graphs sampled will
        be planar. This requires a use of a `cutoff` to account for values of m for which
        planarity is very unlikely (or impossible) or very likely (or guaranteed) resulting
        in very long (or infinite) loops.
        """

        data = np.empty((size,n,n))
        labels = np.zeros((size))
        m_size = size//(end-start) # Number of graphs needed for each m value

        n_added = 0
        for i,m in tqdm(enumerate(range(start,end)),desc=f"generating {split} split..."):
            n_planar = 0
            n_non_planar = 0
            n_iter = 0
            while n_added < (i+1)*m_size and n_iter < cutoff:
                g = nx.gnm_random_graph(n, m)
                is_planar = nx.is_planar(g)
                if is_planar and n_planar < m_size/2:
                    data[n_added] = nx.to_numpy_array(g)
                    labels[n_added] = is_planar
                    n_planar += 1
                    n_added += 1
                elif not is_planar and n_non_planar < m_size/2:
                    data[n_added] = nx.to_numpy_array(g)
                    labels[n_added] = is_planar
                    n_non_planar += 1
                    n_added += 1
                n_iter += 1
            assert labels.sum() == n_added/2
            assert n_iter < cutoff, (
                "Cutoff reached, please adjust start and end so "
                "that the probability of planarity is less extreme."
            )
            np.savez(path,data=data,labels=labels)


def _generate_natural_split(
        n: int,
        split: str,
        start: int,
        end: int,
        path: Path,
        size: int
    ):
        """
        Generate a single split of random graphs. 
        Graphs will be sampled from G(n,m) for each m in [start,end).
        Each value of m will have an equal number of corresponding graphs sampled
        (Therefore number of m values from `start` to `end` must divide both split sizes).
        Adjacency matrices and planarity labels will be saved to a file at the given `path`.

        The split will not be normalized, meaning that for each m the number of planar graphs
        present is representetive of G(n,m).
        """
        
        data = np.empty((size,n,n))
        labels = np.empty((size))
        m_size = size//(end-start) # Number of graphs needed for each m value

        for i,m in tqdm(enumerate(range(start,end)),desc=f"generating {split} split..."):
            for j in range(m_size):
                g = nx.gnm_random_graph(n, m)
                data[i*m_size+j] = nx.to_numpy_array(g)
                labels[i*m_size+j] = nx.is_planar(g)

        np.savez(path,data=data,labels=labels)

def get_stats(n,start,end,m_sample_size,plot: bool = False):
    """
    Generate `m_sample_size` random graphs G~G(n,m)
    for each m in [start,end) and record how many
    are planar.
    """
    stats = np.zeros((end-start))
    for i,m in tqdm(enumerate(range(start,end))):
        for _ in range(m_sample_size):
            g = nx.gnm_random_graph(n, m)
            is_planar = nx.is_planar(g)
            stats[i] += is_planar

    if plot:
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(start,end), stats, color='skyblue')

        plt.title('Planar Counts')
        plt.xlabel('m')
        plt.ylabel('Number of planar graphs in sample')

        plt.show()

    return stats

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

    path.mkdir(parents=True)

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
