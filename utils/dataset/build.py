import networkx as nx
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from jaxtyping import Float,Int

from pathlib import Path
from typing import Tuple, Callable, Union

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
    files in the directory with the given `path`. You can choose to "normalize" either
    split by ensuring that, for each m value, exactly half the graphs that are added to 
    the dataset are planar.
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

    path.mkdir(parents=True)

    for split,size,normalized in split_args:

        split_path = path/Path(f"{split}.npz")

        if normalized:
            generate_normalized_split(n,split,start,end,split_path,size,cutoff)
        else:
            generate_natural_split(n,split,start,end,split_path,size)


def generate_normalized_split(
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
        for i,m in enumerate(tqdm(range(start,end),desc=f"generating {split} split")):
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


def generate_natural_split(
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

        for i,m in enumerate(tqdm(range(start,end),desc=f"generating {split} split")):
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
    for i,m in enumerate(tqdm(range(start,end))):
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

def _generate_graphs_with_property(
    predicate: Callable[[nx.Graph,bool], bool],
    n: int,
    start: int,
    end: int,
    size: int,
    return_labels: bool = False,
    tqdm_desc: str = "generating graphs",
    path: Path | None = None,
    cutoff: int = 100_000
) -> Union[
        Float[Tensor,"size n n"],
        Tuple[Float[Tensor,"size n n"],Int[Tensor,"size"]]
    ]:

    """
    Generate a set of graphs with a certain property. 
    Graphs will be sampled from G(n,m) until `size//(end-start)` graphs, `g` (with `is_planar=nx.is_planar(g)`), 
    have been sampled such that `predicate(g,is_planar)==True`. This is repeated for each m in [start,end).
    Each value of m will have an equal number of corresponding graphs sampled
    (Therefore number of m values from `start` to `end` must divide both split sizes).
    Returns adjacency matrices (and optionally labels) in torch tensor which will be saved as numpy array 
    to a file at the given `path`. A `cutoff` is needed incase the [start,end) includes values for which 
    the property is very unlikely. 
    """

    assert size%(end-start)==0,(
        f"number of m values ({end-start}, from {start} to {end}) must " 
        f"divide `size` ({size}), maybe try `size={(end-start)*(size//(end-start))}`."
    )

    data = np.empty((size,n,n),np.float32)

    if return_labels:
        labels = np.empty((size),dtype=np.int64)

    m_size = size//(end-start) # Number of graphs needed for each m value

    n_added = 0
    for i,m in enumerate(tqdm(range(start,end),desc=tqdm_desc)):
        n_iter = 0
        while n_added < (i+1)*m_size and n_iter < cutoff:

            g = nx.gnm_random_graph(n, m)
            is_planar = nx.is_planar(g)

            if predicate(g,is_planar):
                data[n_added] = nx.to_numpy_array(g)

                if return_labels:
                    labels[n_added] = is_planar

                n_added += 1

            n_iter += 1

        assert n_iter < cutoff, (
            "Cutoff reached, please increase cutoff or adjust start and end so "
            "that the probability of your predicate `{}` evaluating to `True` is larger."
            .format(predicate.__name__)
        )

    if path:
        if return_labels:
            np.savez(path, data=data, labels=labels)
        else:
            np.save(path,data)

    data_tensor = torch.tensor(data)
    
    if return_labels:
        labels_tensor = torch.tensor(labels)
        return data_tensor, labels_tensor 
    else:
        return data_tensor


def generate_planar(
    n: int,
    start: int,
    end: int,
    size: int,
    path: Path | None = None,
    non_planar: bool = False,
    cutoff: int = 100_000
) -> Float[Tensor,"size n n"]:
    
    def planarity_predicate(g,is_planar:bool):
         """
         Returns:
         - `True` if `graph`'s planarity matches our desired planarity condition sepcified by `non_planar`
         - `False` otherwise
         """
         return is_planar != non_planar
    
    return _generate_graphs_with_property(
         predicate=planarity_predicate,
         n=n,
         start=start,
         end=end,
         size=size,
         return_labels=False,
         tqdm_desc="generating {}planar set".format("non-" if non_planar else ""),
         path=path,
         cutoff=cutoff
    )
    

def generate_misclass(
    model,
    start: int,
    end: int,
    size: int,
    path: Path | None = None,
    cutoff: int = 100_000
) -> Tuple[Float[Tensor,"size n n"],Int[Tensor,"size"]]:
    
    def is_misclassified(g,is_planar):
        x = torch.tensor(nx.adjacency_matrix(g).toarray(),dtype=torch.float32)
        logits = model(x)
        return logits.argmax()!=is_planar
    
    return _generate_graphs_with_property(
         predicate=is_misclassified,
         n=model.cfg.n_vertices,
         start=start,
         end=end,
         size=size,
         return_labels=True,
         tqdm_desc="generating misclassified set",
         path=path,
         cutoff=cutoff
    )



