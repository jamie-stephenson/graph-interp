import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class AdjDataset(Dataset):
    """Dataset containing graphs represented by adjacency matrix"""
    def __init__(self,file):
        dataset = np.load(file)
        self.data = torch.tensor(dataset['data'],dtype=torch.float32)
        self.labels = torch.tensor(dataset['labels'],dtype=torch.uint8)

    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    
    def __len__(self):
        return len(self.labels)

class LapDataset(Dataset):
    """Dataset containing graphs represented by laplacian matrix"""
    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def generate_data():
        pass

class PromptDataset(Dataset):
    """Dataset containing planarity verification prompts for each graph"""
    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def generate_data():
        pass

def get_dataloader(format:str,file,batch_size:int):
    if format == "adj":
        dataset = AdjDataset(file)
    elif format == "lap":
        raise NotImplementedError
    elif format == "gpt2":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid dataset format: {format}")
    
    return DataLoader(dataset,batch_size,shuffle=True)