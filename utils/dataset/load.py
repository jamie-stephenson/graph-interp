import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class AdjDataset(Dataset):
    """Dataset containing graphs represented by adjacency matrix"""
    def __init__(self,file):
        dataset = np.load(file)
        self.data = torch.tensor(dataset['data'],dtype=torch.float32)
        self.labels = torch.tensor(dataset['labels'],dtype=torch.int64)

    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    
    def __len__(self):
        return len(self.labels)

def get_dataloader(file,batch_size:int):

    dataset = AdjDataset(file)
    
    return DataLoader(dataset,batch_size,shuffle=True)