import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List
import os
from tqdm import tqdm


class LSTMDataset(Dataset):
    def __init__(self, root: str, subset: str, cl: str):
        self.root = root
        self.subset = subset
        self.cl = cl
        self.data = np.load(f"{root}/{subset}/{cl}.npy", mmap_mode='r')  # load npy per class in mmap mode

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.data[idx]

        return torch.from_numpy(data[:, :-1].copy()).float(), torch.tensor(data[0, -1]).long()
