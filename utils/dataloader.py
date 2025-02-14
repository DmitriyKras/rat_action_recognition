import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict
import os
from tqdm import tqdm
from .augmentation import KeypointsAugmentor


class LSTMDataset(Dataset):
    def __init__(self, config: Dict, subset: str, cl: str, augment: bool = False):
        self.config = config
        self.root = config['root'] + '/lstm_dataset'
        self.subset = subset
        self.cl = cl
        self.data = np.load(f"{self.root}/{subset}/{cl}.npy", mmap_mode='r')  # load npy per class in mmap mode
        self.augment = augment
        self.augmentor = KeypointsAugmentor(config)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.data[idx]
        X, y = torch.from_numpy(data[:, :-1].copy()), torch.tensor(data[0, -1])
        if self.augment:
            X = self.augmentor.augment(X)
        return X.float(), y.long()
