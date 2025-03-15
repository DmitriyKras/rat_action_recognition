import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from typing import List, Dict, Tuple, Literal
import os
from tqdm import tqdm
from math import floor
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from .augmentation import KeypointsAugmentor
from .utils import read_label
import json


class LSTMDataset(Dataset):
    def __init__(self, lstm_path: str, cl: int, seq_length: int = 10,
                 overlap: float = 0, offset: int = 0) -> None:
        super().__init__()
        self.cl = self.cl = torch.tensor(cl)
        self.data = np.load(lstm_path, mmap_mode='r')  # load npy video in mmap mode
        n_frames = self.data.shape[0] // 2 - 2 * offset
        self.offset = offset
        self.step = floor((1 - overlap) * seq_length)  # step of the sliding window
        self.n_steps = (n_frames - seq_length) // self.step + 1 # number of steps for video
        self.seq_length = seq_length

    def __len__(self):
        return self.n_steps
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        n = idx * self.step + self.offset
        kpts = self.data[n : n + self.seq_length].copy()
        kpts = torch.from_numpy(kpts)
        return kpts.float(), self.cl
    

class OpticalFlowDataset(Dataset):
    def __init__(self, flow_path: str, cl: int, seq_length: int = 10,
                 overlap: float = 0, offset: int = 0) -> None:
        super().__init__()
        self.cl = self.cl = torch.tensor(cl)
        self.data = np.load(flow_path, mmap_mode='r')  # load npy per class in mmap mode
        n_frames = self.data.shape[0] // 2 - 2 * offset
        self.offset = offset
        self.step = floor((1 - overlap) * seq_length)  # step of the sliding window
        self.n_steps = (n_frames - seq_length) // self.step + 1 # number of steps for video
        self.seq_length = seq_length

    def __len__(self) -> int:
        return self.n_steps
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        n = (idx * self.step + self.offset) * 2
        frames = self.data[n : n + 2 * self.seq_length].copy()
        frames = torch.from_numpy(frames)
        return frames.float(), self.cl


### Dataset for one video file
class Conv3DDataset(Dataset):
    def __init__(self, video_path: str, label_path: str, input_shape: Tuple[int, int], cl: int, 
                 w_size: int, overlap: float = 0, offset: int = 0, augment: bool = False):
        super().__init__()
        self.w_size = w_size
        self.video_path = video_path
        self.cl = torch.tensor(cl)
        self.input_shape = input_shape
        self.offset = offset
        self.augment = augment
        self.labels, wh = read_label(label_path)
        self.labels = (self.labels[:, 1 : 5] * np.array(wh * 2)).astype(int)  # extract xyxy coordinates of bbox on every frame
        n_frames = self.labels.shape[0] - 2 * offset  # total number of frames to be used
        self.step = floor((1 - overlap) * w_size)  # step of the sliding window
        self.n_steps = (n_frames - w_size) // self.step + 1 # number of steps for video
        self.wh = wh  # save image width, height
        # self.transform = A.Compose([
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5)
        # ], p=0.5)

    def __read_frames(self, idx) -> List[np.ndarray]:
        n = idx * self.step + self.offset
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)  # set pointer to frame
        frames = []
        # Extract specific frames and ROIs
        for i in range(self.w_size):
            _, frame = cap.read()
            x1, y1, x2, y2 = self.labels[n + i, :]
            w, h = x2 - x1, y2 - y1
            x1, x2 = max(0, int(x1 - w * 0.2)), min(self.wh[0], int(x2 + w * 0.2))
            y1, y2 = max(0, int(y1 - h * 0.2)), min(self.wh[1], int(y2 + h * 0.2))
            frames.append(cv2.cvtColor(cv2.resize(frame[y1 : y2, x1 : x2, :], self.input_shape), cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        frames = self.__read_frames(idx)
        # if self.augment:
        #     transformed = self.transform(images=frames)
        #     frames = transformed['images']
        frames = np.array(frames)

        frames = torch.from_numpy(frames) / 255
        return torch.permute(frames.float(), (3, 0, 1, 2)), self.cl


class Conv2DDataset(Dataset):
    def __init__(self, video_path: str, label_path: str, input_shape: Tuple[int, int], cl: int, 
                 offset: int = 0, step: int = 10, augment: bool = False):
        super().__init__()
        self.video_path = video_path
        self.cl = torch.tensor(cl)
        self.input_shape = input_shape
        self.offset = offset
        self.augment = augment
        self.labels, wh = read_label(label_path)
        self.labels = (self.labels[:, 1 : 5] * np.array(wh * 2)).astype(int)  # extract xyxy coordinates of bbox on every frame
        self.n_steps = (self.labels.shape[0] - 2 * offset) // step - 1  # total number of frames to be used
        self.step = step
        self.wh = wh  # save image width, height

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        n = self.offset + idx * self.step
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)  # set pointer to frame
        _, frame = cap.read()  # read frame
        x1, y1, x2, y2 = self.labels[n, :]
        w, h = x2 - x1, y2 - y1
        x1, x2 = max(0, int(x1 - w * 0.2)), min(self.wh[0], int(x2 + w * 0.2))
        y1, y2 = max(0, int(y1 - h * 0.2)), min(self.wh[1], int(y2 + h * 0.2))
        frame = cv2.cvtColor(cv2.resize(frame[y1 : y2, x1 : x2, :], self.input_shape), cv2.COLOR_BGR2RGB)

        # if self.augment:
        #     transformed = self.transform(images=frames)
        #     frames = transformed['images']

        frame = torch.from_numpy(frame) / 255
        return torch.permute(frame.float(), (2, 0, 1)), self.cl
    

class TwoStreamDataset(Dataset):
    def __init__(self, video_path: str, label_path: str, flow_path: str, cl: int,
                 input_shape: Tuple[int, int], step: int, seq_length: int = 10,
                 offset: int = 0) -> None:
        super().__init__()
        self.video_path = video_path
        self.cl = torch.tensor(cl)
        self.input_shape = input_shape
        self.offset = offset
        self.labels, wh = read_label(label_path)
        self.data = np.load(flow_path, mmap_mode='r')  # load npy per class in mmap mode
        n_frames = self.data.shape[0] // 2
        self.labels = (self.labels[:n_frames, 1 : 5] * np.array(wh * 2)).astype(int)  # extract xyxy coordinates of bbox on every frame
        self.n_steps = (self.labels.shape[0] - 2 * offset - seq_length) // step - 1  # total number of frames to be used
        self.step = step
        self.seq_length = seq_length
        self.wh = wh  # save image width, height

    def __len__(self) -> int:
        return self.n_steps
    
    def __getitem__(self, idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        n = self.offset + idx * self.step + self.seq_length // 2
        # Load RGB frame
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)  # set pointer to frame
        _, frame = cap.read()  # read frame
        x1, y1, x2, y2 = self.labels[n, :]
        w, h = x2 - x1, y2 - y1
        x1, x2 = max(0, int(x1 - w * 0.2)), min(self.wh[0], int(x2 + w * 0.2))
        y1, y2 = max(0, int(y1 - h * 0.2)), min(self.wh[1], int(y2 + h * 0.2))
        frame = cv2.cvtColor(cv2.resize(frame[y1 : y2, x1 : x2, :], self.input_shape), cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame) / 255
        frame = torch.permute(frame.float(), (2, 0, 1))
        # Load flow
        frames = self.data[n * 2 - self.seq_length: n * 2 + self.seq_length].copy()
        frames = torch.from_numpy(frames).float()
        return (frame, frames), self.cl


def split_videos_labels_flow(config: Dict, test_size: float = 0.3) -> None:
    root = config['root']
    videos = config['videos']
    labels = config['labels']
    classes = config['classes']
    optical_flow = config['optical_flow']
    kpts = config['kpts_features']
    train_ds, val_ds = [], []
    for cl_id, cl in enumerate(classes):
        v_dir = f"{root}/{cl}/{videos}"
        l_dir = f"{root}/{cl}/{labels}"
        f_dir = f"{root}/{cl}/{optical_flow}"
        k_dir = f"{root}/{cl}/{kpts}"
        videos_list = sorted(os.listdir(v_dir))
        labels_list = sorted(os.listdir(l_dir))
        flow_list = sorted(os.listdir(f_dir))
        kpts_list = sorted(os.listdir(k_dir))
        train_videos, val_videos, train_labels, val_labels, \
        train_flows, val_flows, train_kpts, val_kpts \
        = train_test_split(videos_list, labels_list, flow_list, kpts_list, test_size=test_size)
        train_ds.extend({'video': vid, 'label': label, 'flow': flow, 'kpts_f': kpt, 'class': cl_id}
                         for vid, label, flow, kpt in zip(train_videos, train_labels, train_flows, train_kpts))
        val_ds.extend({'video': vid, 'label': label, 'flow': flow, 'kpts_f': kpt, 'class': cl_id}
                         for vid, label, flow, kpt in zip(val_videos, val_labels, val_flows, val_kpts))

    with open("/home/cv-worker/dmitrii/rat_action_recognition/train_split.json", 'w') as f:
        json.dump(train_ds, f)
    with open("/home/cv-worker/dmitrii/rat_action_recognition/val_split.json", 'w') as f:
        json.dump(val_ds, f)


def build_conv3d_dataset(config: Dict, train_json: str, val_json: str, input_shape: Tuple[int, int],
                         w_size: int, overlap: float = 0, offset: int = 0) -> Tuple[ConcatDataset, ConcatDataset]:
    root = config['root']
    videos = config['videos']
    labels = config['labels']
    classes = config['classes']
    with open(train_json, 'r') as f:
        train_meta = json.load(f)
    with open(val_json, 'r') as f:
        val_meta = json.load(f)
    
    train_ds = [Conv3DDataset(f"{root}/{classes[meta['class']]}/{videos}/{meta['video']}", 
                              f"{root}/{classes[meta['class']]}/{labels}/{meta['label']}", 
                              input_shape, meta['class'], w_size, overlap, offset) 
                         for meta in train_meta]
    
    val_ds = [Conv3DDataset(f"{root}/{classes[meta['class']]}/{videos}/{meta['video']}", 
                              f"{root}/{classes[meta['class']]}/{labels}/{meta['label']}", 
                              input_shape, meta['class'], w_size, overlap, offset) 
                         for meta in val_meta]
    
    return ConcatDataset(train_ds), ConcatDataset(val_ds)


def build_flow_dataset(config: Dict, train_json: str, val_json: str, seq_length: int = 10, overlap: float = 0, 
                       offset: int = 0) -> Tuple[ConcatDataset, ConcatDataset]:
    root = config['root']
    optical_flow = config['optical_flow']
    classes = config['classes']
    with open(train_json, 'r') as f:
        train_meta = json.load(f)
    with open(val_json, 'r') as f:
        val_meta = json.load(f)

    train_ds = [OpticalFlowDataset(f"{root}/{classes[meta['class']]}/{optical_flow}/{meta['flow']}", 
                              meta['class'], seq_length, overlap, offset) 
                         for meta in train_meta]
    
    val_ds = [OpticalFlowDataset(f"{root}/{classes[meta['class']]}/{optical_flow}/{meta['flow']}",  
                              meta['class'], seq_length, overlap, offset) 
                         for meta in val_meta]
    
    return ConcatDataset(train_ds), ConcatDataset(val_ds)


def build_conv2d_dataset(config: Dict, train_json: str, val_json: str, input_shape: Tuple[int, int],
                         offset: int = 0) -> Tuple[ConcatDataset, ConcatDataset]:
    root = config['root']
    videos = config['videos']
    labels = config['labels']
    classes = config['classes']
    
    with open(train_json, 'r') as f:
        train_meta = json.load(f)
    with open(val_json, 'r') as f:
        val_meta = json.load(f)

    train_ds = [Conv2DDataset(f"{root}/{classes[meta['class']]}/{videos}/{meta['video']}", 
                              f"{root}/{classes[meta['class']]}/{labels}/{meta['label']}", 
                              input_shape, meta['class'], offset) 
                         for meta in train_meta]
    
    val_ds = [Conv2DDataset(f"{root}/{classes[meta['class']]}/{videos}/{meta['video']}", 
                              f"{root}/{classes[meta['class']]}/{labels}/{meta['label']}", 
                              input_shape, meta['class'], offset) 
                         for meta in val_meta]
    
    return ConcatDataset(train_ds), ConcatDataset(val_ds)


def build_lstm_dataset(config: Dict, train_json: str, val_json: str, seq_length: int = 32, overlap: float = 0.2, 
                       offset: int = 0) -> Tuple[ConcatDataset, ConcatDataset]:
    root = config['root']
    kpts = config['kpts_features']
    classes = config['classes']
    with open(train_json, 'r') as f:
        train_meta = json.load(f)
    with open(val_json, 'r') as f:
        val_meta = json.load(f)

    train_ds = [LSTMDataset(f"{root}/{classes[meta['class']]}/{kpts}/{meta['flow']}", 
                              meta['class'], seq_length, overlap, offset) 
                         for meta in train_meta]
    
    val_ds = [LSTMDataset(f"{root}/{classes[meta['class']]}/{kpts}/{meta['flow']}",  
                              meta['class'], seq_length, overlap, offset) 
                         for meta in val_meta]
    
    return ConcatDataset(train_ds), ConcatDataset(val_ds)
    


def build_two_stream_dataset(config: Dict, train_json: str, val_json: str, input_shape: Tuple[int, int], 
                             step: int,  seq_length: int = 10, 
                             offset: int = 0) -> Tuple[ConcatDataset, ConcatDataset]:
    root = config['root']
    videos = config['videos']
    labels = config['labels']
    classes = config['classes']
    optical_flow = config['optical_flow']
    
    with open(train_json, 'r') as f:
        train_meta = json.load(f)
    with open(val_json, 'r') as f:
        val_meta = json.load(f)

    train_ds = [TwoStreamDataset(f"{root}/{classes[meta['class']]}/{videos}/{meta['video']}", 
                              f"{root}/{classes[meta['class']]}/{labels}/{meta['label']}",
                              f"{root}/{classes[meta['class']]}/{optical_flow}/{meta['flow']}",
                              meta['class'], input_shape, step, seq_length, offset) 
                         for meta in train_meta]
    
    val_ds = [TwoStreamDataset(f"{root}/{classes[meta['class']]}/{videos}/{meta['video']}", 
                              f"{root}/{classes[meta['class']]}/{labels}/{meta['label']}",
                              f"{root}/{classes[meta['class']]}/{optical_flow}/{meta['flow']}",
                              meta['class'], input_shape, step, seq_length, offset) 
                         for meta in val_meta]

    return ConcatDataset(train_ds), ConcatDataset(val_ds)
