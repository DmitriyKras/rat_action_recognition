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


class LSTMDataset(Dataset):
    def __init__(self, config: Dict, subset: str, cl: str, augment: bool = False):
        super().__init__()
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
        
        n = self.offset + idx * self.step
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
        frames = self.data[n * 2 : n * 2 + 2 * self.seq_length].copy()
        frames = torch.from_numpy(frames).float()
        return (frame, frames), self.cl


def build_conv3d_dataset(config: Dict, input_shape: Tuple[int, int],
                         w_size: int, overlap: float = 0, offset: int = 0) -> Tuple[ConcatDataset, ConcatDataset]:
    root = config['root']
    videos = config['videos']
    labels = config['labels']
    classes = config['classes']
    train_ds, val_ds = [], []
    for cl_id, cl in enumerate(classes):
        v_dir = f"{root}/{cl}/{videos}"
        l_dir = f"{root}/{cl}/{labels}"
        videos_list = sorted(os.listdir(v_dir))
        labels_list = sorted(os.listdir(l_dir))
        train_videos, val_videos, train_labels, val_labels = train_test_split(videos_list, labels_list, test_size=0.2)
        print(f"Number of videos in train set of class {cl_id}: {cl} is {len(train_videos)}")
        print(f"Number of videos in val set of class {cl_id}: {cl} is {len(val_videos)}")
        train_ds.extend([Conv3DDataset(f"{v_dir}/{vid}", f"{l_dir}/{label}", input_shape, cl_id, w_size, overlap, offset) 
                         for vid, label in zip(train_videos, train_labels)])
        val_ds.extend([Conv3DDataset(f"{v_dir}/{vid}", f"{l_dir}/{label}", input_shape, cl_id, w_size, overlap, offset) 
                         for vid, label in zip(val_videos, val_labels)])
    
    return ConcatDataset(train_ds), ConcatDataset(val_ds)


def build_flow_dataset(config: Dict, seq_length: int = 10, overlap: float = 0, 
                       offset: int = 0) -> Tuple[ConcatDataset, ConcatDataset]:
    root = config['root']
    optical_flow = config['optical_flow']
    classes = config['classes']
    train_ds, val_ds = [], []
    for cl_id, cl in enumerate(classes):
        flow_list = os.listdir(f"{root}/{cl}/{optical_flow}")
        train_flow, val_flow = train_test_split(flow_list, test_size=0.2)
        print(f"Number of videos in train set of class {cl_id}: {cl} is {len(train_flow)}")
        print(f"Number of videos in val set of class {cl_id}: {cl} is {len(val_flow)}")
        train_ds.extend([OpticalFlowDataset(f"{root}/{cl}/{optical_flow}/{flow}", cl_id, seq_length, overlap, offset) 
                         for flow in train_flow])
        val_ds.extend([OpticalFlowDataset(f"{root}/{cl}/{optical_flow}/{flow}", cl_id, seq_length, overlap, offset) 
                         for flow in val_flow])
    
    return ConcatDataset(train_ds), ConcatDataset(val_ds)


def build_conv2d_dataset(config: Dict, input_shape: Tuple[int, int],
                         offset: int = 0) -> Tuple[ConcatDataset, ConcatDataset]:
    root = config['root']
    videos = config['videos']
    labels = config['labels']
    classes = config['classes']
    train_ds, val_ds = [], []
    for cl_id, cl in enumerate(classes):
        v_dir = f"{root}/{cl}/{videos}"
        l_dir = f"{root}/{cl}/{labels}"
        videos_list = sorted(os.listdir(v_dir))
        labels_list = sorted(os.listdir(l_dir))
        train_videos, val_videos, train_labels, val_labels = train_test_split(videos_list, labels_list, test_size=0.2)
        print(f"Number of videos in train set of class {cl_id}: {cl} is {len(train_videos)}")
        print(f"Number of videos in val set of class {cl_id}: {cl} is {len(val_videos)}")
        train_ds.extend([Conv2DDataset(f"{v_dir}/{vid}", f"{l_dir}/{label}", input_shape, cl_id, offset) 
                         for vid, label in zip(train_videos, train_labels)])
        val_ds.extend([Conv2DDataset(f"{v_dir}/{vid}", f"{l_dir}/{label}", input_shape, cl_id, offset) 
                         for vid, label in zip(val_videos, val_labels)])
    
    return ConcatDataset(train_ds), ConcatDataset(val_ds)


def build_lstm_dataset(config: Dict) -> Tuple[ConcatDataset, ConcatDataset]:
    train_ds = ConcatDataset([LSTMDataset(config, 'train', cl) for cl in config['classes']])
    val_ds = ConcatDataset([LSTMDataset(config, 'val', cl) for cl in config['classes']])
    return train_ds, val_ds


def build_two_stream_dataset(config: Dict, input_shape: Tuple[int, int], step: int,  seq_length: int = 10, 
                         offset: int = 0) -> Tuple[ConcatDataset, ConcatDataset]:
    root = config['root']
    videos = config['videos']
    labels = config['labels']
    classes = config['classes']
    optical_flow = config['optical_flow']
    train_ds, val_ds = [], []
    for cl_id, cl in enumerate(classes):
        v_dir = f"{root}/{cl}/{videos}"
        l_dir = f"{root}/{cl}/{labels}"
        f_dir = f"{root}/{cl}/{optical_flow}"
        videos_list = sorted(os.listdir(v_dir))
        labels_list = sorted(os.listdir(l_dir))
        flow_list = sorted(os.listdir(f_dir))
        train_videos, val_videos, train_labels, val_labels, \
        train_flows, val_flows = train_test_split(videos_list, labels_list, flow_list, test_size=0.2)
        print(f"Number of videos in train set of class {cl_id}: {cl} is {len(train_videos)}")
        print(f"Number of videos in val set of class {cl_id}: {cl} is {len(val_videos)}")
        train_ds.extend([TwoStreamDataset(f"{v_dir}/{vid}", f"{l_dir}/{label}", f"{f_dir}/{flow}", cl_id, input_shape, 
                                          step, seq_length, offset) 
                         for vid, label, flow in zip(train_videos, train_labels, train_flows)])
        val_ds.extend([TwoStreamDataset(f"{v_dir}/{vid}", f"{l_dir}/{label}", f"{f_dir}/{flow}", cl_id, input_shape, 
                                          step, seq_length, offset) 
                         for vid, label, flow in zip(val_videos, val_labels, val_flows)])
    return ConcatDataset(train_ds), ConcatDataset(val_ds)
