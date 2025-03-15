from typing import List, Dict, Tuple
from mmdet.apis import DetInferencer
from mmpose.apis import inference_topdown, init_model
import cv2
import numpy as np
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from utils import FeatureGenerator, labels2kpts, read_feature, read_label, FarnebackFeatureGenerator
from math import floor


class DatasetGenerator:
    def __init__(self, config: Dict):
        self.ds_config = config
        self.n_kpts = len(config['kpts'])
        self.selected_ids = config['selected_ids']
        self.angle_pairs = config['angle_pairs']
        self.features = config['features']
        self.feature_generator = FeatureGenerator(config)
    
    def generate_stat_dataset(self, w_size: int, n_subwindows: int, save_path: str, overlap: float = 0.3) -> None:
        data = []
        labels_path = self.ds_config['labels']
        class_names = self.ds_config['classes']
        root_path = self.ds_config['root']
        for id, cl in tqdm(enumerate(class_names), total=len(class_names)):
            label_list = os.listdir(f"{root_path}/{cl}/{labels_path}/")  # get txt paths of saved labels
            for path in tqdm(label_list, total=len(label_list)):
            #for path in label_list:
                label, _ = read_label(f"{root_path}/{cl}/{labels_path}/{path}")  # read label as np.ndarray
                label = labels2kpts(label, self.n_kpts, 'bbox' in self.features)  # strip frame number and kpts scores
                n_frames = label.shape[0]  # total number of frames
                step = floor((1 - overlap) * w_size)  # step of the sliding window
                n_steps = (n_frames - w_size) // step  + 1  # number of steps for label
                # print('\n', n_frames)
                # print(step)
                # print(n_steps)
                for i in range(n_steps):
                    f_matrix = self.feature_generator.build_feature_matrix(label[i * step : i * step + w_size].copy())
                    f_vector = self.feature_generator.build_stat_feature_vector(f_matrix, n_subwindows).tolist()
                    f_vector.append(id)
                    data.append(f_vector)

        data = np.array(data)
        #print(data[:, -1])
        np.save(save_path, data)


class AutoLabelActions:
    def __init__(self, config: Dict, 
               mmdet_config: Dict={'model': '/home/cv-worker/dmitrii/weights/ratdet/rtmdet_s_8xb32-300e_ratdataset.py',
                                    'weights': '/home/cv-worker/dmitrii/weights/ratdet/best_200.pth'},
                mmpose_config: Dict={'model': '/home/cv-worker/dmitrii/weights/topviewrodents/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_topviewrodents.py',
                                     'weights': '/home/cv-worker/dmitrii/weights/topviewrodents/best_180.pth'}):
        self.ds_config = config
        self.bbox_detector = DetInferencer(model=mmdet_config['model'], 
                           weights=mmdet_config['weights'])
        self.kpts_detector = init_model(mmpose_config['model'],
                   mmpose_config['weights'], device='cuda')        
    
    def label_bbox_kpts(self) -> None:
        root = self.ds_config['root']
        videos_path = self.ds_config['videos']
        labels_path = self.ds_config['labels']
        classes = self.ds_config['classes']
        for cl in tqdm(classes, total=len(classes)):
            videos = os.listdir(f"{root}/{cl}/{videos_path}")
            os.makedirs(f"{root}/{cl}/{labels_path}", exist_ok=True)
            for video in tqdm(videos, total=len(videos)):
                out = []  # output list with string labels
                cap = cv2.VideoCapture(f"{root}/{cl}/{videos_path}/{video}")
                i = 0  # frame number
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    H, W = frame.shape[:2]
                    result = self.bbox_detector(frame)  # detect rat
                    bbox = result['predictions'][0]['bboxes'][0]  # get bbox
                    x1, y1, x2, y2 = np.array(bbox).astype(int)  # extract roi
                    kpts = inference_topdown(self.kpts_detector, frame.copy(), bboxes=np.array(bbox)[None, ...]) # detect kpts
                    kpts = kpts[0].pred_instances  # extract kpts
                    scores = kpts.keypoint_scores[0]  # get kpts x, y and scores
                    keypoints = kpts.keypoints[0]
                    out_string = str(i)  # string for output
                    out_string = out_string + f" {x1 / W :.3f} {y1 / H :.3f} {x2 / W :.3f} {y2 / H :.3f}"
                    for pt, score in zip(keypoints, scores):  # append kpts
                        out_string = out_string + f" {pt[0] / W :.3f} {pt[1] / H :.3f} {score :.3f}"
                    out_string = out_string + '\n'

                    out.append(out_string)
                    i += 1
                out.insert(0, f"{W} {H}\n")  # insert width and height in the head of file
                with open(f"{root}/{cl}/{labels_path}/{video.replace('.mp4', '.txt')}", 'w') as f:
                    f.writelines(out)

    def label_flow_farneback(self, of_params: Dict, flow_shape: Tuple[int, int], 
                             crop_bbox: bool = True) -> None:
        root = self.ds_config['root']
        videos_path = self.ds_config['videos']
        labels_path = self.ds_config['labels']
        classes = self.ds_config['classes']
        of_path = self.ds_config['optical_flow']
        for cl in tqdm(classes, total=len(classes)):
            videos = sorted(os.listdir(f"{root}/{cl}/{videos_path}"))
            os.makedirs(f"{root}/{cl}/{of_path}", exist_ok=True)
            for video in tqdm(videos, total=len(videos)):
                cap = cv2.VideoCapture(f"{root}/{cl}/{videos_path}/{video}")
                _, frame = cap.read()
                label, _ = read_label(f"{root}/{cl}/{labels_path}/{video.replace('.mp4', '.txt')}")
                W, H = flow_shape
                label = label[1:, 1:5] * np.array(flow_shape * 2)
                label = label.astype(int)
                fg = FarnebackFeatureGenerator(of_params, flow_shape, frame, label.shape[0])
                #print(f"{root}/{cl}/{videos_path}/{video}")
                for box in label:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, int(x1 - w*0.2))
                    x2 = min(W, int(x2 + w*0.2))
                    y1 = max(0, int(y1 - h*0.2))
                    y2 = min(H, int(y2 + h*0.2))
                    _, frame = cap.read()  # read frame
                    if crop_bbox:
                        out = fg.step(frame, roi_shape=(256, 256), bbox=np.array((x1, y1, x2, y2)))
                    else:
                        out = fg.step(frame)
                np.save(f"{root}/{cl}/{of_path}/{video.replace('.mp4', '.npy')}", out.round(2).transpose((2, 0, 1)))

    def label_bbox_kpts_features(self) -> None:
        root = self.ds_config['root']
        labels_path = self.ds_config['labels']
        classes = self.ds_config['classes']
        kpts_path = self.ds_config['kpts_features']
        n_kpts = len(self.ds_config['kpts'])
        features = self.ds_config['features']
        feature_generator = FeatureGenerator(self.ds_config)
        for cl in tqdm(classes, total=len(classes)):
            labels = sorted(os.listdir(f"{root}/{cl}/{labels_path}"))
            os.makedirs(f"{root}/{cl}/{kpts_path}", exist_ok=True)
            for label in tqdm(labels, total=len(labels)):
                bbox_kpts, _ = read_label(f"{root}/{cl}/{labels_path}/{label}")
                bbox_kpts = labels2kpts(bbox_kpts, n_kpts, 'bbox' in features)  # strip frame number and kpts scores
                f_matrix = feature_generator.build_feature_matrix(bbox_kpts)
                np.save(f"{root}/{cl}/{kpts_path}/{label.replace('.txt', '.npy')}", f_matrix)
                
