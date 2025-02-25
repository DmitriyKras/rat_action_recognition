from typing import List, Dict
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
                mmdet_config: Dict={'model': '/home/techtrans2/mmdetection/work_dirs/rtmdet_s_8xb32-300e_ratdataset/rtmdet_s_8xb32-300e_ratdataset.py',
                                    'weights': '/home/techtrans2/mmdetection/work_dirs/rtmdet_s_8xb32-300e_ratdataset/epoch_200.pth'},
                mmpose_config: Dict={'model': '/home/techtrans2/mmpose/work_dirs/topviewrodents/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_topviewrodents.py',
                                     'weights': '/home/techtrans2/mmpose/work_dirs/topviewrodents/best_PCK_epoch_320.pth'}):
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

    def label_img_features(self) -> None:
        preprocess = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([224, 224]),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        feature_extractor = nn.Sequential(*modules).cuda()
        feature_extractor.eval()
        root = self.ds_config['root']
        videos_path = self.ds_config['videos']
        labels_path = self.ds_config['labels']
        classes = self.ds_config['classes']
        for cl in tqdm(classes, total=len(classes)):
            videos = os.listdir(f"{root}/{cl}/{videos_path}")
            os.makedirs(f"{root}/{cl}/img_features", exist_ok=True)  # make dir for labels
            for video in tqdm(videos, total=len(videos)):
                out = []  # output list with string labels
                cap = cv2.VideoCapture(f"{root}/{cl}/{videos_path}/{video}")
                label, (W, H) = read_label(f"{root}/{cl}/{labels_path}/{video.replace('.mp4', '.txt')}")
                label = label[:, 1:5] * np.array((W, H) * 2)
                label = label.astype(int)
                for box in label:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, int(x1 - w*0.15))
                    x2 = min(W, int(x2 + w*0.15))
                    y1 = max(0, int(y1 - h*0.15))
                    y2 = min(H, int(y2 + h*0.15))
                    _, frame = cap.read()  # read frame
                    frame = frame[y1:y2, x1:x2, ::-1]  # extract roi
                    ### EXTRACT FEATURES ###
                    frame = preprocess(frame.copy())[None, ...].cuda()
                    with torch.no_grad():
                        feat = feature_extractor(frame).cpu().numpy().squeeze()
                    out_string = ' '.join(np.round(feat, 5).astype(str).tolist()) + '\n'
                    out.append(out_string)
                with open(f"{root}/{cl}/img_features/{video.replace('.mp4', '.txt')}", 'w') as f:
                    f.writelines(out)

    def label_flow_farneback(self, of_params: Dict) -> None:
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
                label, (W, H) = read_label(f"{root}/{cl}/{labels_path}/{video.replace('.mp4', '.txt')}")
                label = label[1:, 1:5] * np.array((W, H) * 2)
                label = label.astype(int)
                fg = FarnebackFeatureGenerator(of_params, (320, 320), frame, label.shape[0])
                for box in label:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, int(x1 - w*0.15))
                    x2 = min(W, int(x2 + w*0.15))
                    y1 = max(0, int(y1 - h*0.15))
                    y2 = min(H, int(y2 + h*0.15))
                    _, frame = cap.read()  # read frame
                    out = fg.step(frame)  # TODO add bbox support
                np.save(f"{root}/{cl}/{of_path}/{video.replace('.mp4', '.npy')}", out.round(2))


class DatasetGeneratorLSTM:
    def __init__(self, config: Dict):
        self.ds_config = config
        self.n_kpts = len(config['kpts'])
        self.selected_ids = config['selected_ids']
        self.angle_pairs = config['angle_pairs']
        self.features = config['features']
        self.feature_generator = FeatureGenerator(config)
    
    def generate_lstm_dataset(self, w_size: int, save_folder: str, overlap: float = 0.3) -> None:
        labels_path = self.ds_config['labels']
        class_names = self.ds_config['classes']
        root_path = self.ds_config['root']
        img_features = self.ds_config['img_features']
        os.makedirs(f"{root_path}/{save_folder}", exist_ok=True)  # create dirs for dataset
        os.makedirs(f"{root_path}/{save_folder}/train", exist_ok=True)
        os.makedirs(f"{root_path}/{save_folder}/val", exist_ok=True)
        for id, cl in tqdm(enumerate(class_names), total=len(class_names)):
            label_list = sorted(os.listdir(f"{root_path}/{cl}/{labels_path}/"))  # get txt paths of saved labels
            feature_list = sorted(os.listdir(f"{root_path}/{cl}/{img_features}/"))  # get txt paths of saved img features
            train_labels, val_labels, train_feature_list, val_feature_list = train_test_split(label_list, feature_list, test_size=0.2)
            data = []
            for l_path, f_path in tqdm(zip(train_labels, train_feature_list), total=len(label_list)):
                label, _ = read_label(f"{root_path}/{cl}/{labels_path}/{l_path}")  # read label as np.ndarray
                label = labels2kpts(label, self.n_kpts, 'bbox' in self.features)  # strip frame number and kpts scores
                if self.ds_config['crop_features']:
                    img_f = read_feature(f"{root_path}/{cl}/{img_features}/{f_path}")  # read features as np.ndarray
                n_frames = label.shape[0]  # total number of frames
                step = floor((1 - overlap) * w_size)  # step of the sliding window
                n_steps = (n_frames - w_size) // step  + 1  # number of steps for label
                for i in range(n_steps):
                    f_vector = self.feature_generator.build_feature_matrix(label[i * step : i * step + w_size].copy())  # build f vector from kpts
                    if self.ds_config['crop_features']:
                        f_vector = np.concatenate((f_vector,
                                                   img_f[i * step : i * step + w_size].copy()), axis=1)
                    f_vector = np.concatenate((f_vector, np.ones((f_vector.shape[0], 1)) * id), axis=1)
                    data.append(np.round(f_vector, 4))

            data = np.array(data)
            np.save(f"{root_path}/{save_folder}/train/{cl}.npy", data)
            data = []
            for l_path, f_path in tqdm(zip(val_labels, val_feature_list), total=len(label_list)):
                label, _ = read_label(f"{root_path}/{cl}/{labels_path}/{l_path}")  # read label as np.ndarray
                label = labels2kpts(label, self.n_kpts, 'bbox' in self.features)  # strip frame number and kpts scores
                if self.ds_config['crop_features']:
                    img_f = read_feature(f"{root_path}/{cl}/{img_features}/{f_path}")  # read features as np.ndarray
                n_frames = label.shape[0]  # total number of frames
                step = floor((1 - overlap) * w_size)  # step of the sliding window
                n_steps = (n_frames - w_size) // step  + 1  # number of steps for label
                for i in range(n_steps):
                    f_vector = self.feature_generator.build_feature_matrix(label[i * step : i * step + w_size].copy())  # build f vector from kpts
                    if self.ds_config['crop_features']:
                        f_vector = np.concatenate((f_vector,
                                                   img_f[i * step : i * step + w_size].copy()), axis=1)
                    f_vector = np.concatenate((f_vector, np.ones((f_vector.shape[0], 1)) * id), axis=1)
                    data.append(np.round(f_vector, 4))
            np.save(f"{root_path}/{save_folder}/val/{cl}.npy", data)
    