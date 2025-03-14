from mmdet.apis import DetInferencer
import cv2
import numpy as np
from mmpose.apis import inference_topdown, init_model
import joblib
from configs import TOPVIEWRODENTS_CONFIG
from models import LSTMClassifier
from utils import FeatureGenerator
from typing import Dict
from torchvision import transforms, models
from torch import nn
import torch


class Inferencer:
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
        self.preprocess = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([224, 224]),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.feature_extractor = nn.Sequential(*modules).cuda()
        self.feature_extractor.eval()
        self.clf = LSTMClassifier(542, 128, len(TOPVIEWRODENTS_CONFIG['classes'])).cuda()
        self.clf.load_state_dict(torch.load('best_lstm.pt'))
        self.clf.eval()
        self.selected_ids = config['selected_ids']
        self.angle_pairs = config['angle_pairs']
        self.features = config['features']
        self.feature_generator = FeatureGenerator(config)
    
    def img_features(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - w*0.15))
        x2 = min(W, int(x2 + w*0.15))
        y1 = max(0, int(y1 - h*0.15))
        y2 = min(H, int(y2 + h*0.15))
        frame = frame[y1:y2, x1:x2, ::-1]  # extract roi
        ### EXTRACT FEATURES ###
        frame = self.preprocess(frame.copy())[None, ...].cuda()
        with torch.no_grad():
            feat = self.feature_extractor(frame).cpu().numpy().squeeze()
        return feat
    
    def __call__(self, video_path: str, w_size: int) -> None:
        cap = cv2.VideoCapture(video_path)
        stacked_kpts = []
        stacked_f = []
        classes = self.ds_config['classes']
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or cv2.waitKey(0) == ord('q'):
                break
            
            H, W = frame.shape[:2]
            # Get bbox
            result = self.bbox_detector(frame)
            bbox = result['predictions'][0]['bboxes'][0]
            x1, y1, x2, y2 = np.array(bbox).astype(int)
            # Get kpts
            kpts = inference_topdown(self.kpts_detector, frame.copy(), bboxes=np.array(bbox)[None, ...])[0].pred_instances
            keypoints = kpts.keypoints[0]
            # Get features
            img_f = self.img_features(frame.copy(), np.array(bbox).astype(int))
            stacked_f.append(img_f)
            # Draw features
            for pt in keypoints.astype(int)[self.ds_config['selected_ids']]:
                frame = cv2.circle(frame, (pt[0], pt[1]), 3, (0, 0, 255), -1)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            keypoints /= np.array(frame.shape[:2][::-1])
            # Append features
            if 'bbox' in self.ds_config['features']:
                stacked_kpts.append(np.concatenate((
                    np.array((x1 / W, y1 / H, x2 / W, y2 / H)), keypoints.flatten())))
            else:
                stacked_kpts.append(keypoints.flatten())
            if len(stacked_kpts) == w_size:
                f_matrix = self.feature_generator.build_feature_matrix(np.array(stacked_kpts))
                clf_features = np.concatenate((f_matrix, np.array(stacked_f)), axis=1)
                clf_features = torch.from_numpy(clf_features).cuda().float()
                with torch.no_grad():
                    probs = self.clf.forward(clf_features[None, ...]).cpu().numpy().squeeze()
                #print(probs)
                print(classes[probs.argmax()])
                stacked_kpts.pop(0)
                stacked_f.pop(0)
                frame = cv2.putText(frame, classes[probs.argmax()], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('rat', frame)

        cv2.destroyAllWindows()

