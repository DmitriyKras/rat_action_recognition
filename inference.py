from mmdet.apis import DetInferencer
import cv2
import numpy as np
from mmpose.apis import inference_topdown, init_model
from catboost import CatBoostClassifier
import joblib
from dataset_generation import DatasetGenerator
from configs import TOPVIEWRODENTS_CONFIG, W_SIZE
from inference import Inferencer
import time


inf = Inferencer(TOPVIEWRODENTS_CONFIG)
inf('/home/techtrans2/RAT_DATASETS/LAB_RAT_ACTIONS_DATASET/body_cleaning/videos/body_cleaning_22.mp4', 16)

# inferencer = DetInferencer(model='/home/techtrans2/mmdetection/work_dirs/rtmdet_s_8xb32-300e_ratdataset/rtmdet_s_8xb32-300e_ratdataset.py', 
#                            weights='/home/techtrans2/mmdetection/work_dirs/rtmdet_s_8xb32-300e_ratdataset/epoch_200.pth', device='cuda')

# # model = init_model('/home/techtrans2/mmpose/work_dirs/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_ratpose_from_scratch/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_ratpose_from_scratch.py',
# #                    '/home/techtrans2/mmpose/work_dirs/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_ratpose_from_scratch/best_PCK_epoch_240.pth', device='cuda')

# model = init_model('/home/techtrans2/mmpose/work_dirs/topviewrodents/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_topviewrodents.py',
#                    '/home/techtrans2/mmpose/work_dirs/topviewrodents/best_PCK_epoch_320.pth', device='cuda')

# #cap = cv2.VideoCapture('scratching_back_paw/videos/scratching_back_paw_5.mp4')
# cap = cv2.VideoCapture('/home/techtrans2/RAT_DATASETS/rec_2024_07_09_13_42_01_up.mp4')

# stacked_kpts = []
# clf = joblib.load('catboost_model.pkl')

# ds = DatasetGenerator(TOPVIEWRODENTS_CONFIG)
# CLASSES = TOPVIEWRODENTS_CONFIG['classes']


# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret or cv2.waitKey(0) == ord('q'):
#         break
    
#     H, W = frame.shape[:2]

#     result = inferencer(frame)
#     bbox = result['predictions'][0]['bboxes'][0]
#     x1, y1, x2, y2 = np.array(bbox).astype(int)
#     kpts = inference_topdown(model, frame.copy(), bboxes=np.array(bbox)[None, ...])[0].pred_instances
#     scores = kpts.keypoint_scores[0]
#     keypoints = kpts.keypoints[0]
#     for pt in keypoints.astype(int)[TOPVIEWRODENTS_CONFIG['selected_ids']]:
#         frame = cv2.circle(frame, (pt[0], pt[1]), 3, (0, 0, 255), -1)

#     frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

#     keypoints /= np.array(frame.shape[:2][::-1])

#     if 'bbox' in TOPVIEWRODENTS_CONFIG['features']:
#         stacked_kpts.append(np.concatenate((
#             np.array((x1 / W, y1 / H, x2 / W, y2 / H)), keypoints.flatten())))
#     else:
#         stacked_kpts.append(keypoints.flatten())

#     if len(stacked_kpts) == W_SIZE:
#         print(np.array(stacked_kpts).shape)
#         vector = ds.build_stat_feature_vector(np.array(stacked_kpts), 0)
#         print(vector.shape)
#         probs = clf.predict(vector)
#         print(CLASSES[int(probs)])
#         stacked_kpts.pop(0)
#         frame = cv2.putText(frame, CLASSES[int(probs)], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



#     cv2.imshow('rat', frame)

# cv2.destroyAllWindows()