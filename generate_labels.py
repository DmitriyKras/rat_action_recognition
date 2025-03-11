from dataset_generation import AutoLabelActions, DatasetGeneratorLSTM
from configs import TOPVIEWRODENTS_CONFIG
import time

of_params = {'pyr_scale': 0.5, 'levels': 3, 
             'winsize': 15, 
             'iterations': 3, 
             'poly_n': 5, 
             'poly_sigma': 1.2, 'flags': 0}


### PIPELINE FOR DATASET LABELING FROM SCRATCH
ds_config = TOPVIEWRODENTS_CONFIG
# Label TopViewRodents for all videos
ds_config['videos'] = 'videos'
ds_config['labels'] = 'labels_topviewrodents'
labeler = AutoLabelActions(ds_config)
start = time.time()
labeler.label_bbox_kpts()
end = time.time()
labels_tvr = end - start
ds_config['videos'] = 'videos_sick'
ds_config['labels'] = 'labels_topviewrodents_sick'
labeler = AutoLabelActions(ds_config)
start = time.time()
labeler.label_bbox_kpts()
end = time.time()
labels_tvr_sick = end - start
# Label Ratpose for all videos
mmpose_config = {
    'model': '/home/cv-worker/dmitrii/weights/ratpose/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_ratpose_from_scratch.py',
    'weights': '/home/cv-worker/dmitrii/weights/ratpose/best_300.pth'
}
ds_config['videos'] = 'videos'
ds_config['labels'] = 'labels_ratpose'
labeler = AutoLabelActions(ds_config, mmpose_config=mmpose_config)
start = time.time()
labeler.label_bbox_kpts()
end = time.time()
labels_ratpose = end - start
ds_config['videos'] = 'videos_sick'
ds_config['labels'] = 'labels_ratpose_sick'
labeler = AutoLabelActions(ds_config, mmpose_config=mmpose_config)
start = time.time()
labeler.label_bbox_kpts()
end = time.time()
labels_ratpose_sick = end - start
# Label Optical Flow for all videos
ds_config['videos'] = 'videos'
ds_config['labels'] = 'labels_topviewrodents'
ds_config['optical_flow'] = 'optical_flow'
labeler = AutoLabelActions(ds_config)
start = time.time()
labeler.label_flow_farneback(of_params, (960, 480))
end = time.time()
optical_flow = end - start
ds_config['videos'] = 'videos_sick'
ds_config['labels'] = 'labels_topviewrodents_sick'
ds_config['optical_flow'] = 'optical_flow_sick'
labeler = AutoLabelActions(ds_config)
start = time.time()
labeler.label_flow_farneback(of_params, (960, 480))
end = time.time()
optical_flow_sick = end - start

print(f"Labels TopViewRodents: {labels_tvr / 60} min {labels_tvr / 3600} hr")
print(f"Labels TopViewRodents sick: {labels_tvr_sick / 60} min {labels_tvr_sick / 3600} hr")
print(f"Labels Ratpose: {labels_ratpose / 60} min {labels_ratpose / 3600} hr")
print(f"Labels Ratpose sick: {labels_ratpose_sick / 60} min {labels_ratpose_sick / 3600} hr")
print(f"Optical Flow: {optical_flow / 60} min {optical_flow / 3600} hr")
print(f"Optical Flow sick: {optical_flow_sick / 60} min {optical_flow_sick / 3600} hr")

#ds = DatasetGeneratorLSTM(TOPVIEWRODENTS_CONFIG)

#ds.generate_lstm_dataset(32, 'lstm_dataset', 0.3)
# import numpy as np
# data = np.load('lstm_dataset/val/scratching_back_paw.npy', mmap_mode='r')
# print(data.shape)
# print(data[:, :, -1])