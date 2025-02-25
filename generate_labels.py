from dataset_generation import AutoLabelActions, DatasetGeneratorLSTM
from configs import TOPVIEWRODENTS_CONFIG


of_params = {'pyr_scale': 0.5, 'levels': 3, 
             'winsize': 15, 
             'iterations': 3, 
             'poly_n': 5, 
             'poly_sigma': 1.2, 'flags': 0}

import numpy as np

of = np.load('/home/techtrans2/RAT_DATASETS/LAB_RAT_ACTIONS_DATASET/body_cleaning/optical_flow/body_cleaning_6.npy')
print(of[..., 0].max(), of[..., 0].min(), of[..., 0].mean())



# labeler = AutoLabelActions(TOPVIEWRODENTS_CONFIG)
# labeler.label_flow_farneback(of_params)
# labeler.label_bbox_kpts()
# labeler.label_img_features()



#ds = DatasetGeneratorLSTM(TOPVIEWRODENTS_CONFIG)

#ds.generate_lstm_dataset(16, 'lstm_dataset', 0.8)
# import numpy as np
# data = np.load('lstm_dataset/val/scratching_back_paw.npy', mmap_mode='r')
# print(data.shape)
# print(data[:, :, -1])