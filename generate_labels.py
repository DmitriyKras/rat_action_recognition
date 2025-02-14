from dataset_generation import AutoLabelActions, DatasetGeneratorLSTM
from configs import TOPVIEWRODENTS_CONFIG

#labeler = AutoLabelActions(TOPVIEWRODENTS_CONFIG)
#labeler.label_bbox_kpts()
#labeler.label_img_features()


ds = DatasetGeneratorLSTM(TOPVIEWRODENTS_CONFIG)
ds.generate_lstm_dataset(16, 'lstm_dataset', 0.8)
import numpy as np
data = np.load('lstm_dataset/val/scratching_back_paw.npy', mmap_mode='r')
print(data.shape)
print(data[:, :, -1])