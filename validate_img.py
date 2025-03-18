from utils import TwoStreamClassificationTrainer, build_two_stream_dataset
from utils import ClassificationTrainer, build_conv2d_dataset, build_conv3d_dataset, build_flow_dataset
import torch
import pandas as pd
from models import TwoStreamCNNFusionConv, resnet50, resnet3d
from configs import TOPVIEWRODENTS_CONFIG


BATCH_SIZE = 32
ds_config = TOPVIEWRODENTS_CONFIG
SEQ_LENGTH = 10
FRAMES_LENGTH = 32
RGB_WEIGHTS = '/home/cv-worker/dmitrii/weights/action_recognition/best_img_resnet_50.pt'
FLOW_WEIGHTS = '/home/cv-worker/dmitrii/weights/action_recognition/best_flow_resnet_50.pt'
CONV3D_WEIGHTS = '/home/cv-worker/dmitrii/weights/action_recognition/best_conv3d_resnet_18.pt'
TRAIN_JSON = '/home/cv-worker/dmitrii/rat_action_recognition/train_split.json'
VAL_JSON = '/home/cv-worker/dmitrii/rat_action_recognition/val_split.json'

res = []

# Validate Conv2D
train_ds, val_ds = build_conv2d_dataset(ds_config,
                                        TRAIN_JSON,
                                        VAL_JSON,
                                        input_shape=(256, 256),
                                        offset=10,
                                        step=7)

model = resnet50(num_classes=len(ds_config['classes']), in_channels=3)
model.load_state_dict(torch.load(RGB_WEIGHTS))

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='img_resnet_50')
res.append(trainer.validate(BATCH_SIZE))

# Validate Conv3D
train_ds, val_ds = build_conv3d_dataset(ds_config, TRAIN_JSON, VAL_JSON,
                                        input_shape=(256, 256), 
                                        w_size=FRAMES_LENGTH, 
                                        overlap=0.5, 
                                        offset=10)

model = resnet3d('resnet18', n_classes=len(ds_config['classes']))
model.load_state_dict(torch.load(CONV3D_WEIGHTS))

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='conv3d_resnet_18')
res.append(trainer.validate(16))

# Validate Optical Flow
train_ds, val_ds = build_flow_dataset(ds_config, TRAIN_JSON, VAL_JSON,
                                        seq_length=SEQ_LENGTH, 
                                        overlap=0.5, 
                                        offset=10)

model = resnet50(num_classes=len(ds_config['classes']), in_channels=SEQ_LENGTH * 2)

model.load_state_dict(torch.load(FLOW_WEIGHTS))

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='flow_resnet_50')
res.append(trainer.validate(BATCH_SIZE))

# Validate TwoStream
train_ds, val_ds = build_two_stream_dataset(ds_config, TRAIN_JSON, VAL_JSON, input_shape=(256, 256),
                                   step=7, seq_length=SEQ_LENGTH, offset=10)

model = TwoStreamCNNFusionConv(SEQ_LENGTH, len(ds_config['classes']), 'conv')
model.load_state_dict(torch.load('/home/cv-worker/dmitrii/weights/action_recognition/best_two_stream_conv.pt'))

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='two_stream_conv')
res.append(trainer.validate(BATCH_SIZE))


model = TwoStreamCNNFusionConv(SEQ_LENGTH, len(ds_config['classes']), 'mul')
model.load_state_dict(torch.load('/home/cv-worker/dmitrii/weights/action_recognition/best_two_stream_mul.pt'))

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='two_stream_mul')
res.append(trainer.validate(BATCH_SIZE))


res = pd.concat(res, axis=0)
print(res)
res.to_pickle('img.pkl')
