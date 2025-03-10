from utils import build_conv3d_dataset, ClassificationTrainer
from configs import TOPVIEWRODENTS_CONFIG
from models import resnet3d
from torchsummary import summary


BATCH_SIZE = 16
EPOCHS = 100
LR = 10e-4
ds_config = TOPVIEWRODENTS_CONFIG
TRAIN_JSON = '/home/cv-worker/dmitrii/rat_action_recognition/train_split.json'
VAL_JSON = '/home/cv-worker/dmitrii/rat_action_recognition/val_split.json'


train_ds, val_ds = build_conv3d_dataset(ds_config, TRAIN_JSON, VAL_JSON,
                                        input_shape=(256, 256), 
                                        w_size=32, 
                                        overlap=0.2, 
                                        offset=10)


model = resnet3d('resnet18', n_classes=len(ds_config['classes']))
#summary(model, (3, 32, 256, 256))

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='conv3d_resnet_18')
trainer.train(BATCH_SIZE, EPOCHS, LR)