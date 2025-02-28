from utils import build_conv3d_dataset, ClassificationTrainer
from configs import TOPVIEWRODENTS_CONFIG
from models import resnet3d
from torchsummary import summary


BATCH_SIZE = 8
EPOCHS = 100
LR = 10e-4
ds_config = TOPVIEWRODENTS_CONFIG


train_ds, val_ds = build_conv3d_dataset(ds_config, 
                                        input_shape=(256, 256), 
                                        w_size=32, 
                                        overlap=0.2, 
                                        offset=10)


model = resnet3d('resnet18', n_classes=len(ds_config['classes']))
#summary(model, (3, 32, 256, 256))

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='conv3d_resnet_18')
trainer.train(BATCH_SIZE, EPOCHS, LR)