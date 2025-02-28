from utils import build_conv2d_dataset, ClassificationTrainer
from models import resnet34, resnet50
from configs import TOPVIEWRODENTS_CONFIG



BATCH_SIZE = 64
EPOCHS = 100
LR = 10e-4
ds_config = TOPVIEWRODENTS_CONFIG


train_ds, val_ds = build_conv2d_dataset(ds_config, 
                                        input_shape=(256, 256),
                                        offset=10)


model = resnet50(num_classes=len(ds_config['classes']), in_channels=3)

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='img_resnet_50')
trainer.train(BATCH_SIZE, EPOCHS, LR)