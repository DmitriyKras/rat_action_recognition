from utils import build_flow_dataset, ClassificationTrainer
from models import resnet34, resnet50
from configs import TOPVIEWRODENTS_CONFIG



BATCH_SIZE = 32
EPOCHS = 100
LR = 10e-4
ds_config = TOPVIEWRODENTS_CONFIG
SEQ_LENGTH = 10


train_ds, val_ds = build_flow_dataset(ds_config, 
                                        seq_length=SEQ_LENGTH, 
                                        overlap=0.5, 
                                        offset=10)


model = resnet50(num_classes=len(ds_config['classes']), in_channels=SEQ_LENGTH * 2)

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='flow_resnet_50')
trainer.train(BATCH_SIZE, EPOCHS, LR)