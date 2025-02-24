from torch.utils.data import ConcatDataset
from utils import ClassificationTrainer, LSTMDataset
from models import MultiLayerBiLSTMClassifier, LSTMClassifier
from configs import TOPVIEWRODENTS_CONFIG


BATCH_SIZE = 128
EPOCHS = 50
LR = 10e-4
ds_config = TOPVIEWRODENTS_CONFIG


train_ds = ConcatDataset([LSTMDataset(ds_config, 'train', cl, augment=True) for cl in ds_config['classes']])
val_ds = ConcatDataset([LSTMDataset(ds_config, 'val', cl) for cl in ds_config['classes']])


model = LSTMClassifier(30, 128, len(ds_config['classes']))
#model = MultiLayerBiLSTMClassifier(2078, 256, 2, len(ds_config['classes'])).to(device)

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='lstm')
trainer.train(BATCH_SIZE, EPOCHS, LR)
