from utils import ClassificationTrainer, build_lstm_dataset
from models import MultiLayerBiLSTMClassifier, LSTMClassifier
from configs import TOPVIEWRODENTS_CONFIG


BATCH_SIZE = 128
EPOCHS = 100
LR = 10e-4
ds_config = TOPVIEWRODENTS_CONFIG


train_ds, val_ds = build_lstm_dataset(ds_config)


#model = LSTMClassifier(val_ds[0][0].size()[-1], 128, len(ds_config['classes']))

model = MultiLayerBiLSTMClassifier(val_ds[0][0].size()[-1], 256, 2, len(ds_config['classes']))

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='lstm')
trainer.train(BATCH_SIZE, EPOCHS, LR)
