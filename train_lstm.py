from utils import ClassificationTrainer, build_lstm_dataset
from models import MultiLayerBiLSTMClassifier, LSTMClassifier
from configs import TOPVIEWRODENTS_CONFIG


BATCH_SIZE = 128
EPOCHS = 100
LR = 10e-4
ds_config = TOPVIEWRODENTS_CONFIG
SEQ_LENGTH = 32
TRAIN_JSON = '/home/cv-worker/dmitrii/rat_action_recognition/train_split.json'
VAL_JSON = '/home/cv-worker/dmitrii/rat_action_recognition/val_split.json'


train_ds, val_ds = build_lstm_dataset(ds_config,
                                      TRAIN_JSON, VAL_JSON,
                                      SEQ_LENGTH,
                                      overlap=0.5,
                                      offset=10)


#model = LSTMClassifier(SEQ_LENGTH, 128, len(ds_config['classes']))

model = MultiLayerBiLSTMClassifier(SEQ_LENGTH, 256, 2, len(ds_config['classes']))

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='lstm')
trainer.train(BATCH_SIZE, EPOCHS, LR)
