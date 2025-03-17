from utils import ClassificationTrainer, build_lstm_dataset
from models import MultiLayerBiLSTMClassifier, LSTMClassifier
from configs import *


BATCH_SIZE = 128
EPOCHS = 100
LR = 10e-4

# TopViewRodents
ds_config = TOPVIEWRODENTS_CONFIG

# ds_config['selected_ids'] = TOPVIEWRODENTS_SPINE_EARS_KPTS
# ds_config['angle_pairs'] = TOPVIEWRODENTS_SPINE_EARS_ANGLES

# ds_config['selected_ids'] = TOPVIEWRODENTS_ALL_KPTS
# ds_config['angle_pairs'] = TOPVIEWRODENTS_ALL_ANGLES

# WistarRat
# ds_config = WISTAR_RAT_CONFIG

# ds_config['selected_ids'] = WISTAR_RAT_SPINE_HEAD_KPTS
# ds_config['angle_pairs'] = WISTAR_RAT_SPINE_HEAD_ANGLES

# ds_config['selected_ids'] = WISTAR_RAT_ALL_KPTS
# ds_config['angle_pairs'] = WISTAR_RAT_ALL_ANGLES

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

trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name='lstm_tvr_spine')
trainer.train(BATCH_SIZE, EPOCHS, LR)
