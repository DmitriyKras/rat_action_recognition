from typing import Dict
import pandas as pd
from utils import ClassificationTrainer, build_lstm_dataset
from dataset_generation import AutoLabelActions
from models import MultiLayerBiLSTMClassifier, LSTMClassifier
from configs import *


BATCH_SIZE = 128
EPOCHS = 100
LR = 10e-4
SEQ_LENGTH = 32
TRAIN_JSON = '/home/cv-worker/dmitrii/rat_action_recognition/train_split.json'
VAL_JSON = '/home/cv-worker/dmitrii/rat_action_recognition/val_split.json'

def train_val_lstm(name: str, ds_config: Dict) -> pd.DataFrame:
    labeler = AutoLabelActions(ds_config)
    labeler.label_bbox_kpts_features()

    train_ds, val_ds = build_lstm_dataset(ds_config,
                                      TRAIN_JSON, VAL_JSON,
                                      SEQ_LENGTH,
                                      overlap=0.5,
                                      offset=10)

    model = MultiLayerBiLSTMClassifier(val_ds[0][0].size()[-1], 256, 2, len(ds_config['classes']))

    trainer = ClassificationTrainer(ds_config, model, (train_ds, val_ds), name=name)

    return trainer.train(BATCH_SIZE, EPOCHS, LR)

res = []

# TopViewRodents spine
ds_config = TOPVIEWRODENTS_CONFIG
res.append(train_val_lstm('lstm_tvr_spine', ds_config))

# TopViewRodents spine and ears
ds_config['selected_ids'] = TOPVIEWRODENTS_SPINE_EARS_KPTS
ds_config['angle_pairs'] = TOPVIEWRODENTS_SPINE_EARS_ANGLES
res.append(train_val_lstm('lstm_tvr_spine_ears', ds_config))

# TopViewRodents all
ds_config['selected_ids'] = TOPVIEWRODENTS_ALL_KPTS
ds_config['angle_pairs'] = TOPVIEWRODENTS_ALL_ANGLES
res.append(train_val_lstm('lstm_tvr_all', ds_config))

# WistarRat spine
ds_config = WISTAR_RAT_CONFIG
res.append(train_val_lstm('lstm_wistar_spine', ds_config))

# WistarRat spine, ears and eyes
ds_config['selected_ids'] = WISTAR_RAT_SPINE_HEAD_KPTS
ds_config['angle_pairs'] = WISTAR_RAT_SPINE_HEAD_ANGLES
res.append(train_val_lstm('lstm_wistar_spine_head', ds_config))

# WistarRat all
ds_config['selected_ids'] = WISTAR_RAT_ALL_KPTS
ds_config['angle_pairs'] = WISTAR_RAT_ALL_ANGLES
res.append(train_val_lstm('lstm_wistar_all', ds_config))


res = pd.concat(res, axis=0)
print(res)
res.to_pickle('lstm.pkl')
