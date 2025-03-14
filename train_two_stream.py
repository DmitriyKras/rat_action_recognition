from utils import TwoStreamClassificationTrainer, build_two_stream_dataset
from models import TwoStreamCNNFusionConv
from configs import TOPVIEWRODENTS_CONFIG


BATCH_SIZE = 8
EPOCHS = 100
LR = 10e-4
ds_config = TOPVIEWRODENTS_CONFIG
SEQ_LENGTH = 10
RGB_WEIGHTS = '/home/cv-worker/dmitrii/weights/action_recognition/best_img_resnet_50.pt'
FLOW_WEIGHTS = '/home/cv-worker/dmitrii/weights/action_recognition/best_flow_resnet_50.pt'
TRAIN_JSON = '/home/cv-worker/dmitrii/rat_action_recognition/train_split.json'
VAL_JSON = '/home/cv-worker/dmitrii/rat_action_recognition/val_split.json'


train_ds, val_ds = build_two_stream_dataset(ds_config, TRAIN_JSON, VAL_JSON, input_shape=(256, 256),
                                   step=10, seq_length=SEQ_LENGTH, offset=10)
two_stream_model = TwoStreamCNNFusionConv(SEQ_LENGTH, len(ds_config['classes']), RGB_WEIGHTS, FLOW_WEIGHTS)

trainer = TwoStreamClassificationTrainer(ds_config, two_stream_model, (train_ds, val_ds))
trainer.train(BATCH_SIZE, EPOCHS, LR)