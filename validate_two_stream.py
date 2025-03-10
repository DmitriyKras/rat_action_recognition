from utils import TwoStreamValidator, build_two_stream_dataset
import torch
from models import TwoStreamCNNFusionClf, resnet50
from configs import TOPVIEWRODENTS_CONFIG


BATCH_SIZE = 32
ds_config = TOPVIEWRODENTS_CONFIG
SEQ_LENGTH = 10
RGB_WEIGHTS = '/home/cv-worker/dmitrii/'
FLOW_WEIGHTS = '/home/cv-worker/dmitrii/'

dataset = build_two_stream_dataset(ds_config, input_shape=(256, 256),
                                   step=5, seq_length=SEQ_LENGTH, offset=10)
rgb_model = resnet50(in_channels=3, num_classes=len(ds_config['classes']))
flow_model = resnet50(in_channels=2*SEQ_LENGTH, num_classes=len(ds_config['classes']))
rgb_model.load_state_dict(torch.load(RGB_WEIGHTS))
flow_model.load_state_dict(torch.load(FLOW_WEIGHTS))
two_stream_model = TwoStreamCNNFusionClf(2*SEQ_LENGTH, len(ds_config['classes']), RGB_WEIGHTS, FLOW_WEIGHTS)

val = TwoStreamValidator(ds_config, {'rgb': rgb_model, 'flow': flow_model, 'two_stream': two_stream_model}, dataset)
val.validate(BATCH_SIZE)