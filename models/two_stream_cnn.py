import torch.nn as nn
import torch.nn.functional as F
import torch
from .conv2d_classifiers import resnet34, resnet50, resnet101


class TwoStreamCNNFusionClf(nn.Module):
    def __init__(self, of_seq_length: int, n_classes: int, 
                 rgb_weights: str, of_weights: str) -> None:
        super().__init__()
        # Build stream models
        self.rgb_stream = resnet50(num_classes=n_classes, in_channels=3)
        self.of_stream = resnet50(num_classes=n_classes, in_channels=2 * of_seq_length)
        # Load weights
        self.rgb_stream.load_state_dict(torch.load(rgb_weights))
        self.of_stream.load_state_dict(torch.load(of_weights))

    def forward(self, x):
        frame, of_frames = x
        rgb_logits = self.rgb_stream(frame)
        of_logits = self.of_stream(of_frames)
        return F.softmax(rgb_logits) * F.softmax(of_logits)
