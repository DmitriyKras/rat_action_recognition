import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional, Literal
from .conv2d_classifiers import resnet34, resnet50, resnet101
from .torch_utils import state_dict_intersection
from torchsummary import summary


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


class TwoStreamCNNFusionConv(nn.Module):
    def __init__(self, of_seq_length: int, n_classes: int, fusion_type: Literal['mul', 'sum', 'conv'],
                 rgb_weights: Optional[str] = None, of_weights: Optional[str] = None,) -> None:
        super().__init__()
        # Build stream models
        self.rgb_stream = resnet50(num_classes=n_classes, in_channels=3, include_top=False)
        self.of_stream = resnet50(num_classes=n_classes, in_channels=2 * of_seq_length, include_top=False)
        # Load weights
        if rgb_weights is not None:
            self.rgb_stream.load_state_dict(state_dict_intersection(self.rgb_stream.state_dict(), torch.load(rgb_weights)))
        if of_weights is not None:
            self.of_stream.load_state_dict(state_dict_intersection(self.of_stream.state_dict(), torch.load(of_weights)))
        # Freeze two stream backbones
        for param in self.rgb_stream.parameters():
            param.requires_grad_(False)
        for param in self.of_stream.parameters():
            param.requires_grad_(False)
        
        # Define fusing convs
        self.fusion_type = fusion_type
        if fusion_type == 'conv':
            self.fusing_conv = nn.Conv2d(2048*2, 2048, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, n_classes)

    def forward(self, frame, of_frames):
        rgb_fmap = self.rgb_stream(frame)
        of_fmap = self.of_stream(of_frames)
        if self.fusion_type == 'conv':
            x = self.fusing_conv(torch.cat((rgb_fmap, of_fmap), dim=1))
        elif self.fusion_type == 'mul':
            x = rgb_fmap * of_fmap
        elif self.fusion_type == 'sum':
            x = rgb_fmap + of_fmap
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
