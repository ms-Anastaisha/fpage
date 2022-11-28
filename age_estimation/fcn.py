import torch
import torch.nn as nn

class FCN(nn.Sequential):
    def __init__(self, in_channels, num_classes, **kwargs):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        ]

        super(FCN, self).__init__(*layers)


