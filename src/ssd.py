import torch
import torch.nn as nn
import torch.nn.functional as F
from multibox_layer import MultiBoxLayer


class L2Norm(nn.Module):
    def __init__(self, in_channels):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels))

    def forward(self, x):
        """out = scales * x / sqrt(\sum x_i^2)"""
        unsqueezed_weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return unsqueezed_weight * x * x.pow(2).sum(1, keepdim=True).clamp(min=1e-12).rsqrt()


class SSD(nn.Module):
    """SSD300 model."""
    input_size = 300

    def __init__(self):
        super(SSD, self).__init__()

        self.base = VGG16()
        # output feature map size: 38
        self.norm4 = L2Norm(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # output feature map size: 19

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1, ceil_mode=True)
        # output feature map size: 19

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        # output feature map size: 19

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        # output feature map size: 10

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        # output feature map size: 5

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)
        # output feature map size: 3

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)
        # output feature map size: 1

        self.multibox = MultiBoxLayer()

    def forward(self, x):
        hs = []
        h = self.base(x)
        hs.append(self.norm4(h))  # conv4_3
        h = self.pool4(h)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = self.pool5(h)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)  # conv7

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)  # conv8_2

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)  # conv9_2

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)  # conv10_2

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)  # conv11_2

        loc_preds, conf_preds = self.multibox(hs)
        return loc_preds, conf_preds


def VGG16():
    """VGG16 base."""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
    layers = []
    in_channels = 3
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            # feature map sizes after each pooling: [150, 75, 38]
        else:
            layers += [
                nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                # conv uses 'same' padding
                nn.ReLU(True)
            ]
            in_channels = x
    return nn.Sequential(*layers)
