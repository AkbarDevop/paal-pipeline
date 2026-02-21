"""Model definitions."""

import torch
import torch.nn as nn
import torchvision.models as models


class SingleModalModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self._replace_first_conv(backbone, in_channels, pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(backbone.last_channel, num_classes))

    @staticmethod
    def _replace_first_conv(backbone, in_channels, pretrained):
        conv = backbone.features[0][0]
        if in_channels == 3:
            return
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                repeats = (in_channels + 2) // 3
                w = conv.weight.data.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                new_conv.weight.data = w * (3.0 / in_channels)
        backbone.features[0][0] = new_conv

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class DualBranchModel(nn.Module):
    def __init__(self, channels_a=3, channels_b=3, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        a = models.mobilenet_v2(weights=weights)
        b = models.mobilenet_v2(weights=weights)
        SingleModalModel._replace_first_conv(a, channels_a, pretrained)
        SingleModalModel._replace_first_conv(b, channels_b, pretrained)
        self.branch_a = a.features
        self.branch_b = b.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(a.last_channel + b.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_a, x_b):
        f1 = self.pool(self.branch_a(x_a)).flatten(1)
        f2 = self.pool(self.branch_b(x_b)).flatten(1)
        return self.classifier(torch.cat([f1, f2], dim=1))


def build_model(modality="rgb", num_classes=2, pretrained=True):
    from data_loader import MODALITY_CHANNELS

    in_channels = MODALITY_CHANNELS[modality]
    model = SingleModalModel(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: SingleModalModel | Input channels: {in_channels} | Params: {total:,} ({trainable:,} trainable)")
    return model
