"""Model definitions."""

import torch
import torch.nn as nn
import torchvision.models as models

BACKBONES = ["mobilenet_v2", "xception", "densenet121"]


class SingleModalModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, pretrained=True, backbone="mobilenet_v2"):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v2(weights=weights)
            self._adapt_first_conv(net.features[0][0], in_channels, pretrained,
                                   setter=lambda c: setattr(net.features[0], "0", c))
            self.features = net.features
            last_ch = net.last_channel  # 1280

        elif backbone == "xception":
            import timm
            net = timm.create_model("legacy_xception", pretrained=pretrained,
                                    num_classes=0, global_pool="")
            self._adapt_first_conv(net.conv1, in_channels, pretrained,
                                   setter=lambda c: setattr(net, "conv1", c))
            self.features = net
            last_ch = net.num_features  # 2048

        elif backbone == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            net = models.densenet121(weights=weights)
            self._adapt_first_conv(net.features.conv0, in_channels, pretrained,
                                   setter=lambda c: setattr(net.features, "conv0", c))
            self.features = nn.Sequential(net.features, nn.ReLU(inplace=True))
            last_ch = net.classifier.in_features  # 1024

        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {BACKBONES}")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_ch, num_classes))

    @staticmethod
    def _adapt_first_conv(conv, in_channels, pretrained, setter):
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
        setter(new_conv)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(modality="rgb", num_classes=2, pretrained=True, backbone="mobilenet_v2"):
    from data_loader import MODALITY_CHANNELS

    in_channels = MODALITY_CHANNELS[modality]
    model = SingleModalModel(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained, backbone=backbone)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {backbone} | Input channels: {in_channels} | Params: {total:,} ({trainable:,} trainable)")
    return model
