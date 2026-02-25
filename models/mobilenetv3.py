from torch import nn
from torchvision import models as tv_models


class MobileNetV3Backbone(nn.Module):
    def __init__(self, variant="small", pretrained=False):
        super().__init__()
        if variant == "small":
            net = tv_models.mobilenet_v3_small(pretrained=pretrained)
        elif variant == "large":
            net = tv_models.mobilenet_v3_large(pretrained=pretrained)
        else:
            raise ValueError("Unsupported MobileNetV3 variant: {}".format(variant))

        self.features = net.features
        self.avgpool = net.avgpool
        self.embed_dim = net.classifier[0].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x


def mobilenetv3_small(args=None, pretrained=False):
    return MobileNetV3Backbone(variant="small", pretrained=pretrained)


def mobilenetv3_large(args=None, pretrained=False):
    return MobileNetV3Backbone(variant="large", pretrained=pretrained)
