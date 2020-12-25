import torch
import torch.nn as nn
from torch.nn import Conv2d
from torchvision.models.segmentation import fcn_resnet101


class FCN(nn.Module):
    def __init__(self, n_classes: int, n_bands: int, pretrained: bool):
        """
        Instantiate a wrapper for torchvision FCN with a Resnet 101 backbone pretrained on ImageNet and
        a classifier pretrained on a subset of Coco
        :param n_classes: the number of classes in the dataset
        :param n_bands: the number of bands in the satellite imagery
        :param pretrained: whether to load pretrained weights
        """
        super(FCN, self).__init__()
        assert not pretrained or (pretrained and n_bands is 3)  # Pretrained models were trained on RGB images

        self.fcn = fcn_resnet101(pretrained=pretrained, progress=True)

        if pretrained:
            for layer in self.fcn.backbone.parameters():
                layer.requires_grad = False
        self.fcn.classifier[4] = Conv2d(in_channels=512, out_channels=n_classes, kernel_size=(1, 1), stride=(1, 1))

        if n_bands is not 3:
            self.fcn.conv1 = Conv2d(in_channels=n_bands, out_channels=64, kernel_size=(7, 7), stride=(2, 2))

        self.n_bands = n_bands
        self.n_classes = n_classes
        self.pretrained = pretrained

    def forward(self, x: torch.Tensor):
        return self.fcn(x)
