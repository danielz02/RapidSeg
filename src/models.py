from torch.nn import Conv2d
from src.unet_parts import *
from torchvision.models.segmentation import fcn_resnet101


class FCN(nn.Module):
    def __init__(self, n_classes: int, n_bands: int, pretrained: bool):
        """
        Instantiate a wrapper for torchvision FCN with a Resnet 101 backbone pretrained on ImageNet and
        a classifier pretrained on a subset of Coco
        Note: This FCN upsamples by non-learnable bi-linear interpolation
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
            self.fcn.backbone.conv1 = Conv2d(in_channels=n_bands, out_channels=64, kernel_size=(7, 7), stride=(2, 2))

        self.n_bands = n_bands
        self.n_classes = n_classes
        self.pretrained = pretrained

    def forward(self, x: torch.Tensor):
        return self.fcn(x)


# https://github.com/milesial/Pytorch-UNet/blob/381924dfd6396c92b17ceebaa9c607f3f9392106/unet/unet_model.py
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return {"out": logits}
