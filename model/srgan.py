from functools import reduce
from operator import mul

import torch
import torch.nn as nn
from torchvision.models import vgg19


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class FeatureExtractor(nn.Module):
    def __init__(self, vgg19_path=None):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=False)
        if vgg19_path is not None:
            vgg19_model.load_state_dict(torch.load(vgg19_path))

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(
            *list(vgg19_model.features.children())[:12])

    def forward(self, img):
        if img.shape[1] == 1:
            img = torch.cat((img, img, img), dim=1)
        out = self.feature_extractor(img)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16,
                 n_upsample=1):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, 1, 4),
            nn.ReLU()
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64))

        # Upsampling layers
        upsampling = []
        for out_features in range(n_upsample):
            upsampling += [nn.Conv2d(64, 256, 3, 1, 1),
                           nn.BatchNorm2d(256),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU()]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(64, out_channels, 9, 1, 4)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, input_shape=(128, 128)):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        layers = []
        in_filters = in_channels
        for out_filters, stride, normalize in [(64, 1, False),
                                               (64, 2, True),
                                               (128, 1, True),
                                               (128, 2, True),
                                               (256, 1, True),
                                               (256, 2, True),
                                               (512, 1, True),
                                               (512, 2, True), ]:
            layers.extend(
                discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        self.convolution = nn.Sequential(*layers)

        # TODO: Whether we should replace Fully connected layer with FCN

        dummy_input = torch.ones((1, 1, *input_shape), dtype=torch.float)
        dummy_out = self.convolution(dummy_input)
        self.fc_in_elements = reduce(mul, dummy_out.shape, 1)
        self.fcs = nn.Sequential(nn.Linear(self.fc_in_elements, 1024),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(1024, 1))

    def forward(self, img):
        features = self.convolution(img)
        unrolled = features.view(-1, self.fc_in_elements)
        return self.fcs(unrolled)
