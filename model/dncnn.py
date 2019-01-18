import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17, tanh_out=False,
                 built_in_residual=False):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        self.tanh_out = tanh_out
        self.built_in_residual = built_in_residual
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding,
                                    bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        if self.built_in_residual:
            out = x - out
        if self.tanh_out:
            out = torch.tanh(out)
        return out
