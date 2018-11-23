import torch.nn as nn

from toolbox.train import load_model


class CombinedNetworkDenoiseAfter(nn.Module):
    def __init__(self, upsample_model, denoise_model):
        super(CombinedNetworkDenoiseAfter, self).__init__()
        self.upsample = upsample_model
        self.denoise = denoise_model

    def forward(self, low_noise):
        up_sampled = self.upsample(low_noise)
        residual = self.denoise(up_sampled)
        return up_sampled - residual

