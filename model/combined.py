import torch.nn as nn

from toolbox.train import load_model


class CombinedNetworkDenoiseAfter(nn.Module):
    def __init__(self, up_sample_cls: type, denoise_cls: type,
                 up_sample_params: dict, denoise_params: dict,
                 upsample_path: str = None, denoise_path: str = None):
        super(CombinedNetworkDenoiseAfter, self).__init__()
        self.up_sample_net = up_sample_cls(**up_sample_params)
        self.denoise_net = denoise_cls(**denoise_params)
        if upsample_path is not None:
            load_model(upsample_path, self.up_sample_net)
        if denoise_path is not None:
            load_model(denoise_path, self.denoise_net)

    def forward(self, low_noise):
        up_sampled = self.up_sample_net(low_noise)
        residual = self.denoise_net(up_sampled)
        return up_sampled - residual

