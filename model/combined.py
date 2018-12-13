import torch.nn as nn


class CombinedNetworkDenoiseAfter(nn.Module):
    def __init__(self, upsample_model, denoise_model):
        super(CombinedNetworkDenoiseAfter, self).__init__()
        self.upsample = upsample_model
        self.denoise = denoise_model

    def forward(self, low_res_noise):
        up_sampled = self.upsample(low_res_noise)
        residual = self.denoise(up_sampled)
        return up_sampled - residual


class CombinedNetworkDenoiseBefore(nn.Module):
    def __init__(self, denoise_model, upsample_model,
                 dncnn_built_in_residual=False):
        super(CombinedNetworkDenoiseBefore, self).__init__()
        self.upsample = upsample_model
        self.denoise = denoise_model
        self.dncnn_built_in_residual = dncnn_built_in_residual

    def forward(self, low_res_noise):
        out = self.denoise(low_res_noise)
        if not self.dncnn_built_in_residual:
            out = low_res_noise - out
        up_sampled = self.upsample(out)
        return up_sampled


class DenoiseAfterDenoisedUpsample(nn.Module):
    def __init__(self, denoise_model, upsample_model, finetune_denoise):
        super(DenoiseAfterDenoisedUpsample, self).__init__()
        self.denoise = denoise_model
        self.upsample = upsample_model
        self.finetune = finetune_denoise

    def forward(self, low_res_noise):
        residual = self.denoise(low_res_noise)
        up_sampled = self.upsample(low_res_noise - residual)
        fine_tune_res = self.finetune(up_sampled)
        return up_sampled - fine_tune_res
