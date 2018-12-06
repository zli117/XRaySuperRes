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
    def __init__(self, denoise_model, upsample_model):
        super(CombinedNetworkDenoiseBefore, self).__init__()
        self.upsample = upsample_model
        self.denoise = denoise_model

    def forward(self, low_res_noise):
        residual = self.denoise(low_res_noise)
        up_sampled = self.upsample(low_res_noise - residual)
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
