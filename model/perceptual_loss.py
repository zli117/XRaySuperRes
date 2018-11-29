import torch
from torch import nn
from torchvision.models import vgg11


class PerceptualLoss(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        vgg = vgg11()
        vgg.eval()
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path)
            vgg.load_state_dict(state_dict)
            del state_dict
        self.features = vgg.features
        self.mse_loss = nn.MSELoss()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, output_chan1, target_chan1):
        output_chan3 = torch.cat((output_chan1, output_chan1, output_chan1),
                                 dim=1)
        target_chan3 = torch.cat((target_chan1, target_chan1, target_chan1),
                                 dim=1)
        output_features = self.features[:10](output_chan3)
        target_features = self.features[:10](target_chan3)
        return self.mse_loss(output_features, target_features)
