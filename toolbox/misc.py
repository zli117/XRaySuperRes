import torch
from torch import nn


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def save_model(model: nn.Module, file_path, epoch, seed, step=None):
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'seed': seed,
                'step': step}, file_path)


def load_model(file_path, model: nn.Module):
    obj = torch.load(file_path)
    model_state = obj['model']
    model.load_state_dict(model_state)
    del model_state
    return model
