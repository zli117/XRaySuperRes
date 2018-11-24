import argparse
import sys

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam

from defines import *
from model.combined import CombinedNetworkDenoiseAfter
from model.dncnn import DnCNN
from model.espcn import ESPCN
from toolbox.train import TrackedTraining
from util.XRayDataSet import XRayDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 1')
    parser.add_argument('-v', '--valid_portion', type=float, default=0.1,
                        help='portion of train dataset used for validation')
    parser.add_argument('-t', '--train_batch_size', type=int, default=128,
                        help='train batch size')
    parser.add_argument('-b', '--valid_batch_size', type=int, default=512,
                        help='validation batch size')
    parser.add_argument('-e', '--epochs', type=int,
                        help='number of epochs for upsample and denoise')
    parser.add_argument('-y', '--combined_epochs', type=int,
                        help='epochs for training combined model')
    parser.add_argument('-p', '--save_pfx',
                        help='prefix for saving states')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='which device to run on')
    parser.add_argument('-i', '--image_dir', default=TRAIN_IMG,
                        help='input image dir')
    parser.add_argument('-l', '--target_dir', default=TRAIN_TARGET,
                        help='target image dir')

    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    arg = parser.parse_args()
    return arg


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


args = parse_args()

image_files = os.listdir(args.image_dir)

train_split, valid_split = train_test_split(image_files,
                                            test_size=args.valid_portion)
print('train split size: %d' % len(train_split))
print('valid split size: %d' % len(valid_split))


class Train(TrackedTraining):
    def __init__(self, *args, **kwargs):
        self.mse_loss = nn.MSELoss()
        super().__init__(*args, **kwargs)

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['target'])
        return image, target

    def train_loss_fn(self, output, target):
        loss = self.mse_loss(output, target)
        return torch.sqrt(loss)

    def valid_loss_fn(self, output, target):
        loss = self.train_loss_fn(output, target)
        return loss * 255


class TrainDenoise(TrackedTraining):
    def __init__(self, sr_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr_model = sr_model
        self.sr_model.eval()
        self.mse_loss = nn.MSELoss()
        self.curr_val_input = None

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['target'])
        sr_out = self.sr_model(image)
        residual = sr_out - target
        return sr_out, residual

    def parse_valid_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['target'])
        self.curr_val_input = self.sr_model(image)
        return self.curr_val_input, target

    def train_loss_fn(self, output, target):
        loss = self.mse_loss(output, target)
        return torch.sqrt(loss)

    def valid_loss_fn(self, output, target):
        output = self.curr_val_input - output
        return self.train_loss_fn(output, target) * 255


train_dataset = XRayDataset(train_split, args.image_dir, args.target_dir)
valid_dataset = XRayDataset(valid_split, args.image_dir, args.target_dir)

train_loader_config = {'num_workers': 8,
                       'batch_size': args.train_batch_size}
inference_loader_config = {'num_workers': 10,
                           'batch_size': args.valid_batch_size,
                           'shuffle': False}

with torch.cuda.device_ctx_manager(args.device):
    print('On device:', torch.cuda.get_device_name(args.device))
    print('Warning: No save on interrupt')
    print('======== Training ESPCN ========')

    optimizer_config = {'lr': 1e-5}
    espcn = ESPCN(2)
    save_pfx = args.save_pfx + 'espcn'
    train = Train(espcn, train_dataset, valid_dataset, Adam,
                  save_pfx, save_pfx, optimizer_config, train_loader_config,
                  inference_loader_config, epochs=args.epochs)
    espcn = train.train()

    print('======== Training DnCNN ========')
    optimizer_config = {'lr': 7e-6}
    dncnn = DnCNN(1)
    save_pfx = args.save_pfx + 'dncnn'
    train = TrainDenoise(espcn, dncnn, train_dataset, valid_dataset, Adam,
                         save_pfx, save_pfx, optimizer_config,
                         train_loader_config, inference_loader_config,
                         epochs=args.epochs)
    dncnn = train.train()

    print('======== Training Combined ========')
    optimizer_config = {'lr': 1e-5}
    combined = CombinedNetworkDenoiseAfter(espcn, dncnn)
    save_pfx = args.save_pfx + 'combined'
    train = Train(combined, train_dataset, valid_dataset, Adam,
                  save_pfx, save_pfx, optimizer_config, train_loader_config,
                  inference_loader_config, epochs=args.combined_epochs)
    combined = train.train()