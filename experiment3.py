import argparse
import sys

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam

from defines import *
from model.dncnn import DnCNN
from model.espcn import ESPCN
from toolbox.torch_state_samplers import TrackedRandomSampler
from toolbox.train import TrackedTraining, load_model
from util.XRayDataSet import XRayDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 1')
    parser.add_argument('-v', '--valid_portion', type=float, default=0.1,
                        help='portion of train dataset used for validation')
    parser.add_argument('-t', '--train_batch_size', type=int, default=128,
                        help='train batch size')
    parser.add_argument('-b', '--valid_batch_size', type=int, default=512,
                        help='validation batch size')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs')
    parser.add_argument('-s', '--save_prefix',
                        help='prefix for saving model and state')
    parser.add_argument('-r', '--restore_state_path',
                        help='restore the previous trained state and starting '
                             'from there')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='which device to run on')
    parser.add_argument('-i', '--image_dir', default=TRAIN_IMG,
                        help='input image dir')
    parser.add_argument('-l', '--target_dir', default=TRAIN_TARGET,
                        help='target image dir')
    parser.add_argument('-m', '--sr_model', help='path for sr model')

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


class TrainDenoise(TrackedTraining):
    def __init__(self, sr_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr_model = sr_model
        self.sr_model.eval()
        self.mse_loss = nn.MSELoss()

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['target'])
        sr_out = self.sr_model(image)
        residual = sr_out - target
        return sr_out, residual

    def train_loss_fn(self, output, target):
        loss = self.mse_loss(output, target)
        return torch.sqrt(loss)


train_dataset = XRayDataset(train_split, args.image_dir, args.target_dir)
valid_dataset = XRayDataset(valid_split, args.image_dir, args.target_dir)

train_loader_config = {'num_workers': 8,
                       'batch_size': args.train_batch_size,
                       'sampler': TrackedRandomSampler(train_dataset)}
inference_loader_config = {'num_workers': 10,
                           'batch_size': args.valid_batch_size,
                           'shuffle': False}

optimizer_config = {'lr': 5e-6}

dncnn = DnCNN(1)

with torch.cuda.device_ctx_manager(args.device):
    print('On device:', torch.cuda.get_device_name(args.device))
    sr_model = cuda(load_model(args.sr_model, ESPCN(2)))
    train = TrainDenoise(sr_model, dncnn, train_dataset, valid_dataset, Adam,
                         args.save_prefix, args.save_prefix,
                         optimizer_config, train_loader_config,
                         inference_loader_config, epochs=args.epochs,
                         save_optimizer=False)

    if args.restore_state_path is not None:
        state_dict = torch.load(args.restore_state_path)
        train.load_state(state_dict)
        del state_dict

    trained_model = train.train()
