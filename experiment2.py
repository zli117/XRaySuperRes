import argparse
import sys

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import SGD

from defines import *
from model.srresnet import NetG
from toolbox.torch_state_samplers import TrackedRandomSampler
from toolbox.train import TrackedTraining
from util.XRayDataSet import XRayDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 1')
    parser.add_argument('-m', '--mean_pickle',
                        help='mean of all the input images')
    parser.add_argument('-s', '--sd_pickle', help='sd of all the input images')
    parser.add_argument('-v', '--valid_portion', type=float, default=0.1,
                        help='portion of train dataset used for validation')
    parser.add_argument('-t', '--train_batch_size', type=int, default=128,
                        help='train batch size')
    parser.add_argument('-b', '--valid_batch_size', type=int, default=512,
                        help='validation batch size')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs')
    parser.add_argument('-p', '--save_model_prefix',
                        help='prefix for model saving files')
    parser.add_argument('-f', '--save_state_prefix',
                        help='prefix for saving trainer state')
    parser.add_argument('-r', '--restore_state_path',
                        help='restore the previous trained state and starting '
                             'from there')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='which device to run on')

    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    arg = parser.parse_args()
    return arg


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


args = parse_args()

train_split, valid_split = train_test_split(TRAIN_IDX,
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

    def parse_valid_batch(self, batch):
        image, target = self.parse_train_batch(batch)
        return image, target * 255

    def loss_fn(self, output, target):
        loss = self.mse_loss(output, target)
        return torch.sqrt(loss)


train_dataset = XRayDataset(train_split, os.path.join(TRAIN_IMG, 'train_'),
                            os.path.join(TRAIN_TARGET, 'train_'))
valid_dataset = XRayDataset(valid_split, os.path.join(TRAIN_IMG, 'train_'),
                            os.path.join(TRAIN_TARGET, 'train_'))

model = NetG()

train_loader_config = {'num_workers': 8,
                       'batch_size': args.train_batch_size,
                       'sampler': TrackedRandomSampler(train_dataset)}
inference_loader_config = {'num_workers': 10,
                           'batch_size': args.valid_batch_size,
                           'shuffle': False}

optimizer_config = {'lr': 1e-5, 'momentum': 0.9, 'weight_decay': 1e-6}

with torch.cuda.device_ctx_manager(args.device):
    print('On device:', torch.cuda.get_device_name(args.device))
    train = Train(model, train_dataset, valid_dataset, SGD,
                  args.save_model_prefix,
                  args.save_state_prefix, optimizer_config, train_loader_config,
                  inference_loader_config, epochs=args.epochs,
                  save_optimizer=False)

    if args.restore_state_path is not None:
        state_dict = torch.load(args.restore_state_path)
        train.load_state(state_dict)
        del state_dict

    trained_model = train.train()
