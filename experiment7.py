import argparse
import sys

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam

from defines import *
from model.discriminator import Discriminator
from model.dncnn import DnCNN
from toolbox.misc import cuda
from toolbox.train import TrackedTrainingGAN
from util.XRayDataSet import XRayDataset
from util.test import test


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
    parser.add_argument('-p', '--save_dir',
                        help='dir for saving states')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='which device to run on')
    parser.add_argument('-s', '--sr_state_path',
                        help='saved state for sr model')
    parser.add_argument('-n', '--denoise_state_path',
                        help='saved state for denoise model')
    parser.add_argument('-c', '--combine_state_path',
                        help='saved state for combined model')
    parser.add_argument('-i', '--image_dir', default=TRAIN_IMG,
                        help='input image dir')
    parser.add_argument('-l', '--target_dir', default=TRAIN_TARGET,
                        help='target image dir')
    parser.add_argument('-w', '--test_in', default=TEST_IMG,
                        help='test input dir')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='output dir for test')

    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    arg = parser.parse_args()
    return arg


args = parse_args()

image_files = os.listdir(args.image_dir)

train_split, valid_split = train_test_split(image_files,
                                            test_size=args.valid_portion)
print('train split size: %d' % len(train_split))
print('valid split size: %d' % len(valid_split))


class TrainDenoise(TrackedTrainingGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = nn.MSELoss()

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['down_sample'])
        return image, target

    def train_loss_fn(self, output, target):
        output = (output + 1) / 2
        loss = self.mse_loss(output, target)
        return loss


train_loader_config = {'num_workers': 8,
                       'batch_size': args.train_batch_size}
inference_loader_config = {'num_workers': 10,
                           'batch_size': args.valid_batch_size,
                           'shuffle': False}

with torch.cuda.device_ctx_manager(args.device):
    print('On device:', torch.cuda.get_device_name(args.device))

    print('======== Training DnCNN ========')
    train_dataset = XRayDataset(train_split, args.image_dir, args.target_dir,
                                down_sample_target=True)
    valid_dataset = XRayDataset(valid_split, args.image_dir, args.target_dir,
                                down_sample_target=True)
    optimizer_config = {'lr': 5e-6}
    dncnn = DnCNN(1, tanh_out=True, built_in_residual=True)
    discriminator = Discriminator()
    save_pfx = args.save_dir + 'dncnn'
    train = TrainDenoise(discriminator, dncnn, train_dataset, valid_dataset,
                         Adam, save_pfx, optimizer_config,
                         train_loader_config, inference_loader_config,
                         epochs=args.epochs, save_optimizer=False)
    if args.denoise_state_path is not None:
        state_dict = torch.load(args.denoise_state_path)
        train.load(state_dict)
        del state_dict
    dncnn, discriminator = train.train()

    test(dncnn, args.test_in, args.output_dir, args.valid_batch_size)