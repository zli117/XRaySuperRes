import argparse
import sys

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam

from defines import *
from model.edsr import EDSR
from toolbox.misc import cuda
from toolbox.timer import Timer
from toolbox.train import TrackedTraining
from util.XRayDataSet import XRayDataset
from util.test import test

torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Experiment 11 EDSR')
    parser.add_argument('-v', '--valid_portion', type=float, default=0.2,
                        help='portion of train dataset used for validation')
    parser.add_argument('-t', '--train_batch_size', type=int, default=16,
                        help='train batch size')
    parser.add_argument('-b', '--valid_batch_size', type=int, default=16,
                        help='validation batch size')
    parser.add_argument('-c', '--epochs_pretrain', type=int, default=300,
                        help='number of epochs for srresnet')
    parser.add_argument('-p', '--save_dir', required=True,
                        help='dir for saving states')
    parser.add_argument('-g', '--save_optimizer', action='store_true',
                        default=False, help='save optimizer')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='which device to run on')
    parser.add_argument('-s', '--sr_res_state_path', default=None,
                        help='saved state for sr model')
    parser.add_argument('-i', '--image_dir', default=TRAIN_IMG,
                        help='input image dir')
    parser.add_argument('-l', '--target_dir', default=TRAIN_TARGET,
                        help='target image dir')
    parser.add_argument('-w', '--test_in', default=TEST_IMG,
                        help='test input dir')
    parser.add_argument('-x', '--vgg19_path', help='vgg19 pretrained path')
    parser.add_argument('-o', '--output_dir',
                        help='output dir for test')
    parser.add_argument('-y', '--loss', type=str,
                        help='which loss to use: l1 or l2')
    parser.add_argument('-z', '--smaller_edsr', default=False,
                        action='store_true', help='using smaller edsr')

    arg = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)
    return arg


args = parse_args()


class TrainDenoise(TrackedTraining):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = nn.MSELoss()

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['down_sample'])
        return image, target

    def train_loss_fn(self, output, target):
        return torch.sqrt(self.mse_loss(output, target))


class PretrainSRGAN(TrackedTraining):
    def __init__(self, loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        self.mse_loss = mse_loss
        if loss == 'l1':
            print('Using l1 loss')
            self.train_loss = l1_loss
        elif loss == 'l2':
            print('Using l2 loss')
            self.train_loss = lambda output, target: torch.sqrt(
                mse_loss(output, target))
        else:
            print('invalid loss type')
            exit(1)

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['target'])
        return image, target

    def train_loss_fn(self, output, target):
        return self.train_loss(output, target)

    def valid_loss_fn(self, output, target):
        return torch.sqrt(self.mse_loss(output, target))


class TrainFinetuneDenoise(TrackedTraining):
    def __init__(self, sr_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = nn.MSELoss()
        self.sr_model = sr_model
        self.sr_model.eval()

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        enlarged = self.sr_model(image)
        target = cuda(batch['target'])
        return enlarged, target

    def train_loss_fn(self, output, target):
        return torch.sqrt(self.mse_loss(output, target))


train_loader_config = {'num_workers': 20,
                       'batch_size': args.train_batch_size}
inference_loader_config = {'num_workers': 20,
                           'batch_size': args.valid_batch_size,
                           'shuffle': False}

with torch.cuda.device_ctx_manager(args.device):
    print('On device:', torch.cuda.get_device_name(args.device))

    print('Loading noisy image from', args.image_dir)
    image_files = os.listdir(args.image_dir)

    train_split, valid_split = train_test_split(image_files,
                                                test_size=args.valid_portion)

    with Timer():
        print('======== EDSR ========')
        train_dataset = XRayDataset(train_split, args.image_dir,
                                    args.target_dir)
        valid_dataset = XRayDataset(valid_split, args.image_dir,
                                    args.target_dir)
        optimizer_config = {'lr': 5e-5}
        if args.smaller_edsr:
            print('using small network')
            net_config = {'n_res_blocks': 16, 'n_feats': 64}
        else:
            print('using large network')
            net_config = {'n_res_blocks': 32, 'n_feats': 128}
        edsrnet = EDSR(**net_config)
        save_dir = os.path.join(args.save_dir, 'srres')
        train = PretrainSRGAN(args.loss, edsrnet, train_dataset, valid_dataset,
                              Adam, save_dir, optimizer_config,
                              train_loader_config, inference_loader_config,
                              epochs=args.epochs_pretrain,
                              save_optimizer=args.save_optimizer)
        if args.sr_res_state_path is not None:
            state_dict = torch.load(args.sr_res_state_path)
            train.load(state_dict)
            del state_dict
            train_dataset = train.train_dataset
            valid_dataset = train.valid_dataset
        edsrnet = train.train()

    test(edsrnet, args.test_in, args.output_dir, args.valid_batch_size)
