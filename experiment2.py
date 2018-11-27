import argparse
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from defines import *
from model.espcn import ESPCN
from toolbox.kfold import TrackedKFold
from toolbox.torch_state_samplers import TrackedRandomSampler
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
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs')
    parser.add_argument('-f', '--save_dir',
                        help='dir for saving trainer state and model')
    parser.add_argument('-r', '--restore_state_path',
                        help='restore the previous trained state and starting '
                             'from there')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='which device to run on')
    parser.add_argument('-k', '--k_folds', type=int,
                        help='k folds of validation')
    parser.add_argument('-a', '--down_sample', action='store_true',
                        help='Down sampling the target to get input')

    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    arg = parser.parse_args()
    return arg


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


args = parse_args()

if args.down_sample:
    print('Using input as down sampled from target')


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

    def train_loss_fn(self, output, target):
        loss = self.mse_loss(output, target)
        return torch.sqrt(loss)


inference_loader_config = {'num_workers': 10,
                           'batch_size': args.valid_batch_size,
                           'shuffle': False}


class KFold(TrackedKFold):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_obj(self, train_idx):
        train_split, valid_split = train_test_split(
            train_idx, test_size=args.valid_portion)
        train_split = np.array(TRAIN_IDX)[train_split]
        valid_split = np.array(TRAIN_IDX)[valid_split]
        train_dataset = XRayDataset(train_split,
                                    os.path.join(TRAIN_IMG, 'train_'),
                                    os.path.join(TRAIN_TARGET, 'train_'),
                                    down_sample_target=args.down_sample)
        valid_dataset = XRayDataset(valid_split,
                                    os.path.join(TRAIN_IMG, 'train_'),
                                    os.path.join(TRAIN_TARGET, 'train_'),
                                    down_sample_target=args.down_sample)
        train_loader_config = {'num_workers': 8,
                               'batch_size': args.train_batch_size,
                               'sampler': TrackedRandomSampler(train_dataset)}
        save_path_pfx = '%s%d' % (self.state_save_dir, self.fold_idx)
        train_obj = Train(self.model, train_dataset, valid_dataset, Adam,
                          save_path_pfx, save_path_pfx,
                          optimizer_config, train_loader_config,
                          inference_loader_config, epochs=args.epochs)
        return train_obj

    def test(self, test_idx):
        test_split = np.array(TRAIN_IDX)[test_idx]
        test_dataset = XRayDataset(test_split,
                                   os.path.join(TRAIN_IMG, 'train_'),
                                   os.path.join(TRAIN_TARGET, 'train_'),
                                   down_sample_target=args.down_sample)
        return self.train_obj.validate(
            DataLoader(test_dataset, **inference_loader_config))


model = ESPCN(2)

optimizer_config = {'lr': 1e-5}  # , 'momentum': 0.9, 'weight_decay': 1e-6}

with torch.cuda.device_ctx_manager(args.device):
    print('On device:', torch.cuda.get_device_name(args.device))
    k_fold = KFold(args.save_dir, model, args.k_folds, len(TRAIN_IDX))

    if args.restore_state_path is not None:
        state_dict = torch.load(args.restore_state_path)
        k_fold.load(state_dict)
        del state_dict

    k_fold.run()
    print(k_fold.history)
