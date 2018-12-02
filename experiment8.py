import argparse
import sys

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam

from defines import *
from model.perceptual_loss import PerceptualLoss
from model.redcnn import REDCNN
from model.espcn import ESPCN
from toolbox.misc import cuda, load_model
from toolbox.train import TrackedTraining
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
    parser.add_argument('-p', '--save_dir',
                        help='dir for saving states')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='which device to run on')
    parser.add_argument('-n', '--sr_state_path',
                        help='saved state for sr model')
    parser.add_argument('-i', '--image_dir', default=TRAIN_IMG,
                        help='input image dir')
    parser.add_argument('-l', '--target_dir', default=TRAIN_TARGET,
                        help='target image dir')
    parser.add_argument('-w', '--test_in', default=TEST_IMG,
                        help='test input dir')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='output dir for test')
    parser.add_argument('-f', '--denoise_pretrained', required=True,
                        help='location of pretrained denoise model')
    parser.add_argument('-u', '--save_optimizer', action='store_true',
                        default=False, help='Store optimizer or not')
    parser.add_argument('-j', '--vgg11_path', required=True)
    parser.add_argument('-x', '--interpolation', required=True,
                        type=float,
                        help='interpolation between perceptual and mse')

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


class TrainDenoise(TrackedTraining):
    def __init__(self, perceptual_pretrained_path, interpolation, denoise_model,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 <= interpolation <= 1
        self.denoise_model = denoise_model
        self.denoise_model.eval()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = cuda(PerceptualLoss(perceptual_pretrained_path))
        self.loss_interpolation = interpolation

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['target'])
        return self.denoise_model(image), target

    def train_loss_fn(self, output, target):
        mse = self.mse_loss(output, target)
        perceptual = self.perceptual_loss(output, target)
        return perceptual * self.loss_interpolation + mse * (
                1 - self.loss_interpolation)

    def valid_loss_fn(self, output, target):
        return torch.sqrt(self.mse_loss(output, target)) * 255


train_loader_config = {'num_workers': 20,
                       'batch_size': args.train_batch_size}
inference_loader_config = {'num_workers': 20,
                           'batch_size': args.valid_batch_size,
                           'shuffle': False}

with torch.cuda.device_ctx_manager(args.device):
    print('On device:', torch.cuda.get_device_name(args.device))

    train_dataset = XRayDataset(train_split, args.image_dir, args.target_dir)
    valid_dataset = XRayDataset(valid_split, args.image_dir, args.target_dir)
    optimizer_config = {'lr': 1e-5}
    redcnn = REDCNN()
    redcnn = load_model(args.denoise_pretrained, redcnn)
    espcn = ESPCN(2)
    train = TrainDenoise(args.vgg11_path, args.interpolation, redcnn,
                         train_dataset, valid_dataset, Adam,
                         args.save_dir, optimizer_config, train_loader_config,
                         inference_loader_config,
                         epochs=args.epochs,
                         save_optimizer=args.save_optimizer)
    if args.denoise_state_path is not None:
        state_dict = torch.load(args.sr_state_path)
        train.load(state_dict)
        del state_dict
    espcn = train.train()

    test(espcn, args.test_in, args.output_dir, args.valid_batch_size)
