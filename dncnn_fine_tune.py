import argparse
import sys

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam

from defines import *
from model.combined import DenoiseAfterDenoisedUpsample
from model.dncnn import DnCNN
from model.espcn import ESPCN
from toolbox.misc import cuda
from toolbox.timer import Timer
from toolbox.train import TrackedTraining
from util.XRayDataSet import XRayDataset
from util.test import test


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune with DNCNN after denoising and super resolution')
    parser.add_argument('-v', '--valid_portion', type=float, default=0.2,
                        help='portion of train dataset used for validation')
    parser.add_argument('-t', '--train_batch_size', type=int, default=16,
                        help='train batch size')
    parser.add_argument('-b', '--valid_batch_size', type=int, default=16,
                        help='validation batch size')
    parser.add_argument('-e', '--epochs_denoise', type=int, default=1,
                        help='number of epochs for denoise')
    parser.add_argument('-u', '--epochs_upsample', type=int, default=1,
                        help='number of epochs for upsample')
    parser.add_argument('-y', '--finetune_epochs', type=int, default=1,
                        help='epochs for training finetune model')
    parser.add_argument('-p', '--save_dir',
                        help='dir for saving states')
    parser.add_argument('-g', '--save_optimizer', action='store_true',
                        default=False, help='save optimizer')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='which device to run on')
    parser.add_argument('-s', '--sr_state_path',
                        help='saved state for sr model')
    parser.add_argument('-n', '--denoise_state_path',
                        help='saved state for denoise model')
    parser.add_argument('-c', '--finetune_state_path',
                        help='saved state for finetune denoise model')
    parser.add_argument('-i', '--image_dir', default=TRAIN_IMG,
                        help='input image dir')
    parser.add_argument('-l', '--target_dir', default=TRAIN_TARGET,
                        help='target image dir')
    parser.add_argument('-w', '--test_in', default=TEST_IMG,
                        help='test input dir')
    parser.add_argument('-o', '--output_dir', help='output dir for test')
    parser.add_argument('-z', '--interpolation_dncnn', type=float, default=0.0,
                        help='interpolation between perceptual and mse')
    parser.add_argument('-k', '--k_fold_split', help='file for k fold splits',
                        default=None)
    parser.add_argument('-j', '--cv_index', type=int,
                        help='to run validation for which fold')

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = nn.MSELoss()

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['down_sample'])
        residual = image - target
        return image, residual

    def train_loss_fn(self, output, target):
        return torch.sqrt(self.mse_loss(output, target))


class TrainUpSample(TrackedTraining):
    def __init__(self, denoise_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = nn.MSELoss()
        self.denoise_model = denoise_model
        self.denoise_model.eval()

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        denoised = image - self.denoise_model(image)
        target = cuda(batch['target'])
        return denoised, target

    def train_loss_fn(self, output, target):
        return torch.sqrt(self.mse_loss(output, target))


class TrainFinetuneDenoise(TrackedTraining):
    def __init__(self, sr_model, denoise_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = nn.MSELoss()
        self.denoise_model = denoise_model
        self.sr_model = sr_model
        self.denoise_model.eval()
        self.sr_model.eval()

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        denoised = image - self.denoise_model(image)
        enlarged = self.sr_model(denoised)
        target = cuda(batch['target'])
        residual = enlarged - target
        return enlarged, residual

    def train_loss_fn(self, output, target):
        return torch.sqrt(self.mse_loss(output, target))


train_loader_config = {'num_workers': 16,
                       'batch_size': args.train_batch_size}
inference_loader_config = {'num_workers': 16,
                           'batch_size': args.valid_batch_size,
                           'shuffle': False}

with torch.cuda.device_ctx_manager(args.device):
    print('On device:', torch.cuda.get_device_name(args.device))
    with Timer():
        print('======== Training DnCNN ========')
        train_dataset = XRayDataset(train_split, args.image_dir,
                                    args.target_dir,
                                    down_sample_target=True)
        valid_dataset = XRayDataset(valid_split, args.image_dir,
                                    args.target_dir,
                                    down_sample_target=True)
        optimizer_config = {'lr': 5e-5}
        dncnn = DnCNN(1)
        save_dir = os.path.join(args.save_dir, 'dncnn')
        train = TrainDenoise(dncnn, train_dataset, valid_dataset,
                             Adam, save_dir, optimizer_config,
                             train_loader_config, inference_loader_config,
                             epochs=args.epochs_denoise,
                             save_optimizer=args.save_optimizer)
        if args.denoise_state_path is not None:
            state_dict = torch.load(args.denoise_state_path)
            train.load(state_dict)
            del state_dict
            train_split = train.train_dataset.file_names
            valid_split = train.valid_dataset.file_names
        dncnn = train.train()

        print('======== Training ESPCN ========')
        train_dataset = XRayDataset(train_split, args.image_dir,
                                    args.target_dir)
        valid_dataset = XRayDataset(valid_split, args.image_dir,
                                    args.target_dir)

        optimizer_config = {'lr': 2e-5}
        espcn = ESPCN(2)
        save_dir = os.path.join(args.save_dir, 'espcn')
        train = TrainUpSample(dncnn, espcn, train_dataset,
                              valid_dataset, Adam, save_dir, optimizer_config,
                              train_loader_config, inference_loader_config,
                              epochs=args.epochs_upsample,
                              save_optimizer=args.save_optimizer)
        if args.sr_state_path is not None:
            state_dict = torch.load(args.sr_state_path)
            train.load(state_dict)
            del state_dict
            train_dataset = train.train_dataset
            valid_dataset = train.valid_dataset
        espcn = train.train()

        print("======== Denoise Finetune ========")
        train_dataset = XRayDataset(train_split, args.image_dir,
                                    args.target_dir)
        valid_dataset = XRayDataset(valid_split, args.image_dir,
                                    args.target_dir)
        optimizer_config = {'lr': 1e-5}
        dncnn_finetuned = DnCNN(1)
        save_dir = os.path.join(args.save_dir, 'finetune')
        train = TrainFinetuneDenoise(espcn, dncnn, dncnn_finetuned,
                                     train_dataset,
                                     valid_dataset, Adam, save_dir,
                                     optimizer_config,
                                     train_loader_config,
                                     inference_loader_config,
                                     epochs=args.finetune_epochs,
                                     save_optimizer=args.save_optimizer)
        if args.finetune_state_path is not None:
            state_dict = torch.load(args.finetune_state_path)
            train.load(state_dict)
            del state_dict
            train_split = train.train_dataset.file_names
            valid_split = train.valid_dataset.file_names
        dncnn_finetuned = train.train()

    combined = DenoiseAfterDenoisedUpsample(dncnn, espcn, dncnn_finetuned)

    test(combined, args.test_in, args.output_dir, args.valid_batch_size)
