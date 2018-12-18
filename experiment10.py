import argparse
import sys

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam

from defines import *
from model.combined import CombinedNetworkDenoiseBefore
from model.dncnn import DnCNN
from model.srgan import GeneratorResNet, Discriminator, FeatureExtractor
from toolbox.misc import cuda
from toolbox.timer import Timer
from toolbox.train import TrackedTrainingGAN, TrackedTraining
from util.XRayDataSet import XRayDataset
from util.test import test

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Experiment 10 SRGAN')
    parser.add_argument('-v', '--valid_portion', type=float, default=0.2,
                        help='portion of train dataset used for validation')
    parser.add_argument('-t', '--train_batch_size', type=int, default=16,
                        help='train batch size')
    parser.add_argument('-b', '--valid_batch_size', type=int, default=16,
                        help='validation batch size')
    parser.add_argument('-e', '--epochs_denoise', type=int, default=1,
                        help='number of epochs for denoise')
    parser.add_argument('-c', '--epochs_pretrain', type=int, default=1,
                        help='number of epochs for pretraining generator')
    parser.add_argument('-u', '--epochs_upsample', type=int, default=1,
                        help='number of epochs for upsample')
    parser.add_argument('-p', '--save_dir',
                        help='dir for saving states')
    parser.add_argument('-g', '--save_optimizer', action='store_true',
                        default=False, help='save optimizer')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='which device to run on')
    parser.add_argument('-n', '--denoise_state_path',
                        help='saved state for denoise model')
    parser.add_argument('-s', '--sr_state_path',
                        help='saved state for sr model')
    parser.add_argument('-m', '--sr_res_state_path',
                        help='saved state for sr pretrain')
    parser.add_argument('-i', '--image_dir', default=TRAIN_IMG,
                        help='input image dir')
    parser.add_argument('-l', '--target_dir', default=TRAIN_TARGET,
                        help='target image dir')
    parser.add_argument('-w', '--test_in', default=TEST_IMG,
                        help='test input dir')
    parser.add_argument('-x', '--vgg19_path', help='vgg19 pretrained path')
    parser.add_argument('-o', '--output_dir', help='output dir for test')
    parser.add_argument('-y', '--denoise_out', required=True,
                        help='output dir for denoised images')
    parser.add_argument('-k', '--skip_denoise', default=False,
                        action='store_true', help='skip the first dncnn')

    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    arg = parser.parse_args()
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


class PretrainSRGAN(TrainDenoise):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['target'])
        return image, target


class TrainUpSample(TrackedTrainingGAN):
    def __init__(self, feature_extractor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = nn.MSELoss()
        self.feature_extractor = cuda(feature_extractor)
        self.feature_extractor.eval()

    def parse_train_batch(self, batch):
        image = cuda(batch['image'])
        target = cuda(batch['target'])
        return image, target

    def train_loss_fn(self, output, target):
        real_features = self.feature_extractor(target)
        fake_features = self.feature_extractor(output)
        content_loss = torch.sqrt(
            self.mse_loss(output, target)) + 2e-6 * torch.sqrt(
            self.mse_loss(fake_features, real_features))
        return content_loss

    def valid_loss_fn(self, output, target):
        return torch.sqrt(self.mse_loss(output, target)) * 255


train_loader_config = {'num_workers': 20,
                       'batch_size': args.train_batch_size}
inference_loader_config = {'num_workers': 20,
                           'batch_size': args.valid_batch_size,
                           'shuffle': False}

image_files = os.listdir(args.image_dir)

train_split, valid_split = train_test_split(image_files,
                                            test_size=args.valid_portion)

with torch.cuda.device_ctx_manager(args.device):
    print('On device:', torch.cuda.get_device_name(args.device))
    if not args.skip_denoise:
        with Timer():
            print('======== Training DnCNN ========')
            train_dataset = XRayDataset(train_split, args.image_dir,
                                        args.target_dir,
                                        down_sample_target=True)
            valid_dataset = XRayDataset(valid_split, args.image_dir,
                                        args.target_dir,
                                        down_sample_target=True)
            optimizer_config = {'lr': 1e-4}
            dncnn = DnCNN(1, built_in_residual=True)
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
            test(dncnn, args.image_dir, args.denoise_out, args.valid_batch_size)

        print('Loading denoised from', args.denoise_out)
        image_files = os.listdir(args.denoise_out)

        train_split, valid_split = train_test_split(
            image_files, test_size=args.valid_portion)

        srgan_image_in = args.denoise_out
    else:
        srgan_image_in = args.image_dir

    with Timer():
        print('======== Pre-train SRGAN ========')
        train_dataset = XRayDataset(train_split, srgan_image_in,
                                    args.target_dir)
        valid_dataset = XRayDataset(valid_split, srgan_image_in,
                                    args.target_dir)
        optimizer_config = {'lr': 1e-4}
        generator = GeneratorResNet()
        save_dir = os.path.join(args.save_dir, 'srres')
        train = PretrainSRGAN(generator, train_dataset, valid_dataset,
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
        generator = train.train()

    with Timer():
        print('======== Training SRGAN ========')
        g_optimizer_config = {'lr': 1e-4}
        d_optimizer_config = {'lr': 5e-4}
        discriminator = Discriminator()
        feature_extractor = FeatureExtractor(args.vgg19_path)
        save_dir = os.path.join(args.save_dir, 'srgan')
        train = TrainUpSample(feature_extractor, discriminator,
                              generator, train_dataset, valid_dataset, Adam,
                              save_dir, d_optimizer_config, g_optimizer_config,
                              train_loader_config,
                              inference_loader_config, nn.BCEWithLogitsLoss(),
                              epochs=args.epochs_upsample,
                              save_optimizer=args.save_optimizer,
                              discriminator_weight=1e-3)
        if args.sr_state_path is not None:
            state_dict = torch.load(args.sr_state_path)
            train.load(state_dict)
            del state_dict
            train_dataset = train.train_dataset
            valid_dataset = train.valid_dataset
        generator, discriminator = train.train()

    if args.skip_denoise:
        final = generator
    else:
        final = CombinedNetworkDenoiseBefore(dncnn, generator,
                                             dncnn_built_in_residual=True)

    test(final, args.test_in, args.output_dir, args.valid_batch_size)
