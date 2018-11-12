import argparse
import sys

import numpy as np
import torch
from scipy.misc import imsave
from torch import nn
from torch.utils.data import DataLoader

from defines import *
from model.espcn import ESPCN
from toolbox.progress_bar import ProgressBar
from util.XRayDataSet import XRayDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 1')
    parser.add_argument('-m', '--mean_pickle',
                        help='mean of all the input images')
    parser.add_argument('-s', '--sd_pickle', help='sd of all the input images')
    parser.add_argument('-b', '--batch_size', type=int, default=18,
                        help='test batch size')
    parser.add_argument('-f', '--saved_model_file',
                        help='file name for model used for submission')
    parser.add_argument('-o', '--out_dir', help='output dir')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='which gpu will this run on')

    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    arg = parser.parse_args()
    return arg


args = parse_args()


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


test_dataset = XRayDataset(TEST_IDX, os.path.join(TEST_IMG, 'test_'))

model = ESPCN(2)

test_loader = DataLoader(test_dataset, num_workers=8,
                         batch_size=args.batch_size, shuffle=False)


def load_model(file_path, model: nn.Module):
    obj = torch.load(file_path)
    model_state = obj['model']
    model.load_state_dict(model_state)
    del model_state
    return model


def test(model: nn.Module, data_loader: DataLoader, save_path: str):
    model = cuda(model)
    torch.cuda.empty_cache()
    model.eval()
    total_steps = len(data_loader)
    progress_bar = ProgressBar(20, ' batch: %d')
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            image = cuda(batch['image'])
            indices = batch['idx']
            output = model(image).cpu().numpy()
            for j, idx in enumerate(indices):
                out_img = np.zeros((128, 128, 3))
                out_img[:, :, 0] = output[j, 0]
                out_img[:, :, 1] = output[j, 0]
                out_img[:, :, 2] = output[j, 0]
                imsave(os.path.join(save_path, 'test_%05d.png' % idx), out_img)
            progress_bar.progress(i / total_steps * 100, i)


with torch.cuda.device_ctx_manager(args.device):
    print('On Device:', torch.cuda.get_device_name(args.device))
    load_model(args.saved_model_file, model)
    test(model, test_loader, args.out_dir)
