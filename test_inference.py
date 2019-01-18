import argparse
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from defines import *
from model.espcn import ESPCN
from toolbox.progress_bar import ProgressBar
from toolbox.misc import load_model
from util.XRayDataSet import XRayDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 1')
    parser.add_argument('-b', '--batch_size', type=int, default=18,
                        help='test batch size')
    parser.add_argument('-f', '--model_path',
                        help='path for saved model')
    parser.add_argument('-i', '--in_dir', default=TEST_IMG, help='input dir')
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


test_files = os.listdir(args.in_dir)

test_dataset = XRayDataset(test_files, args.in_dir)

test_loader = DataLoader(test_dataset, num_workers=8,
                         batch_size=args.batch_size, shuffle=False)


def test(model: nn.Module, data_loader: DataLoader, save_path: str):
    model = cuda(model)
    torch.cuda.empty_cache()
    model.eval()
    total_steps = len(data_loader)
    progress_bar = ProgressBar(20, ' batch: %d')
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            image = cuda(batch['image'])
            file_names = batch['file_name']
            output = model(image).cpu().numpy()
            for j, file_name in enumerate(file_names):
                out_img = np.zeros((128, 128, 4))
                out_img[:, :, 0] = output[j, 0]
                out_img[:, :, 1] = output[j, 0]
                out_img[:, :, 2] = output[j, 0]
                out_img[:, :, 3] = np.ones((128, 128))
                cv2.imwrite(os.path.join(save_path, file_name),
                            out_img * 255)
            progress_bar.progress(i / total_steps * 100, i)


with torch.cuda.device_ctx_manager(args.device):
    print('On Device:', torch.cuda.get_device_name(args.device))
    espcn = ESPCN(2)
    # dncnn = DnCNN(1)
    # combined = CombinedNetworkDenoiseAfter(espcn, dncnn)
    load_model(args.model_path, espcn)
    test(espcn, test_loader, args.out_dir)
