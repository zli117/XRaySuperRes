import argparse
import os
import sys

import cv2
import numpy as np
from torch.utils.data import DataLoader

from toolbox.progress_bar import ProgressBar
from util.XRayDataSet import XRayDataset

parser = argparse.ArgumentParser('Combine 2 image for pix2pix')
parser.add_argument('-a', help='dir for A images')
parser.add_argument('-b', help='dir for B images')
parser.add_argument('-o', help='output dir')

if len(sys.argv) == 1:
    parser.print_help()
    exit(0)

args = parser.parse_args()

files = os.listdir(args.a)

if not os.path.exists(args.o):
    os.makedirs(args.o)

a_dataset = XRayDataset(files, args.a)
b_dataset = XRayDataset(files, args.b)
a_loader = DataLoader(a_dataset, num_workers=20)
b_loader = DataLoader(b_dataset, num_workers=20)

progress_bar = ProgressBar(20, ' batch: %d')

step = 0
total_steps = len(a_loader)

for batch_a, batch_b in zip(a_loader, b_loader):
    file_name = batch_a['file_name'][0]
    image_a = batch_a['image'][0, 0]
    image_b = batch_b['image'][0, 0]
    image_ab = np.concatenate([image_a, image_b])
    out_img = np.zeros(image_ab.shape + (4,))
    for i in range(3):
        out_img[:, :, i] = image_ab
    out_img[:, :, 3] = np.ones(image_ab.shape)
    cv2.imwrite(os.path.join(args.o, file_name), out_img * 255)
    progress_bar.progress(step / total_steps * 100, step)
    step += 1
