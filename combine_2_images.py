import argparse
import os
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from toolbox.progress_bar import ProgressBar
from util.XRayDataSet import XRayDataset

parser = argparse.ArgumentParser('Combine 2 image for pix2pix')
parser.add_argument('-v', '--valid_portion', type=float, default=0.2,
                    help='portion of train dataset used for validation')
parser.add_argument('-a', help='dir for A images')
parser.add_argument('-b', help='dir for B images')
parser.add_argument('-o', help='output dir')

if len(sys.argv) == 1:
    parser.print_help()
    exit(0)

args = parser.parse_args()

files = os.listdir(args.a)

train_split, valid_split = train_test_split(files, test_size=args.valid_portion)

if not os.path.exists(args.o):
    os.makedirs(args.o)

train_a_dataset = XRayDataset(train_split, args.a)
train_b_dataset = XRayDataset(train_split, args.b)
train_a_loader = DataLoader(train_a_dataset, num_workers=20)
train_b_loader = DataLoader(train_b_dataset, num_workers=20)

val_a_dataset = XRayDataset(valid_split, args.a)
val_b_dataset = XRayDataset(valid_split, args.b)
val_a_loader = DataLoader(val_a_dataset, num_workers=20)
val_b_loader = DataLoader(val_b_dataset, num_workers=20)

progress_bar = ProgressBar(20, ' batch: %d')

output_train = os.path.join(args.o, 'train')
output_val = os.path.join(args.o, 'val')


def generate_img(a_loader, b_loader, output):
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
        cv2.imwrite(os.path.join(output, file_name), out_img * 255)
        progress_bar.progress(step / total_steps * 100, step)
        step += 1


generate_img(train_a_loader, train_b_loader, output_train)
generate_img(val_a_loader, val_b_loader, output_val)
