import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from defines import *
from toolbox.misc import cuda
from toolbox.progress_bar import ProgressBar
from util.XRayDataSet import XRayDataset


def test(model: nn.Module, in_path: str, save_dir: str, batch_size: int):
    test_files = os.listdir(in_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_dataset = XRayDataset(test_files, in_path)

    data_loader = DataLoader(test_dataset, num_workers=8, batch_size=batch_size,
                             shuffle=False)
    if torch.cuda.is_available():
        model.cuda()
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
                out_img = np.zeros(output.shape[2:] + (4,))
                out_img[:, :, 0] = output[j, 0]
                out_img[:, :, 1] = output[j, 0]
                out_img[:, :, 2] = output[j, 0]
                out_img[:, :, 3] = np.ones(output.shape[2:])
                cv2.imwrite(os.path.join(save_dir, file_name),
                            out_img * 255)
            progress_bar.progress(i / total_steps * 100, i)
