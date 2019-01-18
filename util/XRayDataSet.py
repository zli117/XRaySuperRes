import os

import torch
from scipy.misc import imresize
from skimage.data import imread
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from toolbox.states import State, Trackable


class XRayDataset(Dataset, Trackable):
    def __init__(self, file_names, img_dir, target_dir=None,
                 transform=None, chan1=True, down_sample_target=False):
        super().__init__()
        self.file_names = State(file_names)
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.transform = transform
        self.chan1 = chan1
        self.down_sample_target = down_sample_target

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        file_name = self.file_names[item]
        to_tensor = ToTensor()

        # Only take the first channel. All channels are the same (gray scale)
        image = imread(os.path.join(self.img_dir, file_name))
        if len(image.shape) == 3:
            image = to_tensor(image)
        else:
            image = torch.Tensor(image / 65536)
            image = torch.unsqueeze(image, 0)
        if self.chan1:
            image = torch.unsqueeze(image[0], 0)
        result = [('image', image), ('file_name', file_name)]
        if self.target_dir is not None:
            target = imread(os.path.join(self.target_dir, file_name))
            if self.down_sample_target:
                down_sample = imresize(target, 0.5)
                down_sample = torch.unsqueeze(to_tensor(down_sample)[0], 0)
                result.append(('down_sample', down_sample))
            target = torch.unsqueeze(to_tensor(target)[0], 0)
            result.append(('target', target))
        if self.transform is not None:
            result = list(map(lambda x: (x[0], self.transform(x[1])), result))
        return dict(result)
