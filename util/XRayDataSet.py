import os

import torch
from scipy.misc import imresize
from skimage.data import imread
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from toolbox.states import State, Trackable


class XRayDataset(Dataset, Trackable):
    def __init__(self, file_names, img_path_pfx, target_path_pfx=None,
                 transform=None, chan4=False, down_sample_target=False):
        self.file_names = State(file_names)
        self.img_path_pfx = img_path_pfx
        self.target_path_pfx = target_path_pfx
        self.transform = transform
        self.chan4 = chan4
        self.down_sample_target = down_sample_target

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        file_name = self.file_names[item]
        to_tensor = ToTensor()

        if self.down_sample_target:
            target = imread(os.path.join(self.target_path_pfx, file_name))
            image = imresize(target, 0.5)
            target = torch.unsqueeze(to_tensor(target)[0], 0)
            image = torch.unsqueeze(to_tensor(image)[0], 0)
            return {'image': image, 'file_name': file_name, 'target': target}

        # Only take the first channel. All channels are the same (gray scale)
        image = imread(os.path.join(self.img_path_pfx, file_name))
        image = to_tensor(image)
        if not self.chan4:
            image = torch.unsqueeze(image[0], 0)
        result = [('image', image), ('file_name', file_name)]
        if self.target_path_pfx is not None:
            target = imread(os.path.join(self.target_path_pfx, file_name))
            target = torch.unsqueeze(to_tensor(target)[0], 0)
            result.append(('target', target))
        if self.transform is not None:
            result = list(map(lambda x: (x[0], self.transform(x[1])), result))
        return dict(result)
