from skimage.data import imread
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class XRayDataset(Dataset):
    def __init__(self, indices, img_path_pfx, target_path_pfx=None,
                 transform=None):
        self.indices = indices
        self.img_path_pfx = img_path_pfx
        self.target_path_pfx = target_path_pfx
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        # TODO: Make sure to convert to single channel
        idx = self.indices[item]
        to_tensor = ToTensor()
        image = imread('%s%d.png' % (self.img_path_pfx, idx))
        image = to_tensor(image)
        result = [('image', image)]
        if self.target_path_pfx is not None:
            target = imread('%s%d.png' % (self.target_path_pfx, idx))
            target = to_tensor(target)
            result.append(('target', target))
        if self.transform is not None:
            result = list(map(lambda x: (x[0], self.transform(x[1])), result))
        return dict(result)
