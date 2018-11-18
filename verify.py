from torch.utils.data import DataLoader

from defines import *
from util.XRayDataSet import XRayDataset

train_dataset = XRayDataset(TRAIN_IDX, os.path.join(TRAIN_IMG, 'train_'),
                            os.path.join(TRAIN_TARGET, 'train_'), chan4=True)

test_dataset = XRayDataset(TEST_IDX, os.path.join(TEST_IMG, 'train_'),
                           chan4=True)

train_loader = DataLoader(train_dataset, num_workers=5)
test_loader = DataLoader(test_dataset, num_workers=5)


def verify(loader):
    for i, batch in enumerate(loader):
        image = batch['image'][0]
        idx = batch['idx']
        if sum(image[0] != image[1]) == 0 and sum(image[1] != image[2]) == 0:
            continue
        print('%d is not the same' % idx)


verify(train_loader)
verify(test_loader)
