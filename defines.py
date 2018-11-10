import os

DATA_DIR = 'data/'
TRAIN_IMG = os.path.join(DATA_DIR, 'train_images_128x128')
TRAIN_TARGET = os.path.join(DATA_DIR, 'train_images_64x64')
TEST_IMG = os.path.join(DATA_DIR, 'test_images_64x64')
TRAIN_IDX = list(range(4000, 20000))
TEST_IDX = list(range(0, 4000))
