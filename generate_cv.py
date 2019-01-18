import argparse
import pickle
import sys

from sklearn.model_selection import KFold

from defines import *

parser = argparse.ArgumentParser(
    description='Experiment 7 Independent Fine-tune')

parser.add_argument('-i', '--image-dir', default=TRAIN_IMG,
                    help='input image dir')
parser.add_argument('-k', '--k-folds', type=int, help='number of folds')
parser.add_argument('-s', '--shuffle', type=bool, action='store_true',
                    default=False)
parser.add_argument('-o', '--save-file',
                    help='file to save the pickle for the split')

if len(sys.argv) == 1:
    parser.print_help()
    exit(0)

args = parser.parse_args()

image_files = os.listdir(args.image_dir)

k_fold = KFold(args.k_folds, shuffle=args.suffle)

train_test = [split for split in k_fold.split(image_files)]

with open(args.save_file, 'wb') as file:
    pickle.dump(train_test, file)
