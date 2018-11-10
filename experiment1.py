import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 1')
    parser.add_argument('-m', '--mean_pickle',
                        help='mean of all the input images')
    parser.add_argument('-s', '--sd_pickle', help='sd of all the input images')
    parser.add_argument('-v', '--valid_portion', type=float, default=0.1,
                        help='portion of train dataset used for validation')
    parser.add_argument('-t', '--train_batch_size', type=int, default=5,
                        help='train batch size')
    parser.add_argument('-b', '--valid_batch_size', type=int, default=18,
                        help='validation batch size')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs')
    parser.add_argument('-p', '--save_model_prefix',
                        help='prefix for model saving files')
    parser.add_argument('-f', '--save_state_prefix',
                        help='prefix for saving trainer state')
    parser.add_argument('-r', '--restore_state_path',
                        help='restore the previous trained state and starting '
                             'from there')

    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    arg = parser.parse_args()
    return arg


