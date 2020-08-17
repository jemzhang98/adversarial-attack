import os
from argparse import ArgumentParser
from feature_extractor import extract_features_and_save


def main(args):
    if args.train_dir:
        train_files = os.listdir(args.train_dir)
        train_files = [os.path.join(args.train_dir, f) for f in train_files]
        extract_features_and_save(train_files, os.path.join(args.output_dir, 'train'))

    if args.test_dir:
        test_files = os.listdir(args.test_dir)
        test_files = [os.path.join(args.test_dir, f) for f in test_files]
        extract_features_and_save(test_files, os.path.join(args.output_dir, 'test'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='./features/')
    args = parser.parse_args()

    main(args)
