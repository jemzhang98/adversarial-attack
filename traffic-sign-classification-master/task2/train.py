from argparse import ArgumentParser
from pytorch_lightning import Trainer
from stn_cnn import StnCnn


def main(args):
    model = StnCnn(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    # add PROGRAM level args
    # parser.add_argument('--conda_env', type=str, default='some_name')

    # add model specific args
    parser = StnCnn.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)

