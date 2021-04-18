""" Splits motsynth dataset into train, validation and test"""
import logging
import sys
from configparser import ConfigParser
from pathlib import Path

import click
import numpy as np
from IPython.core import ultratb

import mots_tracker

sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dd",
    "--data_dir",
    "data_dir",
    default="/home/vy/university/thesis/datasets/MOTSynth",
    type=click.Path(exists=True),
    help="Path to MOTSynth preprocessed annotations",
)
@click.option(
    "--od",
    "--output_dir",
    "output_dir",
    default="/home/vy/university/thesis/datasets/MOTSynth",
    type=click.Path(exists=True),
    help="Output path of the for the split files",
)
@click.option(
    "--p",
    "--proportions",
    "proportions",
    type=(float, float, float),
    default=(0.6, 0.2, 0.2),
    help="Proportions to split the dataset to train/val/test (should sum up to 1)",
)
@click.version_option(mots_tracker.__version__)
def main(data_dir, output_dir, proportions):
    data_dir = Path(data_dir)
    print("Splitting sequences in proportions: {}".format(proportions))
    np.random.seed(42)
    static_seq_ids, dynamic_seq_ids = [], []
    parser = ConfigParser()
    for seq_path in sorted((data_dir / "all").glob("*"), key=lambda p: str(p)):
        parser.read(str(seq_path / "seqinfo.ini"), encoding=None)
        seq_name = parser.get("Sequence", "name")
        dynamic = parser.getint("Sequence", "dynamic")
        if dynamic:
            dynamic_seq_ids.append(seq_name)
        else:
            static_seq_ids.append(seq_name)
    print("Static sequences: {}".format(len(static_seq_ids)))
    print("Dynamic sequences: {}".format(len(dynamic_seq_ids)))
    static_seq_ids = np.array(static_seq_ids)
    dynamic_seq_ids = np.array(dynamic_seq_ids)

    def split_data(data, p=(0.6, 0.2, 0.2)):
        """ Splits data in train, val"""
        np.random.shuffle(data)
        n = data.shape[0]
        vstart = int((p[0]) * n)
        tstart = int((p[0] + p[1]) * n)
        return np.split(data, [vstart, tstart])

    st_train, st_val, st_test = split_data(static_seq_ids, proportions)
    print(
        "Static split is train: {}, validation: {}, test: {}".format(
            st_train.shape[0], st_val.shape[0], st_test.shape[0]
        )
    )
    d_train, d_val, d_test = split_data(dynamic_seq_ids, proportions)
    print(
        "Dynamic split is train: {}, validation: {}, test: {}".format(
            d_train.shape[0], d_val.shape[0], d_test.shape[0]
        )
    )

    train_ids = np.concatenate((st_train, d_train))
    val_ids = np.concatenate((st_val, d_val))
    test_ids = np.concatenate((st_test, d_test))

    assert not set(train_ids) & set(val_ids)
    assert not set(train_ids) & set(test_ids)
    assert not set(val_ids) & set(test_ids)

    print(
        "Overall split is train: {}, validation: {}, test: {}".format(
            train_ids.shape[0], val_ids.shape[0], test_ids.shape[0]
        )
    )
    output_path = Path(output_dir) / "split_{}_{}_{}".format(*proportions)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(output_path / "train.txt"), train_ids, fmt="%s")
    np.savetxt(str(output_path / "val.txt"), val_ids, fmt="%s")
    np.savetxt(str(output_path / "test.txt"), test_ids, fmt="%s")


if __name__ == "__main__":
    main()
