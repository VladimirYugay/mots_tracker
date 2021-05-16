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
    type=(float, float),
    default=(0.8, 0.2),
    help="Proportions to split the dataset to train/val (should sum up to 1)",
)
@click.option(
    "--ts",
    "--test_seqs_path",
    "test_seqs_path",
    default="/home/vy/university/thesis/datasets/MOTSynth",
    type=click.Path(exists=True),
    help="File with sequences reserved for test",
)
@click.option(
    "--only_dynamic",
    "only_dynamic",
    help="Use only dynamic sequences",
    flag_value=True,
)
@click.version_option(mots_tracker.__version__)
def main(data_dir, output_dir, proportions, test_seqs_path, only_dynamic):
    data_dir = Path(data_dir)
    print("Splitting sequences in proportions: {}".format(proportions))
    np.random.seed(42)

    with open(test_seqs_path, "r") as test_file:
        test_seqs = set([line.strip() for line in test_file.readlines()])
        print(test_seqs)

    print("Test seqs num: {}", len(test_seqs))
    static_seq_ids, dynamic_seq_ids = [], []
    parser = ConfigParser()
    for seq_path in sorted((data_dir / "all").glob("*"), key=lambda p: str(p)):
        parser.read(str(seq_path / "seqinfo.ini"), encoding=None)
        seq_name = parser.get("Sequence", "name")
        dynamic = parser.getint("Sequence", "dynamic")
        if seq_name in test_seqs:
            continue
        if dynamic:
            dynamic_seq_ids.append(seq_name)
        elif not only_dynamic:
            static_seq_ids.append(seq_name)
    print("Static sequences: {}".format(len(static_seq_ids)))
    print("Dynamic sequences: {}".format(len(dynamic_seq_ids)))
    static_seq_ids = np.array(static_seq_ids)
    dynamic_seq_ids = np.array(dynamic_seq_ids)

    def split_data(data, p=(0.8, 0.2)):
        """ Splits data in train, val"""
        if data.shape[0] == 0:
            return np.array([]), np.array([])
        np.random.shuffle(data)
        n = data.shape[0]
        return data[: int(n * p[0])], data[int(n * p[0]) :]

    st_train, st_val = split_data(static_seq_ids, proportions)
    print(
        "Static split is train: {}, validation: {}".format(
            st_train.shape[0], st_val.shape[0]
        )
    )
    d_train, d_val = split_data(dynamic_seq_ids, proportions)
    print(
        "Dynamic split is train: {}, validation: {}".format(
            d_train.shape[0], d_val.shape[0]
        )
    )

    train_ids = np.concatenate((st_train, d_train))
    val_ids = np.concatenate((st_val, d_val))

    assert not set(train_ids) & set(val_ids)

    print(
        "Overall split is train: {}, validation: {}, test: {}".format(
            train_ids.shape[0], val_ids.shape[0], len(test_seqs)
        )
    )
    output_path = Path(output_dir) / "split_{}_{}_{}".format(*proportions, only_dynamic)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(output_path / "train.txt"), train_ids, fmt="%s")
    np.savetxt(str(output_path / "val.txt"), val_ids, fmt="%s")


if __name__ == "__main__":
    main()
