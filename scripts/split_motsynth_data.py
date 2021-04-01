""" Splits motsynth dataset into train, validation and test"""
import logging
import sys
from pathlib import Path

import click
import numpy as np
from IPython.core import ultratb

import mots_tracker

sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data_dir",
    "data_dir",
    default="/home/vy/university/thesis/datasets/MOTSynth",
    type=click.Path(exists=True),
    help="Path to MOTSynth dataset",
)
@click.option(
    "--output_dir",
    "output_dir",
    default="/home/vy/university/thesis/datasets/MOTSynth",
    type=click.Path(exists=True),
    help="Output path of the dataset",
)
@click.option(
    "--proportions",
    "proportions",
    type=(float, float, float),
    default=(0.6, 0.2, 0.2),
    help="Proportions to split the dataset to train/val/test (should sum up to 1)",
)
@click.version_option(mots_tracker.__version__)
def main(data_dir, output_dir, proportions):
    print(proportions)
    np.random.seed(42)
    seq_ids = []
    for seq_id in (Path(data_dir) / "frames").glob("*"):
        seq_ids.append(seq_id.parts[-1])
    seq_ids = np.array(seq_ids)
    np.random.shuffle(seq_ids)
    n = seq_ids.shape[0]
    val_start = int((proportions[0]) * n)
    test_start = int((proportions[0] + proportions[1]) * n)
    train_ids, val_ids, test_ids = np.split(seq_ids, [val_start, test_start])
    assert not set(train_ids) & set(val_ids)
    assert not set(train_ids) & set(test_ids)
    assert not set(val_ids) & set(test_ids)

    output_path = Path(output_dir) / "splits"
    output_path.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(output_path / "train.txt"), train_ids, fmt="%s")
    np.savetxt(str(output_path / "val.txt"), val_ids, fmt="%s")
    np.savetxt(str(output_path / "test.txt"), test_ids, fmt="%s")


if __name__ == "__main__":
    main()
