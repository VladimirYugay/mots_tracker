""" Prepares gt files for evaluation with TrackEval on MOTS
Generates two folders:
bench_name-train
    seq_id
        gt
            gt.txt
        seqinfo.ini
bench_name-test
    seq_id
        gt
            gt.txt
        seqinfo.ini
seqmaps
    bench_name-train.txt
    bench_name-test.txt
    bench_name-all.txt
"""
import logging
import sys
from distutils.dir_util import copy_tree
from pathlib import Path
from shutil import copyfile

import click
from IPython.core import ultratb

import mots_tracker

sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--ap",
    "--ann_path",
    "ann_path",
    default="",
    type=click.Path(exists=True),
    help="Path to MOTSynth preprocessed annotations",
)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="",
    type=click.Path(exists=True),
    help="Output path for the TrackEval compatible gt output",
)
@click.option(
    "--sn",
    "--split_name",
    "split_name",
    default="split_0.8_0_0.2",
    help="Output path of the for the split files",
)
@click.option(
    "--bn",
    "--bench_name",
    "bench_name",
    default="MOTS",
    help="Name of the benchmark",
)
@click.version_option(mots_tracker.__version__)
def main(ann_path, output_path, split_name, bench_name):
    # generate seqmaps files
    ann_path = Path(ann_path)
    output_path = Path(output_path)
    split_path = ann_path / split_name
    train_file = open(str(split_path / "train.txt"), "r")
    test_file = open(str(split_path / "test.txt"), "r")
    train_seqs = sorted([line.strip() for line in train_file.readlines()])
    test_seqs = sorted([line.strip() for line in test_file.readlines()])
    train_file.close()
    test_file.close()

    (output_path / "seqmaps").mkdir(exist_ok=True, parents=True)
    with open(
        str(output_path / "seqmaps" / "{}-train.txt".format(bench_name)), "w"
    ) as file:
        print("name", file=file)
        for seq in train_seqs:
            print(seq, file=file)
    with open(
        str(output_path / "seqmaps" / "{}-test.txt".format(bench_name)), "w"
    ) as file:
        print("name", file=file)
        for seq in test_seqs:
            print(seq, file=file)
    with open(
        str(output_path / "seqmaps" / "{}-all.txt".format(bench_name)), "w"
    ) as file:
        print("name", file=file)
        for line in sorted(train_seqs + test_seqs):
            print(line, file=file)

    # generate gt files for evaluation
    output_path_train = output_path / "{}-train".format(bench_name)
    output_path_train.mkdir(exist_ok=True, parents=True)
    for seq in train_seqs:
        (output_path_train / seq).mkdir(exist_ok=True, parents=True)
        copy_tree(
            str(ann_path / "train" / seq / "gt"), str(output_path_train / seq / "gt")
        )
        copyfile(
            str(ann_path / "train" / seq / "seqinfo.ini"),
            str(output_path_train / seq / "seqinfo.ini"),
        )

    output_path_test = output_path / "{}-test".format(bench_name)
    output_path_test.mkdir(exist_ok=True, parents=True)
    for seq in test_seqs:
        (output_path_test / seq).mkdir(exist_ok=True, parents=True)
        copy_tree(
            str(ann_path / "test" / seq / "gt"), str(output_path_test / seq / "gt")
        )
        copyfile(
            str(ann_path / "test" / seq / "seqinfo.ini"),
            str(output_path_test / seq / "seqinfo.ini"),
        )


if __name__ == "__main__":
    main()
