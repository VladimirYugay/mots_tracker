""" Script for comparing backward consistency """
import json
import os

import numpy as np

from mots_tracker.legacy.old_reader import OldMOTSReader
from mots_tracker.readers import MOTSReader

if __name__ == "__main__":
    reader_cfg_path = "./configs/reader_configs/mots_reader_config.json"
    mots_path = "/home/vy/university/thesis/datasets/MOTS/"
    phase = "train"
    with open(str(reader_cfg_path), "r") as reader_config_file:
        reader_args = json.load(reader_config_file)
    old_reader = OldMOTSReader(os.path.join(mots_path, phase), reader_args)
    new_reader = MOTSReader(os.path.join(mots_path, phase), reader_args)
    assert old_reader.sequence_info.keys() == new_reader.sequence_info.keys()

    for seq_id in old_reader.sequence_info.keys():
        for frame in range(old_reader.sequence_info[seq_id]["length"]):
            old_sample = old_reader.read_sample(seq_id, frame)
            new_sample = new_reader.read_sample(seq_id, frame)
            print("Precessing seq {}, frame {}".format(seq_id, frame))
            assert np.all(new_sample["mask_ids"] == old_sample["mask_ids"])
            assert np.all(new_sample["masks"] == old_sample["masks"])
            assert np.all(new_sample["boxes"] == old_sample["boxes"])
