""" Module for the methods that match MOTS masks and MOT bounding boxes """
import numpy as np
from pycocotools import mask as rletools
from scipy.optimize import linear_sum_assignment
from termcolor import colored

import mots_tracker.readers
from mots_tracker.utils import decode_mask


def check_original_data(seg_path, bb_path, vis_threshold=0.0, verbose=False):
    """Make sure that there are always more or equal bounding boxes than masks
    in every frame in MOTS20 dataset. Should be applied before mask to bb matching"""
    seg_data = np.loadtxt(str(seg_path), dtype=np.str)
    bb_data = np.loadtxt(str(bb_path), delimiter=",")
    n_bb_frames, n_seg_frames = (
        bb_data[:, 0].astype(np.int).max(),
        seg_data[:, 0].astype(np.int).max(),
    )
    if verbose:
        print(
            "BB frames num: {} vs. {} Masks frames num".format(
                n_bb_frames, n_seg_frames
            )
        )
    original_mapping = {}  # original number of boxes to number of masks
    for frame_id in range(1, n_bb_frames + 1):  # frames start from 1 in the gt file
        frame_data = bb_data[
            (bb_data[:, 0] == frame_id)
            & (bb_data[:, 8] >= vis_threshold)
            & (bb_data[:, 6] == 1)
            & (bb_data[:, 7] == 1)
        ]
        num_boxes = frame_data.shape[0]
        mask_ids = seg_data[seg_data[:, 0] == str(frame_id)][:, 1].astype(np.int) % 1000
        num_masks = np.sum(mask_ids != 0)
        original_mapping[frame_id] = [num_boxes, num_masks]
    if verbose:
        for seq_id, (num_boxes, num_masks) in original_mapping.items():
            msg = "Frame: {}, number of boxes: {}, number of masks: {}".format(
                seq_id, num_boxes, num_masks
            )
            if num_boxes < num_masks:
                msg = colored(msg, "blue")
            print(msg)
    return original_mapping


def check_mapped_data(seg_path, bb_path, vis_threshold=0.0, verbose=False):
    """Make sure that there are always more or equal bounding boxes than masks
    in every frame in MOTS20 dataset. Should be applied before mask to bb matching"""
    seg_data = np.loadtxt(str(seg_path), dtype=np.str)
    bb_data = np.loadtxt(str(bb_path), delimiter=",")
    n_bb_frames, n_seg_frames = (
        bb_data[:, 0].astype(np.int).max(),
        seg_data[:, 0].astype(np.int).max(),
    )
    if verbose:
        print(
            "BB frames num: {} vs. {} Masks frames num".format(
                n_bb_frames, n_seg_frames
            )
        )
    for frame_id in range(1, n_bb_frames + 1):  # frames start from 1 in the gt file
        frame_data = bb_data[
            (bb_data[:, 0] == frame_id)
            & (bb_data[:, 8] >= vis_threshold)
            & (bb_data[:, 6] == 1)
            & (bb_data[:, 7] == 1)
        ]
        num_boxes = frame_data.shape[0]
        mask_ids = seg_data[seg_data[:, 0] == str(frame_id)][:, 1].astype(np.int) % 1000
        num_masks = np.sum(mask_ids != 0)
        if verbose:
            msg = "Frame: {}, number of boxes: {}, number of masks: {}".format(
                frame_id, num_boxes, num_masks
            )
            if num_boxes < num_masks:
                msg = colored(msg, "blue")
            print(msg)
        assert num_masks == num_boxes


def iou_mask2box(box, mask):
    """Computes iou for a binary mask and a box
    Args:
        box (ndarray): array of top left, bottom right corner format
        mask (ndarray): boolean array of shape (h, w)
    Returns:
        (float): intersection over union
    """
    width, height = box[2] - box[0], box[3] - box[1]
    intersection = np.sum(
        mask[box[1] : box[1] + height, box[0] : box[0] + width]
    )  # use the fact that mask is binary
    union = width * height + np.sum(mask) - intersection
    return intersection / union


def intersection_mask2box(box, mask):
    width, height = box[2] - box[0], box[3] - box[1]
    return np.sum(mask[box[1] : box[1] + height, box[0] : box[0] + width])


def match_seg2bb(
    mask_path,
    bb_path,
    mask_save_path,
    bb_save_path,
    img_size,
    vis_threshold=0.0,
    verbose=False,
):
    """Matches every segmentation mask to a bounding box
    The main idea is to keep only those boxes which have
    a corresponding segmentation mask
    Moreover, change box id to maks id for the sake of visualization
    """
    mask_data = np.loadtxt(str(mask_path), dtype=np.str)
    bb_data = np.loadtxt(str(bb_path), delimiter=",")
    n_frames = mask_data[:, 0].astype(np.int).max()  # we assume that data was verified
    bb_file, mask_file = open(bb_save_path, "w"), open(mask_save_path, "w")
    for frame_id in range(1, n_frames + 1):
        if verbose:
            print("Processing frame: {}".format(frame_id))
        frame_bb_data = bb_data[
            (bb_data[:, 0] == frame_id)
            & (bb_data[:, 8] >= vis_threshold)
            & (bb_data[:, 6] == 1)
            & (bb_data[:, 7] == 1)
        ]
        frame_boxes = frame_bb_data[:, [2, 3, 4, 5]]
        frame_boxes[:, [0, 1]] -= 1
        frame_boxes[:, 2] = frame_boxes[:, 0] + frame_boxes[:, 2] - 1
        frame_boxes[:, 3] = frame_boxes[:, 1] + frame_boxes[:, 3] - 1
        # clip boxes outside of the image
        img_width, img_height = img_size
        frame_boxes[:, [0, 2]] = np.clip(frame_boxes[:, [0, 2]], 0, img_width - 1)
        frame_boxes[:, [1, 3]] = np.clip(frame_boxes[:, [1, 3]], 0, img_height - 1)
        frame_boxes = frame_boxes.astype(np.int)
        box_ids = frame_bb_data[:, 1].astype(np.int)  # without id modulo

        frame_mask_data = mask_data[mask_data[:, 0] == str(frame_id)]
        mask_ids = frame_mask_data[:, 1].astype(np.int)

        iou_matrix = np.zeros((frame_boxes.shape[0], mask_ids.shape[0]))
        for i in range(box_ids.shape[0]):
            for j in range(mask_ids.shape[0]):
                if mask_ids[j] % 1000 == 0:  # ignore background masks
                    continue
                mask = decode_mask(
                    frame_mask_data[j][3], frame_mask_data[j][4], frame_mask_data[j][5]
                )
                iou_matrix[i][j] = intersection_mask2box(frame_boxes[i], mask)
        rows, cols = linear_sum_assignment(-iou_matrix)
        bb_matches, mask_matches = {}, set()
        for row_id, col_id in zip(rows, cols):
            if iou_matrix[row_id][col_id] > 0:
                bb_matches[box_ids[row_id]] = mask_ids[col_id]
                mask_matches.add(mask_ids[col_id])
        # keep only matched boxes and replace their ids with the matching mask id
        for row in frame_bb_data:
            if int(row[1]) in bb_matches:
                print(
                    "{},{},{},{},{},{},{},{},{}".format(
                        int(row[0]),
                        bb_matches[row[1]],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                    ),
                    file=bb_file,
                )
        # keep only masks which were matched with boxes
        for row in frame_mask_data:
            if int(row[1]) in mask_matches:
                print(
                    "{} {} {} {} {} {}".format(
                        row[0], row[1], row[2], row[3], row[4], row[5]
                    ),
                    file=mask_file,
                )
    bb_file.close()
    mask_file.close()


def generate_mask2bb(reader: mots_tracker.readers.MOTSReader) -> None:
    """Generates bounding boxes from masks and store corresponding .txt files
    Args:
        reader (MOTSReader): reader for handling MOTS dataset
    """
    for seq_id in reader.config["seq_ids"]:
        print("Processing sequence {}".format(seq_id))
        width = reader.sequence_info[seq_id]["img_width"]
        height = reader.sequence_info[seq_id]["img_height"]
        file = open(
            str(reader.data_path / reader.mode / seq_id / "gt" / "gt_bb.txt"), "w"
        )
        for frame_id in range(reader.sequence_info[seq_id]["length"]):
            sample = reader.read_sample(seq_id, frame_id)
            for mask_id, raw_mask in zip(sample["mask_ids"], sample["raw_masks"]):
                box = rletools.toBbox(
                    {
                        "size": [int(height), int(width)],
                        "counts": raw_mask.encode(encoding="UTF-8"),
                    }
                )
                print(
                    "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
                    % (frame_id + 1, mask_id, box[0], box[1], box[2], box[3]),
                    file=file,
                )
        file.close()
