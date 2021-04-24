""" Utils for input and output of tracker data """
import os

from mots_tracker import utils


def print_mot_format(frame_id, obj_id, box, file):
    """Prints results in mot format to the file
    Args:
        frame_id (int): frame id, 1 indexed for MOT
        obj_id (int): id of the object
        box (ndarray): bounding box in (x1, y1, x2, y2) format
        file (File): opened file to write to  (do not close)
    """
    print(
        "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
        % (
            frame_id,
            obj_id,
            box[0],
            box[1],
            box[2] - box[0],
            box[3] - box[1],
        ),
        file=file,
    )


def print_kitti_format(frame_id, obj_id, obj_type, box, file):
    """Prints results in mot format to the file
    Args:
        frame_id (int): frame id, 0 indexed for KITTI
        obj_id (int): id of the object
        obj_type (str): type of the object
        box (ndarray): bounding box in (x1, y1, x2, y2) format
        file (File): opened file to write to  (do not close)
    """
    print(
        "{} {} {} -1 -1 -1 {} {} {} {} -1 -1 -1 -1 -1 -1 -1 -1".format(
            frame_id - 1, obj_id, obj_type, box[0], box[1], box[2], box[3]
        ),
        file=file,
    )


def multi_run_wrapper(args):
    """ Unpacks argument for running on multiple cores """
    return track_objects(*args)


def track_objects(reader, seq_id, output_path, mot_tracker, reader_config):
    """Function to run trackers on multiple cores
    Args:
        reader (Reader): one of the readers implemented in readers module
        seq_id (str): id of the sequence
        output_path (Path): path to save output
        mot_tracker (Tracker): one of the trackers implemented in trackers module
        reader_config (dict): configuration of the reader
    """
    out_file = open(os.path.join(output_path, "{}.txt".format(seq_id)), "w")
    print("Processing %s." % seq_id)
    for frame in range(reader.sequence_info[seq_id]["length"]):
        print("Processing seq: {}, frame: {}".format(seq_id, frame))
        sample = reader.read_sample(seq_id, frame)
        frame += 1
        trackers = mot_tracker.update(sample, sample["intrinsics"])
        for (pred_box, mask, box, idx) in trackers:
            if "resize_shape" in reader_config and reader_config["resize_shape"]:
                width = reader.sequence_info[seq_id]["img_width"]
                height = reader.sequence_info[seq_id]["img_height"]
                box = utils.resize_boxes(
                    box[None, :], reader_config["resize_shape"], (width, height)
                )[0]
            print_mot_format(frame, idx, box, out_file)
    out_file.close()
