""" Utils for input and output of tracker data """


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
