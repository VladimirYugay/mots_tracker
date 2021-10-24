import numpy as np

from mots_tracker import readers
from mots_tracker.io_utils import get_instance, load_yaml


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def run_eval(config_path):
    config = load_yaml(config_path)
    reader_pred = get_instance(readers, "reader", config)
    seq_ids = sorted(reader_pred.sequence_info.keys())
    config["reader"]["args"]["depth_path"] = "gt_depth_new"
    reader_gt = get_instance(readers, "reader", config)
    min_depth, max_depth = 0, 100
    errors = []
    for seq_id in seq_ids:
        for frame in range(reader_pred.sequence_info[seq_id]["length"]):
            gt_depth = reader_gt.read_sample(seq_id, frame)["depth"]
            gt_depth[gt_depth > max_depth] = max_depth

            pred_depth = reader_pred.read_sample(seq_id, frame)["depth"]
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth

            errors.append(compute_errors(gt_depth, pred_depth))
    mean_errors = np.array(errors).mean(0)
    print(
        "\n  "
        + ("{:>8} | " * 7).format(
            "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
        )
    )
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    config_path = "./configs/median_tracker_config.yaml"
    run_eval(config_path)
