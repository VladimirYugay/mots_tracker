""" Calibrates camera given chessboard images """
import logging
import sys
from pathlib import Path

import click
import cv2
import numpy as np

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input_path",
    "input_path",
    default="/home/vy/university/thesis/adl-chessboard",
    type=click.Path(exists=True),
    help="Path to raw annotations",
)
@click.option(
    "--output_path",
    "output_path",
    default="/home/vy/university/thesis/",
    type=click.Path(exists=False),
    help="Path to resulting gt annotations",
)
@click.option(
    "--square_size",
    "square_size",
    default=25,
    help="Chessboard square size in mm",
)
@click.option(
    "--chessboard_size",
    "chessboard_size",
    default=(6, 8),
    help="Chessboard size in squares",
)
@click.option(
    "--display",
    "display",
    help="Display the calibration patterns",
    flag_value=True,
)
def main(input_path, output_path, chessboard_size, square_size, display):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    input_path = Path(input_path)
    output_path = Path(output_path)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, square_size, 0.001)
    height, width = chessboard_size

    board_pts = np.zeros((height * width, 3), np.float32)
    board_pts[:, :2] = np.mgrid[:height, :width].T.reshape(-1, 2)
    board_pts *= square_size

    world_points, image_points = [], []
    for file_path in input_path.glob("**/*"):
        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(image, chessboard_size, None)
        if found:
            world_points.append(board_pts)
            refined_corners = cv2.cornerSubPix(
                image, corners, (11, 11), (-1, -1), criteria
            )
            image_points.append(refined_corners)
            if display:
                cv2.drawChessboardCorners(
                    image, (width, height), refined_corners, found
                )
                cv2.imshow("img", image)
                cv2.waitKey(500)
    cv2.destroyAllWindows()
    _, intrinsics, _, _, _ = cv2.calibrateCamera(
        world_points, image_points, image.shape[::-1], None, None
    )
    np.set_printoptions(suppress=True, precision=3)
    logging.log(logging.INFO, intrinsics)
    intrinsics = np.around(intrinsics, 3)
    np.savetxt(str(output_path / "intrinsics.txt"), intrinsics, fmt="%s")
    logging.log(logging.INFO, "Saved to {}".format(output_path))


if __name__ == "__main__":
    main()
