""" Generates depth from ground truth depth videos  """
import logging
import sys
from pathlib import Path

import click
import cv2

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--ip",
    "--input_path",
    "input_path",
    default="/home/vy/university/thesis/adl-chessboard",
    type=click.Path(exists=True),
    help="Path to depth videos",
)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="/home/vy/university/thesis/",
    type=click.Path(exists=True),
    help="Path to resulting depth frames",
)
def main(input_path, output_path):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    input_path = Path(input_path)
    output_path = Path(output_path)
    for video_path in sorted(input_path.glob("*")):
        video_cap = cv2.VideoCapture(str(video_path))
        video_id = int(video_path.parts[-1].split(".")[0])
        save_path = output_path / "{:03d}".format(video_id) / "gt_depth"
        save_path.mkdir(exist_ok=True, parents=True)

        count = 1
        success = True
        print("Unpacking video: {}".format(video_id))
        while success:
            success, image = video_cap.read()
            if count < 4:
                count += 1
                continue
            if not success or count == 1804:
                break
            file_name = str(save_path / (str(count - 4).zfill(4) + ".png"))
            cv2.imwrite(file_name, image)
            count += 1


if __name__ == "__main__":
    main()
