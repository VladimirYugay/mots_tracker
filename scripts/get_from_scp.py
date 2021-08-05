""" Calibrates camera given chessboard images """
import logging
import sys
from pathlib import Path

import click
import numpy as np
from paramiko import SSHClient
from scp import SCPClient

_logger = logging.getLogger(__name__)


@click.command()
def main():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(
        hostname="atcremers79.vision.in.tum.de",
        port=58022,
        username="yugay",
        password="Xoh?c`a3",
    )

    scp = SCPClient(ssh.get_transport())
    output_path = Path("/home/vy/university/thesis/datasets/test")
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path("/storage/local/yugay/datasets/MOTSynth/frames")

    seqs = [
        "165",
        "028",
        "156",
        "644",
        "736",
        "542",
        "232",
        "241",
        "113",
        "248",
        "492",
        "230",
        "137",
        "204",
        "102",
        "670",
        "238",
        "104",
        "218",
        "337",
    ]
    img_names = np.arange(1800)
    for seq in seqs:
        (output_path / seq).mkdir(parents=True, exist_ok=True)
        for img_name in img_names:
            if img_name % 100 == 0:
                img_name = "{0:04d}".format(img_name) + ".jpg"
                print("Processing seq: {}, img: {}".format(seq, img_name))
                scp.get(
                    str(input_path / seq / img_name),
                    str(output_path / seq / img_name),
                    recursive=True,
                )
    scp.close()


if __name__ == "__main__":
    main()
