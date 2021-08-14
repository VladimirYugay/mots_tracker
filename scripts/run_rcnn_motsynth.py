# Script for running inference of the mask and boxes model
# Can't be run inside this repo due to lack of pytorch
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as rletools
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.transforms import ToTensor


class MaskRCNN_FPN(MaskRCNN):
    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone("resnet50", False)
        super(MaskRCNN_FPN, self).__init__(backbone, num_classes)

        # Cache for feature use
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)
        detections = self([img])[0]

        return (
            detections["boxes"].detach(),
            detections["scores"].detach(),
            detections["masks"].detach(),
        )


def plot_image_boxes(image, boxes, boxes_ids=None):
    fig, ax = plt.subplots(1, dpi=96)
    ax.imshow(image)
    for i, box in enumerate(boxes):
        color = np.random.uniform(0, 1, size=3)
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            linewidth=1.0,
            color=color,
        )
        ax.add_patch(rect)
    plt.show()


def encode_mask(mask):
    """Encodes binary mask to rle string
    Args:
        mask (ndarray): binary mask
    Returns:
        encoding (dict): dict with 'size' and 'counts' string inside
    """
    mask = mask.astype(np.uint8)
    encoding = rletools.encode(np.asfortranarray(mask))
    encoding["counts"] = encoding["counts"]
    return encoding


def decode_mask(height, width, mask_string):
    """ decodes coco segmentation mask string to numpy """
    return rletools.decode(
        {
            "size": [int(height), int(width)],
            "counts": mask_string.encode(encoding="UTF-8"),
        }
    )


if __name__ == "__main__":
    model_name = "mask_rcnn"
    model = MaskRCNN_FPN(2)
    model_path = "/home/vy/Downloads/maskrcnn_motsyn4_epoch_5.pth"
    obj_detect_state_dict = torch.load(
        model_path, map_location=lambda storage, loc: storage
    )
    model.load_state_dict(obj_detect_state_dict["model"])
    model.eval()

    frames_path = Path("/home/vy/university/thesis/datasets/MOTSynth/frames")
    output_path = Path("/home/vy/university/thesis/datasets/MOTSynth_annotations/all")

    height, width = 1080, 1920
    transform = ToTensor()
    for seq in sorted(frames_path.glob("*")):
        print("Processing sequence:", seq.parts[-1])
        (output_path / seq.parts[-1] / model_name).mkdir(parents=True, exist_ok=True)
        boxes_file = open(
            str(output_path / seq.parts[-1] / model_name / "boxes.txt"), "w"
        )
        masks_file = open(
            str(output_path / seq.parts[-1] / model_name / "masks.txt"), "w"
        )
        for i, img_path in enumerate(sorted(seq.glob("*"))):
            print("Processing image: {}, from sequence: {}".format(i, seq.parts[-1]))
            img = np.array(Image.open(str(img_path)))
            img = transform(img)
            boxes, scores, masks = model.detect(img)
            for box, mask, score in zip(boxes, masks, scores):
                print(
                    "{},{},{},{},{},{}".format(
                        i + 1, score, box[0], box[1], box[2] - box[0], box[3] - box[1]
                    ),
                    file=boxes_file,
                )
                mask = mask[0, ...].detach().cpu().numpy()
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                mask = encode_mask(mask)
                print(
                    "{} {} -1 {} {} {}".format(
                        i + 1,
                        score,
                        mask["size"][0],
                        mask["size"][1],
                        mask["counts"].decode("UTF-8"),
                    ),
                    file=masks_file,
                )
        boxes_file.close()
        masks_file.close()
