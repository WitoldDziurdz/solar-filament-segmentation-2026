"""Stage 1 of the model: YOLO11l-seg proposes filament instances.

The detector is Ultralytics YOLO11l-seg with a single class, fine-tuned from
the COCO-pretrained ``yolo11l-seg.pt`` at an input size of 1792 px.

**Inference** (:func:`propose`), exactly as the published predictions were made:

* the 2048 x 2048 grayscale frame is replicated to three channels
  (:func:`detector_input`). The competition JPEGs are single-channel, so this is
  byte-identical to what Ultralytics reads from the file itself;
* Ultralytics letterboxes it to 1792 x 1792 (a plain bilinear resize, because
  the frame is square), scales it to [0, 1] and runs the network in float32;
* proposals are kept down to a confidence of 0.05, with non-maximum suppression
  at IoU 0.5 and at most 60 per frame;
* masks come back at native resolution (``retina_masks=True``): the prototype
  mask logits are upsampled to 2048 px *before* they are thresholded at
  probability 0.5. Thresholding first and resizing the binary mask would shift
  every boundary, and one pixel of boundary costs about a tenth of the score;
* only proposals with confidence >= 0.3 go on to the refiner
  (:attr:`DetectorConfig.confidence`).

The 0.3 / 0.5 operating point was chosen on the tune half of the validation
fold from confidences 0.10-0.80 and NMS IoU 0.0 / 0.5 / 0.7. The 0.05 floor
is lower because one inference pass served that whole sweep; proposals between
the floor and 0.3 never reach the output.

**Training** (``training/train_detector.py`` repeats it on a GPU; the notebook
loads the published weights). Ultralytics 8.4.126, 565 training images (folds 1-4), batch 2, seed
20260816, a 6-hour budget on one GPU (78 epochs), no validation during
training, single class. Augmentation: horizontal and vertical flips (p = 0.5),
rotation up to 180 degrees, mosaic off (every frame is one centred solar disc,
and a mosaic of four partial discs is not an image the test set contains), the
Ultralytics defaults otherwise; ``overlap_mask=False`` so overlapping
filaments keep their own supervision. Targets: per image, the union of all
annotators' filaments, deduplicated by complete-link clustering at IoU > 0.5
with at most one outline per annotator in a cluster; each cluster is
represented by its member with the highest mean IoU to the others, and an
outline drawn as several rings keeps the ring with the most vertices (6,435
annotated outlines became 5,143 targets).

Ultralytics is AGPL-3.0 licensed.
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from ..io import IMAGE_SIZE, mask_to_rle

ULTRALYTICS_VERSION = "8.4.126"


@dataclass(frozen=True)
class DetectorConfig:
    """Every inference setting of the detector.

    ``imgsz``
        Network input side; Ultralytics resizes the 2048 px frame to it.
    ``conf_floor`` / ``nms_iou`` / ``max_det``
        Passed to Ultralytics' ``predict`` as ``conf``, ``iou`` and ``max_det``.
    ``mask_threshold``
        Probability above which a native-resolution mask pixel is foreground.
    ``confidence``
        Operating point: proposals below it are discarded before refinement.
    """

    imgsz: int = 1792
    conf_floor: float = 0.05
    nms_iou: float = 0.5
    max_det: int = 60
    mask_threshold: float = 0.5
    confidence: float = 0.3


DETECTOR = DetectorConfig()


def load_detector(path: str | Path):
    """Build the Ultralytics model from a checkpoint; returns ``ultralytics.YOLO``.

    The device is chosen per call of :func:`propose`. Ultralytics checkpoints
    are pickles, so verify the file's checksum first
    (:func:`filament.pipeline.verify_manifest`). ``YOLO_OFFLINE`` is set unless
    the caller set it: inference from a local file needs no network, and it
    keeps Ultralytics from checking for updates or sending usage statistics.
    """
    os.environ.setdefault("YOLO_OFFLINE", "1")
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        warnings.warn(
            f"ultralytics {ultralytics.__version__} is installed; the published predictions "
            f"were made with {ULTRALYTICS_VERSION} (requirements.txt). Results may differ.",
            RuntimeWarning, stacklevel=2,
        )
    model = YOLO(str(Path(path)))
    if model.task != "segment" or len(model.names) != 1:
        raise ValueError(f"{Path(path).name} is not a one-class segmentation model "
                         f"(task {model.task!r}, {len(model.names)} classes)")
    return model


def detector_input(image: np.ndarray) -> np.ndarray:
    """``(2048, 2048)`` uint8 frame -> ``(2048, 2048, 3)`` uint8, the channel replicated."""
    if image.dtype != np.uint8 or image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected a ({IMAGE_SIZE}, {IMAGE_SIZE}) uint8 frame, "
                         f"got {image.shape} {image.dtype}")
    return np.repeat(image[:, :, None], 3, axis=2)


def masks_to_proposals(
    masks: np.ndarray,
    scores: np.ndarray,
    config: DetectorConfig = DETECTOR,
) -> List[Tuple[str, float]]:
    """Native-resolution model masks -> ``[(rle, score), ...]``, empty masks dropped."""
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"masks must be native ({IMAGE_SIZE}, {IMAGE_SIZE}), got {masks.shape}; "
                         "predict with retina_masks=True")
    pairs = []
    for mask, score in zip(masks, np.asarray(scores)):
        binary = mask > config.mask_threshold
        if binary.any():
            pairs.append((mask_to_rle(binary), float(score)))
    return pairs


def propose(model, image: np.ndarray, config: DetectorConfig = DETECTOR,
            device: str | torch.device = "cpu") -> List[Tuple[str, float]]:
    """Every proposal down to ``config.conf_floor``, best first, as ``(rle, score)``.

    Apply ``config.confidence`` afterwards (:func:`filament.pipeline.predict_instances`).
    """
    device = torch.device(device)
    result = model.predict(
        source=detector_input(image), imgsz=config.imgsz, conf=config.conf_floor,
        iou=config.nms_iou, max_det=config.max_det, retina_masks=True,
        device=0 if device.type == "cuda" else "cpu", verbose=False,
    )[0]
    if result.masks is None:
        return []
    return masks_to_proposals(result.masks.data.cpu().numpy(),
                              result.boxes.conf.cpu().numpy(), config)
