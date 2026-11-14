"""Loading MAGFiLO images and annotations, and reading and writing prediction files.

The dataset has a structure that punishes naive handling: roughly 1154 COCO
image records cover only ~707 *physical* solar observations, because several
experts annotated the same image independently. A record id looks like

    040301-20140609195854Bh
    ^^^^^^ annotator        ^^^^^^^^^^^^^^^^ physical observation stem

Splitting on the record id puts the same Sun image in train and validation under
two different annotators, which inflates local scores and makes them useless for
model selection. Everything here keeps the stem available so that folds can be
grouped on it (see :mod:`filament.folds`).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

IMAGE_SIZE = 2048
IMAGE_SUFFIXES = (".jpeg", ".jpg")

# A file fetched from a Git LFS repository without git-lfs installed is a small
# text pointer that starts with this line instead of the real bytes.
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_coco(path: str | Path) -> dict:
    """Read the competition's COCO annotation file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def physical_stem(record_id: str) -> str:
    """``"040301-20140609195854Bh"`` -> ``"20140609195854Bh"``.

    The stem is the physical observation; it is what folds must be grouped on.
    """
    return record_id.split("-", maxsplit=1)[1]


def annotator_of(record_id: str) -> str:
    """``"040301-20140609195854Bh"`` -> ``"040301"``."""
    return record_id.split("-", maxsplit=1)[0]


def list_images(directory: str | Path) -> List[Path]:
    """Sorted image paths in ``directory``; an absent directory gives an empty list.

    The stem of each path (file name without extension) is the observation id
    used in ``filament_id``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def load_image(path: str | Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """Read one H-alpha frame as a ``(size, size)`` uint8 grayscale array.

    The competition images are 2048 x 2048 single-channel JPEGs. Anything else
    is rejected here, because a resized image would silently shift every mask
    boundary and the metric is very sensitive to that.
    """
    import cv2  # imported here so that the metric and folds need no OpenCV

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    if image.shape != (size, size):
        raise ValueError(f"{path.name} is {image.shape}, expected ({size}, {size})")
    return image


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def polygons_to_rle(
    segmentation: Sequence[Sequence[float]],
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> str:
    """Convert a COCO polygon list into one merged compressed-RLE counts string.

    A filament may be stored as several polygon rings; merging keeps it a single
    instance, which is what the metric matches on.
    """
    rles = mask_util.frPyObjects(list(segmentation), height, width)
    merged = mask_util.merge(rles) if isinstance(rles, list) else rles
    return merged["counts"].decode("ascii")


def mask_to_rle(mask: np.ndarray) -> str:
    """Binary ``(H, W)`` array -> compressed-RLE counts string."""
    encoded = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return encoded["counts"].decode("ascii")


def rle_to_mask(counts: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> np.ndarray:
    """Compressed-RLE counts string -> boolean ``(H, W)`` array."""
    rle = {"size": [height, width], "counts": counts.encode("ascii")}
    return mask_util.decode(rle).astype(bool)


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

def build_gt_frame(
    coco: dict,
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
    stems: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Ground truth in submission shape: ``filament_id``, ``segmentation_rle``.

    ``filament_id`` is ``"<annotator>-<stem>_<index>"``, matching what the
    organizer's evaluator expects on the ground-truth side. The extra columns
    (``record_id``, ``stem``, ``annotator``, ``category_id``, ``area``) are for
    analysis and are ignored by the metric.

    ``stems`` optionally restricts the frame to those physical observations;
    the indices inside a record are the same either way.
    """
    wanted = None if stems is None else set(stems)
    by_record: Dict[str, list] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_record[annotation["image_id"]].append(annotation)

    rows = []
    for record_id, annotations in by_record.items():
        if wanted is not None and physical_stem(record_id) not in wanted:
            continue
        for index, annotation in enumerate(annotations):
            rows.append(
                {
                    "filament_id": f"{record_id}_{index}",
                    "segmentation_rle": polygons_to_rle(
                        annotation["segmentation"], height, width
                    ),
                    "record_id": record_id,
                    "stem": physical_stem(record_id),
                    "annotator": annotator_of(record_id),
                    "category_id": annotation["category_id"],
                    "area": annotation["area"],
                }
            )
    columns = ["filament_id", "segmentation_rle", "record_id", "stem",
               "annotator", "category_id", "area"]
    return pd.DataFrame(rows, columns=columns)


def build_pred_frame(per_image: Dict[str, List[str]]) -> pd.DataFrame:
    """Predictions in submission shape from ``{image_stem: [rle, ...]}``."""
    rows = []
    for stem, rles in per_image.items():
        for index, rle in enumerate(rles):
            rows.append({"filament_id": f"{stem}_{index}", "segmentation_rle": rle})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def records_frame(coco: dict) -> pd.DataFrame:
    """One row per COCO image record, with its stem and annotator broken out."""
    rows = [
        {
            "record_id": image["id"],
            "file_name": image["file_name"],
            "stem": physical_stem(image["id"]),
            "annotator": annotator_of(image["id"]),
            "height": image["height"],
            "width": image["width"],
        }
        for image in coco["images"]
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Prediction files
# --------------------------------------------------------------------------

def prediction_digest(predictions: Dict[str, List[str]]) -> str:
    """SHA-256 of ``{stem: [rle, ...]}`` in canonical JSON (sorted keys, no spaces).

    Stored next to the predictions, it proves later that a file was not edited
    and that two runs produced the same masks.
    """
    payload = json.dumps(predictions, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_prediction_file(path: str | Path) -> tuple[Dict[str, List[str]], Optional[str]]:
    """Read saved predictions as ``({stem: [rle, ...]}, fold_fingerprint)``.

    Three layouts are accepted, all keyed by image stem:

    * ``{"predictions": {stem: [rle, ...]}, ...}`` -- what :func:`save_prediction_file`
      writes, optionally with ``fold_fingerprint`` and ``prediction_sha256``;
    * ``{"predictions": {stem: [{"rle": ..., "score": ...}, ...]}}``;
    * a bare ``{stem: [rle, ...]}``.

    When the file carries ``prediction_sha256`` it is recomputed and a mismatch
    raises ``ValueError``. ``fold_fingerprint`` is ``None`` when the file has none.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object keyed by stem")
    body = payload.get("predictions", payload)
    if not isinstance(body, dict):
        raise ValueError(f"{path}: predictions must be keyed by stem")

    predictions: Dict[str, List[str]] = {}
    for stem, items in body.items():
        if not isinstance(items, list):
            raise ValueError(f"{path}: predictions for {stem} must be a list")
        predictions[stem] = [item["rle"] if isinstance(item, dict) else item for item in items]

    recorded = payload.get("prediction_sha256") if "predictions" in payload else None
    if recorded is not None and prediction_digest(predictions) != recorded:
        raise ValueError(f"{path}: predictions do not match their recorded sha256")
    fingerprint = payload.get("fold_fingerprint") if "predictions" in payload else None
    return predictions, fingerprint


def save_prediction_file(
    path: str | Path,
    predictions: Dict[str, List[str]],
    fold_fingerprint: Optional[str] = None,
    **metadata,
) -> str:
    """Write ``{stem: [rle, ...]}`` with its digest; returns the digest.

    The layout is the first one :func:`load_prediction_file` accepts. Extra
    keyword arguments (model name, thresholds, ...) are stored alongside.
    """
    digest = prediction_digest(predictions)
    payload = dict(metadata)
    payload.update(
        {"fold_fingerprint": fold_fingerprint, "predictions": predictions,
         "prediction_sha256": digest}
    )
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return digest


# --------------------------------------------------------------------------
# Weight files
# --------------------------------------------------------------------------

def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(path: str | Path, expected_sha256: str) -> Path:
    """Fail loudly unless ``path`` holds exactly the expected weight file.

    Three failures are caught before a model is built from the file: it is
    missing, it is a Git LFS pointer (the repository was cloned without
    git-lfs, so only a 130-byte text stub arrived), or its bytes differ from the
    published checksum.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"weight file missing: {path}")
    with open(path, "rb") as handle:
        head = handle.read(len(LFS_POINTER_PREFIX))
    if head == LFS_POINTER_PREFIX:
        raise RuntimeError(
            f"{path.name} is a Git LFS pointer, not the weights. Install git-lfs and "
            "run `git lfs pull` in the repository."
        )
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{path.name}: sha256 {actual[:12]}... != expected {expected_sha256[:12]}...")
    return path
