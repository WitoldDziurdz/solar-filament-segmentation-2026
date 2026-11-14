"""Shared helpers and the optional real-data fixture.

Most tests build tiny synthetic scenes and need no data. Tests marked with the
``real_gt`` fixture read the competition's training annotations and are
skipped when the file is absent. They look for it at, in order:

1. ``$FILAMENT_ANNOTATIONS``;
2. ``$FILAMENT_DATA_ROOT/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json``;
3. ``data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json``
   under the repository root.

The one test that needs a training image looks in ``train/train_images`` of the
same data root. The tests of the published weights read ``weights/`` and need no
data.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util

REPO = Path(__file__).resolve().parents[1]
ANNOTATION_FILE = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def data_root():
    return Path(os.environ.get("FILAMENT_DATA_ROOT", REPO / "data" / "MAGFiLO_1.0_Kaggle_2026"))


def annotations_path():
    if os.environ.get("FILAMENT_ANNOTATIONS"):
        return Path(os.environ["FILAMENT_ANNOTATIONS"])
    return data_root() / "train" / ANNOTATION_FILE


def train_images_dir():
    """``train/train_images`` of the competition data; tests that need it skip without it."""
    return data_root() / "train" / "train_images"


def rect_rle(r0, c0, r1, c1, size):
    """Compressed-RLE counts string for the filled rectangle [r0:r1, c0:c1)."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[r0:r1, c0:c1] = 1
    return mask_util.encode(np.asfortranarray(mask))["counts"].decode("ascii")


def frame(pairs):
    """Submission-shaped frame from ``[(filament_id, rle), ...]``."""
    return pd.DataFrame(
        {"filament_id": [p[0] for p in pairs], "segmentation_rle": [p[1] for p in pairs]},
        columns=["filament_id", "segmentation_rle"],
    )


@pytest.fixture(scope="session")
def real_gt():
    """The full training ground truth as a frame, or skip."""
    path = annotations_path()
    if not path.is_file():
        pytest.skip(f"training annotations not found ({ANNOTATION_FILE})")
    from filament.io import build_gt_frame, load_coco

    return build_gt_frame(load_coco(path))
