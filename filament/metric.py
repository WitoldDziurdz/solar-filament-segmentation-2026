"""Panoptic Quality metric for the Solar Filament Segmentation Challenge 2026.

The competition switched its primary metric from Dice to Panoptic Quality on
7 August 2026:

    PQ = sum(IoU over TP pairs) / (|TP| + 0.5 * |FP| + 0.5 * |FN|)

where a (ground-truth, prediction) pair is a true positive only when its IoU is
strictly greater than 0.5. Equivalently PQ = SQ * RQ, with SQ the mean IoU over
matched pairs and RQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|).

Two implementations live here on purpose:

``pq_reference``
    A line-by-line port of the organizer's ``self-evaluation-notebook``. It
    decodes every mask to a dense (n, H, W) tensor. Slow and memory hungry, but
    it is the definition we are held to.

``pq_score``
    The fast path used everywhere else. It computes the same IoU matrix through
    ``pycocotools.mask.iou``, which works directly on run-length encodings and
    never materialises a dense array. ``tests/test_metric.py`` asserts the two
    agree.

Identifier conventions, which the matching depends on:

    ground truth   "<annotator_id>-<image_stem>_<index>"
    prediction     "<image_stem>_<index>"

Ground truth is grouped per *annotator-image*, so an image annotated by three
experts is scored three times against the same set of predictions. Predictions
for images absent from the ground truth are silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

IMAGE_SIZE = 2048
IOU_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# RLE helpers
# --------------------------------------------------------------------------

def to_rle_dict(counts: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> dict:
    """Wrap an ASCII compressed-RLE counts string in the dict pycocotools wants."""
    if isinstance(counts, bytes):
        payload = counts
    else:
        payload = counts.encode("ascii")
    return {"size": [height, width], "counts": payload}


def rles_to_layers(
    rles: Sequence[str],
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> np.ndarray:
    """Decode compressed RLE strings into an ``(n, H, W)`` float32 binary stack.

    This is the organizer's helper, kept for the reference implementation.
    """
    if len(rles) == 0:
        return np.zeros((0, height, width), dtype=np.float32)
    masks = mask_util.decode([to_rle_dict(r, height, width) for r in rles])
    return masks.transpose(2, 0, 1).astype(np.float32)


def rle_areas(rles: Sequence[str], height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> np.ndarray:
    """Foreground pixel count per RLE, without decoding to dense masks."""
    if len(rles) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(
        [int(mask_util.area(to_rle_dict(r, height, width))) for r in rles],
        dtype=np.int64,
    )


# --------------------------------------------------------------------------
# Overlap matrices
# --------------------------------------------------------------------------

def iou_matrix_from_rles(
    gt_rles: Sequence[str],
    pred_rles: Sequence[str],
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> np.ndarray:
    """IoU matrix of shape ``(n_gt, n_pred)`` computed directly on RLEs.

    ``mask_util.iou`` returns ``(len(dt), len(gt))``, so predictions are passed
    as the detections and the result is transposed. Empty inputs yield a
    correctly shaped empty array rather than an error.
    """
    n_gt, n_pred = len(gt_rles), len(pred_rles)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float64)

    gt_objs = [to_rle_dict(r, height, width) for r in gt_rles]
    pred_objs = [to_rle_dict(r, height, width) for r in pred_rles]

    # iscrowd=0 everywhere: we want plain IoU, not the crowd variant that
    # normalises by the detection area only.
    ious = mask_util.iou(pred_objs, gt_objs, [0] * n_gt)
    return np.asarray(ious, dtype=np.float64).reshape(n_pred, n_gt).T


def dice_matrix_from_iou(
    iou: np.ndarray,
    gt_areas: np.ndarray,
    pred_areas: np.ndarray,
) -> np.ndarray:
    """Recover Dice from IoU: ``dice = 2 * iou / (1 + iou)``.

    Both express the same intersection over the same two areas, so the identity
    holds exactly wherever the union is non-zero.
    """
    del gt_areas, pred_areas  # kept for signature symmetry; identity needs neither
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2.0 * iou / (1.0 + iou)
    return np.nan_to_num(dice, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------

def _prefix_before_underscore(filament_ids: pd.Series) -> pd.Series:
    """Take everything before the first underscore, tolerating an empty frame.

    An empty column arrives as float64, and the ``.str`` accessor rejects it, so
    the empty case is short-circuited. A submission with no rows is legal input
    that must score zero rather than raise.
    """
    if len(filament_ids) == 0:
        return pd.Series([], dtype="object")
    return filament_ids.astype(str).str.split("_", n=1).str[0]


def gt_group_keys(filament_ids: pd.Series) -> pd.Series:
    """Ground-truth ``"<annotator>-<stem>_<idx>"`` -> ``"<annotator>-<stem>"``."""
    return _prefix_before_underscore(filament_ids)


def pred_group_keys(filament_ids: pd.Series) -> pd.Series:
    """Prediction ``"<stem>_<idx>"`` -> ``"<stem>"``."""
    return _prefix_before_underscore(filament_ids)


def image_stem_of(annotator_image: str) -> str:
    """``"040301-20140609195854Bh"`` -> ``"20140609195854Bh"``."""
    return annotator_image.split("-", maxsplit=1)[1]


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PQResult:
    """Panoptic Quality and the pieces it decomposes into."""

    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    n_gt: int
    n_pred: int
    n_groups: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"PQ={self.pq:.4f}  SQ={self.sq:.4f}  RQ={self.rq:.4f}  "
            f"TP={self.tp} FP={self.fp} FN={self.fn}  "
            f"(gt={self.n_gt} pred={self.n_pred} over {self.n_groups} annotator-images)"
        )


def _accumulate(iou: np.ndarray, n_gt: int, n_pred: int) -> Tuple[List[float], int, int]:
    """TP IoUs, FP count and FN count for a single annotator-image."""
    if n_gt == 0:
        return [], n_pred, 0
    if n_pred == 0:
        return [], 0, n_gt

    hit = iou > IOU_THRESHOLD
    # IoU > 0.5 can hold for at most one pair per row and per column, so this
    # is already a one-to-one matching; no Hungarian assignment is needed.
    tp_ious = iou[hit].tolist()
    fp = int((hit.sum(axis=0) == 0).sum())
    fn = int((hit.sum(axis=1) == 0).sum())
    return tp_ious, fp, fn


def pq_score(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> PQResult:
    """Panoptic Quality over every annotator-image present in ``gt_df``.

    Both frames need columns ``filament_id`` and ``segmentation_rle``. Rows in
    ``pred_df`` whose image stem never appears in ``gt_df`` are ignored, which
    mirrors the organizer's implementation.
    """
    gt_keys = gt_group_keys(gt_df["filament_id"])
    pred_keys = pred_group_keys(pred_df["filament_id"])

    # Bucket once instead of filtering the frame per group: the naive version is
    # quadratic and dominates runtime on the full training set.
    gt_by_group: Dict[str, List[str]] = {}
    for key, rle in zip(gt_keys, gt_df["segmentation_rle"]):
        gt_by_group.setdefault(key, []).append(rle)

    pred_by_stem: Dict[str, List[str]] = {}
    for key, rle in zip(pred_keys, pred_df["segmentation_rle"]):
        pred_by_stem.setdefault(key, []).append(rle)

    tp_ious: List[float] = []
    fp_total = 0
    fn_total = 0
    n_pred_total = 0

    for annotator_image, gt_rles in gt_by_group.items():
        pred_rles = pred_by_stem.get(image_stem_of(annotator_image), [])
        iou = iou_matrix_from_rles(gt_rles, pred_rles, height, width)
        ious, fp, fn = _accumulate(iou, len(gt_rles), len(pred_rles))
        tp_ious.extend(ious)
        fp_total += fp
        fn_total += fn
        n_pred_total += len(pred_rles)

    return _finalise(tp_ious, fp_total, fn_total, len(gt_df), n_pred_total, len(gt_by_group))


def _finalise(
    tp_ious: Sequence[float],
    fp: int,
    fn: int,
    n_gt: int,
    n_pred: int,
    n_groups: int,
) -> PQResult:
    tp = len(tp_ious)
    denominator = tp + 0.5 * fp + 0.5 * fn
    iou_sum = float(sum(tp_ious))

    pq = iou_sum / denominator if denominator > 0 else 0.0
    sq = iou_sum / tp if tp > 0 else 0.0
    rq = tp / denominator if denominator > 0 else 0.0

    return PQResult(
        pq=pq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn,
        n_gt=n_gt, n_pred=n_pred, n_groups=n_groups,
    )


def pq_reference(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> float:
    """Dense port of the organizer's notebook. Correctness oracle, not for routine use.

    The published version uses torch; this uses numpy with the same float32
    dtype and the same operation order. Masks are 0/1 and intersections are
    bounded by 2048*2048 = 4.2M, well inside the 16.7M range where float32
    represents integers exactly, so the two agree bit for bit.
    """
    gt_keys = gt_group_keys(gt_df["filament_id"])
    pred_keys = pred_group_keys(pred_df["filament_id"])

    tp_iou_scores: List[float] = []
    fp_count = 0
    fn_count = 0

    for annotator_image in gt_keys.unique():
        image_id = image_stem_of(annotator_image)

        gt_rles = gt_df.loc[gt_keys == annotator_image, "segmentation_rle"].tolist()
        pred_rles = pred_df.loc[pred_keys == image_id, "segmentation_rle"].tolist()

        gt_layers = rles_to_layers(gt_rles, height, width)
        pred_layers = rles_to_layers(pred_rles, height, width)

        n_gt = len(gt_rles)
        n_pred = len(pred_rles)

        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue

        gt_flat = gt_layers.reshape(n_gt, height * width)
        pred_flat = pred_layers.reshape(n_pred, height * width)
        intersection = gt_flat @ pred_flat.T
        gt_areas = gt_flat.sum(axis=1).reshape(-1, 1)
        pred_areas = pred_flat.sum(axis=1).reshape(1, -1)
        union = gt_areas + pred_areas - intersection
        with np.errstate(invalid="ignore", divide="ignore"):
            iou_matrix = np.where(union == 0, 0.0, intersection / union)

        hit_matrix = iou_matrix > IOU_THRESHOLD
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += int((hit_matrix.sum(axis=0) == 0).sum())
        fn_count += int((hit_matrix.sum(axis=1) == 0).sum())

    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
    return sum(tp_iou_scores) / denominator if denominator > 0 else 0.0


# --------------------------------------------------------------------------
# Decision theory: when is emitting an instance worth it?
# --------------------------------------------------------------------------

def emission_break_even(current_pq: float) -> float:
    """Minimum ``p(hit) * E[IoU | hit]`` that justifies emitting one instance.

    Consider one extra predicted instance aimed at one ground-truth filament,
    against a running numerator ``N`` and denominator ``D`` with ``N/D = PQ``:

        stay silent        ->  N        / (D + 0.5)     one FN
        emit and miss      ->  N        / (D + 1.0)     one FP *and* one FN
        emit and hit       -> (N + IoU) / (D + 1.0)

    Emitting beats silence when ``(N + p*IoU)/(D+1) > N/(D+0.5)``, which for
    large ``D`` reduces to ``p * IoU > 0.5 * PQ``.

    The asymmetry is the point: a wrong instance costs twice what an omitted one
    does. At PQ 0.4 an instance is worth emitting only above roughly 0.2 expected
    IoU -- about a 30% chance of clearing the threshold at a typical matched IoU
    of 0.7. Low-confidence detections are better deleted than kept.
    """
    return 0.5 * current_pq


def expected_pq_delta(
    p_hit: float,
    expected_iou_given_hit: float,
    current_pq: float,
) -> float:
    """Signed margin over the break-even point. Positive means emit."""
    return p_hit * expected_iou_given_hit - emission_break_even(current_pq)
