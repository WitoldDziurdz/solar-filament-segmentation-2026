"""Pin our PQ to the organiser's published implementation.

Every threshold this project chooses is chosen against ``filament.metric``. If
it disagrees with the evaluator even slightly, every sweep optimises the wrong
thing and the disagreement shows up only as an unexplained gap between the
held-out number and the leaderboard.

The organiser published ``self-evaluation-notebook``. ``official_pq`` below is
a transcription of its ``get_pq_score``, ``fp_count_hit`` and ``fn_count_hit``,
kept deliberately literal rather than tidied, so the comparison is against
their code and not against our reading of it.

The subtle part, and the reason this is worth a test: the matching is *not*
bijective across annotators. ``tp_iou_scores.extend(iou_matrix[hit_matrix])``
takes every GT/pred pair above 0.5, so one prediction that clears the
threshold against three annotators' records is three true positives, and a
prediction is a false positive only if it clears nothing in that record. A
Hungarian-style one-to-one matching over all annotators -- what any
reimplementation from the name alone would produce -- gives different numbers.
"""

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util

from filament.metric import pq_score

FULL = 256


# --- the organiser's code, transcribed ------------------------------------

def fp_count_hit(hit_matrix):
    gt_matches_per_pred = hit_matrix.sum(axis=0)
    return int((gt_matches_per_pred == 0).sum())


def fn_count_hit(hit_matrix):
    pred_matches_per_gt = hit_matrix.sum(axis=1)
    return int((pred_matches_per_gt == 0).sum())


def get_overlap_matrices(gt_layers, pred_layers):
    n_gt = gt_layers.shape[0]
    n_pred = pred_layers.shape[0]
    gt_flat = gt_layers.reshape(n_gt, -1)
    pred_flat = pred_layers.reshape(n_pred, -1)
    intersection = gt_flat @ pred_flat.T
    gt_areas = gt_flat.sum(axis=1).reshape(-1, 1)
    pred_areas = pred_flat.sum(axis=1).reshape(1, -1)
    union = gt_areas + pred_areas - intersection
    return np.where(union == 0, 0.0, intersection / np.where(union == 0, 1, union))


def official_pq(gt_by_record, pred_by_stem, size=FULL):
    """get_pq_score, over the same per-annotator-record grouping Kaggle uses."""
    def layers(rles):
        if not rles:
            return np.zeros((0, size, size), dtype=np.float32)
        decoded = mask_util.decode([{"size": [size, size], "counts": r.encode("ascii")}
                                    for r in rles])
        return decoded.transpose(2, 0, 1).astype(np.float32)

    tp_iou_scores, fp_count, fn_count = [], 0, 0
    for record, gt_rles in gt_by_record.items():
        image_id = record.split("-", maxsplit=1)[1]
        pred_rles = pred_by_stem.get(image_id, [])
        n_gt, n_pred = len(gt_rles), len(pred_rles)
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue
        iou_matrix = get_overlap_matrices(layers(gt_rles), layers(pred_rles))
        hit_matrix = iou_matrix > 0.5
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)

    denominator = len(tp_iou_scores) + 0.5 * fp_count + 0.5 * fn_count
    return sum(tp_iou_scores) / denominator if denominator > 0 else 0.0


# --- ours, through the same frames the evaluator reads -----------------------

def ours(gt_by_record, pred_by_stem, size=FULL):
    gt = pd.DataFrame(
        [{"filament_id": f"{record}_{k}", "segmentation_rle": rle}
         for record, rles in gt_by_record.items() for k, rle in enumerate(rles)],
        columns=["filament_id", "segmentation_rle"])
    pred = pd.DataFrame(
        [{"filament_id": f"{stem}_{k}", "segmentation_rle": rle}
         for stem, rles in pred_by_stem.items() for k, rle in enumerate(rles)],
        columns=["filament_id", "segmentation_rle"])
    return pq_score(gt, pred, size, size)


def rect(y0, x0, height, width):
    canvas = np.zeros((FULL, FULL), dtype=np.uint8)
    canvas[y0:y0 + height, x0:x0 + width] = 1
    return mask_util.encode(np.asfortranarray(canvas))["counts"].decode("ascii")


SCENARIOS = {
    "perfect single hit": (
        {"a-img": [rect(20, 20, 40, 40)]},
        {"img": [rect(20, 20, 40, 40)]},
    ),
    "partial overlap below threshold": (
        {"a-img": [rect(20, 20, 40, 40)]},
        {"img": [rect(45, 45, 40, 40)]},
    ),
    "one prediction, three annotators": (
        {"a-img": [rect(20, 20, 40, 40)],
         "b-img": [rect(21, 21, 40, 40)],
         "c-img": [rect(60, 60, 30, 30)]},
        {"img": [rect(20, 20, 41, 41)]},
    ),
    "prediction with nothing to match": (
        {"a-img": [rect(20, 20, 40, 40)]},
        {"img": [rect(20, 20, 40, 40), rect(150, 150, 30, 30)]},
    ),
    # A record with no instances cannot appear in a ground-truth CSV; both
    # implementations score the stem 0.
    "annotator drew nothing": (
        {"a-img": []},
        {"img": [rect(20, 20, 40, 40)]},
    ),
    "we predicted nothing": (
        {"a-img": [rect(20, 20, 40, 40), rect(100, 100, 30, 30)]},
        {"img": []},
    ),
    "image absent from predictions": (
        {"a-img": [rect(20, 20, 40, 40)]},
        {},
    ),
    "two GT covered by one large prediction": (
        {"a-img": [rect(20, 20, 40, 40), rect(24, 24, 40, 40)]},
        {"img": [rect(20, 20, 44, 44)]},
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_our_pq_equals_the_organisers(name):
    gt, pred = SCENARIOS[name]
    # 1e-6, not exact: the organiser's code builds the IoU matrix from a
    # float32 matmul over dense layers, while pycocotools counts run-length
    # intersections in integers. The difference is accumulation noise.
    assert ours(gt, pred).pq == pytest.approx(official_pq(gt, pred), abs=1e-6), name


def test_matching_is_not_one_to_one():
    """One prediction clearing 0.5 against two annotators' records is two true positives."""
    gt, pred = SCENARIOS["one prediction, three annotators"]
    scores = ours(gt, pred)
    assert scores.tp == 2, "both agreeing annotators should count separately"
    # Against the third annotator, who traced a filament elsewhere, the same
    # prediction hits nothing: one FP and one FN for that record.
    assert scores.fp == 1 and scores.fn == 1


def test_random_scenes_agree():
    """Fuzz the shapes rather than trusting the hand-picked cases."""
    rng = np.random.default_rng(20260816)
    for _ in range(25):
        gt, pred = {}, {}
        for annotator in "abc":
            gt[f"{annotator}-img"] = [
                rect(*rng.integers(5, 150, size=2).tolist(),
                     *rng.integers(20, 80, size=2).tolist())
                for _ in range(int(rng.integers(1, 4)))
            ]
        pred["img"] = [
            rect(*rng.integers(5, 150, size=2).tolist(),
                 *rng.integers(20, 80, size=2).tolist())
            for _ in range(int(rng.integers(0, 4)))
        ]
        assert ours(gt, pred).pq == pytest.approx(official_pq(gt, pred), abs=1e-6)
