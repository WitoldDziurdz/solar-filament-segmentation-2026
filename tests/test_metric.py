"""The fast metric must equal the organizer's dense implementation, exactly.

Every experiment is promoted or rejected on the number these functions return,
so the fast path is checked against a line-by-line port of the published
``self-evaluation-notebook`` on hand-built cases and on random ones.

Tests run at 64x64 so the dense reference stays cheap; the metric is
resolution-agnostic and the production size is 2048.
"""

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util

from filament.metric import (
    emission_break_even,
    expected_pq_delta,
    iou_matrix_from_rles,
    pq_reference,
    pq_score,
)

H = W = 64


def rect_rle(r0, c0, r1, c1, height=H, width=W):
    """Compressed-RLE counts string for the filled rectangle [r0:r1, c0:c1)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[r0:r1, c0:c1] = 1
    return mask_util.encode(np.asfortranarray(mask))["counts"].decode("ascii")


def frame(pairs):
    """Build a submission-shaped frame from ``[(filament_id, rle), ...]``."""
    return pd.DataFrame(
        {"filament_id": [p[0] for p in pairs], "segmentation_rle": [p[1] for p in pairs]}
    )


def both(gt, pred):
    """Return (fast PQ, reference PQ) for the same inputs."""
    return pq_score(gt, pred, H, W).pq, pq_reference(gt, pred, H, W)


# --------------------------------------------------------------------------
# Agreement between the two implementations
# --------------------------------------------------------------------------

def test_perfect_prediction_scores_one():
    rle = rect_rle(10, 10, 30, 30)
    gt = frame([("ann1-img001_0", rle)])
    pred = frame([("img001_0", rle)])

    fast, ref = both(gt, pred)
    assert fast == pytest.approx(1.0)
    assert fast == pytest.approx(ref)


def test_no_predictions_scores_zero():
    gt = frame([("ann1-img001_0", rect_rle(10, 10, 30, 30))])
    pred = frame([])

    fast, ref = both(gt, pred)
    assert fast == pytest.approx(0.0)
    assert fast == pytest.approx(ref)


def test_empty_masks_score_zero_not_one():
    """The pre-fix evaluator gave 1.00 for empty masks. This one must not."""
    empty = mask_util.encode(np.asfortranarray(np.zeros((H, W), dtype=np.uint8)))
    empty_counts = empty["counts"].decode("ascii")

    gt = frame([("ann1-img001_0", rect_rle(10, 10, 30, 30))])
    pred = frame([("img001_0", empty_counts)])

    result = pq_score(gt, pred, H, W)
    assert result.pq == pytest.approx(0.0)
    assert result.pq == pytest.approx(pq_reference(gt, pred, H, W))
    # An empty mask overlaps nothing, so it is a false positive on top of the miss.
    assert (result.fp, result.fn, result.tp) == (1, 1, 0)


def test_splitting_one_filament_costs_a_double_penalty():
    """A split instance fails IoU>0.5 and is charged as both FP and FN."""
    gt = frame([("ann1-img001_0", rect_rle(10, 10, 30, 30))])
    pred = frame(
        [
            ("img001_0", rect_rle(10, 10, 20, 30)),
            ("img001_1", rect_rle(20, 10, 30, 30)),
        ]
    )

    result = pq_score(gt, pred, H, W)
    assert result.tp == 0
    assert result.fp == 2  # both halves match nothing
    assert result.fn == 1  # the ground truth is missed
    assert result.pq == pytest.approx(0.0)
    assert result.pq == pytest.approx(pq_reference(gt, pred, H, W))


def test_merging_two_filaments_loses_both():
    gt = frame(
        [
            ("ann1-img001_0", rect_rle(10, 10, 20, 30)),
            ("ann1-img001_1", rect_rle(30, 10, 40, 30)),
        ]
    )
    pred = frame([("img001_0", rect_rle(10, 10, 40, 30))])

    result = pq_score(gt, pred, H, W)
    assert result.tp == 0
    assert result.fn == 2
    assert result.pq == pytest.approx(pq_reference(gt, pred, H, W))


def test_iou_just_above_threshold_is_a_hit():
    """The metric has a cliff at 0.5; verify which side of it we land on."""
    gt = frame([("ann1-img001_0", rect_rle(0, 0, 10, 10))])          # 100 px
    pred = frame([("img001_0", rect_rle(0, 0, 10, 16))])             # 160 px, inter 100

    iou = iou_matrix_from_rles(
        gt["segmentation_rle"].tolist(), pred["segmentation_rle"].tolist(), H, W
    )[0, 0]
    assert iou == pytest.approx(100 / 160)  # 0.625 > 0.5

    result = pq_score(gt, pred, H, W)
    assert result.tp == 1
    assert result.pq == pytest.approx(0.625)
    assert result.pq == pytest.approx(pq_reference(gt, pred, H, W))


def test_iou_just_below_threshold_is_a_double_miss():
    gt = frame([("ann1-img001_0", rect_rle(0, 0, 10, 10))])          # 100 px
    pred = frame([("img001_0", rect_rle(0, 0, 10, 21))])             # 210 px, inter 100

    iou = iou_matrix_from_rles(
        gt["segmentation_rle"].tolist(), pred["segmentation_rle"].tolist(), H, W
    )[0, 0]
    assert iou == pytest.approx(100 / 210)  # 0.476 < 0.5

    result = pq_score(gt, pred, H, W)
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)
    assert result.pq == pytest.approx(0.0)


def test_missing_image_is_penalised_as_false_negatives():
    """Omitting an image must not be free -- that was the old evaluator's bug."""
    gt = frame(
        [
            ("ann1-img001_0", rect_rle(10, 10, 30, 30)),
            ("ann1-img002_0", rect_rle(10, 10, 30, 30)),
        ]
    )
    pred = frame([("img001_0", rect_rle(10, 10, 30, 30))])  # img002 omitted entirely

    result = pq_score(gt, pred, H, W)
    assert result.fn == 1
    assert result.pq == pytest.approx(1.0 / 1.5)
    assert result.pq == pytest.approx(pq_reference(gt, pred, H, W))


def test_each_annotator_record_is_scored_separately():
    """Two annotators on one image means the prediction is judged twice."""
    good = rect_rle(10, 10, 30, 30)
    gt = frame(
        [
            ("ann1-img001_0", good),
            ("ann2-img001_0", rect_rle(40, 40, 50, 50)),  # a very different opinion
        ]
    )
    pred = frame([("img001_0", good)])

    result = pq_score(gt, pred, H, W)
    assert result.n_groups == 2
    # Perfect for annotator 1; for annotator 2 the same mask is an FP plus an FN.
    assert (result.tp, result.fp, result.fn) == (1, 1, 1)
    assert result.pq == pytest.approx(1.0 / 2.0)
    assert result.pq == pytest.approx(pq_reference(gt, pred, H, W))


def test_predictions_for_unknown_images_are_ignored():
    gt = frame([("ann1-img001_0", rect_rle(10, 10, 30, 30))])
    pred = frame(
        [
            ("img001_0", rect_rle(10, 10, 30, 30)),
            ("img999_0", rect_rle(10, 10, 30, 30)),  # not in the ground truth
        ]
    )

    result = pq_score(gt, pred, H, W)
    assert result.pq == pytest.approx(1.0)
    assert result.pq == pytest.approx(pq_reference(gt, pred, H, W))


@pytest.mark.parametrize("seed", range(12))
def test_fast_and_reference_agree_on_random_scenes(seed):
    rng = np.random.default_rng(seed)

    gt_rows, pred_rows = [], []
    for image_index in range(3):
        stem = f"img{image_index:03d}"
        for annotator in range(rng.integers(1, 3)):
            for k in range(rng.integers(0, 4)):
                r0, c0 = rng.integers(0, 50, size=2)
                gt_rows.append(
                    (
                        f"ann{annotator}-{stem}_{k}",
                        rect_rle(r0, c0, r0 + rng.integers(2, 14), c0 + rng.integers(2, 14)),
                    )
                )
        for k in range(rng.integers(0, 5)):
            r0, c0 = rng.integers(0, 50, size=2)
            pred_rows.append(
                (
                    f"{stem}_{k}",
                    rect_rle(r0, c0, r0 + rng.integers(2, 14), c0 + rng.integers(2, 14)),
                )
            )

    gt, pred = frame(gt_rows), frame(pred_rows)
    if gt.empty:
        pytest.skip("no ground truth generated for this seed")

    assert pq_score(gt, pred, H, W).pq == pytest.approx(pq_reference(gt, pred, H, W))


def test_pq_decomposes_into_sq_times_rq():
    gt = frame(
        [
            ("ann1-img001_0", rect_rle(0, 0, 10, 10)),
            ("ann1-img001_1", rect_rle(20, 20, 30, 30)),
            ("ann1-img002_0", rect_rle(0, 0, 10, 10)),
        ]
    )
    pred = frame(
        [
            ("img001_0", rect_rle(0, 0, 10, 12)),
            ("img001_1", rect_rle(20, 20, 30, 30)),
            ("img002_0", rect_rle(40, 40, 50, 50)),
        ]
    )

    result = pq_score(gt, pred, H, W)
    assert result.pq == pytest.approx(result.sq * result.rq)
    assert result.pq == pytest.approx(pq_reference(gt, pred, H, W))


# --------------------------------------------------------------------------
# Emission decision rule
# --------------------------------------------------------------------------

def test_break_even_is_half_of_current_pq():
    assert emission_break_even(0.40) == pytest.approx(0.20)
    assert emission_break_even(0.0) == pytest.approx(0.0)


def test_confident_detection_is_worth_emitting():
    # 60% chance of clearing the threshold at a typical matched IoU of 0.7.
    assert expected_pq_delta(0.6, 0.7, current_pq=0.4) > 0


def test_doubtful_detection_is_better_deleted():
    # 20% chance: expected 0.14 against a 0.20 break-even.
    assert expected_pq_delta(0.2, 0.7, current_pq=0.4) < 0


def test_deleting_a_doubtful_instance_really_raises_pq():
    """The decision rule is only worth trusting if the metric agrees with it."""
    gt = frame([("ann1-img001_0", rect_rle(0, 0, 20, 20))])

    kept = frame(
        [
            ("img001_0", rect_rle(0, 0, 20, 20)),      # a hit
            ("img001_1", rect_rle(40, 40, 45, 45)),    # a doubtful extra, matches nothing
        ]
    )
    dropped = frame([("img001_0", rect_rle(0, 0, 20, 20))])

    assert pq_score(gt, dropped, H, W).pq > pq_score(gt, kept, H, W).pq
    assert pq_score(gt, kept, H, W).pq == pytest.approx(1 / 1.5)
    assert pq_score(gt, dropped, H, W).pq == pytest.approx(1.0)
