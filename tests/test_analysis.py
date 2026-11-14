"""The committee's criteria: IoU / Dice of overlapping pairs and split / merge relations."""

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util

from filament.analysis import (
    analyze,
    distribution_summary,
    human_baseline,
    iou_dice_histograms,
    overlap_distribution,
    relation_counts,
)
from filament.metric import pq_score

from .conftest import frame, rect_rle

H = W = 64


def rect(r0, c0, r1, c1):
    return rect_rle(r0, c0, r1, c1, H)


def run(gt_pairs, pred_pairs):
    return analyze(frame(gt_pairs), frame(pred_pairs), H, W)


def relations(analysis):
    table = relation_counts(analysis)
    return {(row.side, row.relation): row.count for row in table.itertuples()}


def test_perfect_prediction():
    a = run([("ann1-img_0", rect(10, 10, 30, 30))], [("img_0", rect(10, 10, 30, 30))])
    assert list(a.pairs["iou"]) == [1.0] and list(a.pairs["dice"]) == [1.0]
    counts = relations(a)
    assert counts[("ground truth", "1:1")] == 1 and counts[("prediction", "1:1")] == 1
    assert a.pq.pq == pytest.approx(1.0)


def test_iou_and_dice_use_the_organizers_formulas():
    # GT 10x10 = 100 px; prediction 10x16 = 160 px sharing all 100 px.
    a = run([("ann1-img_0", rect(0, 0, 10, 10))], [("img_0", rect(0, 0, 10, 16))])
    pair = a.pairs.iloc[0]
    assert (pair.gt_area, pair.pred_area, pair.intersection) == (100, 160, 100)
    assert pair.iou == pytest.approx(100 / (100 + 160 - 100))
    assert pair.dice == pytest.approx(2 * 100 / (100 + 160))


def test_fragmentation_is_one_to_many_and_still_enters_the_distributions():
    """Two halves of one filament: each IoU 0.5, so neither matches, but both overlap."""
    a = run([("ann1-img_0", rect(10, 10, 30, 30))],
            [("img_0", rect(10, 10, 20, 30)), ("img_1", rect(20, 10, 30, 30))])
    counts = relations(a)
    assert counts[("ground truth", "1:n")] == 1
    assert counts[("prediction", "1:1")] == 2
    assert len(a.pairs) == 2 and a.pairs["iou"].tolist() == [0.5, 0.5]
    assert a.pq.tp == 0  # IoU 0.5 is not > 0.5


def test_merging_is_many_to_one():
    a = run([("ann1-img_0", rect(10, 10, 20, 30)), ("ann1-img_1", rect(30, 10, 40, 30))],
            [("img_0", rect(10, 10, 40, 30))])
    counts = relations(a)
    assert counts[("prediction", "n:1")] == 1
    assert counts[("ground truth", "1:1")] == 2


def test_misses_on_both_sides_and_touching_is_not_overlapping():
    a = run([("ann1-img_0", rect(0, 0, 10, 10))],
            [("img_0", rect(10, 0, 20, 10))])  # shares an edge, no pixel
    counts = relations(a)
    assert counts[("ground truth", "1:0")] == 1
    assert counts[("prediction", "0:1")] == 1
    assert a.pairs.empty


def test_an_image_without_predictions_is_all_misses():
    a = run([("ann1-img_0", rect(0, 0, 10, 10)), ("ann1-img_1", rect(20, 20, 30, 30))], [])
    assert relations(a)[("ground truth", "1:0")] == 2
    assert a.pred.empty


def test_each_annotator_record_is_a_separate_comparison():
    gt = [("ann1-img_0", rect(10, 10, 30, 30)), ("ann2-img_0", rect(40, 40, 50, 50))]
    a = run(gt, [("img_0", rect(10, 10, 30, 30))])
    assert len(a.comparisons) == 2
    # The prediction is counted once per record: 1:1 for ann1, 0:1 for ann2.
    counts = relations(a)
    assert counts[("prediction", "1:1")] == 1 and counts[("prediction", "0:1")] == 1
    assert len(a.pred) == 2


def test_predictions_for_unknown_stems_are_ignored():
    a = run([("ann1-img_0", rect(10, 10, 30, 30))],
            [("img_0", rect(10, 10, 30, 30)), ("other_0", rect(0, 0, 5, 5))])
    assert len(a.pred) == 1 and a.pq.pq == pytest.approx(1.0)


def random_scene(seed):
    rng = np.random.default_rng(seed)
    gt, pred = [], []
    for image in range(3):
        stem = f"img{image}"
        for annotator in range(int(rng.integers(1, 4))):
            for k in range(int(rng.integers(1, 5))):
                r0, c0 = rng.integers(0, 50, size=2)
                gt.append((f"ann{annotator}-{stem}_{k}",
                           rect(r0, c0, r0 + rng.integers(2, 16), c0 + rng.integers(2, 16))))
        for k in range(int(rng.integers(0, 6))):
            r0, c0 = rng.integers(0, 50, size=2)
            pred.append((f"{stem}_{k}",
                         rect(r0, c0, r0 + rng.integers(2, 16), c0 + rng.integers(2, 16))))
    return gt, pred


def decode(rle):
    return mask_util.decode({"size": [H, W], "counts": rle.encode("ascii")}).astype(bool)


@pytest.mark.parametrize("seed", range(10))
def test_pairs_and_counts_match_a_dense_computation(seed):
    gt, pred = random_scene(seed)
    a = run(gt, pred)
    masks = {fid: decode(rle) for fid, rle in gt + pred}

    expected_pairs, expected_gt_n = {}, {}
    for gid, _ in gt:
        stem = gid.split("-", 1)[1].rsplit("_", 1)[0]
        n = 0
        for pid, _ in pred:
            if pid.rsplit("_", 1)[0] != stem:
                continue
            inter = int((masks[gid] & masks[pid]).sum())
            if inter > 0:
                n += 1
                total = int(masks[gid].sum() + masks[pid].sum())
                expected_pairs[(gid, pid)] = (inter / (total - inter), 2 * inter / total)
        expected_gt_n[gid] = n

    got = {(r.gt_id, r.pred_id): (r.iou, r.dice) for r in a.pairs.itertuples()}
    assert got.keys() == expected_pairs.keys()
    for key, (iou, dice) in expected_pairs.items():
        assert got[key] == pytest.approx((iou, dice))
    assert dict(zip(a.gt["gt_id"], a.gt["n_overlapping"])) == expected_gt_n


@pytest.mark.parametrize("seed", range(10))
def test_panoptic_quality_agrees_with_the_metric(seed):
    gt, pred = random_scene(seed)
    a = run(gt, pred)
    expected = pq_score(frame(gt), frame(pred), H, W)
    assert (a.pq.tp, a.pq.fp, a.pq.fn) == (expected.tp, expected.fp, expected.fn)
    assert a.pq.pq == pytest.approx(expected.pq)
    summary = distribution_summary(a)
    assert summary["n_matched"] == expected.tp
    if expected.tp:
        assert summary["matched_mean_iou"] == pytest.approx(expected.sq)


def test_histograms_have_fifty_normalised_bins_on_the_unit_interval():
    gt, pred = random_scene(3)
    a = run(gt, pred)
    table = iou_dice_histograms(a)
    assert len(table) == 50
    assert table["bin_lo"].iloc[0] == 0.0 and table["bin_hi"].iloc[-1] == 1.0
    assert table["iou_count"].sum() == len(a.pairs)
    assert table["iou_fraction"].sum() == pytest.approx(1.0)
    assert (table["dice_density"] * 0.02).sum() == pytest.approx(1.0)
    perfect = run([("ann1-img_0", rect(0, 0, 9, 9))], [("img_0", rect(0, 0, 9, 9))])
    assert iou_dice_histograms(perfect)["iou_count"].iloc[-1] == 1  # 1.0 lands in the last bin
    summary = distribution_summary(a)
    assert summary["mean_iou"] == pytest.approx(a.pairs["iou"].mean())


def test_overlap_distribution_counts_every_instance():
    gt, pred = random_scene(5)
    a = run(gt, pred)
    table = overlap_distribution(a, max_n=3)
    assert list(table["n_overlapping"]) == ["0", "1", "2", ">=3"]
    assert table["ground_truth"].sum() == len(a.gt)
    assert table["prediction"].sum() == len(a.pred)


def test_human_baseline_compares_each_record_with_the_others():
    gt = frame([("ann1-img_0", rect(10, 10, 30, 30)),
                ("ann2-img_0", rect(10, 10, 30, 34)),   # same filament, drawn wider
                ("ann1-solo_0", rect(0, 0, 5, 5))])      # one annotator only: skipped
    a = human_baseline(gt, height=H, width=W)
    assert sorted(zip(a.comparisons["source"], a.comparisons["record"])) == [
        ("ann1-img", "ann2-img"), ("ann2-img", "ann1-img")]
    assert a.pairs["iou"].tolist() == pytest.approx([400 / 480, 400 / 480])
    per_source = a.pq_by("source")
    assert per_source["pq"].tolist() == pytest.approx([400 / 480, 400 / 480])

    with_self = human_baseline(gt, include_self=True, height=H, width=W)
    assert len(with_self.comparisons) == 4
    assert with_self.pq_by("source")["tp"].tolist() == [2, 2]


def test_human_baseline_reproduces_the_annotator_anchors(real_gt):
    """0.342 between experts, 0.598 for one expert scored against all, on the training set."""
    between = human_baseline(real_gt).pq_by("source")
    assert len(between) == 743
    assert between["pq"].mean() == pytest.approx(0.3418600962646357, abs=1e-9)
    with_self = human_baseline(real_gt, include_self=True).pq_by("source")
    assert with_self["pq"].mean() == pytest.approx(0.5983446061187612, abs=1e-9)
