"""IoU / Dice distributions and one-to-many / many-to-one relations.

These are the quantitative criteria the evaluation committee computes on top
of Panoptic Quality. The definitions follow the organizers' self-evaluation
notebook and their forum answers:

Unit
    The **annotator record**. Ground-truth ids ``"<annotator>-<stem>_<k>"`` are
    grouped by ``"<annotator>-<stem>"`` and each record is compared with every
    prediction ``"<stem>_<j>"`` of that stem -- the same grouping Panoptic
    Quality uses (:mod:`filament.metric`). An image with three annotators is
    therefore analysed three times, and a prediction appears once per record.

Pairwise scores
    For a ground-truth instance of area ``A`` and a prediction of area ``B``
    that share ``I`` pixels::

        IoU  = I / (A + B - I)
        Dice = 2 I / (A + B)

Distributions
    Every pair with ``IoU > 0`` is kept -- all overlapping pairs, not only the
    ``IoU > 0.5`` matches -- and summarised as a 50-bin histogram on [0, 1],
    normalised, together with its mean (:func:`iou_dice_histograms`,
    :func:`distribution_summary`).

Relations (hit = ``IoU > 0``, any shared pixel)
    Per ground-truth instance, the number ``n`` of predictions it overlaps:
    ``1:0`` (missed), ``1:1``, ``1:n`` with ``n >= 2`` (fragmentation).
    Per prediction, the number ``n`` of ground-truth instances it overlaps:
    ``0:1`` (missed, i.e. a detection matching nothing), ``1:1``,
    ``n:1`` with ``n >= 2`` (merging). See :func:`relation_counts`.

:func:`human_baseline` runs the same analysis with one annotator's record in
place of the prediction and another annotator's record as ground truth, which
gives every number here a human anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .metric import (
    IMAGE_SIZE,
    IOU_THRESHOLD,
    PQResult,
    gt_group_keys,
    image_stem_of,
    iou_matrix_from_rles,
    pred_group_keys,
    rle_areas,
)

N_BINS = 50
MODEL_SOURCE = "model"

PAIR_COLUMNS = ["record", "stem", "source", "gt_id", "pred_id", "gt_area", "pred_area",
                "intersection", "iou", "dice"]
GT_COLUMNS = ["record", "stem", "source", "gt_id", "area", "n_overlapping", "best_iou"]
PRED_COLUMNS = ["record", "stem", "source", "pred_id", "area", "n_overlapping", "best_iou"]
COMPARISON_COLUMNS = ["record", "stem", "source", "n_gt", "n_pred", "tp", "fp", "fn", "iou_sum"]


@dataclass(frozen=True)
class MatchAnalysis:
    """Everything the committee's criteria are computed from.

    ``pairs``
        one row per overlapping (ground truth, prediction) pair, ``IoU > 0``;
    ``gt``
        one row per ground-truth instance and comparison, with the number of
        predictions it overlaps (``n_overlapping``) and its best IoU;
    ``pred``
        one row per prediction and comparison, with the number of
        ground-truth instances it overlaps;
    ``comparisons``
        one row per (annotator record, prediction source) with the Panoptic
        Quality counts ``tp``, ``fp``, ``fn`` and the summed IoU of matches.

    ``source`` is ``"model"`` for :func:`analyze` and the predicting record's
    id for :func:`human_baseline`.
    """

    pairs: pd.DataFrame
    gt: pd.DataFrame
    pred: pd.DataFrame
    comparisons: pd.DataFrame

    @property
    def pq(self) -> PQResult:
        """Panoptic Quality pooled over every comparison (equals ``pq_score``)."""
        return _pq_from_counts(self.comparisons)

    def pq_by(self, column: str) -> pd.DataFrame:
        """Panoptic Quality per ``"stem"``, ``"record"`` or ``"source"``."""
        rows = []
        for key, group in self.comparisons.groupby(column, sort=True):
            result = _pq_from_counts(group)
            rows.append({column: key, "pq": result.pq, "sq": result.sq, "rq": result.rq,
                         "tp": result.tp, "fp": result.fp, "fn": result.fn})
        return pd.DataFrame(rows, columns=[column, "pq", "sq", "rq", "tp", "fp", "fn"])


# --------------------------------------------------------------------------
# Core comparison
# --------------------------------------------------------------------------

def _compare(record: str, stem: str, source: str,
             gt_ids: Sequence[str], gt_rles: Sequence[str],
             pred_ids: Sequence[str], pred_rles: Sequence[str],
             height: int, width: int) -> Tuple[list, list, list, dict]:
    """Rows for one annotator record against one set of predictions."""
    iou = iou_matrix_from_rles(gt_rles, pred_rles, height, width)
    gt_areas = rle_areas(gt_rles, height, width)
    pred_areas = rle_areas(pred_rles, height, width)

    # pycocotools reports IoU = I / U with I and U pixel counts, so I is
    # recovered exactly as an integer: I = IoU (A + B) / (1 + IoU).
    sums = gt_areas[:, None] + pred_areas[None, :]
    intersection = np.rint(iou * sums / (1.0 + iou)).astype(np.int64)
    overlap = intersection > 0

    pairs = []
    for g, p in zip(*np.nonzero(overlap)):
        inter = int(intersection[g, p])
        a, b = int(gt_areas[g]), int(pred_areas[p])
        pairs.append({"record": record, "stem": stem, "source": source,
                      "gt_id": gt_ids[g], "pred_id": pred_ids[p],
                      "gt_area": a, "pred_area": b, "intersection": inter,
                      "iou": inter / (a + b - inter), "dice": 2.0 * inter / (a + b)})

    best_gt = iou.max(axis=1) if iou.size else np.zeros(len(gt_rles))
    best_pred = iou.max(axis=0) if iou.size else np.zeros(len(pred_rles))
    gt_rows = [{"record": record, "stem": stem, "source": source, "gt_id": gt_ids[g],
                "area": int(gt_areas[g]), "n_overlapping": int(overlap[g].sum()),
                "best_iou": float(best_gt[g])} for g in range(len(gt_rles))]
    pred_rows = [{"record": record, "stem": stem, "source": source, "pred_id": pred_ids[p],
                  "area": int(pred_areas[p]), "n_overlapping": int(overlap[:, p].sum()),
                  "best_iou": float(best_pred[p])} for p in range(len(pred_rles))]

    # The evaluator's matching: IoU > 0.5 is one-to-one by construction.
    hit = iou > IOU_THRESHOLD
    tp = int(hit.sum())
    comparison = {"record": record, "stem": stem, "source": source,
                  "n_gt": len(gt_rles), "n_pred": len(pred_rles), "tp": tp,
                  "fp": int((hit.sum(axis=0) == 0).sum()) if len(gt_rles) else len(pred_rles),
                  "fn": int((hit.sum(axis=1) == 0).sum()) if len(pred_rles) else len(gt_rles),
                  "iou_sum": float(iou[hit].sum())}
    return pairs, gt_rows, pred_rows, comparison


def _collect(comparisons: Iterator[tuple], height: int, width: int) -> MatchAnalysis:
    pairs, gts, preds, counts = [], [], [], []
    for args in comparisons:
        p, g, d, c = _compare(*args, height=height, width=width)
        pairs.extend(p)
        gts.extend(g)
        preds.extend(d)
        counts.append(c)
    return MatchAnalysis(
        pairs=pd.DataFrame(pairs, columns=PAIR_COLUMNS),
        gt=pd.DataFrame(gts, columns=GT_COLUMNS),
        pred=pd.DataFrame(preds, columns=PRED_COLUMNS),
        comparisons=pd.DataFrame(counts, columns=COMPARISON_COLUMNS),
    )


def _records(gt_df: pd.DataFrame) -> Dict[str, Tuple[List[str], List[str]]]:
    """``{record: (filament_ids, rles)}`` in frame order."""
    records: Dict[str, Tuple[List[str], List[str]]] = {}
    for key, fid, rle in zip(gt_group_keys(gt_df["filament_id"]), gt_df["filament_id"],
                             gt_df["segmentation_rle"]):
        ids, rles = records.setdefault(key, ([], []))
        ids.append(fid)
        rles.append(rle)
    return records


def analyze(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> MatchAnalysis:
    """Compare predictions with every annotator record present in ``gt_df``.

    Both frames need ``filament_id`` and ``segmentation_rle``, exactly as for
    :func:`filament.metric.pq_score`. Predictions for stems absent from
    ``gt_df`` are ignored, as the evaluator ignores them.
    """
    predictions: Dict[str, Tuple[List[str], List[str]]] = {}
    for stem, fid, rle in zip(pred_group_keys(pred_df["filament_id"]), pred_df["filament_id"],
                              pred_df["segmentation_rle"]):
        ids, rles = predictions.setdefault(stem, ([], []))
        ids.append(fid)
        rles.append(rle)

    def comparisons():
        for record, (gt_ids, gt_rles) in _records(gt_df).items():
            stem = image_stem_of(record)
            pred_ids, pred_rles = predictions.get(stem, ([], []))
            yield record, stem, MODEL_SOURCE, gt_ids, gt_rles, pred_ids, pred_rles

    return _collect(comparisons(), height, width)


def human_baseline(
    gt_df: pd.DataFrame,
    include_self: bool = False,
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
) -> MatchAnalysis:
    """The same analysis with one annotator standing in for the model.

    For every stem with two or more annotator records, each record in turn is
    the prediction (``source``) and every *other* record of that stem is the
    ground truth. ``include_self=True`` also scores each record against itself,
    which is how "one expert replicated perfectly" is measured.

    ``pq_by("source")["pq"].mean()`` over the training set reproduces the two
    human anchors: 0.342 between experts and 0.598 with ``include_self``.
    """
    by_stem: Dict[str, Dict[str, Tuple[List[str], List[str]]]] = {}
    for record, value in _records(gt_df).items():
        by_stem.setdefault(image_stem_of(record), {})[record] = value

    def comparisons():
        for stem in sorted(by_stem):
            records = by_stem[stem]
            if len(records) < 2:
                continue
            for source in sorted(records):
                pred_ids, pred_rles = records[source]
                for record in sorted(records):
                    if record == source and not include_self:
                        continue
                    gt_ids, gt_rles = records[record]
                    yield record, stem, source, gt_ids, gt_rles, pred_ids, pred_rles

    return _collect(comparisons(), height, width)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def _pq_from_counts(comparisons: pd.DataFrame) -> PQResult:
    tp = int(comparisons["tp"].sum())
    fp = int(comparisons["fp"].sum())
    fn = int(comparisons["fn"].sum())
    iou_sum = float(comparisons["iou_sum"].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return PQResult(
        pq=iou_sum / denominator if denominator > 0 else 0.0,
        sq=iou_sum / tp if tp > 0 else 0.0,
        rq=tp / denominator if denominator > 0 else 0.0,
        tp=tp, fp=fp, fn=fn,
        n_gt=int(comparisons["n_gt"].sum()), n_pred=int(comparisons["n_pred"].sum()),
        n_groups=len(comparisons),
    )


def iou_dice_histograms(analysis: MatchAnalysis, bins: int = N_BINS) -> pd.DataFrame:
    """IoU and Dice of all overlapping pairs as ``bins`` equal bins on [0, 1].

    ``*_fraction`` columns sum to 1 (each bin's share of the pairs);
    ``*_density`` columns integrate to 1 (``fraction * bins``), which is what
    ``numpy.histogram(..., density=True)`` returns. A value of exactly 1.0
    falls in the last bin.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    table = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:]})
    total = len(analysis.pairs)
    for name in ("iou", "dice"):
        counts, _ = np.histogram(analysis.pairs[name].to_numpy(), bins=edges)
        fraction = counts / total if total else np.zeros(bins)
        table[f"{name}_count"] = counts
        table[f"{name}_fraction"] = fraction
        table[f"{name}_density"] = fraction * bins
    return table


def distribution_summary(analysis: MatchAnalysis) -> Dict[str, float]:
    """Means and medians of IoU and Dice over all overlapping pairs.

    ``matched_*`` restricts to the ``IoU > 0.5`` matches; ``matched_mean_iou``
    is Panoptic Quality's SQ.
    """
    pairs = analysis.pairs
    matched = pairs[pairs["iou"] > IOU_THRESHOLD]

    def stat(frame, column, how):
        return float(getattr(frame[column], how)()) if len(frame) else float("nan")

    return {
        "n_pairs": int(len(pairs)),
        "mean_iou": stat(pairs, "iou", "mean"),
        "median_iou": stat(pairs, "iou", "median"),
        "mean_dice": stat(pairs, "dice", "mean"),
        "median_dice": stat(pairs, "dice", "median"),
        "n_matched": int(len(matched)),
        "matched_mean_iou": stat(matched, "iou", "mean"),
        "matched_mean_dice": stat(matched, "dice", "mean"),
    }


def relation_counts(analysis: MatchAnalysis) -> pd.DataFrame:
    """Counts and shares of 1:0, 1:1, 1:n (ground truth) and 0:1, 1:1, n:1 (predictions).

    Shares are within each side: per ground-truth instance and comparison, and
    per prediction and comparison.
    """
    rows = []
    sides = (
        ("ground truth", analysis.gt["n_overlapping"],
         (("1:0", "missed", 0), ("1:1", "one prediction", 1), ("1:n", "fragmented", 2))),
        ("prediction", analysis.pred["n_overlapping"],
         (("0:1", "missed", 0), ("1:1", "one ground truth", 1), ("n:1", "merged", 2))),
    )
    for side, n, relations in sides:
        total = len(n)
        for relation, meaning, k in relations:
            count = int((n >= 2).sum()) if k == 2 else int((n == k).sum())
            rows.append({"side": side, "relation": relation, "meaning": meaning,
                         "count": count, "share": count / total if total else float("nan")})
    return pd.DataFrame(rows, columns=["side", "relation", "meaning", "count", "share"])


def overlap_distribution(analysis: MatchAnalysis, max_n: int = 5) -> pd.DataFrame:
    """How many instances overlap 0, 1, 2, ... partners, per side.

    The last row collects ``max_n`` and more.
    """
    labels = [str(k) for k in range(max_n)] + [f">={max_n}"]
    table = {"n_overlapping": labels}
    for side, frame in (("ground_truth", analysis.gt), ("prediction", analysis.pred)):
        n = frame["n_overlapping"].to_numpy()
        table[side] = [int((n == k).sum()) for k in range(max_n)] + [int((n >= max_n).sum())]
    return pd.DataFrame(table)
