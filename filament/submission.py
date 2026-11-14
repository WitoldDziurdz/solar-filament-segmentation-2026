"""Building and validating submission CSVs.

The competition's evaluator is unforgiving about shape and silent about
mistakes: a mask encoded at the wrong resolution, a duplicated id, or a
zero-area instance produces a valid-looking file and a bad score. Every
submission goes through :func:`validate` before it is written.

Note the asymmetry that motivates :func:`drop_low_confidence`: an emitted
instance that misses costs a false positive *and* a false negative, twice the
price of never emitting it. Under Panoptic Quality an instance is worth
emitting only when ``p(IoU > 0.5) * E[IoU | hit] > 0.5 * PQ``
(:func:`filament.metric.emission_break_even`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

IMAGE_SIZE = 2048
COLUMNS = ["filament_id", "segmentation_rle"]


class SubmissionError(ValueError):
    """Raised when a submission would be silently mis-scored."""


def mask_to_rle(mask: np.ndarray) -> str:
    """Binary ``(H, W)`` array -> compressed COCO RLE counts string."""
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def build(per_image: Dict[str, Sequence[str]]) -> pd.DataFrame:
    """``{image_stem: [rle, ...]}`` -> submission frame, one row per instance."""
    rows = []
    for stem, rles in per_image.items():
        for index, rle in enumerate(rles):
            rows.append({"filament_id": f"{stem}_{index}", "segmentation_rle": rle})
    return pd.DataFrame(rows, columns=COLUMNS)


def drop_low_confidence(
    scored: Sequence[Tuple[str, float]],
    threshold: float,
) -> List[str]:
    """Keep only instances at or above ``threshold``.

    ``threshold`` should come from a sweep against real PQ on a held-out fold,
    not from a detector's default -- the PQ-optimal cut sits far above the usual
    0.05, because a doubtful instance is worse than no instance at all.
    """
    return [rle for rle, score in scored if score >= threshold]


def resolve_overlaps(
    rles: Sequence[str],
    scores: Sequence[float] | None = None,
    min_area: int = 1,
    min_retained: float = 0.5,
    size: int = IMAGE_SIZE,
) -> List[str]:
    """Make one image's instances disjoint, in priority order.

    Instances are processed best-first and each claims only pixels no earlier
    instance took. Priority is ``scores`` when given, otherwise area descending.

    An instance that loses more than ``1 - min_retained`` of its pixels is
    dropped rather than emitted as a sliver: a shrunken mask usually falls below
    IoU 0.5 anyway, and a prediction that misses costs a false positive *and* a
    false negative, twice what omitting it costs.
    """
    if not len(rles):
        return []

    order = (
        np.argsort(-np.asarray(scores, dtype=float))
        if scores is not None
        else np.argsort(-np.asarray([
            int(mask_util.area({"size": [size, size], "counts": r.encode("ascii")}))
            for r in rles
        ]))
    )

    claimed = np.zeros((size, size), dtype=bool)
    resolved: List[tuple[int, str]] = []

    for position in order:
        decoded = mask_util.decode(
            {"size": [size, size], "counts": rles[position].encode("ascii")}
        ).astype(bool)
        original = int(decoded.sum())
        if original == 0:
            continue

        decoded &= ~claimed
        retained = int(decoded.sum())
        if retained < min_area or retained < min_retained * original:
            continue

        claimed |= decoded
        resolved.append(
            (
                int(position),
                mask_util.encode(np.asfortranarray(decoded.astype(np.uint8)))["counts"].decode("ascii"),
            )
        )

    # Restore the caller's ordering so ids stay stable and comparable.
    return [rle for _, rle in sorted(resolved, key=lambda item: item[0])]


def postprocess(
    instances: Iterable[Tuple[Union[np.ndarray, str], float]],
    score_threshold: Optional[float] = None,
    min_area: int = 1,
    min_retained: float = 0.5,
    size: int = IMAGE_SIZE,
) -> List[str]:
    """Raw model output for one image -> disjoint, canonical RLE strings.

    ``instances`` is what a model returns for one image: ``(mask, score)`` pairs
    where ``mask`` is a boolean ``(size, size)`` array at native resolution or
    an RLE counts string. The steps are:

    1. drop instances scoring below ``score_threshold`` (``None`` keeps all);
    2. encode every mask as compressed RLE at native resolution;
    3. :func:`resolve_overlaps` best-score-first, dropping instances that keep
       less than ``min_retained`` of their area or fewer than ``min_area`` pixels.

    A mask at any other resolution is rejected: thresholding at low resolution
    and upsampling the binary result shifts every boundary, and one pixel of
    radius costs about a tenth of the score.
    """
    rles: List[str] = []
    scores: List[float] = []
    for mask, score in instances:
        if score_threshold is not None and score < score_threshold:
            continue
        if isinstance(mask, str):
            rle = mask
        else:
            mask = np.asarray(mask)
            if mask.shape != (size, size):
                raise SubmissionError(
                    f"mask has shape {mask.shape}; masks must be ({size}, {size}). "
                    "Resize the probability map before thresholding, never the binary mask."
                )
            rle = mask_to_rle(mask.astype(bool))
        rles.append(rle)
        scores.append(float(score))
    return resolve_overlaps(rles, scores, min_area=min_area, min_retained=min_retained, size=size)


@dataclass(frozen=True)
class SubmissionStats:
    rows: int
    images: int
    min_area: int
    median_instances_per_image: float

    def __str__(self) -> str:
        return (
            f"{self.rows} instances over {self.images} images, "
            f"median {self.median_instances_per_image:.1f} per image, "
            f"smallest {self.min_area} px"
        )


def validate(
    submission: pd.DataFrame,
    expected_stems: Iterable[str] | None = None,
    size: int = IMAGE_SIZE,
) -> SubmissionStats:
    """Check every property the evaluator assumes but does not report on.

    Raises :class:`SubmissionError` on anything that would mis-score. Missing
    images are reported as a warning rather than an error: omitting an image is
    legal and merely costs false negatives, whereas the other failures are bugs.
    """
    if list(submission.columns) != COLUMNS:
        raise SubmissionError(f"columns must be exactly {COLUMNS}, got {list(submission.columns)}")

    if submission.empty:
        raise SubmissionError("submission is empty; this scores zero")

    if not submission["filament_id"].is_unique:
        duplicates = submission.loc[submission["filament_id"].duplicated(), "filament_id"].head()
        raise SubmissionError(f"duplicate filament_id, e.g. {list(duplicates)}")

    if submission["segmentation_rle"].isna().any():
        raise SubmissionError("null segmentation_rle present")

    areas = []
    for filament_id, counts in zip(submission["filament_id"], submission["segmentation_rle"]):
        rle = {"size": [size, size], "counts": counts.encode("ascii")}
        try:
            area = int(mask_util.area(rle))
            decoded = mask_util.decode(rle)
        except Exception as error:  # noqa: BLE001 - surface the offending row
            raise SubmissionError(f"RLE for {filament_id} does not decode at {size}x{size}: {error}")

        if area == 0:
            raise SubmissionError(
                f"zero-area instance {filament_id}: an empty mask matches nothing and is "
                "charged as a false positive on top of the miss it fails to fix"
            )

        # pycocotools does not object to a mask encoded at another resolution --
        # it decodes whatever run lengths it finds against the canvas it is
        # given and returns a plausible-looking mask. Compressed RLE is
        # canonical, so a stream genuinely encoded at this size round-trips to
        # itself; anything else was encoded elsewhere and would be mis-scored.
        if mask_util.encode(np.asfortranarray(decoded))["counts"].decode("ascii") != counts:
            raise SubmissionError(
                f"RLE for {filament_id} is not a canonical {size}x{size} encoding -- "
                "it was probably encoded at a different resolution, which scores as garbage"
            )

        areas.append(area)

    stems = submission["filament_id"].str.rsplit("_", n=1).str[0]
    per_image = stems.value_counts()

    # Panoptic Quality is defined over a *panoptic* segmentation: every pixel
    # belongs to at most one segment. Mask R-CNN instances overlap freely, and
    # Kaggle rejects such a file outright rather than scoring it. Connected
    # components are disjoint by construction, which is why the classical
    # baseline was accepted and the detector run was not.
    overlapping_pairs = 0
    for stem, group in submission.groupby(stems):
        if len(group) < 2:
            continue
        objs = [
            {"size": [size, size], "counts": c.encode("ascii")}
            for c in group["segmentation_rle"]
        ]
        ious = np.asarray(mask_util.iou(objs, objs, [0] * len(objs))).reshape(
            len(objs), len(objs)
        )
        np.fill_diagonal(ious, 0.0)
        overlapping_pairs += int((np.triu(ious, 1) > 0).sum())

    if overlapping_pairs:
        raise SubmissionError(
            f"{overlapping_pairs} pairs of instances overlap. Panoptic Quality "
            "requires disjoint segments and Kaggle rejects overlapping "
            "submissions. Call resolve_overlaps() before writing."
        )

    if expected_stems is not None:
        missing = set(expected_stems) - set(stems)
        if missing:
            print(
                f"WARNING: {len(missing)} of the expected images have no predictions "
                f"(pure false negatives), e.g. {sorted(missing)[:3]}"
            )

    return SubmissionStats(
        rows=len(submission),
        images=int(per_image.size),
        min_area=int(min(areas)),
        median_instances_per_image=float(per_image.median()),
    )


def write(
    submission: pd.DataFrame,
    path: str | Path = "submission.csv",
    expected_stems: Iterable[str] | None = None,
) -> SubmissionStats:
    """Validate, then write. Never write an unvalidated submission."""
    stats = validate(submission, expected_stems)
    submission.to_csv(path, index=False)
    print(f"wrote {Path(path).name}: {stats}")
    return stats
