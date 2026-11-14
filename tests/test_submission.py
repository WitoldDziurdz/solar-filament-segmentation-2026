"""Submission validation must catch the failures the evaluator reports silently."""

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util

from filament.submission import (
    SubmissionError,
    build,
    drop_low_confidence,
    mask_to_rle,
    validate,
)

SIZE = 64


def rect(r0, c0, r1, c1, size=SIZE):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[r0:r1, c0:c1] = 1
    return mask_to_rle(mask)


def test_build_emits_one_row_per_instance():
    submission = build({"imgA": [rect(0, 0, 5, 5), rect(10, 10, 15, 15)], "imgB": [rect(0, 0, 5, 5)]})

    assert len(submission) == 3
    assert list(submission.columns) == ["filament_id", "segmentation_rle"]
    assert set(submission["filament_id"]) == {"imgA_0", "imgA_1", "imgB_0"}


def test_validate_accepts_a_healthy_submission():
    submission = build({"imgA": [rect(0, 0, 5, 5), rect(10, 10, 20, 20)], "imgB": [rect(0, 0, 5, 5)]})

    stats = validate(submission, size=SIZE)
    assert stats.rows == 3
    assert stats.images == 2
    assert stats.min_area == 25


def test_zero_area_instance_is_rejected():
    empty = mask_to_rle(np.zeros((SIZE, SIZE), dtype=np.uint8))
    submission = build({"imgA": [rect(0, 0, 5, 5), empty]})

    with pytest.raises(SubmissionError, match="zero-area"):
        validate(submission, size=SIZE)


def test_duplicate_ids_are_rejected():
    submission = pd.DataFrame(
        {"filament_id": ["imgA_0", "imgA_0"], "segmentation_rle": [rect(0, 0, 5, 5)] * 2}
    )

    with pytest.raises(SubmissionError, match="duplicate"):
        validate(submission, size=SIZE)


def test_empty_submission_is_rejected():
    with pytest.raises(SubmissionError, match="empty"):
        validate(pd.DataFrame(columns=["filament_id", "segmentation_rle"]), size=SIZE)


def test_wrong_columns_are_rejected():
    submission = pd.DataFrame({"id": ["a"], "rle": ["x"]})

    with pytest.raises(SubmissionError, match="columns"):
        validate(submission, size=SIZE)


def test_rle_encoded_at_the_wrong_resolution_is_caught():
    """A 512-encoded mask submitted as 2048 is the classic silent failure."""
    small = mask_to_rle(np.ones((16, 16), dtype=np.uint8))
    submission = pd.DataFrame({"filament_id": ["imgA_0"], "segmentation_rle": [small]})

    # Decoded against a larger canvas the counts no longer describe a valid mask,
    # or describe a different one; either way validation must not pass it as-is.
    with pytest.raises(SubmissionError):
        validate(submission, size=SIZE * 4)


def test_missing_images_warn_but_do_not_raise(capsys):
    submission = build({"imgA": [rect(0, 0, 5, 5)]})

    validate(submission, expected_stems=["imgA", "imgB", "imgC"], size=SIZE)
    assert "2 of the expected images have no predictions" in capsys.readouterr().out


def test_drop_low_confidence_keeps_only_confident_instances():
    scored = [("rleA", 0.9), ("rleB", 0.4), ("rleC", 0.1)]

    assert drop_low_confidence(scored, 0.5) == ["rleA"]
    assert drop_low_confidence(scored, 0.05) == ["rleA", "rleB", "rleC"]


# --------------------------------------------------------------------------
# Overlap handling -- the failure that reached Kaggle twice
# --------------------------------------------------------------------------

def test_overlapping_instances_are_rejected():
    """Kaggle errors on overlapping segments; validation must catch it first."""
    submission = build({"imgA": [rect(0, 0, 20, 20), rect(10, 10, 30, 30)]})

    with pytest.raises(SubmissionError, match="overlap"):
        validate(submission, size=SIZE)


def test_touching_but_disjoint_instances_are_accepted():
    submission = build({"imgA": [rect(0, 0, 10, 10), rect(10, 0, 20, 10)]})

    assert validate(submission, size=SIZE).rows == 2


def test_resolve_overlaps_makes_instances_disjoint():
    from filament.submission import resolve_overlaps

    overlapping = [rect(0, 0, 20, 20), rect(10, 10, 30, 30)]
    resolved = resolve_overlaps(overlapping, min_retained=0.0, size=SIZE)

    submission = build({"imgA": resolved})
    validate(submission, size=SIZE)  # must not raise


def test_resolve_overlaps_gives_contested_pixels_to_the_higher_score():
    from filament.submission import resolve_overlaps

    first, second = rect(0, 0, 20, 20), rect(10, 10, 30, 30)
    areas = [
        mask_util.area({"size": [SIZE, SIZE], "counts": r.encode("ascii")})
        for r in resolve_overlaps([first, second], scores=[0.9, 0.1], min_retained=0.0, size=SIZE)
    ]

    # The 0.9-scored instance keeps its full 400 px; the other loses the overlap.
    assert areas[0] == 400
    assert areas[1] == 400 - 100


def test_resolve_overlaps_drops_instances_gutted_by_the_overlap():
    from filament.submission import resolve_overlaps

    big = rect(0, 0, 30, 30)
    mostly_inside = rect(1, 1, 12, 12)   # 121 px, almost entirely within big

    resolved = resolve_overlaps([big, mostly_inside], scores=[0.9, 0.8], size=SIZE)
    assert len(resolved) == 1


def test_resolve_overlaps_preserves_caller_ordering():
    from filament.submission import resolve_overlaps

    rles = [rect(0, 0, 6, 6), rect(40, 40, 60, 60), rect(20, 20, 26, 26)]
    resolved = resolve_overlaps(rles, min_retained=0.0, size=SIZE)

    assert resolved == rles  # already disjoint, order unchanged


# --------------------------------------------------------------------------
# postprocess() -- raw model output to disjoint canonical RLE
# --------------------------------------------------------------------------

def dense_rect(r0, c0, r1, c1, size=SIZE):
    mask = np.zeros((size, size), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def test_postprocess_drops_instances_below_the_threshold():
    from filament.submission import postprocess

    instances = [(dense_rect(0, 0, 10, 10), 0.9), (dense_rect(30, 30, 40, 40), 0.2)]
    kept = postprocess(instances, score_threshold=0.5, size=SIZE)
    assert kept == [rect(0, 0, 10, 10)]
    assert len(postprocess(instances, score_threshold=None, size=SIZE)) == 2


def test_postprocess_resolves_overlaps_by_score_and_validates():
    from filament.submission import postprocess

    instances = [(dense_rect(0, 0, 20, 20), 0.3), (dense_rect(10, 10, 30, 30), 0.9)]
    kept = postprocess(instances, min_retained=0.0, size=SIZE)
    areas = [mask_util.area({"size": [SIZE, SIZE], "counts": r.encode("ascii")}) for r in kept]
    # The 0.9 instance keeps all 400 px; the 0.3 instance loses the 100 px it shared.
    assert areas == [300, 400]
    validate(build({"imgA": kept}), size=SIZE)


def test_postprocess_accepts_rle_strings_and_is_idempotent():
    from filament.submission import postprocess

    clean = [rect(0, 0, 6, 6), rect(40, 40, 60, 60)]
    assert postprocess([(r, 1.0) for r in clean], size=SIZE) == clean


def test_postprocess_rejects_masks_at_another_resolution():
    from filament.submission import postprocess

    with pytest.raises(SubmissionError, match="shape"):
        postprocess([(np.ones((SIZE // 2, SIZE // 2), dtype=bool), 0.9)], size=SIZE)


def test_postprocess_drops_empty_masks():
    from filament.submission import postprocess

    assert postprocess([(np.zeros((SIZE, SIZE), dtype=bool), 0.9)], size=SIZE) == []
