"""Folds must keep every annotator of one observation together, and be reproducible."""

import pandas as pd
import pytest

from filament.folds import (
    REFERENCE_FINGERPRINT,
    REFERENCE_REPORT_DIGEST,
    REFERENCE_TUNE_DIGEST,
    assert_no_leakage,
    fold_fingerprint,
    grouped_folds,
    stems_digest,
    validation_halves,
)

STEMS = [f"2014{i:04d}120000Bh" for i in range(40)]


def gt_frame(records):
    """Minimal ground-truth frame: one row per (record, instance)."""
    rows = [{"record_id": r, "stem": r.split("-", 1)[1]} for r in records for _ in range(2)]
    return pd.DataFrame(rows)


def test_grouped_folds_is_deterministic_and_covers_every_stem():
    first = grouped_folds(STEMS + STEMS[:5])  # duplicates, as in a ground-truth column
    second = grouped_folds(list(reversed(STEMS)))
    assert first == second
    assert set(first) == set(STEMS)
    sizes = pd.Series(first).value_counts()
    assert len(sizes) == 5 and sizes.max() - sizes.min() <= 1


def test_seed_changes_the_assignment():
    assert grouped_folds(STEMS, seed=1) != grouped_folds(STEMS, seed=2)


def test_fingerprint_detects_one_moved_stem():
    assignment = grouped_folds(STEMS)
    moved = dict(assignment)
    moved[STEMS[0]] = (moved[STEMS[0]] + 1) % 5
    assert fold_fingerprint(assignment) == fold_fingerprint(dict(reversed(list(assignment.items()))))
    assert fold_fingerprint(assignment) != fold_fingerprint(moved)


def test_stem_keyed_assignment_passes_the_leakage_check():
    gt = gt_frame([f"a{k}-{s}" for s in STEMS for k in range(2)])
    assert_no_leakage(grouped_folds(gt["stem"]), gt)


def test_record_keyed_split_that_separates_annotators_is_caught():
    """The naive split: two annotators of one image in different folds."""
    stem = STEMS[0]
    gt = gt_frame([f"a1-{stem}", f"a2-{stem}"])
    with pytest.raises(AssertionError, match="span multiple folds"):
        assert_no_leakage({f"a1-{stem}": 0, f"a2-{stem}": 1}, gt)


def test_unassigned_stems_are_an_error():
    gt = gt_frame([f"a1-{STEMS[0]}", f"a1-{STEMS[1]}"])
    with pytest.raises(AssertionError, match="no fold"):
        assert_no_leakage({STEMS[0]: 0}, gt)


def test_validation_halves_partition_the_fold():
    assignment = grouped_folds(STEMS)
    tune, report = validation_halves(assignment, fold=0)
    fold = sorted(s for s, f in assignment.items() if f == 0)
    assert sorted(tune + report) == fold
    assert not set(tune) & set(report)
    assert tune == fold[::2] and report == fold[1::2]


def test_stems_digest_ignores_order():
    assert stems_digest(STEMS) == stems_digest(reversed(STEMS))
    assert len(stems_digest(STEMS)) == 16


def test_reference_fold_is_reproduced_from_the_annotations(real_gt):
    """The fold every validation number in the report was measured on."""
    assignment = grouped_folds(real_gt["stem"])
    assert_no_leakage(assignment, real_gt)
    assert len(assignment) == 707
    assert fold_fingerprint(assignment) == REFERENCE_FINGERPRINT
    tune, report = validation_halves(assignment)
    assert len(tune) == len(report) == 71
    assert stems_digest(tune) == REFERENCE_TUNE_DIGEST
    assert stems_digest(report) == REFERENCE_REPORT_DIGEST
