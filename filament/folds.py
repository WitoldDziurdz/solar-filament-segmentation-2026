"""Leakage-free cross-validation folds, grouped on the physical observation.

About 1154 COCO image records cover only ~707 physical solar observations,
because up to three experts annotated the same image. A record id is
``<annotator>-<stem>``; splitting on it puts the same Sun image in train and
validation under two different annotators, which inflates local scores and
makes them useless for model selection. Folds here are assigned per stem.

Every validation number of this project comes from one fold assignment:
five folds over the 707 training stems, seed ``20260816``, fold 0 held out
(142 stems), whose SHA-256 fingerprint is :data:`REFERENCE_FINGERPRINT`.
Fold 0 is split once more into two halves (:func:`validation_halves`):
thresholds and other choices are made on the *tune* half only, and the
*report* half is scored once with the choice frozen.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_SEED = 20260816
N_SPLITS = 5
VALIDATION_FOLD = 0

# grouped_folds(<the 707 training stems>, n_splits=5, seed=20260816)
REFERENCE_FINGERPRINT = "b990dbd4719315540906c0389c5edc44d408f7c853f79b1cfcc711f625d7bdee"
# stems_digest() of the two halves of fold 0 under that assignment (71 stems each)
REFERENCE_TUNE_DIGEST = "18cecf114adf8ee0"
REFERENCE_REPORT_DIGEST = "0c77b38a0f82bcce"


def grouped_folds(
    stems: Iterable[str],
    n_splits: int = N_SPLITS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, int]:
    """Assign each physical observation stem to a fold.

    Grouping is on the stem, so every annotator's version of one observation
    lands in the same fold. Deterministic given ``seed``.
    """
    unique = sorted(set(stems))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    return {unique[position]: index % n_splits for index, position in enumerate(order)}


def fold_fingerprint(assignment: Dict[str, int]) -> str:
    """Stable hash of a fold assignment, so runs can be proven comparable."""
    payload = ";".join(f"{stem}:{fold}" for stem, fold in sorted(assignment.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_no_leakage(assignment: Dict[str, int], gt: pd.DataFrame) -> None:
    """Fail loudly if one physical observation spans two folds.

    ``assignment`` may be keyed by physical stem (what :func:`grouped_folds`
    returns) or by COCO record id (``<annotator>-<stem>``, the naive split this
    guards against). Rows of ``gt`` that no fold covers are an error too.
    """
    keys = gt["stem"]
    if "record_id" in gt.columns and set(assignment) & set(gt["record_id"]):
        keys = gt["record_id"]
    folds = keys.map(assignment)
    if folds.isna().any():
        missing = sorted(set(keys[folds.isna()]))
        raise AssertionError(f"{len(missing)} keys have no fold, e.g. {missing[:5]}")

    folds_per_stem = gt.assign(fold=folds).groupby("stem")["fold"].nunique()
    offenders = folds_per_stem[folds_per_stem > 1]
    if len(offenders) > 0:
        raise AssertionError(
            f"{len(offenders)} physical observations span multiple folds: "
            f"{list(offenders.index[:5])}"
        )


def validation_halves(
    assignment: Dict[str, int],
    fold: int = VALIDATION_FOLD,
) -> Tuple[List[str], List[str]]:
    """Split one fold's stems into ``(tune, report)`` halves.

    The fold's stems are sorted and dealt alternately: ``tune = stems[::2]``,
    ``report = stems[1::2]``. Choosing a threshold as the maximum over a grid on
    the same stems it is reported on overstated PQ by 0.05-0.07 on this data, so
    every choice is made on ``tune`` and ``report`` is scored once.
    """
    stems = sorted(stem for stem, assigned in assignment.items() if assigned == fold)
    return stems[::2], stems[1::2]


def stems_digest(stems: Iterable[str]) -> str:
    """First 16 hex digits of SHA-256 over ``";".join(sorted(stems))``."""
    payload = ";".join(sorted(stems))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
