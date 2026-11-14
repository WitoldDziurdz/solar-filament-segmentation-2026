"""The optional training scripts, without training anything.

The scripts run as programs (importing train_detector.py starts training), so
these tests call ``--help`` in a subprocess and lift single functions out of
the source with ``ast`` to exercise them on synthetic input.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pycocotools import mask as mask_util

from .conftest import REPO

TRAINING = REPO / "training"


def lift(script, names, **namespace):
    """Execute only the named top-level functions of ``script`` in ``namespace``."""
    tree = ast.parse((TRAINING / script).read_text(encoding="utf-8"))
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert sorted(node.name for node in found) == sorted(names)
    namespace = {"np": np, "mask_util": mask_util, "Path": Path, "FULL_SIZE": 2048, **namespace}
    exec(compile(ast.Module(body=found, type_ignores=[]), script, "exec"), namespace)
    return namespace


def box(x0, y0, x1, y1):
    """A COCO polygon (one ring) for the rectangle [x0, x1] x [y0, y1]."""
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


@pytest.mark.parametrize("script, flags", [
    ("train_detector.py", ["--data-root", "--output-dir"]),
    ("train_refiner.py", ["--data-root", "--detector-run", "--output-dir"]),
])
def test_help_needs_no_gpu_and_names_every_argument(script, flags):
    env = {**os.environ, "FILAMENT_TORCH_PATCHED": "1"}      # never let a probe reinstall torch here
    done = subprocess.run([sys.executable, str(TRAINING / script), "--help"], capture_output=True,
                          text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    assert all(flag in done.stdout for flag in flags)
    assert "GPU" in done.stdout


def detector_functions():
    def to_rle_dict(counts, size=2048):
        payload = counts if isinstance(counts, bytes) else counts.encode("ascii")
        return {"size": [size, size], "counts": payload}

    return lift("train_detector.py", ["polygons_to_rle", "polygon_to_yolo_line", "deduplicate_polygons"],
                to_rle_dict=to_rle_dict)


def test_yolo_label_lines_are_normalised_and_clipped():
    f = detector_functions()
    line = f["polygon_to_yolo_line"]([[-4.0, 0.0, 2052.0, 0.0, 1024.0, 1024.0]])
    values = [float(v) for v in line.split()[1:]]
    assert line.startswith("0 ") and values == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
    assert f["polygon_to_yolo_line"]([[1.0, 2.0, 3.0, 4.0]]) is None          # fewer than 3 points


def test_annotator_union_is_deduplicated_by_complete_link():
    dedup = detector_functions()["deduplicate_polygons"]
    same = {"a1-s": [box(100, 100, 300, 140)], "a2-s": [box(102, 100, 300, 142)]}
    assert len(dedup(same)) == 1                                   # two experts, one filament
    apart = {"a1-s": [box(100, 100, 300, 140), box(900, 900, 1100, 940)]}
    assert len(dedup(apart)) == 2
    one_record = {"a1-s": [box(100, 100, 300, 140), box(101, 100, 300, 140)]}
    assert len(dedup(one_record)) == 2                             # at most one outline per record
    chain = {"a1-s": [box(0, 0, 100, 40)], "a2-s": [box(30, 0, 130, 40)], "a3-s": [box(60, 0, 160, 40)]}
    assert len(dedup(chain)) == 2                                  # a1 ~ a2 ~ a3, but a1 !~ a3


def rle(x0, y0, x1, y1):
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask_util.encode(np.asfortranarray(mask))["counts"].decode("ascii")


def refiner_functions(**namespace):
    def to_rle_dict(counts):
        return {"size": [2048, 2048], "counts": counts if isinstance(counts, bytes) else counts.encode("ascii")}

    return lift("train_refiner.py", ["file_sha256", "cluster_annotation_masks", "match_proposals", "find_inputs"],
                to_rle_dict=to_rle_dict, hashlib=__import__("hashlib"), **namespace)


def test_proposals_are_matched_to_annotator_clusters():
    f = refiner_functions()
    clusters = f["cluster_annotation_masks"]({"a1-s": [rle(100, 100, 300, 140)],
                                              "a2-s": [rle(102, 100, 300, 142)]})
    assert len(clusters) == 1 and len(clusters[0]) == 2
    proposals = {"rles": [rle(100, 100, 290, 140),        # a positive: every outline of the cluster is a target
                          rle(1500, 1500, 1600, 1540),    # touches nothing: a negative
                          rle(280, 100, 400, 140)],       # 0 < IoU <= 0.1: ambiguous, left out
                 "scores": [0.9, 0.4, 0.2]}
    matched = f["match_proposals"](proposals, clusters)
    assert [row["score"] for row in matched] == [0.9, 0.4]
    assert len(matched[0]["target_rles"]) == 2 and matched[1]["target_rles"] == []


def test_inputs_come_from_the_arguments_and_any_detector_run_is_accepted(tmp_path, capsys):
    data, run = tmp_path / "MAGFiLO_1.0_Kaggle_2026", tmp_path / "detector"
    (data / "train").mkdir(parents=True)
    (data / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps({}))
    hashes = {"run_report.json": "0" * 64, "validation_predictions.json": "1" * 64, "submission.csv": "2" * 64}
    f = refiner_functions(ARGS=SimpleNamespace(data_root=str(data), detector_run=str(run)),
                          SOURCE_HASHES=hashes, WEIGHTS_SHA256="3" * 64)
    with pytest.raises(FileNotFoundError, match="run_report.json"):
        f["find_inputs"]()
    (run / "runs" / "seg" / "weights").mkdir(parents=True)
    for name in hashes:
        (run / name).write_text("{}")
    with pytest.raises(FileNotFoundError, match="last.pt"):
        f["find_inputs"]()
    (run / "runs" / "seg" / "weights" / "last.pt").write_bytes(b"weights")
    annotations, root, source, weights = f["find_inputs"]()
    assert root == data and source == run and weights.name == "last.pt"
    assert "not the published one" in capsys.readouterr().out
