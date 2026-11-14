"""The pipeline: weights are verified before anything loads, and the two stages chain as documented."""

import json
import shutil

import cv2
import numpy as np
import pytest
import torch

from filament import pipeline
from filament.io import LFS_POINTER_PREFIX, file_sha256, load_prediction_file, mask_to_rle, rle_to_mask
from filament.models import detector, refiner
from filament.models.detector import DETECTOR
from filament.models.refiner import REFINER, NativeResidualUNet

from .conftest import REPO, train_images_dir

SIZE = 2048


# --------------------------------------------------------------------------
# Weights and manifest
# --------------------------------------------------------------------------

def fake_weights(folder, sizes=(("detector", 300), ("refiner", 200))):
    """Two small stand-in weight files and a manifest that describes them."""
    entries = []
    for role, size in sizes:
        path = folder / f"{role}.pt"
        path.write_bytes(bytes((7 * i + len(role)) % 251 + 1 for i in range(size)))   # no zero bytes
        entries.append({"name": path.name, "role": role, "size": path.stat().st_size,
                        "sha256": file_sha256(path)})
    (folder / "manifest.json").write_text(json.dumps({"files": entries}))
    return entries


def test_the_published_manifest_matches_the_committed_weights():
    manifest = json.loads((pipeline.WEIGHTS_DIR / "manifest.json").read_text())
    assert sorted(entry["role"] for entry in manifest["files"]) == ["detector", "refiner"]
    for entry in manifest["files"]:
        path = pipeline.WEIGHTS_DIR / entry["name"]
        assert path.stat().st_size == entry["size"]
        assert file_sha256(path) == entry["sha256"]
    verified = pipeline.verify_manifest()
    assert verified == {"detector": pipeline.WEIGHTS_DIR / "detector_yolo11l_seg_1792.pt",
                        "refiner": pipeline.WEIGHTS_DIR / "refiner_native_residual.pt"}


def test_a_valid_manifest_verifies(tmp_path):
    fake_weights(tmp_path)
    assert pipeline.verify_manifest(tmp_path) == {"detector": tmp_path / "detector.pt",
                                                  "refiner": tmp_path / "refiner.pt"}


def test_a_missing_weight_file_fails_loudly(tmp_path):
    fake_weights(tmp_path)
    (tmp_path / "refiner.pt").unlink()
    with pytest.raises(FileNotFoundError, match="refiner.pt"):
        pipeline.verify_manifest(tmp_path)


def test_a_corrupted_weight_file_fails_loudly(tmp_path):
    fake_weights(tmp_path)
    path = tmp_path / "detector.pt"
    data = bytearray(path.read_bytes())
    data[100] ^= 0xFF                          # same size, one byte changed
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        pipeline.verify_manifest(tmp_path)


def test_a_truncated_weight_file_fails_on_its_size(tmp_path):
    fake_weights(tmp_path)
    path = tmp_path / "detector.pt"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="bytes"):
        pipeline.verify_manifest(tmp_path)


def test_a_git_lfs_pointer_fails_loudly(tmp_path):
    entries = fake_weights(tmp_path)
    pointer = LFS_POINTER_PREFIX + b"v1\noid sha256:" + b"0" * 64 + b"\nsize 300\n"
    (tmp_path / "detector.pt").write_bytes(pointer)
    entries[0]["size"] = len(pointer)          # a pointer can even match a hand-edited size
    (tmp_path / "manifest.json").write_text(json.dumps({"files": entries}))
    with pytest.raises(RuntimeError, match="LFS"):
        pipeline.verify_manifest(tmp_path)


@pytest.mark.parametrize("manifest", [{"files": []}, {}, {"files": [{"name": "detector.pt"}]}])
def test_an_incomplete_manifest_is_refused(tmp_path, manifest):
    fake_weights(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        pipeline.verify_manifest(tmp_path)


def test_retrained_weights_get_a_manifest_and_load_through_the_same_checks(tmp_path, monkeypatch):
    trained = tmp_path / "trained"
    trained.mkdir()
    fake_weights(trained)
    weights = pipeline.write_manifest(tmp_path / "weights", {"detector": trained / "detector.pt",
                                                             "refiner": trained / "refiner.pt"})
    assert pipeline.verify_manifest(weights) == {
        "detector": weights / "detector_yolo11l_seg_1792.pt",
        "refiner": weights / "refiner_native_residual.pt"}
    loaded = []
    monkeypatch.setattr(detector, "load_detector", lambda path: loaded.append(path.name))
    monkeypatch.setattr(refiner, "load_refiner", lambda path, device: loaded.append(path.name))
    pipeline.load_pipeline(weights, device="cpu")
    assert loaded == ["detector_yolo11l_seg_1792.pt", "refiner_native_residual.pt"]
    with pytest.raises(FileNotFoundError):
        pipeline.write_manifest(tmp_path / "other", {"detector": trained / "detector.pt",
                                                     "refiner": trained / "missing.pt"})


def test_no_manifest_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        pipeline.verify_manifest(tmp_path)


def test_weights_are_loaded_only_after_verification(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(detector, "load_detector", lambda *a, **k: calls.append("detector"))
    monkeypatch.setattr(refiner, "load_refiner", lambda *a, **k: calls.append("refiner"))
    fake_weights(tmp_path)
    path = tmp_path / "refiner.pt"
    path.write_bytes(b"\x00" * path.stat().st_size)     # right size, wrong bytes
    with pytest.raises(ValueError, match="sha256"):
        pipeline.load_pipeline(tmp_path, device="cpu")
    assert calls == []

    fake_weights(tmp_path)
    pipe = pipeline.load_pipeline(tmp_path, device="cpu")
    assert calls == ["detector", "refiner"]
    assert pipe.weights == {role: file_sha256(tmp_path / f"{role}.pt") for role in ("detector", "refiner")}


# --------------------------------------------------------------------------
# Chaining the two stages
# --------------------------------------------------------------------------

def box_rle(r0, c0, r1, c1):
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask_to_rle(mask)


PROPOSALS = [  # what the detector returns: best first, down to the 0.05 floor
    (box_rle(100, 100, 140, 600), 0.92),
    (box_rle(120, 500, 160, 900), 0.71),    # overlaps the first one
    (box_rle(1500, 300, 1530, 700), 0.30),  # exactly at the operating point: kept
    (box_rle(800, 800, 830, 1200), 0.29),   # below it: never refined
    (box_rle(1800, 1800, 1830, 1900), 0.06),
]


@pytest.fixture
def identity_pipeline(monkeypatch):
    """The real chaining code with a stand-in detector and an untrained (identity) refiner."""
    monkeypatch.setattr(detector, "propose", lambda model, image, config, device: list(PROPOSALS))
    return pipeline.Pipeline(detector=None, refiner=NativeResidualUNet(base_channels=8).eval(),
                             device=torch.device("cpu"), detector_config=DETECTOR,
                             refiner_config=REFINER, weights={})


def test_proposals_below_the_operating_point_are_not_refined(identity_pipeline):
    image = np.full((SIZE, SIZE), 120, dtype=np.uint8)
    instances = pipeline.predict_instances(image, identity_pipeline)
    assert instances == PROPOSALS[:3]       # identity refiner: masks unchanged, order kept


def test_predict_image_returns_native_boolean_masks_with_scores(identity_pipeline):
    image = np.full((SIZE, SIZE), 120, dtype=np.uint8)
    instances = pipeline.predict_image(image, identity_pipeline)
    assert [score for _, score in instances] == [0.92, 0.71, 0.30]
    for (mask, _), (rle, _) in zip(instances, PROPOSALS):
        assert mask.dtype == bool and mask.shape == (SIZE, SIZE)
        assert np.array_equal(mask, rle_to_mask(rle))


def test_predict_dataset_post_processes_every_image(identity_pipeline, tmp_path):
    for stem in ("b_stem", "a_stem"):
        cv2.imwrite(str(tmp_path / f"{stem}.jpeg"), np.full((SIZE, SIZE), 120, dtype=np.uint8))
    timings = {}
    predictions = pipeline.predict_dataset(tmp_path, pipe=identity_pipeline, timings=timings, log_every=0)
    assert list(predictions) == ["a_stem", "b_stem"] and set(timings) == set(predictions)
    for rles in predictions.values():
        masks = [rle_to_mask(rle) for rle in rles]
        assert len(masks) == 3
        assert not np.logical_and(masks[0], masks[1]).any()           # overlap resolved best-first
        assert np.array_equal(masks[0], rle_to_mask(PROPOSALS[0][0]))
    only_one = pipeline.predict_dataset(tmp_path, stems=["b_stem"], pipe=identity_pipeline, log_every=0)
    assert list(only_one) == ["b_stem"]


def test_predict_dataset_refuses_a_missing_image(identity_pipeline, tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.predict_dataset(tmp_path, stems=["nowhere"], pipe=identity_pipeline, log_every=0)


def test_comparing_two_prediction_sets_instance_by_instance():
    a, b, c = box_rle(0, 0, 100, 100), box_rle(500, 500, 600, 700), box_rle(1000, 0, 1010, 50)
    nudged = box_rle(0, 0, 100, 101)                     # a, one column wider
    ours = {"same": [a, b], "drift": [a, b], "lost": [a, c], "empty": []}
    theirs = {"same": [a, b], "drift": [nudged, b], "lost": [a], "empty": []}
    table = pipeline.compare_predictions(ours, theirs).set_index("stem")
    assert list(table.index) == ["drift", "empty", "lost", "same"]
    assert table.loc["same", "identical"] and table.loc["empty", "identical"]
    assert not table.loc["drift", "identical"] and table.loc["drift", "matched"] == 2
    assert table.loc["drift", "min_iou"] == pytest.approx(100 / 101)
    assert table.loc["drift", "pixels_differing"] == 100
    assert (table.loc["lost", ["ours", "theirs", "matched"]] == [2, 1, 1]).all()
    assert table.loc["lost", "pixels_differing"] == 500
    assert np.isnan(table.loc["empty", "mean_iou"])


def test_the_parameter_table_documents_every_stage():
    table = pipeline.parameters()
    assert {"stage", "parameter", "value"} <= set(table.columns)
    assert {"detector", "refiner", "post-processing"} <= set(table["stage"])
    values = dict(zip(table["parameter"], table["value"]))
    assert values["imgsz"] == 1792 and values["confidence"] == 0.3 and values["alpha"] == 1.0


# --------------------------------------------------------------------------
# The published weights, end to end
# --------------------------------------------------------------------------

def test_the_published_pipeline_runs_on_a_frame():
    """Loads both published networks on CPU and runs one synthetic frame through them."""
    pipe = pipeline.load_pipeline(device="cpu")
    assert pipe.device.type == "cpu"
    image = cv2.GaussianBlur(np.random.default_rng(1).integers(90, 160, (SIZE, SIZE), dtype=np.uint8),
                             (0, 0), 3)
    for mask, score in pipeline.predict_image(image, pipe):
        assert mask.dtype == bool and mask.shape == (SIZE, SIZE) and mask.any()
        assert DETECTOR.confidence <= score <= 1.0


def test_reproduces_the_recorded_validation_predictions_on_one_image():
    """Needs the competition's training images; skipped without them.

    The recorded predictions were made on a GPU in mixed precision. On another
    platform a few boundary pixels may differ, never whole instances.
    """
    stem = "20111009110234Ch"                 # first report-half image with 10+ predicted instances
    images = train_images_dir()
    if not (images / f"{stem}.jpeg").is_file():
        pytest.skip("training images not found")
    recorded, _ = load_prediction_file(REPO / "predictions" / "validation_yolo11l_residual.json")
    predicted = pipeline.predict_dataset(images, stems=[stem], log_every=0)[stem]
    assert len(predicted) == len(recorded[stem]) >= 10
    for ours, theirs in zip(predicted, recorded[stem]):
        a, b = rle_to_mask(ours), rle_to_mask(theirs)
        assert (a & b).sum() / (a | b).sum() > 0.98
