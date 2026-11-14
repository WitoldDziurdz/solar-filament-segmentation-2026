"""The other published configurations: "025" (YOLO11m-seg at 2048 px, the refiner at alpha 1.5) and
"032" (the default's settings with a detector trained on all 707 images).

What these tests hold the package to:

* each configuration's settings are the ones its evaluation run used, and they reach the detector and
  the refiner; "032" *is* the default's settings, with other detector weights;
* each detector file is verified like the default's (missing, truncated, Git LFS pointer, corrupted);
* choosing one configuration never touches another configuration's detector;
* the default configuration, "013", is what it was;
* ``training/train_detector.py --recipe 025`` selects that detector's training recipe.

Two tests compare with the "025" evaluation run and are skipped unless ``FILAMENT_SECOND_RUN_DIR``
names the folder that holds its ``run_report.json`` and ``validation_predictions.json``.
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from filament import pipeline
from filament.folds import REFERENCE_FINGERPRINT
from filament.io import LFS_POINTER_PREFIX, file_sha256, load_prediction_file, rle_to_mask
from filament.models import detector, refiner
from filament.models.detector import DETECTOR
from filament.models.refiner import REFINER, NativeResidualUNet

from .conftest import REPO, train_images_dir

SIZE = 2048
DEFAULT_DETECTOR = "detector_yolo11l_seg_1792.pt"
SECOND_DETECTOR = "detector_yolo11m_seg_2048.pt"
FULL_DETECTOR = "detector_yolo11l_seg_1792_full.pt"
REFINER_FILE = "refiner_native_residual.pt"
DETECTOR_FILES = {"013": DEFAULT_DETECTOR, "025": SECOND_DETECTOR, "032": FULL_DETECTOR}

# Each configuration's settings as its evaluation run recorded them, in the run report's terms
# (settings.detector, settings.refiner, selected_alpha). Copied by hand from the reports.
DEFAULT_SETTINGS = {
    "detector": {"imgsz": 1792, "conf_floor": 0.05, "nms_iou": 0.5, "max_det": 60,
                 "mask_threshold": 0.5, "confidence": 0.3},
    "refiner": {"tile": 512, "margin": 64, "batch": 4, "background_sigma": 15.0, "taper_floor": 0.05},
    "selected_alpha": 1.0,
}
SECOND_SETTINGS = {
    "detector": {"imgsz": 2048, "conf_floor": 0.05, "nms_iou": 0.0, "max_det": 60,
                 "mask_threshold": 0.5, "confidence": 0.3},
    "refiner": {"tile": 512, "margin": 64, "batch": 4, "background_sigma": 15.0, "taper_floor": 0.05},
    "selected_alpha": 1.5,
}
REFINER_SHA256 = "5204da7ce821e25fa43473f1bce25b299c7ba1f021c506e5c8df0670195d474b"
WEIGHTS = {
    "013": {"detector": "183a841307827f2167a8d3d981c14df9d1ccbce9147c6a7d60a3536af8bb6eed",
            "refiner": REFINER_SHA256},
    "025": {"detector": "e94255af30540e6cc11e89517c5d5217aacdf68e6caf4297d316a1617929ed44",
            "refiner": REFINER_SHA256},
    "032": {"detector": "6559dea88fc91ba64953fd36bf08dd8605babb25b18d373e48bfc97446220803",
            "refiner": REFINER_SHA256},
}
DETECTOR_SIZES = {"025": 45_638_134, "032": 56_192_097}


def settings_of(configuration):
    """A configuration in its run report's terms."""
    refiner_settings = dataclasses.asdict(configuration.refiner)
    alpha = refiner_settings.pop("alpha")
    return {"detector": dataclasses.asdict(configuration.detector), "refiner": refiner_settings,
            "selected_alpha": alpha}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, recorded", [("013", DEFAULT_SETTINGS), ("025", SECOND_SETTINGS)])
def test_each_configuration_replays_the_settings_its_run_chose(name, recorded):
    assert settings_of(pipeline.CONFIGURATIONS[name]) == recorded


def test_the_default_configuration_is_the_one_every_function_defaults_to():
    assert pipeline.DEFAULT_CONFIGURATION == "013"
    default = pipeline.CONFIGURATIONS["013"]
    assert default.detector is DETECTOR and default.refiner is REFINER


def test_the_full_data_configuration_is_the_default_with_other_detector_weights():
    full, default = pipeline.CONFIGURATIONS["032"], pipeline.CONFIGURATIONS["013"]
    assert full.detector is default.detector and full.refiner is default.refiner     # the same settings
    manifest = json.loads((pipeline.WEIGHTS_DIR / "manifest.json").read_text())
    full_files = {entry["role"]: entry for entry in manifest["configurations"]["032"]["files"]}
    default_files = {entry["role"]: entry for entry in manifest["files"]}
    assert full_files["refiner"] == default_files["refiner"]
    assert full_files["detector"]["name"] != default_files["detector"]["name"]
    assert full_files["detector"]["sha256"] != default_files["detector"]["sha256"]


def test_an_unknown_configuration_is_refused_before_any_file_is_read(tmp_path):
    nowhere = tmp_path / "no-weights-here"           # a lookup of the manifest would raise FileNotFoundError
    with pytest.raises(ValueError, match="'026'"):
        pipeline.load_pipeline(nowhere, device="cpu", configuration="026")
    with pytest.raises(ValueError, match="'026'"):
        pipeline.verify_manifest(nowhere, configuration="026")


# --------------------------------------------------------------------------
# Weights: which files each configuration verifies and loads
# --------------------------------------------------------------------------

def fake_weights(folder):
    """Stand-ins for the four published files, and a manifest in the published layout."""
    def entry(name, role, size, salt):
        path = folder / name
        path.write_bytes(bytes((7 * i + salt) % 251 + 1 for i in range(size)))     # no zero bytes
        return {"name": name, "role": role, "size": size, "sha256": file_sha256(path)}

    shared_refiner = entry(REFINER_FILE, "refiner", 200, 3)
    manifest = {"files": [entry(DEFAULT_DETECTOR, "detector", 300, 1), shared_refiner],
                "configurations": {
                    "025": {"files": [entry(SECOND_DETECTOR, "detector", 280, 2), shared_refiner]},
                    "032": {"files": [entry(FULL_DETECTOR, "detector", 290, 4), shared_refiner]}}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def loaded(monkeypatch):
    """Which weight files the pipeline builds its networks from, in order (no network is built)."""
    names = []
    monkeypatch.setattr(detector, "load_detector", lambda path: names.append(Path(path).name))
    monkeypatch.setattr(refiner, "load_refiner", lambda path, device: names.append(Path(path).name))
    return names


@pytest.mark.parametrize("name", ["025", "032"])
def test_the_published_manifest_lists_each_configuration_beside_the_unchanged_default(name):
    manifest = json.loads((pipeline.WEIGHTS_DIR / "manifest.json").read_text())
    default_files = {entry["role"]: entry for entry in manifest["files"]}
    listed = manifest["configurations"][name]["files"]
    files = {entry["role"]: entry for entry in listed}
    assert len(listed) == 2 and sorted(files) == ["detector", "refiner"]
    assert files["detector"]["name"] == DETECTOR_FILES[name]
    assert files["detector"]["size"] == DETECTOR_SIZES[name]
    assert files["detector"]["sha256"] == WEIGHTS[name]["detector"]
    assert set(files["detector"]) == set(default_files["detector"])            # the same entry format
    assert files["refiner"] == default_files["refiner"]                        # one refiner, listed again
    for entry in files.values():
        path = pipeline.WEIGHTS_DIR / entry["name"]
        assert path.stat().st_size == entry["size"] and file_sha256(path) == entry["sha256"]
    assert pipeline.verify_manifest(configuration=name) == {
        "detector": pipeline.WEIGHTS_DIR / DETECTOR_FILES[name], "refiner": pipeline.WEIGHTS_DIR / REFINER_FILE}


@pytest.mark.parametrize("name", ["025", "032"])
def test_each_configuration_loads_its_own_detector_and_the_shared_refiner(loaded, name):
    pipe = pipeline.load_pipeline(device="cpu", configuration=name)
    assert loaded == [DETECTOR_FILES[name], REFINER_FILE]
    assert pipe.configuration == name and pipe.weights == WEIGHTS[name]
    chosen = pipeline.CONFIGURATIONS[name]
    assert pipe.detector_config is chosen.detector and pipe.refiner_config is chosen.refiner


def test_the_second_configuration_runs_with_its_own_settings(loaded):
    pipe = pipeline.load_pipeline(device="cpu", configuration="025")
    assert settings_of(pipeline.Configuration("025", pipe.detector_config, pipe.refiner_config, "")) == SECOND_SETTINGS


def test_the_default_loads_what_it_always_loaded(loaded):
    for pipe in (pipeline.load_pipeline(device="cpu"), pipeline.load_pipeline(device="cpu", configuration="013")):
        assert pipe.configuration == "013" and pipe.weights == WEIGHTS["013"]
        assert pipe.detector_config is DETECTOR and pipe.refiner_config is REFINER
    assert loaded == [DEFAULT_DETECTOR, REFINER_FILE] * 2


@pytest.mark.parametrize("missing", ["013", "025", "032"])
def test_choosing_one_configuration_never_needs_another_detector(tmp_path, loaded, missing):
    fake_weights(tmp_path)
    (tmp_path / DETECTOR_FILES[missing]).unlink()
    for name in ("013", "025", "032"):
        if name == missing:
            with pytest.raises(FileNotFoundError, match=DETECTOR_FILES[name]):
                pipeline.load_pipeline(tmp_path, device="cpu", configuration=name)
        else:
            pipeline.load_pipeline(tmp_path, device="cpu", configuration=name)
    assert loaded == [file for name in ("013", "025", "032") if name != missing
                      for file in (DETECTOR_FILES[name], REFINER_FILE)]


def flip_one_byte(path):
    data = bytearray(path.read_bytes())
    data[100] ^= 0xFF
    path.write_bytes(bytes(data))


DAMAGE = [
    (lambda path: path.unlink(), FileNotFoundError, None),
    (lambda path: path.write_bytes(path.read_bytes()[:-1]), ValueError, "bytes"),
    (lambda path: path.write_bytes(LFS_POINTER_PREFIX + b"v1\noid sha256:" + b"0" * 64 + b"\nsize 280\n"),
     RuntimeError, "LFS"),
    (flip_one_byte, ValueError, "sha256"),
]


@pytest.mark.parametrize("name", ["025", "032"])
@pytest.mark.parametrize("damage, error, message", DAMAGE, ids=["missing", "truncated", "lfs-pointer", "corrupted"])
def test_a_damaged_detector_fails_loudly_before_anything_loads(tmp_path, loaded, name, damage, error, message):
    fake_weights(tmp_path)
    damage(tmp_path / DETECTOR_FILES[name])
    with pytest.raises(error, match=message or DETECTOR_FILES[name]):
        pipeline.load_pipeline(tmp_path, device="cpu", configuration=name)
    assert loaded == []
    pipeline.load_pipeline(tmp_path, device="cpu")                  # the default does not read that file
    assert loaded == [DEFAULT_DETECTOR, REFINER_FILE]


@pytest.mark.parametrize("name", ["025", "032"])
@pytest.mark.parametrize("listed", [None, {}, {"files": []}, {"files": [{"name": "detector.pt"}]}],
                         ids=["absent", "empty", "no-files", "incomplete"])
def test_a_manifest_without_complete_weights_refuses_that_configuration(tmp_path, loaded, name, listed):
    manifest = fake_weights(tmp_path)
    if listed is None:
        del manifest["configurations"][name]
    else:
        manifest["configurations"][name] = listed
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest.json"):          # the manifest's refusal, not a lookup
        pipeline.load_pipeline(tmp_path, device="cpu", configuration=name)
    assert loaded == []


@pytest.mark.parametrize("name", ["025", "032"])
def test_retrained_weights_load_the_default_and_refuse_the_other_configurations(tmp_path, loaded, name):
    trained = tmp_path / "trained"
    trained.mkdir()
    fake_weights(trained)
    weights = pipeline.write_manifest(tmp_path / "weights", {"detector": trained / DEFAULT_DETECTOR,
                                                             "refiner": trained / REFINER_FILE})
    with pytest.raises(ValueError, match=f"lists no weights for configuration '{name}'"):
        pipeline.load_pipeline(weights, device="cpu", configuration=name)
    pipeline.load_pipeline(weights, device="cpu")
    assert loaded == [DEFAULT_DETECTOR, REFINER_FILE]


@pytest.mark.parametrize("name, model_name, imgsz, epochs", [
    ("013", "yolo11l-seg.pt", 1792, 78),
    ("025", "yolo11m-seg.pt", 2048, 74),
    ("032", "yolo11l-seg.pt", 1792, 49),
])
def test_each_published_detector_was_trained_at_its_configurations_input_size(name, model_name, imgsz, epochs):
    model = detector.load_detector(pipeline.WEIGHTS_DIR / DETECTOR_FILES[name])   # one-class segmentation
    train_args = model.ckpt["train_args"]
    assert (train_args["model"], train_args["imgsz"], train_args["epochs"]) == (model_name, imgsz, epochs)
    assert pipeline.CONFIGURATIONS[name].detector.imgsz == imgsz


# --------------------------------------------------------------------------
# Inference: the chosen configuration's settings reach both networks
# --------------------------------------------------------------------------

class RecordingDetector:
    """Stands in for ``ultralytics.YOLO``: records each predict call's settings, returns two proposals."""

    def __init__(self):
        self.calls = []

    def predict(self, **arguments):
        self.calls.append({key: value for key, value in arguments.items() if key != "source"})
        masks = torch.zeros((2, SIZE, SIZE))
        masks[0, 1000:1040, 900:1300] = 1.0           # confidence 0.9: refined
        masks[1, 300:330, 300:700] = 1.0              # confidence 0.2: below the operating point
        return [SimpleNamespace(masks=SimpleNamespace(data=masks),
                                boxes=SimpleNamespace(conf=torch.tensor([0.9, 0.2])))]


def eroding_refiner():
    """A refiner whose correction is -1.8 everywhere: the proposal survives alpha 1.0 and vanishes at 1.5.

    Inside the proposal the refined logit is 2.197 - alpha * 1.8: +0.40 at alpha 1.0, -0.50 at alpha 1.5.
    """
    model = NativeResidualUNet(base_channels=8).eval()
    with torch.no_grad():
        model.delta_head.bias.fill_(-1.8)
    return model


@pytest.fixture
def stand_in_networks(monkeypatch):
    """Real weight checks and inference code around a recording detector and the eroding refiner."""
    built = []

    def load_detector(path):
        built.append(RecordingDetector())
        return built[-1]

    monkeypatch.setattr(detector, "load_detector", load_detector)
    monkeypatch.setattr(refiner, "load_refiner", lambda path, device: eroding_refiner())
    pipeline.default_pipeline.cache_clear()
    pipeline.published_pipeline.cache_clear()
    yield built
    pipeline.default_pipeline.cache_clear()           # never leave a stand-in in the process-wide caches
    pipeline.published_pipeline.cache_clear()


DEFAULT_CALL = {"imgsz": 1792, "conf": 0.05, "iou": 0.5, "max_det": 60, "retina_masks": True}


@pytest.mark.parametrize("configuration, settings, kept", [
    (None, DEFAULT_CALL, 1),
    ("013", DEFAULT_CALL, 1),
    ("025", {"imgsz": 2048, "conf": 0.05, "iou": 0.0, "max_det": 60, "retina_masks": True}, 0),
    ("032", DEFAULT_CALL, 1),
])
def test_predict_dataset_runs_the_chosen_configuration(stand_in_networks, tmp_path, configuration, settings, kept):
    cv2.imwrite(str(tmp_path / "a_stem.jpeg"), np.full((SIZE, SIZE), 120, dtype=np.uint8))
    predictions = pipeline.predict_dataset(tmp_path, configuration=configuration, log_every=0)
    (detector_model,) = stand_in_networks
    (call,) = detector_model.calls
    assert {key: call[key] for key in settings} == settings
    assert len(predictions["a_stem"]) == kept       # 0.2 is always cut; alpha 1.5 erases the 0.9 proposal


def test_predict_dataset_refuses_a_pipeline_of_another_configuration(stand_in_networks, tmp_path):
    default = pipeline.load_pipeline(device="cpu")
    with pytest.raises(ValueError, match="025"):
        pipeline.predict_dataset(tmp_path, pipe=default, configuration="025", log_every=0)


# --------------------------------------------------------------------------
# Training recipe
# --------------------------------------------------------------------------

def start_detector_training(tmp_path, *arguments):
    """Runs ``train_detector.py`` with no GPU visible, so that it stops at its GPU guard."""
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("FILAMENT_SMOKE", "FILAMENT_ALLOW_CPU")}
    environment.update(FILAMENT_TORCH_PATCHED="1", CUDA_VISIBLE_DEVICES="")   # no reinstall, no GPU
    return subprocess.run([sys.executable, str(REPO / "training" / "train_detector.py"),
                           "--data-root", str(tmp_path), "--output-dir", str(tmp_path / "run"), *arguments],
                          capture_output=True, text=True, env=environment, timeout=300)


@pytest.mark.parametrize("arguments, recipe", [
    ((), "recipe 013: yolo11l-seg.pt at imgsz 1792"),
    (("--recipe", "025"), "recipe 025: yolo11m-seg.pt at imgsz 2048"),
], ids=["default", "second"])
def test_the_training_recipe_option_picks_the_detector(tmp_path, arguments, recipe):
    done = start_detector_training(tmp_path, *arguments)
    assert recipe in done.stdout
    assert done.returncode == 1 and "ABORT" in done.stderr          # the GPU guard, before any training


def test_an_unknown_training_recipe_is_refused(tmp_path):
    done = start_detector_training(tmp_path, "--recipe", "026")
    assert done.returncode == 2 and "invalid choice" in done.stderr


# --------------------------------------------------------------------------
# Against the second configuration's evaluation run (opt-in)
# --------------------------------------------------------------------------

def second_run_dir():
    folder = os.environ.get("FILAMENT_SECOND_RUN_DIR")
    if not folder:
        pytest.skip("FILAMENT_SECOND_RUN_DIR is not set (the second configuration's evaluation run)")
    return Path(folder)


def test_the_second_configuration_equals_its_run_report():
    report = json.loads((second_run_dir() / "run_report.json").read_text())
    assert report["status"] == "complete" and report["fold_fingerprint"] == REFERENCE_FINGERPRINT
    assert settings_of(pipeline.CONFIGURATIONS["025"]) == {
        "detector": report["settings"]["detector"], "refiner": report["settings"]["refiner"],
        "selected_alpha": report["selected_alpha"]}
    assert {"detector": report["inputs"]["detector"]["sha256"],
            "refiner": report["inputs"]["refiner"]["sha256"]} == WEIGHTS["025"]


def test_the_second_configuration_reproduces_a_recorded_validation_image():
    """The recorded predictions were made on this kind of CPU; elsewhere a few boundary pixels may differ."""
    folder = second_run_dir()
    stem = "20111009110234Ch"                 # the report-half image the default's one-image test uses
    images = train_images_dir()
    if not (images / f"{stem}.jpeg").is_file():
        pytest.skip("training images not found")
    recorded, _ = load_prediction_file(folder / "validation_predictions.json")
    predicted = pipeline.predict_dataset(images, stems=[stem], configuration="025", log_every=0)[stem]
    assert len(predicted) == len(recorded[stem]) >= 5
    for ours, theirs in zip(predicted, recorded[stem]):
        a, b = rle_to_mask(ours), rle_to_mask(theirs)
        assert (a & b).sum() / (a | b).sum() > 0.98
