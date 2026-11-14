"""The configurations with the fine-tuned refiner: "013ft" ("013" with it) and "035" ("032" with it).

What these tests hold the package to:

* "013ft" and "035" *are* the default's settings (the same objects, refiner alpha 1.0); only the
  refiner weights differ from "013" and "032";
* the fine-tuned refiner is published like every other file: listed in the manifest with its size
  and SHA-256, verified before anything loads (missing, truncated, Git LFS pointer, corrupted), and
  loaded strictly with ``weights_only=True``; the settings stored beside its weights are the ones the
  package refines with (no train/serve skew), and its weights are not the starting point's;
* choosing a configuration never reads another configuration's detector or refiner file;
* the manifest entries of "013", "025" and "032" are exactly what they were;
* notebook stage 7 knows all five configurations: "013ft" is compared with its recorded report half,
  "035" is not scored (its detector was trained on every validation image);
* ``training/finetune_refiner.py`` is the fine-tune that produced the published file: its training
  code hashes to the value stored inside that file, every function it lifts from the fine-tune run's
  script is unchanged, and its recipe constants are the ones the file says it was trained with.

Two tests are opt-in: one compares with the fine-tune's evaluation run (``FILAMENT_FT_RUN_DIR``, the
folder holding its ``run_report.json``), one with the fine-tune run's own script
(``FILAMENT_FT_SOURCE``, its path).
"""

import ast
import dataclasses
import hashlib
import importlib.util
import inspect
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
from filament.io import LFS_POINTER_PREFIX, file_sha256
from filament.models import detector, refiner
from filament.models.detector import DETECTOR
from filament.models.refiner import PRIOR_LOGIT, REFINER, NativeResidualUNet

from .conftest import REPO

DEFAULT_DETECTOR = "detector_yolo11l_seg_1792.pt"
SECOND_DETECTOR = "detector_yolo11m_seg_2048.pt"
FULL_DETECTOR = "detector_yolo11l_seg_1792_full.pt"
REFINER_FILE = "refiner_native_residual.pt"
FT_REFINER_FILE = "refiner_native_residual_ft.pt"
DETECTOR_FILES = {"013": DEFAULT_DETECTOR, "025": SECOND_DETECTOR, "032": FULL_DETECTOR,
                  "013ft": DEFAULT_DETECTOR, "035": FULL_DETECTOR}
REFINER_FILES = {"013": REFINER_FILE, "025": REFINER_FILE, "032": REFINER_FILE,
                 "013ft": FT_REFINER_FILE, "035": FT_REFINER_FILE}
ALL_CONFIGURATIONS = ("013", "025", "032", "013ft", "035")

# The fine-tuned refiner: the fine-tune's last checkpoint, published as a byte copy.
FT_REFINER_SIZE = 4_403_970
FT_REFINER_SHA256 = "0a8eb466be0e5aa2d07eb662e14538cb6a2c447d12d069614c4f26c5ebb10774"
REFINER_SHA256 = "5204da7ce821e25fa43473f1bce25b299c7ba1f021c506e5c8df0670195d474b"
DETECTOR_SHA256 = {"013": "183a841307827f2167a8d3d981c14df9d1ccbce9147c6a7d60a3536af8bb6eed",
                   "032": "6559dea88fc91ba64953fd36bf08dd8605babb25b18d373e48bfc97446220803"}
WEIGHTS = {"013ft": {"detector": DETECTOR_SHA256["013"], "refiner": FT_REFINER_SHA256},
           "035": {"detector": DETECTOR_SHA256["032"], "refiner": FT_REFINER_SHA256}}

# The settings "013ft" was evaluated with, as the fine-tune's evaluation run recorded them for the
# default detector (settings.detector, settings.refiner, selected_alpha; copied by hand). "035" is
# "032"'s pipeline with the same refiner at the same alpha: no held-out image exists to choose another.
FT_SETTINGS = {
    "detector": {"imgsz": 1792, "conf_floor": 0.05, "nms_iou": 0.5, "max_det": 60,
                 "mask_threshold": 0.5, "confidence": 0.3},
    "refiner": {"tile": 512, "margin": 64, "batch": 4, "background_sigma": 15.0, "taper_floor": 0.05},
    "selected_alpha": 1.0,
}
# The recorded report half of "013ft" (fold 0, 71 images; alpha chosen on the tune half).
FT_REPORT_HALF = {"pq": 0.434551, "tp": 605, "fp": 305, "fn": 386}

# sha256 of json.dumps(block, sort_keys=True, separators=(",", ":")) of each manifest block as it was
# before the fine-tuned configurations were added.
UNCHANGED_BLOCKS = {
    ("files",): "b4e9f742c10b0ab825cbb1142a46932a92acde60d37cfc8c7a909345eb62f857",
    ("configurations", "025"): "541ecbab92d52f88fa9c5c9fbfba9dfda28b63f521c8b19e7b951f93d00337a2",
    ("configurations", "032"): "cda8bff1cb5142139411849aba77b765aef4eb13d804b1e6c0b8590dc591e93d",
}


def settings_of(configuration):
    """A configuration in its run report's terms."""
    refiner_settings = dataclasses.asdict(configuration.refiner)
    alpha = refiner_settings.pop("alpha")
    return {"detector": dataclasses.asdict(configuration.detector), "refiner": refiner_settings,
            "selected_alpha": alpha}


def published_manifest():
    return json.loads((pipeline.WEIGHTS_DIR / "manifest.json").read_text(encoding="utf-8"))


def listed_files(manifest, name):
    """``{role: entry}`` of one configuration in a manifest."""
    entries = manifest["files"] if name == "013" else manifest["configurations"][name]["files"]
    return {entry["role"]: entry for entry in entries}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

def test_five_configurations_are_published_and_the_default_is_unchanged():
    assert sorted(pipeline.CONFIGURATIONS) == sorted(ALL_CONFIGURATIONS)
    assert pipeline.DEFAULT_CONFIGURATION == "013"
    for name, configuration in pipeline.CONFIGURATIONS.items():
        assert configuration.name == name


@pytest.mark.parametrize("name", ["013ft", "035"])
def test_the_fine_tuned_configurations_are_the_defaults_settings(name):
    chosen = pipeline.CONFIGURATIONS[name]
    assert chosen.detector is DETECTOR and chosen.refiner is REFINER          # the same objects
    assert settings_of(chosen) == FT_SETTINGS


# --------------------------------------------------------------------------
# Weights: the manifest, the published file
# --------------------------------------------------------------------------

def test_the_manifest_entries_of_the_other_configurations_are_unchanged():
    manifest = published_manifest()
    for path, digest in UNCHANGED_BLOCKS.items():
        block = manifest
        for key in path:
            block = block[key]
        assert hashlib.sha256(json.dumps(block, sort_keys=True, separators=(",", ":")).encode()).hexdigest() \
            == digest, path


@pytest.mark.parametrize("name, paired_with", [("013ft", "013"), ("035", "032")])
def test_the_manifest_pairs_the_fine_tuned_refiner_with_the_right_detector(name, paired_with):
    manifest = published_manifest()
    listed = manifest["configurations"][name]["files"]
    files, other = {entry["role"]: entry for entry in listed}, listed_files(manifest, paired_with)
    assert len(listed) == 2 and sorted(files) == ["detector", "refiner"]
    assert files["detector"] == other["detector"]                     # the detector entry, verbatim
    ft = files["refiner"]
    assert (ft["name"], ft["size"], ft["sha256"]) == (FT_REFINER_FILE, FT_REFINER_SIZE, FT_REFINER_SHA256)
    assert set(ft) == set(other["refiner"])                            # the existing entry format
    assert manifest["configurations"][name].get("description")
    for entry in files.values():
        path = pipeline.WEIGHTS_DIR / entry["name"]
        assert path.stat().st_size == entry["size"] and file_sha256(path) == entry["sha256"]
    assert pipeline.verify_manifest(configuration=name) == {
        "detector": pipeline.WEIGHTS_DIR / DETECTOR_FILES[name], "refiner": pipeline.WEIGHTS_DIR / FT_REFINER_FILE}


def test_both_fine_tuned_configurations_list_the_same_refiner_entry():
    manifest = published_manifest()
    assert listed_files(manifest, "013ft")["refiner"] == listed_files(manifest, "035")["refiner"]


def fine_tuned_payload():
    return torch.load(pipeline.WEIGHTS_DIR / FT_REFINER_FILE, map_location="cpu", weights_only=True)


def stored_config(payload):
    """The training config stored beside the weights: one extra key next to NativeResidualUNet.checkpoint()'s."""
    extra = sorted(set(payload) - {"format_version", "model_type", "model_config", "model"})
    assert len(extra) == 1 and isinstance(payload[extra[0]], dict)
    return payload[extra[0]]


def test_the_fine_tuned_refiner_loads_strictly_as_the_published_architecture():
    model = refiner.load_refiner(pipeline.WEIGHTS_DIR / FT_REFINER_FILE, "cpu")
    assert isinstance(model, NativeResidualUNet) and model.base_channels == 24
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_096_825
    payload = fine_tuned_payload()
    assert set(payload["model"]) == set(model.state_dict())
    assert all(tensor.dtype == torch.float32 for tensor in payload["model"].values())


def test_the_fine_tuned_refiner_was_trained_with_the_settings_it_refines_with():
    config = stored_config(fine_tuned_payload())
    serving = {key: value for key, value in dataclasses.asdict(REFINER).items() if key != "alpha"}
    assert config["inference"] == serving                               # no train/serve skew
    assert config["prior_logit"] == PRIOR_LOGIT
    assert config["model_config"] == {"base_channels": 24}
    training = config["training"]
    assert training["steps"] == training["completed_steps"] == 6000     # a finished run, not a partial save
    assert training["init"]["sha256"] == REFINER_SHA256                 # fine-tuned from the published refiner
    assert training["tile"] == REFINER.tile


def test_the_fine_tuned_weights_are_not_the_starting_points():
    tuned = fine_tuned_payload()["model"]
    start = torch.load(pipeline.WEIGHTS_DIR / REFINER_FILE, map_location="cpu", weights_only=True)["model"]
    assert set(tuned) == set(start)
    assert sum(not torch.equal(tuned[name], start[name]) for name in start) >= len(start) // 2


# --------------------------------------------------------------------------
# Loading: which files each configuration reads
# --------------------------------------------------------------------------

def fake_weights(folder):
    """Stand-ins for the five published files, and a manifest in the published layout."""
    def entry(name, role, size, salt):
        path = folder / name
        path.write_bytes(bytes((7 * i + salt) % 251 + 1 for i in range(size)))     # no zero bytes
        return {"name": name, "role": role, "size": size, "sha256": file_sha256(path)}

    shared, tuned = entry(REFINER_FILE, "refiner", 200, 3), entry(FT_REFINER_FILE, "refiner", 210, 5)
    default, full = entry(DEFAULT_DETECTOR, "detector", 300, 1), entry(FULL_DETECTOR, "detector", 290, 4)
    manifest = {"files": [default, shared],
                "configurations": {
                    "025": {"files": [entry(SECOND_DETECTOR, "detector", 280, 2), shared]},
                    "032": {"files": [full, shared]},
                    "013ft": {"files": [default, tuned]},
                    "035": {"files": [full, tuned]}}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def loaded(monkeypatch):
    """Which weight files the pipeline builds its networks from, in order (no network is built)."""
    names = []
    monkeypatch.setattr(detector, "load_detector", lambda path: names.append(Path(path).name))
    monkeypatch.setattr(refiner, "load_refiner", lambda path, device: names.append(Path(path).name))
    return names


@pytest.mark.parametrize("name", ["013ft", "035"])
def test_each_fine_tuned_configuration_loads_its_detector_and_the_fine_tuned_refiner(loaded, name):
    pipe = pipeline.load_pipeline(device="cpu", configuration=name)
    assert loaded == [DETECTOR_FILES[name], FT_REFINER_FILE]
    assert pipe.configuration == name and pipe.weights == WEIGHTS[name]
    assert pipe.detector_config is DETECTOR and pipe.refiner_config is REFINER


@pytest.mark.parametrize("missing", [FT_REFINER_FILE, REFINER_FILE])
def test_choosing_a_configuration_never_needs_another_refiner(tmp_path, loaded, missing):
    fake_weights(tmp_path)
    (tmp_path / missing).unlink()
    for name in ALL_CONFIGURATIONS:
        if REFINER_FILES[name] == missing:
            with pytest.raises(FileNotFoundError, match=missing):
                pipeline.load_pipeline(tmp_path, device="cpu", configuration=name)
        else:
            pipeline.load_pipeline(tmp_path, device="cpu", configuration=name)
    assert loaded == [file for name in ALL_CONFIGURATIONS if REFINER_FILES[name] != missing
                      for file in (DETECTOR_FILES[name], REFINER_FILES[name])]


def flip_one_byte(path):
    data = bytearray(path.read_bytes())
    data[100] ^= 0xFF
    path.write_bytes(bytes(data))


DAMAGE = [
    (lambda path: path.unlink(), FileNotFoundError, None),
    (lambda path: path.write_bytes(path.read_bytes()[:-1]), ValueError, "bytes"),
    (lambda path: path.write_bytes(LFS_POINTER_PREFIX + b"v1\noid sha256:" + b"0" * 64 + b"\nsize 210\n"),
     RuntimeError, "LFS"),
    (flip_one_byte, ValueError, "sha256"),
]


@pytest.mark.parametrize("name", ["013ft", "035"])
@pytest.mark.parametrize("damage, error, message", DAMAGE, ids=["missing", "truncated", "lfs-pointer", "corrupted"])
def test_a_damaged_fine_tuned_refiner_fails_loudly_before_anything_loads(tmp_path, loaded, name, damage, error,
                                                                        message):
    fake_weights(tmp_path)
    damage(tmp_path / FT_REFINER_FILE)
    with pytest.raises(error, match=message or FT_REFINER_FILE):
        pipeline.load_pipeline(tmp_path, device="cpu", configuration=name)
    assert loaded == []
    pipeline.load_pipeline(tmp_path, device="cpu")                  # the default does not read that file
    assert loaded == [DEFAULT_DETECTOR, REFINER_FILE]


@pytest.mark.parametrize("name", ["013ft", "035"])
def test_retrained_weights_refuse_the_fine_tuned_configurations(tmp_path, loaded, name):
    trained = tmp_path / "trained"
    trained.mkdir()
    fake_weights(trained)
    weights = pipeline.write_manifest(tmp_path / "weights", {"detector": trained / DEFAULT_DETECTOR,
                                                             "refiner": trained / REFINER_FILE})
    with pytest.raises(ValueError, match=f"lists no weights for configuration '{name}'"):
        pipeline.load_pipeline(weights, device="cpu", configuration=name)
    assert loaded == []


# --------------------------------------------------------------------------
# Notebook stage 7
# --------------------------------------------------------------------------

def notebook_cell(cell_id):
    notebook = json.loads((REPO / "notebooks" / "pipeline.ipynb").read_text(encoding="utf-8"))
    (cell,) = [cell for cell in notebook["cells"] if cell.get("id") == cell_id]
    return "".join(cell["source"])


def recorded_scores():
    """``VALIDATION_RECORDED`` of the notebook's stage 2, read from its source."""
    tree = ast.parse(notebook_cell("e3d24050"))
    (value,) = [node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "VALIDATION_RECORDED"
                        for target in node.targets)]
    return ast.literal_eval(value)


def test_notebook_stage_7_knows_every_configuration():
    recorded = recorded_scores()
    assert sorted(recorded) == sorted(pipeline.CONFIGURATIONS)
    assert recorded["013ft"] == FT_REPORT_HALF                     # compared with its recorded report half
    assert recorded["035"] is None and recorded["032"] is None      # in-sample: not scored
    assert recorded["013"] == {"pq": 0.4312, "tp": 602, "fp": 308, "fn": 389}
    assert recorded["025"] == {"pq": 0.427543, "tp": 566, "fp": 236, "fn": 425}
    assert "VALIDATION_RECORDED[pipe.configuration]" in notebook_cell("77b4cbde")


# --------------------------------------------------------------------------
# Against the fine-tune's evaluation run (opt-in)
# --------------------------------------------------------------------------

def ft_run_dir():
    folder = os.environ.get("FILAMENT_FT_RUN_DIR")
    if not folder:
        pytest.skip("FILAMENT_FT_RUN_DIR is not set (the fine-tune's evaluation run on the default detector)")
    return Path(folder)


def test_the_fine_tuned_configuration_equals_its_evaluation_run():
    report = json.loads((ft_run_dir() / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete" and not report["dry_run"]
    assert report["fold_fingerprint"] == REFERENCE_FINGERPRINT
    assert settings_of(pipeline.CONFIGURATIONS["013ft"]) == {
        "detector": report["settings"]["detector"], "refiner": report["settings"]["refiner"],
        "selected_alpha": report["selected_alpha"]} == FT_SETTINGS
    assert {"detector": report["inputs"]["detector"]["sha256"],
            "refiner": report["inputs"]["refiner"]["sha256"]} == WEIGHTS["013ft"]
    half = report["report_half"]
    assert {key: half[key] for key in FT_REPORT_HALF} == FT_REPORT_HALF


# --------------------------------------------------------------------------
# training/finetune_refiner.py: the fine-tune that produced the published file
# --------------------------------------------------------------------------

FINETUNE = REPO / "training" / "finetune_refiner.py"

# sha256 of inspect.getsource() of every function and class the script takes unchanged from the script of the
# run that fine-tuned the published file. The objects of the training loop are also bound to the file itself:
# the file stores the sha256 of their source (test_the_training_code_is_the_code_that_trained_the_published_file).
LIFTED = {
    "training_ground_truth": "c9a9f43e3362fd605fbf60e9775eb1669ccd4acf001ed17e0f11893180e9baf7",
    "build_examples": "72e5c68558d3f3d2181de8b38df634b2d1fe36e406af7c90f7b98c83640ad1c0",
    "check_examples": "eef3a1cf8925d28ef9fadd9a99e42525c03c8eb93450108cce304dfd4754c2dc",
    "pool_of": "90870fb9465bd5887877243aa7f63cc428ab18fe92a31291efbe83e40928f06e",
    "item_seed": "e26b7916112c45064a774cb799608563db8cb088feb26e45c015a956d500f427",
    "MixedCropDataset": "2040f65246f7a79f838485cd697df89f97ff00030f3a1b2f0f72e3a8427b7ea3",
    "make_pools": "f962b2265d94a363f58844482ec5a09769677502519bfb6f0f6f3fb273bca9ff",
    "assert_mps_ready": "e18541699ae902cac0d5f4a42c28745530cfd38d54e0a4e96211d95761778eb4",
    "assert_on_device": "8f3d2db9a9af5b41035f5ee1855efc36935852888dd3319c55d5620bd3fce5f8",
    "StepGuard": "54866eeb50480c6093e61052a121ba1d4a9e7795f4d3d1f49af94dd32680c434",
    "check_finite": "5ce63cff8526cea7e1ecc3dde9c920a9585867223844d17e06f23d5410513648",
    "check_state_loaded": "e4bcc9250d413ac67386e6d02e144253f92521f714d8a7ad865e169be5153258",
    "load_refiner_checkpoint": "11e06b97d49313e83ab98557ff024107d03061903b1b692a77c82d2b81b3f53b",
    "plain": "c0a5d27b5d66e641bfc81a606405d1ee249f2b7ee2cd1d9932014d49fcde911e",
    "rows_digest": "d6cbb9a47a3a0d6a7dbcd9d8a3763f51ccff85b4b49df2a74af2f1910043d403",
    "save_torch_atomic": "9f76ac26c6608c979eadb088a4378c65d887f72fc6131a8993271ce3e93457a6",
    "synchronize": "1cc37512e905c96375df1fb0a1a78638c54480da98e76a735ef590a54cc592a8",
    "inference_settings": "e5285ebbe8c4957530481d6ff71f40fcad28daa0041a65a9a49ff1964ba2106a",
    "checkpoint_config": "38b4c47f4acf322444319b873cdc4984e1cbb3f47e11ef14048ae8e1f8054f2c",
    "save_checkpoint": "73e9bd210011e5c9efa4545404207189a24c582221710e9fdf3758d1caf58660",
    "TrainSettings": "c4d1116321ef4ab13c29dbbe57ab2d2db0160d6d39a11edc27f3a94fe51f9620",
    "lr_at": "8a05077b3e18921d1b02ec5aa5c301ce6b7a94c7b69aea4832ae3de3b43a89c7",
    "make_loader": "11f8979d8aa4a858486d6ae59d48980b21cb9c3de70a8a2b4719660576b1f1af",
    "optimizer_state_cpu": "c737239591a771895e06c8d1bd40b26e3e4c56eef5fdf5c656ff18f7fbd4a0b2",
    "save_state": "e26fe6969986d6abff00967db5b36377179b7fb98da8d353bd39c68661c12d7e",
    "write_history": "c39c9d95edc35a772668132a8fef2d5c727b23ae5bc92a9cc452af68b7b79350",
    "mps_warmup": "f890ad3a5625a84073a856fcfdd0daad350b767a9882bc59f622288fb54dded4",
    "summary_of": "20957e891137dce8a7bcd0df1903a807226c0a6d902a8a8629c87a89f58165e1",
    "train": "be4db03b4a5cf9c9489cc50b53a373635bf40ebbc9d605afe27b2d1d0372bbdd",
    "summarise_training": "00911740dd4f30347200a4f05dcf76aa54f8c83d6045354e2be787fa12f33a2a",
    "training_code_sha256": "1361450e4e204725a0cd7625c3c800b6465cda54c6889d40785db8bba18be687",
    "Clock": "f107fb24a27f2c4a99dda1f84476ce6239054981d5ae31f17f4d124083a2616a",
}


def load_script(path, name):
    """A script as a module (registered, so that DataLoader workers could unpickle from it); nothing runs."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def finetune():
    """``training/finetune_refiner.py`` as a module: its __main__ block (arguments, torch bootstrap) does not run."""
    yield load_script(FINETUNE, "finetune_refiner_under_test")
    sys.modules.pop("finetune_refiner_under_test", None)


def test_the_fine_tune_help_needs_no_gpu_and_names_every_argument():
    env = {**os.environ, "FILAMENT_TORCH_PATCHED": "1"}      # never let a probe reinstall torch here
    done = subprocess.run([sys.executable, str(FINETUNE), "--help"], capture_output=True, text=True, env=env,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    for flag in ("--data-root", "--refiner-run", "--output-dir", "--detector", "--init-refiner", "--max-hours"):
        assert flag in done.stdout
    assert "GPU" in done.stdout and "MPS" in done.stdout


def test_the_training_code_is_the_code_that_trained_the_published_file(finetune):
    key = stored_config(fine_tuned_payload())["key"]
    assert finetune.training_code_sha256() == key["training_code_sha256"]
    assert finetune.TrainSettings().numeric() == key["settings"]
    assert file_sha256(REPO / "training" / "train_refiner.py") == key["train_refiner_sha256"]
    assert file_sha256(REPO / "filament" / "models" / "refiner.py") == key["refiner_py_sha256"]


@pytest.mark.parametrize("name", sorted(LIFTED))
def test_each_lifted_function_is_unchanged(finetune, name):
    source = inspect.getsource(getattr(finetune, name))
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == LIFTED[name]


def test_the_recipe_is_the_one_the_published_file_was_trained_with(finetune):
    training = stored_config(fine_tuned_payload())["training"]
    assert (finetune.STEPS, finetune.BATCH, finetune.LR, finetune.WEIGHT_DECAY, finetune.WARMUP_STEPS,
            finetune.CLIP_NORM, finetune.SEED) == (training["steps"], training["batch"], training["lr"],
                                                   training["weight_decay"], training["warmup_steps"],
                                                   training["clip_norm"], training["seed"])
    assert finetune.TR.RESIDUAL_WEIGHT == training["residual_weight"] and finetune.TR.TILE == training["tile"]
    assert (training["dtype"], training["autocast"]) == ("float32", False)
    assert finetune.INIT_REFINER_SHA256 == training["init"]["sha256"] == REFINER_SHA256
    assert finetune.INIT_REFINER == pipeline.WEIGHTS_DIR / REFINER_FILE
    assert finetune.POOL_ORDER == ("011", "012") and "pools (011, 012)" in training["mix"]
    assert finetune.CONFIG_KEY in fine_tuned_payload()


def test_the_second_pool_comes_from_configuration_025s_detector(finetune):
    assert finetune.PROPOSER is pipeline.CONFIGURATIONS["025"].detector          # 2048 px, NMS 0.0, floor 0.05
    assert finetune.DETECTOR_CONFIGURATION == "025"
    assert finetune.PROPOSER.conf_floor == 0.05 and finetune.PROPOSER.max_det == 60


@pytest.mark.parametrize("cuda, mps, smoke, expected", [
    (True, True, False, "cuda"), (False, True, False, "mps"), (True, True, True, "cpu"), (False, False, True, "cpu")])
def test_the_device_is_cuda_then_mps_and_the_cpu_only_for_the_smoke_check(finetune, cuda, mps, smoke, expected):
    assert finetune.select_device(cuda, mps, smoke).type == expected


def test_without_a_gpu_the_fine_tune_stops_before_anything_runs(finetune):
    with pytest.raises(SystemExit, match="ABORT"):
        finetune.select_device(False, False, False)


def box_rle(x0, y0, x1, y1):
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return rle_of(mask)


def rle_of(mask):
    from pycocotools import mask as mask_util
    return mask_util.encode(np.asfortranarray(mask))["counts"].decode("ascii")


def refiner_run(folder, train_stems, val_stems, examples, proposals, fingerprint=REFERENCE_FINGERPRINT):
    """A folder in the layout ``training/train_refiner.py`` writes (config.json, examples, proposals)."""
    folder.mkdir(parents=True, exist_ok=True)
    config = {"fold_fingerprint": fingerprint, "train_stems": list(train_stems),
              "tune_stems": list(val_stems[::2]), "report_stems": list(val_stems[1::2])}
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "training_examples.json").write_text(json.dumps(examples))
    (folder / "training_proposals.json").write_text(json.dumps(proposals))
    return folder


def two_stem_run(tmp_path):
    train, val = ["t1", "t2"], ["v1", "v2"]
    proposals = {stem: {"rles": [box_rle(100, 100, 300, 140)], "scores": [0.9]} for stem in train}
    examples = [{"rle": box_rle(100, 100, 300, 140), "score": 0.9, "target_rles": [box_rle(100, 100, 300, 140)],
                 "match_iou": 1.0, "stem": stem} for stem in train]
    return train, val, proposals, examples


def test_the_refiner_run_is_checked_before_it_is_used(finetune, tmp_path):
    train, val, proposals, examples = two_stem_run(tmp_path)
    good = refiner_run(tmp_path / "good", train, val, examples, proposals)
    config, rows, cached = finetune.check_refiner_run(good, train, val)
    assert rows == examples and cached == proposals and config["train_stems"] == train
    bad = refiner_run(tmp_path / "fold", train, val, examples, proposals, fingerprint="0" * 64)
    with pytest.raises(finetune.RefusedInput, match="fold"):
        finetune.check_refiner_run(bad, train, val)
    with pytest.raises(finetune.RefusedInput, match="training stems"):
        finetune.check_refiner_run(good, ["t1", "t3"], val)
    leaked = refiner_run(tmp_path / "leak", train, val, examples + [dict(examples[0], stem="v1")], proposals)
    with pytest.raises(finetune.RefusedInput, match="validation stems"):
        finetune.check_refiner_run(leaked, train, val)
    partial = refiner_run(tmp_path / "partial", train, val, examples, {"t1": proposals["t1"]})
    with pytest.raises(finetune.RefusedInput, match="proposals"):
        finetune.check_refiner_run(partial, train, val)
    with pytest.raises(finetune.RefusedInput, match="config.json"):
        finetune.check_refiner_run(tmp_path / "nowhere", train, val)


def test_the_construction_control_rebuilds_the_first_pool_exactly(finetune):
    gt = {"t1": {"a1-t1": [box_rle(100, 100, 300, 140)], "a2-t1": [box_rle(102, 100, 300, 142)]}}
    proposals = {"t1": {"rles": [box_rle(100, 100, 290, 140), box_rle(1500, 1500, 1600, 1540)],
                        "scores": [0.9, 0.4]}}
    examples = finetune.build_examples(proposals, gt, ["t1"])
    assert [len(row["target_rles"]) for row in examples] == [2, 0]                   # a positive, a negative
    assert finetune.construction_control(proposals, examples, gt, ["t1"])["identical"]
    with pytest.raises(finetune.GuardFailure, match="construction"):
        finetune.construction_control(proposals, examples[:1], gt, ["t1"])


class RecordingDetector:
    """Stands in for ``ultralytics.YOLO``: records each predict call's settings, returns two proposals."""

    def __init__(self):
        self.calls = []

    def predict(self, **arguments):
        self.calls.append({key: value for key, value in arguments.items() if key != "source"})
        masks = torch.zeros((2, 2048, 2048))
        masks[0, 1000:1040, 900:1300] = 1.0
        masks[1, 300:330, 300:700] = 1.0
        return [SimpleNamespace(masks=SimpleNamespace(data=masks), boxes=SimpleNamespace(conf=torch.tensor([0.9, 0.2])))]


def test_the_second_pools_proposals_are_cached_per_image_under_their_key(finetune, tmp_path):
    images = tmp_path / "train_images"
    images.mkdir()
    for stem in ("t1", "t2"):
        cv2.imwrite(str(images / f"{stem}.jpeg"), np.full((2048, 2048), 120, dtype=np.uint8))
    model, cache = RecordingDetector(), tmp_path / "cache"
    key = {"detector_sha256": "x", "settings": dataclasses.asdict(finetune.PROPOSER)}
    found = finetune.propose_training_images(model, ["t1", "t2"], images, cache, key, torch.device("cpu"),
                                             log=lambda *a, **k: None)
    assert sorted(found) == ["t1", "t2"] and found["t1"]["scores"] == [pytest.approx(0.9), pytest.approx(0.2)]
    assert len(model.calls) == 2
    assert {key_: model.calls[0][key_] for key_ in ("imgsz", "conf", "iou", "max_det")} == {
        "imgsz": 2048, "conf": 0.05, "iou": 0.0, "max_det": 60}
    again = finetune.propose_training_images(model, ["t1", "t2"], images, cache, key, torch.device("cpu"),
                                             log=lambda *a, **k: None)
    assert again == found and len(model.calls) == 2                                 # every image reused
    with pytest.raises(finetune.RefusedInput, match="another"):
        finetune.propose_training_images(model, ["t1"], images, cache, dict(key, detector_sha256="y"),
                                         torch.device("cpu"), log=lambda *a, **k: None)


def test_a_short_cpu_fine_tune_writes_the_published_files_layout(finetune, tmp_path):
    images = tmp_path / "train_images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for stem in ("t1", "t2"):
        cv2.imwrite(str(images / f"{stem}.jpeg"), rng.integers(0, 255, (2048, 2048), dtype=np.uint8))
    target = box_rle(1000, 1000, 1100, 1030)
    pools = {name: [{"rle": box_rle(1000, 1000, 1100, 1032), "score": 0.9, "target_rles": [target],
                     "match_iou": 0.9, "stem": stem} for stem in ("t1", "t2")]
             + [{"rle": box_rle(200, 200, 260, 220), "score": 0.4, "target_rles": [], "match_iou": 0.0, "stem": "t1"}]
             for name in finetune.POOL_ORDER}
    model = NativeResidualUNet(base_channels=8)
    settings = finetune.TrainSettings(steps=3, workers=0, checkpoint_every=2, log_every=1)
    summary = finetune.fine_tune(tmp_path / "run", model, pools, images, settings, torch.device("cpu"),
                                 key={"test": True}, clock=finetune.Clock(1.0), tile=64, log=lambda *a, **k: None)
    assert summary["completed_steps"] == 3 and summary["device"] == "cpu"
    written = torch.load(tmp_path / "run" / finetune.CHECKPOINT_NAME, map_location="cpu", weights_only=True)
    published = fine_tuned_payload()
    assert set(written) == set(published)                                           # the published layout
    config, reference = stored_config(written), stored_config(published)
    assert set(config) == set(reference) and set(config["training"]) == set(reference["training"])
    assert config["training"]["completed_steps"] == 3 and config["training"]["tile"] == 64
    assert NativeResidualUNet.from_checkpoint(written).base_channels == 8
    assert len((tmp_path / "run" / "training_history.csv").read_text().splitlines()) == 1 + 3


@pytest.mark.parametrize("device, require_mps", [("mps", True), ("cpu", False), ("cuda", False)])
def test_fine_tune_applies_the_guards_of_its_device(finetune, monkeypatch, tmp_path, device, require_mps):
    """MPS: train()'s own MPS guards. CUDA: the probe, the parameters on the GPU and a warm-up first. CPU: none."""
    calls = {}

    def recording_train(out, model, pools, settings, on, key, clock, require_mps=True, **arguments):
        calls.update(require_mps=require_mps, device=str(on), experts=arguments["max_experts"], pools=len(pools))
        return {"environment": {"note": "train's own record"}, "completed_steps": settings.steps}

    class Model:
        def to(self, target):
            calls["moved_to"] = str(target)
            return self

        def parameters(self):
            return iter(())

    monkeypatch.setattr(finetune, "train", recording_train)
    monkeypatch.setattr(finetune, "assert_cuda_ready", lambda target: {"cuda": "a GPU", "probe": "ran"})
    monkeypatch.setattr(finetune, "assert_on_device", lambda tensors, kind: calls.setdefault("on_device", kind))
    monkeypatch.setattr(finetune, "mps_warmup", lambda model, batch, tile, on, experts, settings:
                        calls.setdefault("warmup", (batch, tile, str(on), experts)) and 0.5)
    target = box_rle(10, 10, 40, 20)
    pools = {name: [{"rle": target, "score": 0.9, "target_rles": [target, target], "match_iou": 1.0, "stem": "t1"}]
             for name in finetune.POOL_ORDER}
    summary = finetune.fine_tune(tmp_path, Model(), pools, tmp_path, finetune.TrainSettings(steps=2, workers=0),
                                 torch.device(device), key={}, clock=None, tile=64)
    assert calls["require_mps"] is require_mps and calls["device"] == device
    assert calls["experts"] == 2 and calls["pools"] == 2
    if device == "cuda":
        assert calls["on_device"] == "cuda" and calls["moved_to"] == "cuda"
        assert calls["warmup"] == (4, 64, "cuda", 2)                 # before train(), whose guard times steps
        assert summary["environment"] == {"cuda": "a GPU", "probe": "ran", "warmup_seconds": 0.5}
    else:
        assert "on_device" not in calls and "warmup" not in calls      # train() warms MPS up itself
        assert summary["environment"] == {"note": "train's own record"}


# --------------------------------------------------------------------------
# Against the fine-tune run's own script (opt-in)
# --------------------------------------------------------------------------

def test_the_lifted_functions_equal_the_fine_tune_runs_script(finetune):
    path = os.environ.get("FILAMENT_FT_SOURCE")
    if not path:
        pytest.skip("FILAMENT_FT_SOURCE is not set (the script of the run that fine-tuned the published file)")
    source = load_script(Path(path), "fine_tune_run_script")
    try:
        for name in LIFTED:
            assert inspect.getsource(getattr(source, name)) == inspect.getsource(getattr(finetune, name)), name
        assert source.training_code_sha256() == finetune.training_code_sha256()
        assert source.TrainSettings().numeric() == finetune.TrainSettings().numeric()
    finally:
        sys.modules.pop("fine_tune_run_script", None)
