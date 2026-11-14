"""Optional retraining, step 3: the fine-tuned refiner of configurations "013ft" and "035".

Nothing in the pipeline needs this: the notebook loads the published weights
from ``weights/``. This is the fine-tune that produced
``weights/refiner_native_residual_ft.pt``. It starts from the published refiner,
``weights/refiner_native_residual.pt``, and trains it further on the proposals
of two detectors on the same training images. It reads the output folder of
``train_refiner.py`` (its examples are the first pool) and runs the 2048 px
detector of configuration ``"025"`` for the second::

    python training/finetune_refiner.py --data-root data/MAGFiLO_1.0_Kaggle_2026 \\
        --refiner-run outputs/retrain/refiner --output-dir outputs/retrain/refiner_ft

**Hardware and time.** One CUDA GPU or Apple MPS (Apple silicon); the script
stops on a machine that has neither. The published run trained on Apple MPS
(M2, 8 GB): 6,000 steps in 1.5 hours, after the 2048 px detector had proposed
on the 565 training images on the CPU (about 5 s per image). ``FILAMENT_SMOKE=1``
runs the whole flow on 4 training images and 4 steps on the CPU, as a check of
the code. The detector run on the CPU needs about 5 GB of memory.

**The recipe**, and nothing else differs from ``train_refiner.py``'s training:

* starting point: the published refiner, loaded strictly, every tensor compared
  with the file;
* examples: pool ``"011"`` is ``train_refiner.py``'s own (the default
  detector's proposals down to confidence 0.05, matched to the annotators'
  outlines); pool ``"012"`` is built the same way from the proposals of the
  2048 px detector (``weights/detector_yolo11m_seg_2048.pt``: imgsz 2048, NMS
  IoU 0.0, the 0.05 floor, at most 60) on the same images. Before the second
  pool is trusted, the first is rebuilt from its own proposals and must come
  out row for row;
* crops: ``train_refiner.py``'s dataset, unchanged (512 px around a proposal,
  jitter, flips, 35 % dilate / erode, one crop in four a negative), half of
  every batch of four from each pool, each crop a pure function of the seed
  and its index;
* optimisation: 6,000 steps, batch 4, AdamW at learning rate 1e-4 (a third of
  the original 3e-4) with weight decay 1e-4, 100 warm-up steps and a cosine
  schedule to 0, gradient clipping at 5, the original loss plus 0.001 x the
  mean squared correction; float32, no mixed precision; seed 20261110.

The fine-tune does not choose alpha and scores nothing. The published
configurations use alpha 1.0, chosen on the tune half of the validation fold
from 0.5 / 1.0 / 1.5 / 2.0 when the fine-tuned weights were evaluated with the
default detector (report half 0.4346 against 0.4312 before the fine-tune).

**Guards.** On a GPU only: on CUDA, the architecture check and torch bootstrap
of ``train_refiner.py`` and a real CUDA operation before anything else; on MPS,
``PYTORCH_ENABLE_MPS_FALLBACK`` must be unset (an operation MPS lacks would then
run on the CPU without a word), the parameters and every batch must be on the
device, and each of the first 20 steps must take at most 3 s, after a warm-up
on a throwaway copy of the model. A non-finite loss stops the run. The last
checkpoint is saved unconditionally, with the training config next to the
weights; a stop (``--max-hours``, Ctrl-C) saves the state, and the same
command resumes from it, drawing exactly the crops it would have drawn.

**Writes to --output-dir:** ``refiner_native_residual_ft.pt`` (weights and
training config; copy it over the published file and update its size and
SHA-256 in ``weights/manifest.json`` to use it), ``train_state.pt`` (resume),
``training_history.csv``, ``proposals_2048/`` (one record per image, reused by a
rerun), ``training_proposals_2048.json``, ``training_examples_2048.json`` and
``run_report.json``.

**Where the code comes from.** Every function and class of the training proper
is the fine-tune run's own, unchanged, and ``tests/test_ft_config.py`` proves
it: the published file stores the SHA-256 of the source of the training loop's
objects (``training_code_sha256``), and the test recomputes it from this file;
the source of every other function taken over is pinned too. New here are the
command line, the device choice and the CUDA guard, the reading of
``train_refiner.py``'s folder, the proposals' cache, ``fine_tune`` (which picks
the guards of the device) and ``memory_snapshot`` (a log line, portable).
"""

import argparse
import copy
import dataclasses
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone


def bootstrap_torch():
    """BOOTSTRAP: reject unsupported CUDA kernels or install the compatible build."""
    import torch as probe
    if probe.cuda.is_available():
        properties = probe.cuda.get_device_properties(0)
        architecture = f"sm_{properties.major}{properties.minor}"
        if architecture not in probe.cuda.get_arch_list():
            if os.environ.get("FILAMENT_TORCH_PATCHED") == "1":
                raise SystemExit("patched torch still lacks the assigned CUDA architecture")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1",
                "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu121",
            ], check=True)
            os.environ["FILAMENT_TORCH_PATCHED"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune the native residual refiner on two detectors' proposals (one CUDA GPU or Apple "
                    "MPS; about 1.5 hours of training on an Apple M2).")
    parser.add_argument("--data-root", required=True,
                        help="the competition folder MAGFiLO_1.0_Kaggle_2026, holding train/ and test/")
    parser.add_argument("--refiner-run", required=True,
                        help="output folder of train_refiner.py: config.json, training_examples.json and "
                             "training_proposals.json (the first pool)")
    parser.add_argument("--output-dir", required=True, help="where the checkpoint and the reports are written")
    parser.add_argument("--detector", default=None,
                        help="weights of the 2048 px detector (default: the published one, configuration 025)")
    parser.add_argument("--init-refiner", default=None,
                        help="the refiner to start from (default: the published weights/refiner_native_residual.pt)")
    parser.add_argument("--max-hours", type=float, default=6.0,
                        help="wall-clock cap; a stop saves the state, and the same command resumes (default 6)")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers (default 2)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    ARGS = parse_args()
    bootstrap_torch()


ROOT = Path(__file__).resolve().parents[1]           # the repository: filament/, weights/, training/
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from filament import pipeline  # noqa: E402
from filament.folds import (REFERENCE_FINGERPRINT, VALIDATION_FOLD, assert_no_leakage,  # noqa: E402
                            fold_fingerprint, grouped_folds, validation_halves)
from filament.io import IMAGE_SIZE, build_gt_frame, file_sha256, load_coco, load_image, verify_weights  # noqa: E402
from filament.models import detector, refiner  # noqa: E402
from filament.models.refiner import NativeResidualUNet, native_refiner_loss, prior_logits_from_input  # noqa: E402


def load_module(name, path):
    """A script as a module under its own name (registered, so DataLoader workers can unpickle from it)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# train_refiner.py, imported (its __main__ block -- argument parsing and the torch bootstrap -- does not run):
# the dataset, collate, worker seeding, matching and its constants are its own objects.
TR = load_module("finetune_train_refiner", Path(__file__).resolve().parent / "train_refiner.py")

ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FULL_SIZE = IMAGE_SIZE
SMOKE = os.environ.get("FILAMENT_SMOKE") == "1"
REQUIRE_GPU = not SMOKE
SMOKE_IMAGES = 4                 # FILAMENT_SMOKE=1: the first training images ...
SMOKE_STEPS = 4                  # ... and optimizer steps, on the CPU

# The budget of the published fine-tune.
SEED = 20261110
STEPS = 6000
BATCH = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 100
CLIP_NORM = 5.0
CHECKPOINT_EVERY = 1000
LOG_EVERY = 100
WORKERS = 2                      # train_refiner's DataLoader workers outside its smoke
MAX_HOURS = 6.0

# MPS guards.
STEP_LIMIT_SECONDS = 3.0
TIMED_STEPS = 20
FALLBACK_VARIABLE = "PYTORCH_ENABLE_MPS_FALLBACK"

# The checkpoint: NativeResidualUNet.checkpoint() plus the training config under CONFIG_KEY, the key the published
# file uses (named after the run that fine-tuned it).
EXPERIMENT = "finetune_refiner"
CONFIG_KEY = "exp034"
CHECKPOINT_NAME = "refiner_native_residual_ft.pt"
STATE_NAME = "train_state.pt"
POOL_ORDER = ("011", "012")      # pool 0: the default detector's (its training run was 011), pool 1: the 2048 px one's

# The starting point: the published refiner (--init-refiner replaces both before training).
INIT_REFINER = ROOT / "weights" / "refiner_native_residual.pt"
INIT_REFINER_SHA256 = "5204da7ce821e25fa43473f1bce25b299c7ba1f021c506e5c8df0670195d474b"

# The second pool's detector settings are configuration "025"'s: imgsz 2048, NMS IoU 0.0, the 0.05 floor, max_det 60.
DETECTOR_CONFIGURATION = "025"
PROPOSER = pipeline.CONFIGURATIONS[DETECTOR_CONFIGURATION].detector


class FinetuneError(RuntimeError):
    exit_code = 1


class RefusedInput(FinetuneError):
    exit_code = 2


class GuardFailure(FinetuneError):
    exit_code = 3


class BudgetStop(FinetuneError):
    exit_code = 4


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def relative(path):
    path = Path(path).absolute()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json_atomic(path, payload, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=indent, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def format_duration(seconds):
    seconds = int(round(max(0.0, seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def refuse_tainted(path):
    """Never read a file under a folder named ``TAINTED*``: the convention for files quarantined from any use."""
    parts = Path(path).absolute().parts
    if any(part.upper().startswith("TAINTED") for part in parts):
        raise RefusedInput(f"refused: {path} lies under a TAINTED-* folder")


def plain(value):
    """``value`` as plain JSON types: ``torch.load(weights_only=True)`` refuses anything else (e.g. the
    ``TorchVersion`` that ``torch.__version__`` is), so every config, key and extra stored in a checkpoint is
    passed through this first."""
    return json.loads(json.dumps(value, allow_nan=False))


def rows_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def save_torch_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device=None):
    """This process's peak RSS, and the device's allocations, for the log."""
    snapshot = {}
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS, KiB on Linux
        snapshot["rss_peak_gb"] = round(peak / (1e9 if sys.platform == "darwin" else 1e6), 2)
    except ImportError:                                                     # no resource module (Windows)
        pass
    if device is not None and torch.device(device).type == "mps":
        snapshot["mps_allocated_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 2)
        snapshot["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    if device is not None and torch.device(device).type == "cuda":
        snapshot["cuda_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return snapshot


class Clock:
    """``max_hours`` of computing; time spent waiting for other jobs extends the deadline."""

    def __init__(self, max_hours):
        self.started = time.time()
        self.deadline = self.started + max_hours * 3600.0
        self.waited = 0.0

    def paused(self, seconds):
        self.deadline += seconds
        self.waited += seconds

    def remaining_hours(self):
        return max(0.0, (self.deadline - time.time()) / 3600.0)

    def check(self, stage):
        if time.time() > self.deadline:
            raise BudgetStop(f"the wall-clock cap was reached during {stage}; rerun the same command to resume")


# ---------------------------------------------------------------------------
# Examples: train_refiner's construction, and no validation stem in them
# ---------------------------------------------------------------------------

def training_ground_truth(coco):
    """Every annotator record's outlines per stem: train_refiner.main's own lines (580-586), copied verbatim."""
    all_gt = defaultdict(list)
    for annotation in coco["annotations"]:
        rles = mask_util.frPyObjects(annotation["segmentation"], FULL_SIZE, FULL_SIZE)
        all_gt[annotation["image_id"]].append(mask_util.merge(rles)["counts"].decode("ascii"))
    per_stem_gt = defaultdict(dict)
    for record, rles in all_gt.items():
        per_stem_gt[record.split("-", 1)[1]][record] = rles
    return per_stem_gt


def build_examples(proposals_by_stem, per_stem_gt, stems):
    """train_refiner.main's example loop (lines 649-655) over cached proposals, stem by stem, in order."""
    examples = []
    for stem in stems:
        proposals = proposals_by_stem[stem]
        clusters = TR.cluster_annotation_masks(per_stem_gt[stem])
        for example in TR.match_proposals(proposals, clusters):
            example["stem"] = stem
            examples.append(example)
    return examples


def check_examples(examples, train_stems, val_stems, what):
    """Every example comes from a training stem and none from a validation stem; counts."""
    train, val = set(train_stems), set(val_stems)
    if train & val:
        raise RefusedInput(f"the training and validation stem lists overlap: {sorted(train & val)[:3]}")
    stems = {row["stem"] for row in examples}
    leaked = sorted(stems & val)
    if leaked:
        raise RefusedInput(f"{what}: {len(leaked)} validation stems in the training examples, e.g. {leaked[:3]}")
    foreign = sorted(stems - train)
    if foreign:
        raise RefusedInput(f"{what}: examples from {len(foreign)} stems that are not a training stem, "
                           f"e.g. {foreign[:3]}")
    positives = sum(1 for row in examples if row["target_rles"])
    if not positives:
        raise RefusedInput(f"{what}: no positive examples")
    return {"rows": len(examples), "positives": positives, "negatives": len(examples) - positives,
            "stems": len(stems)}


def construction_control(proposals, examples, per_stem_gt, stems):
    """The first pool rebuilt from its own proposals by ``build_examples`` must be its examples, row for row.

    That proves the construction the second pool goes through is ``train_refiner.py``'s.
    """
    rebuilt = build_examples(proposals, per_stem_gt, stems)
    wanted = [row for row in examples if row["stem"] in set(stems)]
    if rebuilt != wanted:
        raise GuardFailure(f"rebuilding the first pool from its proposals gave {len(rebuilt)} rows, not its "
                           f"{len(wanted)} (or other ones): the construction is not train_refiner's")
    return {"stems": len(stems), "rows": len(rebuilt), "identical": True,
            "what": "build_examples(training_proposals.json) == training_examples.json"}


def check_refiner_run(folder, train_stems, val_stems):
    """``train_refiner.py``'s output folder: the first pool's examples and the proposals they were built from.

    Its fold must be the reference fold, its training stems the fold's, its examples from training stems only
    (``check_examples``) and its proposals one entry per training stem. Returns ``(config, examples, proposals)``.
    """
    folder = Path(folder)
    paths = {name: folder / name for name in ("config.json", "training_examples.json", "training_proposals.json")}
    for name, path in paths.items():
        refuse_tainted(path)
        if not path.is_file():
            raise RefusedInput(f"{name} not found in {folder}: --refiner-run is the output folder of train_refiner.py")
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    if config.get("fold_fingerprint") != REFERENCE_FINGERPRINT:
        raise RefusedInput(f"the refiner run was made on another fold ({str(config.get('fold_fingerprint'))[:12]}...)")
    if sorted(config.get("train_stems") or []) != sorted(train_stems):
        raise RefusedInput("the refiner run's training stems are not the fold's training stems")
    examples = json.loads(paths["training_examples.json"].read_text(encoding="utf-8"))
    check_examples(examples, train_stems, val_stems, "the refiner run's examples")
    proposals = json.loads(paths["training_proposals.json"].read_text(encoding="utf-8"))
    if sorted(proposals) != sorted(train_stems):
        raise RefusedInput("the refiner run's training proposals do not cover exactly the training stems")
    return config, examples, proposals


def proposals_key(detector_sha256, device):
    """What a cached proposal record depends on: the weights, the settings, the device and the code."""
    return {"detector_sha256": detector_sha256, "settings": dataclasses.asdict(PROPOSER), "device": device.type,
            "torch": str(torch.__version__), "ultralytics": package_version("ultralytics"),
            "detector_py_sha256": file_sha256(detector.__file__)}


def propose_training_images(model, stems, image_dir, cache_dir, key, device, log=print):
    """The second pool's proposals on the training images, down to the floor, one cached record per image.

    ``detector.propose`` with configuration "025"'s settings, exactly as the pipeline proposes. A record is reused
    only under the same key; one written under another key is refused, never overwritten.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    found, computed, began = {}, 0, time.time()
    for position, stem in enumerate(stems, start=1):
        path = cache_dir / f"{stem}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("key") != key or record.get("stem") != stem:
                raise RefusedInput(f"{path} was written by another detector, setting or image; use a fresh "
                                   "--output-dir or delete it")
        else:
            tic = time.perf_counter()
            pairs = detector.propose(model, load_image(Path(image_dir) / f"{stem}.jpeg"), PROPOSER, device)
            record = {"key": key, "stem": stem, "rles": [rle for rle, _ in pairs],
                      "scores": [score for _, score in pairs], "seconds": round(time.perf_counter() - tic, 2)}
            write_json_atomic(path, record, indent=None)
            computed += 1
        found[stem] = {"rles": record["rles"], "scores": record["scores"]}
        if position % 25 == 0 or position == len(stems):
            mean = (time.time() - began) / max(1, computed)
            log(f"  [proposals] {position}/{len(stems)} images ({computed} computed, {mean:.1f} s each), "
                f"{sum(len(item['rles']) for item in found.values())} proposals", flush=True)
    return found


# ---------------------------------------------------------------------------
# The 50/50 mix over train_refiner's own dataset
# ---------------------------------------------------------------------------

def pool_of(index):
    """Crop ``index``'s pool: batches of four alternate (0, 1, 0, 1) and (1, 0, 1, 0).

    train_refiner draws a negative where ``index % 4 == 0``; that index goes to pool 0 in even batches
    and to pool 1 in odd ones, so each pool gets exactly one negative in every four of its crops.
    """
    return (index + index // 4) % 2


def item_seed(seed, index):
    """The global numpy seed for crop ``index``: a pure function of (seed, index)."""
    return int(np.random.SeedSequence([int(seed), int(index)]).generate_state(1)[0])


class MixedCropDataset(Dataset):
    """Half the crops from each detector's ``TR.NativeCropDataset``, deterministic per index.

    ``NativeCropDataset.__getitem__`` seeds its generator from the global numpy RNG; seeding that RNG from
    (seed, index) first makes every crop a pure function of its index, independent of the worker that
    draws it -- so a resumed run draws exactly the crops the uninterrupted run would have drawn. The pool
    is called with the crop's own index, so its one-in-four negative rule applies unchanged.
    """

    def __init__(self, pools, length, seed):
        if len(pools) != 2:
            raise ValueError("the mix takes exactly two pools: run 011's and 012's")
        self.pools, self.length, self.seed = tuple(pools), int(length), int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        np.random.seed(item_seed(self.seed, index))
        return self.pools[pool_of(index)][index]


def make_pools(examples_by_detector, image_dir, tile=None, length=None):
    """One ``TR.NativeCropDataset`` per detector, in :data:`POOL_ORDER`."""
    tile = TR.TILE if tile is None else tile
    length = STEPS * BATCH if length is None else length
    return tuple(TR.NativeCropDataset(examples_by_detector[name], Path(image_dir), tile, length)
                 for name in POOL_ORDER)


# ---------------------------------------------------------------------------
# Device and loading guards
# ---------------------------------------------------------------------------

def select_device(cuda_available, mps_available, smoke):
    """A CUDA GPU if there is one, else Apple MPS; the CPU only for the ``FILAMENT_SMOKE=1`` check."""
    if smoke:
        return torch.device("cpu")
    if cuda_available:
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    raise SystemExit("ABORT: the fine-tune needs a CUDA GPU or Apple MPS; FILAMENT_SMOKE=1 runs a short check on "
                     "the CPU")


def assert_cuda_ready(device):
    """A real CUDA operation on ``device``, whose architecture this torch build supports (see ``bootstrap_torch``)."""
    properties = torch.cuda.get_device_properties(device)
    architecture = f"sm_{properties.major}{properties.minor}"
    if architecture not in torch.cuda.get_arch_list():
        raise GuardFailure(f"{properties.name} is {architecture}, which this torch build lacks "
                           f"({torch.cuda.get_arch_list()})")
    probe = torch.nn.Conv2d(4, 4, 3, padding=1).to(device)
    probe(torch.randn(1, 4, 16, 16, device=device))
    torch.cuda.synchronize()
    return {"cuda": properties.name, "architecture": architecture, "probe": "a convolution ran on the GPU"}


def assert_mps_ready(environ=None, backend=None):
    environ = os.environ if environ is None else environ
    backend = torch.backends.mps if backend is None else backend
    if FALLBACK_VARIABLE in environ:
        raise GuardFailure(f"{FALLBACK_VARIABLE} is set ({environ[FALLBACK_VARIABLE]!r}): an operation MPS lacks "
                           "would then run on the CPU without a word; unset it")
    if not backend.is_available():
        raise GuardFailure("MPS is not available (torch.backends.mps.is_available() is False): this run "
                           "trains on Apple MPS only")
    return {"mps_available": True, FALLBACK_VARIABLE: None}


def assert_on_device(tensors, device_type):
    tensors = list(tensors)
    if not tensors:
        raise GuardFailure("no tensors to check")
    wrong = sorted({tensor.device.type for tensor in tensors if tensor.device.type != device_type})
    if wrong:
        raise GuardFailure(f"tensors on {wrong}, expected all on {device_type}")


class StepGuard:
    """Each of the first ``timed_steps`` steps of a session must take at most ``limit_seconds``."""

    def __init__(self, limit_seconds=STEP_LIMIT_SECONDS, timed_steps=TIMED_STEPS):
        self.limit, self.timed, self.first = limit_seconds, timed_steps, []

    def observe(self, session_step, seconds):
        if session_step >= self.timed:
            return
        self.first.append(float(seconds))
        if seconds > self.limit:
            raise GuardFailure(f"step {session_step + 1} of this session took {seconds:.2f} s > {self.limit} s: "
                               "that is a CPU's pace, not MPS's")


def check_finite(loss, step):
    value = float(loss.detach())
    if not math.isfinite(value):
        raise GuardFailure(f"non-finite loss {value} at step {step}")
    return value


def check_state_loaded(model, state):
    """Every tensor of ``model`` equals the checkpoint's, with no key missing or extra: zero loaded is loud."""
    own = model.state_dict()
    if not own:
        raise GuardFailure("zero tensors loaded")
    missing, unexpected = sorted(set(own) - set(state)), sorted(set(state) - set(own))
    if missing or unexpected:
        raise GuardFailure(f"not every tensor loaded: missing {missing[:5]}, unexpected {unexpected[:5]}")
    differing = [name for name in own if not torch.equal(own[name].detach().cpu(), state[name].detach().cpu())]
    if differing:
        raise GuardFailure(f"{len(differing)} tensors differ from the checkpoint, e.g. {differing[:3]}")
    return {"tensors": len(own), "parameters": int(sum(p.numel() for p in model.parameters())),
            "missing": [], "unexpected": []}


def load_refiner_checkpoint(path, sha256, device):
    """A refiner checkpoint, refused unless its sha256 matches; loaded strictly and compared tensor by tensor."""
    path = Path(path)
    refuse_tainted(path)
    try:
        verify_weights(path, sha256)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise RefusedInput(f"refiner weights refused: {error}") from error
    payload = torch.load(path, map_location="cpu", weights_only=True)
    try:
        model = NativeResidualUNet.from_checkpoint(payload)              # load_state_dict(strict=True)
    except (RuntimeError, ValueError, KeyError, TypeError) as error:
        raise GuardFailure(f"strict load of {path.name} failed: {error}") from error
    check = check_state_loaded(model, payload["model"])
    return model.to(device), check


# ---------------------------------------------------------------------------
# The checkpoint: weights plus the config they were trained with
# ---------------------------------------------------------------------------

def inference_settings():
    """Every refiner inference setting except alpha: exactly what exp025's path refines with."""
    return {key: value for key, value in dataclasses.asdict(refiner.REFINER).items() if key != "alpha"}


def checkpoint_config(steps, completed_steps, base_channels=None, device="mps", extra=None, settings=None,
                      tile=None):
    """What the weights were trained with, stored in the checkpoint; ``inference`` is what serving replays."""
    numeric = (settings or TrainSettings(steps=steps)).numeric()
    config = {
        "experiment": EXPERIMENT, "format": 1,
        "model_config": {"base_channels": TR.BASE_CHANNELS if base_channels is None else base_channels},
        "inference": inference_settings(), "prior_logit": float(refiner.PRIOR_LOGIT),
        "training": {"steps": int(steps), "completed_steps": int(completed_steps), "batch": numeric["batch"],
                     "tile": TR.TILE if tile is None else int(tile), "lr": numeric["lr"],
                     "weight_decay": numeric["weight_decay"], "warmup_steps": numeric["warmup"],
                     "schedule": "train_refiner's: linear warm-up, cosine to 0 over the steps (step-based)",
                     "clip_norm": numeric["clip"], "residual_weight": TR.RESIDUAL_WEIGHT, "seed": numeric["seed"],
                     "device": device, "dtype": "float32", "autocast": False,
                     "mix": "50/50: crop i from pool (i + i // 4) % 2, pools (011, 012)",
                     "negatives": "train_refiner's NativeCropDataset: index % 4 == 0 -> a negative",
                     "init": {"path": relative(INIT_REFINER), "sha256": INIT_REFINER_SHA256}},
    }
    if extra:
        config.update(extra)
    return config


def save_checkpoint(path, model, config):
    payload = model.checkpoint()
    payload[CONFIG_KEY] = plain(config)
    save_torch_atomic(path, payload)


# ---------------------------------------------------------------------------
# Training: train_refiner's step on the fine-tune's budget, resumable
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TrainSettings:
    steps: int = STEPS
    batch: int = BATCH
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    warmup: int = WARMUP_STEPS
    clip: float = CLIP_NORM
    seed: int = SEED
    workers: int = WORKERS
    checkpoint_every: int = CHECKPOINT_EVERY
    log_every: int = LOG_EVERY

    def numeric(self):
        """What the weights depend on (workers and the logging cadence do not change a single crop)."""
        return {key: getattr(self, key) for key in ("steps", "batch", "lr", "weight_decay", "warmup", "clip", "seed")}


def lr_at(step, steps, lr=LR, warmup=WARMUP_STEPS):
    """train_refiner's schedule, step-based: ``lr * warmup * .5 * (1 + cos(pi * progress))``."""
    progress = min(1.0, step / steps)
    warm = min(1.0, (step + 1) / warmup)
    return lr * warm * .5 * (1 + math.cos(math.pi * progress))


def make_loader(dataset, start_step, settings):
    """Batches ``start_step`` .. ``steps - 1`` in order, built by train_refiner's collate and worker seed."""
    sampler = range(start_step * settings.batch, settings.steps * settings.batch)
    extra = {"multiprocessing_context": "spawn"} if settings.workers else {}
    return DataLoader(dataset, batch_size=settings.batch, sampler=sampler, shuffle=False,
                      num_workers=settings.workers, collate_fn=TR.collate_crops, worker_init_fn=TR.seed_worker,
                      generator=torch.Generator().manual_seed(settings.seed), **extra)


def optimizer_state_cpu(state):
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().clone()
    if isinstance(state, dict):
        return {key: optimizer_state_cpu(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(optimizer_state_cpu(value) for value in state)
    return state


def save_state(path, model, optimizer, completed, history, key, extras):
    save_torch_atomic(path, {"key": plain(key), "step": int(completed), "model": model.checkpoint(),
                             "optimizer": optimizer_state_cpu(optimizer.state_dict()), "history": plain(history),
                             "extras": plain(extras)})


def write_history(path, history):
    pd.DataFrame(history, columns=["step", "loss", "learning_rate", "step_seconds", "data_seconds",
                                   "device"]).to_csv(path, index=False)


def mps_warmup(model, batch, tile, device, max_experts, settings=None):
    """Build MPS's kernels before the timed steps: whole training steps on a throwaway copy of the model.

    MPS compiles its kernels on the first call for each shape (about 4 s here), which the 3-second guard
    would take for a CPU's pace. A deep copy with its own AdamW takes forward, loss, backward, clip and step
    on synthetic batches of every expert count; the model being trained and its optimizer are never
    touched, and its weights are compared before and after to prove it.
    """
    settings = settings or TrainSettings()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    began = time.perf_counter()
    scratch = copy.deepcopy(model).to(device).train()
    optimizer = torch.optim.AdamW(scratch.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    generator = torch.Generator().manual_seed(0)
    for experts in range(1, max_experts + 1):
        images = torch.rand(batch, 4, tile, tile, generator=generator)
        images[:, 2] = (images[:, 2] > 0.5).float()
        targets = (torch.rand(batch, experts, tile, tile, generator=generator) > 0.5).float()
        present = torch.ones(batch, experts, dtype=torch.bool)
        images, targets, present = images.to(device), targets.to(device), present.to(device)
        loss = native_refiner_loss(scratch(images), targets, present, residual_weight=TR.RESIDUAL_WEIGHT,
                                   prior_logits=prior_logits_from_input(images))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scratch.parameters(), settings.clip)
        optimizer.step()
        synchronize(device)
    del scratch, optimizer
    after = model.state_dict()
    if any(not torch.equal(before[name], after[name]) for name in before):
        raise GuardFailure("the MPS warm-up changed the weights")
    return time.perf_counter() - began


def summary_of(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {"n": int(array.size), "mean": round(float(array.mean()), 4), "median": round(float(np.median(array)), 4),
            "p95": round(float(np.percentile(array, 95)), 4), "max": round(float(array.max()), 4)}


def train(out, model, pools, settings, device, key, clock, require_mps=True, max_session_steps=None,
          extras=None, max_experts=4, log=print, pause=None):
    """Fine-tune ``model`` on the mix; resumes from ``<out>/train_state.pt`` when its key matches.

    The step is train_refiner's (``train_refiner``, lines 500-528) without its CUDA-only parts: off CUDA its
    GradScaler is disabled and its autocast off, so scale / unscale / step reduce to backward, clip, step,
    which is what runs here, in float32. The state is saved every ``checkpoint_every`` steps, at a stop and
    at the end, unconditionally; the last checkpoint (weights + config) with it. ``pause`` (if given) is
    called every ``log_every`` steps, between two steps: it waits while another heavy job runs.
    """
    device = torch.device(device)
    if require_mps:
        environment = assert_mps_ready()
        if device.type != "mps":
            raise GuardFailure(f"training was asked to run on {device}, not mps")
    else:
        environment = {"mps_available": bool(torch.backends.mps.is_available()), "note": "require_mps=False (tests)"}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state_path, final_path, history_path = out / STATE_NAME, out / CHECKPOINT_NAME, out / "training_history.csv"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    start, history, extras = 0, [], dict(extras or {})
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("key") != key:
            raise RefusedInput(f"{relative(state_path)} belongs to another training run (other settings, examples "
                               "or code); use a fresh --out or delete it")
        restored = NativeResidualUNet.from_checkpoint(state["model"])
        model.load_state_dict(restored.state_dict(), strict=True)
        check_state_loaded(model, state["model"]["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, history = int(state["step"]), list(state["history"])
        extras = dict(state.get("extras") or {})                  # the first session's (parity included)
        extras["resumes"] = list(extras.get("resumes") or []) + [{"at_step": start, "utc": utc_now()}]
        log(f"[train {device}] resuming at step {start}/{settings.steps} from {relative(state_path)}", flush=True)
    if require_mps:
        assert_on_device(model.parameters(), "mps")
    resumed_from = start

    def checkpoint(completed):
        config = checkpoint_config(settings.steps, completed, model.base_channels, device.type,
                                   extra={"key": key, "extras": extras}, settings=settings, tile=pools[0].tile)
        save_state(state_path, model, optimizer, completed, history, key, extras)
        save_checkpoint(final_path, model, config)
        write_history(history_path, history)

    if start >= settings.steps:
        if not final_path.is_file():
            checkpoint(start)
        return summarise_training(history, settings, device, resumed_from, 0, environment, None, extras)

    warmup_seconds = mps_warmup(model, settings.batch, pools[0].tile, device, max_experts, settings) \
        if device.type == "mps" else 0.0
    log(f"[train {device}] warm-up {warmup_seconds:.1f} s (throwaway copy, expert counts 1-{max_experts}); "
        f"memory {memory_snapshot(device)}", flush=True)
    dataset = MixedCropDataset(pools, length=settings.steps * settings.batch, seed=settings.seed)
    loader = make_loader(dataset, start, settings)
    guard, session, completed = StepGuard(), 0, start
    model.train()
    began_session = time.time()
    try:
        waited_at = time.perf_counter()
        for step, (images, targets, present) in zip(range(start, settings.steps), loader):
            data_seconds = time.perf_counter() - waited_at
            clock.check(f"training at step {step}")
            rate = lr_at(step, settings.steps, settings.lr, settings.warmup)
            for group in optimizer.param_groups:
                group["lr"] = rate
            began = time.perf_counter()
            images, targets, present = images.to(device), targets.to(device), present.to(device)
            if require_mps:
                assert_on_device((images, targets, present), "mps")
            logits = model(images)
            if logits.dtype != torch.float32:
                raise GuardFailure(f"logits are {logits.dtype}, not float32")
            loss = native_refiner_loss(logits, targets, present, residual_weight=TR.RESIDUAL_WEIGHT,
                                       prior_logits=prior_logits_from_input(images))
            value = check_finite(loss, step + 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.clip)
            optimizer.step()
            synchronize(device)
            seconds = time.perf_counter() - began
            completed = step + 1
            history.append({"step": completed, "loss": value, "learning_rate": rate, "step_seconds": round(seconds, 4),
                            "data_seconds": round(data_seconds, 4), "device": str(device)})
            try:
                guard.observe(session, seconds)
            except GuardFailure as error:
                raise GuardFailure(f"{error}; data wait {data_seconds:.2f} s; memory {memory_snapshot(device)}") \
                    from error
            session += 1
            if completed % settings.log_every == 0 or session == 1 or completed == settings.steps:
                recent = [row["loss"] for row in history[-settings.log_every:]]
                rate_now = (time.time() - began_session) / session
                log(f"[train {device}] step {completed}/{settings.steps} loss {value:.4f} (mean of last "
                    f"{len(recent)}: {np.mean(recent):.4f}) lr {rate:.2e} | {seconds:.2f} s/step, data wait "
                    f"{data_seconds:.2f} s | elapsed {format_duration(time.time() - began_session)}, ETA "
                    f"{format_duration(rate_now * (settings.steps - completed))}", flush=True)
                write_history(history_path, history)
            if completed % settings.checkpoint_every == 0 and completed < settings.steps:
                checkpoint(completed)
                log(f"[train {device}] checkpoint at step {completed}: {relative(state_path)}, "
                    f"{relative(final_path)}", flush=True)
            if max_session_steps is not None and session >= max_session_steps and completed < settings.steps:
                raise BudgetStop(f"{max_session_steps} steps this session; rerun to resume at step {completed}")
            if pause is not None and completed % settings.log_every == 0 and completed < settings.steps:
                pause()
            waited_at = time.perf_counter()
    except (BudgetStop, KeyboardInterrupt):
        checkpoint(completed)                                     # unconditional: a stop keeps what it learned
        log(f"[train {device}] stopped at step {completed}; state saved to {relative(state_path)}", flush=True)
        raise
    if completed != settings.steps:
        raise GuardFailure(f"the loader ended at step {completed} of {settings.steps}")
    checkpoint(completed)                                         # the last checkpoint, unconditionally
    log(f"[train {device}] done: {completed} steps; {relative(final_path)}", flush=True)
    return summarise_training(history, settings, device, resumed_from, session, environment, guard, extras,
                              warmup_seconds=warmup_seconds, session_seconds=time.time() - began_session)


def summarise_training(history, settings, device, resumed_from, session, environment, guard, extras,
                       warmup_seconds=None, session_seconds=None):
    losses = [row["loss"] for row in history]
    return {"completed_steps": int(history[-1]["step"]) if history else 0, "steps": settings.steps,
            "resumed_from": resumed_from, "session_steps": session, "device": str(device),
            "dtype": "float32", "autocast": False, "environment": environment,
            "first_steps_seconds": guard.first if guard else None,
            "step_seconds": summary_of([row["step_seconds"] for row in history]),
            "data_wait_seconds": summary_of([row["data_seconds"] for row in history]),
            "warmup_seconds": None if warmup_seconds is None else round(warmup_seconds, 2),
            "session_seconds": None if session_seconds is None else round(session_seconds, 1),
            "loss": {"first_100_mean": round(float(np.mean(losses[:100])), 5) if losses else None,
                     "last_100_mean": round(float(np.mean(losses[-100:])), 5) if losses else None,
                     "min": round(float(np.min(losses)), 5) if losses else None},
            "settings": settings.numeric(), "extras": extras}


# The source of everything the trained weights depend on in this file; hashed into the training key, so a
# changed loop is never resumed from another loop's state. The published file stores this hash.
TRAINING_CODE = (pool_of, item_seed, MixedCropDataset, make_pools, lr_at, make_loader, mps_warmup, train,
                 check_finite, StepGuard, save_state, optimizer_state_cpu)


def training_code_sha256():
    return hashlib.sha256("".join(inspect.getsource(obj) for obj in TRAINING_CODE).encode("utf-8")).hexdigest()


def training_key(settings, examples, stems):
    """What the weights depend on, checked before a resume: the settings, the start, both pools, the code."""
    return plain({"recipe": EXPERIMENT, "settings": settings.numeric(), "init_refiner_sha256": INIT_REFINER_SHA256,
                  "examples": {name: rows_digest(rows) for name, rows in examples.items()},
                  "train_refiner_sha256": file_sha256(TR.__file__), "refiner_py_sha256": file_sha256(refiner.__file__),
                  "training_code_sha256": training_code_sha256(), "torch": str(torch.__version__), "smoke": SMOKE,
                  "images": len(stems)})


def fine_tune(out, model, examples_by_pool, image_dir, settings, device, key, clock, tile=None, extras=None,
              log=print):
    """``train`` on the two pools, with the guards of the device it runs on.

    On MPS, ``train``'s own guards (``require_mps``). On CUDA, the probe and the architecture check
    (``assert_cuda_ready``), the parameters on the GPU, and the warm-up ``train`` runs only on MPS: its step
    guard times the first steps on every device, so CUDA's first-call setup runs here first, on a throwaway
    copy. ``train`` then runs without the MPS guards, and the CUDA record replaces its environment record. On
    the CPU (the smoke check), none of these.
    """
    device = torch.device(device)
    pools = make_pools(examples_by_pool, image_dir, tile=tile, length=settings.steps * settings.batch)
    max_experts = max(1, max(len(row["target_rles"]) for rows in examples_by_pool.values() for row in rows))
    environment = None
    if device.type == "cuda":
        environment = assert_cuda_ready(device)
        model = model.to(device)
        assert_on_device(model.parameters(), "cuda")
        environment["warmup_seconds"] = round(mps_warmup(model, settings.batch, pools[0].tile, device, max_experts,
                                                         settings), 2)
    summary = train(out, model, pools, settings, device, key, clock, require_mps=device.type == "mps",
                    extras=extras, max_experts=max_experts, log=log)
    if environment is not None:
        summary["environment"] = environment
    return summary


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def main():
    global INIT_REFINER, INIT_REFINER_SHA256
    started = time.monotonic()
    out = Path(ARGS.output_dir)
    refuse_tainted(out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = select_device(torch.cuda.is_available(), torch.backends.mps.is_available(), SMOKE)
    if REQUIRE_GPU and device.type == "cpu":
        raise SystemExit("ABORT: the production fine-tune requires a CUDA GPU or Apple MPS")
    print(f"fine-tune on {device}{' -- FILAMENT_SMOKE=1: a short check on the CPU' if SMOKE else ''}", flush=True)
    if not SMOKE and (TR.TILE, TR.BASE_CHANNELS, TR.RESIDUAL_WEIGHT) != (512, 24, 0.001):
        raise RefusedInput("train_refiner's production constants are not 512 / 24 / 0.001")

    # the fold, and the first pool from train_refiner.py's folder
    annotations = Path(ARGS.data_root) / "train" / ANNOTATION_NAME
    refuse_tainted(annotations)
    if not annotations.is_file():
        raise RefusedInput(f"competition annotations not found: {annotations}")
    coco = load_coco(annotations)
    gt = build_gt_frame(coco)
    assignment = grouped_folds(gt["stem"])
    assert_no_leakage(assignment, gt)
    fingerprint = fold_fingerprint(assignment)
    if fingerprint != REFERENCE_FINGERPRINT:
        raise RefusedInput(f"the folds of these annotations are not the reference fold ({fingerprint[:12]}...)")
    tune_stems, report_stems = validation_halves(assignment)
    val_stems = sorted(list(tune_stems) + list(report_stems))
    train_stems = sorted(stem for stem, fold in assignment.items() if fold != VALIDATION_FOLD)
    config, examples_011, proposals_011 = check_refiner_run(ARGS.refiner_run, train_stems, val_stems)
    stems = list(config["train_stems"])                        # train_refiner's order: its examples follow it
    used = stems[:SMOKE_IMAGES] if SMOKE else stems
    per_stem_gt = training_ground_truth(coco)
    control = construction_control(proposals_011, examples_011, per_stem_gt, used)
    pool_011 = [row for row in examples_011 if row["stem"] in set(used)] if SMOKE else examples_011
    print(f"fold {fingerprint[:12]}...: {len(train_stems)} training images ({len(used)} used), "
          f"{len(val_stems)} held out; pool 011 rebuilt exactly from its proposals ({control['rows']} rows)",
          flush=True)

    # the second pool: the 2048 px detector's proposals on the same images
    published = pipeline.verify_manifest(configuration=DETECTOR_CONFIGURATION)["detector"]
    detector_path = published if ARGS.detector is None else Path(ARGS.detector)
    refuse_tainted(detector_path)
    if not Path(detector_path).is_file():
        raise RefusedInput(f"detector weights not found: {detector_path}")
    detector_sha256 = file_sha256(detector_path)
    is_published = detector_sha256 == file_sha256(published)
    print("2048 px detector: " + ("the published one (configuration 025), verified against the manifest"
                                  if is_published else f"not the published one ({detector_sha256[:12]}...)"),
          flush=True)
    model = detector.load_detector(detector_path)
    train_args = model.ckpt.get("train_args") or {}
    if int(train_args.get("imgsz", -1)) != PROPOSER.imgsz:
        raise RefusedInput(f"the detector was trained at imgsz {train_args.get('imgsz')}, not {PROPOSER.imgsz}")
    proposer_device = device if device.type == "cuda" else torch.device("cpu")     # ultralytics: CUDA or the CPU
    proposals_012 = propose_training_images(model, used, Path(ARGS.data_root) / "train" / "train_images",
                                            out / "proposals_2048", proposals_key(detector_sha256, proposer_device),
                                            proposer_device)
    del model
    gc.collect()
    pool_012 = build_examples(proposals_012, per_stem_gt, used)
    examples = {"011": pool_011, "012": pool_012}
    counts = {name: check_examples(rows, train_stems, val_stems, f"pool {name}") for name, rows in examples.items()}
    write_json_atomic(out / "training_proposals_2048.json", proposals_012, indent=None)
    write_json_atomic(out / "training_examples_2048.json", pool_012, indent=None)
    print(f"pools: 011 {counts['011']['positives']} + {counts['011']['negatives']}, 012 {counts['012']['positives']} "
          f"+ {counts['012']['negatives']} (positives + negatives)", flush=True)

    # the starting point, strictly loaded, and the fine-tune
    if ARGS.init_refiner is None:
        pipeline.verify_manifest()                             # the published files, this refiner among them
    else:
        INIT_REFINER = Path(ARGS.init_refiner).resolve()
        INIT_REFINER_SHA256 = file_sha256(INIT_REFINER)
        print(f"starting point: not the published refiner ({INIT_REFINER_SHA256[:12]}...)", flush=True)
    init_model, init_load = load_refiner_checkpoint(INIT_REFINER, INIT_REFINER_SHA256, "cpu")
    settings = (TrainSettings(steps=SMOKE_STEPS, workers=0, log_every=1) if SMOKE
                else TrainSettings(workers=ARGS.workers))
    extras = {"init_load": init_load, "examples": counts, "started_utc": utc_now()}
    summary = fine_tune(out, init_model, examples, Path(ARGS.data_root) / "train" / "train_images", settings, device,
                        training_key(settings, examples, used), Clock(ARGS.max_hours), tile=TR.TILE, extras=extras)
    checkpoint = out / CHECKPOINT_NAME
    report = {"recipe": "the refiner fine-tuned on a 50/50 mix of two detectors' training-image proposals",
              "smoke": SMOKE, "fold_fingerprint": fingerprint, "device": str(device),
              "inputs": {"annotations_sha256": file_sha256(annotations),
                         "refiner_run": {name: file_sha256(Path(ARGS.refiner_run) / name) for name in
                                         ("config.json", "training_examples.json", "training_proposals.json")},
                         "detector": {"path": relative(detector_path), "sha256": detector_sha256,
                                      "published": is_published, "settings": dataclasses.asdict(PROPOSER)},
                         "init_refiner": {"path": relative(INIT_REFINER), "sha256": INIT_REFINER_SHA256}},
              "images": len(used), "pools": counts, "construction_control": control, "training": summary,
              "checkpoint": {"path": relative(checkpoint), "sha256": file_sha256(checkpoint),
                             "size": checkpoint.stat().st_size},
              "runtime_seconds": round(time.monotonic() - started, 1),
              "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                              "ultralytics": package_version("ultralytics")}}
    write_json_atomic(out / "run_report.json", report, indent=2)
    print(f"done: {relative(checkpoint)} (sha256 {report['checkpoint']['sha256'][:12]}...); run_report.json",
          flush=True)
    if SMOKE:
        assert summary["completed_steps"] == SMOKE_STEPS
        print("SMOKE PASS: proposals, both pools, real optimizer steps and a strict checkpoint on the CPU", flush=True)


if __name__ == "__main__":
    try:
        main()
    except FinetuneError as error:
        print(f"STOP ({type(error).__name__}, exit {error.exit_code}): {error}", flush=True)
        sys.exit(error.exit_code)
