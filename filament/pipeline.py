"""The whole model: detector proposals, refined at native resolution.

::

    from filament import pipeline

    instances = pipeline.predict_image(image)         # one frame -> [(mask_2048, score), ...]
    predictions = pipeline.predict_dataset(folder)    # every frame -> {stem: [rle, ...]}, final

For one ``(2048, 2048)`` uint8 frame (:func:`filament.io.load_image`):

1. **Detector** (:mod:`filament.models.detector`): YOLO11l-seg at 1792 px
   proposes filaments with a confidence and a native-resolution mask.
   Proposals below confidence 0.3 are discarded.
2. **Refiner** (:mod:`filament.models.refiner`): each remaining mask is
   corrected on native-resolution tiles around it. A proposal whose refined
   mask is empty is discarded.
3. **Post-processing** (:func:`filament.submission.postprocess`): the
   instances are made disjoint best-score-first, and an instance left with
   less than half of its pixels is discarded. :func:`predict_dataset` does
   this step; after :func:`predict_image` the caller does.

**Configurations.** Five models are published (:data:`CONFIGURATIONS`),
named after the runs that evaluated them. ``"013"``, the default, is the one
above. ``"025"`` uses YOLO11m-seg at 2048 px as the detector (NMS IoU 0.0, the
same 0.05 floor, 60 proposals and 0.3 operating point) and the same refiner
weights at alpha 1.5. ``"032"`` is ``"013"`` in every setting, with a detector
trained by the same recipe on all 707 training images. ``"013ft"`` and
``"035"`` are ``"013"`` and ``"032"`` with the fine-tuned refiner weights
(``training/finetune_refiner.py``) at the same alpha 1.0. Tiling and
post-processing are the same in all five.
``load_pipeline(configuration="025")`` or
``predict_dataset(folder, configuration="025")`` selects one.

Weights live in ``weights/`` next to the package. :func:`load_pipeline` checks
every file of the chosen configuration for its size and SHA-256 against
``weights/manifest.json`` before either network is built, so a missing,
truncated, corrupted or Git-LFS-pointer file stops the run instead of
producing predictions from the wrong weights. Another configuration's
weight files are never read.

With the published weights and ``requirements.txt`` this reproduces the
recorded validation predictions; on hardware other than the GPU they were made
on, a few boundary pixels can differ (the refiner ran in float16 there and runs
in float32 on a CPU).
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

from .io import (IMAGE_SIZE, LFS_POINTER_PREFIX, file_sha256, list_images, load_image, rle_to_mask,
                 verify_weights)
from .models import detector, refiner
from .models.detector import DETECTOR, DetectorConfig
from .models.refiner import PRIOR_LOGIT, REFINER, RefinerConfig, to_rle_dict
from .submission import postprocess

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
MANIFEST_NAME = "manifest.json"
ROLES = ("detector", "refiner")
# The published file name of each role's weights.
WEIGHT_FILES = {"detector": "detector_yolo11l_seg_1792.pt", "refiner": "refiner_native_residual.pt"}

# Post-processing: an instance that keeps less than this share of its pixels
# after better-scoring instances have claimed theirs is dropped.
MIN_RETAINED = 0.5


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Configuration:
    """One published model: its name, every inference setting, and what it is in one line.

    Its weight files are listed in ``weights/manifest.json``: the default
    configuration's under ``files``, any other's under
    ``configurations.<name>.files``.
    """

    name: str
    detector: DetectorConfig
    refiner: RefinerConfig
    description: str


# Named after the runs that evaluated them. Every function uses the default
# unless it is told otherwise. "025"'s settings are the ones its evaluation run
# used: the detector's NMS IoU and confidence chosen on the tune half by the
# detector's own training run, alpha chosen on the tune half from 0.5 / 1 / 1.5.
# "032" shares the default's settings objects: only its detector weights differ.
# "013ft" and "035" share them too: they are "013" and "032" with the fine-tuned
# refiner weights, at the alpha the fine-tune's evaluation chose on the tune half.
DEFAULT_CONFIGURATION = "013"
CONFIGURATIONS = {
    "013": Configuration("013", DETECTOR, REFINER,
                         "YOLO11l-seg at 1792 px, NMS IoU 0.5; the refiner at alpha 1.0"),
    "025": Configuration(
        "025",
        DetectorConfig(imgsz=2048, conf_floor=0.05, nms_iou=0.0, max_det=60, mask_threshold=0.5,
                       confidence=0.3),
        RefinerConfig(tile=512, margin=64, batch=4, background_sigma=15.0, alpha=1.5, taper_floor=0.05),
        "YOLO11m-seg at 2048 px, NMS IoU 0.0; the same refiner at alpha 1.5"),
    "032": Configuration("032", DETECTOR, REFINER,
                         "the default with its detector trained on all 707 training images"),
    "013ft": Configuration("013ft", DETECTOR, REFINER,
                           "the default with the fine-tuned refiner at alpha 1.0"),
    "035": Configuration("035", DETECTOR, REFINER,
                         "\"032\" with the fine-tuned refiner at alpha 1.0"),
}


def configuration_named(name: str) -> Configuration:
    """The published configuration called ``name``; any other name raises ``ValueError``."""
    if name not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration {name!r}; the published ones are {sorted(CONFIGURATIONS)}")
    return CONFIGURATIONS[name]


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------

def _manifest_entries(weights_dir: Path, configuration: str = DEFAULT_CONFIGURATION) -> List[dict]:
    manifest_path = Path(weights_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {Path(weights_dir).name}/")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if configuration == DEFAULT_CONFIGURATION:
        where, entries = MANIFEST_NAME, manifest.get("files") or []
    else:
        listed = (manifest.get("configurations") or {}).get(configuration)
        if not isinstance(listed, dict):
            raise ValueError(f"{MANIFEST_NAME} lists no weights for configuration {configuration!r}")
        where, entries = f"{MANIFEST_NAME} (configuration {configuration!r})", listed.get("files") or []
    roles = sorted(str(entry.get("role")) for entry in entries)
    if roles != sorted(ROLES):
        raise ValueError(f"{where} must list one weight file for each of {ROLES}, "
                         f"found roles {roles}")
    for entry in entries:
        missing = {"name", "role", "size", "sha256"} - set(entry)
        if missing:
            raise ValueError(f"{where}: entry {entry.get('name')!r} lacks {sorted(missing)}")
    return entries


def verify_manifest(weights_dir: Path = WEIGHTS_DIR,
                    configuration: str = DEFAULT_CONFIGURATION) -> Dict[str, Path]:
    """Check one configuration's weight files against ``weights/manifest.json``; ``{role: path}``.

    The manifest is ``{"files": [{"name", "role", "size", "sha256", ...}, ...],
    "configurations": {name: {"files": [...]}}}``: ``files`` lists the default
    configuration's weights and ``configurations`` the others', each with one
    file per role in :data:`ROLES`. Only the chosen configuration's files are
    read. Each must exist, must not be a Git LFS pointer, must have the
    recorded size in bytes, and must match its SHA-256. Any failure raises;
    nothing is loaded here.
    """
    configuration_named(configuration)           # an unknown name fails before any file is read
    weights_dir = Path(weights_dir)
    verified = {}
    for entry in _manifest_entries(weights_dir, configuration):
        path = weights_dir / entry["name"]
        if not path.is_file():
            raise FileNotFoundError(f"weight file missing: {weights_dir.name}/{entry['name']}")
        size = path.stat().st_size
        if size != entry["size"]:
            with open(path, "rb") as handle:
                if handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                    raise RuntimeError(f"{path.name} is a Git LFS pointer, not the weights. "
                                       "Install git-lfs and run `git lfs pull` in the repository.")
            raise ValueError(f"{path.name} has {size} bytes, expected {entry['size']}: "
                             "the file is truncated or not the published one")
        verified[entry["role"]] = verify_weights(path, entry["sha256"])
    return verified


def write_manifest(weights_dir: Path, checkpoints: Dict[str, Path]) -> Path:
    """Set up retrained weights to load exactly like the published ones.

    ``checkpoints`` maps each role in :data:`ROLES` to a freshly trained file
    (``runs/seg/weights/last.pt`` of ``training/train_detector.py`` and
    ``refiner_last.pt`` of ``training/train_refiner.py``). They are copied into
    ``weights_dir`` under the published names, and a ``manifest.json`` with their
    sizes and SHA-256 is written next to them; returns ``weights_dir``.
    """
    weights_dir = Path(weights_dir)
    missing = [str(checkpoints[role]) for role in ROLES if not Path(checkpoints[role]).is_file()]
    if missing:
        raise FileNotFoundError(f"trained weights not found: {missing}")
    weights_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for role in ROLES:
        target = weights_dir / WEIGHT_FILES[role]
        shutil.copyfile(checkpoints[role], target)
        entries.append({"name": target.name, "role": role, "size": target.stat().st_size,
                        "sha256": file_sha256(target)})
    (weights_dir / MANIFEST_NAME).write_text(json.dumps({"files": entries}, indent=2) + "\n",
                                             encoding="utf-8")
    return weights_dir


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@dataclass
class Pipeline:
    """Both networks, the device they run on, and the settings they run with."""

    detector: object                     # ultralytics.YOLO
    refiner: torch.nn.Module             # filament.models.refiner.NativeResidualUNet
    device: torch.device
    detector_config: DetectorConfig = DETECTOR
    refiner_config: RefinerConfig = REFINER
    weights: Dict[str, str] = field(default_factory=dict)   # role -> sha256 of the loaded file
    configuration: str = DEFAULT_CONFIGURATION              # the name in CONFIGURATIONS


def select_device(device: Optional[str | torch.device] = None) -> torch.device:
    """``None`` picks the first CUDA GPU if there is one, else the CPU."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_pipeline(weights_dir: Path = WEIGHTS_DIR,
                  device: Optional[str | torch.device] = None,
                  configuration: str = DEFAULT_CONFIGURATION) -> Pipeline:
    """Verify one configuration's weight files, then build both networks on ``device``.

    ``configuration`` is a name in :data:`CONFIGURATIONS`; the default is the
    published default model.
    """
    chosen = configuration_named(configuration)
    paths = verify_manifest(weights_dir, chosen.name)          # raises before anything is loaded
    entries = {entry["role"]: entry for entry in _manifest_entries(weights_dir, chosen.name)}
    device = select_device(device)
    return Pipeline(
        detector=detector.load_detector(paths["detector"]),
        refiner=refiner.load_refiner(paths["refiner"], device),
        device=device,
        detector_config=chosen.detector,
        refiner_config=chosen.refiner,
        weights={role: entries[role]["sha256"] for role in ROLES},
        configuration=chosen.name,
    )


@lru_cache(maxsize=1)
def default_pipeline() -> Pipeline:
    """The published pipeline, loaded once per process."""
    return load_pipeline()


@lru_cache(maxsize=None)
def published_pipeline(configuration: str = DEFAULT_CONFIGURATION) -> Pipeline:
    """A published configuration's pipeline, loaded once per process.

    The default configuration is :func:`default_pipeline`'s object, so it is
    never loaded twice.
    """
    chosen = configuration_named(configuration)
    if chosen.name == DEFAULT_CONFIGURATION:
        return default_pipeline()
    return load_pipeline(configuration=chosen.name)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def predict_instances(image: np.ndarray, pipe: Optional[Pipeline] = None) -> List[Tuple[str, float]]:
    """Instances of one frame as ``[(rle, score), ...]``, before overlap resolution.

    Steps 1 and 2 of the module docstring. Masks are compressed COCO RLE at
    2048 x 2048, best score first; they may overlap.
    """
    if image.dtype != np.uint8 or image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected a ({IMAGE_SIZE}, {IMAGE_SIZE}) uint8 frame, "
                         f"got {image.shape} {image.dtype}")
    pipe = pipe or default_pipeline()
    proposals = detector.propose(pipe.detector, image, pipe.detector_config, pipe.device)
    kept = [(rle, score) for rle, score in proposals if score >= pipe.detector_config.confidence]
    return refiner.refine_proposals(pipe.refiner, image, kept, pipe.device, pipe.refiner_config)


def predict_image(image: np.ndarray, pipe: Optional[Pipeline] = None) -> List[Tuple[np.ndarray, float]]:
    """Instances of one ``(2048, 2048)`` uint8 frame: ``[(mask_2048, score), ...]``.

    Masks are boolean at native resolution, best score first, and may overlap;
    :func:`filament.submission.postprocess` makes them disjoint.
    """
    return [(rle_to_mask(rle), score) for rle, score in predict_instances(image, pipe)]


def predict_dataset(
    image_dir: Path,
    stems: Optional[Iterable[str]] = None,
    pipe: Optional[Pipeline] = None,
    min_area: int = 1,
    timings: Optional[Dict[str, float]] = None,
    log_every: int = 10,
    configuration: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Final, disjoint predictions for every image: ``{stem: [rle, ...]}``.

    ``stems`` selects ``<stem>.jpeg`` files in ``image_dir`` (all images when
    ``None``, in sorted order). ``timings``, if given, receives the seconds
    spent on each image. Progress is printed every ``log_every`` images
    (0 is silent). ``configuration`` names a published configuration
    (:data:`CONFIGURATIONS`), loaded once per process, when no ``pipe`` is
    given; given both, they must agree. Given neither, the default runs.
    """
    if configuration is not None:
        configuration_named(configuration)
    image_dir = Path(image_dir)
    if stems is None:
        paths = list_images(image_dir)
    else:
        paths = [image_dir / f"{stem}.jpeg" for stem in stems]
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} of {len(paths)} images are missing from "
                                    f"{image_dir.name}/, e.g. {missing[:3]}")
    if configuration is None:
        pipe = pipe or default_pipeline()
    elif pipe is None:
        pipe = published_pipeline(configuration)
    elif pipe.configuration != configuration:
        raise ValueError(f"the pipeline given is configuration {pipe.configuration!r}, "
                         f"not {configuration!r}")
    predictions: Dict[str, List[str]] = {}
    began = time.perf_counter()
    for index, path in enumerate(paths, start=1):
        start = time.perf_counter()
        instances = predict_instances(load_image(path), pipe)
        predictions[path.stem] = postprocess(instances, score_threshold=None, min_area=min_area,
                                             min_retained=MIN_RETAINED)
        if timings is not None:
            timings[path.stem] = time.perf_counter() - start
        if log_every and (index % log_every == 0 or index == len(paths)):
            print(f"  {index}/{len(paths)} images, "
                  f"{(time.perf_counter() - began) / index:.1f} s per image", flush=True)
    return predictions


def compare_predictions(ours: Dict[str, List[str]], theirs: Dict[str, List[str]]) -> pd.DataFrame:
    """How far apart two prediction sets are, one row per image (sorted by stem).

    Both sets must be final (disjoint within an image), so pairing instances
    one-to-one at IoU > 0.5 is unambiguous. Columns: ``ours`` / ``theirs``
    (instance counts), ``identical`` (the same RLE strings in the same order),
    ``matched`` (pairs at IoU > 0.5), ``mean_iou`` / ``min_iou`` over the pairs
    (NaN without one), and ``pixels_differing`` (pixels in exactly one of the
    two sets' unions).
    """
    rows = []
    for stem in sorted(set(ours) | set(theirs)):
        a, b = list(ours.get(stem, [])), list(theirs.get(stem, []))
        row = {"stem": stem, "ours": len(a), "theirs": len(b), "identical": a == b,
               "matched": 0, "mean_iou": np.nan, "min_iou": np.nan, "pixels_differing": 0}
        if a and b:
            ious = np.asarray(mask_util.iou([to_rle_dict(x) for x in a], [to_rle_dict(y) for y in b],
                                            [0] * len(b))).reshape(len(a), len(b))
            pairs = ious[ious > 0.5]
            if pairs.size:
                row.update(matched=int(pairs.size), mean_iou=float(pairs.mean()), min_iou=float(pairs.min()))
        if not row["identical"]:
            unions = []
            for rles in (a, b):
                canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
                for rle in rles:
                    canvas |= rle_to_mask(rle)
                unions.append(canvas)
            row["pixels_differing"] = int((unions[0] ^ unions[1]).sum())
        rows.append(row)
    return pd.DataFrame(rows, columns=["stem", "ours", "theirs", "identical", "matched",
                                       "mean_iou", "min_iou", "pixels_differing"])


# --------------------------------------------------------------------------
# Documentation
# --------------------------------------------------------------------------

def parameters(detector_config: DetectorConfig = DETECTOR,
               refiner_config: RefinerConfig = REFINER) -> pd.DataFrame:
    """Every setting between the raw frame and the submitted masks, one row each."""
    d, r = detector_config, refiner_config
    rows = [
        ("input", "frame", f"{IMAGE_SIZE} x {IMAGE_SIZE} uint8, one channel",
         "native resolution; every output mask is predicted at it"),
        ("detector", "channels", "gray replicated to 3", "byte-identical to Ultralytics reading the JPEG"),
        ("detector", "imgsz", d.imgsz, "frame resized 2048 -> 1792 (bilinear) by Ultralytics, /255"),
        ("detector", "conf_floor", d.conf_floor, "Ultralytics conf: proposals kept from here"),
        ("detector", "nms_iou", d.nms_iou, "chosen on the tune half from 0.0 / 0.5 / 0.7"),
        ("detector", "max_det", d.max_det, "proposals per frame at most"),
        ("detector", "retina_masks", True, "mask logits upsampled to 2048 before thresholding"),
        ("detector", "mask_threshold", d.mask_threshold, "probability"),
        ("detector", "confidence", d.confidence, "operating point, chosen on the tune half from 0.10-0.80"),
        ("refiner", "tile", r.tile, "native pixels, no resizing"),
        ("refiner", "stride", r.tile // 2, "tiles overlap by half"),
        ("refiner", "margin", r.margin, "pixels around the proposal's bounding box"),
        ("refiner", "batch", r.batch, "tiles per forward pass"),
        ("refiner", "taper_floor", r.taper_floor, "Hann blending weight at tile borders"),
        ("refiner", "channel 0", "gray / 255", ""),
        ("refiner", "background_sigma", r.background_sigma, "Gaussian blur behind channel 1"),
        ("refiner", "channel 1", "clip((background - gray) / 32, -2, 2)", "local contrast"),
        ("refiner", "channel 2", "proposal mask (0 / 1)", "exactly the detector's"),
        ("refiner", "channel 3", "clip(signed distance / 32, -1, 1)", "to the proposal outline"),
        ("refiner", "prior_logit", round(PRIOR_LOGIT, 4), "proposal as logits: +inside / -outside"),
        ("refiner", "alpha", r.alpha, "weight of the correction; chosen on the tune half from 0 / 0.5 / 1"),
        ("post-processing", "overlap order", "score, best first", "each instance keeps unclaimed pixels"),
        ("post-processing", "min_retained", MIN_RETAINED, "share of its pixels an instance must keep"),
        ("post-processing", "min_area", 1, "pixels"),
    ]
    return pd.DataFrame(rows, columns=["stage", "parameter", "value", "note"])
