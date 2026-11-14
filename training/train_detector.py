"""Optional retraining, step 1 of 2: the detector, YOLO11l-seg at 1792 px.

Nothing in the pipeline needs this: the notebook loads the published weights
from ``weights/``. This is the training run that produced
``weights/detector_yolo11l_seg_1792.pt``, as it ran on Kaggle, changed only to
take its paths as arguments::

    python training/train_detector.py --data-root data/MAGFiLO_1.0_Kaggle_2026 \\
        --output-dir outputs/retrain/detector

Run it as a script from the repository root; importing it starts training.

**Recipe of the second configuration** (``--recipe 025``). The same run with
YOLO11m-seg (COCO-pretrained ``yolo11m-seg.pt``) at imgsz 2048 in place of
YOLO11l-seg at 1792 produced ``weights/detector_yolo11m_seg_2048.pt``: on a
Tesla T4 the same 6-hour budget gave 74 epochs, and its tune half chose NMS
IoU 0.0 and confidence 0.30. Nothing else differs (data, targets, folds,
augmentation, budget, seed, the threshold sweep): the two published
checkpoints record the same training arguments apart from the model, imgsz
and the number of epochs the budget allowed. The second configuration reuses
the refiner of step 2, which is trained on the default detector's proposals,
so ``train_refiner.py`` takes a ``--recipe 013`` run and refuses any other.

**Hardware and time.** One CUDA GPU. The published run used a Tesla T4
(16 GB): 6 hours of training (a time budget, which gave 78 epochs) and about
25 minutes of validation and test inference. The script stops at once on a
CPU unless ``FILAMENT_ALLOW_CPU=1`` is set. ``FILAMENT_SMOKE=1`` runs the whole
flow on 8 training images with a nano model on a CPU, as a check of the code.
Internet is needed: Ultralytics downloads the COCO-pretrained starting weights
(``yolo11l-seg.pt``), and the script asks pip for ``ultralytics==8.4.126``,
which ``requirements.txt`` has already installed.

**Writes to --output-dir:** ``yolo_ds/`` (the training set in YOLO format),
``runs/seg/weights/last.pt`` (the checkpoint, published as
``weights/detector_yolo11l_seg_1792.pt``), ``run_report.json`` (folds, the
thresholds chosen on the tune half, the report-half score),
``validation_predictions.json`` (every validation proposal, which
``train_refiner.py`` reads) and ``submission.csv`` (the detector alone on the
test images).

**Changes from the archived Kaggle source**, and nothing else: this docstring;
``--data-root`` and ``--output-dir`` instead of ``/kaggle/input`` and
``/kaggle/working``, parsed before anything else so that ``--help`` needs no
GPU; ``--recipe``, which sets the model and imgsz of the second
configuration's run (two lines after ``IMGSZ``) and prints the recipe; and
references to other teams' leaderboard scores removed from one paragraph of
the next docstring, one comment and one printed line.
"""

import argparse

_parser = argparse.ArgumentParser(
    description="Retrain the YOLO11l-seg detector, or with --recipe 025 the YOLO11m-seg one "
                "(one CUDA GPU, about 6.5 hours on a Tesla T4).")
_parser.add_argument("--data-root", required=True,
                     help="the competition folder MAGFiLO_1.0_Kaggle_2026, holding train/ and test/")
_parser.add_argument("--output-dir", required=True,
                     help="where the training set, the checkpoint and the reports are written")
_parser.add_argument("--recipe", choices=("013", "025"), default="013",
                     help="013 (default): YOLO11l-seg at 1792 px, the default configuration's detector; "
                          "025: YOLO11m-seg at 2048 px, the second configuration's detector")
ARGS = _parser.parse_args()

"""BOOTSTRAP: make the assigned GPU usable before anything else runs.

Kaggle assigns a Tesla P100 (sm_60) to this account, and its preinstalled torch
2.10+cu128 ships kernels only for sm_70 and newer, so every CUDA call fails with
"no kernel image is available for execution on the device". Requesting a T4 via
`kaggle kernels push --accelerator` is silently ignored -- the flag accepts any
string, including nonsense, and changes nothing.

PyTorch's own cu121 wheels still include sm_60, so the fix is to install one and
re-exec. That costs a few minutes and needs internet enabled, and it is the
difference between a 1-2 hour run and a 12-hour one that gets cancelled.
"""

import os
import subprocess
import sys

if os.environ.get("FILAMENT_TORCH_PATCHED") != "1":
    import torch as _probe_torch

    _needs_patch = False
    if _probe_torch.cuda.is_available():
        _properties = _probe_torch.cuda.get_device_properties(0)
        _architecture = f"sm_{_properties.major}{_properties.minor}"
        if _architecture not in _probe_torch.cuda.get_arch_list():
            print(f"BOOTSTRAP: {_properties.name} is {_architecture}; torch "
                  f"{_probe_torch.__version__} lacks it. Installing a cu121 build.", flush=True)
            _needs_patch = True

    if _needs_patch:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "torch==2.5.1", "torchvision==0.20.1",
             "--index-url", "https://download.pytorch.org/whl/cu121"],
            check=True,
        )
        os.environ["FILAMENT_TORCH_PATCHED"] = "1"
        print("BOOTSTRAP: reinstalled torch, restarting the script", flush=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)

"""Solar Filament Segmentation 2026 -- YOLO instance segmentation.

This replaces the Mask R-CNN line, and the reason is evidence rather than taste.

Public solutions to this competition use YOLO instance segmentation at
1280-1792 px (one at confidence 0.30, another a YOLO11s detector with a crop
refiner), an architecture we were not using. Our Mask R-CNN line, tuned hard
for a week, sits at 0.31.

Our own measurements say why, and they were taken before this notebook was read:

*Mask resolution.* Held-out SQ is 0.655 and -- the tell -- does not improve with
detector confidence: E[IoU|hit] is 0.63-0.69 in every score bin from 0.45 to
1.00. A ceiling that flat across confidence is a resolution limit, not a
learning problem. Mask R-CNN pools each box to 14x14 and deconvolves once to
28x28, so a 500px filament has its outline quantised to ~18px steps. YOLO-seg
instead assembles every instance from prototype masks at imgsz/4 -- 320x320 at
imgsz 1280 -- which is a different order of detail for a long thin object.

*Recall.* Held-out RQ is 0.505 against an oracle 0.754, and false negatives (192)
outnumber false positives (98) two to one. The reliability curve says this
cannot be bought by lowering the threshold: candidates scoring 0.45-0.60 hit
only 5% of the time, worth 0.030 against a break-even of 0.165. The detector
simply does not propose the filaments. Both public solutions run at 1280-1792
where we ran at 1024, and small thin features are exactly what resolution buys.

So the change is architecture *and* input size together, which is also why no
amount of post-processing moved the number: six such hypotheses were falsified
in a row, each one reshaping a mask whose detail had already been discarded.

Kept from the Mask R-CNN work, because these were expensive to learn:

*Grouped folds.* Splitting on COCO record ids leaks: several annotators trace
the same physical observation, so the same sun appears on both sides of the
split. Folds are grouped by image stem.

*Honest threshold selection.* Sweeping configurations against the whole
validation fold and quoting the maximum inflated PQ by 0.046-0.054 -- the
winner's curse -- which is how a "0.353 OOF" became 0.30 on the leaderboard.
Thresholds are chosen on one half of the validation stems and reported on the
other, and the gap between the two is printed.

*Disjoint output.* Panoptic Quality is defined over a partition, and Kaggle
rejects overlapping instances outright rather than scoring them. This cost two
submissions before it was found, so overlap resolution runs inside the sweep as
well as before writing.
"""

import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FULL_SIZE = 2048
SEED = 20260816
N_FOLDS = 5
VAL_FOLD = 0

# SMOKE is set from the environment by the generated *_smoke kernel, which ships
# a byte-identical copy of this file with a header that exports FILAMENT_SMOKE=1
# and runs on CPU. That variant is free; the weekly GPU quota is not, and it has
# to shrink the *data*, not merely the epoch count, or it is too slow to ever be
# run as a check -- which is how twelve GPU-hours were once spent discovering
# that a torch bootstrap had been copied as a docstring without the code.
SMOKE = os.environ.get("FILAMENT_SMOKE") == "1"

# Sized for the data, not to be safe: the images are 2048 native, and filaments
# are thin enough that every halving of resolution costs recall directly. A
# large model at 1792 with batch 2 is ~5 GPU-hours of the 30/week, which leaves
# room to iterate.
MODEL_NAME = "yolov8n-seg.pt" if SMOKE else "yolo11l-seg.pt"
IMGSZ = 320 if SMOKE else 1792
# --recipe 025: the second configuration's detector. The smoke run stays tiny.
if ARGS.recipe == "025" and not SMOKE:
    MODEL_NAME, IMGSZ = "yolo11m-seg.pt", 2048
print(f"recipe {ARGS.recipe}: {MODEL_NAME} at imgsz {IMGSZ}", flush=True)
# A time budget leaves room for native-resolution validation and test inference.
# Ultralytics adapts the epoch schedule to this budget and saves its final model.
EPOCHS = 2 if SMOKE else 60
TRAIN_HOURS = None if SMOKE else 6.0
ULTRALYTICS_VERSION = "8.4.126"
BATCH = 2
N_VAL_STEMS = 4 if SMOKE else None
SMOKE_TRAIN_STEMS = 8
SMOKE_TEST_IMAGES = 3

# One inference pass has to serve the whole sweep, so predict generously and
# filter afterwards. MIN_KEEP_CONF must sit below every value in CONF_GRID.
MIN_KEEP_CONF = 0.05
MAX_DET = 60
MASK_THRESHOLD = 0.5

# Two-sided on purpose. Every earlier sweep in this project ended at 0.80, PQ
# rose monotonically to it every time, and 0.80 was then called the optimum --
# but it had never won a comparison, it was simply the last value tried. The
# leader runs at 0.30 and the crop-refiner write-up at 0.28, so the grid has to
# reach well below where our Mask R-CNN line settled.
CONF_GRID = (0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80)
# The leader suppresses any overlapping box (iou=0.0); the refiner write-up uses
# 0.50. Neither is assumed, and each value needs its own inference pass.
NMS_IOU_GRID = (0.5,) if SMOKE else (0.0, 0.5, 0.7)

WORK = Path(ARGS.output_dir)
WORK.mkdir(parents=True, exist_ok=True)
DATASET = WORK / "yolo_ds"

np.random.seed(SEED)
torch.manual_seed(SEED)

if SMOKE:
    print("SMOKE: whole flow on tiny data -- dataset build, training, val "
          "inference, honest sweep, submission schema", flush=True)


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = select_device()
print(f"device: {DEVICE}", flush=True)

# A GPU kernel that quietly resolves to CPU is the single most expensive failure
# this project has had: twelve hours of the weekly quota spent training on CPU
# and cancelled at epoch 5 of 12. Fail in two minutes instead.
REQUIRE_GPU = not SMOKE and os.environ.get("FILAMENT_ALLOW_CPU") != "1"
if REQUIRE_GPU and DEVICE.type != "cuda":
    raise SystemExit(
        "ABORT: this kernel requests a GPU but resolved to CPU. Training here "
        "would burn the weekly quota to no purpose. Check the torch bootstrap "
        "above (it must re-exec, not merely describe itself), or set "
        "FILAMENT_ALLOW_CPU=1 to run on CPU deliberately."
    )

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                f"ultralytics=={ULTRALYTICS_VERSION}"], check=True)
from ultralytics import YOLO   # noqa: E402

# Ultralytics otherwise phones home and writes into a read-only config dir.
os.environ["YOLO_OFFLINE"] = "0"
os.environ["WANDB_DISABLED"] = "true"


def find_data_root() -> Path:
    root = Path(ARGS.data_root)
    if (root / "test" / "test_images").is_dir():
        return root
    raise FileNotFoundError(f"{root} has no test/test_images: pass the MAGFiLO_1.0_Kaggle_2026 folder")


DATA_ROOT = find_data_root()
TRAIN_DIR = DATA_ROOT / "train" / "train_images"
TEST_DIR = DATA_ROOT / "test" / "test_images"
ANNOTATIONS = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


# ---------------------------------------------------------------------------
# Metric -- the real one, not a proxy
# ---------------------------------------------------------------------------

def to_rle_dict(counts, size=FULL_SIZE):
    payload = counts if isinstance(counts, bytes) else counts.encode("ascii")
    return {"size": [size, size], "counts": payload}


def pq_from_groups(gt_by_record, pred_by_stem):
    """PQ against every annotator's record separately, exactly as Kaggle scores.

    Each annotator traced the same sun independently, and each of their records
    is scored against our single prediction set. A filament three annotators
    drew is three chances to be right, which is why emitting a doubtful instance
    is cheaper than it first looks.
    """
    tp_ious, fp, fn = [], 0, 0
    for record, gt_rles in gt_by_record.items():
        stem = record.split("-", 1)[1]
        pred_rles = pred_by_stem.get(stem, [])
        if not gt_rles:
            fp += len(pred_rles)
            continue
        if not pred_rles:
            fn += len(gt_rles)
            continue
        ious = mask_util.iou([to_rle_dict(r) for r in pred_rles],
                             [to_rle_dict(r) for r in gt_rles], [0] * len(gt_rles))
        iou = np.asarray(ious, dtype=np.float64).reshape(len(pred_rles), len(gt_rles)).T
        hit = iou > 0.5
        tp_ious.extend(iou[hit].tolist())
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())
    tp = len(tp_ious)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {"pq": sum(tp_ious) / denominator if denominator else 0.0,
            "sq": sum(tp_ious) / tp if tp else 0.0,
            "rq": tp / denominator if denominator else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def resolve_overlaps(rles, scores, min_retained=0.5):
    """Make one image's instances disjoint, best-scoring first.

    An instance left under half its original area is dropped: it would almost
    certainly fall below IoU 0.5, and a prediction that misses costs a false
    positive *and* a false negative -- twice what omitting it costs.
    """
    if not rles:
        return []
    order = np.argsort(-np.asarray(scores, dtype=float))
    claimed = np.zeros((FULL_SIZE, FULL_SIZE), dtype=bool)
    resolved = []
    for position in order:
        decoded = mask_util.decode(to_rle_dict(rles[position])).astype(bool)
        original = int(decoded.sum())
        if original == 0:
            continue
        decoded &= ~claimed
        retained = int(decoded.sum())
        if retained == 0 or retained < min_retained * original:
            continue
        claimed |= decoded
        resolved.append(mask_util.encode(
            np.asfortranarray(decoded.astype(np.uint8)))["counts"].decode("ascii"))
    return resolved


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

with open(ANNOTATIONS, "r", encoding="utf-8") as handle:
    coco = json.load(handle)

polygons_by_record = defaultdict(list)
for annotation in coco["annotations"]:
    polygons_by_record[annotation["image_id"]].append(annotation["segmentation"])

stem_of_record = {i["id"]: i["id"].split("-", 1)[1] for i in coco["images"]}
records = [r for r in polygons_by_record if (TRAIN_DIR / f"{stem_of_record[r]}.jpeg").exists()]
stems = sorted({stem_of_record[r] for r in records})
rng = np.random.default_rng(SEED)
order = rng.permutation(len(stems))
fold_of_stem = {stems[p]: i % N_FOLDS for i, p in enumerate(order)}
fold_fingerprint = hashlib.sha256(";".join(
    f"{stem}:{fold}" for stem, fold in sorted(fold_of_stem.items())
).encode("utf-8")).hexdigest()

val_stems = sorted({s for s in stems if fold_of_stem[s] == VAL_FOLD})[:N_VAL_STEMS]
train_stems = sorted({s for s in stems if fold_of_stem[s] != VAL_FOLD})
if SMOKE:
    train_stems = train_stems[:SMOKE_TRAIN_STEMS]
assert not set(train_stems) & set(val_stems), "physical observations leak across splits"
assert all(fold_of_stem[s] != VAL_FOLD for s in train_stems)
print(f"fold fingerprint: {fold_fingerprint}", flush=True)
tune_stems, report_stems = val_stems[::2], val_stems[1::2]
print(f"train {len(train_stems)} stems | validation {len(val_stems)} "
      f"-> tune {len(tune_stems)}, report {len(report_stems)}", flush=True)


def polygons_to_rle(segmentation):
    rles = mask_util.frPyObjects(list(segmentation), FULL_SIZE, FULL_SIZE)
    merged = mask_util.merge(rles) if isinstance(rles, list) else rles
    return merged["counts"].decode("ascii")


val_set = set(val_stems)
gt_by_record = {r: [polygons_to_rle(p) for p in polygons_by_record[r]]
                for r in records if stem_of_record[r] in val_set}


def polygon_to_yolo_line(segmentation):
    """One COCO annotation as a YOLO segmentation label line, or None.

    YOLO takes a single polygon per instance, while a COCO annotation may carry
    several rings, so the largest is kept and slivers are dropped. Coordinates
    are normalised by the native 2048 and clipped: a traced outline that runs a
    fraction of a pixel off the frame is common and would otherwise be silently
    rejected by the loader, costing training instances without any error.
    """
    rings = [np.asarray(r, dtype=np.float64).reshape(-1, 2)
             for r in segmentation if len(r) >= 6]
    if not rings:
        return None
    ring = max(rings, key=len)
    flat = (ring / FULL_SIZE).clip(0.0, 1.0).reshape(-1)
    return "0 " + " ".join(f"{v:.6f}" for v in flat)


def deduplicate_polygons(record_polygons, iou_threshold=0.5):
    """One original outline per cross-annotator instance, retaining singletons.

    A cluster contains at most one instance from each annotator record. All of
    its members must overlap above the cutoff, so a chain of overlapping
    filaments cannot collapse into one target. Its representative is the
    original polygon with greatest mean IoU to the other members.
    """
    entries = []
    for record in sorted(record_polygons):
        for polygon in record_polygons[record]:
            if polygon_to_yolo_line(polygon) is None:
                continue
            rle = to_rle_dict(polygons_to_rle(polygon))
            if int(mask_util.area(rle)) > 0:
                entries.append((record, polygon, rle))
    if not entries:
        return []
    rles = [e[2] for e in entries]
    ious = np.asarray(mask_util.iou(rles, rles, [0] * len(rles)))
    groups = []
    for index, (record, _, _) in enumerate(entries):
        eligible = [g for g in groups
                    if all(entries[j][0] != record for j in g)
                    and float(ious[index, g].min()) > iou_threshold]
        if eligible:
            best = max(eligible, key=lambda g: float(ious[index, g].mean()))
            best.append(index)
        else:
            groups.append([index])
    representatives = []
    for group in groups:
        mean_iou = ious[np.ix_(group, group)].mean(axis=1)
        representatives.append(entries[group[int(np.argmax(mean_iou))]][1])
    return representatives


def write_yolo_dataset():
    """Deduplicated annotator union, one label file per physical observation."""
    if DATASET.exists():
        shutil.rmtree(DATASET)
    counts = {}
    raw_counts = {}
    for split, split_stems in (("train", train_stems), ("val", val_stems)):
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        instances = 0
        raw_instances = 0
        for stem in split_stems:
            source = TRAIN_DIR / f"{stem}.jpeg"
            link = image_dir / f"{stem}.jpeg"
            try:
                os.symlink(source, link)
            except OSError:
                shutil.copy2(source, link)
            per_record = {r: polygons_by_record[r] for r in records
                          if stem_of_record[r] == stem}
            raw_instances += sum(len(p) for p in per_record.values())
            lines = [polygon_to_yolo_line(p) for p in deduplicate_polygons(per_record)]
            instances += len(lines)
            (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[split] = (len(split_stems), instances)
        raw_counts[split] = raw_instances

    yaml_path = DATASET / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: filament\n", encoding="utf-8")
    print(f"dataset: train {counts['train'][0]} images / {counts['train'][1]} instances, "
          f"val {counts['val'][0]} images / {counts['val'][1]} instances", flush=True)
    print(f"raw annotator labels: {raw_counts}; deduplicated targets: {counts}", flush=True)
    return yaml_path, counts, raw_counts


started = time.time()
yaml_path, dataset_counts, raw_target_counts = write_yolo_dataset()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

print(f"training {MODEL_NAME} at imgsz {IMGSZ} for {EPOCHS} epochs", flush=True)
model = YOLO(MODEL_NAME)
model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    time=TRAIN_HOURS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=0 if DEVICE.type == "cuda" else "cpu",
    workers=2,
    seed=SEED,
    project=str(WORK / "runs"),
    name="seg",
    exist_ok=True,
    val=False,
    plots=False,
    verbose=True,
    # Filaments are dark absorption features on a disc with no meaningful
    # left/right or up/down orientation, so flips are free extra data. Mosaic
    # would paste four suns into one frame, which is not an image the test set
    # ever contains, so it is turned off for the last epochs.
    fliplr=0.5,
    flipud=0.5,
    degrees=180.0,
    # Mosaic pastes four images into one frame. For photographs of many objects
    # that is free variety; here every frame is one solar disc centred in it, so
    # a mosaic tile is four partial suns -- an image the test set never contains,
    # and one where a filament's position on the disc, which is most of the
    # context, becomes meaningless. Off entirely rather than merely closed late.
    mosaic=0.0,
    # Even after deduplication, distinct adjacent filaments can overlap.
    # Preserve each instance's supervision instead of overwriting its pixels.
    overlap_mask=False,
    single_cls=True,
)
weights = WORK / "runs" / "seg" / "weights" / "last.pt"
assert weights.exists(), f"training produced no weights at {weights}"
print(f"trained: {weights} ({weights.stat().st_size / 1e6:.1f} MB)", flush=True)

detector = YOLO(str(weights))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def masks_to_rles(masks, scores):
    """Model masks, whatever grid they arrive on, as full-resolution RLEs.

    The frames are square, so ultralytics letterboxes them without padding and a
    straight resize back to 2048 is exact -- which also means this does not care
    whether masks come back at the prototype size or already upsampled to imgsz.
    Squareness is asserted rather than assumed: a padded non-square mask would
    rescale into a subtly wrong place and still encode, still validate, and
    simply score badly.
    """
    rles, kept_scores = [], []
    for mask, score in zip(masks, scores):
        mask = np.asarray(mask, dtype=np.float32)
        assert mask.ndim == 2 and mask.shape[0] == mask.shape[1], (
            f"expected a square mask, got {mask.shape}")
        probability = cv2.resize(mask, (FULL_SIZE, FULL_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
        binary = (probability > MASK_THRESHOLD).astype(np.uint8)
        if binary.sum() == 0:
            continue
        rles.append(mask_util.encode(np.asfortranarray(binary))["counts"].decode("ascii"))
        kept_scores.append(float(score))
    return rles, kept_scores


def predict_instances(path, nms_iou):
    """Native masks: resize mask logits before Ultralytics binarises them.

    result.masks.data is already binary. retina_masks avoids resampling those
    binaries, which cannot restore outline detail lost at the inference grid.
    """
    result = detector.predict(
        source=str(path), imgsz=IMGSZ, conf=MIN_KEEP_CONF, iou=nms_iou,
        max_det=MAX_DET, verbose=False, retina_masks=True,
        device=0 if DEVICE.type == "cuda" else "cpu",
    )[0]
    if result.masks is None:
        return [], []
    return masks_to_rles(result.masks.data.cpu().numpy(),
                         result.boxes.conf.cpu().numpy())


cache = {}
for nms_iou in NMS_IOU_GRID:
    began = time.time()
    for index, stem in enumerate(val_stems):
        cache[(nms_iou, stem)] = predict_instances(TRAIN_DIR / f"{stem}.jpeg", nms_iou)
        if (index + 1) % 10 == 0:
            print(f"  nms {nms_iou} val {index + 1}/{len(val_stems)} "
                  f"{(time.time() - began) / (index + 1):.1f}s/img", flush=True)


def build(nms_iou, stem, conf):
    rles, scores = cache[(nms_iou, stem)]
    keep = [i for i, s in enumerate(scores) if s >= conf]
    return resolve_overlaps([rles[i] for i in keep], [scores[i] for i in keep])


def score_on(subset, nms_iou, conf):
    chosen = set(subset)
    gt = {r: v for r, v in gt_by_record.items() if stem_of_record[r] in chosen}
    return pq_from_groups(gt, {s: build(nms_iou, s, conf) for s in subset})


# ---------------------------------------------------------------------------
# Honest selection: choose on one half, report on the other
# ---------------------------------------------------------------------------

print()
print(" nms_iou   conf |     PQ     SQ     RQ |    TP     FP     FN")
best = None
for nms_iou in NMS_IOU_GRID:
    for conf in CONF_GRID:
        scores = score_on(tune_stems, nms_iou, conf)
        print(f"    {nms_iou:.2f}   {conf:.2f} | {scores['pq']:.4f} {scores['sq']:.4f} "
              f"{scores['rq']:.4f} | {scores['tp']:5d} {scores['fp']:6d} {scores['fn']:5d}")
        if best is None or scores["pq"] > best[0]["pq"]:
            best = (scores, nms_iou, conf)

tune_scores, nms_iou, conf = best
report_scores = score_on(report_stems, nms_iou, conf)
at_edge = conf in (CONF_GRID[0], CONF_GRID[-1]) or (
    len(NMS_IOU_GRID) > 1 and nms_iou in (NMS_IOU_GRID[0], NMS_IOU_GRID[-1]))

print()
print(f"chosen on the tune half: nms_iou={nms_iou} conf={conf}")
print(f"  tune half   PQ={tune_scores['pq']:.4f}")
print(f"  REPORT half PQ={report_scores['pq']:.4f}   <- the honest number")
print(f"  selection bias {tune_scores['pq'] - report_scores['pq']:+.4f}")
if at_edge:
    print("  WARNING: the winner sits at the edge of its grid, so it never won a "
          "two-sided comparison -- widen the grid before believing it")
print("reference points -- all-observation union oracle 0.767; "
      "human agreement 0.342, single expert oracle 0.598")

# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

test_paths = sorted(TEST_DIR.glob("*.jpeg"))
if SMOKE:
    test_paths = test_paths[:SMOKE_TEST_IMAGES]
print()
print(f"predicting {len(test_paths)} test images", flush=True)

rows = []
for index, path in enumerate(test_paths):
    rles, scores = predict_instances(path, nms_iou)
    keep = [i for i, s in enumerate(scores) if s >= conf]
    for k, rle in enumerate(resolve_overlaps([rles[i] for i in keep], [scores[i] for i in keep])):
        rows.append({"filament_id": f"{path.stem}_{k}", "segmentation_rle": rle})
    if (index + 1) % 25 == 0:
        print(f"  {index + 1}/{len(test_paths)}", flush=True)

submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
submission.to_csv(WORK / "submission.csv", index=False)
covered = {r["filament_id"].rsplit("_", 1)[0] for r in rows}

Path(WORK / "run_report.json").write_text(json.dumps({
    "experiment": "YOLO instance segmentation, thresholds chosen honestly",
    "model": MODEL_NAME, "imgsz": IMGSZ, "epochs": EPOCHS,
    "completed_epochs": len(pd.read_csv(WORK / "runs" / "seg" / "results.csv")),
    "train_hours_budget": TRAIN_HOURS, "ultralytics_version": ULTRALYTICS_VERSION,
    "fold_fingerprint": fold_fingerprint, "seed": SEED, "val_fold": VAL_FOLD,
    "train_stems": train_stems, "val_stems": val_stems,
    "tune_stems": tune_stems, "report_stems": report_stems,
    "target_policy": "complete-link cross-annotator union, original medoid outline",
    "raw_target_counts": raw_target_counts,
    "train_images": dataset_counts["train"][0], "train_instances": dataset_counts["train"][1],
    "nms_iou": nms_iou, "conf": conf, "mask_threshold": MASK_THRESHOLD,
    "tune_half": tune_scores, "report_half": report_scores,
    "selection_bias": tune_scores["pq"] - report_scores["pq"],
    "threshold_at_grid_edge": at_edge,
    "n_rows": int(len(submission)), "n_images_covered": len(covered),
    "runtime_s": round(time.time() - started, 1),
}, indent=2), encoding="utf-8")
Path(WORK / "validation_predictions.json").write_text(json.dumps({
    "fold_fingerprint": fold_fingerprint,
    "predictions": [{"nms_iou": n, "stem": s, "rles": r, "scores": c}
                    for (n, s), (r, c) in cache.items()],
}), encoding="utf-8")

print(f"wrote submission.csv with {len(submission)} rows over {len(covered)} images")
print(f"honest report-half PQ {report_scores['pq']:.4f}")

if SMOKE:
    # Prove the artifact, not merely that the code reached the end. These are
    # the two failures that actually cost submissions: overlapping instances,
    # and RLEs encoded at the wrong resolution.
    decoded = [mask_util.decode(to_rle_dict(r)) for r in submission["segmentation_rle"]]
    for fid, mask in zip(submission["filament_id"], decoded):
        assert mask.shape == (FULL_SIZE, FULL_SIZE), f"{fid} is not {FULL_SIZE}x{FULL_SIZE}"
    by_image = defaultdict(list)
    for fid, mask in zip(submission["filament_id"], decoded):
        by_image[fid.rsplit("_", 1)[0]].append(mask)
    for stem, masks in by_image.items():
        assert np.sum(masks, axis=0).max() <= 1, f"{stem} overlaps -- Kaggle rejects this"
    for rle in submission["segmentation_rle"]:
        again = mask_util.encode(np.asfortranarray(mask_util.decode(to_rle_dict(rle))))
        assert again["counts"].decode("ascii") == rle, "RLE is not canonical"
    assert dataset_counts["train"][1] > 0, "no training instances were written"
    # An empty submission satisfies every loop above vacuously, so the gate
    # would read SMOKE PASS while proving nothing at all. Two epochs on eight
    # images cannot be required to *detect* anything, but the emission path
    # must still be shown to carry a real instance end to end.
    # Two epochs of a nano model on eight images cannot be *required* to detect
    # anything, so demanding a real detection here only made the gate fail for
    # the wrong reason. What must be proven is the conversion: a model mask on
    # its own grid has to land in the right place at 2048.
    probe_masks = np.zeros((1, 64, 64), dtype=np.float32)
    probe_masks[0, 16:32, 40:48] = 1.0
    probe_rles, _ = masks_to_rles(probe_masks, [0.9])
    assert probe_rles, "a solid blob survived neither the resize nor the threshold"
    probe = mask_util.decode(to_rle_dict(probe_rles[0]))
    assert probe.shape == (FULL_SIZE, FULL_SIZE)
    rows, columns = np.nonzero(probe)
    assert abs(rows.mean() - FULL_SIZE * 24 / 64) < 32, "mask landed at the wrong row"
    assert abs(columns.mean() - FULL_SIZE * 44 / 64) < 32, "mask landed at the wrong column"
    print()
    print(f"SMOKE PASS: every stage ran and the submission is valid "
          f"({len(submission)} instances, {len(by_image)} images, disjoint, canonical). "
          "This flow is now safe to run on GPU quota.")
