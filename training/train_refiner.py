"""Optional retraining, step 2 of 2: the native residual refiner.

Nothing in the pipeline needs this: the notebook loads the published weights
from ``weights/``. This is the training run that produced
``weights/refiner_native_residual.pt``, as it ran on Kaggle, changed only to
take its paths as arguments. It reads the output folder of
``train_detector.py``::

    python training/train_refiner.py --data-root data/MAGFiLO_1.0_Kaggle_2026 \\
        --detector-run outputs/retrain/detector --output-dir outputs/retrain/refiner

**Hardware and time.** One CUDA GPU; the script stops on a CPU. The published
run trained for 4 hours (a time budget: 31,743 steps) and took 4.4 hours in
all, with the detector's proposals on the training, validation and test images.
``FILAMENT_SMOKE=1`` runs the whole flow on a few images on a CPU, as a check of
the code. Internet is needed only for pip to confirm ``ultralytics==8.4.126``.

**What it does.** The frozen detector proposes filaments on the 565 training
images, and each proposal is matched to the annotators' outlines. The refiner
is trained on native-resolution crops (the recipe is in the docstring of
``filament/models/refiner.py``). The detector's validation proposals are then
refined with alpha 0, 0.5 and 1.0, alpha is chosen on the tune half, the report
half is scored once with a paired bootstrap against the detector alone, and the
test images are predicted. The run refuses to start if the detector run chose
other thresholds than confidence 0.3 and NMS IoU 0.5, or if the detector cannot
reproduce that run's test predictions. No competition submission is sent by
this runner.

**Writes to --output-dir:** ``refiner_last.pt`` (the checkpoint, published as
``weights/refiner_native_residual.pt``), ``config.json``,
``evaluation_report.json`` and ``run_report.json`` (scores, the chosen alpha,
the bootstrap interval), ``validation_alpha_*.json`` (validation predictions per
alpha), ``submission.csv``, ``training_history.csv`` and the cached proposals.

**Changes from the archived Kaggle source**, and nothing else:

* this docstring; ``--data-root``, ``--detector-run`` and ``--output-dir``
  instead of searching ``/kaggle/input`` and writing to ``/kaggle/working``,
  parsed before anything else so that ``--help`` needs no GPU; the Kaggle-only
  guard that refused to run anywhere else is gone;
* the network, its loss and self-test, ``patch_features`` and ``tile_starts``
  are imported from ``filament.models.refiner`` instead of being embedded: the
  same code, byte for byte, so training and inference share one implementation;
* the detector run need not be the published one: its files are hashed and the
  hashes compared with the published run's, and the outcome is printed instead
  of stopping the run. When alpha 0 wins, the output is checked against that
  run's own ``submission.csv`` instead of the published file's hash.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import defaultdict, OrderedDict


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
        description="Retrain the native residual refiner (one CUDA GPU, about 4.5 hours on a Tesla T4).")
    parser.add_argument("--data-root", required=True,
                        help="the competition folder MAGFiLO_1.0_Kaggle_2026, holding train/ and test/")
    parser.add_argument("--detector-run", required=True,
                        help="output folder of train_detector.py: run_report.json, validation_predictions.json, "
                             "submission.csv and runs/seg/weights/last.pt")
    parser.add_argument("--output-dir", required=True, help="where the checkpoint and the reports are written")
    return parser.parse_args(argv)


if __name__ == "__main__":
    ARGS = parse_args()
    bootstrap_torch()


# The network, its loss and self-test, and the feature and tiling functions come
# from the package (filament/models/refiner.py), byte for byte the block this
# runner embedded on Kaggle, so training and inference share one implementation.
REFINER_MODEL_SOURCE_SHA256 = "9036fe0eeea89edf26222ec21f46a67312850092213442b2d9dbf974e2e976b8"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from filament.models.refiner import (  # noqa: E402
    PRIOR_LOGIT,
    NativeResidualUNet,
    native_refiner_loss,
    patch_features,
    prior_logits_from_input,
    self_test,
    tile_starts,
)


import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader, Dataset


FULL_SIZE = 2048
SEED = 20261003
FOLD_SEED = 20260816
SMOKE = os.environ.get("FILAMENT_SMOKE") == "1"
REQUIRE_GPU = not SMOKE
FINGERPRINT = "b990dbd4719315540906c0389c5edc44d408f7c853f79b1cfcc711f625d7bdee"
# The published detector run's files; WORK and SOURCE_REF are set from the
# command line at the end of this file.
SOURCE_HASHES = {
    "run_report.json": "379273446ea8f201de79439025e7bf4ca6672c1fb8dfd0a1969df6a8c0b98487",
    "validation_predictions.json": "39b5843bb9832d9ed3b18a3c77c612d42fe9073c6ae1e3562e61d1687c553a15",
    "submission.csv": "ac5fcd1e4a55846f5755f3c948b88924075ea0acaf7f9918cc3e2f036c270dff",
}
WEIGHTS_SHA256 = "183a841307827f2167a8d3d981c14df9d1ccbce9147c6a7d60a3536af8bb6eed"
DETECTOR_CONF = 0.3
DETECTOR_NMS = 0.5
DETECTOR_FLOOR = 0.05
ULTRALYTICS_VERSION = "8.4.126"
ALPHAS = (0.0, 0.5, 1.0)
TILE = 128 if SMOKE else 512
BATCH = 2 if SMOKE else 4
BASE_CHANNELS = 8 if SMOKE else 24
TRAIN_SECONDS = 90.0 if SMOKE else 4 * 3600.0
MAX_STEPS = 4 if SMOKE else 40000
LR = 3e-4
RESIDUAL_WEIGHT = 0.001


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prediction_hash(predictions):
    payload = json.dumps(predictions, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_rle_dict(counts):
    return {"size": [FULL_SIZE, FULL_SIZE],
            "counts": counts if isinstance(counts, bytes) else counts.encode("ascii")}


def encode_mask(mask):
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def decode_mask(rle):
    return mask_util.decode(to_rle_dict(rle)).astype(np.uint8)


def resolve_overlaps(rles, scores, min_retained=0.5):
    """Verbatim algorithm/order from the source YOLO evaluation."""
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
        resolved.append(encode_mask(decoded))
    return resolved


def contributions(gt_by_record, predictions, stems):
    """Per physical image: sum IoU, TP, FP, FN; strict >.5, every hit pair."""
    result = {stem: np.zeros(4, dtype=np.float64) for stem in stems}
    for record, gt in gt_by_record.items():
        stem = record.split("-", 1)[1]
        if stem not in result:
            continue
        pred = predictions.get(stem, [])
        if not gt:
            result[stem][2] += len(pred)
        elif not pred:
            result[stem][3] += len(gt)
        else:
            iou = np.asarray(mask_util.iou(
                [to_rle_dict(r) for r in pred], [to_rle_dict(r) for r in gt], [0] * len(gt)
            ), dtype=np.float64).reshape(len(pred), len(gt)).T
            hit = iou > 0.5
            result[stem] += [float(iou[hit].sum()), int(hit.sum()),
                             int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum())]
    return result


def score_contributions(per_stem):
    total = np.sum(list(per_stem.values()), axis=0) if per_stem else np.zeros(4)
    numerator, tp, fp, fn = total
    denominator = tp + 0.5 * (fp + fn)
    return {"pq": float(numerator / denominator) if denominator else 0.0,
            "sq": float(numerator / tp) if tp else 0.0,
            "rq": float(tp / denominator) if denominator else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def paired_bootstrap(baseline, candidate, repetitions=2000):
    stems = sorted(baseline)
    assert stems == sorted(candidate)
    base = np.asarray([baseline[s] for s in stems])
    changed = np.asarray([candidate[s] for s in stems])
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(repetitions):
        rows = rng.integers(0, len(stems), len(stems))
        a, b = base[rows].sum(0), changed[rows].sum(0)
        da, db = a[1] + .5 * (a[2] + a[3]), b[1] + .5 * (b[2] + b[3])
        differences.append((b[0] / db if db else 0) - (a[0] / da if da else 0))
    low, high = np.quantile(differences, [.025, .975])
    return {"repetitions": repetitions, "unit": "physical observation",
            "seed": SEED, "lower_95": float(low), "upper_95": float(high)}


def cluster_annotation_masks(record_masks):
    """Complete link >.5; at most one outline per record; preserve every outline."""
    entries = [(record, rle) for record in sorted(record_masks)
               for rle in record_masks[record] if mask_util.area(to_rle_dict(rle)) > 0]
    if not entries:
        return []
    rles = [to_rle_dict(rle) for _, rle in entries]
    ious = np.asarray(mask_util.iou(rles, rles, [0] * len(rles)))
    groups = []
    for index, (record, _) in enumerate(entries):
        eligible = [group for group in groups
                    if all(entries[j][0] != record for j in group)
                    and float(ious[index, group].min()) > .5]
        if eligible:
            max(eligible, key=lambda group: float(ious[index, group].mean())).append(index)
        else:
            groups.append([index])
    return [[entries[index][1] for index in group] for group in groups]


def match_proposals(proposals, clusters):
    """Keep IoU>.1 positives and exactly zero-overlap negatives; omit ambiguity."""
    flat = [rle for group in clusters for rle in group]
    memberships = [index for index, group in enumerate(clusters) for _ in group]
    matched = []
    if flat and proposals["rles"]:
        iou = np.asarray(mask_util.iou(
            [to_rle_dict(r) for r in proposals["rles"]],
            [to_rle_dict(r) for r in flat], [0] * len(flat)
        )).reshape(len(proposals["rles"]), len(flat))
    else:
        iou = np.zeros((len(proposals["rles"]), len(flat)))
    for index, (rle, confidence) in enumerate(zip(proposals["rles"], proposals["scores"])):
        best_iou = float(iou[index].max()) if flat else 0.0
        if best_iou > .1:
            targets = clusters[memberships[int(iou[index].argmax())]]
        elif best_iou == 0.0:
            targets = []
        else:
            continue
        matched.append({"rle": rle, "score": confidence, "target_rles": targets,
                        "match_iou": best_iou})
    return matched


def patch_origin(prior, tile, rng):
    ys, xs = np.nonzero(prior)
    if len(xs):
        index = int(rng.integers(len(xs)))
        cx, cy = int(xs[index]), int(ys[index])
    else:
        cx, cy = FULL_SIZE // 2, FULL_SIZE // 2
    jitter = max(1, tile // 5)
    x = int(np.clip(cx - tile // 2 + rng.integers(-jitter, jitter + 1), 0, FULL_SIZE - tile))
    y = int(np.clip(cy - tile // 2 + rng.integers(-jitter, jitter + 1), 0, FULL_SIZE - tile))
    return x, y


class NativeCropDataset(Dataset):
    def __init__(self, examples, image_dir, tile, length):
        self.positive = [row for row in examples if row["target_rles"]]
        self.negative = [row for row in examples if not row["target_rles"]]
        if not self.positive:
            raise ValueError("no matched detector proposals available for training")
        self.image_dir, self.tile, self.length = image_dir, tile, length
        self.images = OrderedDict()

    def __len__(self):
        return self.length

    def _image(self, stem):
        if stem not in self.images:
            gray = cv2.imread(str(self.image_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (FULL_SIZE, FULL_SIZE):
                raise ValueError(f"invalid native image: {stem}")
            self.images[stem] = (gray, cv2.GaussianBlur(gray, (0, 0), 15))
            if len(self.images) > 4:
                self.images.popitem(last=False)
        self.images.move_to_end(stem)
        return self.images[stem]

    def __getitem__(self, index):
        rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
        pool = self.negative if index % 4 == 0 and self.negative else self.positive
        row = pool[int(rng.integers(len(pool)))]
        gray, background = self._image(row["stem"])
        prior = decode_mask(row["rle"])
        x, y = patch_origin(prior, self.tile, rng)
        if row["target_rles"] and rng.random() < .35:
            radius = int(rng.integers(1, 5))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
            changed = cv2.dilate(prior, kernel) if rng.random() < .5 else cv2.erode(prior, kernel)
            if changed.any():
                prior = changed
        features = patch_features(gray, background, prior, x, y, self.tile)
        masks = [decode_mask(r)[y:y + self.tile, x:x + self.tile]
                 for r in row["target_rles"]]
        targets = (np.asarray(masks, dtype=np.float32) if masks
                   else np.zeros((0, self.tile, self.tile), dtype=np.float32))
        for axis in (-1, -2):
            if rng.random() < .5:
                features, targets = np.flip(features, axis), np.flip(targets, axis)
        return (torch.from_numpy(np.ascontiguousarray(features)),
                torch.from_numpy(np.ascontiguousarray(targets)))


def collate_crops(items):
    images = torch.stack([item[0] for item in items])
    experts = max(1, max(item[1].shape[0] for item in items))
    targets = torch.zeros((len(items), experts, *images.shape[-2:]), dtype=torch.float32)
    present = torch.zeros((len(items), experts), dtype=torch.bool)
    for index, (_, masks) in enumerate(items):
        count = masks.shape[0]
        targets[index, :count] = masks
        present[index, :count] = True
    return images, targets, present


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


@torch.inference_mode()
def refine_proposal(model, gray, background, rle, device, alphas):
    prior = decode_mask(rle)
    x, y, width, height = mask_util.toBbox(to_rle_dict(rle))
    a, b = max(0, int(x) - 64), max(0, int(y) - 64)
    c = min(FULL_SIZE, int(math.ceil(x + width)) + 64)
    d = min(FULL_SIZE, int(math.ceil(y + height)) + 64)
    windows = [(px, py) for py in tile_starts(b, d, TILE)
               for px in tile_starts(a, c, TILE)]
    total = np.zeros((d - b, c - a), dtype=np.float32)
    weights = np.zeros_like(total)
    taper = np.maximum(np.hanning(TILE).astype(np.float32), .05)
    window_weight = taper[:, None] * taper[None, :]
    for start in range(0, len(windows), BATCH):
        locations = windows[start:start + BATCH]
        batch = torch.from_numpy(np.stack([
            patch_features(gray, background, prior, px, py, TILE) for px, py in locations
        ])).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            residual = model(batch) - prior_logits_from_input(batch)
        changes = residual[:, 0].float().cpu().numpy()
        for change, (px, py) in zip(changes, locations):
            xa, ya, xb, yb = max(a, px), max(b, py), min(c, px + TILE), min(d, py + TILE)
            weight = window_weight[ya - py:yb - py, xa - px:xb - px]
            total[ya - b:yb - b, xa - a:xb - a] += change[ya - py:yb - py, xa - px:xb - px] * weight
            weights[ya - b:yb - b, xa - a:xb - a] += weight
    if not np.all(weights > 0) or not np.isfinite(total).all():
        raise ValueError("invalid native tile coverage or nonfinite model output")
    delta = total / weights
    prior_logits = (prior[b:d, a:c].astype(np.float32) * 2 - 1) * PRIOR_LOGIT
    result = {0.0: rle}
    for alpha in alphas:
        if alpha == 0:
            continue
        full = np.zeros_like(prior)
        full[b:d, a:c] = (prior_logits + alpha * delta > 0).astype(np.uint8)
        result[alpha] = encode_mask(full) if full.any() else None
    return result


def refine_image(model, path, proposals, device, alphas):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert gray is not None and gray.shape == (FULL_SIZE, FULL_SIZE), str(path)
    background = cv2.GaussianBlur(gray, (0, 0), 15)
    rles = {alpha: [] for alpha in alphas}
    scores = {alpha: [] for alpha in alphas}
    for rle, confidence in zip(proposals["rles"], proposals["scores"]):
        if confidence < DETECTOR_CONF:
            continue
        variants = (refine_proposal(model, gray, background, rle, device, alphas)
                    if any(alpha != 0 for alpha in alphas) else {0.0: rle})
        for alpha in alphas:
            if variants[alpha] is not None:
                rles[alpha].append(variants[alpha])
                scores[alpha].append(confidence)
    return {alpha: resolve_overlaps(rles[alpha], scores[alpha]) for alpha in alphas}


def predict_detector(detector, path, device):
    # Smoke uses a cheaper proposer grid; production exactly uses source imgsz.
    output = detector.predict(source=str(path), imgsz=640 if SMOKE else 1792,
                              conf=DETECTOR_FLOOR, iou=DETECTOR_NMS, max_det=60,
                              retina_masks=True, device=0 if device.type == "cuda" else "cpu",
                              verbose=False)[0]
    if output.masks is None:
        return {"rles": [], "scores": []}
    masks = output.masks.data.cpu().numpy()
    assert masks.shape[-2:] == (FULL_SIZE, FULL_SIZE), "retina masks must be native"
    pairs = [(encode_mask(mask > .5), float(score)) for mask, score in
             zip(masks, output.boxes.conf.cpu().numpy()) if (mask > .5).any()]
    return {"rles": [r for r, _ in pairs], "scores": [score for _, score in pairs]}


def validate_submission(frame, allowed_stems):
    if list(frame.columns) != ["filament_id", "segmentation_rle"]:
        raise ValueError("unexpected submission schema")
    if not frame.filament_id.is_unique:
        raise ValueError("duplicate filament ids")
    groups = defaultdict(list)
    for row in frame.itertuples(index=False):
        stem = row.filament_id.rsplit("_", 1)[0]
        if stem not in allowed_stems:
            raise ValueError(f"unexpected test stem {stem}")
        groups[stem].append(row.segmentation_rle)
    for stem, rles in groups.items():
        claimed = np.zeros((FULL_SIZE, FULL_SIZE), dtype=bool)
        for rle in rles:
            mask = decode_mask(rle).astype(bool)
            if not mask.any() or np.any(claimed & mask) or encode_mask(mask) != rle:
                raise ValueError(f"empty, overlapping or noncanonical mask on {stem}")
            claimed |= mask
    return {"rows": len(frame), "covered_images": len(groups),
            "allowed_images": len(allowed_stems), "disjoint": True, "canonical": True}


def find_inputs():
    annotation_path = Path(ARGS.data_root) / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"competition annotations not found: {annotation_path}")
    data_root = annotation_path.parent.parent
    source = Path(ARGS.detector_run)
    for filename in SOURCE_HASHES:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"detector run output missing: {source / filename}")
    weights = source / "runs" / "seg" / "weights" / "last.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"frozen detector checkpoint missing: {weights}")
    published = (all(file_sha256(source / filename) == expected for filename, expected in SOURCE_HASHES.items())
                 and file_sha256(weights) == WEIGHTS_SHA256)
    print("detector run: " + ("the published one, every file matches its hash" if published else
                              "not the published one; the refiner is trained on this detector's proposals"),
          flush=True)
    return annotation_path, data_root, source, weights


def train_refiner(model, examples, image_dir, device):
    dataset = NativeCropDataset(examples, image_dir, TILE, MAX_STEPS * BATCH)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0 if SMOKE else 2,
                        collate_fn=collate_crops, worker_init_fn=seed_worker, generator=generator,
                        pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, start = [], time.monotonic()
    model.train()
    for step, (images, targets, present) in enumerate(loader):
        elapsed = time.monotonic() - start
        if step and elapsed >= TRAIN_SECONDS:
            break
        progress = min(1.0, max(step / MAX_STEPS, elapsed / TRAIN_SECONDS))
        warmup = min(1.0, (step + 1) / (1 if SMOKE else 100))
        rate = LR * warmup * .5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = rate
        images, targets, present = images.to(device), targets.to(device), present.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = native_refiner_loss(logits, targets, present, residual_weight=RESIDUAL_WEIGHT,
                                       prior_logits=prior_logits_from_input(images))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        history.append({"step": step + 1, "loss": float(loss.detach()),
                        "learning_rate": rate, "training_seconds": time.monotonic() - start})
        if step % 100 == 0 or SMOKE:
            print("training", history[-1], flush=True)
            pd.DataFrame(history).to_csv(WORK / "training_history.csv", index=False)
        if (step + 1) % 500 == 0:
            torch.save(model.checkpoint(), WORK / "refiner_last.pt")
    if not history:
        raise ValueError("training completed no optimizer steps")
    torch.save(model.checkpoint(), WORK / "refiner_last.pt")
    pd.DataFrame(history).to_csv(WORK / "training_history.csv", index=False)
    model.eval()
    return {"steps": len(history), "seconds": time.monotonic() - start,
            "positive_examples": len(dataset.positive), "negative_examples": len(dataset.negative)}


def main():
    started = time.monotonic()
    WORK.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cv2.setNumThreads(0)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if REQUIRE_GPU and device.type != "cuda":
        raise SystemExit("ABORT: production refiner requires a working GPU")
    if device.type == "cuda":
        probe = torch.nn.Conv2d(4, 4, 3, padding=1).to(device)
        probe(torch.randn(1, 4, 16, 16, device=device))
        torch.cuda.synchronize()
    print("model self-test", self_test(), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    f"ultralytics=={ULTRALYTICS_VERSION}"], check=True)
    from ultralytics import YOLO
    annotation_path, data_root, source, detector_weights = find_inputs()
    report = json.loads((source / "run_report.json").read_text())
    cached = json.loads((source / "validation_predictions.json").read_text())
    assert report["fold_fingerprint"] == cached["fold_fingerprint"] == FINGERPRINT
    assert report["model"] == "yolo11l-seg.pt" and report["imgsz"] == 1792
    assert report["nms_iou"] == DETECTOR_NMS and report["conf"] == DETECTOR_CONF
    train_stems = report["train_stems"][:8] if SMOKE else report["train_stems"]
    val_stems = report["val_stems"][:4] if SMOKE else report["val_stems"]
    tune_stems, report_stems = val_stems[::2], val_stems[1::2]
    if not SMOKE:
        assert len(train_stems) == 565 and len(tune_stems) == len(report_stems) == 71
        assert tune_stems == report["tune_stems"] and report_stems == report["report_stems"]
    assert not set(train_stems) & set(val_stems)
    assert not set(tune_stems) & set(report_stems)
    coco = json.loads(annotation_path.read_text())
    all_stems = sorted({annotation["image_id"].split("-", 1)[1] for annotation in coco["annotations"]})
    order = np.random.default_rng(FOLD_SEED).permutation(len(all_stems))
    assignment = {all_stems[position]: i % 5 for i, position in enumerate(order)}
    fingerprint = hashlib.sha256(";".join(
        f"{stem}:{fold}" for stem, fold in sorted(assignment.items())).encode()).hexdigest()
    assert fingerprint == FINGERPRINT
    assert all(assignment[s] != 0 for s in train_stems)
    assert all(assignment[s] == 0 for s in val_stems)
    all_gt = defaultdict(list)
    for annotation in coco["annotations"]:
        rles = mask_util.frPyObjects(annotation["segmentation"], FULL_SIZE, FULL_SIZE)
        all_gt[annotation["image_id"]].append(mask_util.merge(rles)["counts"].decode("ascii"))
    per_stem_gt = defaultdict(dict)
    for record, rles in all_gt.items():
        per_stem_gt[record.split("-", 1)[1]][record] = rles
    val_cache = {row["stem"]: row for row in cached["predictions"]
                 if row["nms_iou"] == DETECTOR_NMS and row["stem"] in set(val_stems)}
    assert set(val_cache) == set(val_stems)
    baseline = {}
    for stem, row in val_cache.items():
        keep = [(r, score) for r, score in zip(row["rles"], row["scores"]) if score >= DETECTOR_CONF]
        baseline[stem] = resolve_overlaps([r for r, _ in keep], [score for _, score in keep])
    baseline_parts = {"tune": contributions(all_gt, baseline, tune_stems),
                      "report": contributions(all_gt, baseline, report_stems)}
    baseline_scores = {split: score_contributions(parts) for split, parts in baseline_parts.items()}
    if not SMOKE:
        for split in ("tune", "report"):
            for key, value in baseline_scores[split].items():
                assert abs(value - report[split + "_half"][key]) < 1e-12, (split, key)
    source_hashes = {name: file_sha256(source / name) for name in SOURCE_HASHES}
    source_hashes["detector_weights"] = file_sha256(detector_weights)
    config = {"source": SOURCE_REF, "source_hashes": source_hashes,
              "runner_sha256": file_sha256(Path(__file__)),
              "model_source_sha256": REFINER_MODEL_SOURCE_SHA256,
              "fold_fingerprint": FINGERPRINT, "train_stems": train_stems,
              "tune_stems": tune_stems, "report_stems": report_stems,
              "smoke": SMOKE, "tile": TILE, "base_channels": BASE_CHANNELS, "batch": BATCH,
              "training_budget_seconds": TRAIN_SECONDS, "max_steps": MAX_STEPS,
              "learning_rate": LR, "residual_weight": RESIDUAL_WEIGHT,
              "alphas": ALPHAS, "detector_confidence": DETECTOR_CONF, "detector_nms": DETECTOR_NMS,
              "training_proposal_floor": DETECTOR_FLOOR,
              "baseline_prediction_sha256": prediction_hash(baseline),
              "baseline_instance_count": sum(map(len, baseline.values())),
              "baseline_scores": baseline_scores,
              "limitations": ["Teacher proposals on refiner training images are in-sample.",
                              "One physical-image fold; no temporal embargo.",
                              "Report partition was previously inspected for detector comparison."]}
    write_json(WORK / "config.json", config)
    write_json(WORK / "baseline_predictions.json", baseline)
    train_dir, test_dir = data_root / "train" / "train_images", data_root / "test" / "test_images"
    detector = YOLO(str(detector_weights))
    test_paths = sorted(test_dir.glob("*.jpeg"))
    if SMOKE:
        test_paths = test_paths[:3]
    allowed = {path.stem for path in test_paths}
    source_submission = pd.read_csv(source / "submission.csv")
    source_test = defaultdict(list)
    for row in source_submission.itertuples(index=False):
        source_test[row.filament_id.rsplit("_", 1)[0]].append(row.segmentation_rle)
    # Cache test proposals before training: a changed detector replay must fail
    # before consuming the four-hour optimization budget. No test labels enter.
    test_cache, parity = {}, []
    for index, path in enumerate(test_paths):
        proposals = predict_detector(detector, path, device)
        test_cache[path.stem] = proposals
        keep = [(r, score) for r, score in zip(proposals["rles"], proposals["scores"])
                if score >= DETECTOR_CONF]
        replay = resolve_overlaps([r for r, _ in keep], [score for _, score in keep])
        same = replay == source_test[path.stem]
        parity.append({"stem": path.stem, "exact_source_baseline": same})
        write_json(WORK / "test_baseline_parity.json", parity)
        if not SMOKE and not same:
            raise ValueError(f"frozen detector test replay differs from source on {path.stem}")
        if index % 25 == 0 or SMOKE:
            print(f"test proposal preflight {index + 1}/{len(test_paths)}", flush=True)
    write_json(WORK / "test_proposals.json", test_cache)
    train_cache, examples = {}, []
    for index, stem in enumerate(train_stems):
        proposals = predict_detector(detector, train_dir / f"{stem}.jpeg", device)
        train_cache[stem] = proposals
        clusters = cluster_annotation_masks(per_stem_gt[stem])
        for example in match_proposals(proposals, clusters):
            example["stem"] = stem
            examples.append(example)
        if index % 25 == 0 or SMOKE:
            print(f"training proposals {index + 1}/{len(train_stems)}; examples {len(examples)}", flush=True)
    write_json(WORK / "training_proposals.json", train_cache)
    write_json(WORK / "training_examples.json", examples)
    del train_cache
    detector.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = NativeResidualUNet(base_channels=BASE_CHANNELS).to(device)
    training = train_refiner(model, examples, train_dir, device)
    predictions = {alpha: {} for alpha in ALPHAS}
    for index, stem in enumerate(val_stems):
        variants = refine_image(model, train_dir / f"{stem}.jpeg", val_cache[stem], device, ALPHAS)
        assert variants[0.0] == baseline[stem], f"alpha=0 changed source masks for {stem}"
        for alpha in ALPHAS:
            predictions[alpha][stem] = variants[alpha]
        if index % 10 == 0 or SMOKE:
            print(f"validation inference {index + 1}/{len(val_stems)}", flush=True)
    assert prediction_hash(predictions[0.0]) == config["baseline_prediction_sha256"]
    for alpha, value in predictions.items():
        write_json(WORK / f"validation_alpha_{alpha:.1f}.json", {
            "fold_fingerprint": FINGERPRINT, "alpha": alpha, "predictions": value,
            "prediction_sha256": prediction_hash(value)})
    tune_scores = {alpha: score_contributions(contributions(all_gt, value, tune_stems))
                   for alpha, value in predictions.items()}
    # Stable ordering means a tie preserves alpha=0 and all detector masks.
    selected_alpha = max(ALPHAS, key=lambda alpha: tune_scores[alpha]["pq"])
    write_json(WORK / "frozen_selection.json", {"alpha": selected_alpha,
               "selected_using": "tune_stems only", "tune_scores": tune_scores})
    selected_parts = (baseline_parts["report"] if selected_alpha == 0 else
                      contributions(all_gt, predictions[selected_alpha], report_stems))
    report_score = score_contributions(selected_parts)
    confidence = paired_bootstrap(baseline_parts["report"], selected_parts,
                                  repetitions=100 if SMOKE else 2000)
    gain = report_score["pq"] - baseline_scores["report"]["pq"]
    promotion = bool(not SMOKE and selected_alpha > 0 and gain >= .01 and confidence["lower_95"] > 0)
    write_json(WORK / "evaluation_report.json", {
        "fold_fingerprint": FINGERPRINT, "training": training,
        "selected_alpha": selected_alpha, "tune_scores": tune_scores,
        "report_half": report_score, "baseline_report_half": baseline_scores["report"],
        "report_gain": gain, "paired_bootstrap": confidence, "promotion_passed": promotion})
    rows = []
    if selected_alpha == 0:
        submission = source_submission[source_submission.filament_id.map(
            lambda fid: fid.rsplit("_", 1)[0] in allowed)].copy()
    else:
        for index, path in enumerate(test_paths):
            proposals = test_cache[path.stem]
            variants = refine_image(model, path, proposals, device, (0.0, selected_alpha))
            same = variants[0.0] == source_test[path.stem]
            if not SMOKE and not same:
                raise ValueError(f"frozen detector test replay differs from source on {path.stem}")
            for instance, rle in enumerate(variants[selected_alpha]):
                rows.append({"filament_id": f"{path.stem}_{instance}", "segmentation_rle": rle})
            if index % 20 == 0 or SMOKE:
                print(f"test inference {index + 1}/{len(test_paths)}", flush=True)
        submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    validation = validate_submission(submission, allowed)
    submission.to_csv(WORK / "submission.csv", index=False)
    if selected_alpha == 0 and not SMOKE:
        assert file_sha256(WORK / "submission.csv") == file_sha256(source / "submission.csv")
    result = {"experiment": "Native residual mask refinement with frozen own detector",
              "smoke": SMOKE, "fold_fingerprint": FINGERPRINT, "source": SOURCE_REF,
              "config": config, "training": training, "selected_alpha": selected_alpha,
              "tune_scores": tune_scores, "report_half": report_score,
              "baseline_report_half": baseline_scores["report"], "report_gain": gain,
              "paired_bootstrap": confidence, "promotion_passed": promotion,
              "promotion_rule": "report gain >= .01 and paired 95% CI lower > 0",
              "submission_validation": validation, "submission_sha256": file_sha256(WORK / "submission.csv"),
              "runtime_seconds": time.monotonic() - started,
              "annotation_context": {"human_agreement": .342, "single_expert_reference": .598,
                                     "note": "Different populations; not upper bounds."}}
    write_json(WORK / "run_report.json", result)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
    (WORK / "requirements_frozen.txt").write_text(freeze.stdout, encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if SMOKE:
        assert training["steps"] >= 1
        print("SMOKE PASS: real optimization, native inference, scoring and disjoint CSV", flush=True)


if __name__ == "__main__":
    WORK = Path(ARGS.output_dir)
    SOURCE_REF = str(Path(ARGS.detector_run))
    main()
