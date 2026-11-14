"""Stage 2 of the model: native-resolution residual refinement of each proposal.

The detector's masks are assembled from 448 x 448 prototype masks and upsampled
to 2048 px, so their outlines are smooth approximations. The refiner looks at
the image at native resolution around one proposal and predicts a correction
``delta`` to that proposal's mask on the logit scale:

    refined = (prior + alpha * delta > 0),   prior = +2.197 inside, -2.197 outside

``prior`` is the logit of p = 0.9 / 0.1, and ``alpha = 1.0`` was chosen on the
tune half of the validation fold from 0, 0.5 and 1.0. With ``delta = 0`` the
refined mask is exactly the proposal, which is also what the untrained network
predicts (its output head starts at zero).

**Input**, four channels on the native pixel grid (:func:`patch_features`):

0. ``gray / 255``;
1. local contrast ``clip((background - gray) / 32, -2, 2)``, where
   ``background`` is the frame blurred with a Gaussian of sigma 15 px
   (:func:`background_of`); filaments are darker than their surroundings;
2. the proposal mask exactly as the detector produced it (0 / 1);
3. signed distance to the proposal outline, ``clip((inside - outside) / 32, -1, 1)``.

**Tiling** (:func:`refine_proposal`). Each proposal is refined inside its
bounding box plus a 64 px margin; nothing outside that window can change. The
window is covered by 512 x 512 tiles at a 256 px stride, clamped to the image
(:func:`tile_starts`), and the tile outputs are blended with a Hann taper whose
floor is 0.05, so tile borders carry little weight. No resizing happens
anywhere in this stage.

**Training** (``training/train_refiner.py`` repeats it on a GPU; the notebook
loads the published weights). Examples are the frozen detector's proposals (down to confidence
0.05) on the 565 training images of folds 1-4. Each proposal is matched to a
cluster of annotator
outlines (complete link at IoU > 0.5, at most one outline per annotator): best
IoU > 0.1 makes it a positive whose targets are every outline in that cluster,
no overlap at all makes it a negative with an empty target, and anything in
between is left out (7,713 positives, 2,667 negatives). Crops of 512 px are
centred on a random pixel of the proposal with a jitter of up to 102 px,
flipped at random along both axes, and in 35 % of positives the proposal is
dilated or eroded by 1-4 px first, so the network learns to undo boundary
errors. One in four crops is a negative. The loss is
:func:`native_refiner_loss` (BCE + soft Dice, averaged over the annotators who
drew the matched filament) plus 0.001 x the mean squared correction. AdamW at
learning rate 3e-4 with weight decay 1e-4, 100 warm-up steps and a cosine
schedule, gradient clipping at 5, batch 4, mixed precision; 31,743 steps in
4 hours on one GPU. The checkpoint holds the weights after the last step.

The model definition below, from :data:`PRIOR_LOGIT` to :func:`self_test`, is
the code these weights were trained with, unchanged. Its original notes:

    Native-pixel refinement with an exact detector-mask control.

    Input is floating point B x 4 x H x W, with H,W >= 8. Channels are grayscale,
    local contrast, an EXACT binary proposal mask, and signed distance. The caller
    normalizes the image/contrast/distance channels and keeps every channel on the
    same native pixel grid. Output is float32 B x 1 x H x W foreground logits.

    The network has four scales, including a full-resolution skip. Its zero-initialized
    head predicts a correction to prior logits +/-2.1972246 (probabilities .1/.9).
    Therefore sigmoid(output) > .5 equals the proposal before any optimization.
    The caller owns tiling, proposal assignment, overlap handling and threshold choice.

    Shape loss uses only experts who annotated the matched instance. Missing experts
    are excluded, not treated as empty masks. A proposal with no matched expert gets
    one all-zero target. Every proposal has equal weight regardless of expert count.

    Checkpoint convention (file I/O belongs in the runner)::

        torch.save(model.checkpoint(), checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = NativeResidualUNet.from_checkpoint(payload).to(device).eval()

    ``checkpoint()`` contains CPU tensor copies, architecture configuration and a
    format version; loading is strict. This module performs no train/test data I/O.
    ``self_test()`` checks the tensor contracts on CPU without optimizer steps.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_util

from ..io import IMAGE_SIZE, mask_to_rle

FULL_SIZE = IMAGE_SIZE


# --------------------------------------------------------------------------
# Model definition (unchanged from the training run)
# --------------------------------------------------------------------------

PRIOR_LOGIT = 2.1972246


def prior_logits_from_input(inputs):
    """Return B x 1 x H x W float32 prior logits from binary channel 2.

    Binary values are an input contract, deliberately not checked with a device
    synchronization on every forward pass. Validate cached proposals in the runner.
    """
    if inputs.ndim != 4 or inputs.shape[1] != 4:
        raise ValueError("expected B x 4 x H x W refiner inputs")
    return (inputs[:, 2:3].float() * 2.0 - 1.0) * PRIOR_LOGIT


def _refiner_groups(channels):
    return next(groups for groups in (8, 4, 2, 1) if channels % groups == 0)


class _RefinerConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_refiner_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_refiner_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class NativeResidualUNet(nn.Module):
    """Four-scale U-Net: base 24 for experiments, base 8 for CPU smoke.

    Spatial sizes need not be divisible by eight: decoder interpolation follows
    each skip's exact dimensions. There is no normalization across the batch.
    """

    def __init__(self, base_channels=24):
        super().__init__()
        if not isinstance(base_channels, int) or base_channels < 4:
            raise ValueError("base_channels must be an integer >= 4")
        self.base_channels = base_channels
        widths = [base_channels * 2 ** level for level in range(4)]
        self.encoders = nn.ModuleList(
            _RefinerConvBlock(4 if level == 0 else widths[level - 1], width)
            for level, width in enumerate(widths)
        )
        self.decoders = nn.ModuleList(
            _RefinerConvBlock(widths[level + 1] + widths[level], widths[level])
            for level in reversed(range(3))
        )
        self.delta_head = nn.Conv2d(widths[0], 1, kernel_size=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("expected B x 4 x H x W refiner inputs")
        if inputs.shape[0] < 1 or min(inputs.shape[-2:]) < 8:
            raise ValueError("refiner requires a nonempty batch and H,W >= 8")
        prior = prior_logits_from_input(inputs)
        skips = []
        features = inputs
        for level, encoder in enumerate(self.encoders):
            if level:
                features = F.max_pool2d(features, kernel_size=2)
            features = encoder(features)
            skips.append(features)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            features = F.interpolate(
                features, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            features = decoder(torch.cat((features, skip), dim=1))
        return prior + self.delta_head(features).float()

    def checkpoint(self):
        """Snapshot portable tensors; subsequent updates cannot mutate this copy."""
        return {
            "format_version": 1,
            "model_type": "native_residual_unet",
            "model_config": {"base_channels": self.base_channels},
            "model": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(cls, payload):
        """Construct on CPU and strictly restore our versioned checkpoint format."""
        if payload.get("format_version") != 1:
            raise ValueError("unsupported native refiner checkpoint version")
        if payload.get("model_type") != "native_residual_unet":
            raise ValueError("checkpoint is not a native residual U-Net")
        model = cls(**payload["model_config"])
        model.load_state_dict(payload["model"], strict=True)
        return model


def native_refiner_loss(
    logits,
    expert_masks,
    expert_present,
    *,
    bce_weight=1.0,
    dice_weight=1.0,
    residual_weight=0.0,
    prior_logits: Optional[torch.Tensor] = None,
    dice_epsilon=1.0,
):
    """Mean proposal loss, averaging BCE + soft Dice over each present expert.

    ``logits``: B x 1 x H x W, raw foreground logits.
    ``expert_masks``: B x R x H x W, binary or soft masks in [0,1].
    ``expert_present``: B x R boolean validity for the matched instance. R may
    be zero. Invalid slots may contain arbitrary values and are ignored.

    No matched expert means one explicit zero mask, including for Dice. Positive
    proposals average per-expert Dice rather than Dice against an averaged mask.
    Float32 calculations and reductions remain stable under an outer autocast.
    Optional residual regularization is the mean squared logit correction; it
    requires B x 1 x H x W ``prior_logits`` and never backpropagates into them.
    """
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape[0] < 1:
        raise ValueError("logits must be nonempty B x 1 x H x W")
    if expert_masks.ndim != 4 or (
        expert_masks.shape[0] != logits.shape[0]
        or expert_masks.shape[-2:] != logits.shape[-2:]
    ):
        raise ValueError("expert_masks must be B x R x H x W matching logits")
    if expert_present.shape != expert_masks.shape[:2]:
        raise ValueError("expert_present must be B x R matching expert_masks")
    if any(weight < 0 for weight in (bce_weight, dice_weight, residual_weight)):
        raise ValueError("loss weights must be nonnegative")
    if dice_epsilon <= 0:
        raise ValueError("dice_epsilon must be positive")
    if residual_weight and (
        prior_logits is None or prior_logits.shape != logits.shape
    ):
        raise ValueError("residual regularization requires matching prior_logits")

    with torch.autocast(device_type=logits.device.type, enabled=False):
        scores = logits.float()
        present = expert_present.to(device=logits.device, dtype=torch.bool)
        masks = expert_masks.to(device=logits.device, dtype=torch.float32)
        masks = torch.where(present[:, :, None, None], masks, torch.zeros_like(masks))

        # Append one valid zero target only for proposals without a matched expert.
        # This also handles an empty expert axis without a separate loss branch.
        negative = ~present.any(dim=1, keepdim=True)
        present = torch.cat((present, negative), dim=1)
        masks = torch.cat((masks, torch.zeros_like(scores)), dim=1)
        weights = present.float()
        expanded_scores = scores.expand_as(masks)
        bce = F.binary_cross_entropy_with_logits(
            expanded_scores, masks, reduction="none"
        ).mean(dim=(-2, -1))
        probability = torch.sigmoid(scores)
        intersection = (probability * masks).sum(dim=(-2, -1))
        total = probability.sum(dim=(-2, -1)) + masks.sum(dim=(-2, -1))
        dice = 1.0 - (2.0 * intersection + dice_epsilon) / (total + dice_epsilon)
        per_expert = bce_weight * bce + dice_weight * dice
        loss = ((per_expert * weights).sum(dim=1) / weights.sum(dim=1)).mean()
        if residual_weight:
            correction = scores - prior_logits.detach().to(
                device=logits.device, dtype=torch.float32
            )
            loss = loss + residual_weight * correction.square().mean()
    return loss


def self_test():
    """Exercise identity, expert weighting, gradients, autocast and checkpoints.

    Designed for the runner's CPU smoke. Uses synthetic tensors only and performs
    no optimizer steps or filesystem access. Raises AssertionError on failure.
    """
    checks = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261003)
        model = NativeResidualUNet(base_channels=8).eval()
        inputs = torch.randn(2, 4, 33, 35)
        inputs[:, 2] = (torch.rand(2, 33, 35) > 0.75).float()
        prior = prior_logits_from_input(inputs)
        with torch.no_grad():
            logits = model(inputs)
        assert logits.shape == (2, 1, 33, 35)
        assert torch.equal(logits, prior)
        assert torch.equal(torch.sigmoid(logits) > 0.5, inputs[:, 2:3].bool())
        checks.append("exact_identity_at_odd_spatial_sizes")

        # Missing experts cannot dilute a positive target or affect its loss.
        target = inputs[:, 2:3].clone()
        one_present = torch.ones(2, 1, dtype=torch.bool)
        one_loss = native_refiner_loss(logits, target, one_present)
        missing_target = torch.cat((target, torch.full_like(target, float("nan"))), 1)
        present = torch.tensor([[True, False], [True, False]])
        missing_loss = native_refiner_loss(logits, missing_target, present)
        duplicate_loss = native_refiner_loss(
            logits, target.repeat(1, 3, 1, 1), one_present.repeat(1, 3)
        )
        torch.testing.assert_close(one_loss, missing_loss)
        torch.testing.assert_close(one_loss, duplicate_loss)
        checks.append("missing_experts_excluded_and_duplicates_normalized")

        # Each proposal has equal weight even when expert counts differ.
        diverse_targets = torch.cat((target, 1.0 - target), dim=1)
        mixed_validity = torch.tensor([[True, False], [True, True]])
        mixed_loss = native_refiner_loss(logits, diverse_targets, mixed_validity)
        first_loss = native_refiner_loss(logits[:1], target[:1], one_present[:1])
        second_loss = native_refiner_loss(
            logits[1:], diverse_targets[1:], mixed_validity[1:]
        )
        torch.testing.assert_close(mixed_loss, (first_loss + second_loss) / 2.0)
        checks.append("equal_proposal_weight_with_unequal_expert_counts")

        absent_loss = native_refiner_loss(logits, target, ~one_present)
        zero_loss = native_refiner_loss(logits, torch.zeros_like(target), one_present)
        empty_loss = native_refiner_loss(logits, target[:, :0], one_present[:, :0])
        torch.testing.assert_close(absent_loss, zero_loss)
        torch.testing.assert_close(empty_loss, zero_loss)
        checks.append("negative_proposals_equal_explicit_zero_targets")

        probability = logits.sigmoid()
        expected_dice = []
        for expert in range(2):
            mask = diverse_targets[:, expert:expert + 1]
            numerator = 2.0 * (probability * mask).sum((-2, -1)) + 1.0
            denominator = probability.sum((-2, -1)) + mask.sum((-2, -1)) + 1.0
            expected_dice.append((1.0 - numerator / denominator).mean())
        actual_dice = native_refiner_loss(
            logits, diverse_targets, torch.ones(2, 2, dtype=torch.bool), bce_weight=0
        )
        torch.testing.assert_close(actual_dice, torch.stack(expected_dice).mean())
        checks.append("dice_averages_experts_not_masks")

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            autocast_logits = model(inputs)
            autocast_loss = native_refiner_loss(autocast_logits, target, one_present)
        assert autocast_logits.dtype == torch.float32
        assert autocast_loss.dtype == torch.float32 and torch.isfinite(autocast_loss)
        assert torch.equal(autocast_logits, prior)
        extreme = torch.where(target.bool(), -10000.0, 10000.0).half()
        extreme_loss = native_refiner_loss(extreme, target, one_present)
        assert extreme_loss.dtype == torch.float32 and torch.isfinite(extreme_loss)
        checks.append("float32_loss_and_identity_under_autocast")

        identity_penalty_loss = native_refiner_loss(
            logits, target, one_present, residual_weight=0.1, prior_logits=prior
        )
        torch.testing.assert_close(identity_penalty_loss, one_loss)
        shifted = logits + 0.25
        plain_loss = native_refiner_loss(shifted, target, one_present)
        penalty_loss = native_refiner_loss(
            shifted, target, one_present, residual_weight=0.1, prior_logits=prior
        )
        torch.testing.assert_close(penalty_loss - plain_loss, torch.tensor(0.00625))
        checks.append("residual_penalty_is_zero_at_identity")

        model.zero_grad(set_to_none=True)
        differentiable_logits = model(inputs)
        loss = native_refiner_loss(differentiable_logits, 1.0 - target, one_present)
        loss.backward()
        gradient = model.delta_head.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        checks.append("finite_nonzero_correction_gradient")

        with torch.no_grad():
            model.delta_head.bias.fill_(0.125)
            expected = model(inputs)
        checkpoint = model.checkpoint()
        restored = NativeResidualUNet.from_checkpoint(checkpoint).eval()
        with torch.no_grad():
            model.delta_head.bias.add_(1.0)
            actual = restored(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert checkpoint["model"]["delta_head.bias"].item() == 0.125
        checks.append("strict_checkpoint_roundtrip_and_snapshot_independence")

        try:
            native_refiner_loss(logits, target, one_present, residual_weight=0.1)
        except ValueError:
            pass
        else:
            raise AssertionError("missing prior logits must be rejected")
        checks.append("regularizer_requires_prior")
    return {"passed": True, "checks": checks, "parameters": sum(
        parameter.numel() for parameter in model.parameters()
    )}


# --------------------------------------------------------------------------
# Inference at native resolution
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinerConfig:
    """Every inference setting of the refiner, as the training run used it.

    ``tile`` / ``batch``
        Tile side in native pixels, and tiles per forward pass.
    ``margin``
        Native pixels added on each side of a proposal's bounding box.
    ``background_sigma``
        Gaussian sigma, in pixels, of the background behind channel 1.
    ``alpha``
        Weight of the predicted correction; 0 returns the detector mask.
    ``taper_floor``
        Lowest blending weight at a tile border (Hann window, clipped).
    """

    tile: int = 512
    margin: int = 64
    batch: int = 4
    background_sigma: float = 15.0
    alpha: float = 1.0
    taper_floor: float = 0.05


REFINER = RefinerConfig()


def load_refiner(path: str | Path, device: str | torch.device = "cpu") -> NativeResidualUNet:
    """Build the refiner from a checkpoint written by :meth:`NativeResidualUNet.checkpoint`.

    ``weights_only=True`` loads tensors and plain values, never code, and the
    state dict is restored strictly: a missing or unexpected tensor raises.
    Verify the file's checksum first (:func:`filament.pipeline.verify_manifest`).
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return NativeResidualUNet.from_checkpoint(payload).to(device).eval()


def to_rle_dict(counts: str) -> dict:
    return {"size": [FULL_SIZE, FULL_SIZE], "counts": counts.encode("ascii")}


def decode_mask(counts: str) -> np.ndarray:
    """Compressed RLE -> ``(2048, 2048)`` uint8 mask of 0 / 1."""
    return mask_util.decode(to_rle_dict(counts))


def background_of(gray: np.ndarray, config: RefinerConfig = REFINER) -> np.ndarray:
    """The smooth background behind the contrast channel: a Gaussian blur of the frame."""
    return cv2.GaussianBlur(gray, (0, 0), config.background_sigma)


def patch_features(gray, background, prior, x, y, tile):
    image = gray[y:y + tile, x:x + tile].astype(np.float32)
    smooth = background[y:y + tile, x:x + tile].astype(np.float32)
    mask = np.ascontiguousarray(prior[y:y + tile, x:x + tile])
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    signed = np.clip((inside - outside) / 32.0, -1.0, 1.0)
    return np.stack((image / 255.0, np.clip((smooth - image) / 32.0, -2.0, 2.0),
                     mask.astype(np.float32), signed)).astype(np.float32)


def tile_starts(lower, upper, tile, full_size=FULL_SIZE):
    """Cover a native interval without resize, including both image borders."""
    stride = tile // 2
    first = min(max(0, int(lower)), full_size - tile)
    starts = [first]
    while starts[-1] + tile < upper:
        following = min(starts[-1] + stride, full_size - tile)
        if following == starts[-1]:
            raise ValueError("cannot cover requested interval")
        starts.append(following)
    return starts


@torch.inference_mode()
def refine_proposal(
    model: NativeResidualUNet,
    gray: np.ndarray,
    background: np.ndarray,
    rle: str,
    device: torch.device,
    config: RefinerConfig = REFINER,
) -> Optional[str]:
    """The refined mask of one proposal as compressed RLE, or ``None`` if it vanishes.

    ``gray`` is the raw ``(2048, 2048)`` uint8 frame, ``background`` its
    :func:`background_of`, ``rle`` the proposal. The arithmetic is the training
    run's: on a GPU the network runs under float16 autocast, on a CPU in float32.
    """
    if config.alpha == 0:
        return rle
    tile, margin = config.tile, config.margin
    prior = decode_mask(rle)
    x, y, width, height = mask_util.toBbox(to_rle_dict(rle))
    a, b = max(0, int(x) - margin), max(0, int(y) - margin)
    c = min(FULL_SIZE, int(math.ceil(x + width)) + margin)
    d = min(FULL_SIZE, int(math.ceil(y + height)) + margin)
    windows = [(px, py) for py in tile_starts(b, d, tile)
               for px in tile_starts(a, c, tile)]
    total = np.zeros((d - b, c - a), dtype=np.float32)
    weights = np.zeros_like(total)
    taper = np.maximum(np.hanning(tile).astype(np.float32), config.taper_floor)
    window_weight = taper[:, None] * taper[None, :]
    for start in range(0, len(windows), config.batch):
        locations = windows[start:start + config.batch]
        batch = torch.from_numpy(np.stack([
            patch_features(gray, background, prior, px, py, tile) for px, py in locations
        ])).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            residual = model(batch) - prior_logits_from_input(batch)
        changes = residual[:, 0].float().cpu().numpy()
        for change, (px, py) in zip(changes, locations):
            xa, ya, xb, yb = max(a, px), max(b, py), min(c, px + tile), min(d, py + tile)
            weight = window_weight[ya - py:yb - py, xa - px:xb - px]
            total[ya - b:yb - b, xa - a:xb - a] += change[ya - py:yb - py, xa - px:xb - px] * weight
            weights[ya - b:yb - b, xa - a:xb - a] += weight
    if not np.all(weights > 0) or not np.isfinite(total).all():
        raise ValueError("invalid native tile coverage or nonfinite model output")
    delta = total / weights
    prior_logits = (prior[b:d, a:c].astype(np.float32) * 2 - 1) * PRIOR_LOGIT
    full = np.zeros_like(prior)
    full[b:d, a:c] = (prior_logits + config.alpha * delta > 0).astype(np.uint8)
    return mask_to_rle(full) if full.any() else None


def refine_proposals(
    model: NativeResidualUNet,
    gray: np.ndarray,
    proposals: List[Tuple[str, float]],
    device: torch.device,
    config: RefinerConfig = REFINER,
) -> List[Tuple[str, float]]:
    """Refine every ``(rle, score)`` proposal of one frame; vanished ones are dropped.

    Order and scores are kept. The refined masks may overlap; overlap
    resolution is post-processing (:func:`filament.submission.postprocess`).
    """
    background = background_of(gray, config)
    refined = []
    for rle, score in proposals:
        mask = refine_proposal(model, gray, background, rle, device, config)
        if mask is not None:
            refined.append((mask, score))
    return refined
