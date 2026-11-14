"""The native residual refiner: model contract, tiling, and refinement of one proposal."""

import numpy as np
import pytest
import torch

from filament.io import mask_to_rle, rle_to_mask
from filament.models import refiner
from filament.models.refiner import (
    REFINER,
    NativeResidualUNet,
    RefinerConfig,
    background_of,
    patch_features,
    refine_proposal,
    tile_starts,
)
from filament.pipeline import WEIGHTS_DIR

SIZE = 2048


def frame(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(40, 200, size=(SIZE, SIZE), dtype=np.uint8)


def box_mask(r0, c0, r1, c1):
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def identity_model():
    """An untrained refiner: its zero-initialised head predicts no correction."""
    return NativeResidualUNet(base_channels=8).eval()


def test_the_embedded_self_test_passes():
    result = refiner.self_test()
    assert result["passed"] and len(result["checks"]) == 10


def test_tiles_cover_the_interval_at_native_resolution():
    for lower, upper in [(0, 100), (300, 900), (1500, 2048), (0, 2048), (1900, 2048), (700, 1212)]:
        starts = tile_starts(lower, upper, 512)
        assert starts[0] <= lower and starts[-1] + 512 >= upper
        assert all(0 <= s <= SIZE - 512 for s in starts)
        assert all(b - a <= 256 for a, b in zip(starts, starts[1:]))   # half-tile stride, no gaps


@pytest.mark.parametrize("box", [(900, 900, 960, 1300),     # one tile
                                 (100, 50, 180, 1400),      # several tiles along x
                                 (0, 0, 40, 40),            # touches two image borders
                                 (2000, 1990, 2048, 2048)])  # the opposite corner
def test_an_untrained_refiner_returns_the_proposal_unchanged(box):
    image = frame()
    proposal = mask_to_rle(box_mask(*box))
    refined = refine_proposal(identity_model(), image, background_of(image), proposal,
                              torch.device("cpu"), RefinerConfig(batch=2))
    assert refined == proposal


def test_only_the_window_around_the_proposal_can_change():
    model = identity_model()
    with torch.no_grad():
        model.delta_head.bias.fill_(10.0)        # pushes every pixel of every tile to foreground
    image = frame()
    r0, c0, r1, c1 = 1000, 1000, 1040, 1100
    refined = rle_to_mask(refine_proposal(model, image, background_of(image),
                                          mask_to_rle(box_mask(r0, c0, r1, c1)),
                                          torch.device("cpu"), REFINER))
    rows, cols = np.nonzero(refined)
    margin = REFINER.margin
    assert rows.min() == r0 - margin and rows.max() == r1 + margin - 1
    assert cols.min() == c0 - margin and cols.max() == c1 + margin - 1


def test_a_refinement_that_erases_the_proposal_returns_none():
    model = identity_model()
    with torch.no_grad():
        model.delta_head.bias.fill_(-10.0)
    image = frame()
    assert refine_proposal(model, image, background_of(image), mask_to_rle(box_mask(10, 10, 50, 300)),
                           torch.device("cpu"), REFINER) is None


def test_alpha_zero_is_the_detector_mask():
    model = identity_model()
    with torch.no_grad():
        model.delta_head.bias.fill_(-10.0)
    image = frame()
    proposal = mask_to_rle(box_mask(10, 10, 50, 300))
    assert refine_proposal(model, image, background_of(image), proposal, torch.device("cpu"),
                           RefinerConfig(alpha=0.0)) == proposal


def test_patch_features_are_the_four_documented_channels():
    image = frame()
    prior = box_mask(100, 100, 150, 400).astype(np.uint8)
    features = patch_features(image, background_of(image), prior, 0, 0, 512)
    assert features.shape == (4, 512, 512) and features.dtype == np.float32
    gray, contrast, mask, signed = features
    assert np.allclose(gray, image[:512, :512] / 255.0)
    assert contrast.min() >= -2.0 and contrast.max() <= 2.0
    assert set(np.unique(mask)) == {0.0, 1.0}
    assert signed.min() >= -1.0 and signed.max() <= 1.0
    assert (signed[mask == 1] > 0).all() and (signed[mask == 0] < 0).all()


def test_background_is_a_sigma_15_gaussian_blur():
    import cv2

    image = frame()
    assert np.array_equal(background_of(image), cv2.GaussianBlur(image, (0, 0), 15))


def test_the_published_refiner_loads_strictly(tmp_path):
    model = refiner.load_refiner(WEIGHTS_DIR / "refiner_native_residual.pt")
    assert isinstance(model, NativeResidualUNet) and not model.training
    assert model.base_channels == 24
    assert sum(p.numel() for p in model.parameters()) == 1_096_825

    payload = torch.load(WEIGHTS_DIR / "refiner_native_residual.pt", map_location="cpu", weights_only=True)
    del payload["model"]["delta_head.bias"]
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(RuntimeError, match="delta_head.bias"):
        refiner.load_refiner(broken)
