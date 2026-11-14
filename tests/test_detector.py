"""The YOLO11l-seg detector: input format, mask conversion and the published checkpoint."""

import numpy as np
import pytest

from filament.io import rle_to_mask
from filament.models import detector
from filament.models.detector import DETECTOR, detector_input, masks_to_proposals


def test_detector_input_is_the_frame_replicated_to_three_channels():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(2048, 2048), dtype=np.uint8)
    out = detector_input(image)
    assert out.shape == (2048, 2048, 3) and out.dtype == np.uint8
    assert all(np.array_equal(out[..., c], image) for c in range(3))


def test_detector_input_rejects_anything_but_a_native_uint8_frame():
    with pytest.raises(ValueError, match="uint8"):
        detector_input(np.zeros((1792, 1792), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        detector_input(np.zeros((2048, 2048), dtype=np.float32))


def test_masks_become_native_rles_with_their_scores_in_order():
    masks = np.zeros((3, 2048, 2048), dtype=np.uint8)
    masks[0, 10:20, 30:400] = 1
    masks[2, 1000:1100, 5:9] = 1          # masks[1] is empty and must be dropped
    proposals = masks_to_proposals(masks, np.array([0.9, 0.8, 0.4], dtype=np.float32))
    assert [score for _, score in proposals] == pytest.approx([0.9, 0.4])
    assert np.array_equal(rle_to_mask(proposals[0][0]), masks[0].astype(bool))
    assert np.array_equal(rle_to_mask(proposals[1][0]), masks[2].astype(bool))
    assert all(isinstance(score, float) for _, score in proposals)


def test_masks_at_another_resolution_are_refused():
    with pytest.raises(ValueError, match="2048"):
        masks_to_proposals(np.zeros((1, 448, 448), dtype=np.uint8), np.array([0.9]))


def test_the_operating_point_is_the_one_chosen_on_the_tune_half():
    assert (DETECTOR.imgsz, DETECTOR.nms_iou, DETECTOR.confidence) == (1792, 0.5, 0.3)
    assert DETECTOR.conf_floor < DETECTOR.confidence and DETECTOR.max_det == 60


def test_the_published_detector_is_a_one_class_segmentation_model():
    from filament.pipeline import WEIGHTS_DIR

    model = detector.load_detector(WEIGHTS_DIR / "detector_yolo11l_seg_1792.pt")
    assert model.task == "segment" and len(model.names) == 1
