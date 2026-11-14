"""Loading, geometry, frames, prediction files and weight checks."""

import json

import numpy as np
import pytest
from pycocotools import mask as mask_util

from filament.io import (
    LFS_POINTER_PREFIX,
    annotator_of,
    build_gt_frame,
    build_pred_frame,
    file_sha256,
    list_images,
    load_image,
    load_prediction_file,
    mask_to_rle,
    physical_stem,
    polygons_to_rle,
    prediction_digest,
    rle_to_mask,
    save_prediction_file,
    verify_weights,
)

SIZE = 64


def area(rle, size=SIZE):
    return int(mask_util.area({"size": [size, size], "counts": rle.encode("ascii")}))


def test_record_ids_split_into_annotator_and_stem():
    assert physical_stem("040301-20140609195854Bh") == "20140609195854Bh"
    assert annotator_of("040301-20140609195854Bh") == "040301"


def test_polygon_rings_merge_into_one_instance():
    ring_a = [0, 0, 10, 0, 10, 10, 0, 10]
    ring_b = [30, 30, 40, 30, 40, 40, 30, 40]
    one = polygons_to_rle([ring_a], SIZE, SIZE)
    both = polygons_to_rle([ring_a, ring_b], SIZE, SIZE)
    assert isinstance(both, str)
    assert area(both) == 2 * area(one) > 0


def test_mask_rle_round_trip():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[5:20, 7:9] = True
    assert (rle_to_mask(mask_to_rle(mask), SIZE, SIZE) == mask).all()


def tiny_coco():
    square = [[0, 0, 8, 0, 8, 8, 0, 8]]
    return {
        "images": [{"id": "a1-stemA", "file_name": "stemA.jpeg", "height": SIZE, "width": SIZE},
                   {"id": "a2-stemA", "file_name": "stemA.jpeg", "height": SIZE, "width": SIZE},
                   {"id": "a1-stemB", "file_name": "stemB.jpeg", "height": SIZE, "width": SIZE}],
        "annotations": [
            {"image_id": "a1-stemA", "segmentation": square, "category_id": 1, "area": 64},
            {"image_id": "a1-stemA", "segmentation": square, "category_id": 2, "area": 64},
            {"image_id": "a2-stemA", "segmentation": square, "category_id": 3, "area": 64},
            {"image_id": "a1-stemB", "segmentation": square, "category_id": 1, "area": 64},
        ],
    }


def test_gt_frame_ids_follow_the_evaluator_convention():
    gt = build_gt_frame(tiny_coco(), SIZE, SIZE)
    assert list(gt["filament_id"]) == ["a1-stemA_0", "a1-stemA_1", "a2-stemA_0", "a1-stemB_0"]
    assert list(gt["stem"]) == ["stemA", "stemA", "stemA", "stemB"]
    assert list(gt["annotator"]) == ["a1", "a1", "a2", "a1"]


def test_gt_frame_can_be_restricted_to_stems():
    gt = build_gt_frame(tiny_coco(), SIZE, SIZE, stems=["stemB"])
    assert list(gt["filament_id"]) == ["a1-stemB_0"]
    empty = build_gt_frame(tiny_coco(), SIZE, SIZE, stems=[])
    assert empty.empty and "filament_id" in empty.columns


def test_pred_frame_numbers_instances_per_stem():
    pred = build_pred_frame({"stemA": ["x", "y"], "stemB": ["z"]})
    assert list(pred["filament_id"]) == ["stemA_0", "stemA_1", "stemB_0"]


def test_list_images_filters_and_sorts(tmp_path):
    for name in ("b.jpeg", "a.jpg", "c.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.jpg", "b.jpeg"]
    assert list_images(tmp_path / "absent") == []


def test_load_image_checks_size(tmp_path):
    import cv2

    good = tmp_path / "good.jpeg"
    cv2.imwrite(str(good), np.full((2048, 2048), 90, dtype=np.uint8))
    image = load_image(good)
    assert image.shape == (2048, 2048) and image.dtype == np.uint8

    small = tmp_path / "small.jpeg"
    cv2.imwrite(str(small), np.zeros((100, 100), dtype=np.uint8))
    with pytest.raises(ValueError, match="expected"):
        load_image(small)
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpeg")


def test_prediction_file_round_trip_and_tamper_check(tmp_path):
    predictions = {"stemA": ["abc", "def"], "stemB": []}
    path = tmp_path / "pred.json"
    digest = save_prediction_file(path, predictions, fold_fingerprint="f" * 64, model="demo")
    assert digest == prediction_digest(predictions)

    loaded, fingerprint = load_prediction_file(path)
    assert loaded == predictions and fingerprint == "f" * 64

    payload = json.loads(path.read_text())
    payload["predictions"]["stemA"].append("ghi")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sha256"):
        load_prediction_file(path)


def test_prediction_file_accepts_bare_and_scored_layouts(tmp_path):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"stemA": ["abc"]}))
    assert load_prediction_file(bare) == ({"stemA": ["abc"]}, None)

    scored = tmp_path / "scored.json"
    scored.write_text(json.dumps({"predictions": {"stemA": [{"rle": "abc", "score": 0.7}]}}))
    assert load_prediction_file(scored) == ({"stemA": ["abc"]}, None)


def test_verify_weights_catches_missing_pointer_and_corrupt_files(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"\x00weights\x01" * 100)
    assert verify_weights(weights, file_sha256(weights)) == weights

    with pytest.raises(ValueError, match="sha256"):
        verify_weights(weights, "0" * 64)
    with pytest.raises(FileNotFoundError):
        verify_weights(tmp_path / "absent.pt", "0" * 64)

    pointer = tmp_path / "pointer.pt"
    pointer.write_bytes(LFS_POINTER_PREFIX + b"v1\noid sha256:abc\nsize 123\n")
    with pytest.raises(RuntimeError, match="LFS"):
        verify_weights(pointer, "0" * 64)
