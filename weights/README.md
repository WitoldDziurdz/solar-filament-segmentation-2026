# Model weights

The weights of the five published configurations live in this folder, in the
repository itself, so the notebook loads them with no download step.
`filament/pipeline.py` reads `manifest.json`, checks the size and SHA-256 of
every file the chosen configuration lists there, and only then builds the
networks. A missing, truncated or corrupted file, or a Git LFS pointer in place
of the weights, stops the notebook with an error instead of producing
predictions from the wrong weights.

## Files

Five files, 166,831,699 B (159.1 MiB) in all: three detectors and two
refiners, each configuration using one of each.

| File | Size | SHA-256 | Configuration | What it is |
|---|---|---|---|---|
| `detector_yolo11l_seg_1792.pt` | 56,196,577 B (53.6 MiB) | `183a841307827f2167a8d3d981c14df9d1ccbce9147c6a7d60a3536af8bb6eed` | `"013"`, the default, and `"013ft"` | YOLO11l-seg, one class, trained at 1792 px with Ultralytics 8.4.126 on the 565 training images of folds 1-4. An Ultralytics checkpoint: a pickle with float16 weights and no optimizer state. |
| `detector_yolo11m_seg_2048.pt` | 45,638,134 B (43.5 MiB) | `e94255af30540e6cc11e89517c5d5217aacdf68e6caf4297d316a1617929ed44` | `"025"` | YOLO11m-seg, one class, trained at 2048 px with Ultralytics 8.4.126 on the same 565 images. Same format as the default detector. |
| `detector_yolo11l_seg_1792_full.pt` | 56,192,097 B (53.6 MiB) | `6559dea88fc91ba64953fd36bf08dd8605babb25b18d373e48bfc97446220803` | `"032"` and `"035"` | YOLO11l-seg, one class, trained at 1792 px with Ultralytics 8.4.126 on all 707 training images, so no image is held out from it. Same format. |
| `refiner_native_residual.pt` | 4,400,921 B (4.2 MiB) | `5204da7ce821e25fa43473f1bce25b299c7ba1f021c506e5c8df0670195d474b` | `"013"`, `"025"` and `"032"` | The native residual refiner (`NativeResidualUNet`, base 24 channels, 1,096,825 parameters, float32), trained on the default detector's proposals on the same 565 images. Tensors only, loaded with `torch.load(..., weights_only=True)` and restored strictly. |
| `refiner_native_residual_ft.pt` | 4,403,970 B (4.2 MiB) | `0a8eb466be0e5aa2d07eb662e14538cb6a2c447d12d069614c4f26c5ebb10774` | `"013ft"` and `"035"` | The same refiner fine-tuned for 6,000 steps on a 50/50 mix of the default detector's and the 2048 px detector's proposals on the same 565 images (`training/finetune_refiner.py`). Same network and format, plus the training configuration it was fine-tuned with, stored as plain values under one extra key; loaded the same way, and the extra key is not read by the network. |

The detectors and the first refiner were trained on one Kaggle GPU, the
fine-tuned refiner on Apple MPS; the README of the repository describes how.
`manifest.json` holds the same names, sizes and checksums in machine-readable
form. The default configuration's two files are listed under `files`; each
other configuration lists its own two files, a shared one included, under
`configurations.<name>.files`:

```json
{
  "files": [
    {"name": "detector_yolo11l_seg_1792.pt", "role": "detector", "size": 56196577, "sha256": "183a8413..."},
    {"name": "refiner_native_residual.pt", "role": "refiner", "size": 4400921, "sha256": "5204da7c..."}
  ],
  "configurations": {
    "025": {"files": [
      {"name": "detector_yolo11m_seg_2048.pt", "role": "detector", "size": 45638134, "sha256": "e94255af..."},
      {"name": "refiner_native_residual.pt", "role": "refiner", "size": 4400921, "sha256": "5204da7c..."}
    ]},
    "032": {"files": [
      {"name": "detector_yolo11l_seg_1792_full.pt", "role": "detector", "size": 56192097, "sha256": "6559dea8..."},
      {"name": "refiner_native_residual.pt", "role": "refiner", "size": 4400921, "sha256": "5204da7c..."}
    ]},
    "013ft": {"files": [
      {"name": "detector_yolo11l_seg_1792.pt", "role": "detector", "size": 56196577, "sha256": "183a8413..."},
      {"name": "refiner_native_residual_ft.pt", "role": "refiner", "size": 4403970, "sha256": "0a8eb466..."}
    ]},
    "035": {"files": [
      {"name": "detector_yolo11l_seg_1792_full.pt", "role": "detector", "size": 56192097, "sha256": "6559dea8..."},
      {"name": "refiner_native_residual_ft.pt", "role": "refiner", "size": 4403970, "sha256": "0a8eb466..."}
    ]}
  }
}
```

A configuration is checked and loaded from its own list only, so choosing one
never reads another configuration's files.

Check a file by hand with `shasum -a 256 weights/<file>` or
`python -c "from filament.io import file_sha256; print(file_sha256('weights/<file>'))"`.

The Ultralytics checkpoints are Python pickles, and loading a pickle can run
code. That is why the checksum is verified first: only these exact bytes are
ever unpickled.

## How the files are stored

All five files are plain Git objects, each under GitHub's 100 MiB limit for a
single file; the two YOLO11l-seg detectors are above the 50 MiB size at which
GitHub prints a warning on push. A clone therefore contains the weights, and
no Git LFS or other tool is needed. If the files are ever moved to Git LFS, a
clone made without git-lfs installed would hold 130-byte pointer files
instead; the pipeline detects this and says so.

Do not split a file into parts or point to an external download: the
organizers require the weights to be in the repository and loaded by the
notebook.
