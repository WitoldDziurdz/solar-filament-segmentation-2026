# Solar filament instance segmentation

Code, weights and a pipeline notebook for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE Big Data Cup 2026): every solar filament in a full-disk GONG H-alpha image
is segmented as its own instance, and submissions are scored by Panoptic
Quality (PQ).

## Kaggle submissions

Every Kaggle submission this repository rebuilds, and the configuration that
made it (*Published configurations* describes each one):

| Kaggle submission | Public score | Configuration | Recipe | Local PQ, report half (TP / FP / FN) | Rebuilt by (notebook stage 2) |
|---|---|---|---|---|---|
| 58242720 | 0.39 | `"013ft"` | YOLO11l-seg at 1792 px, folds 1-4 (`detector_yolo11l_seg_1792.pt`) + fine-tuned refiner (`refiner_native_residual_ft.pt`), alpha 1.0 | 0.4346 (605 / 305 / 386) | `CONFIGURATION = "013ft"` |
| 58242724 | 0.39 | `"035"` | YOLO11l-seg at 1792 px, all 707 images (`detector_yolo11l_seg_1792_full.pt`) + fine-tuned refiner (`refiner_native_residual_ft.pt`), alpha 1.0 | LB-only: detector trained on all 707 training images | `CONFIGURATION = "035"` |
| 58181980 | 0.39 | `"032"` | YOLO11l-seg at 1792 px, all 707 images (`detector_yolo11l_seg_1792_full.pt`) + refiner (`refiner_native_residual.pt`), alpha 1.0 | LB-only: detector trained on all 707 training images | `CONFIGURATION = "032"` |
| 56828482 | 0.38 | `"013"` (default) | YOLO11l-seg at 1792 px, folds 1-4 (`detector_yolo11l_seg_1792.pt`) + refiner (`refiner_native_residual.pt`), alpha 1.0 | 0.4312 (602 / 308 / 389) | `CONFIGURATION = "013"` (as shipped) |
| 58179404 | 0.38 | `"025"` | YOLO11m-seg at 2048 px, NMS IoU 0.0, folds 1-4 (`detector_yolo11m_seg_2048.pt`) + refiner (`refiner_native_residual.pt`), alpha 1.5 | 0.4275 (566 / 236 / 425) | `CONFIGURATION = "025"` |

**Final selections: 58242720 (`"013ft"`) and 58242724 (`"035"`).**

To rebuild a row's CSV, set `CONFIGURATION` in stage 2 of
`notebooks/pipeline.ipynb` (it ships as `"013"`) and run the whole notebook
after *Setup* and *Data*: `.venv/bin/jupyter execute notebooks/pipeline.ipynb`.
It writes the CSV to `outputs/submission.csv`. On the laptop CPU of *Hardware
and runtime*, the package rebuilt the files submitted for `"013ft"`, `"035"`,
`"032"` and `"025"` byte for byte, and the one submitted for `"013"`, which was
made on a GPU, with the same 1,245 instances up to 201 boundary pixels
(*Reproducing*).

*Local PQ* is measured on the report half of validation fold 0 (71 images,
126 annotator records; *How the pipeline is evaluated*). For scale: annotators
agree with each other at 0.342, one annotator against all annotators reaches
0.598, and the union of all annotators' instances 0.767. *LB-only*: the
detector was trained on all 707 training images, fold 0 included, so no image
is held out from it and its evidence is its public score. *Folds 1-4* are the
565 training images outside fold 0. The public scores cover about half of the
180 test images and are shown with two decimals, so 0.38 against 0.39 is not
significant.

The weights are in the repository, and the notebook loads them, so the
submitted predictions can be rebuilt without retraining and without
downloading anything except the competition data.

## The pipeline

```
2048 x 2048 H-alpha frame (one channel)
  -> 1. detector   YOLO11l-seg at 1792 px          proposals, confidence >= 0.3, masks at 2048 px
  -> 2. refiner    native residual U-Net            each mask corrected on 512 px native tiles
  -> 3. post-processing                             disjoint, best score first
  -> submission.csv                                 one compressed RLE per filament
```

1. **Detector** (`filament/models/detector.py`). Ultralytics YOLO11l-seg with
   one class, fine-tuned at an input size of 1792 px. The frame is replicated
   to three channels and resized by Ultralytics; proposals are kept from
   confidence 0.05 with non-maximum suppression at IoU 0.5 and at most 60 per
   frame, and only those with confidence of at least 0.3 go on. Masks come back
   at native resolution: the mask logits are upsampled to 2048 px before they
   are thresholded, because thresholding first and resizing the binary mask
   shifts every boundary.
2. **Refiner** (`filament/models/refiner.py`). The detector builds its masks
   from 448 x 448 prototypes, so outlines are smooth approximations, and PQ
   is very sensitive to boundaries: a match needs IoU > 0.5 against every
   annotator separately. A four-scale U-Net with 1.1 million parameters looks
   at the native-resolution image around each proposal (grayscale, local
   contrast against a blurred background, the proposal mask, and the signed
   distance to its outline) and predicts a correction to the mask's logits,
   blended over 512 px tiles. Its output layer starts at zero, so an untrained
   refiner returns the detector's mask unchanged.
3. **Post-processing** (`filament/submission.py`). Instances of one image are
   made disjoint best-score-first; an instance that keeps less than half its
   pixels is dropped, because a shrunken mask rarely reaches IoU 0.5 and a
   miss costs a false positive on top of the false negative.

`filament.pipeline.parameters()` lists every setting between the raw frame and
the submitted masks, with where each value came from; stage 3 of the notebook
prints it. The confidence threshold, the NMS IoU and the refiner's weight were
chosen on the tune half of the validation fold; nothing was tuned on the
report half (see *How the pipeline is evaluated*).

## Contents

```
filament/                 the package; the notebook only calls it
  io.py                   images, COCO annotations, polygon -> RLE, prediction files, weight checks
  folds.py                folds grouped on the physical observation, fingerprint, tune / report halves
  models/detector.py      YOLO11l-seg: input format, inference settings, proposals
  models/refiner.py       the refiner network (as trained), tiling, refinement of one proposal
  pipeline.py             weight verification and loading; predict_image, predict_dataset
  submission.py           post-processing, overlap resolution, validation, CSV writing
  metric.py               Panoptic Quality exactly as the organizers compute it
  analysis.py             IoU / Dice distributions, 1:n and n:1 relations, human baseline
  viz.py                  overlays and distribution plots
notebooks/pipeline.ipynb  the whole pipeline, stage by stage
training/                 optional retraining: the training runs behind the published weights
  train_detector.py       step 1: YOLO11l-seg (one GPU, about 6.5 h); --recipe 025: YOLO11m-seg
  train_refiner.py        step 2: the refiner, on the detector's proposals (one GPU, about 4.5 h)
  finetune_refiner.py     step 3: the fine-tuned refiner of "013ft" and "035" (one GPU or Apple MPS, about 1.5 h)
weights/                  the weights of the five configurations, and manifest.json with their sizes and SHA-256
predictions/              the validation predictions recorded when the model was evaluated (fold 0)
tests/                    pytest suite
requirements.txt          every package pinned
LICENSE                   GNU Affero General Public License v3.0 (see *Licence*)
```

The model's contract, for one `(2048, 2048)` uint8 frame from
`filament.io.load_image`:

```python
from filament import pipeline

pipe = pipeline.load_pipeline()                    # verifies weights/, then loads both networks
instances = pipeline.predict_image(image, pipe)    # [(mask_2048 bool, score), ...], may overlap
predictions = pipeline.predict_dataset(folder)     # {stem: [rle, ...]}, post-processed
```

`load_pipeline` and `predict_dataset` take `configuration="025"`, `"032"`,
`"013ft"` or `"035"` for the other published configurations (see
*Published configurations*); without it they run the default, `"013"`.

## Setup

Python 3.10 (tested with 3.10.19).

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

This installs PyTorch 2.5.1 and Ultralytics 8.4.126, the versions the
predictions were reproduced with (on Linux x86-64, the CUDA build of PyTorch,
which also runs on a CPU). A CUDA GPU is used when there is one; nothing needs
one.

## Data

The competition data is not redistributed here. Download it with the Kaggle
command-line client (Kaggle API credentials and accepted competition rules are
needed) and unzip it under `data/`:

```bash
kaggle competitions download -c filament-segmentation-2026 -p data/
unzip data/filament-segmentation-2026.zip -d data/
```

which gives the layout of the competition's Data page:

```
data/MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── MAGFiLO_1.0_Annotations_kaggle2026_train.json   1,154 annotator records, 707 images
│   └── train_images/<stem>.jpeg                         2048 x 2048 grayscale
└── test/
    └── test_images/<stem>.jpeg                          180 images
```

To keep the data elsewhere, set `FILAMENT_DATA_ROOT` to the folder that
contains `train/` and `test/`. Every other path is relative to the repository.

**What the models were trained on.** Only the competition's training data:
the 707 training images and their annotation file,
`MAGFiLO_1.0_Annotations_kaggle2026_train.json`, from the competition's Data
page. Every network was trained, and every setting chosen, on those alone. No
other data was used: no other release or part of MAGFiLO, no other annotations,
no other H-alpha or GONG images, no test labels, and no pseudo-labels on the
test images, which are only ever predicted. The one outside input is where the
detectors start from: Ultralytics' COCO-pretrained `yolo11l-seg.pt` and
`yolo11m-seg.pt` (see *Retraining*). The refiner was trained from scratch, and
its fine-tuned version was trained further from it.

## Reproducing

```bash
.venv/bin/python -m pytest tests -q                  # about 30 s; needs no data, 3 tests use it if present
.venv/bin/jupyter execute notebooks/pipeline.ipynb   # or open it in Jupyter and run the cells in order
```

The notebook loads the weights, predicts the 180 test images and writes
`submission.csv`, then predicts the 71 report-half validation images again and
checks that they reproduce the recorded validation score. It writes to
`outputs/` (override with `FILAMENT_OUTPUT_DIR`): `submission.csv`,
`validation_report_half.json`, `iou_dice_histograms.csv`, `distributions.png`,
`overlays.png`. For a quick look, set `TEST_LIMIT` (a number of test images)
and `RECOMPUTE_VALIDATION = False` in stage 2.

**What reproduces, and how exactly.** The recorded predictions were made on
Kaggle GPUs with a newer PyTorch (the detector on a Tesla T4 with PyTorch
2.10), and the refiner ran there in float16 mixed precision; on a CPU it runs
in float32. On a laptop CPU (Apple M2, PyTorch 2.5.1) the notebook gave:

- *Validation, report half:* TP / FP / FN 602 / 308 / 389, identical to the
  record, and PQ 0.431255 against the recorded 0.431248 (+7e-6). The
  detector's proposals were bit-identical on all 71 images; after refinement 28
  images are bit-identical and the rest differ by 85 boundary pixels in total
  (lowest IoU between an instance and its recorded counterpart 0.997). Stage 7
  prints both sets of counts and the PQ difference, and warns loudly, without
  stopping, when the counts differ or the PQ moves by more than 1e-4.
- *Test set:* the rebuilt `submission.csv` has the same 1,245 instances over
  178 of the 180 images as the submitted file, with the same ids in the same
  order. Every instance matches its submitted counterpart at IoU > 0.5 (mean
  IoU 0.99991, lowest 0.996); 81 images are bit-identical and 201 pixels
  differ in all, so the two files are not byte-identical.

## Hardware and runtime

| What | Hardware | Time |
|---|---|---|
| Model, one frame (detector and refiner) | Apple M2 CPU (8 cores), 8 GB RAM, no GPU | 6.7 to 7.9 s on average over the 180 test frames, 7.4 to 7.5 s over the 71 validation frames (two complete runs) |
| The whole notebook: 180 test and 71 validation frames | same | 30 to 33 min |
| Configurations `"025"` / `"032"`, one frame | same | 8.6 s / 6.9 s on average over the 180 test frames; 10.6 s for `"025"` over the 142 validation frames |
| Configurations `"013ft"` / `"035"`, one frame | same | 7.0 s / 6.8 s on average over the 180 test frames; 8.0 s for `"013ft"` over the 142 validation frames |
| Peak memory of the model | same | about 4.75 GiB |
| Tests | same | about 30 s |
| Optional retraining (`training/`, `RETRAIN = True`) | one CUDA GPU; the published runs used a Tesla T4 | detector about 6.5 h (6 h training), refiner about 4.5 h (4 h training) |
| Optional fine-tune of the refiner (`training/finetune_refiner.py`) | one CUDA GPU or Apple MPS; the published run used an Apple M2 (MPS) | 2048 px detector on the 565 training frames about 47 min (on the CPU), then 6,000 steps in 1.5 h |

Of a frame's time on that CPU, the detector takes about 3.4 s once warmed up,
and the refiner 0.5 to 6 s (3.5 s on average), depending on how many filaments
the frame holds and how large they are.

## How the pipeline is evaluated

**Validation split.** The 1,154 annotator records cover only 707 physical
observations, because up to three experts annotated the same image. Folds are
assigned per observation (`filament/folds.py`), so all annotators of one image
fall in the same fold. Five folds, seed 20260816; both networks were trained
on folds 1-4 (565 images), and fold 0 (142 images) is held out. Its
fingerprint `b990dbd4...7bdee` is checked by the notebook and by
`tests/test_folds.py`. Fold 0 is split into a tune half and a report half of
71 images each: thresholds are chosen on the tune half only, and the report
half is scored once. The report half was also used to decide between
candidate pipelines, so the score of the one chosen carries a small selection
effect.

**Metric.** Panoptic Quality as in the organizers' self-evaluation notebook
(`filament/metric.py`, checked against a literal transcription of their code in
`tests/test_metric_matches_official.py`). Every annotator record of an image is
a separate ground truth; a prediction matches when IoU > 0.5, and
PQ = SQ x RQ. Measured on the training annotations with the same code, two
annotators agree with each other at PQ 0.342, one annotator scored against all
annotators of an image reaches 0.598, and the union of all annotators'
instances reaches 0.767. The first two are reproduced by
`tests/test_analysis.py`.

**Committee criteria** (`filament/analysis.py`), per annotator record:

- IoU = I / (A + B - I) and Dice = 2I / (A + B) for every overlapping pair
  (IoU > 0), reported as 50-bin normalised histograms on [0, 1] with their means;
- per annotated filament, the number of predictions it overlaps:
  1:0 missed, 1:1, 1:n fragmented;
- per prediction, the number of annotated filaments it overlaps:
  0:1 missed, 1:1, n:1 merged.

The same numbers with one annotator in place of the model
(`analysis.human_baseline`) give the human reference.

## Results

Validation fold 0, report half (71 images, 126 annotator records); the
thresholds were chosen on the tune half beforehand.

| Model | PQ | SQ | RQ | TP | FP | FN | Public leaderboard |
|---|---|---|---|---|---|---|---|
| Detector and refiner (this repository) | **0.4312** | 0.681 | 0.633 | 602 | 308 | 389 | 0.38 |
| Detector alone | 0.4153 | 0.669 | 0.621 | 590 | 319 | 401 | 0.36 |

The refiner adds 0.016 PQ (paired bootstrap over images, 2,000 resamples:
95 % interval 0.006 to 0.027). For scale: annotators agree with each other at
0.342, and one annotator against all annotators reaches 0.598. On the tune
half, where the settings were chosen, the pipeline scores 0.4556
(TP 525, FP 277, FN 248). The public leaderboard scores cover about half of the
180 test images and are shown with two decimals.

The committee's criteria on the report half (stage 7 of the notebook prints
them with the human baseline next to them):

| | Model | Annotator vs annotator, all training images |
|---|---|---|
| mean IoU, overlapping pairs | 0.584 | 0.517 |
| mean Dice, overlapping pairs | 0.715 | 0.661 |
| annotated filaments missed (1:0) | 21.7 % | 17.8 % |
| annotated filaments fragmented (1:n) | 3.1 % | 3.2 % |
| predictions matching nothing (0:1) | 15.6 % | 17.8 % |
| predictions merging filaments (n:1) | 4.5 % | 3.2 % |

`predictions/validation_yolo11l_residual.json` holds the recorded validation
predictions as `{"predictions": {stem: [rle, ...]}, "fold_fingerprint": ...,
"prediction_sha256": ...}`; `filament.io.load_prediction_file` verifies the
digest on load.

## Published configurations

Five configurations of the same pipeline are published. The names are the
numbers of the runs that evaluated them (`"013ft"` is `"013"` with the
fine-tuned refiner), and `"013"`, the model described above, is the default
everywhere. `"025"` and `"032"` change the detector, and
`"025"` also the refiner's weight; `"013ft"` and `"035"` are `"013"` and
`"032"` with the fine-tuned refiner weights. Nothing else differs:

| | `"013"` (default) | `"025"` | `"032"` | `"013ft"` | `"035"` |
|---|---|---|---|---|---|
| Detector | YOLO11l-seg at 1792 px, trained on the 565 images of folds 1-4 | YOLO11m-seg at 2048 px (the frame is not resized), trained on the same 565 images | YOLO11l-seg at 1792 px, the default's recipe trained on all 707 training images | the default's | `"032"`'s |
| Detector weights | `weights/detector_yolo11l_seg_1792.pt` | `weights/detector_yolo11m_seg_2048.pt` | `weights/detector_yolo11l_seg_1792_full.pt` | `weights/detector_yolo11l_seg_1792.pt` | `weights/detector_yolo11l_seg_1792_full.pt` |
| NMS IoU | 0.5 | 0.0 | 0.5 | 0.5 | 0.5 |
| Confidence floor, proposals per frame, operating point | 0.05, 60, 0.3 | 0.05, 60, 0.3 | 0.05, 60, 0.3 | 0.05, 60, 0.3 | 0.05, 60, 0.3 |
| Refiner | `weights/refiner_native_residual.pt`, alpha 1.0 | the same weights, alpha 1.5 | the same weights, alpha 1.0 | `weights/refiner_native_residual_ft.pt`, alpha 1.0 | the fine-tuned weights, alpha 1.0 |
| Tiling, post-processing, `submission.csv` writer | as described above | the same | the same | the same | the same |

What each was measured at, on the report half of validation fold 0 (71
images, 126 annotator records) and on the public leaderboard:

| | PQ | SQ | RQ | TP | FP | FN | Public leaderboard |
|---|---|---|---|---|---|---|---|
| `"013"` | **0.4312** | 0.681 | 0.633 | 602 | 308 | 389 | 0.38 (submission 56828482) |
| `"025"` | **0.4275** | 0.677 | 0.631 | 566 | 236 | 425 | 0.38 (submission 58179404) |
| `"025"`, its detector alone | 0.4188 | 0.669 | 0.626 | 561 | 241 | 430 | |
| `"032"` | no local score: its detector was trained on every validation image | | | | | | 0.39 (submission 58181980) |
| `"013ft"` | **0.4346** | 0.683 | 0.637 | 605 | 305 | 386 | 0.39 (submission 58242720) |
| `"035"` | no local score: `"032"`'s detector, trained on every validation image | | | | | | 0.39 (submission 58242724) |

For scale: annotators agree with each other at 0.342, one annotator against
all annotators reaches 0.598, and the union of all annotators' instances
0.767. The public scores cover about half of the 180 test images and are shown
with two decimals, so a difference of 0.01 between them is not significant.

- **`"025"`.** Its detector's NMS IoU and operating point were chosen on the
  tune half by that detector's own training run, as the default's were, from
  NMS IoU 0.0 / 0.5 / 0.7 and confidences 0.10-0.80. The refiner's alpha was chosen on the tune half from
  0.5 / 1.0 / 1.5, and 1.5 is the edge of that grid. The refiner adds 0.009
  PQ on this detector, against 0.016 on the default's; on the tune half the
  configuration scores 0.4493 (TP 491, FP 228, FN 282).
- **`"032"`.** The default's settings, unchanged: its operating point and
  alpha were chosen for the default detector on the tune half, and with every
  image used for training none is left to choose them again.
- **`"013ft"`.** The default with the refiner fine-tuned from the published
  one (`training/finetune_refiner.py`, see *Retraining*). Its alpha was chosen
  on the tune half from 0.5 / 1.0 / 1.5 / 2.0; on the tune half the
  configuration scores 0.4569 (TP 526, FP 280, FN 247). On the report half it
  is 0.0033 PQ above `"013"` (paired bootstrap over images, 2,000 resamples:
  95 % interval 0.0009 to 0.0056). Of that, 0.0014 comes from better outlines
  on the 66 images whose TP / FP / FN do not change, and 0.0019 from 5 images
  where a refined outline crossed IoU 0.5 (four matches gained, one lost).
  Only the outline part replicates: on the tune half, and for the fine-tuned
  refiner on `"025"`'s detector (against `"025"`) on both halves, it is 0.0011
  to 0.0016 each time, while the part from crossings has no consistent sign
  (-0.0012 to +0.0019 over the four comparisons). Over all 142 validation
  images `"013ft"` is 0.0024 above `"013"` (the same bootstrap: 95 % interval
  0.0004 to 0.0043), and that is the gain we credit it with: the report
  half's 0.0033 carries the selection effect described in *How the pipeline
  is evaluated*, and the tune half alone gives 0.0013. Its test predictions
  are nearly the same instances as `"013"`'s.
- **`"035"`.** `"032"` with the fine-tuned refiner at `"013ft"`'s alpha: with
  every image used for training its detector, none is left to choose it
  again, so its evidence is its public score.

**Running them.**

```python
pipe = pipeline.load_pipeline(configuration="025")      # verifies its two files, then loads them
predictions = pipeline.predict_dataset(folder, configuration="013ft")
```

In the notebook, set `CONFIGURATION` in stage 2; stages 4 to 8 then use it.
Choosing one configuration never reads another configuration's weight files.
The parameter table of stage 3 still describes the default. Stage 7 compares
the recomputed report half of `"025"` and of `"013ft"` with the TP / FP / FN
and PQ recorded when each was evaluated; their per-image predictions are not
in `predictions/`, so this needs `RECOMPUTE_VALIDATION = True`. Stage 7 does
not score `"032"` or `"035"`, because their detector was trained on every
validation image and a score there would be in-sample: it shows the
annotators' numbers only, and stage 8 draws no figures.

**Reproduction.** On the laptop CPU of *Hardware and runtime*, all four
rebuild exactly what was recorded when they were evaluated and submitted:

- `"025"`: all 142 validation images are bit-identical to the recorded
  predictions (report half PQ 0.4275, TP 566 / FP 236 / FN 425), and the
  rebuilt `submission.csv` has the submitted file's 1,121 rows, unchanged;
- `"032"`: the rebuilt `submission.csv` has the submitted file's 1,206 rows,
  unchanged;
- `"013ft"`: all 142 validation images are bit-identical to the recorded
  predictions (report half PQ 0.4346, TP 605 / FP 305 / FN 386), and the
  rebuilt `submission.csv` has the submitted file's 1,246 rows, unchanged;
- `"035"`: the rebuilt `submission.csv` has the submitted file's 1,209 rows,
  unchanged.

All four recorded runs used the refiner in float32 on a CPU like this one. On
a GPU it runs in float16, so a few boundary pixels can differ there (see
*Reproducing*). A frame takes about 9-11 s with `"025"` and about 7 s with
`"032"`, `"013ft"` or `"035"`.

**Retraining.** The detector of `"025"` has its own recipe option,
`training/train_detector.py --recipe 025` (see *Retraining*). The detector of
`"032"` and `"035"` was trained by the default recipe in a full-data mode that
`training/train_detector.py` does not have, since the script keeps fold 0 out
for evaluation: the two checkpoints record the same training arguments apart
from the number of epochs the 6-hour budget allowed (78 on 565 images, 49 on
707). `"013"`, `"025"` and `"032"` use the refiner trained on the default
detector's proposals; `"013ft"` and `"035"` use its fine-tuned version, which
`training/finetune_refiner.py` trains (see *Retraining*).

## Retraining (optional)

The notebook loads the published weights and trains nothing by default. The
scripts in `training/` are the training runs that produced those weights: the
first two as they ran on Kaggle, changed only to take their paths as
arguments (each script's docstring lists what changed), and the third, the
fine-tune, as it ran on a laptop. Run them from the repository root, the
detector first:

```bash
.venv/bin/python training/train_detector.py --data-root data/MAGFiLO_1.0_Kaggle_2026 \
    --output-dir outputs/retrain/detector
.venv/bin/python training/train_refiner.py --data-root data/MAGFiLO_1.0_Kaggle_2026 \
    --detector-run outputs/retrain/detector --output-dir outputs/retrain/refiner
```

or set `RETRAIN = True` in stage 2 of the notebook, which runs both and then
uses the new weights for every later stage (after writing a manifest of their
checksums).

The first two scripts need one CUDA GPU and stop at once on a CPU. On the Tesla T4
the published weights were trained on, the detector took about 6.5 hours (a
6-hour training budget, which gave 78 epochs) and the refiner about 4.5 hours
(a 4-hour budget, 31,743 steps). `FILAMENT_SMOKE=1` runs either script on a
handful of images on a CPU, as a check that the code runs. The first script
downloads Ultralytics' COCO-pretrained `yolo11l-seg.pt`, so it needs internet.
Each run also repeats the evaluation, choosing its thresholds on the tune half
and scoring the report half once, and writes its reports next to the
checkpoint.

Note: `training/train_refiner.py` expects the detector run report to match the shipped
detector's recipe (YOLO11l-seg at imgsz 1792, conf 0.30, NMS 0.5) and stops otherwise. A
retrained detector with other inference settings needs those asserts updated before the
refiner is trained.

The detector of configuration `"025"` has its own recipe option:
`training/train_detector.py --recipe 025` trains YOLO11m-seg (from the COCO-pretrained
`yolo11m-seg.pt`) at imgsz 2048 instead, with everything else unchanged; on a Tesla T4 the
same 6-hour budget gave 74 epochs, and its tune half chose NMS IoU 0.0 and confidence 0.30.
`weights/detector_yolo11m_seg_2048.pt` is a byte copy of the `runs/seg/weights/last.pt`
that run wrote. The refiner is not retrained for it: `"025"` uses the refiner trained on the
default detector's proposals, at alpha 1.5. The detector of `"032"`, trained on all 707
images, has no option here (see *Published configurations*). The notebook's `RETRAIN` path
retrains the default configuration only.

The fine-tuned refiner of `"013ft"` and `"035"` is the third step,
`training/finetune_refiner.py`. It reads the refiner run's folder (its examples: the default
detector's proposals matched to the annotators' outlines), runs the 2048 px detector of
`"025"` on the same 565 images for a second set of examples, and fine-tunes the published
refiner on a 50/50 mix of the two:

```bash
.venv/bin/python training/finetune_refiner.py --data-root data/MAGFiLO_1.0_Kaggle_2026 \
    --refiner-run outputs/retrain/refiner --output-dir outputs/retrain/refiner_ft
```

It needs one CUDA GPU or Apple MPS (Apple silicon) and stops at once on a machine with
neither. The published weights were fine-tuned on Apple MPS (M2, 8 GB): 6,000 steps in 1.5
hours, after the 2048 px detector had run on the CPU for about 47 minutes. It does not choose
alpha or score anything: the published configurations use alpha 1.0, which the evaluation of
the fine-tuned weights chose on the tune half. `FILAMENT_SMOKE=1` runs it on 4 images and 4
steps on a CPU, as a check of the code. Its output, `refiner_native_residual_ft.pt`, replaces
the published file once its size and SHA-256 are written into `weights/manifest.json` for
`"013ft"` and `"035"`.

The recipes, on the 565 images of folds 1-4:

- **Detector:** Ultralytics 8.4.126, fine-tuned from the COCO-pretrained
  `yolo11l-seg.pt` at imgsz 1792, batch 2, seed 20260816, a 6-hour budget
  (78 epochs), single class. Augmentation: horizontal and vertical flips,
  rotation up to 180 degrees, no mosaic (each frame is one centred solar
  disc), Ultralytics defaults otherwise. Targets: per image, the union of all
  annotators' filaments, deduplicated by complete-link clustering at
  IoU > 0.5 with one representative outline per cluster (6,435 annotated
  outlines became 5,143 targets).
- **Refiner:** the frozen detector's proposals on the same images, each
  matched to a cluster of annotator outlines (a positive when its best IoU
  exceeds 0.1, a negative when it overlaps nothing). 512 px crops around a
  proposal, random flips, and in 35 % of positives the proposal dilated or
  eroded by 1-4 px first. Loss: binary cross-entropy plus soft Dice, averaged
  over the annotators who drew the matched filament, plus a small penalty on
  the size of the correction. AdamW, learning rate 3e-4 with a cosine
  schedule, batch 4, 31,743 steps in 4 hours.
- **Fine-tuned refiner:** the published refiner, trained further for 6,000
  steps on the same images: half of every batch of four from the default
  detector's examples (7,713 positives, 2,667 negatives) and half from the
  2048 px detector's (4,357 positives, 1,691 negatives), with the refiner's
  own crops, augmentation and loss. AdamW, learning rate 1e-4 with 100
  warm-up steps and a cosine schedule to 0, batch 4, float32, seed 20261110.

The docstrings of `filament/models/detector.py` and `filament/models/refiner.py`
give the full recipes. A retrained model will not reproduce the published
predictions pixel for pixel. The first two runs stop on a time budget, so how
many steps they take depends on the GPU, and GPU training is not bit-for-bit
deterministic in any case; the fine-tune takes 6,000 steps on any device, and
draws the same crops, but its device's kernels are not deterministic either.

## Credits

- Data: MAGFiLO, A. Ahmadzadeh et al., "A dataset of manually annotated
  filaments from H-alpha observations", *Scientific Data* 11, 1031 (2024),
  doi:10.1038/s41597-024-03876-y. Images from the GONG network.
- Metric: the organizers' `self-evaluation-notebook` (A. Ahmadzadeh), ported in
  `filament/metric.py`; Panoptic Quality from A. Kirillov et al., "Panoptic
  Segmentation", CVPR 2019, doi:10.1109/CVPR.2019.00963.
- Detector: Ultralytics YOLO11 (G. Jocher and J. Qiu, 2024,
  https://github.com/ultralytics/ultralytics), AGPL-3.0.
- Refiner: a U-Net (O. Ronneberger, P. Fischer and T. Brox, "U-Net:
  Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015,
  doi:10.1007/978-3-319-24574-4_28) with group normalisation, in PyTorch.
- Run-length encoding and IoU on masks: `pycocotools`; image operations: OpenCV.

## Licence

This repository is released under the GNU Affero General Public License v3.0
(AGPL-3.0); the full text is in `LICENSE`. That covers our code (`filament/`,
`training/`, `tests/`), the notebook, and the weights in `weights/`.

- **Ultralytics YOLO11**, which the detectors are built on, is licensed under
  AGPL-3.0 too. It is a dependency, installed by `requirements.txt` and not
  included here. The three detector checkpoints in `weights/` were fine-tuned
  from Ultralytics' COCO-pretrained `yolo11l-seg.pt` and `yolo11m-seg.pt`, and
  each records AGPL-3.0 as its licence. `LICENSE` is the copy of the licence
  that Ultralytics 8.4.126 distributes, unchanged.
- **The refiner** (`filament/models/refiner.py` and the two
  `refiner_native_residual*.pt` files) is our own network, trained from
  scratch on the competition data; the fine-tuned file was trained further
  from the first.
- **The competition data** are not redistributed here; download them from
  Kaggle under the competition's rules (*Data*).
- The other packages in `requirements.txt` are installed from PyPI under
  their own licences and are not included here.
