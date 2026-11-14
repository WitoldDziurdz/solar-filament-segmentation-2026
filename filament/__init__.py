"""Instance segmentation of solar filaments in GONG H-alpha images.

The package holds everything the pipeline notebook runs. Each module does one
job:

``io``
    Image and COCO annotation loading, polygon -> RLE conversion, ground-truth
    and prediction frames, prediction files, weight-file checks.
``folds``
    Cross-validation folds grouped on the physical observation, their
    fingerprint, and the tune / report halves of the validation fold.
``models.detector``
    YOLO11l-seg at 1792 px (YOLO11m-seg at 2048 px in configuration
    ``"025"``), which proposes filament instances.
``models.refiner``
    The native residual refiner, which corrects each proposal's mask at full
    resolution (fine-tuned weights in configurations ``"013ft"`` and
    ``"035"``).
``pipeline``
    Weight verification and loading, and the two models chained:
    ``predict_image`` for one frame, ``predict_dataset`` for a folder. Five
    configurations are published (``CONFIGURATIONS``): ``"013"`` (the
    default), ``"025"``, ``"032"``, ``"013ft"`` and ``"035"``. On the report
    half of validation fold 0, ``"013"`` scores PQ 0.4312 (TP 602 / FP 308 /
    FN 389), ``"025"`` 0.4275 (566 / 236 / 425) and ``"013ft"`` 0.4346
    (605 / 305 / 386); ``"032"`` and ``"035"`` have no local score, because
    their detector was trained on every validation image, and are judged by
    their public leaderboard score. For scale: two annotators agree at 0.342,
    one annotator against all of them reaches 0.598, and the union of all
    annotators 0.767. The README gives the details.
``submission``
    Post-processing of raw model output, overlap resolution, validation and
    CSV writing.
``metric``
    Panoptic Quality exactly as the competition computes it.
``analysis``
    IoU and Dice distributions of overlapping pairs, one-to-many /
    many-to-one counts, and the same numbers between annotators.
``viz``
    Overlay figures of annotations and predictions (needs matplotlib).

Every mask in this package is a 2048 x 2048 binary image stored as a
compressed COCO run-length encoding (the ``counts`` string).
"""

__version__ = "1.0.0"

IMAGE_SIZE = 2048
