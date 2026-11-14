"""The two networks of the pipeline.

``detector``
    YOLO11l-seg (Ultralytics) at 1792 px: proposes filament instances, each with
    a confidence score and a native-resolution mask.
``refiner``
    A four-scale U-Net on native-resolution tiles: predicts a per-pixel
    correction to each proposal's mask (the native residual refiner).

:mod:`filament.pipeline` chains them. Neither module reads anything but the
weight file it is given.
"""
