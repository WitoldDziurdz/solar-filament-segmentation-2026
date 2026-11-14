"""Figures: annotation and prediction overlays, and the committee's distributions.

Colour carries one meaning throughout: blue is human annotation (or the
human-vs-human baseline), orange is the model. Needs matplotlib.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .analysis import MatchAnalysis, iou_dice_histograms, relation_counts
from .io import IMAGE_SIZE, rle_to_mask

ANNOTATION_COLOR = "#2a78d6"   # categorical slot 1
PREDICTION_COLOR = "#eb6834"   # categorical slot 2
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"


def apply_style() -> None:
    """Recessive axes and gridlines, small sans text; sized for a two-column paper."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "legend.frameon": False,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": AXIS,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------

def densest_window(rles: Iterable[str], side: int = 1024, step: int = 64,
                   size: int = IMAGE_SIZE) -> Tuple[int, int, int, int]:
    """``(y0, y1, x0, x1)`` of the ``side`` x ``side`` window holding the most instances.

    An instance counts when its bounding-box centre is inside the window; ties
    go to the window closest to the image centre, so the choice is deterministic.
    """
    from pycocotools import mask as mask_util

    centres = []
    for rle in rles:
        x, y, w, h = mask_util.toBbox({"size": [size, size], "counts": rle.encode("ascii")})
        centres.append((y + h / 2.0, x + w / 2.0))
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    best, best_key = (0, side, 0, side), None
    for y0 in range(0, size - side + 1, step):
        for x0 in range(0, size - side + 1, step):
            inside = ((centres[:, 0] >= y0) & (centres[:, 0] < y0 + side)
                      & (centres[:, 1] >= x0) & (centres[:, 1] < x0 + side)).sum()
            off_centre = abs(y0 + side / 2 - size / 2) + abs(x0 + side / 2 - size / 2)
            key = (int(inside), -off_centre)
            if best_key is None or key > best_key:
                best, best_key = (y0, y0 + side, x0, x0 + side), key
    return best


def draw_overlay(ax, image: np.ndarray, annotation_rles: Sequence[str],
                 prediction_rles: Sequence[str],
                 window: Optional[Tuple[int, int, int, int]] = None,
                 linewidth: float = 0.6) -> None:
    """Grayscale frame with annotation outlines (blue) and prediction outlines (orange).

    ``annotation_rles`` may hold several annotators' instances; each instance
    is outlined separately so disagreements stay visible.
    """
    y0, y1, x0, x1 = window or (0, image.shape[0], 0, image.shape[1])
    ax.imshow(image[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    for rles, color in ((annotation_rles, ANNOTATION_COLOR), (prediction_rles, PREDICTION_COLOR)):
        for rle in rles:
            crop = rle_to_mask(rle, image.shape[0], image.shape[1])[y0:y1, x0:x1]
            if crop.any():
                ax.contour(crop.astype(float), levels=[0.5], colors=[color],
                           linewidths=linewidth)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------

def legend_handles(entries: Dict[str, str]):
    """Line handles for a figure legend: ``{label: colour}``."""
    from matplotlib.lines import Line2D

    return [Line2D([], [], color=color, linewidth=1.5, label=label)
            for label, color in entries.items()]


def plot_distributions(axes, analyses: Dict[str, MatchAnalysis],
                       colors: Optional[Dict[str, str]] = None) -> None:
    """Three panels: IoU histogram, Dice histogram, relation shares.

    ``analyses`` maps a label (e.g. "model", "annotator vs annotator") to its
    :class:`~filament.analysis.MatchAnalysis`. Histograms are the 50-bin
    normalised distributions of all overlapping pairs; the dashed rule at
    IoU 0.5 is the matching threshold of Panoptic Quality. One legend for the
    figure is placed above the panels.
    """
    labels = list(analyses)
    palette = colors or {label: c for label, c in zip(labels, (PREDICTION_COLOR, ANNOTATION_COLOR))}
    for ax, name, title in ((axes[0], "iou", "IoU of overlapping pairs"),
                            (axes[1], "dice", "Dice of overlapping pairs")):
        for label in labels:
            table = iou_dice_histograms(analyses[label])
            ax.stairs(table[f"{name}_fraction"], np.r_[table["bin_lo"], 1.0],
                      color=palette[label], linewidth=1.2,
                      label=f"mean {analyses[label].pairs[name].mean():.2f}")
        if name == "iou":
            ax.axvline(0.5, color=INK_MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("IoU" if name == "iou" else "Dice")
        ax.set_ylabel("share of pairs")
        ax.set_title(title, loc="left", color=INK)
        ax.legend(loc="upper left", handlelength=1.2)

    ax = axes[2]
    x = np.array([0.0, 1.0, 2.0, 3.6, 4.6, 5.6])
    width = 0.8 / max(len(labels), 1)
    for i, label in enumerate(labels):
        shares = relation_counts(analyses[label])["share"].to_numpy()
        ax.bar(x + (i - (len(labels) - 1) / 2) * width, shares, width=width * 0.9,
               color=palette[label])
    ax.set_xticks(x, ["1:0", "1:1", "1:n", "0:1", "1:1", "n:1"])
    for centre, text in ((1.0, "annotated filament:\nmissed / one / split"),
                         (4.6, "prediction:\nmissed / one / merged")):
        ax.text(centre, -0.30, text, ha="center", va="top", color=INK_SECONDARY,
                transform=ax.get_xaxis_transform(), fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share")
    ax.set_title("Relations, hit = any overlap", loc="left", color=INK)

    axes[0].figure.legend(handles=legend_handles({label: palette[label] for label in labels}),
                          loc="outside upper center", ncol=len(labels))
