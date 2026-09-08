"""
metrics.py
==========
Pixel-level measurements used to detect the failure modes defined in
``taxonomy.py``.

Two families of metrics live here:

* **No-reference** metrics, computable from the colorized output alone
  (colourfulness, mean chroma, edge colour-bleeding, region inconsistency,
  halo energy, implausible-hue flags). These are what you rely on in the wild,
  where no colour ground-truth exists.

* **Full-reference** metrics, computable only when a colour ground-truth is
  available (PSNR, SSIM, CIEDE2000, ab-MSE, colourfulness gap). We obtain a
  ground-truth "for free" by taking a colour photo, converting it to grey,
  colorizing it, and comparing against the original.

All images are expected as HxWx3 uint8 RGB unless stated otherwise.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise ImportError("opencv-python is required for metrics.py") from e

from skimage import color as skcolor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.segmentation import slic


# --------------------------------------------------------------------------- #
# Colour-space helpers
# --------------------------------------------------------------------------- #
def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float Lab (L in [0,100], a,b roughly [-128,127])."""
    return skcolor.rgb2lab(img_rgb.astype(np.float32) / 255.0)


def chroma_map(img_rgb: np.ndarray) -> np.ndarray:
    """Per-pixel chroma = sqrt(a^2 + b^2) from Lab."""
    lab = rgb_to_lab(img_rgb)
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


# --------------------------------------------------------------------------- #
# No-reference metrics
# --------------------------------------------------------------------------- #
def colorfulness(img_rgb: np.ndarray) -> float:
    """
    Hasler & Suesstrunk (2003) colourfulness metric.
    Higher = more colourful. Natural photos usually sit in ~15..80.
    """
    R, G, B = (img_rgb[..., i].astype(np.float32) for i in range(3))
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def mean_chroma(img_rgb: np.ndarray) -> float:
    return float(chroma_map(img_rgb).mean())


def chroma_percentile(img_rgb: np.ndarray, q: float = 99.0) -> float:
    return float(np.percentile(chroma_map(img_rgb), q))


def global_cast_strength(img_rgb: np.ndarray) -> float:
    """
    Measures how much a single hue dominates. We look at the mean (a, b) vector:
    a large mean-chroma relative to the spread of hues means the whole image is
    pushed toward one colour direction (a cast).

    Returns a value in ~[0, 1] where higher = stronger cast.
    """
    lab = rgb_to_lab(img_rgb)
    a, b = lab[..., 1].ravel(), lab[..., 2].ravel()
    mean_vec = np.array([a.mean(), b.mean()])
    mean_len = np.linalg.norm(mean_vec)
    spread = np.sqrt(a.var() + b.var()) + 1e-6
    return float(mean_len / (mean_len + spread))


def edge_bleeding(img_rgb: np.ndarray, l_channel: Optional[np.ndarray] = None) -> float:
    """
    Colour bleeding across boundaries.

    Idea: real objects change colour *at* luminance edges, but a good colorizer
    keeps chroma piecewise-smooth and aligned to those edges. Bleeding shows up
    as chroma *gradients that are strong but NOT aligned with luminance edges*
    (colour drifting across a flat-luminance area) OR as chroma smeared across a
    luminance edge.

    We compute the mean chroma-gradient magnitude in a thin band around
    luminance edges, normalised by overall chroma-gradient energy. High values
    mean chroma structure concentrates on edges in an unstable way.
    """
    lab = rgb_to_lab(img_rgb)
    L = lab[..., 0] if l_channel is None else l_channel.astype(np.float32)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)

    # luminance edges
    lgx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    lgy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    lmag = np.sqrt(lgx ** 2 + lgy ** 2)
    edge_mask = lmag > np.percentile(lmag, 90)

    # dilate to a band straddling the edge
    band = cv2.dilate(edge_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    cgx = cv2.Sobel(chroma, cv2.CV_32F, 1, 0, ksize=3)
    cgy = cv2.Sobel(chroma, cv2.CV_32F, 0, 1, ksize=3)
    cmag = np.sqrt(cgx ** 2 + cgy ** 2)

    on_edge = cmag[band].mean() if band.any() else 0.0
    off_edge = cmag[~band].mean() + 1e-6
    return float(on_edge / off_edge)


def region_inconsistency(img_rgb: np.ndarray, n_segments: int = 120) -> float:
    """
    Average within-superpixel chroma variance.

    We over-segment on luminance so segments respect structure, then measure how
    much (a, b) varies inside each segment. A coherent object should have low
    internal colour variance; patchiness inflates this number.
    """
    lab = rgb_to_lab(img_rgb)
    L = lab[..., 0]
    # SLIC on the luminance replicated to 3 channels (structure-driven segments)
    seg = slic(
        np.repeat((L / 100.0)[..., None], 3, axis=2),
        n_segments=n_segments,
        compactness=10,
        start_label=0,
        channel_axis=2,
    )
    ab = lab[..., 1:]
    variances = []
    for s in np.unique(seg):
        mask = seg == s
        if mask.sum() < 20:
            continue
        variances.append(ab[mask].var(axis=0).sum())
    return float(np.mean(variances)) if variances else 0.0


def halo_energy(img_rgb: np.ndarray) -> float:
    """
    High-frequency chroma energy near strong edges, a proxy for ringing/halo
    artefacts introduced by upsampling a coarse ab map.
    """
    lab = rgb_to_lab(img_rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    lmag = np.abs(cv2.Laplacian(L, cv2.CV_32F))
    edge_mask = lmag > np.percentile(lmag, 95)
    band = cv2.dilate(edge_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ha = np.abs(cv2.Laplacian(a, cv2.CV_32F))
    hb = np.abs(cv2.Laplacian(b, cv2.CV_32F))
    hf = ha + hb
    return float(hf[band].mean()) if band.any() else 0.0


# Rough plausibility gamut for a few common object cues. We do NOT run semantic
# segmentation here (keeps the project dependency-light); instead we flag hues
# that are globally rare in natural photographs — strong green/magenta chroma at
# mid luminance, which most commonly indicates skin/foliage/sky failures.
def implausible_hue_fraction(img_rgb: np.ndarray) -> float:
    """
    Fraction of moderately-to-strongly chromatic pixels whose hue falls in the
    'rarely natural' band. This is deliberately conservative and content-free;
    it catches gross errors (e.g. large green-skin or magenta-sky regions).
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[..., 0] * 2.0            # OpenCV hue is 0..179 -> degrees 0..358
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    chromatic = (s > 0.35) & (v > 0.2) & (v < 0.95)
    if chromatic.sum() == 0:
        return 0.0
    # "magenta/purple-ish" band (300..340 deg) is uncommon in natural scenes at
    # high saturation; treat as a weak implausibility indicator.
    rare = (h >= 295) & (h <= 345)
    return float((rare & chromatic).sum() / chromatic.sum())


def no_reference_metrics(img_rgb: np.ndarray) -> Dict[str, float]:
    return {
        "colorfulness": colorfulness(img_rgb),
        "mean_chroma": mean_chroma(img_rgb),
        "chroma_p99": chroma_percentile(img_rgb, 99.0),
        "global_cast": global_cast_strength(img_rgb),
        "edge_bleeding": edge_bleeding(img_rgb),
        "region_inconsistency": region_inconsistency(img_rgb),
        "halo_energy": halo_energy(img_rgb),
        "implausible_hue_frac": implausible_hue_fraction(img_rgb),
    }


# --------------------------------------------------------------------------- #
# Full-reference metrics (need colour ground-truth)
# --------------------------------------------------------------------------- #
def ciede2000(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    """Mean CIEDE2000 colour difference. Lower is better."""
    lab_p = rgb_to_lab(pred_rgb)
    lab_g = rgb_to_lab(gt_rgb)
    de = skcolor.deltaE_ciede2000(lab_g, lab_p)
    return float(np.mean(de))


def ab_mse(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    lab_p = rgb_to_lab(pred_rgb)
    lab_g = rgb_to_lab(gt_rgb)
    return float(np.mean((lab_p[..., 1:] - lab_g[..., 1:]) ** 2))


def reference_metrics(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> Dict[str, float]:
    if pred_rgb.shape != gt_rgb.shape:
        gt_rgb = cv2.resize(gt_rgb, (pred_rgb.shape[1], pred_rgb.shape[0]))
    psnr = float(peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=255))
    ssim = float(
        structural_similarity(gt_rgb, pred_rgb, channel_axis=2, data_range=255)
    )
    return {
        "psnr": psnr,
        "ssim": ssim,
        "ciede2000": ciede2000(pred_rgb, gt_rgb),
        "ab_mse": ab_mse(pred_rgb, gt_rgb),
        "colorfulness_gap": colorfulness(gt_rgb) - colorfulness(pred_rgb),
    }


# --------------------------------------------------------------------------- #
# Stability (compares two predictions, not to GT)
# --------------------------------------------------------------------------- #
def stability_ab_shift(pred_a_rgb: np.ndarray, pred_b_rgb: np.ndarray) -> float:
    """
    Mean absolute ab shift between two colorizations of the *same scene* whose
    inputs differ only by a colour-preserving perturbation. Large shift =
    fragile / unstable colorizer.
    """
    if pred_a_rgb.shape != pred_b_rgb.shape:
        pred_b_rgb = cv2.resize(
            pred_b_rgb, (pred_a_rgb.shape[1], pred_a_rgb.shape[0])
        )
    lab_a = rgb_to_lab(pred_a_rgb)
    lab_b = rgb_to_lab(pred_b_rgb)
    return float(np.mean(np.abs(lab_a[..., 1:] - lab_b[..., 1:])))
