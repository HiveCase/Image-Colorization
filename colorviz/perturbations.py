"""
perturbations.py
================
Stress-test generators. Each generator takes a grayscale L-channel input (the
thing a colorizer actually consumes) and returns a modified L-channel that
exercises a specific failure condition from the project brief.

Design choice
-------------
A colorizer's input is *luminance only*. So a fair, reproducible stress test
perturbs the luminance the model sees, then we colorize and measure what
happened. Perturbations are grouped by the four stress conditions:

    complex_lighting : gamma, exposure, backlight, harsh shadow, highlight clip
    fine_texture     : blur, down-up sampling, sensor noise, JPEG blocking
    out_of_domain    : luminance inversion, posterisation, sketch/edge map,
                       extreme equalisation  (make the *domain* unfamiliar)
    semantic_boundary: handled at the metric level + a synthetic multi-object
                       test chart (real multi-object photos supplied by user)

Every perturbation is *content-preserving at the scene level* except the OOD
ones, which are meant to change the input domain. This matters for which
metrics are valid (see analysis.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import cv2
import numpy as np

from .taxonomy import StressCondition


# An L-channel here is a float32 array in [0, 100] (CIE-Lab L) OR a uint8 grey
# image in [0, 255]; we operate in uint8 grey for simplicity and convert at the
# colorizer boundary. All functions below take & return uint8 HxW grey.

Perturbation = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- #
# Complex lighting
# --------------------------------------------------------------------------- #
def gamma(grey: np.ndarray, g: float) -> np.ndarray:
    x = grey.astype(np.float32) / 255.0
    return np.clip((x ** g) * 255.0, 0, 255).astype(np.uint8)


def exposure_shift(grey: np.ndarray, delta: int) -> np.ndarray:
    return np.clip(grey.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def backlight(grey: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """Darken a central subject as if lit from behind (bright surround)."""
    h, w = grey.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = r / r.max()
    # center dark, edges bright
    mask = 1.0 - strength * (1.0 - r)
    return np.clip(grey.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def harsh_shadow(grey: np.ndarray, drop: float = 0.55) -> np.ndarray:
    """Cast a hard diagonal shadow over part of the frame."""
    h, w = grey.shape
    yy, xx = np.mgrid[0:h, 0:w]
    shadow = (xx + yy) < (0.9 * (h + w) / 2)
    out = grey.astype(np.float32)
    out[shadow] *= (1.0 - drop)
    return np.clip(out, 0, 255).astype(np.uint8)


def clip_highlights(grey: np.ndarray, thresh: int = 200) -> np.ndarray:
    """Blow out highlights, destroying luminance detail in bright areas."""
    out = grey.copy()
    out[out > thresh] = 255
    return out


# --------------------------------------------------------------------------- #
# Fine texture
# --------------------------------------------------------------------------- #
def gaussian_blur(grey: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    k = max(3, int(sigma * 4) | 1)
    return cv2.GaussianBlur(grey, (k, k), sigma)


def down_up(grey: np.ndarray, factor: int = 4) -> np.ndarray:
    """Destroy fine texture by downsampling then upsampling back."""
    h, w = grey.shape
    small = cv2.resize(grey, (max(1, w // factor), max(1, h // factor)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def sensor_noise(grey: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, grey.shape)
    return np.clip(grey.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_blocking(grey: np.ndarray, quality: int = 15) -> np.ndarray:
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode(".jpg", grey, enc)
    if not ok:
        return grey
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


# --------------------------------------------------------------------------- #
# Out-of-domain
# --------------------------------------------------------------------------- #
def invert(grey: np.ndarray) -> np.ndarray:
    """Photographic negative — same structure, alien luminance statistics."""
    return (255 - grey).astype(np.uint8)


def posterize(grey: np.ndarray, levels: int = 4) -> np.ndarray:
    step = 256 // levels
    return ((grey // step) * step).astype(np.uint8)


def sketch(grey: np.ndarray) -> np.ndarray:
    """Edge/line-drawing style input — far from photographic training domain."""
    inv = 255 - grey
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    dodge = cv2.divide(grey, 255 - blur, scale=256)
    return np.clip(dodge, 0, 255).astype(np.uint8)


def extreme_equalize(grey: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(4, 4))
    return clahe.apply(grey)


# --------------------------------------------------------------------------- #
# Semantic boundary — synthetic multi-object chart
# --------------------------------------------------------------------------- #
def synthetic_boundary_chart(size: int = 512) -> np.ndarray:
    """
    A grey test chart with many adjacent, semantically-distinct-looking regions
    sharing hard luminance edges. Colour bleeding and halos are easy to see and
    measure on it. Returned as uint8 grey.
    """
    img = np.zeros((size, size), np.uint8)
    rng = np.random.default_rng(0)
    step = size // 8
    for i in range(8):
        for j in range(8):
            val = int(rng.integers(30, 226))
            img[i * step:(i + 1) * step, j * step:(j + 1) * step] = val
    # add a few fine-textured tiles
    for _ in range(6):
        y = int(rng.integers(0, size - step))
        x = int(rng.integers(0, size - step))
        tex = rng.integers(0, 256, (step, step), dtype=np.uint8)
        img[y:y + step, x:x + step] = tex
    return img


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class Variant:
    name: str
    condition: StressCondition
    fn: Perturbation
    reference_valid: bool  # True if original colour GT still applies afterwards


def build_variant_registry() -> List[Variant]:
    """
    The full stress suite. ``reference_valid`` marks whether comparing to the
    original colour photo is meaningful: lighting/texture keep scene colour, so
    GT metrics apply; OOD transforms change the domain, so we fall back to
    no-reference + stability metrics only.
    """
    V = Variant
    C = StressCondition
    reg: List[Variant] = [
        V("identity", C.BASELINE, lambda g: g, True),

        # complex lighting (scene colour unchanged -> GT valid)
        V("gamma_low", C.COMPLEX_LIGHTING, lambda g: gamma(g, 2.2), True),
        V("gamma_high", C.COMPLEX_LIGHTING, lambda g: gamma(g, 0.45), True),
        V("underexpose", C.COMPLEX_LIGHTING, lambda g: exposure_shift(g, -70), True),
        V("overexpose", C.COMPLEX_LIGHTING, lambda g: exposure_shift(g, 70), True),
        V("backlight", C.COMPLEX_LIGHTING, lambda g: backlight(g, 0.7), True),
        V("harsh_shadow", C.COMPLEX_LIGHTING, lambda g: harsh_shadow(g, 0.55), True),
        V("clip_highlights", C.COMPLEX_LIGHTING, lambda g: clip_highlights(g, 200), True),

        # fine texture (scene colour unchanged -> GT valid)
        V("blur", C.FINE_TEXTURE, lambda g: gaussian_blur(g, 2.5), True),
        V("down_up_4x", C.FINE_TEXTURE, lambda g: down_up(g, 4), True),
        V("noise", C.FINE_TEXTURE, lambda g: sensor_noise(g, 12.0), True),
        V("jpeg_q15", C.FINE_TEXTURE, lambda g: jpeg_blocking(g, 15), True),

        # out of domain (domain changed -> GT NOT valid)
        V("negative", C.OUT_OF_DOMAIN, invert, False),
        V("posterize", C.OUT_OF_DOMAIN, lambda g: posterize(g, 4), False),
        V("sketch", C.OUT_OF_DOMAIN, sketch, False),
        V("clahe_extreme", C.OUT_OF_DOMAIN, extreme_equalize, False),
    ]
    return reg


VARIANT_REGISTRY: Dict[str, Variant] = {
    v.name: v for v in build_variant_registry()
}
