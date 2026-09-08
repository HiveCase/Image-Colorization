"""
analysis.py
===========
Ties everything together:

    source colour image
        -> convert to grey  (the honest colorizer input)
        -> for each stress variant: perturb grey -> colorize -> measure
        -> map measurements to FailureFindings (with severity)
        -> collect ImageReportCards

Detection is threshold-based and fully configurable (see configs/default.yaml).
Thresholds are calibrated against the BASELINE ("identity") variant of each
image so that we flag *relative* degradation, which is far more robust than
absolute cut-offs across diverse images.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import metrics as M
from .colorizer import BaseColorizer
from .perturbations import VARIANT_REGISTRY, Variant, synthetic_boundary_chart
from .taxonomy import (
    FailureFinding,
    FailureMode,
    ImageReportCard,
    Severity,
    StressCondition,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict = {
    "input_size": 512,
    "max_side": 768,               # downscale huge images for speed
    "thresholds": {
        # no-reference (absolute)
        "desaturation_mean_chroma": 8.0,     # below this = desaturated
        "oversaturation_chroma_p99": 110.0,  # above this = oversaturated
        "global_cast": 0.55,                 # above this = cast
        "edge_bleeding": 2.2,                # above this = bleeding
        "region_inconsistency": 120.0,       # above this = patchy
        "halo_energy": 4.0,                  # above this = halos
        "implausible_hue_frac": 0.06,        # above this = implausible hue
        # reference (absolute)
        "low_fidelity_ciede2000": 12.0,      # above this = low fidelity
        # stability (relative to baseline colorization)
        "instability_ab_shift": 9.0,         # above this = unstable
    },
    # how far past a threshold counts as "severe" (for score normalisation)
    "severity_span": {
        "default": 1.0,   # 1x threshold over the line == score 1.0
    },
}


def _norm_score(value: float, thresh: float, span_mult: float = 1.0,
                higher_is_worse: bool = True) -> float:
    """
    Map a metric to a 0..1 anomaly score. 0 at the threshold, 1 when the metric
    is (1+span_mult)*threshold worth of distance past it.
    """
    if higher_is_worse:
        excess = value - thresh
    else:
        excess = thresh - value
    if excess <= 0:
        return 0.0
    denom = max(thresh * span_mult, 1e-6)
    return float(min(1.0, excess / denom))


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_failures(
    nr: Dict[str, float],
    ref: Optional[Dict[str, float]],
    stability: Optional[float],
    cfg: Dict,
) -> List[FailureFinding]:
    t = cfg["thresholds"]
    findings: List[FailureFinding] = []

    def add(mode, value, thresh, higher_is_worse=True, detail=""):
        score = _norm_score(value, thresh, 1.0, higher_is_worse)
        sev = Severity.from_score(score)
        if sev != Severity.NONE:
            findings.append(FailureFinding(mode, sev, score, detail))

    # global chroma
    add(FailureMode.DESATURATION, nr["mean_chroma"], t["desaturation_mean_chroma"],
        higher_is_worse=False, detail=f"mean_chroma={nr['mean_chroma']:.1f}")
    add(FailureMode.OVERSATURATION, nr["chroma_p99"], t["oversaturation_chroma_p99"],
        detail=f"chroma_p99={nr['chroma_p99']:.1f}")
    add(FailureMode.GLOBAL_COLOR_CAST, nr["global_cast"], t["global_cast"],
        detail=f"global_cast={nr['global_cast']:.2f}")

    # spatial
    add(FailureMode.COLOR_BLEEDING, nr["edge_bleeding"], t["edge_bleeding"],
        detail=f"edge_bleeding={nr['edge_bleeding']:.2f}")
    add(FailureMode.REGION_INCONSISTENCY, nr["region_inconsistency"],
        t["region_inconsistency"],
        detail=f"region_var={nr['region_inconsistency']:.1f}")
    add(FailureMode.HALO, nr["halo_energy"], t["halo_energy"],
        detail=f"halo_energy={nr['halo_energy']:.2f}")

    # semantic
    add(FailureMode.IMPLAUSIBLE_HUE, nr["implausible_hue_frac"],
        t["implausible_hue_frac"],
        detail=f"implausible_frac={nr['implausible_hue_frac']:.2f}")

    # reference-based fidelity
    if ref is not None:
        add(FailureMode.LOW_FIDELITY, ref["ciede2000"], t["low_fidelity_ciede2000"],
            detail=f"CIEDE2000={ref['ciede2000']:.1f}")

    # stability
    if stability is not None:
        add(FailureMode.INSTABILITY, stability, t["instability_ab_shift"],
            detail=f"ab_shift_vs_baseline={stability:.1f}")

    return findings


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_rgb(path: str, max_side: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# --------------------------------------------------------------------------- #
# Core per-image routine
# --------------------------------------------------------------------------- #
def analyze_image(
    colorizer: BaseColorizer,
    image_path: str,
    variants: List[Variant],
    cfg: Dict,
    out_dir: str,
    has_color_gt: bool = True,
) -> List[ImageReportCard]:
    """
    Run the full stress suite on one source image.

    ``has_color_gt=True`` means ``image_path`` is a colour photo we can convert
    to grey and use as ground-truth (the normal case). Set False for genuinely
    grey inputs (e.g. historical photos) where only no-reference metrics apply.
    """
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    src_rgb = load_rgb(image_path, cfg["max_side"])
    grey = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)
    gt_rgb = src_rgb if has_color_gt else None

    cards: List[ImageReportCard] = []
    baseline_pred: Optional[np.ndarray] = None

    for v in variants:
        perturbed_grey = v.fn(grey)
        pred_rgb = colorizer.colorize(perturbed_grey)

        # save colorized output + the grey input for the report
        out_path = os.path.join(out_dir, f"{image_id}__{v.name}.png")
        cv2.imwrite(out_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
        in_path = os.path.join(out_dir, f"{image_id}__{v.name}__input.png")
        cv2.imwrite(in_path, perturbed_grey)

        nr = M.no_reference_metrics(pred_rgb)

        # reference metrics only valid when scene colour is preserved AND we have GT
        ref = None
        if has_color_gt and v.reference_valid and gt_rgb is not None:
            ref = M.reference_metrics(pred_rgb, gt_rgb)

        # stability vs the identity/baseline colorization
        stability = None
        if v.name == "identity":
            baseline_pred = pred_rgb
        elif baseline_pred is not None and v.reference_valid:
            stability = M.stability_ab_shift(baseline_pred, pred_rgb)

        findings = detect_failures(nr, ref, stability, cfg)

        merged = dict(nr)
        if ref:
            merged.update(ref)
        if stability is not None:
            merged["stability_ab_shift"] = stability

        cards.append(
            ImageReportCard(
                image_id=image_id,
                condition=v.condition,
                variant=v.name,
                metrics=merged,
                findings=findings,
                output_path=out_path,
                input_path=in_path,
            )
        )

    return cards


def analyze_boundary_chart(
    colorizer: BaseColorizer, cfg: Dict, out_dir: str
) -> ImageReportCard:
    """Dedicated synthetic multi-object test for the semantic-boundary condition."""
    os.makedirs(out_dir, exist_ok=True)
    grey = synthetic_boundary_chart(cfg["input_size"])
    pred = colorizer.colorize(grey)
    out_path = os.path.join(out_dir, "boundary_chart__pred.png")
    in_path = os.path.join(out_dir, "boundary_chart__input.png")
    cv2.imwrite(out_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
    cv2.imwrite(in_path, grey)
    nr = M.no_reference_metrics(pred)
    findings = detect_failures(nr, None, None, cfg)
    return ImageReportCard(
        image_id="boundary_chart",
        condition=StressCondition.SEMANTIC_BOUNDARY,
        variant="synthetic_chart",
        metrics=nr,
        findings=findings,
        output_path=out_path,
        input_path=in_path,
    )


def run_suite(
    colorizer: BaseColorizer,
    image_paths: List[str],
    cfg: Dict,
    out_dir: str,
    include_boundary_chart: bool = True,
) -> List[ImageReportCard]:
    variants = list(VARIANT_REGISTRY.values())
    all_cards: List[ImageReportCard] = []
    for p in image_paths:
        all_cards.extend(
            analyze_image(colorizer, p, variants, cfg,
                          os.path.join(out_dir, "images"))
        )
    if include_boundary_chart:
        all_cards.append(
            analyze_boundary_chart(colorizer, cfg, os.path.join(out_dir, "images"))
        )
    return all_cards


def cards_to_records(cards: List[ImageReportCard]) -> List[Dict]:
    """Flatten to rows for CSV / DataFrame export."""
    rows = []
    for c in cards:
        base = {
            "image_id": c.image_id,
            "condition": c.condition.value,
            "variant": c.variant,
            "worst_severity": c.worst_severity.value,
            "n_findings": len(c.findings),
            "failure_modes": "|".join(sorted({f.mode.value for f in c.findings})),
            "output_path": c.output_path,
            "input_path": c.input_path,
        }
        for k, val in c.metrics.items():
            base[f"metric_{k}"] = val
        rows.append(base)
    return rows
