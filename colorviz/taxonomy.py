"""
taxonomy.py
===========
Failure taxonomy for automatic image colorization.

The taxonomy is the conceptual backbone of the whole project. Every quantitative
metric produced by ``metrics.py`` is mapped onto one of these failure modes so
that the final report can say *where* the model fails and *in what way*.

The four top-level *stress conditions* come directly from the project brief:

    1. Out-of-domain (OOD) inputs
    2. Complex lighting
    3. Fine textures
    4. Multi-object semantic boundaries

Each stress condition is expressed through one or more concrete, observable
*failure modes*. A failure mode is something we can detect from pixels (with or
without a colour ground-truth), assign a severity to, and count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class StressCondition(str, Enum):
    """Top-level categories of stress from the project specification."""

    OUT_OF_DOMAIN = "out_of_domain"
    COMPLEX_LIGHTING = "complex_lighting"
    FINE_TEXTURE = "fine_texture"
    SEMANTIC_BOUNDARY = "semantic_boundary"
    BASELINE = "baseline"  # unperturbed reference


class FailureMode(str, Enum):
    """
    Observable ways a colorizer output can be wrong.

    These are intentionally *symptom-level* (what you see in the image), not
    architecture-level (what happened inside the network), because symptoms are
    what we can measure and what a reader of the report cares about.
    """

    # --- Global chroma problems -------------------------------------------
    DESATURATION = "desaturation"          # output collapses to grey / sepia
    OVERSATURATION = "oversaturation"      # garish, unrealistic chroma
    GLOBAL_COLOR_CAST = "global_color_cast"  # whole image tinted one hue

    # --- Spatial / structural problems ------------------------------------
    COLOR_BLEEDING = "color_bleeding"      # colour crosses object edges
    REGION_INCONSISTENCY = "region_inconsistency"  # one object, many colours
    HALO = "halo"                          # chroma ring around high-contrast edges

    # --- Semantic problems ------------------------------------------------
    IMPLAUSIBLE_HUE = "implausible_hue"    # green skin, purple grass, blue foliage
    MULTIMODAL_AVERAGING = "multimodal_averaging"  # brownish "safe" average colour

    # --- Robustness problems ----------------------------------------------
    INSTABILITY = "instability"            # tiny input change -> large colour change

    # --- Reference-based fidelity (needs colour GT) -----------------------
    LOW_FIDELITY = "low_fidelity"          # far from ground-truth colour (high dE)


# Human-readable descriptions, used verbatim in the generated report.
FAILURE_DESCRIPTIONS: Dict[FailureMode, str] = {
    FailureMode.DESATURATION: (
        "The prediction is nearly achromatic or collapses to a narrow "
        "brown/sepia band. Typical when the model is unsure and hedges toward a "
        "low-risk average colour."
    ),
    FailureMode.OVERSATURATION: (
        "Chroma is pushed far beyond what a natural photo would contain, giving "
        "a cartoonish or neon appearance."
    ),
    FailureMode.GLOBAL_COLOR_CAST: (
        "A single hue dominates the whole frame (e.g. everything tinted teal), "
        "usually triggered by an unusual global luminance statistic."
    ),
    FailureMode.COLOR_BLEEDING: (
        "Colour leaks across object boundaries so that an object's hue spills "
        "onto its neighbour. Measured as chroma variation concentrated on "
        "luminance edges."
    ),
    FailureMode.REGION_INCONSISTENCY: (
        "A single semantic region is painted with several inconsistent colours "
        "(patchiness), instead of one coherent colour."
    ),
    FailureMode.HALO: (
        "A thin ring of anomalous chroma hugs high-contrast edges, a classic "
        "artefact of upsampling a low-resolution ab prediction."
    ),
    FailureMode.IMPLAUSIBLE_HUE: (
        "An object is given a hue that essentially never occurs for that object "
        "class in the real world (green skin, blue foliage, purple grass)."
    ),
    FailureMode.MULTIMODAL_AVERAGING: (
        "When an object could plausibly be many colours, the model averages the "
        "modes and produces a dull, desaturated 'safe' colour."
    ),
    FailureMode.INSTABILITY: (
        "A small, colour-preserving change to the input (mild lighting shift, "
        "light blur) produces a disproportionately large change in the output "
        "colours."
    ),
    FailureMode.LOW_FIDELITY: (
        "With a colour ground-truth available, the predicted colours are far "
        "from the true colours (high CIEDE2000 / ab error)."
    ),
}


# Which failure modes each stress condition is *expected* to provoke. This is a
# prior, used to organise the report and to sanity-check that our stress set is
# actually exercising the intended weaknesses.
CONDITION_TO_EXPECTED_MODES: Dict[StressCondition, List[FailureMode]] = {
    StressCondition.OUT_OF_DOMAIN: [
        FailureMode.DESATURATION,
        FailureMode.GLOBAL_COLOR_CAST,
        FailureMode.IMPLAUSIBLE_HUE,
        FailureMode.MULTIMODAL_AVERAGING,
    ],
    StressCondition.COMPLEX_LIGHTING: [
        FailureMode.GLOBAL_COLOR_CAST,
        FailureMode.DESATURATION,
        FailureMode.INSTABILITY,
        FailureMode.LOW_FIDELITY,
    ],
    StressCondition.FINE_TEXTURE: [
        FailureMode.REGION_INCONSISTENCY,
        FailureMode.HALO,
        FailureMode.INSTABILITY,
    ],
    StressCondition.SEMANTIC_BOUNDARY: [
        FailureMode.COLOR_BLEEDING,
        FailureMode.HALO,
        FailureMode.REGION_INCONSISTENCY,
    ],
    StressCondition.BASELINE: [],
}


class Severity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"

    @staticmethod
    def from_score(score: float) -> "Severity":
        """Map a normalised 0..1 anomaly score to a discrete severity band."""
        if score < 0.25:
            return Severity.NONE
        if score < 0.50:
            return Severity.MINOR
        if score < 0.75:
            return Severity.MODERATE
        return Severity.SEVERE

    @property
    def rank(self) -> int:
        return {"none": 0, "minor": 1, "moderate": 2, "severe": 3}[self.value]


@dataclass
class FailureFinding:
    """A single detected failure for one image under one stress condition."""

    mode: FailureMode
    severity: Severity
    score: float              # raw normalised anomaly score in [0, 1]
    detail: str = ""          # short human-readable justification


@dataclass
class ImageReportCard:
    """All findings for one (image, condition) pair."""

    image_id: str
    condition: StressCondition
    variant: str                      # e.g. "gamma_low", "identity", "blur"
    metrics: Dict[str, float] = field(default_factory=dict)
    findings: List[FailureFinding] = field(default_factory=list)
    output_path: str = ""
    input_path: str = ""

    @property
    def worst_severity(self) -> Severity:
        if not self.findings:
            return Severity.NONE
        return max((f.severity for f in self.findings), key=lambda s: s.rank)

    def modes(self) -> List[FailureMode]:
        return [f.mode for f in self.findings]
