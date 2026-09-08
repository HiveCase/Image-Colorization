"""colorviz — DDColor deployment + colorization failure analysis toolkit."""

__version__ = "1.0.0"

from .colorizer import BaseColorizer, DDColorColorizer, MockColorizer, build_colorizer
from .taxonomy import FailureMode, StressCondition, Severity

__all__ = [
    "BaseColorizer",
    "DDColorColorizer",
    "MockColorizer",
    "build_colorizer",
    "FailureMode",
    "StressCondition",
    "Severity",
]
