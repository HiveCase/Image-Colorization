"""
Minimal smoke tests — run the framework end to end on tiny synthetic images
using the mock colorizer. No torch or weights required.

    python -m pytest tests/ -q      (or)     python tests/test_smoke.py
"""
import os
import sys
import tempfile

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorviz import metrics as M
from colorviz.analysis import DEFAULT_CONFIG, run_suite, cards_to_records
from colorviz.colorizer import MockColorizer
from colorviz.perturbations import VARIANT_REGISTRY
from colorviz.report import generate_report


def _make_image(path, size=128):
    img = np.zeros((size, size, 3), np.uint8)
    img[: size // 2] = (120, 170, 220)
    img[size // 2:] = (60, 140, 70)
    cv2.circle(img, (size // 2, size // 2), 25, (200, 150, 120), -1)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def test_metrics_ranges():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    nr = M.no_reference_metrics(img)
    for k in ["colorfulness", "mean_chroma", "edge_bleeding",
              "region_inconsistency", "halo_energy"]:
        assert k in nr and np.isfinite(nr[k])


def test_variants_registered():
    assert "identity" in VARIANT_REGISTRY
    assert len(VARIANT_REGISTRY) >= 15


def test_colorizer_shapes():
    grey = (np.random.rand(96, 96) * 255).astype(np.uint8)
    out = MockColorizer().colorize(grey)
    assert out.shape == (96, 96, 3) and out.dtype == np.uint8


def test_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        img_path = os.path.join(d, "t.png")
        _make_image(img_path)
        cards = run_suite(MockColorizer(), [img_path], DEFAULT_CONFIG,
                          os.path.join(d, "out"))
        assert len(cards) >= len(VARIANT_REGISTRY)
        rows = cards_to_records(cards)
        assert rows and "condition" in rows[0]
        paths = generate_report(cards, os.path.join(d, "report"), "mock")
        assert os.path.exists(paths["markdown"])
        assert os.path.exists(paths["html"])
    print("end-to-end OK")


if __name__ == "__main__":
    test_metrics_ranges()
    test_variants_registered()
    test_colorizer_shapes()
    test_pipeline_end_to_end()
    print("ALL SMOKE TESTS PASSED")
