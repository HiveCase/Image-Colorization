"""
colorizer.py
============
Deployment wrapper around the DDColor model (ICCV 2023, DAMO / Alibaba).

Paper : "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders",
        Kang et al., ICCV 2023.  arXiv:2212.11613
Code  : https://github.com/piddnad/DDColor
Weights (HuggingFace Hub):
        piddnad/ddcolor_modelscope   (default, best general model)
        piddnad/ddcolor_paper
        piddnad/ddcolor_artistic
        piddnad/ddcolor_paper_tiny   (fast, low VRAM)

Two colorizer backends are provided behind one interface:

* ``DDColorColorizer`` -- the real model. Requires torch + the DDColor
  architecture (installed by ``setup_ddcolor.sh``, which clones the official
  repo so its ``DDColor`` nn.Module is importable) + the HF weights.

* ``MockColorizer`` -- a dependency-free stand-in that produces a plausible but
  deliberately naive colorization. It exists so the *entire* analysis pipeline
  (perturbations -> metrics -> taxonomy -> report) can be exercised and tested
  on any machine, including CI, without downloading a model. It is NOT part of
  the deliverable analysis; ``--mock`` must be passed explicitly.

The preprocessing in ``DDColorColorizer.colorize`` reproduces the official
DDColor inference recipe: take the original L channel at full resolution,
resize the grey image to the model's input size, run the network to predict the
ab channels, upsample ab back to full resolution, recombine with the original L
and convert to BGR/RGB.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


class BaseColorizer(ABC):
    """Common interface. Input: HxW uint8 grey. Output: HxWx3 uint8 RGB."""

    name: str = "base"

    @abstractmethod
    def colorize(self, grey: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def _as_grey(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)


class DDColorColorizer(BaseColorizer):
    """
    Real DDColor inference. Lazily imports torch and the DDColor architecture so
    that importing this module never forces a heavy dependency.
    """

    name = "ddcolor"

    def __init__(
        self,
        model_name: str = "piddnad/ddcolor_modelscope",
        input_size: int = 512,
        device: Optional[str] = None,
    ):
        import torch  # local import on purpose

        from huggingface_hub import PyTorchModelHubMixin

        # The DDColor nn.Module comes from the official repo, made importable by
        # setup_ddcolor.sh (it installs the package / adds it to PYTHONPATH).
        try:
            from ddcolor_arch import DDColor  # type: ignore
        except Exception:
            # fall back to the path used inside the official repo
            from basicsr.archs.ddcolor_arch import DDColor  # type: ignore

        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)

        self.torch = torch
        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DDColorHF.from_pretrained(model_name).to(self.device).eval()

    @property
    def _F(self):
        import torch.nn.functional as F
        return F

    def colorize(self, grey: np.ndarray) -> np.ndarray:
        torch = self.torch
        F = self._F
        grey = self._as_grey(grey)
        h, w = grey.shape

        # full-res original L (from the grey we were given)
        img_rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        orig_l = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)[:, :, :1]  # (h,w,1)

        # model input: resized grey -> Lab L -> 3-channel grey RGB
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        l_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2Lab)[:, :, :1]
        gray_lab = np.concatenate(
            [l_resized, np.zeros_like(l_resized), np.zeros_like(l_resized)], axis=-1
        )
        gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_Lab2RGB)

        tensor = (
            torch.from_numpy(gray_rgb.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            out_ab = self.model(tensor).cpu()  # (1,2,H,W)

        out_ab = F.interpolate(out_ab, size=(h, w), mode="bilinear",
                               align_corners=False)[0]
        out_ab = out_ab.numpy().transpose(1, 2, 0)  # (h,w,2)

        out_lab = np.concatenate([orig_l, out_ab], axis=-1)
        out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB)
        return (np.clip(out_rgb, 0, 1) * 255).round().astype(np.uint8)


class MockColorizer(BaseColorizer):
    """
    Dependency-free stand-in for testing the pipeline end-to-end.

    Strategy: assign chroma by luminance band with a fixed, hand-made palette and
    a little spatial smoothing. This intentionally exhibits several of the real
    failure modes (desaturation in shadows, implausible hues, cast) so the
    detectors have something to find during a smoke test.
    """

    name = "mock"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def colorize(self, grey: np.ndarray) -> np.ndarray:
        grey = self._as_grey(grey)
        L = grey.astype(np.float32) * (100.0 / 255.0)  # into Lab L range

        # luminance-driven ab: dark->blue-ish, mid->warm, bright->desaturated
        a = np.zeros_like(L)
        b = np.zeros_like(L)
        dark = grey < 80
        mid = (grey >= 80) & (grey < 180)
        bright = grey >= 180
        a[dark], b[dark] = -6, -18
        a[mid], b[mid] = 14, 26
        a[bright], b[bright] = 2, 6

        # smooth the ab so it doesn't perfectly track edges
        a = cv2.GaussianBlur(a, (9, 9), 0)
        b = cv2.GaussianBlur(b, (9, 9), 0)

        lab = np.stack([L, a, b], axis=-1)
        rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
        return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


def build_colorizer(mock: bool = False, **kwargs) -> BaseColorizer:
    """Factory used by the scripts."""
    if mock:
        return MockColorizer()
    return DDColorColorizer(**kwargs)
