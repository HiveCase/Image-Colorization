# Image Colorization Model & Failure Analysis

Deploys a peer-reviewed (ICCV 2023) open-source image colorization model and
runs a systematic **failure analysis** against it: it stress-tests the model,
detects failures, maps them onto a taxonomy, and generates a report describing
*where* the model fails and *in what way*.

This directly implements the project brief:
1. **Model implementation** — deploy an open-source colorization model published
   in a peer-reviewed venue in 2023+.
2. **Failure analysis & taxonomy** — identify, catalog, and stress-test failure
   conditions: out-of-domain inputs, complex lighting, fine textures, and
   multi-object semantic boundaries.
3. **Deliverable** — a report describing the failures.

## Model

**DDColor** — *"DDColor: Towards Photo-Realistic Image Colorization via Dual
Decoders"*, Kang et al., **ICCV 2023** (DAMO Academy, Alibaba Group).
Paper: arXiv:2212.11613 · Code: https://github.com/piddnad/DDColor ·
Weights: HuggingFace `piddnad/ddcolor_modelscope` (Apache-2.0).

Chosen because it is peer-reviewed and recent, fully open-source with pretrained
weights, state-of-the-art on automatic colorization, and cheap to deploy for
inference on a single GPU (or CPU).

## Method in one picture

```
 colour photo ──► to grayscale ──► [16 stress variants] ──► DDColor ──► metrics
                     (ground-truth)   lighting / texture /                 │
                                      OOD / boundary                       ▼
                                                        taxonomy mapping + report
```

We convert colour photos to grayscale to obtain a **free ground-truth**: the
model must recover colour we already know. Each image is expanded into ~16
stress variants (see `colorviz/perturbations.py`), colorized, and measured with
both no-reference metrics (colourfulness, chroma, edge-bleeding, region
inconsistency, halo, implausible-hue) and full-reference metrics (PSNR, SSIM,
CIEDE2000, ab-MSE) plus a stability metric. Measurements are thresholded into
severity-graded **failure findings** organised by the taxonomy in
`colorviz/taxonomy.py`.

## Install

```bash
# 1. install everything + fetch the DDColor architecture and weights path
bash setup_ddcolor.sh
```

If you only want to try the *framework* without the model (no torch/weights):

```bash
pip install numpy opencv-python scikit-image matplotlib pandas PyYAML
```

## Quickstart

```bash
# optional: grab a few sample colour photos (or add your own to data/raw/)
python scripts/download_samples.py

# smoke-test the whole pipeline with a dependency-free mock colorizer
python scripts/02_run_analysis.py --input data/raw --out outputs/run_mock --mock

# the real thing (after setup_ddcolor.sh)
python scripts/02_run_analysis.py --input data/raw --out outputs/run1

# just colorize images
python scripts/01_colorize.py --input data/raw --out outputs/colorized
```

Open `outputs/run1/report/REPORT.html` for the deliverable.

## Outputs

```
outputs/run1/
├── images/            colorized outputs + grey inputs for every stress variant
├── metrics.csv        one row per (image, variant): all metrics + detected modes
└── report/
    ├── REPORT.md      written failure-analysis report (organised by taxonomy)
    ├── REPORT.html    standalone HTML version
    └── figures/       failures-by-condition, severity heatmap, metric boxplots,
                       worst-case gallery
```

## Repository layout

```
colorviz/
├── colorizer.py      DDColor deployment wrapper (+ mock backend for testing)
├── perturbations.py  the 16 stress-test generators, grouped by condition
├── metrics.py        no-reference, full-reference, and stability metrics
├── taxonomy.py       stress conditions, failure modes, severity, report cards
├── analysis.py       orchestration: perturb -> colorize -> measure -> detect
└── report.py         figures + Markdown/HTML report generation
scripts/
├── 01_colorize.py       deploy + colorize image(s)
├── 02_run_analysis.py   full stress suite + report
└── download_samples.py  fetch sample colour photos
configs/default.yaml   model + detection thresholds
data/                  put colour images here (see data/README.md)
setup_ddcolor.sh       installs deps + official DDColor code/weights path
```

## Failure taxonomy (summary)

| Stress condition        | Failure modes it typically provokes                          |
|-------------------------|--------------------------------------------------------------|
| Out-of-domain           | desaturation, global colour cast, implausible hue, mode-averaging |
| Complex lighting        | colour cast, desaturation, instability, low fidelity         |
| Fine texture            | region inconsistency, halos, instability                     |
| Multi-object boundaries | colour bleeding, halos, region inconsistency                 |

Each mode has a pixel-level detector in `metrics.py` and a written definition in
`taxonomy.py` that appears verbatim in the report.

## Notes & honest limitations

- Detection thresholds (`configs/default.yaml`) are sensible defaults, not
  universal truths — calibrate them on your dataset before quoting failure
  rates as absolute.
- The implausible-hue detector is deliberately content-free (no semantic
  segmentation) so the project stays dependency-light; it catches gross errors,
  not subtle ones. Plugging in a segmentation model would sharpen the
  semantic-boundary and implausible-hue analyses.
- Reference metrics are only applied to scene-colour-preserving variants;
  out-of-domain variants use no-reference + stability metrics, since the
  original colour is no longer a valid target.

## Citation

```bibtex
@inproceedings{kang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={328--338},
  year={2023}
}
```
