#!/usr/bin/env python
"""
02_run_analysis.py — run the full stress-test suite and build the report.

Pipeline:  images -> perturb -> colorize -> metrics -> taxonomy -> report

Examples
--------
  # full run with the real DDColor model
  python scripts/02_run_analysis.py --input data/raw --out outputs/run1

  # end-to-end smoke test with the mock colorizer (no weights needed)
  python scripts/02_run_analysis.py --input data/raw --out outputs/run_mock --mock

Outputs (under --out):
  images/            colorized outputs + grey inputs for every variant
  metrics.csv        one row per (image, variant) with all metrics + findings
  report/REPORT.md   the written failure-analysis report
  report/REPORT.html standalone HTML version
  report/figures/    summary figures + worst-case gallery
"""
import argparse
import csv
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colorviz.analysis import DEFAULT_CONFIG, cards_to_records, run_suite  # noqa
from colorviz.colorizer import build_colorizer  # noqa: E402
from colorviz.report import generate_report  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def gather(path):
    if os.path.isdir(path):
        files = []
        for e in IMG_EXT:
            files += glob.glob(os.path.join(path, f"*{e}"))
            files += glob.glob(os.path.join(path, f"*{e.upper()}"))
        return sorted(files)
    return [path]


def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        if "analysis" in user:
            cfg["max_side"] = user["analysis"].get("max_side", cfg["max_side"])
        if "model" in user:
            cfg["input_size"] = user["model"].get("input_size", cfg["input_size"])
        if "thresholds" in user:
            cfg["thresholds"].update(user["thresholds"])
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image file or folder (colour photos)")
    ap.add_argument("--out", default="outputs/run1")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="piddnad/ddcolor_modelscope")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-chart", action="store_true",
                    help="skip the synthetic boundary chart")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    colorizer = build_colorizer(
        mock=args.mock, model_name=args.model, input_size=cfg["input_size"]
    )
    model_name = "MockColorizer (smoke test)" if args.mock else args.model
    print(f"[colorviz] backend = {colorizer.name} | model = {model_name}")

    images = gather(args.input)
    if not images:
        print(f"No images found under {args.input}")
        return
    print(f"[colorviz] {len(images)} source image(s)")

    cards = run_suite(
        colorizer, images, cfg, args.out,
        include_boundary_chart=not args.no_chart,
    )

    # metrics.csv
    rows = cards_to_records(cards)
    csv_path = os.path.join(args.out, "metrics.csv")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"[colorviz] wrote {csv_path}")

    # report
    report_paths = generate_report(
        cards, os.path.join(args.out, "report"), model_name
    )
    print(f"[colorviz] report: {report_paths['markdown']}")
    print(f"[colorviz] report: {report_paths['html']}")

    n_fail = sum(1 for c in cards if c.findings)
    print(f"[colorviz] {n_fail}/{len(cards)} evaluations flagged with failures")


if __name__ == "__main__":
    main()
