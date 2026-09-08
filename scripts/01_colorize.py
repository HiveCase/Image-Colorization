#!/usr/bin/env python
"""
01_colorize.py — deploy the model and colorize image(s).

Examples
--------
  # real DDColor model (after running setup_ddcolor.sh)
  python scripts/01_colorize.py --input data/raw/photo.jpg --out outputs/colorized

  # whole folder
  python scripts/01_colorize.py --input data/raw --out outputs/colorized

  # smoke test without any weights
  python scripts/01_colorize.py --input data/raw --out outputs/colorized --mock
"""
import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colorviz.colorizer import build_colorizer  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def gather(path):
    if os.path.isdir(path):
        files = []
        for e in IMG_EXT:
            files += glob.glob(os.path.join(path, f"*{e}"))
            files += glob.glob(os.path.join(path, f"*{e.upper()}"))
        return sorted(files)
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out", default="outputs/colorized")
    ap.add_argument("--model", default="piddnad/ddcolor_modelscope")
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--mock", action="store_true",
                    help="use dependency-free mock colorizer (testing only)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    colorizer = build_colorizer(
        mock=args.mock, model_name=args.model, input_size=args.input_size
    )
    print(f"[colorviz] backend = {colorizer.name}")

    files = gather(args.input)
    if not files:
        print(f"No images found under {args.input}")
        return

    for f in files:
        grey = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if grey is None:
            print(f"  skip (unreadable): {f}")
            continue
        rgb = colorizer.colorize(grey)
        name = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(args.out, f"{name}_color.png")
        cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"  {f} -> {out_path}")


if __name__ == "__main__":
    main()
