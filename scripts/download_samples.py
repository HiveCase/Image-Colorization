#!/usr/bin/env python
"""
download_samples.py — fetch a few permissively-licensed colour photos to test on.

These are only convenience samples. For a real study, build a dataset that
deliberately covers the four stress conditions (see data/README.md).

Note: requires network access. If it fails, just drop your own colour JP/PNG
files into data/raw/.
"""
import os
import sys
import urllib.request

# Wikimedia Commons images (public domain / CC). Chosen to include skin, foliage,
# sky, fabric and multi-object scenes — good colorization stress cases.
SAMPLES = {
    "portrait.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/"
        "Gull_portrait_ca_usa.jpg/640px-Gull_portrait_ca_usa.jpg",
    "landscape.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
        "Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg",
    "market.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/"
        "Good_Food_Display_-_NCI_Visuals_Online.jpg/640px-"
        "Good_Food_Display_-_NCI_Visuals_Online.jpg",
}


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "raw")
    os.makedirs(out, exist_ok=True)
    for name, url in SAMPLES.items():
        dst = os.path.join(out, name)
        if os.path.exists(dst):
            print(f"exists: {dst}")
            continue
        try:
            print(f"downloading {name} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "colorviz/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(dst, "wb") as f:
                f.write(r.read())
            print(f"  -> {dst}")
        except Exception as e:
            print(f"  failed ({e}). Add your own images to {out} instead.",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
