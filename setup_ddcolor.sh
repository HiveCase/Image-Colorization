#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_ddcolor.sh
# Installs the official DDColor (ICCV 2023) code so its network architecture is
# importable, then installs this project's Python requirements.
#
# The HuggingFace weights (piddnad/ddcolor_modelscope) download automatically on
# first inference; no manual weight download is needed.
#
# Usage:
#   bash setup_ddcolor.sh
# ---------------------------------------------------------------------------
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="${PROJECT_DIR}/third_party"
mkdir -p "${THIRD_PARTY}"

echo "==> Installing Python requirements"
pip install -r "${PROJECT_DIR}/requirements.txt"

echo "==> Cloning official DDColor repo (ICCV 2023, piddnad/DDColor)"
if [ ! -d "${THIRD_PARTY}/DDColor" ]; then
  git clone --depth 1 https://github.com/piddnad/DDColor.git "${THIRD_PARTY}/DDColor"
fi

echo "==> Making the DDColor architecture importable"
# The DDColor nn.Module lives in the repo. We expose it two ways so
# colorizer.py can import it as either `ddcolor_arch` or
# `basicsr.archs.ddcolor_arch`.
cd "${THIRD_PARTY}/DDColor"

# Install BasicSR-based package if the repo ships a setup file.
if [ -f "setup.py" ]; then
  pip install -e . || echo "   (editable install skipped; using PYTHONPATH fallback)"
fi

# Copy the architecture to a top-level importable module name as a fallback.
ARCH_SRC="$(find "${THIRD_PARTY}/DDColor" -name 'ddcolor_arch.py' | head -n1 || true)"
if [ -n "${ARCH_SRC}" ]; then
  cp "${ARCH_SRC}" "${PROJECT_DIR}/ddcolor_arch.py"
  echo "   copied $(basename "${ARCH_SRC}") -> ${PROJECT_DIR}/ddcolor_arch.py"
fi

cat <<EOF

==> Done.

To use the real model, run e.g.:

  python scripts/02_run_analysis.py --input data/raw --out outputs/run1

First run downloads the weights from HuggingFace (piddnad/ddcolor_modelscope).
If you hit an import error for the architecture, prepend the repo to PYTHONPATH:

  export PYTHONPATH="${THIRD_PARTY}/DDColor:\$PYTHONPATH"

EOF
