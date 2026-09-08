"""
report.py
=========
Turns a list of ImageReportCards into the project deliverable:

    * a metrics table  (CSV)
    * summary figures   (PNG): failures per stress condition, severity heatmap,
                               metric distributions
    * a qualitative gallery of the worst failures (side-by-side grey/colour)
    * a written report  (Markdown + standalone HTML)

The written report is organised around the taxonomy so it reads as
"where does the model fail, and in what way".
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Dict, List

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .taxonomy import (
    CONDITION_TO_EXPECTED_MODES,
    FAILURE_DESCRIPTIONS,
    FailureMode,
    ImageReportCard,
    Severity,
    StressCondition,
)


SEVERITY_COLOR = {
    "none": "#e8f5e9",
    "minor": "#fff9c4",
    "moderate": "#ffcc80",
    "severe": "#ef9a9a",
}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_failures_by_condition(cards: List[ImageReportCard], path: str) -> None:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for c in cards:
        for f in c.findings:
            counts[c.condition.value][f.mode.value] += 1

    conditions = [c.value for c in StressCondition if c != StressCondition.BASELINE]
    modes = [m.value for m in FailureMode]
    data = np.array([[counts[c][m] for c in conditions] for m in modes])

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(conditions))
    for i, m in enumerate(modes):
        ax.bar(conditions, data[i], bottom=bottom, label=m)
        bottom += data[i]
    ax.set_ylabel("failure findings")
    ax.set_title("Failure findings by stress condition and mode")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_severity_heatmap(cards: List[ImageReportCard], path: str) -> None:
    variants = sorted({c.variant for c in cards})
    images = sorted({c.image_id for c in cards})
    sev_rank = {s.value: s.rank for s in Severity}
    grid = np.zeros((len(images), len(variants)))
    for c in cards:
        i = images.index(c.image_id)
        j = variants.index(c.variant)
        grid[i, j] = sev_rank[c.worst_severity.value]

    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 0.6),
                                    max(4, len(images) * 0.4)))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=3)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(images)))
    ax.set_yticklabels(images, fontsize=7)
    ax.set_title("Worst failure severity per image x variant")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["none", "minor", "moderate", "severe"])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_metric_by_condition(cards: List[ImageReportCard], metric: str,
                            path: str) -> None:
    groups: Dict[str, List[float]] = defaultdict(list)
    for c in cards:
        if metric in c.metrics:
            groups[c.condition.value].append(c.metrics[metric])
    if not groups:
        return
    labels = list(groups.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([groups[k] for k in labels], labels=labels)
    ax.set_title(f"{metric} across stress conditions")
    ax.set_ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def gallery_worst(cards: List[ImageReportCard], out_path: str,
                  top_k: int = 8) -> List[ImageReportCard]:
    """Side-by-side grey input vs colorized output for the worst failures."""
    ranked = sorted(
        cards,
        key=lambda c: (c.worst_severity.rank,
                       sum(f.score for f in c.findings)),
        reverse=True,
    )
    picked = [c for c in ranked if c.findings][:top_k]
    if not picked:
        return []

    n = len(picked)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for row, c in enumerate(picked):
        gin = cv2.imread(c.input_path, cv2.IMREAD_GRAYSCALE)
        out = cv2.cvtColor(cv2.imread(c.output_path), cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(gin, cmap="gray")
        axes[row, 0].set_title(f"{c.image_id} | {c.variant}\ninput", fontsize=8)
        axes[row, 1].imshow(out)
        modes = ", ".join(sorted({f.mode.value for f in c.findings}))
        axes[row, 1].set_title(
            f"{c.worst_severity.value.upper()}\n{modes}", fontsize=8
        )
        for a in axes[row]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return picked


# --------------------------------------------------------------------------- #
# Written report
# --------------------------------------------------------------------------- #
def _summary_stats(cards: List[ImageReportCard]) -> Dict:
    total = len(cards)
    with_failure = sum(1 for c in cards if c.findings)
    mode_counts = Counter()
    cond_fail = Counter()
    cond_total = Counter()
    for c in cards:
        cond_total[c.condition.value] += 1
        if c.findings:
            cond_fail[c.condition.value] += 1
        for f in c.findings:
            mode_counts[f.mode.value] += 1
    return {
        "total": total,
        "with_failure": with_failure,
        "mode_counts": mode_counts,
        "cond_fail": cond_fail,
        "cond_total": cond_total,
    }


def build_markdown(cards: List[ImageReportCard], model_name: str,
                   figures: Dict[str, str], picked: List[ImageReportCard]) -> str:
    s = _summary_stats(cards)
    lines: List[str] = []
    add = lines.append

    add(f"# Failure Analysis Report — {model_name}\n")
    add("## 1. Overview\n")
    add(f"- Total (image, stress-variant) evaluations: **{s['total']}**")
    fail_pct = 100.0 * s["with_failure"] / max(1, s["total"])
    add(f"- Evaluations exhibiting at least one failure: "
        f"**{s['with_failure']} ({fail_pct:.0f}%)**\n")

    add("### Failure rate by stress condition\n")
    add("| Stress condition | Evaluated | With failure | Rate |")
    add("|---|---:|---:|---:|")
    for cond in s["cond_total"]:
        tot = s["cond_total"][cond]
        fl = s["cond_fail"][cond]
        add(f"| {cond} | {tot} | {fl} | {100.0 * fl / max(1, tot):.0f}% |")
    add("")

    add("### Most common failure modes\n")
    add("| Failure mode | Count |")
    add("|---|---:|")
    for mode, cnt in s["mode_counts"].most_common():
        add(f"| {mode} | {cnt} |")
    add("")

    add("## 2. Figures\n")
    for caption, rel in figures.items():
        if caption == "gallery":
            continue  # rendered in section 4
        add(f"**{caption}**\n")
        add(f"![{caption}]({rel})\n")

    add("## 3. Taxonomy — where and how the model fails\n")
    for cond in StressCondition:
        if cond == StressCondition.BASELINE:
            continue
        add(f"### {cond.value}\n")
        expected = CONDITION_TO_EXPECTED_MODES[cond]
        observed = Counter()
        for c in cards:
            if c.condition == cond:
                for f in c.findings:
                    observed[f.mode] += 1
        add(f"- Expected failure modes: "
            f"{', '.join(m.value for m in expected) or '—'}")
        if observed:
            obs_str = ", ".join(f"{m.value} ({n})"
                                for m, n in observed.most_common())
        else:
            obs_str = "none detected"
        add(f"- Observed failure modes: {obs_str}\n")
        for mode, _ in observed.most_common(3):
            add(f"  - **{mode.value}** — {FAILURE_DESCRIPTIONS[mode]}")
        add("")

    add("## 4. Qualitative examples (worst cases)\n")
    if "gallery" in figures:
        add(f"![worst cases]({figures['gallery']})\n")
    for c in picked:
        modes = ", ".join(sorted({f.mode.value for f in c.findings}))
        add(f"- `{c.image_id}` / `{c.variant}` — "
            f"**{c.worst_severity.value}** — {modes}")
    add("")

    add("## 5. Method notes\n")
    add("- Ground-truth is obtained by converting colour photos to grayscale, "
        "colorizing, and comparing to the original. Reference metrics (PSNR, "
        "SSIM, CIEDE2000, ab-MSE) are only applied to scene-colour-preserving "
        "variants (lighting, texture); out-of-domain variants use no-reference "
        "and stability metrics only.")
    add("- Severity is graded relative to detection thresholds "
        "(`configs/default.yaml`); tune these to your dataset before drawing "
        "hard conclusions.")
    add("- The synthetic boundary chart isolates colour-bleeding and halo "
        "behaviour on hard luminance edges.\n")
    return "\n".join(lines)


def markdown_to_html(md_text: str, title: str) -> str:
    """Minimal, dependency-free markdown->HTML (headings, tables, images, lists)."""
    html_lines = []
    in_table = False
    for raw in md_text.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue  # separator row
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            tag = "td"
            html_lines.append("<tr>" + "".join(
                f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            html_lines.append("</table>")
            in_table = False
        if line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("!["):
            alt = line[2:line.index("]")]
            src = line[line.index("(") + 1:line.index(")")]
            html_lines.append(
                f'<img src="{src}" alt="{alt}" style="max-width:100%;">')
        elif line.startswith("- ") or line.startswith("  - "):
            indent = "margin-left:2em;" if line.startswith("  -") else ""
            html_lines.append(f'<div style="{indent}">• {line.strip()[2:]}</div>')
        elif line == "":
            html_lines.append("<br>")
        else:
            html_lines.append(f"<p>{line}</p>")
    if in_table:
        html_lines.append("</table>")

    body = "\n".join(html_lines)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 960px; margin: 2rem auto; padding: 0 1rem; color:#222; }}
 h1 {{ border-bottom: 3px solid #333; padding-bottom:.3rem; }}
 h2 {{ margin-top: 2rem; border-bottom:1px solid #ccc; }}
 table {{ border-collapse: collapse; margin: 1rem 0; }}
 td {{ border: 1px solid #bbb; padding: 4px 10px; }}
 img {{ border:1px solid #ddd; border-radius:6px; margin:.5rem 0; }}
 code {{ background:#f2f2f2; padding:1px 4px; border-radius:3px; }}
</style></head><body>
{body}
</body></html>"""


def generate_report(cards: List[ImageReportCard], out_dir: str,
                    model_name: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    figures: Dict[str, str] = {}

    p = os.path.join(fig_dir, "failures_by_condition.png")
    fig_failures_by_condition(cards, p)
    figures["Failures by stress condition"] = os.path.relpath(p, out_dir)

    p = os.path.join(fig_dir, "severity_heatmap.png")
    fig_severity_heatmap(cards, p)
    figures["Severity heatmap"] = os.path.relpath(p, out_dir)

    for metric in ["mean_chroma", "ciede2000", "edge_bleeding"]:
        p = os.path.join(fig_dir, f"metric_{metric}.png")
        fig_metric_by_condition(cards, metric, p)
        if os.path.exists(p):
            figures[f"Distribution of {metric}"] = os.path.relpath(p, out_dir)

    gpath = os.path.join(fig_dir, "gallery_worst.png")
    picked = gallery_worst(cards, gpath)
    if picked:
        figures["gallery"] = os.path.relpath(gpath, out_dir)

    md = build_markdown(cards, model_name, figures, picked)
    md_path = os.path.join(out_dir, "REPORT.md")
    with open(md_path, "w") as f:
        f.write(md)

    html = markdown_to_html(md, f"Failure Analysis — {model_name}")
    html_path = os.path.join(out_dir, "REPORT.html")
    with open(html_path, "w") as f:
        f.write(html)

    return {"markdown": md_path, "html": html_path, "figures_dir": fig_dir}
