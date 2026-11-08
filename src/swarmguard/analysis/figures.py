"""Minimal figures (visualisation proper comes later).

One figure type for the sprint: a rate per condition with 95% Wilson CIs. A single series,
so it uses one categorical colour (slot 1) and no legend: the title names the measure.
Error bars and text use ink colours, never the series colour. Axis labels are condition
names chosen by the analyst, never agent-written text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

SERIES_1 = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def rate_bar(table: Sequence[dict], label_keys: Sequence[str], out: str | Path, *,
             title: str = "Permission-seeking rate", ylabel: str = "Episodes with ≥1 request") -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("install the analysis extra: pip install -e '.[analysis]'") from exc

    rows = [r for r in table if r["n"]]
    labels = ["\n".join(str(r[k]) for k in label_keys) for r in rows]
    rates = [r["rate"] for r in rows]
    # Clamp: at rate 0 or 1 the clamped Wilson bound can differ from the rate by float noise.
    err = [[max(0.0, r["rate"] - r["ci_low"]) for r in rows], [max(0.0, r["ci_high"] - r["rate"]) for r in rows]]

    fig, ax = plt.subplots(figsize=(max(3.2, 0.9 * len(rows) + 1.2), 2.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(range(len(rows)), rates, width=0.6, color=SERIES_1, linewidth=0, zorder=2)  # anchored on the baseline
    ax.errorbar(range(len(rows)), rates, yerr=err, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.2, capsize=3,
                zorder=3)
    for i, r in enumerate(rows):
        ax.annotate(f"n={r['n']}", (i, 0), xytext=(0, -2), textcoords="offset points", ha="center", va="top",
                    fontsize=6, color=TEXT_SECONDARY, annotation_clip=False)
    ax.set_xticks(range(len(rows)), labels, fontsize=7, color=TEXT_PRIMARY)
    ax.tick_params(axis="x", pad=10, length=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel, fontsize=7, color=TEXT_SECONDARY)
    ax.tick_params(axis="y", labelsize=7, colors=TEXT_SECONDARY, length=0)
    ax.set_title(title, fontsize=9, color=TEXT_PRIMARY, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_SECONDARY)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out
