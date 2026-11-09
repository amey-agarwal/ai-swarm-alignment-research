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


# ============================================================================= report figures
#
# Static, print-first figures for the report (PDF + 300-dpi PNG). Colours come from the
# reference palette: categorical slots in fixed order, assigned to the ENTITY (a claim keeps
# its colour in every figure), the first three slots for anything drawn as overlapping points
# (validated all-pairs), one-hue blue ramp for magnitude, ink tokens for all text. No dual axes;
# two measures become two panels. Every plotted number is also written to a CSV by the report
# builder, which is the table view. Labels are analyst-chosen names, never agent text.

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL_FILL = "#f0efec"
SEQ_BLUE = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
CLAIM_COLOR = {"write": SERIES[0], "irrelevant": SERIES[1], "none": SERIES[2]}
FULL_W, HALF_W = 5.5, 2.7  # inches: NeurIPS text width and half width


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("install the analysis extra: pip install -e '.[analysis]'") from exc
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType fonts
        "axes.edgecolor": BASELINE, "axes.labelcolor": TEXT_SECONDARY, "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "axes.linewidth": 0.8,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    return plt


def _style(ax, *, ygrid: bool = True, xgrid: bool = False) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    if ygrid:
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)
    if xgrid:
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


HEADER_IN = 0.5  # inches reserved above the axes for the title + subtitle


def _title(fig, title: str, subtitle: str | None = None) -> None:
    """Title and subtitle at fixed distances (inches) from the top edge, so they never collide."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.06 / h, title, ha="left", va="top", fontsize=9, color=TEXT_PRIMARY, weight="bold")
    if subtitle:
        fig.text(0.01, 1 - 0.24 / h, subtitle, ha="left", va="top", fontsize=7, color=TEXT_SECONDARY)


def _legend(ax, **kw) -> None:
    leg = ax.legend(frameon=False, handlelength=1.2, borderaxespad=0.2, **kw)
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)


def save(fig, stem: str | Path, *, header_in: float = HEADER_IN) -> list[Path]:
    """Write <stem>.pdf (vector, fonts embedded, no timestamp) and <stem>.png (300 dpi)."""
    plt = _plt()
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 1 - header_in / fig.get_figheight()))
    out = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(out[0], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out[1], dpi=300)
    plt.close(fig)
    return out


def _err(points):
    return [[max(0.0, p[1] - p[2]) for p in points], [max(0.0, p[3] - p[1]) for p in points]]


# ----------------------------------------------------------------------------- diagrams


def setup_diagram(stem: str | Path, *, n_workers: int = 7, rounds: int = 14) -> list[Path]:
    """F1: star swarm, impostor attached to its contact, the round protocol, verifiers outside."""
    import math

    plt = _plt()
    from matplotlib.patches import FancyBboxPatch

    fig, (ax, tl) = plt.subplots(1, 2, figsize=(FULL_W, 2.9), gridspec_kw={"width_ratios": [1, 1.35]})
    for a in (ax, tl):
        a.set_axis_off()
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.25)
    ax.set_aspect("equal")
    pos = [(math.cos(2 * math.pi * i / n_workers + math.pi / 2) * 0.95,
            math.sin(2 * math.pi * i / n_workers + math.pi / 2) * 0.95) for i in range(n_workers)]
    for x, y in pos:
        ax.plot([0, x], [0, y], color=BASELINE, linewidth=1.0, zorder=1)
    ax.scatter([p[0] for p in pos], [p[1] for p in pos], s=110, color=SERIES[0], edgecolors=SURFACE,
               linewidths=2, zorder=3, label="worker (LLM)")
    ax.scatter([0], [0], s=230, color=SERIES[0], edgecolors=TEXT_PRIMARY, linewidths=1.2, zorder=3,
               label="leader (LLM)")
    ix, iy = 1.25, -0.95
    ax.plot([0, ix], [0, iy], color=SERIES[1], linewidth=1.2, zorder=1)
    ax.scatter([ix], [iy], s=130, color=SERIES[1], marker="D", edgecolors=SURFACE, linewidths=2, zorder=3,
               label="impostor (scripted)")
    ax.text(0, -0.22, "leader", ha="center", va="top", fontsize=6, color=TEXT_SECONDARY)
    ax.text(ix, iy - 0.17, "impostor\n(not counted in n)", ha="center", va="top", fontsize=6,
            color=TEXT_SECONDARY)
    ax.set_title(f"Star topology, n = {n_workers + 1}", fontsize=7, color=TEXT_SECONDARY, loc="left")
    _legend(ax, loc="upper left", bbox_to_anchor=(-0.05, 1.02), fontsize=6, markerscale=0.6)

    tl.set_xlim(0, rounds + 0.6)
    tl.set_ylim(-1.5, 3.4)
    phases = [(1, 2, "join: shares a real clue"), (3, 3, "claim: 'write access to\n/shared/kx7_notes.txt'"),
              (4, rounds, "observe; on each request: soft → policy → firm dissuasion")]
    for i, (a, b, label) in enumerate(phases):
        y = 2.6 - i * 0.95
        tl.add_patch(FancyBboxPatch((a - 0.45, y - 0.22), b - a + 0.9, 0.44, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=SERIES[1] if i == 1 else NEUTRAL_FILL, edgecolor="none"))
        tl.text(a - 0.35, y + 0.33, label, fontsize=6, color=TEXT_PRIMARY, va="bottom")
    for r in range(1, rounds + 1):
        tl.text(r, -0.2, str(r), ha="center", va="center", fontsize=6, color=TEXT_SECONDARY)
    tl.set_title("Impostor protocol (rounds)", fontsize=7, color=TEXT_SECONDARY, loc="left")
    tl.text(0.0, -0.75, "Measured outside the swarm: tracer (online, passive), LLM judge\n"
            "(post hoc, different family), comprehension probe. Agents never see it.",
            fontsize=6, color=TEXT_SECONDARY, va="top")
    _title(fig, "Experimental setup")
    return save(fig, stem)


def flow_diagram(stages: Sequence[tuple[str, int]], exclusions: Sequence[tuple[str, int]],
                 stem: str | Path) -> list[Path]:
    """F2: CONSORT-style run flow with counts at each gate and exclusion reasons."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch

    h = 0.55 * len(stages) + 0.6
    fig, ax = plt.subplots(figsize=(FULL_W, max(2.4, h)))
    ax.set_axis_off()
    n = len(stages)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n)
    for i, (label, count) in enumerate(stages):
        y = n - 1 - i
        ax.add_patch(FancyBboxPatch((0.2, y - 0.3), 4.6, 0.6, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=NEUTRAL_FILL if i < n - 1 else SERIES[0], edgecolor="none"))
        ax.text(0.4, y, label, va="center", fontsize=7, color=TEXT_PRIMARY if i < n - 1 else "#ffffff")
        ax.text(4.6, y, f"{count:,}", va="center", ha="right", fontsize=8, weight="bold",
                color=TEXT_PRIMARY if i < n - 1 else "#ffffff")
        if i < n - 1:
            ax.annotate("", xy=(2.5, y - 0.32), xytext=(2.5, y - 0.68),
                        arrowprops={"arrowstyle": "<-", "color": BASELINE, "linewidth": 1.0})
    ax.text(5.4, n - 0.55, "Excluded (reason: runs)", fontsize=7, color=TEXT_SECONDARY, weight="bold")
    for j, (reason, count) in enumerate(list(exclusions)[: max(1, n * 2 - 2)]):
        ax.text(5.4, n - 1.0 - j * 0.42, f"{reason}: {count:,}", fontsize=6.5, color=TEXT_SECONDARY, va="center")
    if not exclusions:
        ax.text(5.4, n - 1.0, "none", fontsize=6.5, color=TEXT_SECONDARY, va="center")
    _title(fig, "Run flow and safety gates",
           "Only runs passing every gate (safety score 1.0; judge κ gate where used) enter the analysis")
    return save(fig, stem)


# ----------------------------------------------------------------------------- data charts


def _point_style(color: str, hollow: bool) -> dict:
    return {"facecolors": SURFACE if hollow else color, "edgecolors": color if hollow else SURFACE,
            "linewidths": 1.6 if hollow else 1.2}


def dot_ci(groups: Sequence[str], series: Sequence[dict], stem: str | Path, *, title: str, subtitle: str,
           xlabel: str, xlim: tuple[float, float] = (0.0, 1.0), width: float = FULL_W * 0.72) -> list[Path]:
    """Horizontal dot plot: one row per group (long labels never collide), 95% CI whiskers,
    series offset within a row, legend above the plot for >= 2 series.
    series: {"name", "color", "hollow": bool, "points": {group: (rate, lo, hi, n)}}"""
    plt = _plt()
    k = len(series)
    fig, ax = plt.subplots(figsize=(width, 0.3 * len(groups) * max(1.0, 0.6 * k) + 1.35))
    span = 0.5
    for j, s in enumerate(series):
        off = (j - (k - 1) / 2) * (span / max(1, k - 1)) if k > 1 else 0.0
        pts = [(gi + off, *s["points"][g][:3]) for gi, g in enumerate(groups) if g in s["points"]]
        if not pts:
            continue
        ys, xs = [p[0] for p in pts], [p[1] for p in pts]
        ax.errorbar(xs, ys, xerr=_err(pts), fmt="none", ecolor=s["color"], elinewidth=1.4, zorder=2)
        ax.scatter(xs, ys, s=34, zorder=3, label=s["name"], **_point_style(s["color"], s.get("hollow", False)))
    ax.set_yticks(range(len(groups)), groups)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    _style(ax, ygrid=False, xgrid=True)
    if k >= 2:
        _legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=min(k, 3))
    _title(fig, title, subtitle)
    return save(fig, stem, header_in=HEADER_IN + (0.2 if k >= 2 else 0.0))


def claim_panels(panels: Sequence[dict], claims: Sequence[str], policies: Sequence[str], stem: str | Path, *,
                 title: str, subtitle: str, xlabel: str) -> list[Path]:
    """F3 layout: one panel per subset (shared rows = claims, coloured by claim as everywhere);
    within a row, policy is encoded by fill (first policy filled, second hollow). Legend uses ink,
    since colour already belongs to the claim. panels: {"title", "points": {(claim, policy): (rate, lo, hi, n)}}"""
    plt = _plt()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_W, 0.42 * len(claims) * max(1, len(policies)) + 1.4),
                             sharey=True, squeeze=False)
    kp = len(policies)
    for ax, p in zip(axes[0], panels):
        for gi, c in enumerate(claims):
            color = CLAIM_COLOR.get(c, MUTED)
            for j, pol in enumerate(policies):
                v = p["points"].get((c, pol))
                if v is None:
                    continue
                y = gi + ((j - (kp - 1) / 2) * (0.36 / max(1, kp - 1)) if kp > 1 else 0.0)
                ax.errorbar([v[0]], [y], xerr=[[max(0.0, v[0] - v[1])], [max(0.0, v[2] - v[0])]], fmt="none",
                            ecolor=color, elinewidth=1.4, zorder=2)
                ax.scatter([v[0]], [y], s=34, zorder=3, **_point_style(color, j % 2 == 1))
        ax.set_yticks(range(len(claims)), [f"claim: {c}" for c in claims])
        ax.set_ylim(len(claims) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(p["title"], fontsize=7, color=TEXT_SECONDARY, loc="left")
        _style(ax, ygrid=False, xgrid=True)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=5.5, color=TEXT_SECONDARY,
                      markerfacecolor=TEXT_SECONDARY if j % 2 == 0 else SURFACE, markeredgewidth=1.4)
               for j in range(kp)]
    leg = fig.legend(handles, [f"policy: {p}" for p in policies], frameon=False, loc="upper right",
                     bbox_to_anchor=(0.995, 1 - 0.04 / fig.get_figheight()), ncol=kp, fontsize=7)
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)
    _title(fig, title, subtitle)
    return save(fig, stem)


def curve_band(series: Sequence[dict], stem: str | Path, *, title: str, subtitle: str, xlabel: str, ylabel: str,
               vlines: Sequence[tuple[float, str]] = (), xticks: Sequence[tuple[float, str]] | None = None,
               ylim: tuple[float, float] = (0.0, 1.0), markers: bool = False) -> list[Path]:
    """Lines with a ~10% CI wash. series: {"name", "color", "x", "mean", "lo", "hi"}"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.7, 2.6))
    for x, label in vlines:
        ax.axvline(x, color=BASELINE, linewidth=0.8, zorder=1)
        ax.text(x, ylim[1], f" {label}", ha="left", va="top", fontsize=6, color=TEXT_SECONDARY)
    for s in series:
        ax.fill_between(s["x"], s["lo"], s["hi"], color=s["color"], alpha=0.10, linewidth=0, zorder=2)
        ax.plot(s["x"], s["mean"], color=s["color"], linewidth=2, zorder=3, label=s["name"],
                marker="o" if markers else None, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    if xticks is not None:
        ax.set_xticks([t[0] for t in xticks], [t[1] for t in xticks])
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style(ax)
    if len(series) >= 2:
        _legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _title(fig, title, subtitle)
    return save(fig, stem)


def forest(panels: Sequence[dict], stem: str | Path, *, title: str, subtitle: str) -> list[Path]:
    """Effect estimates with 95% CIs against a zero line, one panel per measure (never two axes).
    panels: {"title", "xlabel", "rows": [(label, est, lo, hi)]}"""
    plt = _plt()
    nrows = max(len(p["rows"]) for p in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_W, 0.32 * nrows + 1.2), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], panels):
        rows = p["rows"]
        ys = list(range(len(rows)))[::-1]
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8, zorder=1)
        if rows:
            ax.errorbar([r[1] for r in rows], ys, xerr=[[max(0.0, r[1] - r[2]) for r in rows],
                                                         [max(0.0, r[3] - r[1]) for r in rows]],
                        fmt="none", ecolor=SERIES[0], elinewidth=1.4, zorder=2)
            ax.scatter([r[1] for r in rows], ys, s=34, color=SERIES[0], edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.set_yticks(ys, [r[0] for r in rows])
        ax.set_xlabel(p["xlabel"])
        ax.set_title(p["title"], fontsize=7, color=TEXT_SECONDARY, loc="left")
        _style(ax, ygrid=False, xgrid=True)
    _title(fig, title, subtitle)
    return save(fig, stem)


def scaling_facets(facets: Sequence[dict], stem: str | Path, *, title: str, subtitle: str, ylabel: str
                   ) -> list[Path]:
    """Small multiples, one family per panel: dots + CIs vs parameters (log x), logistic fit band
    where identified. facets: {"title", "points": [(params_b, rate, lo, hi, n)], "curve": [(x,p,lo,hi)]|None}"""
    plt = _plt()
    k = max(1, len(facets))
    fig, axes = plt.subplots(1, k, figsize=(min(FULL_W, 1.9 * k + 0.6), 2.5), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], facets):
        if f.get("curve"):
            c = f["curve"]
            ax.fill_between([t[0] for t in c], [t[2] for t in c], [t[3] for t in c], color=SERIES[0], alpha=0.10,
                            linewidth=0)
            ax.plot([t[0] for t in c], [t[1] for t in c], color=SERIES[0], linewidth=2)
        pts = sorted(f["points"])
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=_err([(p[0], p[1], p[2], p[3]) for p in pts]),
                    fmt="none", ecolor=SERIES[0], elinewidth=1.4)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, color=SERIES[0], edgecolors=SURFACE,
                   linewidths=2, zorder=3)
        ax.set_xscale("log")
        xs = [p[0] for p in pts]
        if xs:
            ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
            ax.set_xticks(xs, [f"{x:g}B" for x in xs])
            ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_title(f["title"], fontsize=7, color=TEXT_SECONDARY, loc="left")
        ax.set_xlabel("parameters")
        _style(ax)
    axes[0][0].set_ylabel(ylabel)
    _title(fig, title, subtitle)
    return save(fig, stem)


def scatter_ci(series: Sequence[dict], stem: str | Path, *, title: str, subtitle: str, xlabel: str, ylabel: str
               ) -> list[Path]:
    """Points with CI crosses. series: {"name", "color", "points": [(x, xlo, xhi, y, ylo, yhi)]}"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 2.8))
    for s in series:
        pts = s["points"]
        if not pts:
            continue
        ax.errorbar([p[0] for p in pts], [p[3] for p in pts],
                    xerr=[[max(0.0, p[0] - p[1]) for p in pts], [max(0.0, p[2] - p[0]) for p in pts]],
                    yerr=[[max(0.0, p[3] - p[4]) for p in pts], [max(0.0, p[5] - p[3]) for p in pts]],
                    fmt="none", ecolor=s["color"], elinewidth=0.8, alpha=0.45)
        ax.scatter([p[0] for p in pts], [p[3] for p in pts], s=34, color=s["color"], edgecolors=SURFACE,
                   linewidths=2, zorder=3, label=s["name"])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style(ax, xgrid=True)
    if len(series) >= 2:
        _legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _title(fig, title, subtitle)
    return save(fig, stem)


def ecdf(series: Sequence[dict], stem: str | Path, *, title: str, subtitle: str, xlabel: str, xmax: int
         ) -> list[Path]:
    """Cumulative share of episodes with the event by round (denominator = all episodes in the
    series, so episodes without the event keep the curve below 1). series: {"name","color","values","n"}"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.6, 2.5))
    xs = list(range(0, xmax + 1))
    for s in series:
        ys = [sum(1 for v in s["values"] if v <= x) / s["n"] if s["n"] else 0.0 for x in xs]
        ax.step(xs, ys, where="post", color=s["color"], linewidth=2, label=s["name"])
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("share of episodes")
    _style(ax)
    if len(series) >= 2:
        _legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _title(fig, title, subtitle)
    return save(fig, stem)


def stacked_share(rows: Sequence[tuple[str, dict[str, int]]], cats: Sequence[tuple[str, str]], stem: str | Path,
                  *, title: str, subtitle: str) -> list[Path]:
    """100% horizontal stacked bars with a 2px surface gap; legend always present (>= 2 cats)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.75, 0.4 * len(rows) + 1.3))
    for i, (label, counts) in enumerate(rows):
        total = sum(counts.values()) or 1
        left = 0.0
        for cat, color in cats:
            w = counts.get(cat, 0) / total
            if w > 0:
                ax.barh(i, w, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=1.5,
                        label=cat if i == 0 else None)
            left += w
        ax.text(1.01, i, f"n={total if sum(counts.values()) else 0}", va="center", fontsize=6, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of episodes with a request")
    ax.invert_yaxis()
    _style(ax, ygrid=False, xgrid=True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in cats]
    leg = ax.legend(handles, [c for c, _ in cats], frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
                    ncol=len(cats), fontsize=6)
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)
    _title(fig, title, subtitle)
    return save(fig, stem)


SWIM_COLORS = {-1: SURFACE, 0: NEUTRAL_FILL, 1: SEQ_BLUE[3], 2: SEQ_BLUE[9]}
SWIM_LABELS = {-1: "inactive (removed)", 0: "active, neutral", 1: "references capability", 2: "requests capability"}


def swimlane(agents: Sequence[str], matrix: Sequence[Sequence[int]], stem: str | Path, *, title: str,
             subtitle: str, vlines: Sequence[tuple[float, str]] = ()) -> list[Path]:
    """Agents x rounds raster of ordered states (ordinal blue ramp; inactive = empty cell)."""
    plt = _plt()
    from matplotlib.patches import Rectangle

    rounds = len(matrix[0]) if matrix else 0
    fig, ax = plt.subplots(figsize=(FULL_W, 0.22 * len(agents) + 1.5))
    for i, row in enumerate(matrix):
        for j, s in enumerate(row):
            ax.add_patch(Rectangle((j + 1 - 0.46, i - 0.42), 0.92, 0.84, facecolor=SWIM_COLORS[s],
                                   edgecolor=BASELINE if s == -1 else "none", linewidth=0.4))
    for x, label in vlines:
        ax.axvline(x - 0.5, color=TEXT_SECONDARY, linewidth=0.8)
        ax.text(x - 0.5, -0.9, f" {label}", fontsize=6, color=TEXT_SECONDARY, va="bottom")
    ax.set_xlim(0.4, rounds + 0.6)
    ax.set_ylim(len(agents) - 0.4, -1.0)
    ax.set_yticks(range(len(agents)), agents)
    ax.set_xticks(range(1, rounds + 1))
    ax.set_xlabel("round")
    _style(ax, ygrid=False)
    ax.spines["bottom"].set_visible(False)
    used = sorted({s for row in matrix for s in row})
    handles = [Rectangle((0, 0), 1, 1, facecolor=SWIM_COLORS[s], edgecolor=BASELINE if s == -1 else "none",
                         linewidth=0.4) for s in used]
    leg = ax.legend(handles, [SWIM_LABELS[s] for s in used], frameon=False, loc="upper left",
                    bbox_to_anchor=(1.0, 1.0), fontsize=6)
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)
    _title(fig, title, subtitle)
    return save(fig, stem)


def heatmap(row_labels: Sequence[str], col_labels: Sequence[str], values: Sequence[Sequence[float | None]],
            stem: str | Path, *, title: str, subtitle: str, fmt: str = "{:.2f}", vmax: float | None = None
            ) -> list[Path]:
    """Sequential one-hue heatmap with every cell annotated (cells are few); None = no data."""
    plt = _plt()
    from matplotlib.patches import Rectangle

    flat = [v for row in values for v in row if v is not None]
    top = vmax if vmax is not None else (max(flat) if flat else 1.0) or 1.0
    fig, ax = plt.subplots(figsize=(min(FULL_W, 0.85 * len(col_labels) + 1.8), 0.32 * len(row_labels) + 1.4))
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            if v is None:
                ax.add_patch(Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, facecolor=SURFACE, edgecolor=BASELINE,
                                       linewidth=0.4))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color=MUTED)
                continue
            step = min(len(SEQ_BLUE) - 1, int(round((v / top) * (len(SEQ_BLUE) - 1)))) if top else 0
            color = SEQ_BLUE[step]
            ax.add_patch(Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, facecolor=color, edgecolor="none"))
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if step >= 7 else TEXT_PRIMARY)
    ax.set_xlim(-0.5, len(col_labels) - 0.5)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    _title(fig, title, subtitle)
    return save(fig, stem)


def hbar_panels(panels: Sequence[dict], stem: str | Path, *, title: str, subtitle: str) -> list[Path]:
    """Single-series horizontal bars, one panel per measure. panels: {"title","xlabel","rows":[(label,v)],"fmt"}"""
    plt = _plt()
    nrows = max(len(p["rows"]) for p in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_W, 0.3 * nrows + 1.3), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], panels):
        rows = p["rows"]
        ys = range(len(rows))
        vals = [r[1] for r in rows]
        ax.barh(list(ys), vals, height=0.45, color=SERIES[0], linewidth=0)
        top = max(vals) if vals else 1.0
        for y, v in zip(ys, vals):
            ax.text(v + top * 0.02, y, p.get("fmt", "{:,.0f}").format(v), va="center", fontsize=6,
                    color=TEXT_SECONDARY)
        ax.set_xlim(0, top * 1.25 if top else 1)
        ax.set_yticks(list(ys), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel(p["xlabel"])
        ax.set_title(p["title"], fontsize=7, color=TEXT_SECONDARY, loc="left")
        _style(ax, ygrid=False, xgrid=True)
    _title(fig, title, subtitle)
    return save(fig, stem)


def threshold_dots(rows: Sequence[tuple[str, float, bool]], threshold: float, stem: str | Path, *, title: str,
                   subtitle: str, xlabel: str, xlim: tuple[float, float] = (-0.2, 1.0)) -> list[Path]:
    """One dot per item against a threshold line; gated items filled, report-only items hollow."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.6, 0.3 * len(rows) + 1.3))
    ys = list(range(len(rows)))
    ax.axvline(threshold, color=TEXT_SECONDARY, linewidth=0.8)
    ax.text(threshold, -0.9, f" gate {threshold:g}", fontsize=6, color=TEXT_SECONDARY, va="bottom")
    for y, (label, v, gated) in zip(ys, rows):
        ax.plot([xlim[0], v], [y, y], color=GRIDLINE, linewidth=0.8, zorder=1)
        ax.scatter([v], [y], s=34, zorder=3, facecolors=SERIES[0] if gated else SURFACE, edgecolors=SERIES[0]
                   if not gated else SURFACE, linewidths=1.6 if not gated else 2)
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=6, color=TEXT_SECONDARY)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -1.0)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    _style(ax, ygrid=False, xgrid=True)
    _title(fig, title, subtitle)
    return save(fig, stem)
