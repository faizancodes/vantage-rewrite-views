"""Generate publication figures for the VANTAGE paper.

The script intentionally uses only the final numbers reported in the paper and
the synced Modal artifacts summarized by the analysis scripts. It produces
PDF and PNG copies so the LaTeX source can include vector figures while the
artifacts remain easy to inspect in file browsers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


COLORS = {
    # Light print adaptation of the landing-page component system:
    # white canvas, low-chrome cards, sharp borders, muted hierarchy, one teal signal.
    "bg": "#ffffff",
    "surface_1": "#fafafa",
    "surface_2": "#ffffff",
    "surface_3": "#f4f5f5",
    "border": "#e5e7eb",
    "border_strong": "#d1d5db",
    "text": "#111111",
    "text_secondary": "#4b5563",
    "text_tertiary": "#6b7280",
    "muted": "#9ca3af",
    "disabled": "#c7ccd1",
    "accent": "#00b894",
    "accent_dark": "#008f72",
    "accent_dim": "#00b8941a",
    "warning": "#f59e0b",
    "warning_dim": "#f59e0b24",
    "error": "#ef4444",
    "error_dim": "#ef444424",
    "info": "#3D56F0",
}


def choose_font(candidates: list[str]) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return candidates[-1]


SANS_FONT = choose_font(["Geist Sans", "Geist", "Inter", "DejaVu Sans"])
MONO_FONT = choose_font(["Geist Mono", "SF Mono", "Menlo", "DejaVu Sans Mono"])
LOGO_FONT = choose_font(["Helvetica Neue", "Avenir Next", "Avenir", "Arial", "DejaVu Sans"])

plt.rcParams.update(
    {
        "font.family": SANS_FONT,
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["surface_1"],
        "axes.edgecolor": COLORS["border_strong"],
        "axes.labelcolor": COLORS["text_secondary"],
        "xtick.color": COLORS["text_tertiary"],
        "ytick.color": COLORS["text_tertiary"],
        "text.color": COLORS["text"],
        "savefig.facecolor": COLORS["bg"],
        "savefig.edgecolor": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_panel(ax, xy, wh, *, face=COLORS["surface_2"], edge=COLORS["border"], lw=0.9):
    x, y = xy
    w, h = wh
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.006",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(panel)
    return panel


def draw_card(ax, xy, wh, label, title, body, *, accent=False):
    x, y = xy
    w, h = wh
    draw_panel(
        ax,
        xy,
        wh,
        face=COLORS["surface_2"] if not accent else COLORS["surface_3"],
        edge=COLORS["accent"] if accent else COLORS["border"],
        lw=1.0 if accent else 0.8,
    )
    if accent:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.016, y + 0.026),
                0.005,
                h - 0.052,
                boxstyle="round,pad=0,rounding_size=0.003",
                linewidth=0,
                facecolor=COLORS["accent"],
                alpha=0.95,
            )
        )
    pad_x = 0.036 if accent else 0.026
    ax.text(
        x + pad_x,
        y + h - 0.050,
        label.upper(),
        ha="left",
        va="top",
        fontsize=6.3,
        weight=600,
        color=COLORS["muted"],
    )
    ax.text(
        x + pad_x,
        y + h - 0.092,
        title,
        ha="left",
        va="top",
        fontsize=10.2,
        weight=500,
        color=COLORS["text"],
    )
    ax.text(
        x + pad_x,
        y + h - 0.140,
        body,
        ha="left",
        va="top",
        fontsize=7.1,
        color=COLORS["text_secondary"],
        linespacing=1.12,
    )


def chip(ax, x, y, text, *, active=False, warning=False):
    edge = COLORS["accent"] if active else COLORS["border_strong"]
    face = COLORS["accent_dim"] if active else COLORS["surface_3"]
    text_color = COLORS["accent"] if active else COLORS["text_secondary"]
    if warning:
        edge = COLORS["warning"]
        face = COLORS["warning_dim"]
        text_color = COLORS["warning"]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.075,
            0.045,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            linewidth=0.8,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(x + 0.0375, y + 0.023, text, ha="center", va="center", fontsize=6.7, weight=600, color=text_color)


def arrow(ax, start, end, *, color=COLORS["border_strong"], lw=1.0, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def logo(out_dir: Path) -> None:
    """Generate the compact VANTAGE wordmark used in the paper title."""
    fig, ax = plt.subplots(figsize=(5.45, 0.94))
    ax.set_xlim(-0.03, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # A compact icon for the paper's metaphor:
    # low-light context (pale aperture), exploitable AST structure (node graph),
    # and a vantage-like strike path to a predicted future point.
    ax.add_patch(
        Circle(
            (0.135, 0.52),
            0.155,
            facecolor=COLORS["surface_1"],
            edgecolor=COLORS["border"],
            linewidth=1.0,
            zorder=0,
        )
    )
    ast_nodes = [(0.082, 0.37), (0.105, 0.54), (0.158, 0.57), (0.184, 0.40)]
    ast_edges = [(0, 1), (1, 2), (1, 3)]
    for start, end in ast_edges:
        x0, y0 = ast_nodes[start]
        x1, y1 = ast_nodes[end]
        ax.plot([x0, x1], [y0, y1], color=COLORS["border_strong"], linewidth=1.0, zorder=1)
    for x, y in ast_nodes:
        ax.add_patch(Circle((x, y), 0.010, facecolor=COLORS["bg"], edgecolor=COLORS["border_strong"], linewidth=0.8, zorder=2))

    wing_left = Polygon(
        [(0.070, 0.665), (0.132, 0.595), (0.103, 0.555)],
        closed=True,
        facecolor=COLORS["text"],
        edgecolor="none",
        zorder=3,
    )
    wing_right = Polygon(
        [(0.132, 0.595), (0.225, 0.665), (0.157, 0.555)],
        closed=True,
        facecolor=COLORS["text"],
        edgecolor="none",
        zorder=3,
    )
    body = Polygon(
        [(0.115, 0.602), (0.151, 0.602), (0.139, 0.535), (0.126, 0.535)],
        closed=True,
        facecolor=COLORS["text"],
        edgecolor="none",
        zorder=4,
    )
    ax.add_patch(wing_left)
    ax.add_patch(wing_right)
    ax.add_patch(body)
    ax.add_patch(
        FancyArrowPatch(
            (0.155, 0.560),
            (0.214, 0.435),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.1,
            color=COLORS["accent"],
            connectionstyle="arc3,rad=-0.28",
            zorder=5,
        )
    )
    ax.add_patch(Circle((0.218, 0.425), 0.015, facecolor=COLORS["accent_dim"], edgecolor=COLORS["accent"], linewidth=1.0, zorder=6))
    ax.add_patch(Circle((0.218, 0.425), 0.005, facecolor=COLORS["accent"], edgecolor="none", zorder=7))
    ax.plot([0.256, 0.282], [0.52, 0.52], color=COLORS["border_strong"], linewidth=1.0, solid_capstyle="round")

    ax.text(
        0.318,
        0.55,
        "VANTAGE",
        ha="left",
        va="center",
        fontsize=28,
        weight=300,
        color=COLORS["text"],
        fontfamily=LOGO_FONT,
    )
    save(fig, out_dir, "vantage_logo")


def system_diagram(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.91, "VANTAGE local-structure decode step", fontsize=16.5, weight=300, color=COLORS["text"])
    ax.text(
        0.04,
        0.845,
        "LIVE CODE STATE  ->  LOCAL REUSE  ->  TAIL FALLBACK  ->  LOSSLESS VERIFY",
        fontsize=7.4,
        weight=600,
        color=COLORS["muted"],
    )

    card_w = 0.158
    card_h = 0.255
    card_y = 0.485
    cards = [
        ((0.04, card_y), (card_w, card_h), "input", "Partial code", "incomplete prefix\nat cursor", False),
        ((0.235, card_y), (card_w, card_h), "state", "Live AST", "deepest type\nidentifier/literal", False),
        ((0.43, card_y), (card_w, card_h), "reuse", "Local suffix", "nearby tokens\nsame-file matches", True),
        ((0.625, card_y), (card_w, card_h), "fallback", "Tail W4", "branch at the\nfrontier", True),
        ((0.82, card_y), (card_w, card_h), "check", "Verify", "target accepts\nor rejects", False),
    ]
    for xy, wh, label, title, body, accent in cards:
        draw_card(ax, xy, wh, label, title, body, accent=accent)

    for (xy, wh, *_), (next_xy, *_rest) in zip(cards, cards[1:]):
        arrow(ax, (xy[0] + wh[0], 0.615), (next_xy[0], 0.615), color=COLORS["border_strong"], lw=1.1)

    router_xy = (0.43, 0.145)
    router_wh = (0.345, 0.195)
    draw_panel(ax, router_xy, router_wh, face=COLORS["surface_1"], edge=COLORS["border"])
    ax.text(0.455, 0.300, "CHEAP VERIFIED PROPOSERS", fontsize=6.5, weight=600, color=COLORS["muted"], va="top")
    ax.text(0.455, 0.265, "CPU-only proposals run before neural fallback.", fontsize=8.0, color=COLORS["text_secondary"], va="top")
    chip(ax, 0.455, 0.175, "ID")
    chip(ax, 0.538, 0.175, "LIT")
    chip(ax, 0.621, 0.175, "SUF", active=True)
    chip(ax, 0.704, 0.175, "MAC", warning=True)

    arrow(ax, (0.704, card_y), (0.704, 0.34), color=COLORS["accent"], lw=1.0)
    arrow(ax, (0.82, 0.54), (0.775, 0.31), color=COLORS["border_strong"], lw=0.8, rad=-0.18)

    low_xy = (0.04, 0.145)
    low_wh = (0.315, 0.195)
    draw_panel(ax, low_xy, low_wh, face=COLORS["surface_1"], edge=COLORS["border"])
    ax.text(0.065, 0.300, "BENCHMARK REUSE CONTEXTS", fontsize=6.5, weight=600, color=COLORS["muted"], va="top")
    ax.text(
        0.065,
        0.262,
        "identifiers / literals / tests\nnearby repeated scaffolds",
        fontsize=7.7,
        color=COLORS["text_secondary"],
        linespacing=1.2,
        va="top",
    )
    ax.text(0.065, 0.185, "Verified by the same target rejection rule.", fontsize=7.4, color=COLORS["text_tertiary"], va="top")

    save(fig, out_dir, "vantage_system")


def rewrite_anchor_diagram(out_dir: Path) -> None:
    """Diagram the final PLD-vs-rewrite-anchor mechanism."""
    fig, ax = plt.subplots(figsize=(8.8, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.070, 0.915, "Reference drift: exact lookup misses, rewrite-aware anchoring verifies transformed spans", fontsize=10.0, weight=600, color=COLORS["text"])
    fig.text(0.070, 0.850, "Example edit map: user -> account", fontsize=7.5, color=COLORS["text_secondary"])

    x0, y0, h = 0.045, 0.145, 0.655
    w_context, w_mid, w_right = 0.405, 0.205, 0.245
    gap = 0.045
    x1 = x0 + w_context + gap
    x2 = x1 + w_mid + gap

    def draw_code_segments(x: float, y: float, segments: list[tuple[str, bool]], *, fontsize: float = 8.7) -> None:
        # Axes-coordinate advance tuned for the selected monospace font at this figure size.
        # Drawing the highlighted span as text with a bbox keeps the box glued to the word.
        char_w = 0.01255
        highlighted_char_w = 0.0109
        cursor = x
        for text, highlighted in segments:
            bbox = None
            if highlighted:
                bbox = {
                    "boxstyle": "round,pad=0.09,rounding_size=0.09",
                    "linewidth": 0.9,
                    "edgecolor": COLORS["warning"],
                    "facecolor": COLORS["warning_dim"],
                }
            ax.text(
                cursor,
                y,
                text,
                fontsize=fontsize,
                family=MONO_FONT,
                color=COLORS["text"],
                va="center",
                bbox=bbox,
            )
            cursor += len(text) * (highlighted_char_w if highlighted else char_w)

    # Left: input context.
    draw_panel(ax, (x0, y0), (w_context, h), face=COLORS["surface_2"], edge=COLORS["border"], lw=0.9)
    ax.text(x0 + 0.030, y0 + h - 0.070, "INPUT", fontsize=6.3, weight=700, color=COLORS["muted"], va="center")
    ax.text(x0 + 0.030, y0 + h - 0.145, "reference", fontsize=6.0, weight=700, color=COLORS["text_tertiary"], va="center")
    draw_code_segments(x0 + 0.030, y0 + h - 0.205, [("name = ", False), ("user", True), (".name.strip()", False)])
    ax.text(x0 + 0.030, y0 + h - 0.260, "...  return name", fontsize=8.7, family=MONO_FONT, color=COLORS["text"], va="center")
    ax.plot([x0 + 0.030, x0 + w_context - 0.030], [y0 + h - 0.315, y0 + h - 0.315], color=COLORS["border"], linewidth=0.8)
    ax.text(x0 + 0.030, y0 + h - 0.385, "generated prefix", fontsize=6.0, weight=700, color=COLORS["text_tertiary"], va="center")
    draw_code_segments(x0 + 0.030, y0 + h - 0.445, [("name = ", False), ("account", True), (".name.strip()", False)])

    ax.add_patch(FancyBboxPatch((x0 + 0.030, y0 + 0.055), 0.170, 0.060, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=0, facecolor=COLORS["accent_dim"]))
    ax.text(x0 + 0.045, y0 + 0.085, "user -> account", fontsize=6.8, weight=700, color=COLORS["accent_dark"], va="center")

    # Middle: exact PLD failure. Keep this compact so the teal path can route below it.
    pld_y, pld_h = y0 + 0.285, 0.370
    draw_panel(ax, (x1, pld_y), (w_mid, pld_h), face=COLORS["surface_2"], edge=COLORS["warning"], lw=1.0)
    ax.text(x1 + 0.025, pld_y + pld_h - 0.060, "EXACT PLD", fontsize=6.3, weight=700, color=COLORS["warning"], va="center")
    ax.text(x1 + 0.025, pld_y + pld_h - 0.125, "matches tokens", fontsize=8.2, weight=700, color=COLORS["text"], va="center")
    ax.text(x1 + 0.025, pld_y + pld_h - 0.205, "current suffix", fontsize=5.7, weight=700, color=COLORS["text_tertiary"])
    ax.text(x1 + 0.025, pld_y + pld_h - 0.250, "account.name", fontsize=6.9, family=MONO_FONT, color=COLORS["text_secondary"])
    ax.text(x1 + 0.025, pld_y + pld_h - 0.315, "reference", fontsize=5.7, weight=700, color=COLORS["text_tertiary"])
    ax.text(x1 + 0.025, pld_y + pld_h - 0.360, "user.name", fontsize=6.9, family=MONO_FONT, color=COLORS["text_secondary"])
    ax.text(x1 + w_mid - 0.030, pld_y + pld_h - 0.060, "MISS", fontsize=6.8, weight=800, color=COLORS["warning"], ha="right", va="center")

    # Right: rewrite-aware recovery.
    draw_panel(ax, (x2, y0), (w_right, h), face=COLORS["surface_2"], edge=COLORS["accent"], lw=1.1)
    ax.text(x2 + 0.025, y0 + h - 0.070, "VANTAGE", fontsize=6.3, weight=700, color=COLORS["accent_dark"], va="center")
    ax.text(x2 + 0.025, y0 + h - 0.145, "anchors by rewrite", fontsize=8.0, weight=700, color=COLORS["text"], va="center")

    steps = [
        ("1", "inverse map", "account -> user"),
        ("2", "align", "prefix to reference"),
        ("3", "transform", "user -> account"),
        ("4", "verify", "target argmax"),
    ]
    for idx, (num, title, body) in enumerate(steps):
        row = y0 + h - 0.235 - idx * 0.108
        ax.add_patch(Circle((x2 + 0.040, row), 0.0135, facecolor=COLORS["accent_dim"], edgecolor=COLORS["accent"], linewidth=0.9))
        ax.text(x2 + 0.040, row, num, ha="center", va="center", fontsize=5.7, weight=800, color=COLORS["accent_dark"])
        ax.text(x2 + 0.066, row + 0.017, title, ha="left", va="center", fontsize=6.4, weight=800, color=COLORS["text"])
        ax.text(x2 + 0.066, row - 0.018, body, ha="left", va="center", fontsize=5.8, color=COLORS["text_secondary"])
        if idx < len(steps) - 1:
            arrow(ax, (x2 + 0.040, row - 0.018), (x2 + 0.040, row - 0.073), color=COLORS["accent"], lw=0.7)

    # Flow arrows.
    arrow(ax, (x0 + w_context, y0 + 0.435), (x1, pld_y + 0.185), color=COLORS["warning"], lw=1.0)
    teal_y = y0 + 0.235
    ax.plot([x0 + w_context, x2 - 0.025], [teal_y, teal_y], color=COLORS["accent"], linewidth=1.1)
    arrow(ax, (x2 - 0.025, teal_y), (x2, teal_y), color=COLORS["accent"], lw=1.1)
    save(fig, out_dir, "rewrite_anchor_diagram")


def local_reuse_plot(out_dir: Path) -> None:
    benchmarks = ["HumanEval-Py", "MBPP", "HumanEval-TS"]
    series = [
        ("EAGLE k2", [1.149, 1.169, 1.117], COLORS["muted"]),
        ("Tree W4", [1.210, 1.230, 1.186], COLORS["info"]),
        ("Suffix", [1.722, 1.407, 1.282], COLORS["accent"]),
        ("CodeTail W4", [1.681, 1.440, 1.350], COLORS["warning"]),
    ]
    x = list(range(len(benchmarks)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    fig.subplots_adjust(top=0.74, left=0.11, right=0.97, bottom=0.18)
    ax.set_facecolor(COLORS["surface_1"])
    for idx, (label, values, color) in enumerate(series):
        offsets = [v + (idx - 1.5) * width for v in x]
        bars = ax.bar(offsets, values, width=width, label=label, color=color, alpha=0.88)
        if label in {"Suffix", "CodeTail W4"}:
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.026,
                    f"{value:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    color=COLORS["text_secondary"],
                    family=MONO_FONT,
                )

    fig.text(0.11, 0.94, "LOCAL STRUCTURE REUSE", fontsize=7.2, weight=600, color=COLORS["muted"])
    fig.text(0.11, 0.855, "Same-file suffix reuse changes the headline", fontsize=16, weight=300, color=COLORS["text"])
    fig.text(
        0.11,
        0.795,
        "Speedup vs vanilla from the valid v2 headline runs; all candidates remain target-verified.",
        fontsize=8.6,
        color=COLORS["text_secondary"],
    )
    ax.axhline(1.0, color=COLORS["border_strong"], linewidth=0.9)
    ax.set_ylim(0.95, 1.86)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=8.8)
    ax.set_ylabel("speedup vs vanilla", fontsize=8.8, color=COLORS["text_secondary"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["border_strong"])
    ax.tick_params(axis="y", labelsize=8.4, length=3, color=COLORS["border_strong"])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=COLORS["border"], linewidth=0.65, alpha=0.85)
    ax.grid(axis="x", visible=False)
    legend = fig.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        ncol=4,
        loc="upper left",
        bbox_to_anchor=(0.11, 0.725),
        fontsize=7.8,
        labelcolor=COLORS["text_secondary"],
    )
    for text in legend.get_texts():
        text.set_color(COLORS["text_secondary"])
    save(fig, out_dir, "local_reuse_speedups")


def drift_crossover_plot(out_dir: Path) -> None:
    """Plot the controlled rename-drift crossover against BlazEdit-style PLD."""
    # From vantage_phase2_renamepct_full_v1. The x-axis uses realized
    # token-level drift, not the requested rename percentage bucket.
    buckets = ["0%", "10%", "25%", "50%", "100%"]
    realized_drift = [0.0, 6.6, 11.7, 14.2, 14.4]
    ratios = [0.544, 2.199, 1.268, 1.156, 1.160]
    copy_share = [100.0, 93.4, 88.3, 85.8, 85.6]

    fig, ax = plt.subplots(figsize=(7.7, 4.05))
    fig.subplots_adjust(top=0.79, left=0.12, right=0.97, bottom=0.18)
    ax.set_facecolor(COLORS["surface_1"])

    ax.axhspan(1.0, 2.45, facecolor=COLORS["accent_dim"], alpha=0.45, zorder=0)
    ax.axhspan(0.0, 1.0, facecolor=COLORS["warning_dim"], alpha=0.38, zorder=0)
    ax.axhline(1.0, color=COLORS["text_secondary"], linewidth=1.1, linestyle=(0, (4, 3)))
    ax.text(14.85, 1.03, "parity with PLD", ha="right", va="bottom", fontsize=7.6, color=COLORS["text_secondary"])

    ax.plot(realized_drift, ratios, color=COLORS["accent_dark"], linewidth=1.8, marker="o", markersize=5.2, zorder=3)
    ax.scatter([realized_drift[0]], [ratios[0]], color=COLORS["warning"], edgecolor=COLORS["bg"], linewidth=0.8, s=42, zorder=4)
    ax.scatter(realized_drift[1:], ratios[1:], color=COLORS["accent"], edgecolor=COLORS["bg"], linewidth=0.8, s=42, zorder=4)

    label_offsets = {
        "0%": (-2, 18, "center"),
        "10%": (0, -38, "center"),
        "25%": (-24, 22, "right"),
        "50%": (-34, 28, "right"),
        "100%": (38, 6, "left"),
    }
    for x, y, bucket in zip(realized_drift, ratios, buckets):
        dx, dy, ha = label_offsets[bucket]
        ax.annotate(
            f"{bucket}\n{y:.2f}x",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7.3,
            color=COLORS["text_secondary"],
            family=MONO_FONT,
            linespacing=1.05,
            arrowprops=dict(arrowstyle="-", color=COLORS["border_strong"], linewidth=0.6, shrinkA=3, shrinkB=5)
            if bucket in {"50%", "100%"}
            else None,
        )

    fig.text(0.12, 0.94, "REFERENCE-DRIFT CROSSOVER", fontsize=7.2, weight=600, color=COLORS["muted"])
    fig.text(0.12, 0.855, "VANTAGE crosses over once exact recurrence breaks", fontsize=15.6, weight=300, color=COLORS["text"])
    fig.text(
        0.12,
        0.795,
        "Ratio is rewrite-aware Anchor+PLD over BlazEdit-style PLD-w80; each point summarizes 50 tasks.",
        fontsize=8.4,
        color=COLORS["text_secondary"],
    )
    ax.set_xlim(-0.8, 16.2)
    ax.set_ylim(0.35, 2.50)
    ax.set_xlabel("realized reference drift intensity (% token edits)", fontsize=8.8, color=COLORS["text_secondary"])
    ax.set_ylabel("VANTAGE / PLD throughput", fontsize=8.8, color=COLORS["text_secondary"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["border_strong"])
    ax.tick_params(axis="both", labelsize=8.4, length=3, color=COLORS["border_strong"])
    ax.grid(axis="y", color=COLORS["border"], linewidth=0.65, alpha=0.85)
    ax.grid(axis="x", color=COLORS["border"], linewidth=0.55, alpha=0.55)
    ax.text(
        0.38,
        0.54,
        "verbatim copy:\nPLD wins",
        ha="left",
        va="center",
        fontsize=7.5,
        color=COLORS["warning"],
        weight=600,
    )
    ax.text(
        11.1,
        2.25,
        "drift regime:\nanchor recovers PLD misses",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["accent_dark"],
        weight=600,
    )
    save(fig, out_dir, "drift_crossover")


def frontier_plot(out_dir: Path) -> None:
    depths = [1, 2, 3, 4]
    survival = [1.00, 0.67, 0.225, 0.15]
    labels = ["guaranteed", "frontier", "collapse", "unprofitable"]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.subplots_adjust(top=0.82, left=0.12, right=0.97, bottom=0.16)
    ax.set_facecolor(COLORS["surface_1"])
    bar_colors = [COLORS["surface_3"], COLORS["accent"], COLORS["warning_dim"], COLORS["error_dim"]]
    edge_colors = [COLORS["border_strong"], COLORS["accent"], COLORS["warning"], COLORS["error"]]
    bars = ax.bar(depths, survival, width=0.58, color=bar_colors, edgecolor=edge_colors, linewidth=0.9)
    ax.plot(depths, survival, color=COLORS["text_secondary"], linewidth=1.25, marker="o", markersize=4.5)
    ax.axhline(0.50, color=COLORS["muted"], linestyle=(0, (4, 3)), linewidth=1.0)
    ax.text(4.33, 0.515, "threshold", va="bottom", ha="right", fontsize=7.5, color=COLORS["muted"], weight=600)
    for bar, label, value in zip(bars, labels, survival):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.038,
            label.upper(),
            ha="center",
            fontsize=6.8,
            weight=600,
            color=COLORS["accent"] if label == "frontier" else COLORS["text_tertiary"],
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.055, value - 0.085),
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=9.0,
            color=COLORS["text"] if value > 0.35 else COLORS["text_secondary"],
            family=MONO_FONT,
        )
    ax.text(0.0, 1.18, "FRONTIER SURVIVAL", transform=ax.transAxes, fontsize=7.2, weight=600, color=COLORS["muted"])
    ax.text(0.0, 1.08, "Survival collapses after depth 2", transform=ax.transAxes, fontsize=16, weight=300, color=COLORS["text"])
    ax.text(
        0.0,
        1.005,
        "Conditional draft survival in final diagnostics; depth 2 is the useful uncertainty frontier.",
        transform=ax.transAxes,
        fontsize=8.6,
        color=COLORS["text_secondary"],
    )
    ax.set_ylim(0, 1.12)
    ax.set_xlim(0.45, 4.55)
    ax.set_xticks(depths)
    ax.set_xlabel("draft depth", fontsize=8.8, color=COLORS["text_secondary"])
    ax.set_ylabel("survival probability", fontsize=8.8, color=COLORS["text_secondary"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["border_strong"])
    ax.tick_params(axis="both", labelsize=8.6, length=3, color=COLORS["border_strong"])
    ax.grid(axis="y", color=COLORS["border"], linewidth=0.65, alpha=0.85)
    ax.grid(axis="x", visible=False)
    save(fig, out_dir, "frontier_survival")


def overhead_plot(out_dir: Path) -> None:
    methods = ["Tail", "Router-v2\nno retrieval", "Router-v2\n+ retrieval", "Full\nretrieval/scope"]
    confidence = [0.0, 4.803, 4.803, 4.565]
    retrieval = [0.0, 0.0, 1.023, 0.861]
    scope_route = [0.0, 0.014, 0.014, 0.138]

    fig, ax = plt.subplots(figsize=(7.8, 4.25))
    fig.subplots_adjust(top=0.67, left=0.26, right=0.97, bottom=0.15)
    ax.set_facecolor(COLORS["surface_1"])
    y = list(reversed(range(len(methods))))
    ax.barh(y, confidence, label="confidence", color=COLORS["accent"], alpha=0.82, height=0.46)
    ax.barh(y, retrieval, left=confidence, label="retrieval", color=COLORS["warning"], alpha=0.85, height=0.46)
    lefts = [a + b for a, b in zip(confidence, retrieval)]
    ax.barh(y, scope_route, left=lefts, label="scope + route", color=COLORS["surface_3"], edgecolor=COLORS["border_strong"], height=0.46)

    totals = [a + b + c for a, b, c in zip(confidence, retrieval, scope_route)]
    for yi, total in zip(y, totals):
        ax.text(total + 0.12, yi, f"{total:.1f} ms", ha="left", va="center", fontsize=8.8, color=COLORS["text"], family=MONO_FONT)
    fig.text(0.26, 0.935, "ROUTER OVERHEAD", fontsize=7.2, weight=600, color=COLORS["muted"])
    fig.text(0.26, 0.845, "Signals, not routing, dominate cost", fontsize=16, weight=300, color=COLORS["text"])
    fig.text(
        0.26,
        0.785,
        "Tail has no online signal pass; confidence and retrieval accounting consume milliseconds per step.",
        fontsize=8.6,
        color=COLORS["text_secondary"],
    )
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=8.8, color=COLORS["text_secondary"])
    ax.set_xlabel("extra signal cost per step (ms)", fontsize=8.8, color=COLORS["text_secondary"])
    ax.set_xlim(0, 6.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["border_strong"])
    ax.tick_params(axis="x", labelsize=8.4, length=3, color=COLORS["border_strong"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=COLORS["border"], linewidth=0.65, alpha=0.85)
    ax.grid(axis="y", visible=False)
    legend = fig.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.26, 0.735),
        fontsize=7.8,
        labelcolor=COLORS["text_secondary"],
    )
    for text in legend.get_texts():
        text.set_color(COLORS["text_secondary"])
    save(fig, out_dir, "router_overhead")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    logo(out_dir)
    system_diagram(out_dir)
    rewrite_anchor_diagram(out_dir)
    local_reuse_plot(out_dir)
    drift_crossover_plot(out_dir)
    frontier_plot(out_dir)
    overhead_plot(out_dir)
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
