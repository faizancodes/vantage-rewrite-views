#!/usr/bin/env python3
"""Generate light-theme README charts for the VANTAGE repository."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media"

COLORS = {
    "bg": "#fbfbfb",
    "surface": "#ffffff",
    "surface_2": "#f4f4f5",
    "border": "#d4d4d8",
    "grid": "#e4e4e7",
    "text": "#0a0a0a",
    "secondary": "#52525b",
    "muted": "#71717a",
    "accent": "#00b894",
    "accent_dark": "#008f75",
    "pld": "#a1a1aa",
}

FONT = "Inter, Geist, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
MONO = "'SF Mono', 'Geist Mono', Consolas, Menlo, monospace"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_shell(
    width: int,
    height: int,
    title: str,
    subtitle: str,
    body: str,
    eyebrow: str = "VANTAGE / controlled fp32/sdpa",
) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{esc(title)}</title>
  <desc id="desc">{esc(subtitle)}</desc>
  <rect width="{width}" height="{height}" fill="{COLORS['bg']}"/>
  <rect x="24" y="24" width="{width - 48}" height="{height - 48}" rx="0" fill="{COLORS['surface']}" stroke="{COLORS['border']}"/>
  <text x="48" y="62" font-family="{FONT}" font-size="11" font-weight="700" fill="{COLORS['muted']}" text-transform="uppercase">{esc(eyebrow)}</text>
  <text x="48" y="96" font-family="{FONT}" font-size="28" font-weight="300" fill="{COLORS['text']}">{esc(title)}</text>
  <text x="48" y="122" font-family="{FONT}" font-size="14" fill="{COLORS['secondary']}">{esc(subtitle)}</text>
  {body}
</svg>
"""


def mechanism_diagram() -> str:
    width, height = 1100, 690

    def line_text(
        x: int,
        y: int,
        lines: list[str],
        size: float = 12.5,
        color: str = COLORS["secondary"],
        leading: int = 18,
        weight: int = 400,
    ) -> str:
        return "\n  ".join(
            f'<text x="{x}" y="{y + i * leading}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}">{esc(line)}</text>'
            for i, line in enumerate(lines)
        )

    def code_box(
        x: int,
        y: int,
        w: int,
        lines: list[tuple[str, str | None]],
        bg: str = COLORS["surface_2"],
    ) -> str:
        h = 28 + 26 * len(lines)
        parts = [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{bg}" stroke="{COLORS["grid"]}"/>'
        ]
        for i, (text, color) in enumerate(lines):
            parts.append(
                f'<text x="{x + 16}" y="{y + 30 + i * 26}" font-family="{MONO}" font-size="14.5" font-weight="700" fill="{color or COLORS["text"]}">{esc(text)}</text>'
            )
        return "\n  ".join(parts)

    def mini_arrow(x1: int, y: int, x2: int, color: str = COLORS["accent_dark"]) -> str:
        return f"""
  <line x1="{x1}" y1="{y}" x2="{x2 - 12}" y2="{y}" stroke="{color}" stroke-width="2.4"/>
  <path d="M{x2 - 12} {y - 7} L{x2} {y} L{x2 - 12} {y + 7}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
"""

    def contribution_card(
        x: int,
        y: int,
        w: int,
        title: str,
        lines: list[str],
        num: str,
        accent: bool = False,
    ) -> str:
        stroke = COLORS["accent_dark"] if accent else COLORS["border"]
        parts = [
            f'<rect x="{x}" y="{y}" width="{w}" height="104" rx="4" fill="{COLORS["surface"]}" stroke="{stroke}" stroke-width="{2 if accent else 1}"/>',
            f'<circle cx="{x + 24}" cy="{y + 28}" r="13" fill="{COLORS["accent"]}"/>',
            f'<text x="{x + 24}" y="{y + 33}" font-family="{MONO}" font-size="12" font-weight="700" fill="{COLORS["text"]}" text-anchor="middle">{esc(num)}</text>',
            f'<text x="{x + 46}" y="{y + 26}" font-family="{FONT}" font-size="12" font-weight="700" fill="{COLORS["text"]}">{esc(title)}</text>',
            line_text(x + 46, y + 50, lines, size=11.5, leading=16),
        ]
        return "\n  ".join(parts)

    parts: list[str] = []

    # Left: the failure mode for literal prompt lookup.
    left_x, top_y = 54, 164
    left_w, top_h = 472, 260
    parts.append(
        f'<rect x="{left_x}" y="{top_y}" width="{left_w}" height="{top_h}" rx="4" fill="{COLORS["surface"]}" stroke="{COLORS["border"]}"/>'
    )
    parts.append(
        f'<text x="{left_x + 22}" y="{top_y + 34}" font-family="{FONT}" font-size="11" font-weight="700" fill="{COLORS["muted"]}" text-transform="uppercase">Ordinary PLD sees only literal recurrence</text>'
    )
    parts.append(
        line_text(
            left_x + 22,
            top_y + 66,
            ["The prompt has old code.", "The edit wants account.name instead."],
            size=12.5,
        )
    )
    parts.append(
        code_box(
            left_x + 22,
            top_y + 112,
            198,
            [("reference:", COLORS["muted"]), ("return user.name", COLORS["text"])],
        )
    )
    parts.append(mini_arrow(left_x + 235, top_y + 156, left_x + 285, COLORS["pld"]))
    parts.append(
        code_box(
            left_x + 300,
            top_y + 112,
            146,
            [("PLD draft:", COLORS["muted"]), ("user.name", COLORS["pld"])],
        )
    )
    parts.append(
        f'<rect x="{left_x + 22}" y="{top_y + 207}" width="{left_w - 44}" height="30" rx="4" fill="{COLORS["surface_2"]}" stroke="{COLORS["grid"]}"/>'
    )
    parts.append(
        f'<text x="{left_x + 38}" y="{top_y + 227}" font-family="{FONT}" font-size="12" font-weight="700" fill="{COLORS["secondary"]}">Result: short drafts, because the useful span is no longer literal.</text>'
    )

    # Right: VANTAGE's contribution.
    right_x, right_w = 574, 472
    parts.append(
        f'<rect x="{right_x}" y="{top_y}" width="{right_w}" height="{top_h}" rx="4" fill="{COLORS["surface"]}" stroke="{COLORS["accent_dark"]}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{right_x + 22}" y="{top_y + 34}" font-family="{FONT}" font-size="11" font-weight="700" fill="{COLORS["muted"]}" text-transform="uppercase">VANTAGE adds a hidden rewritten draft source</text>'
    )
    parts.append(
        line_text(
            right_x + 22,
            top_y + 66,
            ["VANTAGE uses the visible rename to build", "a hidden draft that already says account.name."],
            size=12.5,
        )
    )
    parts.append(
        code_box(
            right_x + 22,
            top_y + 112,
            188,
            [("visible map:", COLORS["muted"]), ("user -> account", COLORS["text"])],
        )
    )
    parts.append(mini_arrow(right_x + 224, top_y + 156, right_x + 270))
    parts.append(
        code_box(
            right_x + 284,
            top_y + 112,
            164,
            [("hidden view:", COLORS["muted"]), ("account.name", COLORS["accent_dark"])],
        )
    )
    parts.append(
        f'<rect x="{right_x + 22}" y="{top_y + 207}" width="{right_w - 44}" height="30" rx="4" fill="#ecfdf7" stroke="{COLORS["accent"]}"/>'
    )
    parts.append(
        f'<text x="{right_x + 38}" y="{top_y + 227}" font-family="{FONT}" font-size="12" font-weight="700" fill="{COLORS["accent_dark"]}">Result: longer accepted drafts without changing the visible prompt.</text>'
    )

    # Bottom: paper contributions and validated evidence.
    parts.append(
        f'<text x="54" y="466" font-family="{FONT}" font-size="11" font-weight="700" fill="{COLORS["muted"]}" text-transform="uppercase">Core contributions shown in the paper</text>'
    )
    cards = [
        (
            54,
            "Rewrite-View Lookup",
            ["Builds a hidden reference", "from prompt-visible rewrites."],
            "1",
            False,
        ),
        (
            308,
            "SafeRoute",
            ["Chooses PLD or rewrite-view", "drafts; falls back when needed."],
            "2",
            False,
        ),
        (
            562,
            "Verifier contract",
            ["Drafts are checked by target;", "argmax tokens come from target."],
            "3",
            False,
        ),
        (
            816,
            "Controlled evidence",
            ["1.28x-1.64x vs tuned PLD;", "100/100 greedy parity audited."],
            "4",
            True,
        ),
    ]
    for x, title, lines, num, accent in cards:
        parts.append(contribution_card(x, 488, 230, title, lines, num, accent=accent))

    parts.append(
        f'<rect x="54" y="616" width="992" height="34" rx="4" fill="{COLORS["surface_2"]}" stroke="{COLORS["grid"]}"/>'
    )
    parts.append(
        f'<text x="72" y="638" font-family="{FONT}" font-size="12" fill="{COLORS["secondary"]}">VANTAGE is useful when an explicit edit changes repeated code names: it restores copy opportunities while preserving the target model&apos;s fixed-prompt behavior.</text>'
    )

    return svg_shell(
        width,
        height,
        "Why VANTAGE helps code-edit decoding",
        "When an edit renames code, literal PLD proposes old text; VANTAGE proposes the renamed span for target verification.",
        "\n  ".join(parts),
        eyebrow="VANTAGE / Rewrite-View Lookup",
    )


def speedup_chart() -> str:
    width, height = 1100, 430
    chart_x, chart_y = 278, 176
    chart_w, row_h = 620, 58
    max_v = 1.95
    rows = [
        ("Field substitutions", 1.275, 1.202, 1.358),
        ("Identifier-style substitutions", 1.639, 1.475, 1.825),
        ("Mixed controlled suite", 1.328, 1.207, 1.482),
    ]
    parts: list[str] = []
    # Axis and baseline marker.
    for tick in (1.0, 1.25, 1.5, 1.75):
        x = chart_x + (tick / max_v) * chart_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{chart_y - 18}" x2="{x:.1f}" y2="{chart_y + row_h * len(rows) + 8}" stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{chart_y + row_h * len(rows) + 34}" font-family="{MONO}" font-size="11" fill="{COLORS["muted"]}" text-anchor="middle">{tick:.2g}x</text>'
        )
    parts.append(
        f'<text x="{chart_x + (1.0 / max_v) * chart_w:.1f}" y="{chart_y - 30}" font-family="{FONT}" font-size="11" fill="{COLORS["muted"]}" text-anchor="middle">tuned PLD</text>'
    )
    for i, (label, value, lo, hi) in enumerate(rows):
        y = chart_y + i * row_h
        bar_w = (value / max_v) * chart_w
        lo_x = chart_x + (lo / max_v) * chart_w
        hi_x = chart_x + (hi / max_v) * chart_w
        cy = y + 17
        parts.append(
            f'<text x="48" y="{cy + 5}" font-family="{FONT}" font-size="14" fill="{COLORS["text"]}">{esc(label)}</text>'
        )
        parts.append(
            f'<rect x="{chart_x}" y="{y}" width="{chart_w}" height="34" rx="4" fill="{COLORS["surface_2"]}" stroke="{COLORS["border"]}"/>'
        )
        parts.append(
            f'<rect x="{chart_x}" y="{y}" width="{bar_w:.1f}" height="34" rx="4" fill="{COLORS["accent"]}"/>'
        )
        parts.append(
            f'<line x1="{lo_x:.1f}" y1="{cy}" x2="{hi_x:.1f}" y2="{cy}" stroke="{COLORS["accent_dark"]}" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{lo_x:.1f}" y1="{cy - 7}" x2="{lo_x:.1f}" y2="{cy + 7}" stroke="{COLORS["accent_dark"]}" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{hi_x:.1f}" y1="{cy - 7}" x2="{hi_x:.1f}" y2="{cy + 7}" stroke="{COLORS["accent_dark"]}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{chart_x + chart_w + 22}" y="{cy + 5}" font-family="{MONO}" font-size="14" font-weight="700" fill="{COLORS["text"]}">{value:.3f}x</text>'
        )
        parts.append(
            f'<text x="{chart_x + chart_w + 22}" y="{cy + 24}" font-family="{MONO}" font-size="11" fill="{COLORS["muted"]}">[{lo:.3f}, {hi:.3f}]</text>'
        )
    parts.append(
        f'<text x="48" y="374" font-family="{FONT}" font-size="12" fill="{COLORS["secondary"]}">Intervals are paired task-bootstrap intervals from the displayed exact-path artifact.</text>'
    )
    return svg_shell(
        width,
        height,
        "VANTAGE/SafeRoute is faster than tuned PLD",
        "Speedup ratios on controlled explicit-map Python edit workloads.",
        "\n  ".join(parts),
    )


def verifier_steps_chart() -> str:
    width, height = 1100, 500
    chart_x, chart_y = 278, 194
    chart_w, row_h = 620, 66
    max_steps = 1300
    rows = [
        ("Field substitutions", 752, 388, "48.4% fewer"),
        ("Identifier-style substitutions", 1200, 404, "66.3% fewer"),
        ("Mixed controlled suite", 901, 498, "44.7% fewer"),
    ]
    parts: list[str] = []
    for tick in (0, 400, 800, 1200):
        x = chart_x + (tick / max_steps) * chart_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{chart_y - 18}" x2="{x:.1f}" y2="{chart_y + row_h * len(rows) + 4}" stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{chart_y + row_h * len(rows) + 34}" font-family="{MONO}" font-size="11" fill="{COLORS["muted"]}" text-anchor="middle">{tick}</text>'
        )
    parts.append(
        f'<rect x="48" y="138" width="10" height="10" fill="{COLORS["pld"]}"/><text x="66" y="148" font-family="{FONT}" font-size="12" fill="{COLORS["secondary"]}">PLD verifier steps</text>'
    )
    parts.append(
        f'<rect x="204" y="138" width="10" height="10" fill="{COLORS["accent"]}"/><text x="222" y="148" font-family="{FONT}" font-size="12" fill="{COLORS["secondary"]}">VANTAGE/SR verifier steps</text>'
    )
    for i, (label, pld, vantage, reduction) in enumerate(rows):
        y = chart_y + i * row_h
        pld_w = (pld / max_steps) * chart_w
        v_w = (vantage / max_steps) * chart_w
        parts.append(
            f'<text x="48" y="{y + 20}" font-family="{FONT}" font-size="14" fill="{COLORS["text"]}">{esc(label)}</text>'
        )
        parts.append(
            f'<rect x="{chart_x}" y="{y - 4}" width="{pld_w:.1f}" height="18" rx="4" fill="{COLORS["pld"]}"/>'
        )
        parts.append(
            f'<rect x="{chart_x}" y="{y + 22}" width="{v_w:.1f}" height="18" rx="4" fill="{COLORS["accent"]}"/>'
        )
        parts.append(
            f'<text x="{chart_x + pld_w + 10:.1f}" y="{y + 10}" font-family="{MONO}" font-size="12" fill="{COLORS["secondary"]}">{pld}</text>'
        )
        parts.append(
            f'<text x="{chart_x + v_w + 10:.1f}" y="{y + 36}" font-family="{MONO}" font-size="12" font-weight="700" fill="{COLORS["text"]}">{vantage}</text>'
        )
        parts.append(
            f'<text x="{chart_x + chart_w + 28}" y="{y + 24}" font-family="{FONT}" font-size="13" font-weight="700" fill="{COLORS["accent_dark"]}">{esc(reduction)}</text>'
        )
    parts.append(
        f'<text x="48" y="454" font-family="{FONT}" font-size="12" fill="{COLORS["secondary"]}">Verifier steps are logged decode/verification steps under the shared root/catch-up convention.</text>'
    )
    return svg_shell(
        width,
        height,
        "Hidden views reduce verifier work",
        "The measured speedup is explained by fewer target-verifier steps on structured rows.",
        "\n  ".join(parts),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "readme-vantage-flow.svg").write_text(
        mechanism_diagram(), encoding="utf-8"
    )
    (OUT / "readme-speedups.svg").write_text(speedup_chart(), encoding="utf-8")
    (OUT / "readme-verifier-steps.svg").write_text(
        verifier_steps_chart(), encoding="utf-8"
    )
    print(OUT / "readme-vantage-flow.svg")
    print(OUT / "readme-speedups.svg")
    print(OUT / "readme-verifier-steps.svg")


if __name__ == "__main__":
    main()
