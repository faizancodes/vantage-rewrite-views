#!/usr/bin/env bash
set -euo pipefail
MAIN_TEX="${1:-main.tex}"
OUT="${2:-vantage_arxiv_source.tar.gz}"
if [[ ! -f "$MAIN_TEX" ]]; then
  echo "Missing main TeX file: $MAIN_TEX" >&2
  exit 1
fi
MAIN_DIR="$(cd "$(dirname "$MAIN_TEX")" && pwd)"
OUT_ABS="$OUT"
if [[ "$OUT" != /* ]]; then
  OUT_ABS="$PWD/$OUT"
fi
required_figs=(
  "figures/rewrite_anchor_diagram.pdf"
  "figures/compliance_field_rename.pdf"
  "figures/compliance_style_rewrite.pdf"
)
for f in "${required_figs[@]}"; do
  if [[ ! -f "$MAIN_DIR/$f" ]]; then
    echo "Missing required figure: $f" >&2
    exit 1
  fi
done
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
cp "$MAIN_TEX" "$tmpdir/main.tex"
mkdir -p "$tmpdir/figures"
for f in "${required_figs[@]}"; do
  cp "$MAIN_DIR/$f" "$tmpdir/$f"
done
(
  cd "$tmpdir"
  if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  else
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
  fi
  rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.synctex.gz
  tar czf "$OUT_ABS" main.tex figures
)
echo "Wrote $OUT"
