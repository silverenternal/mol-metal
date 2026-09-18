#!/usr/bin/env bash
# WF-Paper-Repair Phase 2: build paper/main.pdf via 4-pass pdflatex+bibtex.
# Build from project root or from paper/ — paths are relative to paper/.
set -e
cd "$(dirname "$0")/../../paper"
echo "=== Working dir: $(pwd) ==="
echo "=== Pass 1: pdflatex ==="
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/pdflatex_pass1.log 2>&1 || {
    echo "FAIL on pass 1; tail of log:"
    tail -40 /tmp/pdflatex_pass1.log
    exit 1
}
echo "=== Pass 2: bibtex ==="
bibtex main > /tmp/bibtex_pass2.log 2>&1 || {
    echo "FAIL on pass 2 bibtex; tail of log:"
    tail -40 /tmp/bibtex_pass2.log
    exit 1
}
echo "=== Pass 3: pdflatex ==="
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/pdflatex_pass3.log 2>&1 || {
    echo "FAIL on pass 3; tail of log:"
    tail -40 /tmp/pdflatex_pass3.log
    exit 1
}
echo "=== Pass 4: pdflatex ==="
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/pdflatex_pass4.log 2>&1 || {
    echo "FAIL on pass 4; tail of log:"
    tail -40 /tmp/pdflatex_pass4.log
    exit 1
}
echo "=== Verify main.pdf ==="
ls -la main.pdf
echo "=== DONE ==="
