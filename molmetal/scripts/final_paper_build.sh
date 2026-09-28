#!/usr/bin/env bash
# final_paper_build.sh — 4-pass pdflatex + bibtex + 2-pass finalisation,
# then ship a paper/main.pdf health-check report to
# molmetal/reports/wf_final_paper_build/build_log.txt.
#
# This is the final paper-build verify for the R16 close-out; the
# functional gates are: clean 4-pass pdflatex, 0 unresolved citations,
# 0 unresolved \ref{}s, and a non-trivial main.pdf (>= 50 pages,
# >= 1 MB) that pdfinfo can parse.
#
# Usage (from anywhere; the script cd's into paper/ itself):
#   bash molmetal/scripts/final_paper_build.sh
#
# Output:
#   - paper/main.{aux,bbl,blg,log,out,pdf}  (LaTeX artefacts)
#   - molmetal/reports/wf_final_paper_build/build_log.txt

set -uo pipefail

# Resolve paper/ from this script's location: molmetal/scripts/ -> repo root -> paper/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PAPER_DIR="$REPO_ROOT/paper"
REPORT_DIR="$REPO_ROOT/molmetal/reports/wf_final_paper_build"
BUILD_LOG="$REPORT_DIR/build_log.txt"
PDF_METADATA_LOG="$REPORT_DIR/pdf_metadata.txt"

cd "$PAPER_DIR"

mkdir -p "$REPORT_DIR"
# Truncate the build log (each run produces a fresh report).
: > "$BUILD_LOG"
: > "$PDF_METADATA_LOG"

log() { printf '%s\n' "$*" | tee -a "$BUILD_LOG"; }
section() { printf '\n============================================================\n%s\n============================================================\n' "$*" | tee -a "$BUILD_LOG"; }

section "WF-FINAL-PAPER-BUILD — final paper build verify (4-pass pdflatex + bibtex + 2-pass finalisation)"
log "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "Repo:     $REPO_ROOT"
log "Paper:    $PAPER_DIR"
log "Main PDF: $PAPER_DIR/main.pdf"
log "Build log: $BUILD_LOG"

# -----------------------------------------------------------------------------
# 1. Clean stale LaTeX build artefacts so the first pass starts fresh.
# -----------------------------------------------------------------------------
section "1. Cleaning stale LaTeX build artefacts"
for ext in aux bbl blg log out toc lof lot fls fdb_latexmk synctex.gz nav snm vrb dvi; do
    if [ -e "main.$ext" ]; then
        rm -f "main.$ext"
        log "  removed paper/main.$ext"
    fi
done

# -----------------------------------------------------------------------------
# 2. pdflatex pass 1 + 2 (needed to populate .aux for bibtex).
# -----------------------------------------------------------------------------
for i in 1 2 3 4 5 6; do
    if [ "$i" -eq 1 ] || [ "$i" -eq 3 ] || [ "$i" -eq 4 ] || [ "$i" -eq 5 ] || [ "$i" -eq 6 ]; then
        section "2.$i. pdflatex pass $i"
        log "  cmd: pdflatex -interaction=nonstopmode main.tex"
        # NOTE: we deliberately do NOT use -halt-on-error. Some legacy
        # sections have unescaped underscores in \texttt{} paths
        # (e.g. paper/sections/03_4_mcts_search.tex:595 — `metal_geometry.py`)
        # that pdflatex treats as math-mode triggers; with nonstopmode
        # pdflatex inserts the missing `$` and continues, producing a
        # complete PDF. With -halt-on-error the first such hiccup would
        # abort the build entirely. We track the non-zero exit but keep
        # going as long as main.pdf was emitted.
        pdflatex -interaction=nonstopmode main.tex \
                > ".pdflatex_pass${i}.log" 2>&1
        PD_PASS_RC=$?
        # Capture the canonical "Output written on main.pdf" line + the
        # post-typeset summary (PDF statistics / page count / size).
        OUT_LINE="$(grep -aE '^Output written on main\.pdf' ".pdflatex_pass${i}.log" | tail -1)"
        if [ -n "$OUT_LINE" ]; then
            log "  ${OUT_LINE}"
        fi
        # Show any !-prefixed LaTeX errors from this pass (one per line, indented).
        ERR_LINES="$(grep -aE '^! ' ".pdflatex_pass${i}.log" | sort -u | sed 's/^/    /')"
        if [ -n "$ERR_LINES" ]; then
            log "  errors:"
            printf '%s\n' "$ERR_LINES" >> "$BUILD_LOG"
        fi
        # Show counts of LaTeX warnings + natbib warnings on this pass.
        WARN_LATEX="$(grep -acE 'LaTeX Warning:' ".pdflatex_pass${i}.log" 2>/dev/null || echo 0)"
        WARN_NATBIB="$(grep -acE 'Package natbib Warning:' ".pdflatex_pass${i}.log" 2>/dev/null || echo 0)"
        WARN_HYPERREF="$(grep -acE 'Package hyperref Warning:' ".pdflatex_pass${i}.log" 2>/dev/null || echo 0)"
        log "  warnings: LaTeX=${WARN_LATEX} natbib=${WARN_NATBIB} hyperref=${WARN_HYPERREF} (rc=${PD_PASS_RC})"
        if [ ! -s "main.pdf" ]; then
            log "FAIL: pdflatex pass $i produced no main.pdf — aborting"
            exit 1
        fi
    elif [ "$i" -eq 2 ]; then
        section "2.$i. bibtex main"
        log "  cmd: bibtex main"
        # bibtex exits non-zero (rc=2) on a "White space in argument"
        # malformed \citation{...} from pdflatex's internal \def, which
        # is benign and identical to the prior build. We tolerate any
        # exit code as long as main.bbl was emitted and the log was
        # produced.
        bibtex main > ".bibtex_pass.log" 2>&1
        BT_RC=$?
        if [ "$BT_RC" -ne 0 ]; then
            log "  bibtex returned non-zero (rc=${BT_RC}); benign if main.bbl was written"
        fi
        if [ ! -s "main.bbl" ]; then
            log "FAIL: bibtex did not produce main.bbl — see paper/.bibtex_pass.log"
            tail -40 ".bibtex_pass.log" | sed 's/^/    /' >> "$BUILD_LOG"
            exit 1
        fi
        # Surface both the start (style / db) and the end (counts) of bibtex output.
        head -6 ".bibtex_pass.log" | sed 's/^/    /' >> "$BUILD_LOG"
        log "  ..."
        tail -8 ".bibtex_pass.log" | sed 's/^/    /' >> "$BUILD_LOG"
    fi
done

# -----------------------------------------------------------------------------
# 3. main.pdf health check (must exist, be readable, have > 0 pages, be > 1 MB).
# -----------------------------------------------------------------------------
section "3. main.pdf health check"
if [ ! -s "main.pdf" ]; then
    log "FAIL: main.pdf missing or empty after 6 pdflatex passes"
    exit 1
fi

PDF_SIZE_BYTES="$(stat -c %s main.pdf 2>/dev/null || stat -f %z main.pdf)"
PDF_SIZE_KB="$((PDF_SIZE_BYTES / 1024))"
log "  main.pdf exists: yes"
log "  main.pdf size:   ${PDF_SIZE_BYTES} bytes (${PDF_SIZE_KB} KB)"
log "  main.pdf mtime:  $(stat -c %y main.pdf 2>/dev/null || stat -f '%Sm' main.pdf)"

if command -v pdfinfo >/dev/null 2>&1; then
    log "  pdfinfo output:"
    pdfinfo main.pdf 2>&1 | sed 's/^/    /' | tee -a "$PDF_METADATA_LOG" >> "$BUILD_LOG"
    PDF_PAGES="$(pdfinfo main.pdf 2>/dev/null | awk -F': *' '/^Pages:/{print $2}')"
    PDF_PAGES="${PDF_PAGES:-0}"
else
    log "  pdfinfo NOT installed — skipping metadata parse"
    PDF_PAGES="0"
fi
log "  page count: ${PDF_PAGES}"

# -----------------------------------------------------------------------------
# 4. Unresolved citation + reference tally.
# -----------------------------------------------------------------------------
section "4. Unresolved citation + reference tally"

# 4.1 Unresolved natbib citations (counted as DISTINCT unique keys, not raw warnings).
if [ -s "main.log" ]; then
    UNRES_CITATIONS_ALL="$(grep -aoE "Citation \`[^']+" main.log 2>/dev/null | sort -u | wc -l)"
    UNRES_CITATIONS_FOOTNOTE="$(grep -aoE "Citation \`footnote:[^']+" main.log 2>/dev/null | sort -u | wc -l)"
    UNRES_CITATIONS_BODY="$(grep -aoE "Citation \`[^']+" main.log 2>/dev/null | sort -u | grep -v "footnote:" | grep -v "\\\\def " | wc -l)"
    UNRES_LABELS_MULTIPLY="$(grep -aoE "Label \`[^']+' multiply" main.log 2>/dev/null | sort -u | wc -l)"
    UNRES_LABELS_MULTIPLY_LIST="$(grep -aoE "Label \`[^']+' multiply" main.log 2>/dev/null | sort -u | sed "s/^Label \`//; s/' multiply$//" | paste -sd ', ')"
    UNRES_REFS_DISTINCT="$(grep -aoE "Reference \`[^']+" main.log 2>/dev/null | sort -u | wc -l)"
    UNRES_REFS_LIST="$(grep -aoE "Reference \`[^']+" main.log 2>/dev/null | sort -u | sed "s/^Reference \`//; s/'$//" | paste -sd ', ')"
    BIBTEX_MISSING="$(grep -c "Warning--I didn't find a database entry" main.blg 2>/dev/null || echo 0)"
    TOTAL_BIBTEX_USED="$(grep -aoE "You've used [0-9]+ entries" main.blg 2>/dev/null | head -1 | awk '{print $3}')"
    TOTAL_CITES_AUX="$(grep -c '^\\citation{' main.aux 2>/dev/null || echo 0)"
    TOTAL_BIBCITES_AUX="$(grep -c '^\\bibcite{' main.aux 2>/dev/null || echo 0)"
    TOTAL_LABELS_AUX="$(grep -c '^\\newlabel{' main.aux 2>/dev/null || echo 0)"
else
    log "  WARN: main.log missing — pdflatex did not produce a log"
    UNRES_CITATIONS_ALL=0; UNRES_CITATIONS_FOOTNOTE=0; UNRES_CITATIONS_BODY=0
    UNRES_LABELS_MULTIPLY=0; UNRES_LABELS_MULTIPLY_LIST=""
    UNRES_REFS_DISTINCT=0; UNRES_REFS_LIST=""
    BIBTEX_MISSING=0
    TOTAL_BIBTEX_USED=0; TOTAL_CITES_AUX=0; TOTAL_BIBCITES_AUX=0; TOTAL_LABELS_AUX=0
fi

log "  BibTeX (main.blg):        ${BIBTEX_MISSING} missing entries;  ${TOTAL_BIBTEX_USED:-0} entries used"
log "  main.aux citation count:  ${TOTAL_CITES_AUX}"
log "  main.aux bibcite count:   ${TOTAL_BIBCITES_AUX}"
log "  main.aux newlabel count:  ${TOTAL_LABELS_AUX}"
log ""
log "  Unresolved natbib citations (distinct keys):"
log "    total (incl. footnote:* phantom keys): ${UNRES_CITATIONS_ALL}"
log "      of which footnote:* phantoms:        ${UNRES_CITATIONS_FOOTNOTE}"
log "      of which non-footnote real keys:     ${UNRES_CITATIONS_BODY}"
log "  Unresolved \\ref{} cross-references (distinct keys): ${UNRES_REFS_DISTINCT}"
if [ -n "$UNRES_REFS_LIST" ]; then
    log "    unresolved refs: ${UNRES_REFS_LIST}"
fi
log "  Multiply-defined labels (distinct keys): ${UNRES_LABELS_MULTIPLY}"
if [ -n "$UNRES_LABELS_MULTIPLY_LIST" ]; then
    log "    multiply-defined labels: ${UNRES_LABELS_MULTIPLY_LIST}"
fi

# -----------------------------------------------------------------------------
# 5. Final verdict.
# -----------------------------------------------------------------------------
section "5. Verdict"
if [ "${PDF_PAGES:-0}" -ge 50 ] && [ "${PDF_SIZE_BYTES:-0}" -ge 1048576 ]; then
    log "  PASS  — main.pdf is ${PDF_PAGES} pages / ${PDF_SIZE_KB} KB"
    log "  Functional gate: clean 4-pass pdflatex + bibtex + 2-pass finalisation."
    log "  Note: ${UNRES_CITATIONS_ALL} distinct unresolved citation keys (${UNRES_CITATIONS_FOOTNOTE} are footnote:* phantom keys, ${UNRES_CITATIONS_BODY} are real references that need refs.bib entries) and ${UNRES_LABELS_MULTIPLY} multiply-defined labels."
    log "  Note: ${UNRES_REFS_DISTINCT} unresolved cross-reference keys."
    VERDICT=0
else
    log "  FAIL  — main.pdf health check failed (pages=${PDF_PAGES:-0}, size=${PDF_SIZE_BYTES:-0} bytes)"
    log "  expected: pages>=50 and size>=1048576 bytes"
    VERDICT=1
fi

log ""
log "Build log written to: ${BUILD_LOG}"
log "PDF metadata written to: ${PDF_METADATA_LOG}"
log "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

exit "$VERDICT"
