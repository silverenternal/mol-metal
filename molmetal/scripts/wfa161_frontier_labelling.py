#!/usr/bin/env python
"""Wf-A161 — Frontier labelling convention for connectivity / validity percentages.

WHY
---
Mol-Metal prints a headline "8.51 %" connectivity figure that renders
identically for SEVEN different populations.  A published "92.7-94 %"
band also circulates in this literature, but it is a FRONTIER composite
and NOT ours, and it is NOT one homogeneous quantity: the repo holds
TWO competing provenance readings for it, and under NEITHER of them is
it a like-for-like comparison to our number.

    Wf-A176 (2026-11-01) CORRECTION — the superseded sentence, PRESERVED
    VERBATIM per AGENTS.md §5.5 rule #6:

        Mol-Metal prints a headline "8.51 %" connectivity figure that
        renders identically for SEVEN different populations, and a
        published "92.7-94 %" band that comes from a DIFFERENT perception
        branch (Open Babel) and a DIFFERENT post-processing than any
        frontier paper's.

    That sentence was wrong in three ways: the band is NOT ours (it is a
    frontier composite — `metrics/by_metric/wfa155_metal_coordination_sphere.json`,
    `comparability_statement.summary`: *"This metric is NOT comparable to
    the published 92.7-94 % frontier band"*), its endpoints are NOT
    connectivity rates (`frontier_matrix.json`, `band_provenance`:
    *"THE BAND IS NOT A CONNECTIVITY BAND"*), and no Open Babel branch is
    named as their source anywhere in the band-provenance record.

    Wf-A176 (2026-11-01) CORRECTION 2 — the paragraph above is itself
    SUPERSEDED (preserved VERBATIM per AGENTS.md §5.5 rule #6).  It
    asserted that NO candidate endpoint of the band is a
    connected-component count.  That is TRUE under ONE provenance
    reading and FALSE under the OTHER, so it may not be published as
    settled.  The repo holds TWO competing readings, both in
    `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`:

        Reading 1  JSON PATH `band_provenance`: GeoLDM's Validity & Uniq
                   92.7 % and HLTF's PB-Valid 94.0 %, both with
                   `post_processed: false`; a third paper quotes 92.7 %
                   as a per-bond-type classification F1.  Under this
                   reading no endpoint is a connected-component count.
        Reading 2  JSON PATH `sources`.`EF-TALFM`, EF-TALFM
                   arXiv:2609.08333v4: the band mixes EF-TALFM's OWN
                   Table 3(a) raw connectivity cell (C. = 92.98 %, counted
                   BEFORE the largest-component retention) with its
                   post-processed N. & U. & S. companion cell (94.33 %).
                   Under THIS reading the lower endpoint IS a
                   connectivity count.

    The invariants that hold under BOTH readings — and are therefore the
    only claims this script may publish — are: (i) the band is EXTERNAL,
    (ii) it is NOT one homogeneous quantity, (iii) it is NOT on our
    scale.  OUR 8.51 % IS a connected-component count on the RAW emitted
    SMILES, and nothing else.

    Wf-A183 (2026-11-01) CORRECTION — the CORRECTION 2 block above carried
    TWO locator defects and ONE stale attribution.  The superseded
    rendering is PRESERVED VERBATIM per AGENTS.md §5.5 rule #6 (it is also
    reproduced in the `## Wf-A183 CORRECTION (2026-11-01)` section this
    script EMITS into `report.md`):

        Reading 1  `band_provenance` (:51-89): GeoLDM's Validity & Uniq
                   92.7 % and HLTF's PB-Valid 94.0 %, both with
                   `post_processed: false`; a third paper quotes 92.7 %
                   as a per-bond-type classification F1.  Under this
                   reading no endpoint is a connected-component count.
        Reading 2  `frontier_axis_mapping` (:360, :378-379), EF-TALFM
                   arXiv:2609.08333v4: the band mixes EF-TALFM's OWN
                   Table 3(a) raw connectivity cell (C. = 92.98 %, counted
                   BEFORE the largest-component retention) with its
                   post-processed N. & U. & S. companion cell (94.33 %).
                   Under THIS reading the lower endpoint IS a
                   connectivity count.

    WHAT WAS WRONG, and what the corrected rendering above now says:

    * `frontier_axis_mapping` is NOT a JSON key of the frontier matrix
      record.  It survives only as a STRING VALUE, so pointing a reader
      at it is a DEFECT.  The ROBUST locator for Reading 2 is the JSON
      PATH `sources`.`EF-TALFM`.
    * LINE-NUMBER locators (`:51-89`, `:360, :378-379`) rot as the
      record gains keys.  Cite JSON PATHS, not line numbers.
    * The Wf-A176 CORRECTION block above quotes "THE BAND IS NOT A
      CONNECTIVITY BAND" as the record's own wording.  That wording is NO
      LONGER the LIVE rendering of `band_provenance`.`headline_finding` —
      the live rendering at that JSON PATH is now the reading-safe clause,
      and the quoted wording is PRESERVED VERBATIM at
      `band_provenance`.`headline_finding_superseded_2026_11_01` (see
      also `headline_finding_superseded_note_2026_11_01`).  The quote is
      an X1-class claim: TRUE under Reading 1 and FALSE under Reading 2,
      so it may not be published as settled.
    * The invariants that hold under BOTH readings are unchanged, and they
      are the only claims this script may publish: (i) the band is
      EXTERNAL, (ii) it is NOT one homogeneous quantity, (iii) it is NOT
      on our scale, (iv) the band's source cells are NOT settled on disk.

    The question this script answers is empirical, not rhetorical:

    When a frontier generative-molecular paper prints a connectivity /
    validity percentage, WHAT QUALIFIER does it print BESIDE the number?
    Is the population (N), the post-processing (largest-component
    retention? relaxation?) and the definition (per-molecule?
    per-pool?) stated AT the number — or only in a distant table caption
    or an appendix subsection?

METHOD
------
1. Fetch the arXiv e-print (LaTeX source) for each paper, in memory.
   A local cache directory is used so a re-run is offline-capable and
   reproducible; the exact URL, HTTP outcome and sha256 of every
   retrieved tarball are recorded.
2. Split each .tex file into "sites": LaTeX float environments
   (table/figure, caption included) and running-text paragraphs.
3. Keep the sites that report a connectivity / validity percentage.
4. For every kept site, classify FOUR qualifier classes:
      population  — the denominator N (how many molecules)
      postproc    — largest-component retention, relaxation, sanitisation,
                    edge resampling, valency correction ...
      definition  — what the metric MEANS (one fragment? RDKit
                    sanitizable? fraction of what?)
      hydrogens   — explicit vs implicit hydrogen handling
   Classifying "AT the number" means: the qualifier string occurs inside
   the SAME site as the number (the table caption that carries the
   numbers counts as "at"; a separate appendix subsection counts as
   "elsewhere").
5. Derive one verdict per paper from the machine-measured evidence:
      AGREES       — population + post-processing + definition are all
                     printed AT the number (the convention we should adopt)
      DISTINGUISHES — the qualifier exists but in a DIFFERENT place or a
                     DIFFERENT form (e.g. a second "w/o correction" column)
      NOT-STATED   — the qualifier is absent (or the metric is absent
                     altogether)
      NOT-RETRIEVED — the fetch failed; no number is guessed

The verdict is DERIVED, not hand-written, so it can be re-checked.
The raw per-site evidence is stored alongside the verdict so a reviewer
can see exactly which string produced which label.

READ-ONLY with respect to the repo: this script only writes into its own
output directory (``--out-dir``).

CLI
---
    python molmetal/scripts/wfa161_frontier_labelling.py            # fetch + write
    python molmetal/scripts/wfa161_frontier_labelling.py --offline  # cache only
    python molmetal/scripts/wfa161_frontier_labelling.py --print    # stdout only

Cross-references
----------------
AGENTS.md §5.2 (MEASURED vs PROJECTED), §5.6 (atomic write protocol),
§5.9 (single venv).  Sibling: ``molmetal/reports/wfa161_frontier_labelling/``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # allow `python molmetal/scripts/...` from repo root
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
except Exception as exc:  # pragma: no cover - fallback for direct execution
    atomic_write_file = None  # type: ignore[assignment]
    _ATOMIC_IMPORT_ERROR = exc


DEFAULT_CACHE_DIR = Path("/tmp/wfa161_arxiv_cache")
DEFAULT_OUT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wfa161_frontier_labelling"

USER_AGENT = "mol-metal-wfa161/1.0 (frontier labelling convention audit; contact: repo owner)"

# ---------------------------------------------------------------------------
# The paper set
# ---------------------------------------------------------------------------
# `metric_terms` are curated per paper (case-insensitive regexes) so the
# extractor does not fire on unrelated uses of the word "valid" (e.g.
# "validation of the atom-number predictor").  Every term is recorded in
# sources.json so a reviewer can audit the selection.
PAPERS: list[dict[str, Any]] = [
    {
        "key": "ef_talfm",
        "arxiv_id": "2609.08333",
        "title": "Fixed-Dimensional Latent Flow for Generating Variable-Size 3D Molecules",
        "short": "EF-TALFM",
        "venue": "arXiv preprint (EF-TALFM); the paper our bond rule came from",
        "metric_label": "Connectivity (C.) = fraction of molecules with exactly one connected component",
        "metric_terms": [r"connectiv", r"connected"],
        "notes": (
            "LaTeX source was pre-cached by Wf-A160 at /tmp/wfa160_arxiv_cache/2609.08333.tar.gz "
            "(sha256 e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224) and is "
            "re-fetched live from the arXiv e-print endpoint otherwise."
        ),
    },
    {
        "key": "flowmol3",
        "arxiv_id": "2508.12629",
        "title": "FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation",
        "short": "FlowMol3",
        "venue": "Digital Discovery (2026)",
        "metric_label": "% Valid = percent of molecules that can be sanitized by RDKit; % PB-Valid adds the PoseBusters suite (incl. one-component check)",
        "metric_terms": [r"%\s*valid", r"pb-?valid", r"\bvalidity\b", r"\bvalid\b", r"(?<![A-Za-z])Val\."],
        "notes": "RSC achemso template; the title lives in a graphicx box, not in a \\title macro.",
    },
    {
        "key": "ggflow",
        "arxiv_id": "2411.05676",
        "title": "Improving Molecular Graph Generation with Flow Matching and Optimal Transport",
        "short": "GGFlow",
        "venue": "TMLR (retitled 2026); originally NeurIPS 2024",
        "metric_label": "Validity = fraction of valid molecules AFTER valency correction or edge resampling; Val. w/o corr. = the same fraction BEFORE correction",
        "metric_terms": [r"\bvalidity\b", r"(?<![A-Za-z])Val\."],
        "notes": (
            "The headline validity column is post-correction for every method (100 for all rows); "
            "the informative number is the second column, 'Val. w/o corr.'"
        ),
    },
    {
        "key": "paflow",
        "arxiv_id": "2509.01486",
        "title": "Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number",
        "short": "PAFlow",
        "venue": "NeurIPS 2025",
        "metric_label": "(none reported) — the paper reports Vina affinity, high-affinity rate, QED, SA and Diversity; no connectivity / validity percentage",
        "metric_terms": [r"\bvalidity\b", r"%\s*valid", r"connectiv"],
        "notes": "Negative control: the paper prints no connectivity/validity percentage at all.",
    },
    {
        "key": "geom_dd",
        "arxiv_id": "2401.02683",
        "title": "Geometric-Facilitated Denoising Diffusion Model for 3D Molecule Generation",
        "short": "GEOM-DD / GFMDiff",
        "venue": "arXiv preprint (v2 title) — AAAI-style source (paper_aaai24_cr.tex)",
        "metric_label": "Mol. Stability (%) = molecules whose atoms are all stable (single fragment); Validity (%) = molecules that are theoretically correct",
        "metric_terms": [r"\bvalidity\b", r"\bvalid\b", r"stability\s*\(%\)", r"connectiv"],
        "notes": (
            "Fifth paper, located via the arXiv API rather than by guessing an identifier "
            "(several candidate identifiers resolved to unrelated papers)."
        ),
    },
]


# ---------------------------------------------------------------------------
# Qualifier patterns
# ---------------------------------------------------------------------------
# Each entry is (compiled regex, human-readable label).  "population" is the
# denominator N; "postproc" is any named post-processing step; "definition"
# is what the metric MEANS; "hydrogens" is explicit-vs-implicit H handling.
POPULATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{1,3}(?:,\d{3})*\s+generated\s+(?:samples|molecules)\b", re.I), "generated-sample count"),
    (re.compile(r"\brates?\s+(?:are|is)\s+computed\s+over\s+[\d,]+\s+generated\s+molecules\b", re.I), "rates-computed-over-N"),
    (re.compile(r"\bresults?\s+of\s+[\d,]+\s+generated\s+samples\b", re.I), "results-of-N-generated-samples"),
    (re.compile(r"\bover\s+[\d,]+\s+generated\s+(?:samples|molecules)\b", re.I), "over-N-generated"),
    (re.compile(r"\bsampled\s+[\d,]+\s+(?:molecules|samples)\b", re.I), "sampled-molecule count"),
    (re.compile(r"\bwe\s+sampled\s+[\d,]+\s+(?:molecules|samples)\b", re.I), "we-sampled-N"),
    (re.compile(r"\bN\s*=\s*[\d,]+\b"), "N="),
    (re.compile(r"\bper\s+[\d,]+\s+generated\b", re.I), "per-N-generated"),
    (re.compile(r"\btotal\s+of\s+[\d,]+\s+generated\b", re.I), "total-of-N-generated"),
    (re.compile(r"\b[\d,]+\s+generated\s+samples\s+are\s+reported\b", re.I), "N-generated-samples-reported"),
]

POSTPROC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"largest\s+connected\s+component", re.I), "largest-connected-component-retained"),
    (re.compile(r"largest\s+fragment", re.I), "largest-fragment-retained"),
    (re.compile(r"largest\s+component", re.I), "largest-component-retained"),
    (re.compile(r"retain(?:ing|s|ed)?\s+(?:the\s+)?(?:atom-count\s+)?largest", re.I), "largest-component-retained"),
    (re.compile(r"after\s+(?:rdkit\s+)?sanitiz", re.I), "after-sanitization"),
    (re.compile(r"edge\s+resampl", re.I), "edge-resampling"),
    (re.compile(r"valency\s+correction", re.I), "valency-correction"),
    (re.compile(r"\brelax(?:ation|ed|ing)\b", re.I), "relaxation"),
    (re.compile(r"post-?\s?process", re.I), "post-processing"),
    (re.compile(r"\brefin(?:e|ement|ed)\b", re.I), "geometry-refinement"),
]

DEFINITION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"exactly\s+one\s+connected\s+component", re.I), "one-fragment-definition"),
    (re.compile(r"(?:%|percent)\s+of\s+molecules", re.I), "percent-of-molecules"),
    (re.compile(r"\bfraction\s+of\b", re.I), "fraction-of"),
    (re.compile(r"denotes\s+the\s+fraction", re.I), "denotes-the-fraction"),
    (re.compile(r"can\s+be\s+sanitized", re.I), "rdkit-sanitizable-definition"),
    (re.compile(r"measures\s+the\s+proportion", re.I), "proportion-definition"),
    (re.compile(r"is\s+defined\s+as", re.I), "is-defined-as"),
    (re.compile(r"quantifies\s+the\s+percentage", re.I), "quantifies-the-percentage"),
    (re.compile(r"forms\s+one\s+component", re.I), "one-component-definition"),
    (re.compile(r"not\s+fragmented", re.I), "not-fragmented-definition"),
    (re.compile(r"molecules\s+whose\s+atoms\s+are\s+all\s+stable", re.I), "molecule-stability-definition"),
    (re.compile(r"theoretically\s+correct", re.I), "theoretically-correct-definition"),
    (re.compile(r"\%\s*of\s+(?:the\s+)?molecules\s+with", re.I), "percent-with-one-fragment"),
]

HYDROGEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"explicit\s+hydrogen", re.I), "explicit-hydrogens"),
    (re.compile(r"implicit\s+hydrogen", re.I), "implicit-hydrogens"),
    (re.compile(r"heavy[-\s]atom", re.I), "heavy-atom-only"),
    (re.compile(r"all-atom", re.I), "all-atom"),
]

QUADRANT_ORDER = ("population", "postproc", "definition", "hydrogens")
QUADRANT_PATTERNS = {
    "population": POPULATION_PATTERNS,
    "postproc": POSTPROC_PATTERNS,
    "definition": DEFINITION_PATTERNS,
    "hydrogens": HYDROGEN_PATTERNS,
}

# Percentage-ish token inside a LaTeX source: 12.3\%, 12.3%, $99.9 \pm 0.1$
PERCENT_TOKEN = re.compile(r"\\?%")
# A number that carries a percentage sign: 99.9%, 92.98\%, 10,000%
PERCENT_VALUE = re.compile(r"\d[\d.,]*\s*\\?%")
# A bare number in a validity column (no percent sign) — only meaningful
# inside a tabular whose header carries the metric term.
NUMERIC_CELL = re.compile(r"\d+\.\d+|\b\d{2,3}\b")

# LaTeX float environment boundaries.
_FLOAT_RE = re.compile(
    r"\\begin\{(table\*?|figure\*?|tabular\*?)\}(.*?)\\end\{\1\}",
    re.DOTALL | re.IGNORECASE,
)

VERDICTS = ("AGREES", "DISTINGUISHES", "NOT-STATED", "NOT-RETRIEVED")

# The keys of the papers in `PAPERS`, in order.  Exposed so the regression
# test can check the committed artefact covers the script's paper list.
PAPER_KEYS = tuple(p["key"] for p in PAPERS)

# Loose "is this sentence about the metric at all?" test, used only to
# RANK evidence hits (a strict term match is used to SELECT sites).
METRIC_RELEVANCE = re.compile(r"valid|connectiv|stabil|Val\.", re.I)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def eprint_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/e-print/{arxiv_id}"


def abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def fetch_tarball(arxiv_id: str, cache_dir: Path, *, offline: bool) -> tuple[bytes | None, dict[str, Any]]:
    """Return (tarball_bytes_or_None, provenance_record).

    Provenance always records the URL and, on failure, the reason.  No
    number is ever inferred from a failed fetch.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{arxiv_id}.tar.gz"
    record: dict[str, Any] = {
        "url": eprint_url(arxiv_id),
        "abs_url": abs_url(arxiv_id),
        "cache_path": str(cache_path),
        "retrieved": False,
        "http_status": None,
        "sha256": None,
        "bytes": None,
        "error": None,
    }

    if cache_path.exists() and cache_path.stat().st_size > 0:
        data = cache_path.read_bytes()
        record.update(
            retrieved=True,
            http_status="cache",
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
        )
        return data, record

    if offline:
        record["error"] = "not cached and --offline was requested"
        return None, record

    req = urllib.request.Request(eprint_url(arxiv_id), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed https host
            data = resp.read()
            record["http_status"] = getattr(resp, "status", 200)
    except urllib.error.HTTPError as exc:
        record["error"] = f"HTTPError {exc.code}"
        return None, record
    except Exception as exc:  # network down, TLS failure, timeout ...
        record["error"] = f"{type(exc).__name__}: {exc}"
        return None, record

    if not data:
        record["error"] = "empty response body"
        return None, record

    record.update(
        retrieved=True,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )
    tmp = cache_path.with_suffix(".tar.gz.part")
    tmp.write_bytes(data)
    os.replace(tmp, cache_path)
    return data, record


def extract_tex_files(tarball: bytes) -> dict[str, str]:
    """Extract every ``.tex`` member into memory.  Never touches the repo."""
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".tex"):
                continue
            if member.size > 8 * 1024 * 1024:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            raw = fh.read()
            out[member.name] = raw.decode("utf-8", errors="replace")
    return out


# ---------------------------------------------------------------------------
# TeX -> sites
# ---------------------------------------------------------------------------
def strip_comments(text: str) -> str:
    """Remove LaTeX comments (unescaped %)."""
    lines = []
    for line in text.splitlines():
        buf = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                buf.append(line[i : i + 2])
                i += 2
                continue
            if ch == "%":
                break
            buf.append(ch)
            i += 1
        lines.append("".join(buf))
    return "\n".join(lines)


def tex_to_text(text: str) -> str:
    """Collapse LaTeX markup into readable prose (approximate, not lossy)."""
    s = strip_comments(text)
    s = re.sub(r"\\(?:label|ref|cref|Cref|autoref)\{[^}]*\}", " ", s)
    s = re.sub(r"\\cite[a-zA-Z]*\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", s)
    s = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", s)
    s = s.replace("~", " ").replace("\\\\", " ")
    s = s.replace("{", " ").replace("}", " ").replace("$", " ")
    s = re.sub(r"\\%", "%", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_caption(raw: str) -> str:
    """Return the verbatim ``\\caption{...}`` body of a float, if any."""
    m = re.search(r"\\caption(?:of\{[^}]*\})?\s*\{", raw)
    if not m:
        return ""
    i = raw.index("{", m.start())
    depth = 0
    buf = ""
    j = i
    while j < len(raw):
        if raw[j] == "{":
            depth += 1
        elif raw[j] == "}":
            depth -= 1
            if depth == 0:
                break
        buf += raw[j]
        j += 1
    return buf


def split_sites(text: str) -> list[dict[str, Any]]:
    """Split a .tex file into float environments and running-text paragraphs.

    A float environment keeps its caption, because the caption is the
    place where a frontier paper most often states the population.

    Float spans are blanked out (newlines preserved) before the
    paragraph split, so a running-text sentence is never merged with the
    caption of the float that follows it.  Without that, a bare sentence
    would inherit the caption's qualifier and the "is the qualifier AT
    the number" test would answer wrongly.
    """
    sites: list[dict[str, Any]] = []
    consumed: list[tuple[int, int]] = []

    for m in _FLOAT_RE.finditer(text):
        kind = "float:" + m.group(1).lower()
        start_line = text[: m.start()].count("\n") + 1
        sites.append({"kind": kind, "raw": m.group(0), "line": start_line})
        consumed.append((m.start(), m.end()))

    # Blank the float spans (keeping newlines so line numbers survive).
    chars = list(text)
    for a, b in consumed:
        for i in range(a, min(b, len(chars))):
            if chars[i] != "\n":
                chars[i] = " "
    masked = "".join(chars)

    # The complement of the float spans, split into blank-line paragraphs.
    # A float is a HARD separator: a running-text sentence never absorbs
    # the caption of the float that follows it.
    regions: list[tuple[int, int]] = []
    cursor = 0
    for a, b in sorted(consumed):
        if a > cursor:
            regions.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < len(text):
        regions.append((cursor, len(text)))

    for a, b in regions:
        seg_masked = masked[a:b]
        pos = 0
        for chunk in seg_masked.split("\n\n"):
            idx = seg_masked.find(chunk, pos)
            if idx < 0:
                pos += len(chunk) + 2
                continue
            pos = idx + len(chunk)
            if not chunk.strip():
                continue
            abs_start = a + idx
            original = text[abs_start : abs_start + len(chunk)]
            if not original.strip():
                continue
            line = text[:abs_start].count("\n") + 1
            sites.append({"kind": "paragraph", "raw": original, "line": line})

    sites.sort(key=lambda s: s["line"])
    return sites


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(])", flat)
    return [p.strip() for p in parts if p.strip()]


def _find_qualifiers(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for rx, label in patterns:
        for m in rx.finditer(text):
            key = f"{label}@{m.start()}"
            if key in seen:
                continue
            seen.add(key)
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            hits.append(
                {
                    "label": label,
                    "match": m.group(0)[:120],
                    "context": text[start:end].strip()[:220],
                }
            )
    return hits


# ---------------------------------------------------------------------------
# Site -> classification
# ---------------------------------------------------------------------------
def classify_site(site: dict[str, Any], metric_terms: list[str]) -> dict[str, Any]:
    raw = site["raw"]
    text = tex_to_text(raw)
    is_float = site["kind"].startswith("float:")

    term_hits = []
    for term in metric_terms:
        if re.search(term, text, re.I):
            term_hits.append(term)

    has_percent = bool(PERCENT_TOKEN.search(raw))
    has_tabular = "tabular" in raw.lower()
    has_numeric = bool(NUMERIC_CELL.search(raw))
    # A number that actually carries a percentage sign.
    has_percent_value = bool(PERCENT_VALUE.search(raw))
    # A results table whose COLUMN HEADER declares the unit as a percent
    # (e.g. "Validity (\%)"), even though the cells are bare numbers.
    header_declares_percent = bool(re.search(r"\\%", raw))
    # The table prints bare numbers with no unit anywhere in the float.
    unitless_table = bool(has_tabular and has_numeric) and not has_percent and not header_declares_percent

    is_metric_site = bool(term_hits) and (has_percent or (has_tabular and has_numeric))
    if not is_metric_site:
        return {"kind": site["kind"], "line": site["line"], "is_metric_site": False}

    at: dict[str, list[dict[str, str]]] = {}
    for quad in QUADRANT_ORDER:
        at[quad] = _find_qualifiers(text, QUADRANT_PATTERNS[quad])

    caption_raw = extract_caption(raw) if is_float else ""
    caption_text = tex_to_text(caption_raw) if caption_raw else ""

    # Sentences inside this site that actually carry a percentage AND a
    # metric term -> the "headline excerpt".
    headline: list[str] = []
    for sent in split_sentences(text):
        if not PERCENT_TOKEN.search(sent):
            continue
        if any(re.search(term, sent, re.I) for term in metric_terms):
            headline.append(sent)
    if not headline:
        # Table block: the metric term may live in the header while the
        # number lives in a row.  Prefer the caption sentences, then the
        # metric-term sentences.
        for sent in split_sentences(caption_text):
            if any(re.search(term, sent, re.I) for term in metric_terms):
                headline.append(sent)
        if not headline:
            for sent in split_sentences(text):
                if any(re.search(term, sent, re.I) for term in metric_terms):
                    headline.append(sent)

    return {
        "kind": site["kind"],
        "line": site["line"],
        "is_metric_site": True,
        "metric_terms_matched": term_hits,
        "has_percent_sign": has_percent,
        "has_percent_value": has_percent_value or (is_float and header_declares_percent),
        "header_declares_percent": header_declares_percent,
        "unitless_table": unitless_table,
        "caption": caption_text[:1200],
        "headline_excerpt": headline[:6],
        "qualifiers_at_number": at,
    }


def _qualifier_strength(site: dict[str, Any]) -> int:
    """How many of the three load-bearing quadrants are printed AT the number."""
    at = site.get("qualifiers_at_number", {})
    return sum(1 for quad in ("population", "postproc", "definition") if at.get(quad))


def _quadrants_populated(site: dict[str, Any]) -> int:
    at = site.get("qualifiers_at_number", {})
    return sum(1 for quad in QUADRANT_ORDER if at.get(quad))


def _rank_hits(hits: list[dict[str, str]], metric_terms: list[str]) -> list[dict[str, str]]:
    """Put the metric-relevant hits first.

    A qualifier string can occur in an unrelated part of the paper (e.g.
    "the policy is refined" matching a post-processing pattern).  A hit
    whose own context mentions the metric is far more likely to be the
    qualifier the reader is looking for, so rank it first.  The relevance
    test is deliberately LOOSER than the paper's curated metric terms:
    "the fraction of valid molecules after valency correction" is the
    definition of the metric even though it never says "validity".
    """
    def _relevant(h: dict[str, str]) -> int:
        blob = f"{h.get('context', '')} {h.get('match', '')}"
        if any(re.search(t, blob, re.I) for t in metric_terms):
            return 2
        return 1 if METRIC_RELEVANCE.search(blob) else 0

    return sorted(hits, key=lambda h: (-_relevant(h), h.get("file", ""), h.get("line", 0)))


def _site_score(site: dict[str, Any]) -> tuple[int, int, int]:
    """Ranking key for "which site is the headline number?".

    1. carries a percentage value at all (a number a reader would quote)
    2. how many load-bearing quadrants are printed AT the number
    3. how many quadrants (incl. hydrogen convention) are printed AT it
    """
    return (
        1 if site.get("has_percent_value") else 0,
        _qualifier_strength(site),
        _quadrants_populated(site),
    )


def elsewhere_hits(doc: dict[str, str], skip: set[tuple[str, int]]) -> dict[str, list[dict[str, str]]]:
    """Qualifier occurrences anywhere else in the paper (informational)."""
    out: dict[str, list[dict[str, str]]] = {q: [] for q in QUADRANT_ORDER}
    for fname, text in doc.items():
        for lineno, line in enumerate(text.splitlines(), 1):
            if (fname, lineno) in skip:
                continue
            flat = tex_to_text(line)
            if not flat:
                continue
            for quad in QUADRANT_ORDER:
                if len(out[quad]) >= 6:
                    continue
                hits = _find_qualifiers(flat, QUADRANT_PATTERNS[quad])
                for h in hits:
                    if len(out[quad]) >= 6:
                        break
                    out[quad].append({"file": fname, "line": lineno, **h})
    return out


# ---------------------------------------------------------------------------
# Per-paper driver
# ---------------------------------------------------------------------------
def analyse_paper(paper: dict[str, Any], cache_dir: Path, *, offline: bool) -> dict[str, Any]:
    arxiv_id = paper["arxiv_id"]
    tarball, prov = fetch_tarball(arxiv_id, cache_dir, offline=offline)

    row: dict[str, Any] = {
        "key": paper["key"],
        "short": paper["short"],
        "arxiv_id": arxiv_id,
        "title": paper["title"],
        "venue": paper["venue"],
        "metric_label": paper["metric_label"],
        "url": prov["url"],
        "abs_url": prov["abs_url"],
        "retrieved": prov["retrieved"],
        "http_status": prov["http_status"],
        "sha256": prov["sha256"],
        "bytes": prov["bytes"],
        "error": prov["error"],
        "notes": paper.get("notes", ""),
        "metric_terms": paper["metric_terms"],
        "sites": [],
        "headline_site": None,
        "qualifier_summary_at_number": {},
        "qualifiers_elsewhere": {},
        "verdict": "NOT-RETRIEVED",
        "verdict_reason": "",
    }

    if not tarball:
        row["verdict"] = "NOT-RETRIEVED"
        row["verdict_reason"] = (
            f"fetch failed ({prov['error']}); no connectivity/validity number is reported or guessed."
        )
        return row

    try:
        doc = extract_tex_files(tarball)
    except Exception as exc:
        row["retrieved"] = False
        row["error"] = f"tar extraction failed: {type(exc).__name__}: {exc}"
        row["verdict"] = "NOT-RETRIEVED"
        row["verdict_reason"] = row["error"]
        return row

    row["tex_files"] = sorted(doc)

    sites: list[dict[str, Any]] = []
    for fname, text in doc.items():
        for site in split_sites(text):
            classified = classify_site(site, paper["metric_terms"])
            classified["file"] = fname
            sites.append(classified)
    sites.sort(key=lambda s: (s["file"], s["line"]))
    row["sites"] = sites

    metric_sites = [s for s in sites if s.get("is_metric_site")]
    row["n_metric_sites"] = len(metric_sites)

    if not metric_sites:
        row["verdict"] = "NOT-STATED"
        row["verdict_reason"] = (
            "No site in the retrieved source reports a connectivity/validity percentage; "
            "the metric is absent from the paper, so no qualifier can be present."
        )
        return row

    # Headline sites = the metric sites a reader would actually quote a
    # percentage from, ranked by (carries a percentage value, number of
    # load-bearing quadrants AT the number, number of quadrants AT it).
    ranked = sorted(metric_sites, key=_site_score, reverse=True)
    number_sites = [s for s in ranked if s.get("has_percent_value")]
    row["number_site_fallback"] = not number_sites
    chosen = number_sites if number_sites else ranked
    best = chosen[0]

    row["headline_sites"] = [
        {
            "file": s["file"],
            "line": s["line"],
            "kind": s["kind"],
            "has_percent_value": s.get("has_percent_value", False),
            "header_declares_percent": s.get("header_declares_percent", False),
            "unitless_table": s.get("unitless_table", False),
            "caption": s.get("caption", "")[:600],
            "headline_excerpt": s["headline_excerpt"],
            "qualifiers_at_number": {
                q: [h["label"] for h in s["qualifiers_at_number"].get(q, [])] for q in QUADRANT_ORDER
            },
        }
        for s in chosen[:3]
    ]
    row["headline_site"] = row["headline_sites"][0]
    row["qualifier_summary_at_number"] = {
        q: [h["label"] for h in best["qualifiers_at_number"].get(q, [])]
        for q in QUADRANT_ORDER
    }

    # Document-wide qualifier inventory (no skipping) — this is where the
    # qualifier actually lives when it is not at the number.
    row["qualifiers_in_paper"] = {
        q: _rank_hits(elsewhere_hits(doc, set()).get(q, []), paper["metric_terms"])
        for q in QUADRANT_ORDER
    }
    # And the subset that is NOT at the headline site.
    skip = set()
    for s in chosen[:3]:
        for off in range(0, 6):
            skip.add((s["file"], s["line"] + off))
    row["qualifiers_elsewhere"] = {
        q: _rank_hits(elsewhere_hits(doc, skip).get(q, []), paper["metric_terms"])
        for q in QUADRANT_ORDER
    }

    at = row["qualifier_summary_at_number"]
    has_pop = bool(at["population"])
    has_post = bool(at["postproc"])
    has_def = bool(at["definition"])
    in_paper = row["qualifiers_in_paper"]

    if has_pop and has_post and has_def:
        row["verdict"] = "AGREES"
        row["verdict_reason"] = (
            "population + post-processing + definition are all printed AT the number "
            "(in the caption of the table that carries the number)."
        )
    elif any(in_paper.get(q) for q in QUADRANT_ORDER) or has_pop or has_post or has_def:
        row["verdict"] = "DISTINGUISHES"
        row["verdict_reason"] = (
            "the qualifier exists but in a different place or a different form: "
            f"AT the number population={at['population'] or 'none'}, "
            f"post-processing={at['postproc'] or 'none'}, "
            f"definition={at['definition'] or 'none'}."
        )
    else:
        row["verdict"] = "NOT-STATED"
        row["verdict_reason"] = "no population / post-processing / definition qualifier found anywhere."

    # How many metric sites carry NO qualifier at the number?  This is the
    # "bare restatement" count that a reader can be misled by.
    row["sites_without_qualifier_at_number"] = [
        {"file": s["file"], "line": s["line"], "kind": s["kind"], "excerpt": s["headline_excerpt"][:2]}
        for s in metric_sites
        if _qualifier_strength(s) == 0
    ]

    return row


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
HOUSE_FORMAT_STRING = (
    "{metric} {value}% — population: N={N} {population_desc}; "
    "post-processing: {postproc}; denominator: {denominator}; "
    "hydrogen convention: {hydrogens}; definition: {definition}"
)

PROPOSED_HOUSE_FORMAT = (
    "connectivity {v:.2f}% "
    "(single-fragment rate over N={n} decoded samples; "
    "post-processing: {postproc}; denominator: {denominator}; "
    "hydrogens: {hydrogens}; definition: exactly one connected component)"
)


# Wf-A194 (2026-11-02) -- the Wf-A176 and Wf-A176-2 dated CORRECTION
# sections, PRESERVED VERBATIM and now EMITTED by this producer.
#
# BEFORE this wave the producer emitted only the Wf-A183 and Wf-A187 dated
# sections; the Wf-A176 and Wf-A176-2 sections existed ONLY in the published
# `report.md` and were copied forward by `_carry_dated_sections`, which
# returns "" when the target `report.md` does not exist.  A run with a FRESH
# `--out-dir` therefore DROPPED both dated sections -- MEASURED 2026-11-02:
# a fresh `--out-dir` produced 31,087 B / 304 lines against the published
# 38,505 B / 429 lines, a 3-hunk diff in which 127 published-only lines
# (the two Wf-A176 sections) were missing.  A re-run would have ERASED dated
# history (AGENTS.md section 5.5 rule #6).
#
# The two sections below are the published bytes of those two sections,
# VERBATIM, extracted from
# `molmetal/reports/wfa161_frontier_labelling/report.md` on 2026-11-02 (the
# `## Wf-A176 CORRECTION (2026-11-01)` block and the
# `## Wf-A176 CORRECTION 2 (2026-11-01)` block).  Emitting them makes a fresh
# `--out-dir` reproduce every dated block.
WFA176_CORRECTION_SECTION = """\
## Wf-A176 CORRECTION (2026-11-01) — §"Comparability breaks" item 2 is WRONG

**This section is APPEND-ONLY. The superseded rendering above is preserved
VERBATIM per AGENTS.md §5.5 rule #6 and is NOT edited.**

### The superseded sentence (verbatim, line 139 of this file)

> 2. **Post-processing.** The frontier's connectivity rates are computed on the atom-count largest connected component after RDKit sanitization (EF-TALFM) or with no stated fragment handling at all (GEOM-DD, GGFlow). Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale.

### Why it is wrong — three independent defects

1. **OWNERSHIP.** The 92.7-94 % band is **NOT ours**. It is a **frontier
   composite**. The repo's own metric artefact
   (`metrics/by_metric/wfa155_metal_coordination_sphere.json`,
   `comparability_statement.summary`) already says: *"This metric is NOT
   comparable to the published 92.7-94 % frontier band ... Four independent
   breaks separate the two."*
2. **METRIC TYPE.** The sentence compares the band to "the frontier's
   connectivity rates". The band's endpoints are **not connectivity rates
   at all**. MEASURED from
   `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`
   (`band_provenance.endpoint_candidates`):
   - **92.7 %** = GeoLDM's **Validity & Uniq** on QM9, tabulated by HLTF
     (arXiv:2603.27113v1, Table 1(b) row `Val&Uniq (%)`, column `GeoLDM`) —
     *"a per-molecule RDKit-sanitisable AND non-duplicate rate"* over
     10,000 unconditional QM9 samples, `post_processed: false`.
   - **94 %** = HLTF's **PB-Valid** on QM9 (same arXiv:2603.27113v1,
     Table 1(b) row `PB-Valid (%)`, column `HLTF (ours)`) — *"PoseBusters
     validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,
     steric clash, aromatic and double-bond flatness, internal energy)"*.
   - A **third** paper in the same scan (YuelBond, bioRxiv
     10.1101/2025.05.06.652517v1) quotes **92.7 %** as a *"multi-class
     BOND-TYPE classification F1"* on 1,000 coordinate-noised molecules.
3. **PROVENANCE.** *"comes from a DIFFERENT perception branch (Open
   Babel)"* is not supported by any artefact on disk. Every candidate
   endpoint above is a **frontier table cell**; no Open Babel branch is
   named anywhere in the `band_provenance` block.

### The authoritative replacement (verbatim from the repo's own artefact)

`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`,
`band_provenance.headline_finding`:

> THE BAND IS NOT A CONNECTIVITY BAND. Its two endpoints are a
> VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %);
> a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE
> CLASSIFICATION F1. None of the three is a connected-component count. Our
> 8.51 % is a connected-component count on the RAW emitted SMILES. The
> comparison the band invites is therefore not merely apples-to-oranges --
> it is a comparison of two different observables, and the post-processing
> axis is a THIRD, separate axis on top of it.

So the corrected reading of item 2 is:

> **2. Post-processing.** The frontier's connectivity rates are computed on
> the atom-count largest connected component after RDKit sanitization
> (EF-TALFM) or with no stated fragment handling at all (GEOM-DD, GGFlow).
> The published 92.7-94 % band is **NOT ours** and is **not a connectivity
> band**: its candidate endpoints are GeoLDM's Validity & Uniq (92.7 %) and
> HLTF's PB-Valid (94.0 %) on QM9 — a sanitisation-and-duplication rate and
> a six-check pose verdict — while a third paper in the same scan quotes
> 92.7 % as a per-bond-type classification F1. None of the three is a
> connected-component count
> (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`,
> `band_provenance`), so the band cannot be placed on the frontier's
> connectivity scale — while OUR 8.51 % IS a connected-component count on
> the RAW emitted SMILES, and nothing else.

### What is STILL not established (honest-negative, unchanged)

The **exact paper** the repo's band was assembled from is **NOT
established on disk**. Verbatim from the same artefact
(`band_provenance.what_this_does_not_settle`):

> This branch did NOT establish which paper the repo's band was assembled
> from, because the band was assembled by a literature scan whose selection
> rule is not recorded on disk.

### The repair

* **Generator** (`molmetal/scripts/wfa161_frontier_labelling.py`) — repaired
  2026-11-01 (Wf-A176): the module docstring and the §"Comparability
  breaks" item-2 emission now carry the corrected text, with the
  superseded sentence preserved VERBATIM in dated comments. The report is
  **not** re-run, precisely so this file's superseded rendering survives.
* **Reach**: MEASURED 2026-11-01 — the superseded sentence occurs in **0**
  of 9,551 HEAD blobs and **0** of 134 `paper/` files. It has **not**
  reached the paper.
* **CPU-only**, 0 GPU. No metric JSON, no `paper/**` write.
"""

WFA176_CORRECTION2_SECTION = """\
## Wf-A176 CORRECTION 2 (2026-11-01) — the "none of the endpoints is a connected-component count" claim is NOT ESTABLISHED

**This section is APPEND-ONLY. Everything above it is preserved VERBATIM per AGENTS.md §5.5 rule #6 and is NOT edited.**

### What is superseded

The "authoritative replacement" quoted in the section above states:

> None of the three is a connected-component count ... so the band cannot be placed on the frontier's connectivity scale

That claim is TRUE under ONE provenance reading on disk and FALSE under the OTHER, so it cannot be published as settled. The repo holds TWO competing readings of what the 92.7-94 % band is, both in the SAME artefact, `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`:

**Reading 1 — `band_provenance` (lines 51-89).** The band's endpoint candidates are GeoLDM's Validity & Uniq (92.7 %, `post_processed: false`) and HLTF's PB-Valid (94.0 %, `post_processed: false`), with YuelBond's per-bond-type F1 (92.7 %) as a third candidate for the same number. Under this reading neither endpoint is a connected-component count.

**Reading 2 — `sources.EF-TALFM` in the same artefact (lines 360, 378-379), corroborated by the `wfa168_frontier_axis_mapping` branch; EF-TALFM (arXiv:2609.08333v4).** The band mixes EF-TALFM's OWN Table 3(a) cells: the RAW connectivity cell `C. = 92.98 %` (counted BEFORE the largest-component retention) and the post-processed `N. & U. & S. = 94.33 %` companion cell. Verbatim, line 378: *“The '92.7-94 %' band the repo quotes mixes the LOWEST connectivity in that table with a post-processed companion cell (N. & U. & S. = 94.33).”* Under THIS reading the lower endpoint IS a connectivity count.

**Why this matters for the correction above.** The sentence it quoted — *"None of the three is a connected-component count"* — is Reading 1's claim, stated as if it were settled. It is not: the repo's own axis-mapping branch records Reading 2, and under Reading 2 the lower endpoint is a connectivity count. A published sentence must not assert a contested provenance as fact.

### The corrected, reading-safe replacement

> The published 92.7-94 % band is **NOT ours** — it is a FRONTIER composite, and the repo holds **TWO competing provenance readings** for it, neither of which makes it a like-for-like comparison to our number. Reading 1: its endpoints are GeoLDM's Validity & Uniq (92.7 %) and HLTF's PB-Valid (94.0 %) on QM9 — a sanitisation-and-duplication rate and a six-check pose verdict — both carrying `post_processed: false`. Reading 2: it mixes EF-TALFM's OWN Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the largest-component retention) with its post-processed N. & U. & S. companion cell (94.33 %). Under **BOTH** readings the band is external, is **not one homogeneous quantity**, and is **not on our scale** — while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. (The exact paper the band was assembled from is NOT established on disk: it came from a literature scan whose selection rule was not recorded.)

### What is NOT changed by this correction

* The superseded section above stays VERBATIM.
* No number moves: the 92.7 %, 94.0 %, 92.98 %, 94.33 %, 8.51 % are all as measured on disk.
* The claim that the band is external, heterogeneous, and not on our scale is unchanged — it holds under both readings.
* `paper/**`, `metrics/**`, `AGENTS.md`, `TODO/**`, `docs/HONEST_NEGATIVE_CATALOG.md` are NOT touched.
* CPU-only, 0 GPU.
"""


WFA183_HEADING = "## Wf-A183 CORRECTION (2026-11-01)"

# Wf-A183 (2026-11-01) — the dated CORRECTION section this producer EMITS
# into `report.md`.  It exists because the producer used to re-emit
# `report.md` WHOLESALE with NO dated correction section of its own, so a
# re-run ERASED the rule-#6 history (AGENTS.md §5.5 rule #6) that a
# previous run had left in the published artefact — the producer could not
# reproduce its own artefact.  With this section emitted, a re-run
# REPRODUCES the correction instead of erasing it.
#
# NO NEW NUMERALS: every number below (92.7 %, 94.0 %, 92.98 %, 94.33 %,
# 8.51 %, arXiv:2609.08333v4, Table 3(a)) is already on disk — the
# canonical clause and the four invariants are quoted VERBATIM from the
# frontier matrix record.
WFA183_CORRECTION_SECTION = """
---

## Wf-A183 CORRECTION (2026-11-01) — the band's locators are JSON PATHS, and its provenance is NOT settled on disk

**This section is APPEND-ONLY. Every superseded rendering it replaces is preserved
VERBATIM below per AGENTS.md §5.5 rule #6. Nothing above this line is edited.**

### The canonical reading-safe band statement (VERBATIM from the record)

Verbatim from `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, JSON PATH
`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement`:

> The published band's provenance is NOT settled on disk: the frontier post-processing matrix record
> (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under
> `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus
> YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE
> the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even
> agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk.  Under BOTH
> readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a
> connected-component count on the RAW emitted SMILES, and nothing else.

### The four invariants (VERBATIM from the record)

Verbatim from the same JSON PATH, `the_only_claims_that_may_be_published`:

* **(i)** The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is a FRONTIER composite, not our measurement.
* **(ii)** It is NOT one homogeneous quantity: even under Reading A its two endpoints are two different observables (a sanitisation-and-duplication rate and a six-check pose verdict), and under Reading B one endpoint is a raw connectivity count while the other is a post-processed composite.
* **(iii)** It is NOT on our scale.  OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.
* **(iv)** The source cells of the band are NOT settled on disk.

### The corrected locators — JSON PATHS, not line numbers

| Reading | ROBUST locator (JSON PATH of `frontier_matrix.json`) | What it is NOT |
|---|---|---|
| Reading 1 | `band_provenance` | NOT a line range |
| Reading 2 | `sources`.`EF-TALFM` | NOT `frontier_axis_mapping` — that string is NOT a key of the record; it survives only as a STRING VALUE |

### The X1-X4 forbidden-claim list

* **X1 — FORBIDDEN as a settled claim:** "THE BAND IS NOT A CONNECTIVITY BAND" / "None of the three is a connected-component count".  TRUE under Reading 1, FALSE under Reading 2 (whose RAW connectivity cell is `C. = 92.98 %`).  The record's LIVE `band_provenance`.`headline_finding` no longer carries this wording — it is preserved VERBATIM at `band_provenance`.`headline_finding_superseded_2026_11_01`.
* **X2 — FORBIDDEN as a locator:** citing `frontier_axis_mapping` as a JSON key of `frontier_matrix.json`.  It is not a key; use the JSON PATH `sources`.`EF-TALFM`.
* **X3 — FORBIDDEN as a locator:** line-number locators (`:51-89`, `:360, :378-379`).  Line numbers rot as the record gains keys; cite the JSON PATHS `band_provenance` and `sources`.`EF-TALFM`.
* **X4 — FORBIDDEN:** attributing the band to Mol-Metal, or to a "DIFFERENT perception branch (Open Babel)".  The band is EXTERNAL, and no Open Babel branch is named anywhere in the band-provenance record.

### The superseded renderings, preserved VERBATIM

**The producer's module docstring, pre-Wf-A183 (the Reading 1 / Reading 2 locator lines):**

>         Reading 1  `band_provenance` (:51-89): GeoLDM's Validity & Uniq
>                    92.7 % and HLTF's PB-Valid 94.0 %, both with
>                    `post_processed: false`; a third paper quotes 92.7 %
>                    as a per-bond-type classification F1.  Under this
>                    reading no endpoint is a connected-component count.
>         Reading 2  `frontier_axis_mapping` (:360, :378-379), EF-TALFM
>                    arXiv:2609.08333v4: the band mixes EF-TALFM's OWN
>                    Table 3(a) raw connectivity cell (C. = 92.98 %, counted
>                    BEFORE the largest-component retention) with its
>                    post-processed N. & U. & S. companion cell (94.33 %).
>                    Under THIS reading the lower endpoint IS a
>                    connectivity count.

**The producer's emitted §"Comparability breaks" item 2, pre-Wf-A183:**

> Reading 2 (the same file -> `frontier_axis_mapping`, EF-TALFM arXiv:2609.08333v4): it mixes EF-TALFM's OWN Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the retention) with its post-processed N. & U. & S. companion cell (94.33 %).

**The record's superseded `band_provenance`.`headline_finding` — the X1 wording — preserved VERBATIM at `band_provenance`.`headline_finding_superseded_2026_11_01`:**

> THE BAND IS NOT A CONNECTIVITY BAND.  Its two endpoints are a VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %); a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE CLASSIFICATION F1.  None of the three is a connected-component count.  Our 8.51 % is a connected-component count on the RAW emitted SMILES.  The comparison the band invites is therefore not merely apples-to-oranges -- it is a comparison of two different observables, and the post-processing axis is a THIRD, separate axis on top of it.

### What this correction does NOT change

* No number moves: 92.7 %, 94.0 %, 92.98 %, 94.33 %, 8.51 % are the values measured on disk.
* The four invariants above are unchanged — they hold under BOTH readings.
* `paper/**`, `metrics/**`, `registry/**`, `AGENTS.md`, `TODO/**`, `docs/**` are NOT touched.
* CPU-only, 0 GPU.
"""

# Wf-A187 (2026-11-01) -- the CONSUMER-side repair of the band-provenance defect class.
#
# The Wf-A176 CORRECTION section above ends with a subsection headed "### The
# authoritative replacement (verbatim from the repo's own artefact)" that calls its
# content authoritative and quotes the record's X1 rendering VERBATIM.  Two LATER
# dated sections in the SAME file -- Wf-A176 CORRECTION 2 and Wf-A183 CORRECTION --
# already supersede it, but the superseded subsection itself carries no supersession
# marker of its own, so a reader who stops there still meets a REFUTED claim
# presented as authoritative.  This section ADDS that marker.  APPEND-ONLY
# (AGENTS.md §5.5 rule #6): nothing above it is edited or removed.
#
# The section below is EMITTED by this producer (see `render_report`), so a re-run
# REPRODUCES it instead of dropping it.
WFA187_CORRECTION_SECTION = """
---

## Wf-A187 CORRECTION (2026-11-01) — the "authoritative replacement" subsection's X1 rendering is SUPERSEDED

**This section is APPEND-ONLY. Every superseded rendering it names is preserved
VERBATIM below per AGENTS.md §5.5 rule #6. Nothing above this line is edited.**

### The superseded subsection, named

The subsection this correction supersedes is the one headed

> ### The authoritative replacement (verbatim from the repo's own artefact)

which sits inside `## Wf-A176 CORRECTION (2026-11-01) — §"Comparability breaks" item 2 is WRONG`.
It presented itself as *the authoritative replacement* and quoted the record's then-live
`band_provenance`.`headline_finding` VERBATIM:

> THE BAND IS NOT A CONNECTIVITY BAND. Its two endpoints are a
> VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %);
> a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE
> CLASSIFICATION F1. None of the three is a connected-component count. Our
> 8.51 % is a connected-component count on the RAW emitted SMILES. The
> comparison the band invites is therefore not merely apples-to-oranges --
> it is a comparison of two different observables, and the post-processing
> axis is a THIRD, separate axis on top of it.

### It is superseded — by what

That wording is **X1**, and it is **SUPERSEDED**:

* by `## Wf-A176 CORRECTION 2 (2026-11-01)` in this same file;
* by `## Wf-A183 CORRECTION (2026-11-01)` in this same file;
* and by the owning record itself — `band_provenance`.`headline_finding` no longer carries
  this wording (it is preserved VERBATIM at `band_provenance`.`headline_finding_superseded_2026_11_01`).

X1 is TRUE under Reading 1 and FALSE under Reading 2 (whose RAW connectivity cell is
`C. = 92.98 %`), so it must not be published as a settled claim. What this correction
withdraws is the superseded subsection's self-description — the heading that called its
content "the authoritative replacement" — and nothing else.

### The reading-safe replacement (the only form that may be quoted)

Verbatim from `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, JSON PATH
`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement`:

> The published band's provenance is NOT settled on disk: the frontier post-processing matrix record
> (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under
> `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus
> YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE
> the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even
> agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk.  Under BOTH
> readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a
> connected-component count on the RAW emitted SMILES, and nothing else.

### The four invariants (VERBATIM from the record)

Verbatim from the same JSON PATH, `the_only_claims_that_may_be_published`:

* **(i)** The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is a FRONTIER composite, not our measurement.
* **(ii)** It is NOT one homogeneous quantity: even under Reading A its two endpoints are two different observables (a sanitisation-and-duplication rate and a six-check pose verdict), and under Reading B one endpoint is a raw connectivity count while the other is a post-processed composite.
* **(iii)** It is NOT on our scale.  OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.
* **(iv)** The source cells of the band are NOT settled on disk.

(The record's Reading A / Reading B are this file's Reading 1 / Reading 2.)

### The band's provenance is NOT settled on disk

Which frontier cells the "92.7-94 %" band names is NOT settled on disk: the record holds TWO
competing readings of them and they do not even agree on the numerals (92.7/94.0 vs
92.98/94.33). The four invariants above are the only publishable claims.

### What this correction does NOT change

* No number moves: 92.7 %, 94.0 %, 92.98 %, 94.33 %, 8.51 % are the values measured on disk.
* The superseded subsection above stays VERBATIM — this section ADDS a supersession marker for it
  and nothing else (AGENTS.md §5.5 rule #6 — append-only).
* `paper/**`, `metrics/**`, `registry/**`, `AGENTS.md`, `TODO/**`, `docs/**` are NOT touched.
* CPU-only, 0 GPU.
"""

# Wf-A194 (2026-11-02) -- the dated CORRECTION section this producer EMITS
# into `report.md`.  It records (a) the carry-forward repair above, (b) the
# live X3 / X4 rendering that the producer's item 2 no longer emits, and
# (c) the locator staleness that the repair creates for the older dated
# section.  APPEND-ONLY (AGENTS.md section 5.5 rule #6): nothing above it is
# edited or removed.
WFA194_CORRECTION_SECTION = """\
## Wf-A194 CORRECTION (2026-11-02) — every dated block is now emitted by the producer, and the live item 2 no longer carries the refuted X3 / X4 rendering

**This section is APPEND-ONLY. Every superseded rendering it names is preserved
VERBATIM elsewhere in this file per AGENTS.md §5.5 rule #6. Nothing above this
line is edited.**

### 1. What this producer changed (MEASURED, 2026-11-02)

Before this wave this producer emitted ONLY the Wf-A183 and Wf-A187 dated
CORRECTION sections. The Wf-A176 and Wf-A176-2 sections existed ONLY in the
published `report.md` and were copied forward by `_carry_dated_sections`, which
returns "" when the target `report.md` does not exist. A run with a FRESH
`--out-dir` therefore DROPPED both dated sections:

* fresh `--out-dir` (2026-11-02, `--offline`, cache `/tmp/wfa161_arxiv_cache`):
  **31,087 B / 304 lines**;
* published `report.md` before this wave: **38,505 B / 429 lines**;
* `diff -u` between the two: **3 hunks**, of which **127 lines are
  published-only** — the two Wf-A176 sections, and nothing else.

A re-run would therefore have ERASED dated history (AGENTS.md §5.5 rule #6).
`WFA176_CORRECTION_SECTION` and `WFA176_CORRECTION2_SECTION` below hold those
two sections VERBATIM and are EMITTED, so a fresh `--out-dir` reproduces every
dated block.

### 2. The live rendering this closes

The `## Comparability breaks` item 2 of the PREVIOUS published `report.md`
asserted, as a current fact, that the frontier's connectivity rates are
computed on the atom-count largest connected component after RDKit
sanitisation (EF-TALFM), and that the published band comes from a DIFFERENT
perception branch (Open Babel) and a DIFFERENT post-processing. BOTH
renderings are REFUTED on disk:

* **the retention claim** — EF-TALFM counts connectivity BEFORE the
  largest-component retention; its `C.` cell is the reconstructed graph's own
  fragment count;
* **the Open-Babel / "DIFFERENT post-processing" claim** — the band is
  EXTERNAL, it is a FRONTIER composite, and no Open Babel branch is named
  anywhere in the band-provenance record.

This producer's item 2 has emitted the corrected, reading-safe form since
Wf-A176 CORRECTION 1; this wave makes the PUBLISHED file carry it too.

**The superseded item-2 sentence survives VERBATIM** in this file, in the
`## Wf-A176 CORRECTION (2026-11-01)` section, under the sub-heading
`### The superseded sentence (verbatim, line 139 of this file)`. It is NOT
deleted (AGENTS.md §5.5 rule #6).

**LOCATOR STALENESS (MEASURED, 2026-11-02).** That sub-heading's phrase
"line 139 of this file" is a LINE-NUMBER locator and it is now STALE: line 139
of this file carries the CORRECTED item 2, and the superseded sentence stands
in the Wf-A176 CORRECTION section instead. Cite JSON PATHS, never line
numbers: `band_provenance` and `sources`.`EF-TALFM` in
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`. This
dated section does NOT rewrite the older dated section (AGENTS.md §5.5
rule #6 — append-only).

**CROSS-WAVE CONSEQUENCE (MEASURED, 2026-11-02).** Two published artefacts
outside this producer's scope cite the OLD line 139 by line number and quote
the old sentence:
`molmetal/reports/wfa175_frontier_comparability/report.md` and
`molmetal/reports/wfa175_frontier_comparability/sources.json`. They are stale
after this wave and their owners must re-point them.

### 3. The canonical reading-safe band statement (VERBATIM from the record)

Verbatim from `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, JSON PATH
`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement`:

> The published band's provenance is NOT settled on disk: the frontier post-processing matrix record
> (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under
> `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus
> YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE
> the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even
> agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk.  Under BOTH
> readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a
> connected-component count on the RAW emitted SMILES, and nothing else.

### 4. The four invariants — cited BY JSON PATH, not re-quoted

`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`the_only_claims_that_may_be_published`
holds the four publishable invariants. They are quoted VERBATIM in the Wf-A183
and Wf-A187 CORRECTION sections above; they are cited HERE BY PATH so that this
section does not duplicate the band numeral.

### 5. What this correction does NOT change

* No number moves: 92.7 %, 94.0 %, 92.98 %, 94.33 %, 8.51 % are the values measured on disk.
* Every dated block that was in the published file is still in it, byte-identical.
* `paper/**`, `metrics/**`, `registry/**`, `AGENTS.md`, `TODO/**`, `docs/**` are NOT touched.
* CPU-only, 0 GPU.
"""


# Wf-A183 (2026-11-01) — the dated CORRECTION sections a previous run of
# this producer left in `report.md` are carried forward verbatim, so that
# a re-run REPRODUCES the rule-#6 history instead of erasing it.
_CORRECTION_HEADING_RE = re.compile(r"^##\s+Wf-A\d+[^\n]*CORRECTION[^\n]*$", re.MULTILINE)


def _strip_trailing_separators(block: str) -> str:
    """Drop the ``---`` rule and the blank lines that separate a section from the NEXT one.

    Wf-A187 (2026-11-01): the carry-forward used to splice the trailing ``---`` of a
    carried section back into the report, so a re-run produced a DOUBLE ``---`` before
    every carried section.  The separator belongs to the JOIN, not to the section.
    """
    lines = block.rstrip("\n").split("\n")
    while lines and lines[-1].strip() in ("", "---"):
        lines.pop()
    return "\n".join(lines)


def _dated_body(section: str) -> str:
    """The BODY of a dated CORRECTION section constant: the heading and its
    text, WITHOUT the leading ``---`` separator and WITHOUT a trailing newline.

    Wf-A194 (2026-11-02): the producer now emits every dated section itself, so
    the separator between two sections is owned by the EMISSION code rather
    than by the constant.
    """
    lines = section.split("\n")
    while lines and lines[0].strip() == "":
        lines.pop(0)
    if lines and lines[0].strip() == "---":
        lines.pop(0)
        while lines and lines[0].strip() == "":
            lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)


def dated_correction_sections(text: str) -> list[str]:
    """Return the dated CORRECTION sections of ``text``, each verbatim."""
    matches = list(_CORRECTION_HEADING_RE.finditer(text))
    out: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append(_strip_trailing_separators(text[m.start():end]))
    return out


def _carry_dated_sections(report_path: Path, new_text: str) -> str:
    """Return the dated CORRECTION sections of the EXISTING ``report_path``
    that ``new_text`` does not already carry, ready to be inserted before
    the newest section.  Empty string when there is nothing to carry."""
    if not report_path.exists():
        return ""
    try:
        existing = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    # Wf-A187 (2026-11-01): the "already carried" test used to be a plain
    # substring search over the whole new text.  A dated section that NAMES an
    # older section in its prose (the Wf-A187 section quotes the heading of the
    # Wf-A176 CORRECTION section) then looked "already present" and was
    # DROPPED from the carry-forward -- i.e. a re-run erased the very history
    # the carry-forward exists to preserve.  The test is now against the SET of
    # section HEADINGS in the new text, which is what "already carried" means.
    new_headings = {m.group(0).strip() for m in _CORRECTION_HEADING_RE.finditer(new_text)}
    blocks = []
    for sec in dated_correction_sections(existing):
        heading = sec.splitlines()[0].strip()
        if heading and heading not in new_headings:
            blocks.append(sec)
    if not blocks:
        return ""
    # Wf-A187 (2026-11-01): the JOIN reproduces the separator convention the
    # published file already uses between a carried section and the next one --
    # two blank lines, then the rule, then a blank line.  The trailing rule that
    # used to sit inside the carried block itself is stripped by
    # `_strip_trailing_separators`, so it is emitted here exactly once.
    return "\n---\n\n" + "\n\n\n---\n\n".join(blocks) + "\n"


def render_report(rows: list[dict[str, Any]], generated_at: str) -> str:
    lines: list[str] = []
    A = lines.append
    A("# Wf-A161 — Frontier labelling convention for connectivity / validity percentages")
    A("")
    A(f"Generated: {generated_at}")
    A("")
    A(
        "**Question.** When a frontier generative-molecular paper prints a connectivity / "
        "validity percentage, what qualifier does it print *beside* the number — and is the "
        "population (N), the post-processing (largest-component retention? relaxation?) and the "
        "definition (per-molecule? per-pool?) stated AT the number, or only in a distant table "
        "caption / appendix subsection?"
    )
    A("")
    A("**Method.** arXiv e-print LaTeX source fetched per paper (URL, HTTP status and sha256 "
      "recorded in `sources.json`), split into LaTeX float environments (caption included) and "
      "running-text paragraphs, and each metric-bearing site classified for four qualifier "
      "classes (population / post-processing / definition / hydrogen convention). The verdict is "
      "DERIVED from the machine-measured evidence, not hand-written.")
    A("")
    A("## Headline finding")
    A("")
    tally = {v: 0 for v in VERDICTS}
    for r in rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    n_agrees = tally.get("AGREES", 0)
    n_total = len(rows)
    A(
        f"**The frontier convention is NOT 'population + post-processing + definition AT the "
        f"number'.** Of {n_total} frontier papers fetched, {n_agrees} print all three at the "
        f"number, {tally.get('DISTINGUISHES', 0)} qualify it in a DIFFERENT place or a "
        f"DIFFERENT form, and {tally.get('NOT-STATED', 0)} print no such metric at all."
    )
    A("")
    A(
        "Concretely: the population (N) is usually a TABLE-CAPTION fact; the post-processing is "
        "usually an APPENDIX fact; and the DEFINITION is usually a METRICS-SUBSECTION fact. The "
        "one paper that prints all three together does it in the caption of the table that "
        "carries the number (EF-TALFM) — never in the running text. A reader who copies "
        "'92.98 %' out of the body text of EF-TALFM gets a number with no population, no "
        "post-processing and no definition attached to it."
    )
    A("")
    A(
        "**Therefore our house format string must be STRICTER than the frontier convention, not "
        "merely equal to it.** The frontier will not disambiguate our numbers for us; we have to "
        "print the population, the post-processing, the denominator and the definition next to "
        "the number ourselves."
    )
    A("")
    A("## Comparison table")
    A("")
    A("| Paper | Metric reported | N stated AT the number? | Post-processing AT the number? | "
      "Definition AT the number? | H convention AT the number? | Where the qualifier sits | Verdict |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r["retrieved"]:
            A(f"| {r['short']} | — | — | — | — | — | fetch failed | **{r['verdict']}** |")
            continue
        at = r.get("qualifier_summary_at_number", {})
        pop = ", ".join(at.get("population", [])) or "—"
        post = ", ".join(at.get("postproc", [])) or "—"
        dfn = ", ".join(at.get("definition", [])) or "—"
        hyd = ", ".join(at.get("hydrogens", [])) or "—"
        hs = r.get("headline_site")
        where = f"`{hs['file']}:{hs['line']}`" if hs else "—"
        A(
            f"| {r['short']} | {r['metric_label']} | {pop} | {post} | {dfn} | {hyd} | {where} | "
            f"**{r['verdict']}** |"
        )
    A("")
    A("## Per-paper detail")
    A("")
    for r in rows:
        A(f"### {r['short']} — arXiv:{r['arxiv_id']}")
        A("")
        A(f"- title: {r['title']}")
        A(f"- venue: {r['venue']}")
        A(f"- URL: {r['url']}")
        A(f"- retrieved: {'yes' if r['retrieved'] else 'no'}"
          + (f" (sha256 `{r['sha256']}`)" if r.get("sha256") else "")
          + (f" — error: {r['error']}" if r.get("error") else ""))
        A(f"- verdict: **{r['verdict']}** — {r['verdict_reason']}")
        if r.get("number_site_fallback"):
            A("- NOTE: no site in the paper prints a number WITH a percent sign; the metric is "
              "printed as bare numbers under a unit-less column header.")
        for i, hs in enumerate(r.get("headline_sites", [])):
            tag = "headline" if i == 0 else f"runner-up {i}"
            A(f"- {tag} site: `{hs['file']}:{hs['line']}` ({hs['kind']}, "
              f"percent-value={hs['has_percent_value']}, "
              f"header-declares-percent={hs['header_declares_percent']})")
            for ex in hs.get("headline_excerpt", [])[:4]:
                A(f"  - > {ex}")
            if hs.get("caption"):
                A(f"  - caption: > {hs['caption'][:500]}")
        bare = r.get("sites_without_qualifier_at_number", [])
        if bare:
            A(f"- bare restatements (metric site with NO qualifier AT the number): {len(bare)}")
            for b in bare[:5]:
                A(f"  - `{b['file']}:{b['line']}` ({b['kind']})")
        elsewhere = r.get("qualifiers_elsewhere", {})
        in_paper = r.get("qualifiers_in_paper", {})
        for quad in QUADRANT_ORDER:
            hits = elsewhere.get(quad) or []
            where = "outside the headline site"
            if not hits:
                hits = in_paper.get(quad) or []
                where = "elsewhere in the paper"
            if hits:
                h = hits[0]
                A(f"- {quad} {where}: `{h['file']}:{h['line']}` — {h['context'][:200]}")
        A("")
    A("## Proposed house format string (PROPOSAL ONLY — not applied)")
    A("")
    A("```")
    A(PROPOSED_HOUSE_FORMAT)
    A("```")
    A("")
    A(
        "Every slot must be filled from a value that is measured on disk. The four slots that "
        "the frontier papers most often leave to a distant caption — population, post-processing, "
        "denominator and definition — become mandatory text next to the number itself."
    )
    A("")
    A("## Comparability breaks against the frontier convention")
    A("")
    A(
        "1. **Population.** Our headline 8.51 % renders identically for SEVEN different "
        "populations. The frontier papers state N in a caption (EF-TALFM: 10,000; GEOM-DD: 10,000) "
        "or not at all (FlowMol3 states no N for its own models). Our house rule must therefore "
        "print N *at* the number, because the frontier does not reliably do it for us."
    )
    # Wf-A176 (2026-11-01) CORRECTION — the pre-Wf-A176 emission, PRESERVED
    # VERBATIM per AGENTS.md §5.5 rule #6:
    #     "2. **Post-processing.** The frontier's connectivity rates are computed on the "
    #     "atom-count largest connected component after RDKit sanitization (EF-TALFM) or with no "
    #     "stated fragment handling at all (GEOM-DD, GGFlow). Our 92.7-94 % band comes from a "
    #     "DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not "
    #     "on the frontier's scale."
    #
    # WHY it was wrong: the band is NOT ours (it is a frontier composite —
    # see metrics/by_metric/wfa155_metal_coordination_sphere.json,
    # `comparability_statement.summary`), no Open Babel branch is named as
    # its source anywhere in the band-provenance record, and "a DIFFERENT
    # post-processing" is refuted by that record's own
    # `post_processed: false` on BOTH of its endpoint candidates.
    #
    # Wf-A176 (2026-11-01) CORRECTION 1 — the correction-1 emission, PRESERVED
    # VERBATIM per AGENTS.md §5.5 rule #6, and itself SUPERSEDED:
    #     "2. **Post-processing.** The frontier's connectivity rates are computed on the "
    #     "atom-count largest connected component after RDKit sanitization (EF-TALFM) or with no "
    #     "stated fragment handling at all (GEOM-DD, GGFlow). The published 92.7-94 % band is "
    #     "**NOT ours** and is **not a connectivity band**: its candidate endpoints are GeoLDM's "
    #     "Validity & Uniq (92.7 %) and HLTF's PB-Valid (94.0 %) on QM9 — a sanitisation-and-"
    #     "duplication rate and a six-check pose verdict — while a third paper in the same scan "
    #     "quotes 92.7 % as a per-bond-type classification F1. None of the three is a "
    #     "connected-component count "
    #     "(`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, "
    #     "`band_provenance`), so the band cannot be placed on the frontier's connectivity scale — "
    #     "while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing "
    #     "else. (The exact paper the band was assembled from is NOT established on disk: it came "
    #     "from a literature scan whose selection rule was not recorded.)"
    #
    # WHY correction 1 was SUPERSEDED (adversarial verify, Wf-A176): it
    # asserted "None of the three is a connected-component count" as if
    # settled.  The repo holds TWO competing provenance readings for the
    # band (`frontier_matrix.json` -> `band_provenance` :51-89 vs
    # `frontier_axis_mapping` :360/:378-379), and under the SECOND the
    # lower endpoint IS a connectivity count.  Only the invariants that
    # hold under BOTH readings may be published.
    #
    # Wf-A183 (2026-11-01): the LOCATOR quoted in the three lines above is
    # SUPERSEDED — `frontier_axis_mapping` is NOT a JSON key of
    # `frontier_matrix.json` (it survives only as a STRING VALUE) and the
    # line numbers rot.  The ROBUST locators are the JSON PATHS
    # `band_provenance` and `sources`.`EF-TALFM`.
    #
    # Wf-A176 (2026-11-01) CORRECTION 2 ALSO REPAIRS Wf-A168's open defect
    # D1 (MEDIUM): "The frontier's connectivity rates are computed on the
    # atom-count largest connected component after RDKit sanitization
    # (EF-TALFM)" is FALSE — EF-TALFM counts connectivity BEFORE the
    # largest-component retention (`frontier_matrix.json:362`: "Connectivity
    # (C.) is counted BEFORE the retention; every companion cell in the
    # same table is counted AFTER it."); Wf-A168's own defect table lists
    # this exact clause as D1 and its "never" list item 8 says "Never say
    # the frontier's connectivity is computed after largest-component
    # retention."
    #
    # Wf-A183 (2026-11-01) CORRECTION — the pre-Wf-A183 EMISSION is
    # PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #     "Reading 1 (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> "
    #     "`band_provenance`): its endpoints are GeoLDM's Validity & Uniq (92.7 %) and HLTF's "
    #     "PB-Valid (94.0 %) on QM9 — a sanitisation-and-duplication rate and a six-check pose "
    #     "verdict — both carrying `post_processed: false`. Reading 2 (the same file -> "
    #     "`frontier_axis_mapping`, EF-TALFM arXiv:2609.08333v4): it mixes EF-TALFM's OWN "
    #     "Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the retention) with its "
    #     "post-processed N. & U. & S. companion cell (94.33 %)."
    #
    # WHY it was wrong: `frontier_axis_mapping` is NOT a JSON key of the
    # frontier matrix record — it survives only as a STRING VALUE, so
    # pointing a reader at it is a DEFECT (it was PUBLISHED, because this
    # string is emitted into `report.md`).  The ROBUST locators are the
    # JSON PATHS `band_provenance` and `sources`.`EF-TALFM`; LINE-NUMBER
    # locators rot and are not used.
    A(
        "2. **Post-processing.** The frontier's connectivity rates are NOT computed on the "
        "atom-count largest connected component after RDKit sanitization: EF-TALFM counts "
        "connectivity BEFORE the largest-component retention (its `C.` cell is the reconstructed "
        "graph's own fragment count), while its COMPANION cells (S., U. & S., N. & U. & S.) are "
        "counted AFTER the retention + strict sanitisation; GEOM-DD and GGFlow state no fragment "
        "handling at all. The published 92.7-94 % band is **NOT ours** — it is a FRONTIER "
        "composite, and the repo holds **TWO competing provenance readings** for it, neither of "
        "which makes it a like-for-like comparison to our number. Reading 1 "
        "(`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> JSON PATH "
        "`band_provenance`): its endpoints are GeoLDM's Validity & Uniq (92.7 %) and HLTF's "
        "PB-Valid (94.0 %) on QM9 — a sanitisation-and-duplication rate and a six-check pose "
        "verdict — both carrying `post_processed: false`. Reading 2 (the same file -> JSON PATH "
        "`sources`.`EF-TALFM`, EF-TALFM arXiv:2609.08333v4): it mixes EF-TALFM's OWN "
        "Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the retention) with its "
        "post-processed N. & U. & S. companion cell (94.33 %). Under BOTH readings the band is "
        "external, is NOT one homogeneous quantity, and is NOT on our scale — while OUR 8.51 % IS "
        "a connected-component count on the RAW emitted SMILES, and nothing else. (The exact "
        "paper the band was assembled from is NOT established on disk: it came from a literature "
        "scan whose selection rule was not recorded.)"
    )
    A(
        "3. **Pre- vs post-processing.** GGFlow answers the ambiguity by printing TWO columns — "
        "the post-correction validity (100 for every method, hence uninformative) and the "
        "pre-correction validity side by side. That is the single most transferable idea found "
        "in this audit: a headline post-processing number alone hides the raw rate."
    )
    A(
        "4. **Definition.** EF-TALFM splits the denominator rule: connectivity, sanitization, "
        "uniqueness and novelty are computed over ALL generated samples, while the PoseBusters "
        "check rates are computed over the novel/unique/sanitized subset. A bare percentage "
        "without its denominator is ambiguous across the whole frontier."
    )
    A("")
    A("## Honest framing")
    A("")
    A(
        "MEASURED: every excerpt above is verbatim from the retrieved LaTeX source whose sha256 "
        "is recorded in `sources.json`. NO GPU was used and no molecule was decoded. The only "
        "numbers quoted are the ones printed by the fetched papers."
    )
    A(
        "PROJECTED: the house format string above is a proposal. It has NOT been applied to any "
        "metric, JSON or document in this repo."
    )
    A("")
    # Wf-A194 (2026-11-02): EVERY dated CORRECTION section is EMITTED by the
    # producer itself, so a re-run REPRODUCES the rule-#6 history instead of
    # erasing it (AGENTS.md §5.5 rule #6).  Before this wave only the Wf-A183
    # and Wf-A187 sections were emitted; the Wf-A176 and Wf-A176-2 sections were
    # carried forward from the published `report.md` and were DROPPED by a run
    # with a fresh `--out-dir` (MEASURED 2026-11-02: 31,087 B / 304 lines
    # against the published 38,505 B / 429 lines).
    #
    # The separators below are the ones the published file already uses
    # (MEASURED 2026-11-02): two blank lines before the FIRST dated section and
    # before the Wf-A176-2 section, one blank line before the others.
    dated_sections = (
        ("\n\n---\n\n", WFA176_CORRECTION_SECTION),
        ("\n\n\n---\n\n", WFA176_CORRECTION2_SECTION),
        ("\n\n---\n\n", WFA183_CORRECTION_SECTION),
        ("\n\n---\n\n", WFA187_CORRECTION_SECTION),
        ("\n\n---\n\n", WFA194_CORRECTION_SECTION),
    )
    out = "\n".join(lines)
    for sep, sec in dated_sections:
        out += sep + _dated_body(sec)
    return out + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--offline", action="store_true", help="use the cache only; never hit the network")
    ap.add_argument("--print", dest="print_only", action="store_true", help="stdout only; write nothing")
    args = ap.parse_args(list(argv) if argv is not None else None)

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)

    rows: list[dict[str, Any]] = []
    for paper in PAPERS:
        t0 = time.time()
        row = analyse_paper(paper, cache_dir, offline=args.offline)
        row["analysis_seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        print(
            f"[wfa161] {row['short']:<14} retrieved={row['retrieved']!s:<5} "
            f"verdict={row['verdict']:<14} sites={row.get('n_metric_sites', 0)}",
            file=sys.stderr,
        )

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    payload = {
        "artifact": "wfa161_frontier_labelling",
        "generated_at": generated_at,
        "question": (
            "When a frontier generative-molecular paper prints a connectivity / validity "
            "percentage, what qualifier does it print beside it?"
        ),
        "verdict_vocabulary": list(VERDICTS),
        "proposed_house_format": PROPOSED_HOUSE_FORMAT,
        "sources": rows,
    }

    if args.print_only:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    sources_path = out_dir / "sources.json"
    report_path = out_dir / "report.md"

    json_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    report_text = render_report(rows, generated_at)
    # Wf-A183 (2026-11-01): carry the dated CORRECTION sections that a
    # previous run of this producer left in `report.md` forward VERBATIM,
    # so a re-run REPRODUCES the rule-#6 history instead of erasing it
    # (AGENTS.md §5.5 rule #6).  The newly emitted Wf-A183 section stays
    # LAST (it is the newest).
    carried = _carry_dated_sections(report_path, report_text)
    if carried:
        marker = "\n---\n\n" + WFA183_HEADING
        idx = report_text.rfind(marker)
        if idx != -1:
            report_text = report_text[:idx] + carried + report_text[idx:]

    if atomic_write_file is not None:
        atomic_write_file(sources_path, json_text)
        atomic_write_file(report_path, report_text)
    else:  # pragma: no cover
        sources_path.write_text(json_text, encoding="utf-8")
        report_path.write_text(report_text, encoding="utf-8")

    print(f"[wfa161] wrote {sources_path}", file=sys.stderr)
    print(f"[wfa161] wrote {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
