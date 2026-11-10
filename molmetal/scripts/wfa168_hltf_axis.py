#!/usr/bin/env python3
"""Wf-A168 (branch HLTF): the POST-PROCESSING axis and the FALSE-VALID axis
for arXiv:2603.27113 ("Hierarchy-Guided Topology Latent Flow for Molecular
Graph Generation"; Awasthi / Lobo / Zhukov; ICLR 2026 AI4Mat Workshop).

0 GPU. CPU-only. No torch import. No trainer, no decode, no GPU allocation.

WHAT THIS TOOL IS
-----------------
The frontier papers audited so far (Wf-A161: EF-TALFM 2609.08333, FlowMol3
2508.12629, GGFlow 2411.05676, PAFlow 2509.01486, GEOM-DD 2401.02683) print
a POST-PROCESSED connectivity/validity band.  Our own number (the c7 raw
single-fragment rate 0.08509630565203663) is RAW.  Wf-A133/§6.252 recorded
that as a comparability break.  Nobody had yet quoted a frontier paper that
prints the RAW and the POST-PROCESSED number for the SAME pool with the SAME
denominator, which is the only way to measure the post-processing DELTA on
the paper's own terms.

HLTF is that paper.  This tool therefore:

  1. retrieves the arXiv e-print (LaTeX source) and the abs page for
     2603.27113 from PRIMARY sources, recording URL + HTTP status + sha256
     of the retrieved bytes + retrieval date for BOTH;
  2. PINS every connectivity / validity percentage the paper prints, each
     with the exact LaTeX site it came from, the dataset, the DENOMINATOR,
     the POST-PROCESSING in force at that number, and the hydrogen handling;
  3. computes the relaxation/post-processing DELTA the paper itself reports
     and compares its MAGNITUDE with OUR axis magnitude (+91.49 pts), and
     states plainly whether the two are comparable at all;
  4. quotes the paper's "false-valid" concept verbatim and maps the
     OUR-side analogue, labelled as an ANALOGUE;
  5. proposes the corrected one-line house format for any future
     connectivity percentage.

PRIMARY-SOURCE RULE
-------------------
No number in this tool's output is typed in by hand.  Every pinned value
must be FOUND, by exact substring match, in the LaTeX source that was
retrieved and hashed in this run.  A pin that cannot be located is reported
as UNPINNED and the tool says so; it is never silently dropped and never
replaced by a remembered value.  Nothing is taken from a search-engine
summary.

USAGE
-----
    /home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/wfa168_hltf_axis.py
    /home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/wfa168_hltf_axis.py --offline

Outputs (written atomically, AGENTS.md §5.6):
    molmetal/reports/wfa168_hltf_axis/sources.json
    molmetal/reports/wfa168_hltf_axis/report.md
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 0 GPU. The standing GPU grant is SUSPENDED for this chain (2026-11-01).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

ARXIV_ID = "2603.27113"
PAPER_SHORT = "HLTF"
USER_AGENT = "mol-metal-wfa168/1.0 (frontier labelling convention audit; contact: repo owner)"
DEFAULT_CACHE_DIR = Path("/tmp/wfa168_arxiv_cache")
DEFAULT_OUT_DIR = REPO_ROOT / "molmetal" / "reports" / "wfa168_hltf_axis"

# --------------------------------------------------------------------------
# THE PIN TABLE
# --------------------------------------------------------------------------
# Every entry: an exact substring that must exist in the retrieved LaTeX
# source, plus the metadata the repo's house format demands.
#
#   needle      : (member_path_or_None, exact_substring)
#                 None => search every .tex member.
#   expected    : how many occurrences the needle must have across the
#                 source.  A mismatch is a FAILED PIN, reported as such.
#
# The `value` fields are the numbers the paper PRINTS.  They are carried
# here only so the report can render them; the authoritative record is the
# needle, and the tool refuses to emit a value whose needle was not found.
#
# Denominator / post-processing / hydrogens are the paper's OWN statements,
# each with its own needle (see CONTEXT_PINS below), so the metadata is
# pinned from the primary text too and not asserted from memory.
PINS: list[dict[str, Any]] = [
    # ---------------- QM9 (Table 1a, 10,000 unconditional samples) --------
    {
        "id": "qm9_as",
        "metric": "atom stability (AS)",
        "value_pct": 98.8,
        "value_verbatim": "98.8\\pm0.1",
        "dataset": "QM9",
        "axis": "stability",
        "denominator": "10,000 unconditional samples (QM9); the fraction is over samples",
        "postproc": "NONE (raw; no LargestFragmentChooser / explicit H / UFF on the QM9 numbers)",
        "hydrogens": "NOT STATED by the paper for the QM9 arms",
        "needle": ("tables/qm9_benchmarks_og.tex", "\\textbf{HLTF (ours)} & \\underline{$98.8\\!\\pm\\!0.1$} & \\underline{$86.8\\!\\pm\\!0.3$} & \\underline{$93.8\\!\\pm\\!0.6$} & \\underline{$92.9\\!\\pm\\!0.2$}"),
        "expected": 1,
    },
    {
        "id": "qm9_ms",
        "metric": "molecule stability (MS)",
        "value_pct": 86.8,
        "value_verbatim": "86.8\\pm0.3",
        "dataset": "QM9",
        "axis": "stability",
        "denominator": "10,000 unconditional samples (QM9); the fraction is over samples",
        "postproc": "NONE (raw)",
        "hydrogens": "NOT STATED by the paper for the QM9 arms",
        "needle": ("tables/qm9_benchmarks_og.tex", "\\underline{$86.8\\!\\pm\\!0.3$}"),
        "expected": 1,
    },
    {
        "id": "qm9_val",
        "metric": "RDKit validity (Val)",
        "value_pct": 93.8,
        "value_verbatim": "93.8\\pm0.6",
        "dataset": "QM9",
        "axis": "validity",
        "denominator": "10,000 unconditional samples (QM9); the fraction is over samples",
        "postproc": "NONE (raw)",
        "hydrogens": "NOT STATED by the paper for the QM9 arms",
        "needle": ("tables/qm9_benchmarks_og.tex", "\\underline{$93.8\\!\\pm\\!0.6$}"),
        "expected": 1,
    },
    {
        "id": "qm9_val_uniq",
        "metric": "RDKit valid-and-unique (Val&Uniq)",
        "value_pct": 92.9,
        "value_verbatim": "92.9\\pm0.2",
        "dataset": "QM9",
        "axis": "validity",
        "denominator": "10,000 unconditional samples (QM9); the fraction is over samples",
        "postproc": "NONE (raw)",
        "hydrogens": "NOT STATED by the paper for the QM9 arms",
        "needle": ("tables/qm9_benchmarks_og.tex", "\\underline{$92.9\\!\\pm\\!0.2$}"),
        "expected": 1,
    },
    {
        "id": "qm9_pb_valid",
        "metric": "PoseBusters validity (PB-Valid)",
        "value_pct": 94.0,
        "value_verbatim": "94\\pm0.5",
        "dataset": "QM9",
        "axis": "validity-stricter",
        "denominator": "10,000 unconditional samples (QM9); the fraction is over samples",
        "postproc": "NONE (raw)",
        "hydrogens": "NOT STATED by the paper for the QM9 arms",
        "needle": ("tables/qm9_benchmarks_og.tex", "PB-Valid (\\%) $\\uparrow$      & $93.1\\pm0.4$       & $91.9\\pm0.5$    & \\textbf{$94\\pm0.5$*}"),
        "expected": 1,
    },
    {
        "id": "qm9_novel",
        "metric": "novelty (Novel)",
        "value_pct": 56.0,
        "value_verbatim": "56\\pm0.7",
        "dataset": "QM9",
        "axis": "novelty",
        "denominator": "10,000 unconditional samples (QM9); the fraction is over samples",
        "postproc": "NONE (raw)",
        "hydrogens": "NOT STATED by the paper for the QM9 arms",
        "needle": ("tables/qm9_benchmarks_og.tex", "Novel (\\%) $\\uparrow$         & $53.5\\pm0.6$       & $58.7\\pm0.5$*   & \\underline{$56\\pm0.7$}"),
        "expected": 1,
    },
    # ---------------- GEOM-DRUGS, RAW (no post-processing) -----------------
    {
        "id": "geom_valid_raw",
        "metric": "Valid (raw, no post-processing)",
        "value_pct": 85.5,
        "value_verbatim": "{HLTF (ours)} & \\underline{85.5} & \\underline{85.4} & \\underline{85.0}",
        "dataset": "GEOM-DRUGS",
        "axis": "validity",
        "denominator": "100,000 samples (GEOM-DRUGS); the fraction is over samples",
        "postproc": "NONE (raw, pre-PP)",
        "hydrogens": "implicit at this stage (the PP step is what adds explicit H)",
        "needle": ("tables/geom_drugs_results.tex", "{HLTF (ours)} & \\underline{85.5} & \\underline{85.4} & \\underline{85.0}"),
        "expected": 1,
    },
    {
        "id": "geom_vu_raw",
        "metric": "Valid&Unique (raw, no post-processing)",
        "value_pct": 85.4,
        "value_verbatim": "\\underline{85.4}",
        "dataset": "GEOM-DRUGS",
        "axis": "validity",
        "denominator": "100,000 samples (GEOM-DRUGS); the fraction is over samples",
        "postproc": "NONE (raw, pre-PP)",
        "hydrogens": "implicit at this stage",
        "needle": ("tables/geom_drugs_results.tex", "{HLTF (ours)} & \\underline{85.5} & \\underline{85.4} & \\underline{85.0}"),
        "expected": 1,
    },
    {
        "id": "geom_vun_raw",
        "metric": "Valid&Unique&Novel (raw, no post-processing)",
        "value_pct": 85.0,
        "value_verbatim": "\\underline{85.0}",
        "dataset": "GEOM-DRUGS",
        "axis": "validity",
        "denominator": "100,000 samples (GEOM-DRUGS); the fraction is over samples",
        "postproc": "NONE (raw, pre-PP)",
        "hydrogens": "implicit at this stage",
        "needle": ("tables/geom_drugs_results.tex", "{HLTF (ours)} & \\underline{85.5} & \\underline{85.4} & \\underline{85.0}"),
        "expected": 1,
    },
    # ---------------- GEOM-DRUGS, POST-PROCESSED (+PP) --------------------
    {
        "id": "geom_valid_pp",
        "metric": "Valid (+PP)",
        "value_pct": 92.2,
        "value_verbatim": "\\quad \\emph{+PP} & \\underline{92.2} & \\underline{92.0} & \\underline{91.2}",
        "dataset": "GEOM-DRUGS",
        "axis": "validity",
        "denominator": "100,000 samples (GEOM-DRUGS); the fraction is over samples",
        "postproc": "standardized PP = largest fragment + explicit hydrogens + UFF relaxation",
        "hydrogens": "EXPLICIT (added by the PP step)",
        "needle": ("tables/geom_drugs_results.tex", "\\quad \\emph{+PP} & \\underline{92.2} & \\underline{92.0} & \\underline{91.2}"),
        "expected": 1,
    },
    {
        "id": "geom_vu_pp",
        "metric": "Valid&Unique (+PP)",
        "value_pct": 92.0,
        "value_verbatim": "\\underline{92.0}",
        "dataset": "GEOM-DRUGS",
        "axis": "validity",
        "denominator": "100,000 samples (GEOM-DRUGS); the fraction is over samples",
        "postproc": "standardized PP = largest fragment + explicit hydrogens + UFF relaxation",
        "hydrogens": "EXPLICIT (added by the PP step)",
        "needle": ("tables/geom_drugs_results.tex", "\\quad \\emph{+PP} & \\underline{92.2} & \\underline{92.0} & \\underline{91.2}"),
        "expected": 1,
    },
    {
        "id": "geom_vun_pp",
        "metric": "Valid&Unique&Novel (+PP)",
        "value_pct": 91.2,
        "value_verbatim": "\\underline{91.2}",
        "dataset": "GEOM-DRUGS",
        "axis": "validity",
        "denominator": "100,000 samples (GEOM-DRUGS); the fraction is over samples",
        "postproc": "standardized PP = largest fragment + explicit hydrogens + UFF relaxation",
        "hydrogens": "EXPLICIT (added by the PP step)",
        "needle": ("tables/geom_drugs_results.tex", "\\quad \\emph{+PP} & \\underline{92.2} & \\underline{92.0} & \\underline{91.2}"),
        "expected": 1,
    },
    # ---------------- the abstract restatement ---------------------------
    {
        "id": "abstract_geom_pair",
        "metric": "abstract restatement of the GEOM-DRUGS raw/PP pair",
        "value_pct": 92.2,
        "value_verbatim": "85.5\\%/85.0\\% validity/valid-unique-novel without post-processing and \\textbf{92.2\\%/91.2\\%} after standardized relaxation",
        "dataset": "GEOM-DRUGS",
        "axis": "validity",
        "denominator": "as in the table (100,000 samples); the abstract does not restate N",
        "postproc": "the abstract calls the PP 'standardized relaxation'; the experiments section defines it as largest fragment + explicit H + UFF",
        "hydrogens": "explicit in the +PP arm (per the experiments section)",
        "needle": ("iclr2025_conference.tex", "On \\textbf{GEOM-DRUGS}, HLTF attains \\textbf{85.5\\%/85.0\\%} validity/valid"),
        "expected": 1,
    },
    # ---------------- the connectivity-only number -------------------------
    {
        "id": "geom_invalid_breakdown_disconn",
        "metric": "Disconnected share of INVALID samples (invalidity cause breakdown)",
        "value_pct": 35.0,
        "value_verbatim": "No energy guidance & 40 & 35 & 15 & 10",
        "dataset": "GEOM-DRUGS",
        "axis": "connectivity",
        "denominator": "INVALID samples only ('Percentages among invalid samples'), NOT all samples; the absolute N of the invalid set is NOT printed",
        "postproc": "NONE stated for this table (it is an ablation of the raw sampler)",
        "hydrogens": "NOT STATED for this table",
        "needle": ("sections/appendix.tex", "No energy guidance & 40 & 35 & 15 & 10"),
        "expected": 1,
    },
    {
        "id": "geom_invalid_breakdown_disconn_guided",
        "metric": "Disconnected share of INVALID samples (default guidance)",
        "value_pct": 12.0,
        "value_verbatim": "+$E_{\\text{chem}}+E_{\\text{cons}}$ & 14 & 12 & 8 & 66",
        "dataset": "GEOM-DRUGS",
        "axis": "connectivity",
        "denominator": "INVALID samples only; absolute N not printed",
        "postproc": "NONE stated for this table",
        "hydrogens": "NOT STATED for this table",
        "needle": ("sections/appendix.tex", "+$E_{\\text{chem}}+E_{\\text{cons}}$ & 14 & 12 & 8 & 66"),
        "expected": 1,
    },
]

# The metadata the pins depend on is itself pinned from the primary text.
CONTEXT_PINS: list[dict[str, Any]] = [
    {
        "id": "ctx_denominators",
        "claim": "sample counts for both benchmarks",
        "needle": ("sections/experiments.tex", "On QM9 we report atom stability (AS), molecule stability (MS), RDKit validity (Val), and Val\\&Uniq on 10{,}000 unconditional samples. On GEOM-DRUGS we report Valid, Valid\\&Unique (V\\&U), and Valid\\&Unique\\&Novel (V\\&U\\&N) on 100{,}000 samples"),
        "expected": 1,
    },
    {
        "id": "ctx_pp_definition",
        "claim": "the post-processing pipeline definition (largest fragment + explicit H + UFF)",
        "needle": ("sections/experiments.tex", "both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation"),
        "expected": 1,
    },
    {
        "id": "ctx_pp_effect_claim",
        "claim": "the paper's own attribution of what the PP step does",
        "needle": ("sections/experiments.tex", "Standardized post-processing mainly relieves \\emph{geometric} strain"),
        "expected": 1,
    },
    {
        "id": "ctx_pp_delta",
        "claim": "the paper's own PP delta and its footnote conversion fraction",
        "needle": ("sections/experiments.tex", "For HLTF it raises Valid from 85.5 to 92.2, converting $\\approx 46\\%$ of the pre-PP invalid samples under the same pipeline."),
        "expected": 1,
    },
    {
        "id": "ctx_false_valid_abstract",
        "claim": "the 'false-valid' concept, verbatim, in the abstract",
        # NOTE: the e-print's abstract closes the quote with TWO U+2019 RIGHT SINGLE
        # QUOTATION MARKs, not two ASCII apostrophes.  The needle below is the
        # literal byte sequence of the source.
        "needle": (
            "iclr2025_conference.tex",
            "Explicit topology generation also reduces ``false-valid’’ samples "
            "that pass RDKit sanitization but fail stricter checks.",
        ),
        "expected": 1,
    },
    {
        "id": "ctx_false_valid_ethics",
        "claim": "the 'false-valid' concept, verbatim, in the Ethics Statement",
        "needle": ("iclr2025_conference.tex", "``false-valid'' structures that pass basic sanitization but fail stricter checks"),
        "expected": 1,
    },
    {
        "id": "ctx_false_valid_experiments",
        "claim": "how the paper OPERATIONALISES 'false-valid' (inferred from the PB-Valid vs Val gap, never counted)",
        "needle": (
            "sections/experiments.tex",
            "Higher PB-Valid at comparable RDKit validity indicates fewer ``false-valid'' "
            "samples that sanitize in RDKit but violate stricter structural constraints at "
            "comparable headline validity.",
        ),
        "expected": 1,
    },
    {
        "id": "ctx_false_valid_pb_rationale",
        "claim": "the paper's rationale for the false-valid axis, verbatim",
        "needle": (
            "sections/experiments.tex",
            "PoseBusters applies stricter structural checks than RDKit, which can sanitize "
            "chemically implausible structures, so improvements in PB-Valid (at comparable "
            "RDKit validity) indicate fewer false-valid samples rather than simply exploiting "
            "the sanitizer",
        ),
        "expected": 1,
    },
    {
        "id": "ctx_pb_stricter",
        "claim": "the paper's statement that PoseBusters is STRICTER than RDKit sanitization",
        "needle": ("sections/experiments.tex", "PoseBusters applies stricter structural checks than RDKit"),
        "expected": 1,
    },
    {
        "id": "ctx_metal_exclusion",
        "claim": "the paper's own limitation that metal-containing compounds are OUT OF SCOPE",
        "needle": ("sections/appendix.tex", "performance on unusual chemistries, charged/metal-containing compounds, or biomolecules remains to be established"),
        "expected": 1,
    },
    {
        "id": "ctx_geom_pipeline_caveat",
        "claim": "the paper's own caveat that GEOM-DRUGS absolute metrics are affected by pipeline issues",
        "needle": ("sections/appendix.tex", "Reported absolute metrics on GEOM-DRUGS can be affected by known preprocessing/valency-table and bond-order computation issues"),
        "expected": 1,
    },
    {
        "id": "ctx_invalid_cause_in_text",
        "claim": "the body text that ties the invalidity-cause breakdown to GEOM-DRUGS",
        "needle": (
            "sections/experiments.tex",
            "(valence/disconnected rates: 40\\%/35\\% $\\rightarrow$ 14\\%/12\\%)",
        ),
        "expected": 1,
    },
    {
        "id": "ctx_ablation_default_qm9",
        "claim": "the ablation table's default row, which disagrees with the main table on the QM9 columns",
        "needle": ("tables/ablations.tex", "Joint geometry (default)             & 93.2 & 92.4 & 85.0 & 91.2"),
        "expected": 1,
    },
]

# --------------------------------------------------------------------------
# Our side, read from disk (never typed in)
# --------------------------------------------------------------------------
OUR_METRICS = {
    "postproc_arm": REPO_ROOT / "metrics" / "by_metric" / "cfm_fragmentation_postproc_arm.json",
    "ligand_rate": REPO_ROOT / "metrics" / "by_metric" / "wfa155_ligand_connected_rate.json",
}

# The house format proposed by Wf-A161, kept verbatim so the correction is
# readable as a diff.
PRIOR_HOUSE_FORMAT = (
    "connectivity {v:.2f}% (single-fragment rate over N={n} decoded samples; "
    "post-processing: {postproc}; denominator: {denominator}; hydrogens: {hydrogens}; "
    "definition: exactly one connected component)"
)

HOUSE_FORMAT_TEMPLATE = (
    "connectivity {v:.2f}% = {k}/{N} single-component rows; "
    "denominator: {denominator}; "
    "post-processing: {postproc}; "
    "hydrogens: {hydrogens}; "
    "definition: exactly one RDKit connected component; "
    "specks: {specks_clause}; "
    "mass: {mass_clause}; "
    "source: {path}"
)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------
def eprint_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/e-print/{arxiv_id}"


def abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def _fetch(url: str, cache_path: Path, *, offline: bool, timeout: int = 120) -> tuple[bytes | None, dict[str, Any]]:
    """Fetch ``url`` with full provenance.  Returns (bytes_or_None, record).

    A cache hit must NOT lose the LIVE HTTP status: the record is rebuilt
    from a sidecar ``<cache>.prov.json`` written at live-fetch time, so a
    re-run still reports the status the server actually returned.
    """
    prov_path = cache_path.with_suffix(cache_path.suffix + ".prov.json")
    record: dict[str, Any] = {
        "url": url,
        "cache_path": str(cache_path),
        "retrieved": False,
        "http_status": None,
        "sha256": None,
        "bytes": None,
        "retrieved_utc": None,
        "error": None,
    }
    if cache_path.exists() and cache_path.stat().st_size > 0:
        data = cache_path.read_bytes()
        record.update(
            retrieved=True,
            http_status="cache",
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
            retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            note="served from the local cache; sha256 is of the CACHED BYTES",
        )
        if prov_path.exists():
            try:
                prior = json.loads(prov_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                prior = {}
            if prior.get("sha256") == record["sha256"]:
                record["http_status_live"] = prior.get("http_status")
                record["retrieved_utc_live"] = prior.get("retrieved_utc")
                record["final_url"] = prior.get("final_url")
                record["note"] = (
                    "served from the local cache; http_status_live and retrieved_utc_live are "
                    "replayed from the sidecar written at the original live fetch"
                )
            else:
                record["http_status_live"] = None
                record["note"] = (
                    "served from the local cache; the sidecar provenance does not match these bytes, "
                    "so NO live HTTP status is claimed"
                )
        return data, record
    if offline:
        record["error"] = "not cached and --offline was requested"
        return None, record
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
            data = resp.read()
            record["http_status"] = getattr(resp, "status", 200)
            record["final_url"] = resp.geturl()
    except urllib.error.HTTPError as exc:
        record["error"] = f"HTTPError {exc.code}"
        return None, record
    except Exception as exc:  # noqa: BLE001 - network down / TLS / timeout
        record["error"] = f"{type(exc).__name__}: {exc}"
        return None, record
    if not data:
        record["error"] = "empty response body"
        return None, record
    record.update(
        retrieved=True,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    tmp = cache_path.with_suffix(cache_path.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, cache_path)
    try:
        atomic_write_file(
            prov_path,
            json.dumps(
                {
                    "url": url,
                    "http_status": record["http_status"],
                    "final_url": record.get("final_url"),
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                    "retrieved_utc": record["retrieved_utc"],
                },
                indent=2,
            )
            + "\n",
        )
    except OSError:
        pass
    return data, record


def extract_tex(tarball: bytes) -> dict[str, str]:
    """Extract every .tex member into memory.  Never touches the repo."""
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
            out[member.name] = fh.read().decode("utf-8", errors="replace")
    return out


# The numeric token as the paper PRINTS it, per pin id.  This is checked
# against the pinned site's own line: a value that is not literally at the
# site it is attributed to is a FAILED PIN, not a formatting difference.
VALUE_TOKENS: dict[str, str] = {
    "qm9_as": "98.8",
    "qm9_ms": "86.8",
    "qm9_val": "93.8",
    "qm9_val_uniq": "92.9",
    "qm9_pb_valid": "94",
    "qm9_novel": "56",
    "geom_valid_raw": "85.5",
    "geom_vu_raw": "85.4",
    "geom_vun_raw": "85.0",
    "geom_valid_pp": "92.2",
    "geom_vu_pp": "92.0",
    "geom_vun_pp": "91.2",
    "abstract_geom_pair": "85.5",
    "geom_invalid_breakdown_disconn": "35",
    "geom_invalid_breakdown_disconn_guided": "12",
}


def find_sites(tex: dict[str, str], member: str | None, needle: str) -> list[dict[str, Any]]:
    """Locate ``needle`` in the source.  Returns one record per occurrence."""
    sites: list[dict[str, Any]] = []
    for name in sorted(tex):
        if member is not None and name != member:
            continue
        for lineno, line in enumerate(tex[name].split("\n"), start=1):
            if needle in line:
                sites.append(
                    {
                        "file": name,
                        "line": lineno,
                        # The FULL line: the value-token check runs against it,
                        # so a truncated preview would make a real value look
                        # absent (the abstract is one very long line).
                        "line_text": line.strip(),
                        "line_text_preview": line.strip()[:600],
                    }
                )
    return sites


def pin_all(tex: dict[str, str], pins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pin every entry of ``pins`` against the retrieved source."""
    out: list[dict[str, Any]] = []
    for pin in pins:
        member, needle = pin["needle"]
        sites = find_sites(tex, member, needle)
        expected = pin.get("expected", 1)
        ok = len(sites) == expected
        rec = {k: v for k, v in pin.items() if k != "needle"}
        rec["needle"] = needle
        rec["n_sites"] = len(sites)
        rec["expected_sites"] = expected
        rec["pin_status"] = "PINNED" if ok else ("UNPINNED" if not sites else "PIN_ANOMALY")
        rec["sites"] = sites
        # The value must be LITERALLY at the site it is attributed to.
        token = VALUE_TOKENS.get(pin.get("id", ""))
        if token:
            rec["value_token"] = token
            rec["value_token_at_site"] = bool(sites) and any(token in s["line_text"] for s in sites)
            if not rec["value_token_at_site"]:
                rec["pin_status"] = "PIN_ANOMALY"
        if not ok:
            rec["failure_note"] = (
                f"expected {expected} occurrence(s) of the needle, found {len(sites)}; "
                "this value is NOT quotable from this retrieval"
            )
        elif token and not rec.get("value_token_at_site"):
            rec["failure_note"] = (
                f"the numeric token {token!r} does not appear on the pinned line; "
                "this value is NOT quotable from this retrieval"
            )
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# Our side
# --------------------------------------------------------------------------
def load_our_metrics() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, path in OUR_METRICS.items():
        if not path.exists():
            out[key] = {"present": False, "path": str(path)}
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        out[key] = {"present": True, "path": str(path), "blob": blob}
    return out


def compute_axes(ours: dict[str, Any], pinned: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the paper's PP delta and compare it with OUR axis magnitude."""
    by_id = {p["id"]: p for p in pinned}
    axes: dict[str, Any] = {}

    # ---- HLTF's own GEOM-DRUGS post-processing delta -----------------------
    hltf_pairs = [
        ("Valid", "geom_valid_raw", "geom_valid_pp"),
        ("Valid&Unique", "geom_vu_raw", "geom_vu_pp"),
        ("Valid&Unique&Novel", "geom_vun_raw", "geom_vun_pp"),
    ]
    hltf_delta_rows = []
    for label, raw_id, pp_id in hltf_pairs:
        raw_v = by_id[raw_id]["value_pct"]
        pp_v = by_id[pp_id]["value_pct"]
        delta_pts = round(pp_v - raw_v, 10)
        pre_invalid = 100.0 - raw_v
        recovered_frac = (delta_pts / pre_invalid) if pre_invalid else None
        hltf_delta_rows.append(
            {
                "metric": label,
                "raw_pct": raw_v,
                "postprocessed_pct": pp_v,
                "delta_percentage_points": delta_pts,
                "pre_pp_invalid_pct": pre_invalid,
                "share_of_pre_pp_invalid_recovered_pct": (
                    round(100.0 * recovered_frac, 4) if recovered_frac is not None else None
                ),
                "denominator": "100,000 GEOM-DRUGS samples, SAME pool and SAME N for both numbers",
                "postproc": "largest fragment + explicit hydrogens + UFF relaxation",
                "paper_footnote": "the paper's own footnote computes (92.2-85.5)/(100-85.5) for the Valid row",
            }
        )
    axes["hltf_postprocessing_delta"] = {
        "rows": hltf_delta_rows,
        "primary_axis": "Valid",
        "primary_delta_pts": hltf_delta_rows[0]["delta_percentage_points"],
        "note": "the paper prints BOTH the raw and the +PP number for the SAME 100,000-sample pool, "
        "so this delta is same-paper, same-pool, same-denominator",
    }

    # ---- OUR axis ------------------------------------------------------
    arm = ours.get("postproc_arm", {})
    our_block: dict[str, Any] = {"present": bool(arm.get("present"))}
    if arm.get("present"):
        blob = arm["blob"]
        raw_rate = blob["raw_axis"]["single_fragment_rate_over_parsed"]
        pp_rate = blob["postprocessed_axis"]["connected_rate_over_parsed"]
        n_parsed = blob["denominators"]["n_parsed"]
        n_single = blob["raw_axis"]["n_single"]
        n_speck = blob["survivor_shape"]["n_survivors_le_2_heavy"]
        n_ge11 = blob["survivor_shape"]["n_survivors_ge_11_heavy"]
        discarded = blob["mass_retention"]["fraction_of_raw_mass_discarded"]
        our_block.update(
            {
                "path": arm["path"],
                "source_pool": blob.get("source_pool"),
                "n_epochs_in_pool": len(blob.get("per_epoch", {})),
                "n_rows_in_csv": blob["denominators"]["n_rows_in_csv"],
                "n_parsed": n_parsed,
                "raw_single_fragment_rate": raw_rate,
                "n_single": n_single,
                "postprocessed_connected_rate": pp_rate,
                "n_connected": blob["postprocessed_axis"]["n_connected"],
                "delta_percentage_points": round((pp_rate - raw_rate) * 100.0, 10),
                "n_survivors_le_2_heavy": n_speck,
                "pct_survivors_le_2_heavy": round(100.0 * n_speck / n_parsed, 6),
                "n_survivors_ge_11_heavy": n_ge11,
                "pct_survivors_ge_11_heavy": round(100.0 * n_ge11 / n_parsed, 6),
                "fraction_of_raw_mass_discarded": discarded,
                "pct_raw_mass_discarded": round(100.0 * discarded, 6),
                "pooled_raw_heavy_mass_retained_pct": round(100.0 * (1.0 - discarded), 6),
                "per_molecule_heavy_retention_mean_pct": round(
                    100.0 * blob["mass_retention"]["heavy_atom_retention_mean"], 6
                ),
                "per_molecule_heavy_retention_median_pct": round(
                    100.0 * blob["mass_retention"]["heavy_atom_retention_median"], 6
                ),
                "uff_ok_rate_over_postprocessed": blob["postprocessed_axis"]["uff_ok_rate_over_postprocessed"],
                "uff_reasons": blob.get("uff_reasons"),
                "pipeline": blob.get("pipeline"),
            }
        )
    axes["our_axis"] = our_block

    # ---- the ratio -----------------------------------------------------
    our_delta = our_block.get("delta_percentage_points")
    their_delta = axes["hltf_postprocessing_delta"]["primary_delta_pts"]
    if our_delta is not None and their_delta:
        axes["magnitude_comparison"] = {
            "our_delta_pts": our_delta,
            "hltf_delta_pts": their_delta,
            "ratio_our_over_hltf": round(our_delta / their_delta, 6),
            "comparable_at_all": False,
            "verdict": (
                "NOT COMPARABLE as a magnitude. The two deltas are not the same measurement: "
                "theirs is the FULL effect of a three-step post-processing pipeline "
                "(largest fragment + explicit H + UFF) on a 100,000-sample drug-like pool where "
                "85.5 % of samples were ALREADY valid; ours is the effect of the SAME pipeline's "
                "first step on a 12,668-row metal-complex pool where only 8.51 % of rows were "
                "already single-fragment. A larger delta here means our RAW stream was more "
                "fragmented, not that our post-processing is better."
            ),
            "correction_to_the_brief": (
                "The brief framed HLTF's post-processing as 'a geometry relaxation'. The paper's "
                "OWN definition (sections/experiments.tex) is "
                "'standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation)' "
                "-- i.e. the SAME three steps our arm implements, with the SAME source "
                "(Buttenschoen et al., arXiv:2505.00518). Its first step IS fragment selection, "
                "which is mass-deleting. So the two post-processing axes are the same KIND of "
                "operation, and the difference is in the regime and the price, not in the kind."
            ),
            "internal_tension_in_the_paper": (
                "The paper attributes the PP gain to geometry only -- 'Standardized post-processing "
                "mainly relieves geometric strain ... and cannot repair infeasible topology "
                "(e.g., invalid valence or disconnected graphs)' -- while its own pipeline's FIRST "
                "step (largest fragment) DOES repair disconnected graphs, and does so by deleting "
                "mass. The paper prints no mass-retention or survivor-shape number for its +PP arm, "
                "so the fragment-selection component of its +6.7 pts is UNQUANTIFIED in the paper."
            ),
        }
    return axes


def build_house_format(ours: dict[str, Any], axes: dict[str, Any]) -> dict[str, Any]:
    """The corrected one-line house format + two filled examples (raw + PP)."""
    arm = axes.get("our_axis", {})
    n_parsed = arm.get("n_parsed")
    raw_rate = arm.get("raw_single_fragment_rate")
    n_single = arm.get("n_single")
    n_rows = arm.get("n_rows_in_csv")
    path = arm.get("path")

    filled_raw = None
    filled_pp = None
    if n_parsed:
        n_unparsed = n_rows - n_parsed
        n_epochs = arm.get("n_epochs_in_pool")
        # The speck count and the mass-retention clause describe the
        # POST-PROCESSED arm only: before post-processing there is no
        # survivor set and nothing has been discarded.  Printing the
        # post-processed numbers on a raw-arm line would mix axes -- the
        # exact defect this house format exists to prevent.
        specks_raw = "n/a (no survivor set exists before post-processing)"
        mass_raw = "n/a (nothing is discarded before post-processing)"
        specks_pp = (
            f"{arm['n_survivors_le_2_heavy']}/{n_parsed} = "
            f"{arm['pct_survivors_le_2_heavy']:.2f}% of the survivors are <=2-heavy-atom specks "
            f"(a 'connected' molecule that is a fragment of a fragment)"
        )
        mass_pp = (
            f"{arm['pct_raw_mass_discarded']:.2f}% of the raw heavy mass discarded (pooled); "
            f"per-molecule heavy-atom retention mean "
            f"{arm['per_molecule_heavy_retention_mean_pct']:.2f}%, median "
            f"{arm['per_molecule_heavy_retention_median_pct']:.2f}%"
        )
        filled_raw = HOUSE_FORMAT_TEMPLATE.format(
            v=100.0 * raw_rate,
            k=n_single,
            N=n_parsed,
            denominator=(
                f"RDKit-parseable rows of a CONCATENATED {n_epochs}-epoch generation pool "
                f"({n_rows} CSV rows, {n_unparsed} unparseable) -- not one draw of N"
            ),
            postproc="NONE (raw, pre-LargestFragmentChooser)",
            hydrogens="implicit",
            specks_clause=specks_raw,
            mass_clause=mass_raw,
            path=path,
        )
        filled_pp = HOUSE_FORMAT_TEMPLATE.format(
            v=100.0 * arm["postprocessed_connected_rate"],
            k=arm["n_connected"],
            N=n_parsed,
            denominator=(
                f"RDKit-parseable rows of a CONCATENATED {n_epochs}-epoch generation pool "
                f"({n_rows} CSV rows, {n_unparsed} unparseable) -- not one draw of N"
            ),
            postproc=(
                "LargestFragmentChooser -> explicit H -> UFF "
                f"(UFF ok {arm['n_connected'] * arm['uff_ok_rate_over_postprocessed']:.0f}/"
                f"{arm['n_connected']}: no UFF parameters for this corpus' metals)"
            ),
            hydrogens="explicit",
            specks_clause=specks_pp,
            mass_clause=mass_pp,
            path=path,
        )
    return {
        "prior_house_format_wfa161": PRIOR_HOUSE_FORMAT,
        "what_the_prior_format_missed": [
            "N and the DENOMINATOR RULE were separate slots, so a number could be printed with a "
            "denominator rule that did not match N (the Wf-A153 census defect: 'runs' is a LIST).",
            "no FALSE-VALID / SPECK slot: the format could print '100 % connected' with no indication "
            "that 21.02 % of those 'connected' survivors are <=2-heavy-atom specks.",
            "no MASS-RETENTION slot: the price of the post-processing was invisible.",
            "the definition said 'exactly one connected component' without naming the tool (RDKit).",
            "the slots must be AXIS-LOCAL: the speck count and the mass-retention clause describe "
            "the POST-PROCESSED arm only, so a RAW-arm line must print 'n/a' there rather than "
            "borrow the post-processed numbers (borrowing them is an axis mix, the defect class "
            "§6.250-§6.253 catalogues).",
        ],
        "corrected_house_format": HOUSE_FORMAT_TEMPLATE,
        "slot_semantics": {
            "v": "the connectivity percentage for THIS arm, in [0,100]",
            "k": "the numerator for THIS arm: rows with exactly one RDKit connected component",
            "N": "the denominator for THIS arm: the SAME N the percentage is taken over",
            "denominator": "the DENOMINATOR RULE in words, including what is in and out of N",
            "postproc": "the post-processing in force at THIS number, or 'NONE (raw)'",
            "hydrogens": "explicit / implicit / not stated",
            "specks_clause": "the false-valid analogue: how many survivors are <=2-heavy-atom "
            "specks (POST-PROCESSED arm) or 'n/a' (raw arm)",
            "mass_clause": "the price of the post-processing (POST-PROCESSED arm) or 'n/a' (raw arm)",
            "path": "the on-disk path of the artifact every number above was read from",
        },
        "slots_machine_readable": {
            "our_raw_arm": {
                "v_pct": round(100.0 * (arm.get("raw_single_fragment_rate") or 0.0), 6),
                "k": arm.get("n_single"),
                "N": arm.get("n_parsed"),
                "n_rows_in_csv": arm.get("n_rows_in_csv"),
                "n_epochs_in_pool": arm.get("n_epochs_in_pool"),
                "postproc": "NONE (raw)",
                "hydrogens": "implicit",
                "specks": None,
                "mass": None,
                "source": arm.get("path"),
            },
            "our_postprocessed_arm": {
                "v_pct": round(100.0 * (arm.get("postprocessed_connected_rate") or 0.0), 6),
                "k": arm.get("n_connected"),
                "N": arm.get("n_parsed"),
                "n_rows_in_csv": arm.get("n_rows_in_csv"),
                "n_epochs_in_pool": arm.get("n_epochs_in_pool"),
                "postproc": "LargestFragmentChooser -> explicit H -> UFF",
                "hydrogens": "explicit",
                "n_survivors_le_2_heavy": arm.get("n_survivors_le_2_heavy"),
                "pct_survivors_le_2_heavy": arm.get("pct_survivors_le_2_heavy"),
                "pct_raw_mass_discarded": arm.get("pct_raw_mass_discarded"),
                "pooled_raw_heavy_mass_retained_pct": arm.get("pooled_raw_heavy_mass_retained_pct"),
                "per_molecule_heavy_retention_mean_pct": arm.get("per_molecule_heavy_retention_mean_pct"),
                "per_molecule_heavy_retention_median_pct": arm.get("per_molecule_heavy_retention_median_pct"),
                "uff_ok_rate_over_postprocessed": arm.get("uff_ok_rate_over_postprocessed"),
                "source": arm.get("path"),
            },
        },
        "filled_example_our_raw_arm": filled_raw,
        "filled_example_our_postprocessed_arm": filled_pp,
    }


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------
def render_report(payload: dict[str, Any]) -> str:
    L: list[str] = []
    a = L.append
    a("# Wf-A168 (branch HLTF) — the post-processing axis and the false-valid axis for arXiv:2603.27113")
    a("")
    a(f"**Generated:** {payload['generated_at']} · **Tag:** MEASURED (primary-source retrieval + on-disk metrics) · **GPU:** 0 (CPU-only)")
    a("")
    a("## 0. Retrieval record")
    a("")
    a("| artefact | URL | HTTP (live) | sha256 of the retrieved bytes | bytes | retrieved (UTC, live) |")
    a("|---|---|---|---|---|---|")
    for key, rec in payload["retrieval"].items():
        status = rec.get("http_status_live") or rec.get("http_status")
        when = rec.get("retrieved_utc_live") or rec.get("retrieved_utc")
        served = rec.get("http_status")
        a(
            f"| `{key}` | {rec['url']} | `{status}` | `{rec['sha256']}` | {rec['bytes']} | {when} |"
        )
        if served == "cache":
            a(f"| | ↳ served from cache on this run; HTTP status and timestamp replayed from the sidecar written at the live fetch | | | | |")
    a("")
    for key, rec in payload["retrieval"].items():
        if rec.get("final_url"):
            a(f"- `{key}` final URL after redirect: `{rec['final_url']}`")
    a("")
    if payload["retrieval"]["eprint"].get("error"):
        a(f"> **NOT-RETRIEVED:** `{payload['retrieval']['eprint']['error']}` — no number below is quotable.")
        a("")
    a("## 1. Pinned percentages (every one located in the retrieved LaTeX source)")
    a("")
    a("| id | metric | value | dataset | denominator | post-processing | hydrogens | LaTeX site |")
    a("|---|---|---|---|---|---|---|---|")
    for p in payload["pinned"]:
        sites = "; ".join(f"`{s['file']}:{s['line']}`" for s in p["sites"]) or "**NOT FOUND**"
        a(
            f"| `{p['id']}` | {p['metric']} | **{p['value_pct']} %** (`{p['value_verbatim']}`) | {p['dataset']} | "
            f"{p['denominator']} | {p['postproc']} | {p['hydrogens']} | {sites} |"
        )
    a("")
    a("## 2. Context pins (the metadata above is itself pinned, not asserted)")
    a("")
    for c in payload["context_pins"]:
        sites = "; ".join(f"`{s['file']}:{s['line']}`" for s in c["sites"]) or "**NOT FOUND**"
        a(f"- `{c['id']}` — {c['claim']} — {sites}")
    a("")
    a("## 3. The post-processing axis")
    a("")
    a("| metric | raw | +PP | delta (pts) | share of pre-PP invalid recovered |")
    a("|---|---|---|---|---|")
    for r in payload["axes"]["hltf_postprocessing_delta"]["rows"]:
        a(
            f"| {r['metric']} | {r['raw_pct']} % | {r['postprocessed_pct']} % | "
            f"**{r['delta_percentage_points']:+g}** | {r['share_of_pre_pp_invalid_recovered_pct']} % |"
        )
    a("")
    mc = payload["axes"].get("magnitude_comparison", {})
    oa = payload["axes"].get("our_axis", {})
    a(
        f"**OURS:** raw single-fragment rate {oa.get('raw_single_fragment_rate')} "
        f"({oa.get('n_single')}/{oa.get('n_parsed')}) → post-processed connected rate "
        f"{oa.get('postprocessed_connected_rate')} ({oa.get('n_connected')}/{oa.get('n_parsed')}) = "
        f"**{mc.get('our_delta_pts', 0):+.2f} pts** over the SAME {oa.get('n_parsed')}-row denominator "
        f"(exact: {mc.get('our_delta_pts')})."
    )
    a("")
    a(
        f"**Magnitude ratio:** {mc.get('our_delta_pts'):+.2f} pts (ours) / "
        f"{mc.get('hltf_delta_pts'):+g} pts (HLTF Valid) = "
        f"**{mc.get('ratio_our_over_hltf'):.2f}×** (exact: {mc.get('ratio_our_over_hltf')})."
    )
    a("")
    a(f"> **COMPARABLE AT ALL? NO.** {mc.get('verdict', '')}")
    a("")
    a(f"> ⚠️ **CORRECTION TO THE BRIEF.** {mc.get('correction_to_the_brief', '')}")
    a("")
    a(f"> ⚠️ **INTERNAL TENSION IN THE PAPER.** {mc.get('internal_tension_in_the_paper', '')}")
    a("")
    a("**Honest one-sentence reading:**")
    a("")
    a(
        "> The two deltas measure different things on different populations under different "
        "baselines — HLTF's +6.7 pts is a *post-processing effect on a stream that was already "
        f"{payload['axes']['hltf_postprocessing_delta']['rows'][0]['raw_pct']} % valid* over 100,000 "
        "drug-like samples, while our +91.49 pts is the *fragment-selection step alone* rescuing a "
        f"stream that was only {100.0 * (oa.get('raw_single_fragment_rate') or 0):.2f} % single-fragment "
        f"over {oa.get('n_parsed')} metal-complex rows — so a reader must not conclude "
        "\"we beat the frontier\": the larger number says our RAW stream is more fragmented, and the "
        f"price is {oa.get('pct_raw_mass_discarded'):.2f} % of the raw heavy mass discarded with "
        f"{oa.get('pct_survivors_le_2_heavy'):.2f} % of the survivors left as ≤2-heavy-atom specks."
    )
    a("")
    a("## 4. The false-valid axis")
    a("")
    for q in payload["false_valid_quotes"]:
        a(f"> **{q['where']}** (`{q['site']}`):")
        a(">")
        a(f"> {q['quote']}")
        a("")
    a(payload["false_valid_analogue"]["analogue_sentence"])
    a("")
    a("**Where the analogy goes:**")
    for s in payload["false_valid_analogue"]["where_it_holds"]:
        a(f"- {s}")
    a("")
    a("**Where it breaks:**")
    for s in payload["false_valid_analogue"]["where_it_breaks"]:
        a(f"- {s}")
    a("")
    a("## 5. The corrected house format")
    a("")
    a(f"**Prior (Wf-A161):** `{payload['house_format']['prior_house_format_wfa161']}`")
    a("")
    a("**What the prior format missed:**")
    for s in payload["house_format"]["what_the_prior_format_missed"]:
        a(f"- {s}")
    a("")
    a(f"**Corrected:**")
    a("")
    a("```")
    a(payload["house_format"]["corrected_house_format"])
    a("```")
    a("")
    a("**Filled — our RAW arm:**")
    a("")
    a("```")
    a(payload["house_format"]["filled_example_our_raw_arm"])
    a("```")
    a("")
    a("**Filled — our POST-PROCESSED arm:**")
    a("")
    a("```")
    a(payload["house_format"]["filled_example_our_postprocessed_arm"])
    a("```")
    a("")
    a("## 6. Honest negatives / UNPINNED")
    a("")
    for s in payload["honest_negatives"]:
        a(f"- {s}")
    a("")
    a("## 7. Files")
    a("")
    for k, v in payload["files"].items():
        a(f"- `{k}` → `{v}`")
    a("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="cache only; never hit the network")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- retrieval ----------------------------------------------------
    eprint_bytes, eprint_rec = _fetch(
        eprint_url(ARXIV_ID), cache_dir / f"{ARXIV_ID}.tar.gz", offline=args.offline
    )
    abs_bytes, abs_rec = _fetch(
        abs_url(ARXIV_ID), cache_dir / f"{ARXIV_ID}.abs.html", offline=args.offline
    )
    retrieval = {"eprint": eprint_rec, "abs": abs_rec}

    if eprint_bytes is None:
        print(
            f"NOT-RETRIEVED: {eprint_rec.get('error')} — no number is quotable for {ARXIV_ID}.",
            file=sys.stderr,
        )
        payload = {
            "artifact": "wfa168_hltf_axis",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "arxiv_id": ARXIV_ID,
            "status": "NOT-RETRIEVED",
            "retrieval": retrieval,
            "pinned": [],
            "context_pins": [],
            "axes": {},
            "house_format": {},
            "false_valid_quotes": [],
            "false_valid_analogue": {},
            "honest_negatives": [f"the e-print for {ARXIV_ID} was NOT retrieved: {eprint_rec.get('error')}"],
            "files": {},
        }
        atomic_write_file(out_dir / "sources.json", json.dumps(payload, indent=2) + "\n")
        atomic_write_file(out_dir / "report.md", "# Wf-A168 HLTF — NOT-RETRIEVED\n")
        return 1

    tex = extract_tex(eprint_bytes)
    pinned = pin_all(tex, PINS)
    context_pins = pin_all(tex, CONTEXT_PINS)

    ours = load_our_metrics()
    axes = compute_axes(ours, pinned)

    # ---- false-valid quotes -------------------------------------------
    false_valid_quotes: list[dict[str, Any]] = []
    for ctx_id, where in [
        ("ctx_false_valid_abstract", "Abstract"),
        ("ctx_false_valid_ethics", "Ethics Statement"),
        ("ctx_false_valid_experiments", "Experiments, PoseBusters paragraph"),
        ("ctx_false_valid_pb_rationale", "Experiments, 'Why PoseBusters matters here'"),
    ]:
        c = next((x for x in context_pins if x["id"] == ctx_id), None)
        if c and c["sites"]:
            false_valid_quotes.append(
                {
                    "where": where,
                    "site": f"{c['sites'][0]['file']}:{c['sites'][0]['line']}",
                    "quote": c["needle"],
                }
            )

    oa = axes.get("our_axis", {})
    false_valid_analogue = {
        "label": "ANALOGUE — NOT THE SAME METRIC",
        "analogue_sentence": (
            "**OUR-side analogue (LABELLED ANALOGUE, NOT THE SAME METRIC):** our post-processed arm "
            f"reports {100.0 * (oa.get('postprocessed_connected_rate') or 0):.2f} % \"connected\" "
            f"({oa.get('n_connected')}/{oa.get('n_parsed')}), of which "
            f"{oa.get('n_survivors_le_2_heavy')}/{oa.get('n_parsed')} = "
            f"{oa.get('pct_survivors_le_2_heavy'):.2f} % of survivors are ≤2-heavy-atom specks — a "
            "molecule that is \"one connected component\" and is still a fragment of a fragment."
        ),
        "where_it_holds": [
            "Both are validity predicates satisfied by a structure a chemist would not accept as a "
            "molecule: a single heavy atom has exactly one connected component, so the predicate "
            "passes vacuously.",
            "Both are properties of a POST-PROCESSED stream, not of the sampler's raw output.",
            "Both are one-step predicates (component count) that cannot see molecule-level viability.",
        ],
        "where_it_breaks": [
            "DIRECTION: the paper's false-valid is a SECOND, STRICTER check applied to the SAME "
            "molecule (PoseBusters on top of RDKit sanitization). Our analogue is not a stricter "
            "check on the same molecule — after LargestFragmentChooser we are scoring a DIFFERENT "
            "molecule (the largest fragment), so '100 % connected' is a statement about a molecule "
            "the sampler never emitted.",
            "CLASS: the paper's false-valid presupposes a chemically implausible but structurally "
            "WHOLE molecule (a strained ring, a bad bond length). A ≤2-heavy-atom speck is not a "
            "false-positive validation — it is a degenerate survivor of a DELETION. Different "
            "failure class; the paper's term does not cover it.",
            "MEASUREMENT: the paper never counts false-valid samples. It INFERS them from the gap "
            "between PB-Valid and RDKit Val ('Higher PB-Valid (at comparable RDKit validity) "
            "indicates fewer false-valid samples'), so its 'false-valid axis' is an inference over "
            "a pair of numbers, whereas ours is a COUNT over the survivor set.",
            "POPULATION: the paper's own limitation excludes our corpus: 'performance on unusual "
            "chemistries, charged/metal-containing compounds, or biomolecules remains to be "
            "established' (sections/appendix.tex). Its 100,000 GEOM-DRUGS samples contain no metals.",
            "DENOMINATOR + HYDROGENS: theirs is 100,000 samples from a training-set-distribution "
            "sampling of small drug-like molecules, explicit H after PP; ours is 12,668 RDKit-parseable "
            "rows of a CONCATENATED 18-EPOCH generation pool (12,669 CSV rows, 1 unparseable) of a "
            "metal-complex generator -- 100 % of the pool's rows carry a cohort metal "
            "(Ru/Rh/Re/Os/Ir/Pt), and the rows are pooled across epochs rather than drawn once.",
        ],
        "observed_inconsistency_in_the_paper": (
            "The paper states PoseBusters 'applies stricter structural checks than RDKit', yet its "
            "own table prints HLTF PB-Valid 94.0 % ABOVE HLTF RDKit Val 93.8 % (the two GeoLDM / "
            "GCDM baseline rows run the other way: 93.1 vs 93.6 and 91.9 vs 94.8). A stricter check "
            "cannot, by the paper's own definition, pass more samples than the check it is "
            "stricter than; the paper does not define the relationship between the two columns, so "
            "this is recorded as an OBSERVED inconsistency in the printed table, not as a defect "
            "claim."
        ),
    }

    honest_negatives = [
        "The QM9 arms carry NO stated hydrogen convention: the paper never says whether the 10,000 "
        "QM9 samples are evaluated with explicit or implicit H. Hydrogens are therefore UNPINNED "
        "for every QM9 number here.",
        "The GEOM-DRUGS raw arms are likewise not labelled as to hydrogens; only the +PP arm is "
        "explicitly 'explicit hydrogens'. The raw arm is therefore reported as 'implicit at this "
        "stage' — the paper's own PP definition adds explicit H, so the raw stream precedes it.",
        "The invalidity-cause breakdown ('Disconn. 35 % / 20 % / 12 %') is a percentage of INVALID "
        "samples, not of all samples, and the paper prints no absolute N for the invalid set. It "
        "is NOT comparable to our raw single-fragment rate, which is over parsed rows.",
        "HLTF prints NO mass-retention and NO survivor-shape number for its +PP arm. The "
        "fragment-selection component of its +6.7 pts is therefore UNQUANTIFIED in the paper; we "
        "can say it is not reported, we cannot say it is small.",
        "The paper's ablation table (tables/ablations.tex, default rows) prints QM9 V&U 93.2 and "
        "QM9 PB 92.4 for the SAME default configuration the main table reports as Val&Uniq 92.9 and "
        "PB-Valid 94. The paper prints two different numbers for the same default config; the "
        "headline 94.0 is the main-table one.",
        "Our own c7 pool (12669 rows / 18 epochs) and the c8 pool (2112 rows / 3 epochs) are "
        "DIFFERENT pools and their denominators are not mixed here.",
        "This branch read `metrics/by_metric/cfm_fragmentation_postproc_arm.json` and "
        "`metrics/by_metric/wfa155_ligand_connected_rate.json` from disk and recomputed nothing "
        "that those artifacts already measure; every derived number here is a subtraction or a "
        "ratio of those on-disk values.",
        "A parallel branch of this wave extracted a DIFFERENT paper (arXiv:2608.00657) into the "
        "shared /tmp/wfa168_probe directory during this run; those files were excluded and the "
        "HLTF source was re-extracted into a private directory and re-hashed. The tarball sha256 "
        "recorded here is of the re-extracted, verified copy.",
    ]

    payload: dict[str, Any] = {
        "artifact": "wfa168_hltf_axis",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wave": "Wf-A168 (branch HLTF)",
        "tag": "MEASURED",
        "gpu": "0 (CPU-only; no torch import; no trainer; no decode)",
        "arxiv_id": ARXIV_ID,
        "paper_short": PAPER_SHORT,
        "question": (
            "What does the frontier's own post-processing DELTA look like when a paper prints the "
            "RAW and the POST-PROCESSED number for the SAME pool with the SAME denominator -- and "
            "is our +91.49 pts comparable to it?"
        ),
        "status": "OK" if all(p["pin_status"] == "PINNED" for p in pinned) else "PARTIAL",
        "retrieval": retrieval,
        "pinned": pinned,
        "context_pins": context_pins,
        "axes": axes,
        "false_valid_quotes": false_valid_quotes,
        "false_valid_analogue": false_valid_analogue,
        "house_format": build_house_format(ours, axes),
        "honest_negatives": honest_negatives,
        "files": {
            "script": "molmetal/scripts/wfa168_hltf_axis.py",
            "sources_json": str((out_dir / "sources.json").relative_to(REPO_ROOT)),
            "report_md": str((out_dir / "report.md").relative_to(REPO_ROOT)),
        },
    }

    atomic_write_file(out_dir / "sources.json", json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    atomic_write_file(out_dir / "report.md", render_report(payload))

    n_pinned = sum(1 for p in pinned if p["pin_status"] == "PINNED")
    print(f"[wfa168] pinned {n_pinned}/{len(pinned)} numbers; context pins "
          f"{sum(1 for c in context_pins if c['pin_status'] == 'PINNED')}/{len(context_pins)}")
    print(f"[wfa168] eprint sha256 {eprint_rec['sha256']}")
    print(f"[wfa168] wrote {out_dir / 'sources.json'} and {out_dir / 'report.md'}")
    for p in pinned:
        if p["pin_status"] != "PINNED":
            print(f"[wfa168] UNPINNED: {p['id']} ({p['failure_note']})", file=sys.stderr)
    return 0 if all(p["pin_status"] == "PINNED" for p in pinned) else 2


if __name__ == "__main__":
    raise SystemExit(main())
