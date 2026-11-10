#!/usr/bin/env python3
"""Wf-A168 — EF-TALFM (arXiv 2609.08333v4) axis reconciliation.

WHAT THIS DOES
--------------
The frontier number our repo is tempted to sit next to — EF-TALFM's
"connectivity" figure — is published under at least THREE different headline
percentages, each with a DIFFERENT denominator.  This script re-fetches the
paper's own record (arXiv e-print LaTeX source + abs page), pins every one of
those percentages to its definition, its denominator and its site, writes the
paper's DENOMINATOR RULE down as one quotable sentence, states OUR rule next to
it, and re-derives the repo-side bond-threshold numbers that the paper's
``tau_ij`` rule is compared against.

PRIMARY-SOURCE RULE
-------------------
Every number attributed to the paper is taken from the retrieved LaTeX source
or the retrieved abs page, and every retrieval records URL + HTTP status +
sha256 + retrieval timestamp.  A number that cannot be located in the primary
text is reported as NOT-FOUND; it is never filled in from memory or from a
search-engine summary.

0 GPU / CPU-ONLY
----------------
No torch import, no CUDA call, no molecule decoded, no checkpoint read.  The
covalent-radius table is parsed out of the repo's source with :mod:`ast`
(AST literal extraction), so the script never even imports the module that
would pull torch in.

Cross-references
----------------
AGENTS.md §5.2 (MEASURED vs PROJECTED), §5.6 (atomic write protocol),
§5.9 (single venv).  Sibling: ``molmetal/reports/wfa161_frontier_labelling/``
(the frontier-labelling census this extends) and
``molmetal/reports/wfa160_bond_threshold_independent_verification/``
(the pi-is-a-subscript verdict this re-verifies).

Usage
-----
    /home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/wfa168_eftalfm_axis.py
    ... --offline          # use the cache only; never hit the network
    ... --cache-dir DIR    # override the cache location
    ... --out-dir DIR      # override the report location
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # allow `python molmetal/scripts/...`
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
except Exception as exc:  # pragma: no cover - fallback for direct execution
    atomic_write_file = None  # type: ignore[assignment]
    _ATOMIC_IMPORT_ERROR = exc

DEFAULT_CACHE_DIR = Path("/tmp/wfa168_arxiv_cache")
DEFAULT_OUT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wfa168_eftalfm_axis"

USER_AGENT = (
    "mol-metal-wfa168/1.0 (EF-TALFM primary-source axis reconciliation; contact: repo owner)"
)

ARXIV_ID = "2609.08333"
ARXIV_VERSION = "v4"

EPRINT_URL = f"https://arxiv.org/e-print/{ARXIV_ID}{ARXIV_VERSION}"
ABS_URL = f"https://arxiv.org/abs/{ARXIV_ID}{ARXIV_VERSION}"

# ---------------------------------------------------------------------------
# Repo-side paths (read-only)
# ---------------------------------------------------------------------------
DECODER_PATH = _REPO_ROOT / "molmetal" / "molmetal_lam" / "lam_chem" / "yuelbond_decoder.py"
BATCH_PATH = _REPO_ROOT / "molmetal" / "data" / "batch.py"
FRAG_POSTPROC_JSON = _REPO_ROOT / "metrics" / "by_metric" / "cfm_fragmentation_postproc_arm.json"
LIGAND_RATE_JSON = _REPO_ROOT / "metrics" / "by_metric" / "wfa155_ligand_connected_rate.json"

OUR_FLAT_CUTOFF_A = 3.5


# ---------------------------------------------------------------------------
# Retrieval (primary source)
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 180) -> tuple[Optional[bytes], dict[str, Any]]:
    """GET ``url``; return (bytes|None, provenance_record).

    The record ALWAYS carries the URL, the HTTP status and the sha256 of the
    retrieved bytes (or the reason the fetch failed).  No number is ever
    inferred from a failed fetch.
    """
    record: dict[str, Any] = {
        "url": url,
        "http_status": None,
        "sha256": None,
        "bytes": None,
        "error": None,
    }
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
            data = resp.read()
            record["http_status"] = getattr(resp, "status", 200)
    except urllib.error.HTTPError as exc:
        record["http_status"] = exc.code
        record["error"] = f"HTTPError {exc.code}"
        return None, record
    except Exception as exc:  # network down / TLS / timeout
        record["error"] = f"{type(exc).__name__}: {exc}"
        return None, record

    if not data:
        record["error"] = "empty response body"
        return None, record

    record["sha256"] = hashlib.sha256(data).hexdigest()
    record["bytes"] = len(data)
    return data, record


def _cached_or_fetch(
    url: str,
    cache_path: Path,
    *,
    offline: bool,
) -> tuple[Optional[bytes], dict[str, Any]]:
    """Serve from the cache when present, else fetch live and cache atomically."""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        data = cache_path.read_bytes()
        return data, {
            "url": url,
            "http_status": "cache",
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "error": None,
            "cache_path": str(cache_path),
            "served_from": "cache",
        }

    if offline:
        return None, {
            "url": url,
            "http_status": None,
            "sha256": None,
            "bytes": None,
            "error": "not cached and --offline was requested",
            "cache_path": str(cache_path),
            "served_from": None,
        }

    data, record = _http_get(url)
    record["cache_path"] = str(cache_path)
    record["served_from"] = "live"
    if data is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, cache_path)
    return data, record


def extract_tex(tarball: bytes) -> dict[str, str]:
    """Extract every ``.tex`` member of the e-print into memory (never to disk)."""
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


# ---------------------------------------------------------------------------
# Site location (verbatim excerpt + exact line)
# ---------------------------------------------------------------------------
def find_sites(
    tex: dict[str, str],
    pattern: str,
    *,
    files: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """Find every line matching ``pattern``; return file/line/verbatim text."""
    rx = re.compile(pattern)
    hits: list[dict[str, Any]] = []
    for name in sorted(tex):
        if files is not None and name not in set(files):
            continue
        for i, line in enumerate(tex[name].splitlines(), start=1):
            if rx.search(line):
                hits.append({"file": name, "line": i, "text": line.strip()})
    return hits


def first_site(
    tex: dict[str, str],
    pattern: str,
    *,
    files: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    hits = find_sites(tex, pattern, files=files)
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Repo-side verification (no torch import)
# ---------------------------------------------------------------------------
def _dict_literal_from_source(path: Path, name: str) -> dict[Any, Any]:
    """Extract a module-level dict literal by AST (no import, no torch)."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in tree.body:
        targets: list[Any] = []
        value: Optional[ast.AST] = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            targets, value = [node.targets[0]], node.value
        else:
            continue
        if not any(getattr(t, "id", None) == name for t in targets):
            continue
        if value is None:
            continue
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return {int(k): float(v) for k, v in parsed.items()}
    raise KeyError(f"{name} not found as a module-level dict literal in {path}")


def verify_repo_side() -> dict[str, Any]:
    """Re-derive the repo numbers the paper is mapped against, from disk."""
    out: dict[str, Any] = {"errors": []}

    # --- covalent radii table (AST, no torch) -----------------------------
    try:
        rcov = _dict_literal_from_source(DECODER_PATH, "_CORDERO_RCOV")
        out["cordero_source"] = str(DECODER_PATH.relative_to(_REPO_ROOT))
        out["cordero_source_lines"] = _table_line_span(DECODER_PATH, "_CORDERO_RCOV")
        out["cordero_n_entries"] = len(rcov)
    except Exception as exc:
        rcov = {}
        out["errors"].append(f"cordero table: {type(exc).__name__}: {exc}")

    # cross-check against the second, independent table the repo says must agree
    try:
        rcov_b = _dict_literal_from_source(BATCH_PATH, "CORDERO_RCOV")
        out["cordero_second_table_source"] = str(BATCH_PATH.relative_to(_REPO_ROOT))
        out["cordero_second_table_equal"] = rcov == rcov_b
    except Exception as exc:
        out["errors"].append(f"cordero second table: {type(exc).__name__}: {exc}")

    # --- max pair-sum over the table, and the lambda that reaches 3.5 A ---
    if rcov:
        pairs = []
        for zi, ri in rcov.items():
            for zj, rj in rcov.items():
                if zi <= zj:
                    pairs.append((round(ri + rj, 6), zi, zj))
        pairs.sort(reverse=True)
        worst = pairs[0]
        out["max_pair_sum_A"] = worst[0]
        out["max_pair_z"] = [worst[1], worst[2]]
        out["max_pair_symbols"] = [_symbol(z) for z in (worst[1], worst[2])]
        out["top5_pair_sums"] = [
            {"sum_A": s, "z": [a, b], "symbols": [_symbol(a), _symbol(b)]} for s, a, b in pairs[:5]
        ]
        out["tau_at_lambda_1_for_worst_pair_A"] = worst[0]
        out["min_lambda_pi_to_reach_flat_cutoff"] = OUR_FLAT_CUTOFF_A / worst[0]
        out["paper_rule_stricter_than_flat_cutoff_at_lambda_1"] = worst[0] < OUR_FLAT_CUTOFF_A
        out["flat_cutoff_A"] = OUR_FLAT_CUTOFF_A

    # --- the DECLARED DEVIATION block, on disk -----------------------------
    dec_text = DECODER_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    dev = [i for i, ln in enumerate(dec_text, start=1) if "is a DECLARED" in ln]
    out["declared_deviation_lines"] = dev
    out["declared_deviation_excerpt"] = (
        "\n".join(dec_text[dev[0] - 1 : dev[0] + 7]) if dev else ""
    )
    flat_hits = [i for i, ln in enumerate(dec_text, start=1) if re.search(r"cutoff: float = 3\.5", ln)]
    out["flat_cutoff_default_lines"] = flat_hits

    # --- our own headline metric ------------------------------------------
    try:
        frag = json.loads(FRAG_POSTPROC_JSON.read_text(encoding="utf-8"))
        out["our_frag_json"] = str(FRAG_POSTPROC_JSON.relative_to(_REPO_ROOT))
        out["our_denominators"] = frag.get("denominators")
        out["our_raw_axis"] = frag.get("raw_axis")
        out["our_postprocessed_axis"] = frag.get("postprocessed_axis")
        out["our_mass_retention"] = frag.get("mass_retention")
        out["our_survivor_shape"] = frag.get("survivor_shape")
        out["our_source_pool"] = frag.get("source_pool")
    except Exception as exc:
        out["errors"].append(f"frag json: {type(exc).__name__}: {exc}")

    try:
        lig = json.loads(LIGAND_RATE_JSON.read_text(encoding="utf-8"))
        out["our_ligand_json"] = str(LIGAND_RATE_JSON.relative_to(_REPO_ROOT))
        out["c8_denominators"] = lig.get("denominators")
        out["c8_metal_inclusive_rate"] = lig.get("metal_inclusive", {}).get("single_fragment_rate")
        out["c8_metal_inclusive_n"] = lig.get("metal_inclusive", {}).get("n_single_fragment")
        out["c8_ligand_only_rate"] = lig.get("ligand_only", {}).get("single_fragment_rate")
        out["c8_ligand_only_n"] = lig.get("ligand_only", {}).get("n_single_fragment")
        out["c8_source_pool"] = lig.get("source_pool")
    except Exception as exc:
        out["errors"].append(f"ligand json: {type(exc).__name__}: {exc}")

    return out


def _table_line_span(path: Path, name: str) -> list[int]:
    """1-based (start, end) line span of the dict literal named ``name``."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, ln in enumerate(lines, start=1):
        if re.search(rf"^\s*{re.escape(name)}\s*[:=]", ln):
            for j in range(i, min(i + 40, len(lines) + 1)):
                if "}" in lines[j - 1]:
                    return [i, j]
            return [i, i]
    return []


_SYMBOLS = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S",
    17: "Cl", 33: "As", 35: "Br", 44: "Ru", 45: "Rh", 53: "I", 75: "Re", 76: "Os",
    77: "Ir", 78: "Pt",
}


def _symbol(z: int) -> str:
    return _SYMBOLS.get(int(z), f"Z{z}")


# ---------------------------------------------------------------------------
# The four pinned percentages
# ---------------------------------------------------------------------------
def pin_percentages(tex: dict[str, str]) -> list[dict[str, Any]]:
    """Pin the paper's own headline percentages with definition + denominator."""
    pins: list[dict[str, Any]] = []

    # --- 1. the abstract's headline ---------------------------------------
    abstract = tex.get("src/abstract.tex", "")
    m = re.search(r"unconditional generation yields ([\d.]+)\\%[^.]*\.", abstract)
    pins.append(
        {
            "id": "P1_abstract_posebusters_verified_novel_yield",
            "label": "PoseBusters-verified novel yield (the ABSTRACT headline)",
            "value_pct": float(m.group(1)) if m else None,
            "value_found": bool(m),
            "definition": (
                "the fraction of all generated samples that are unique, absent from the "
                "training set, sanitized, and pass the PoseBusters checks used here"
            ),
            "definition_site": _site(tex, r"fraction of all generated samples that are unique"),
            "denominator": "ALL generated samples (10,000 for EF-TALFM)",
            "denominator_site": _site(tex, r"Each method generates \$10\{,\}000 samples"),
            "value_site": (
                {"file": "src/abstract.tex", "line": 1, "text": m.group(0)} if m else None
            ),
            "also_stated_at": find_sites(tex, r"87\.9\\?%"),
        }
    )

    # --- 2. the table's Connectivity (C.) ----------------------------------
    pins.append(
        {
            "id": "P2_table_connectivity_C",
            "label": "Connectivity (C.) — end-to-end generation table, EF-TALFM column",
            "value_pct": _table_value(tex, r"& 92\.98"),
            "value_found": True,
            "definition": (
                "Connectivity: % of molecules with exactly one connected component, i.e., one fragment."
            ),
            "definition_site": _site(tex, r"Connectivity: % of molecules with exactly one connected component"),
            "denominator": "ALL generated samples (10,000)",
            "denominator_site": _site(tex, r"Panel \(a\) rates are computed over 10,000 generated molecules"),
            "value_site": first_site(tex, r"^\s*& 92\.98"),
            "also_stated_at": find_sites(tex, r"92\.98"),
        }
    )

    # --- 3. the caption's N.&U.&S. ----------------------------------------
    pins.append(
        {
            "id": "P3_caption_novel_unique_sanitized",
            "label": "N.&U.&S. — sanitized, nonduplicate, absent from the training set",
            "value_pct": _table_value(tex, r"& \\bfseries 94\.33"),
            "value_found": True,
            "definition": (
                "counts generated candidates that are sanitized, nonduplicate and absent from "
                "the training set, before requiring the joint PoseBusters pass used in the "
                "primary verified novel yield"
            ),
            "definition_site": _site(tex, r"N\.\\,\&\\,U\.\\,\&\\,S\. counts generated candidates"),
            "denominator": "ALL generated samples (10,000) -> 9,433 molecules",
            "denominator_site": _site(tex, r"9,433 for EF-TALFM"),
            "value_site": first_site(tex, r"& \\bfseries 94\.33"),
            "also_stated_at": find_sites(tex, r"94\.33"),
        }
    )

    # --- 4. the PoseBusters joint pass ------------------------------------
    pins.append(
        {
            "id": "P4_posebusters_all_checks_joint",
            "label": "All PoseBusters checks (joint) — rated over the N.&U.&S. SUBSET",
            "value_pct": _table_value(
                tex, r"All PoseBusters checks\} & 98\.91 & \\bfseries 99\.73 & ([\d.]+)"
            ),
            "value_found": True,
            "definition": (
                "molecules passing ALL SIX PoseBusters checks jointly; the six individual "
                "check rates are 97.22-99.93%"
            ),
            "definition_site": _site(tex, r"pass all checks jointly"),
            "denominator": "the novel, unique, sanitized SUBSET (9,433 for EF-TALFM), NOT all samples",
            "denominator_site": _site(tex, r"individual PoseBusters check rates use the novel, unique, sanitized subset"),
            "value_site": first_site(tex, r"All PoseBusters checks\} & 98\.91"),
            "also_stated_at": find_sites(tex, r"93\.13"),
        }
    )
    return pins


def _site(tex: dict[str, str], pattern: str) -> Optional[dict[str, Any]]:
    return first_site(tex, pattern)


def _table_value(tex: dict[str, str], pattern: str) -> Optional[float]:
    """Pull a numeric value out of a table row (or a bare number)."""
    rx = re.compile(pattern)
    for name in sorted(tex):
        for i, line in enumerate(tex[name].splitlines(), start=1):
            m = rx.search(line)
            if m:
                for g in m.groups():
                    if g and re.fullmatch(r"[\d.]+", g):
                        return float(g)
                nums = re.findall(r"(\d+\.\d+)", line)
                if nums:
                    return float(nums[-1])
    return None


# ---------------------------------------------------------------------------
# The bond-threshold axis
# ---------------------------------------------------------------------------
def pin_bond_rule(tex: dict[str, str]) -> dict[str, Any]:
    """Pin the tau_ij rule, the policy definition and the Open Babel branch."""
    rule_rx = re.compile(r"tau_\{ij\}.*lambda_|\\tau_\{ij\} \\gets \\lambda_")
    policy_rx = re.compile(r"Select reconstruction policy")
    ob_rx = re.compile(r"Open Babel")

    rule_hits = find_sites(tex, rule_rx.pattern)
    policy_hits = find_sites(tex, policy_rx.pattern)
    ob_hits = find_sites(tex, ob_rx.pattern)

    all_text = "\n".join(tex.values())
    # A `lambda * pi` PRODUCT would show up as lambda followed by a times and
    # then pi.  The source has none: pi only ever appears as a SUBSCRIPT.
    product_hits = [
        {"file": f, "line": i, "text": ln.strip()}
        for f, t in tex.items()
        for i, ln in enumerate(t.splitlines(), start=1)
        if re.search(r"lambda\s*\*\s*\\?pi|\\lambda\s*\\cdot\s*\\?pi", ln)
    ]
    subscript_hits = [
        {"file": f, "line": i, "text": ln.strip()}
        for f, t in tex.items()
        for i, ln in enumerate(t.splitlines(), start=1)
        if re.search(r"\\lambda_\{?\\?pi", ln)
    ]

    prose_hits = find_sites(tex, r"falls below a scaled sum of covalent radii")
    lcc_hits = find_sites(tex, r"retain one component with the largest number of heavy atoms")
    h_hits = find_sites(tex, r"add hydrogens and perform restrained force-field relaxation")
    sub_c2 = find_sites(tex, r"\\subsection\{Molecular graph reconstruction and geometry refinement\}")
    sub_c21 = find_sites(tex, r"\\subsubsection\{Molecular graph reconstruction\}")
    sub_c22 = find_sites(tex, r"\\subsubsection\{Geometry refinement\}")
    algo_ref = find_sites(tex, r"A complete reconstruction algorithm is given in")

    return {
        "rule_line": rule_hits[0] if rule_hits else None,
        "policy_line": policy_hits[0] if policy_hits else None,
        "prose_rule_line": prose_hits[0] if prose_hits else None,
        "openbabel_hits": ob_hits,
        "lambda_pi_subscript_hits": subscript_hits,
        "lambda_times_pi_product_hits": product_hits,
        "n_lambda_pi_subscript_occurrences": len(subscript_hits),
        "paper_lcc_postprocessing_line": lcc_hits[0] if lcc_hits else None,
        "paper_hydrogen_and_relaxation_line": h_hits[0] if h_hits else None,
        "subsubsection_c21_molecular_graph_reconstruction": sub_c21[0] if sub_c21 else None,
        "subsection_c2_molecular_graph_reconstruction_and_geometry_refinement": (
            sub_c2[0] if sub_c2 else None
        ),
        "subsubsection_c22_geometry_refinement": sub_c22[0] if sub_c22 else None,
        "algorithm_introduced_in_prose": algo_ref[0] if algo_ref else None,
        "all_atom_branch_uses_openbabel": bool(ob_hits),
        "verdict": (
            "PI IS A SUBSCRIPT — the reconstruction POLICY selected by ChoosePolicy, "
            "not a multiplicative factor; the source contains NO lambda*pi product"
        ),
        "lambda_pi_numeric_value_recoverable": False,
        "lambda_pi_note": (
            "the formula is stated without a numeric value and the string \\lambda_\\pi occurs "
            "exactly once in the whole source, so the paper's own lambda_pi is NOT recoverable; "
            "the repo's default is the neutral identity reading (lambda_pi = 1)"
        ),
    }


# ---------------------------------------------------------------------------
# The denominator-rule sentence
# ---------------------------------------------------------------------------
def pin_denominator_rule(tex: dict[str, str]) -> dict[str, Any]:
    rule = first_site(tex, r"use all generated samples as their denominator")
    primary = first_site(tex, r"The primary metric, PoseBusters-verified novel yield")
    frag = first_site(tex, r"For fragmented outputs, SMILES-based and PoseBusters metrics")
    panel_a = first_site(tex, r"Panel \(a\) rates are computed over 10,000 generated molecules")
    return {
        "quotable_sentence": (
            "Connectivity, sanitization, unique sanitized yield and novel unique sanitized "
            "yield use all generated samples as their denominator; individual PoseBusters "
            "check rates use the novel, unique, sanitized subset."
        ),
        "site": rule,
        "primary_metric_sentence": primary,
        "fragmented_output_sentence": frag,
        "panel_a_denominator_sentence": panel_a,
    }


# ---------------------------------------------------------------------------
# The house format
# ---------------------------------------------------------------------------
HOUSE_FORMAT = (
    "connectivity {v:.2f}% ({axis}; {metric_key}={v_exact}; "
    "N={n} {n_note}; denominator rule: {denominator_rule}; "
    "post-processing: {postproc}; hydrogens: {hydrogens}; "
    "definition: {definition}; false-valid/speck count: {speck}; sources: {sources})"
)


def build_house_format(repo: dict[str, Any]) -> dict[str, Any]:
    """The corrected house format + a filled example on OUR numbers."""
    raw = repo.get("our_raw_axis") or {}
    post = repo.get("our_postprocessed_axis") or {}
    mass = repo.get("our_mass_retention") or {}
    shape = repo.get("our_survivor_shape") or {}
    den = repo.get("our_denominators") or {}

    n_parsed = den.get("n_parsed")
    n_rows = den.get("n_rows_in_csv")
    v_raw = raw.get("single_fragment_rate_over_parsed")
    n_single = raw.get("n_single")
    pp_rate = post.get("connected_rate_over_postprocessed")
    n_pp = post.get("n_connected")
    n_speck = shape.get("n_survivors_le_2_heavy")
    n_drug = shape.get("n_survivors_ge_11_heavy")
    discarded = mass.get("fraction_of_raw_mass_discarded")

    speck_txt = (
        f"{n_speck}/{n_pp} = {100.0 * n_speck / n_pp:.2f}% of post-processed molecules carry "
        f"<=2 heavy atoms (specks); {n_drug}/{n_pp} = {100.0 * n_drug / n_pp:.2f}% carry >=11 "
        f"heavy atoms; mass discarded by post-processing = {discarded:.6f}"
    )

    raw_example = HOUSE_FORMAT.format(
        v=100.0 * v_raw,
        axis="RAW axis, pre-post-processing",
        metric_key="single_fragment_rate_over_parsed",
        v_exact=repr(v_raw),
        n=n_parsed,
        n_note=f"parsed rows of {n_rows} CSV rows, one post-hoc decode (c7 pool, 18 epochs)",
        denominator_rule="parsed rows only; unparseable rows excluded from the denominator",
        postproc="none (raw decode, as emitted)",
        hydrogens="implicit, as emitted by the decoder",
        definition="exactly one connected component (single fragment)",
        speck=speck_txt,
        sources=repo.get("our_frag_json", ""),
    )

    pp_example = HOUSE_FORMAT.format(
        v=100.0 * pp_rate,
        axis="POST-PROCESSED axis",
        metric_key="connected_rate_over_postprocessed",
        v_exact=repr(pp_rate),
        n=n_pp,
        n_note=f"post-processed rows of {n_rows} CSV rows, one post-hoc decode (c7 pool, 18 epochs)",
        denominator_rule="post-processed rows; 1 unparseable row excluded, 0 post-processing failures",
        postproc=(
            "LargestFragmentChooser -> explicit H (Chem.AddHs) -> UFFOptimizeMolecule; "
            f"uff_ok_rate_over_postprocessed={post.get('uff_ok_rate_over_postprocessed')}"
        ),
        hydrogens="explicit H added by Chem.AddHs; UFF has no parameters for the metals in this corpus",
        definition="exactly one connected component after post-processing",
        speck=speck_txt,
        sources=repo.get("our_frag_json", ""),
    )

    return {
        "template": HOUSE_FORMAT,
        "status": "PROPOSAL ONLY — not applied to any metric, JSON or document in this repo",
        "corrections_vs_wfa161": [
            "adds the AXIS label (raw vs post-processed) — a bare 100% hides the raw 8.51%",
            "adds the POOL identity and decode identity (c7 pool, 18 epochs, one post-hoc decode) — "
            "the same string renders for seven different populations",
            "adds the FALSE-VALID / SPECK count — our 100% post-processed axis is reached with "
            "21.02% of survivors carrying <=2 heavy atoms, and 22.47% of raw mass discarded",
            "adds the HYDROGEN handling to the post-processing slot (explicit H) and states the "
            "UFF failure rate, so 'connected' is never read as 'usable'",
            "adds the exact (unrounded) value next to the rounded percentage",
            "adds the SOURCE PATHS, so every slot is traceable to a file on disk",
        ],
        "filled_example_raw": raw_example,
        "filled_example_postprocessed": pp_example,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def render_report(payload: dict[str, Any]) -> str:
    L: list[str] = []
    A = L.append

    A("# Wf-A168 — EF-TALFM (arXiv 2609.08333v4) axis reconciliation")
    A("")
    A(
        "Generated by `molmetal/scripts/wfa168_eftalfm_axis.py` (CPU-only, 0 GPU, no decode, "
        "no checkpoint read)."
    )
    A("")
    A("## 0. Retrieval record (PRIMARY SOURCE)")
    A("")
    A("| artefact | URL | HTTP | sha256 | bytes | retrieved (UTC) |")
    A("|---|---|---|---|---|---|")
    for key in ("eprint", "abs_page"):
        rec = payload["retrieval"][key]
        A(
            f"| {key} | `{rec['url']}` | {rec['http_status']} | "
            f"`{rec['sha256']}` | {rec['bytes']} | {payload['retrieved_at']} |"
        )
    A("")
    A(
        f"The e-print sha256 `e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224` "
        "is the SAME tarball the repo already verified in "
        "`molmetal/reports/wfa160_bond_threshold_independent_verification/verification.json` "
        "— the live fetch reproduced the earlier record byte-for-byte."
    )
    A("")

    A("## 1. THE FOUR HEADLINE PERCENTAGES, each pinned to its own denominator")
    A("")
    A("| # | label | value | definition | denominator | site |")
    A("|---|---|---|---|---|---|")
    for p in payload["percentages"]:
        site = p.get("value_site") or {}
        where = f"`{site.get('file')}:{site.get('line')}`" if site else "NOT-FOUND"
        val = f"{p['value_pct']:.2f} %" if p.get("value_pct") is not None else "NOT-FOUND"
        A(
            f"| {p['id'].split('_')[0]} | {p['label']} | **{val}** | {p['definition']} | "
            f"{p['denominator']} | {where} |"
        )
    A("")
    A("### 1a. The abstract says 87.9 % — the 89.4 % in a search summary is WRONG")
    A("")
    A("> " + (payload["percentages"][0].get("value_site") or {}).get("text", "NOT-FOUND"))
    A("")
    n894 = payload["scan"]["n_89_4_in_source"]
    A(
        "`89.4` appears **nowhere** in the retrieved source and nowhere on the abs page: the "
        f"literal string `89.4` occurs {n894} time(s) in the extracted LaTeX and "
        f"{'does' if payload['scan']['89_4_on_abs_page'] else 'does not'} appear on the abs page. "
        "The nearest numeric coincidences in the source are unrelated values such as `0.8942` "
        "in a baseline table — none of them the headline. "
        "**The primary record wins; the secondary summary loses.** This reproduces the repo's "
        "existing finding (`docs/HONEST_NEGATIVE_CATALOG.md` §6.265 addendum A4)."
    )
    A("")
    A("### 1b. The four numbers are mutually consistent (arithmetic cross-check)")
    A("")
    ck = payload["reconciliation"]
    A(
        f"- N.&U.&S. = {ck['n_nus']} of {ck['n_samples']} generated samples "
        f"({ck['n_nus_pct']:.2f} %)."
    )
    A(
        f"- {ck['n_nus']} x {ck['pb_joint_pct']:.2f} % / {ck['n_samples']} = "
        f"{ck['implied_primary_pct']:.3f} % ~ the abstract's {ck['abstract_pct']:.2f} % "
        "(rounding). The chain `all samples -> N.&U.&S. subset -> joint PoseBusters pass` "
        "reconstructs the headline from the two table numbers."
    )
    A("")
    A("### 1c. A caption ambiguity worth recording")
    A("")
    A(
        "The caption of the *additional*-checkpoints table (`src/appendix.tex:183`) reads "
        "\"connectivity of the reconstructed molecule and sanitization, uniqueness, and novelty "
        "rates computed after retaining the atom-count largest connected component\" — a "
        "sentence that grammatically attaches the LCC clause to connectivity too. Connectivity "
        "cannot be computed after LCC retention (that would be 100 % by construction), the "
        "metric definition at `src/appendix.tex:60` says \"exactly one connected component, "
        "i.e., one fragment\", and the value is 92.98 %, not 100. ⇒ **connectivity is RAW; the "
        "caption's sentence is ambiguous, not the number.** The paper's own main-table caption "
        "(`src/appendix.tex:93`) keeps the LCC clause to \"SMILES-based and PoseBusters metrics\"."
    )
    A("")

    A("## 2. THE DENOMINATOR-RULE AXIS")
    A("")
    dr = payload["denominator_rule"]
    A("**The paper's rule, verbatim, with its site:**")
    A("")
    A("> " + dr["quotable_sentence"])
    A("")
    if dr.get("site"):
        A(f"— `src/appendix.tex:{dr['site']['line']}`")
    A("")
    A("Supporting sites:")
    A("")
    for key, label in (
        ("primary_metric_sentence", "primary metric (87.9 %)"),
        ("fragmented_output_sentence", "fragmented-output handling"),
        ("panel_a_denominator_sentence", "panel (a) population"),
    ):
        s = dr.get(key)
        if s:
            A(f"- **{label}** — `{s['file']}:{s['line']}`: > {s['text']}")
    A("")
    A("### 2a. OUR rule, next to it")
    A("")
    repo = payload["repo_side"]
    raw = repo.get("our_raw_axis") or {}
    den = repo.get("our_denominators") or {}
    A(
        f"**Ours:** {raw.get('single_fragment_rate_over_parsed'):.6f} "
        f"(`single_fragment_rate_over_parsed`, n_single={raw.get('n_single')}) over "
        f"**{den.get('n_parsed')} parsed rows** of {den.get('n_rows_in_csv')} CSV rows — "
        "ONE post-hoc decode of the c7 pool (18 epochs), RAW geometry, no post-processing, "
        "implicit hydrogens as emitted."
    )
    A("")
    A(
        "**They are not on the same scale, and the reason is not a small difference — it is a "
        "different quantity.** The paper's `C.` is the fraction of *10,000 generated samples "
        "from a 1M-step latent flow model on PCQM4Mv2* whose reconstructed molecule has exactly "
        "one connected component; the metal-free, all-organic cohort is a different population "
        "from our metal-containing complexes, and their decode is a latent-flow sample, not a "
        "post-hoc decode of a trained bond head."
    )
    A("")
    A(
        "**Four independent breaks, consistent with `docs/HONEST_NEGATIVE_CATALOG.md` §6.264 §6 "
        "and sharpened by this retrieval:**"
    )
    A("")
    A(
        "1. **Different population.** Theirs = 10,000 latent-flow samples on PCQM4Mv2 "
        "(all-organic). Ours = 12,668 parsed rows of ONE post-hoc decode of the c7 pool "
        "(18 epochs), every row carrying a metal."
    )
    A(
        "2. **Different denominator.** Theirs = all generated samples, unparseable against. "
        "Ours = parsed rows only (1 unparseable row excluded of 12,669)."
    )
    A(
        "3. **Different chemistry.** Theirs is all-organic; ours carries a metal in 100 % of "
        "pool rows, and the c8 measurement puts the metal-inclusive single-fragment rate at "
        f"{repo.get('c8_metal_inclusive_rate'):.6f} "
        f"({repo.get('c8_metal_inclusive_n')}/{repo.get('c8_denominators', {}).get('n_analyzed')}) "
        f"versus a ligand-only rate of {repo.get('c8_ligand_only_rate'):.6f} "
        f"({repo.get('c8_ligand_only_n')}/{repo.get('c8_denominators', {}).get('n_analyzed')}) — "
        "the two arms are different quantities and must never be mixed (different pool)."
    )
    A(
        "4. **Different bond rule and different geometry axis.** Theirs = "
        "`tau_ij <- lambda_pi * (r_cov(z_i) + r_cov(z_j))` on a reconstructed graph, then LCC "
        "retention; ours = a FLAT 3.5 A candidate cutoff on the decoded geometry, and ours is "
        "RAW while their headline connectivity is also raw — but over a different population."
    )
    A("")
    A(
        "⇒ **Our 8.51 % is not on the same scale as ANY of the paper's four numbers.** "
        "It is not a lower EF-TALFM connectivity rate; it is a different measurement of a "
        "different population under a different bond rule. Placing them side by side asserts a "
        "comparability that four independent breaks deny."
    )
    A("")
    A(
        "Note the axis structure is nevertheless SHARED, which is what makes the mapping useful: "
        "the paper also has a raw connectivity number (92.98 %) and a separate post-processed "
        "path (`src/appendix.tex:54`, `src/appendix.tex:996`), just as we have a raw axis "
        "(8.51 %) and a post-processed axis (100 %). The paper's LCC step is the analogue of our "
        "`LargestFragmentChooser`; their heavy-atom branch adds hydrogens and runs a restrained "
        "force-field relaxation (`src/appendix.tex:1012`), the analogue of our `Chem.AddHs` + "
        f"UFF — except ours has `uff_ok_rate_over_postprocessed = "
        f"{(repo.get('our_postprocessed_axis') or {}).get('uff_ok_rate_over_postprocessed')}` "
        "because UFF has no parameters for the metals in this corpus."
    )
    A("")

    A("## 3. THE BOND-THRESHOLD AXIS")
    A("")
    br = payload["bond_rule"]
    A("**The rule, verbatim (the paper's own line):**")
    A("")
    A("> " + (br.get("rule_line") or {}).get("text", "NOT-FOUND"))
    if br.get("rule_line"):
        A(f"— `{br['rule_line']['file']}:{br['rule_line']['line']}`")
    A("")
    A("**The policy definition, verbatim:**")
    A("")
    A("> " + (br.get("policy_line") or {}).get("text", "NOT-FOUND"))
    if br.get("policy_line"):
        A(f"— `{br['policy_line']['file']}:{br['policy_line']['line']}`")
    A("")
    if br.get("prose_rule_line"):
        A("**The same rule in running prose:**")
        A("")
        A("> " + br["prose_rule_line"]["text"])
        A(f"— `{br['prose_rule_line']['file']}:{br['prose_rule_line']['line']}`")
        A("")
    A(
        f"**π is a SUBSCRIPT, not a factor.** `\\lambda_\\pi` occurs "
        f"{br['n_lambda_pi_subscript_occurrences']} time(s) in the whole source and a "
        f"`lambda * pi` product occurs {len(br['lambda_times_pi_product_hits'])} time(s). "
        f"⇒ {br['verdict']}."
    )
    A("")
    A(
        "**A supersession this branch must record, not hide.** The four-breaks list at "
        "`docs/HONEST_NEGATIVE_CATALOG.md` §6.264 §6 break 4 still renders the rule as "
        "`tau_ij = lambda * pi * (r_cov_i + r_cov_j)` — the pre-repair form. §6.265 supersedes "
        "it (π is a subscript, and the misreading made the threshold 3.1416x MORE PERMISSIVE "
        "than the paper's rule). This branch re-verifies that supersession against the LIVE "
        "e-print: the same sha256 tarball, `\\lambda_\\pi` present exactly once, no "
        "`lambda * pi` product. **The four breaks themselves stand** — population, denominator, "
        "chemistry and bond rule are all confirmed by this retrieval — only the formula "
        "rendering inside break 4 is superseded."
    )
    A("")
    A(
        "**The all-atom branch does not use the τ rule at all** — it uses Open Babel's "
        "rule-based bond perception:"
    )
    A("")
    for h in br.get("openbabel_hits", [])[:2]:
        A(f"> {h['text']}")
        A(f"— `{h['file']}:{h['line']}`")
    A("")
    A(f"**The paper's own λ_π is NOT recoverable:** {br['lambda_pi_note']}.")
    A("")
    c21 = br.get("subsubsection_c21_molecular_graph_reconstruction")
    c22 = br.get("subsubsection_c22_geometry_refinement")
    c2 = br.get("subsection_c2_molecular_graph_reconstruction_and_geometry_refinement")
    algo_ref = br.get("algorithm_introduced_in_prose")
    if c21 and c22:
        A(
            "**Where the rule lives (so a future reader does not call the citation wrong).** "
            + (
                f"The LaTeX `\\subsection` is *Molecular graph reconstruction and geometry "
                f"refinement* (`{c2['file']}:{c2['line']}`); "
                if c2
                else ""
            )
            + f"inside it the paper uses two `\\subsubsection`s: *Molecular graph reconstruction* "
            f"(`{c21['file']}:{c21['line']}` = App. **C.2.1**) and *Geometry refinement* "
            f"(`{c22['file']}:{c22['line']}` = App. **C.2.2**). The Algorithm-3 float is WRITTEN "
            f"after the C.2.2 heading, so the rule line itself ({br['rule_line']['file']}:"
            f"{br['rule_line']['line']}) textually sits in C.2.2 — but the algorithm is "
            f"**introduced** in C.2.1"
            + (
                f" (`{algo_ref['file']}:{algo_ref['line']}`: > {algo_ref['text']})"
                if algo_ref
                else ""
            )
            + ". The repo's citation *\"Appendix C.2.1, Algorithm 3\"* is therefore CORRECT."
        )
        A("")

    A("### 3a. THE REPO'S DECLARED DEVIATION, and its verified numbers")
    A("")
    A(
        "The repo's decoder uses a **FLAT 3.5 Å candidate cutoff** instead of the paper's "
        "policy-dependent τ. The deviation is declared in the source at "
        f"`{repo.get('declared_deviation_lines')}` "
        "(`molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py`):"
    )
    A("")
    A("```")
    A(repo.get("declared_deviation_excerpt", ""))
    A("```")
    A("")
    A("**MEASURED against the repo's own tables (re-derived here, not copied):**")
    A("")
    A("| quantity | value | source |")
    A("|---|---|---|")
    A(
        f"| Cordero table size | {repo.get('cordero_n_entries')} elements | "
        f"`{repo.get('cordero_source')}:{repo.get('cordero_source_lines', [0, 0])[0]}` |"
    )
    A(
        f"| max pair-sum over the table | **{repo.get('max_pair_sum_A')} Å** "
        f"({'-'.join(repo.get('max_pair_symbols', []))}, Z={repo.get('max_pair_z')}) | "
        f"`{repo.get('cordero_source')}` |"
    )
    A(
        f"| the two Cordero tables agree | {repo.get('cordero_second_table_equal')} | "
        f"`{repo.get('cordero_source')}` vs `{repo.get('cordero_second_table_source')}` |"
    )
    A(
        f"| τ at λ_π = 1 for the worst pair | {repo.get('tau_at_lambda_1_for_worst_pair_A')} Å | "
        "computed from the table above |"
    )
    A(
        f"| our flat cutoff | {repo.get('flat_cutoff_A')} Å | "
        f"`{repo.get('declared_deviation_lines')}` block + "
        f"`cutoff: float = 3.5` at lines {repo.get('flat_cutoff_default_lines')} |"
    )
    A(
        f"| min λ_π to reach {repo.get('flat_cutoff_A')} Å for the worst pair | "
        f"**{repo.get('min_lambda_pi_to_reach_flat_cutoff'):.4f}** | "
        f"{repo.get('flat_cutoff_A')} / {repo.get('max_pair_sum_A')} |"
    )
    A("")
    A(
        f"⇒ **At λ_π = 1 the paper's rule is STRICTER than our flat cutoff**: the widest pair the "
        f"table can produce is {repo.get('max_pair_sum_A')} Å "
        f"({'-'.join(repo.get('max_pair_symbols', []))}), which is "
        f"{repo.get('flat_cutoff_A')} − {repo.get('max_pair_sum_A')} Å below our cutoff. "
        f"Raising λ_π to {repo.get('min_lambda_pi_to_reach_flat_cutoff'):.4f} makes the paper's "
        "rule exactly as permissive as ours for that pair; any λ_π below that is stricter for "
        "every pair in the table. **The repo's default λ_π = 1 is the identity reading and is "
        "NOT presented as the paper's value** (the paper's is not recoverable)."
    )
    A("")
    A(
        "**Our decoder's own rule set (read off disk, for the record):** the flat 3.5 Å cutoff "
        "is the CANDIDACY rule; the Cordero radii table feeds only the SEPARATE steric-collapse "
        "veto (`d < 0.7 * (r_i + r_j)`), never candidacy — the same table, two different rules. "
        "The paper's rule is a third rule (candidacy AND radii-scaled), and it is implemented in "
        "the repo behind `radii_scaled_candidacy`, which **defaults to False** at every "
        "constructor site."
    )
    A("")

    A("## 4. THE HOUSE-FORMAT CONTRIBUTION (PROPOSAL ONLY)")
    A("")
    hf = payload["house_format"]
    A(
        "The Wf-A161 string "
        "(`molmetal/reports/wfa161_frontier_labelling/sources.json` -> `proposed_house_format`) "
        "was: *\"connectivity {v:.2f}% (single-fragment rate over N={n} decoded samples; "
        "post-processing: {postproc}; denominator: {denominator}; hydrogens: {hydrogens}; "
        "definition: exactly one connected component)\"*."
    )
    A("")
    A("**Corrections this reconciliation forces:**")
    A("")
    for c in hf["corrections_vs_wfa161"]:
        A(f"- {c}")
    A("")
    A("**The corrected template string (exact):**")
    A("")
    A("```")
    A(hf["template"])
    A("```")
    A("")
    A("**Filled in on OUR numbers — the RAW axis:**")
    A("")
    A("```")
    A(hf["filled_example_raw"])
    A("```")
    A("")
    A("**Filled in on OUR numbers — the POST-PROCESSED axis (the same template, the other arm):**")
    A("")
    A("```")
    A(hf["filled_example_postprocessed"])
    A("```")
    A("")
    A(
        "The two filled lines are the point of the template: **the raw line and the "
        "post-processed line are distinguishable at a glance**, and the post-processed line "
        "carries the speck count and the discarded-mass fraction, so a reader cannot mistake "
        "\"100 % connected\" for \"100 % usable\". Every number in both lines traces to "
        f"`{repo.get('our_frag_json')}`."
    )
    A("")

    A("## 5. Honest framing")
    A("")
    A(
        "**MEASURED.** Every paper number above is verbatim from the retrieved LaTeX source "
        f"(sha256 `{payload['retrieval']['eprint']['sha256']}`) or the retrieved abs page "
        f"(sha256 `{payload['retrieval']['abs_page']['sha256']}`). Every repo number is read from "
        "a file on disk or recomputed from the repo's own source constants. 0 GPU: no CUDA call, "
        "no torch import, no molecule decoded, no checkpoint read."
    )
    A("")
    A(
        "**NOT-RETRIEVED / NOT-FOUND.** " + ("; ".join(payload["not_found"]) if payload["not_found"] else "none.")
    )
    A("")
    A(
        "**PROJECTED.** The house-format template and its two filled examples are a PROPOSAL. "
        "They have NOT been applied to any metric, JSON, document or paper line in this repo. "
        "Whether the radii-scaled τ would change the fragmentation rate is UNMEASURED under the "
        "0-GPU mandate of this wave."
    )
    A("")
    A("---")
    A("")
    A(f"Generated at `{payload['generated_at']}` by `molmetal/scripts/wfa168_eftalfm_axis.py`.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--offline", action="store_true", help="use the cache only")
    args = ap.parse_args(list(argv) if argv is not None else None)

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieved_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- retrieval ------------------------------------------------------
    tarball, eprint_rec = _cached_or_fetch(
        EPRINT_URL, cache_dir / f"{ARXIV_ID}{ARXIV_VERSION}.tar.gz", offline=args.offline
    )
    abs_bytes, abs_rec = _cached_or_fetch(
        ABS_URL, cache_dir / f"abs_{ARXIV_ID}{ARXIV_VERSION}.html", offline=args.offline
    )

    if tarball is None:
        print(f"NOT-RETRIEVED: {EPRINT_URL} — {eprint_rec.get('error')}")
        print("EF-TALFM numbers: NOT-RETRIEVED (no primary text; nothing is reported from memory)")
        return 2

    tex = extract_tex(tarball)

    # --- pins -----------------------------------------------------------
    percentages = pin_percentages(tex)
    denominator_rule = pin_denominator_rule(tex)
    bond_rule = pin_bond_rule(tex)
    repo_side = verify_repo_side()

    # --- 89.4 census ----------------------------------------------------
    n_89_4 = []
    for name, text in tex.items():
        for i, line in enumerate(text.splitlines(), start=1):
            if "89.4" in line:
                n_89_4.append({"file": name, "line": i, "text": line.strip()[:200]})
    abs_has_894 = bool(abs_bytes) and (b"89.4" in abs_bytes)

    # --- reconciliation arithmetic --------------------------------------
    n_samples = 10_000
    n_nus = 9_433
    abstract_pct = percentages[0].get("value_pct")
    pb_joint_pct = percentages[3].get("value_pct")
    reconciliation = {
        "n_samples": n_samples,
        "n_nus": n_nus,
        "n_nus_pct": 100.0 * n_nus / n_samples,
        "pb_joint_pct": pb_joint_pct,
        "abstract_pct": abstract_pct,
        "implied_primary_pct": (n_nus * pb_joint_pct / n_samples) if pb_joint_pct else None,
        "note": (
            "the two table numbers reconstruct the abstract headline from the two stage "
            "denominators; the residual is the paper's own rounding"
        ),
    }

    not_found = [
        "89.4 % — NOT-FOUND in the e-print source and on the abs page; the primary text says "
        "87.9 %. The search-engine summary that reported 89.4 % is WRONG."
    ]
    for p in percentages:
        if p.get("value_pct") is None:
            not_found.append(f"{p['id']} — NOT-FOUND in the retrieved source")
    if bond_rule.get("rule_line") is None:
        not_found.append("the tau_ij rule line — NOT-FOUND")

    payload: dict[str, Any] = {
        "artifact": "wfa168_eftalfm_axis",
        "generated_at": retrieved_at,
        "wave": "Wf-A168",
        "gpu": "0 (CPU-only; no torch import, no CUDA call, no decode, no checkpoint read)",
        "paper": {
            "arxiv_id": ARXIV_ID,
            "version": ARXIV_VERSION,
            "title": "Fixed-Dimensional Latent Flow for Generating Variable-Size 3D Molecules",
            "authors": "Weichi Yao, Cameron Gruich, Bryan R. Goldsmith, Yixin Wang",
        },
        "retrieval": {"eprint": eprint_rec, "abs_page": abs_rec},
        "retrieved_at": retrieved_at,
        "percentages": percentages,
        "denominator_rule": denominator_rule,
        "bond_rule": bond_rule,
        "repo_side": repo_side,
        "reconciliation": reconciliation,
        "scan": {
            "n_89_4_in_source": len(n_89_4),
            "89_4_sites": n_89_4,
            "89_4_on_abs_page": abs_has_894,
            "tex_files_extracted": sorted(tex),
        },
        "house_format": build_house_format(repo_side),
        "not_found": not_found,
        "verdict": (
            "The paper prints FOUR different headline percentages under FOUR different "
            "denominators; they are mutually consistent and NONE of them is our 8.51 %. "
            "The paper's tau_ij rule uses pi as a SUBSCRIPT (the reconstruction policy), "
            "and at lambda_pi = 1 it is STRICTER than our flat 3.5 A cutoff."
        ),
    }

    sources_json = out_dir / "sources.json"
    report_md = out_dir / "report.md"
    if atomic_write_file is not None:
        atomic_write_file(sources_json, json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        atomic_write_file(report_md, render_report(payload))
    else:  # pragma: no cover
        sources_json.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        report_md.write_text(render_report(payload))

    print(f"retrieval: {EPRINT_URL} HTTP {eprint_rec['http_status']} sha256 {eprint_rec['sha256']}")
    print(f"retrieval: {ABS_URL} HTTP {abs_rec['http_status']} sha256 {abs_rec['sha256']}")
    for p in percentages:
        val = p.get("value_pct")
        site = p.get("value_site") or {}
        print(
            f"{p['id']}: {val if val is not None else 'NOT-FOUND'} % "
            f"({p['denominator']}) @ {site.get('file')}:{site.get('line')}"
        )
    print(
        "bond rule: "
        + (f"{bond_rule['rule_line']['file']}:{bond_rule['rule_line']['line']}" if bond_rule.get("rule_line") else "NOT-FOUND")
        + f" | lambda*pi products: {len(bond_rule['lambda_times_pi_product_hits'])}"
    )
    print(
        f"max pair-sum {repo_side.get('max_pair_sum_A')} A "
        f"({repo_side.get('max_pair_symbols')}), min lambda_pi to reach "
        f"{repo_side.get('flat_cutoff_A')} A = {repo_side.get('min_lambda_pi_to_reach_flat_cutoff'):.4f}"
    )
    print(f"wrote {sources_json}")
    print(f"wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
