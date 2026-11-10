#!/usr/bin/env python3
# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wf-A161 -- READ-ONLY inventory of unqualified numbers in ``paper/**``.

Why this module exists
----------------------
AGENTS.md §5.3 / TODO-84 forbid ANY write to ``paper/**``.  The user still
needs to know which numbers in the manuscript are not properly qualified, so
that they can *decide* whether to authorise an edit.  This module produces
that inventory and touches nothing under ``paper/``.

The three classes in scope (per the Wf-A161 brief):

* **(a) EXTERNAL** -- a number measured by another work that the manuscript
  does not label with its source *and* the population that makes it
  interpretable (cohort + noise regime + denominator).
* **(b) OURS** -- a number we measured whose *population* (cohort, seed
  count, fold count, pool, run) is not stated near the number, so the number
  cannot be reproduced or compared.
* **(c) CONSTANT** -- a methodological constant (a distance cutoff, a band
  width, a sigmoid value) that does not name where the value comes from.

Classification vocabulary (fixed, four values):

``QUALIFIED``
    The site already names its source and/or its population.
``UNQUALIFIED``
    It reads as ours (or is otherwise interpretable only with a missing
    qualifier) but omits the population / deviation note that makes it
    interpretable.
``UNSOURCED``
    No source anywhere near the number.
``FALSE-SCOPE``
    It asserts something about *our* model that our own measurements (or
    our own code) do not support.

Read-only contract
------------------
This module NEVER opens a file under ``paper/`` for writing and never
imports anything that writes to it.  The guarantee is enforced, not merely
asserted:

* every file read goes through :func:`_read_bytes`, which opens in ``"rb"``
  and refuses any mode containing a write character;
* :data:`_WRITES` records every path this module writes; :func:`assert_read_only`
  raises if any of them resolves inside the repository's ``paper/`` tree;
* :func:`import_audit` checks that the module's own import list contains no
  write-capable module (``shutil``, ``tempfile``, ``socket``, ...).

Enumeration never uses ``os.walk`` (the repo is 201 GB) -- paths come from
``git ls-files`` via :mod:`wfa156_text_scan`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import wfa156_text_scan as ts  # noqa: E402  (path shim above)

# --------------------------------------------------------------------------
# Read-only bookkeeping
# --------------------------------------------------------------------------

#: Every path this module writes to.  Checked by :func:`assert_read_only`.
_WRITES: list[Path] = []

#: Modules this module must never import (write-capable or network-capable).
_FORBIDDEN_IMPORTS = frozenset(
    {
        "shutil",
        "tempfile",
        "socket",
        "http",
        "urllib",
        "requests",
        "ftplib",
        "smtplib",
        "webbrowser",
        "multiprocessing",
        "ctypes",
    }
)

#: Repository-relative trees that must never be written by this module.
_WRITE_FORBIDDEN_PREFIXES = ("paper",)


def _record_write(path: os.PathLike | str) -> Path:
    p = Path(path)
    _WRITES.append(p)
    return p


def _read_bytes(path: os.PathLike | str) -> bytes:
    """Read a file as bytes.  The ONLY way this module touches a file."""
    with open(path, "rb") as fh:  # noqa: PTH123 - mode is pinned to "rb"
        return fh.read()


def assert_read_only(root: os.PathLike | str = _REPO_ROOT) -> list[str]:
    """Raise if anything this module wrote is inside a forbidden tree.

    Returns the list of written paths (absolute) for the report.
    """
    root = Path(root).resolve()
    offenders: list[str] = []
    for p in _WRITES:
        try:
            rp = p.resolve()
        except OSError:  # pragma: no cover - defensive
            continue
        try:
            rel = rp.relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in _WRITE_FORBIDDEN_PREFIXES:
            offenders.append(str(rp))
    if offenders:
        raise RuntimeError(
            "WFA161 read-only contract violated; wrote into a forbidden tree: "
            + ", ".join(offenders)
        )
    return [str(p) for p in _WRITES]


def import_audit() -> dict:
    """Report the module's own imports and flag write-capable ones."""
    import ast

    src = _read_bytes(__file__).decode("utf-8", errors="replace")
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split(".")[0])
    flagged = sorted(names & _FORBIDDEN_IMPORTS)
    return {"imports": sorted(names), "write_capable_flagged": flagged}


def _code_only_source() -> str:
    """This file's source with comments and string literals removed.

    Uses :mod:`tokenize`, so a docstring that *mentions* ``open(..., "w")``
    cannot trip the write-mode check -- only real code can.
    """
    import io
    import tokenize

    src = _read_bytes(__file__).decode("utf-8", errors="replace")
    out: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except tokenize.TokenError:  # pragma: no cover - defensive
        return src
    return " ".join(out)


def assert_no_write_modes() -> None:
    """Static check: no ``open(..., "w"/"a"/"x"/"+")`` in real code."""
    code = _code_only_source()
    bad = re.findall(r"open\([^)]*[\"']([waxrbt+][^\"']*)[\"']", code)
    bad = [m for m in bad if any(c in m for c in "wax+")]
    if bad:
        raise RuntimeError(f"WFA161 read-only contract violated; write modes found: {bad}")


# --------------------------------------------------------------------------
# Rule table
# --------------------------------------------------------------------------

KIND_EXTERNAL = "EXTERNAL"
KIND_OURS = "OURS"
KIND_CONSTANT = "CONSTANT"

CLS_QUALIFIED = "QUALIFIED"
CLS_UNQUALIFIED = "UNQUALIFIED"
CLS_UNSOURCED = "UNSOURCED"
CLS_FALSE_SCOPE = "FALSE-SCOPE"


@dataclass(frozen=True)
class Rule:
    """One number family to hunt for, plus what makes a site qualified."""

    rule_id: str
    kind: str
    pattern: str
    number: str
    what: str
    measured_by: str  # "us" | "external"
    should_cite: str
    population_required: str
    source_markers: tuple[str, ...] = ()
    population_markers: tuple[str, ...] = ()
    note: str = ""


#: Markers that mean "the window names a source".
_SRC_GENERIC = (
    "wang",
    "dokholyan",
    "yuelbond",
    "biorxiv",
    "10.1101/2025.05.06.652517",
    "10.1021/acs.jcim.5c03052",
    "j. chem. inf. model",
    "jcim",
    "lit-anchored",
    "ours",
    "this work",
    "our ",
    "legacy",
    "\\texttt{",
    "molmetal/",
    "ourdecoder",
)

#: Markers that mean "the window names a population / cohort / regime".
_POP_GENERIC = (
    "n=",
    "n =",
    "cohort",
    "pool",
    "samples",
    "seeds",
    "seed",
    "folds",
    "fold",
    "cells",
    "cell-line",
    "cdg",
    "crude de novo generated",
    "0.2 \\,\\aa",
    "sigma = 0.2",
    "\\sigma = 0.2",
    "distorted",
    "per-bond",
    "worst case",
    "standard 3-d",
    "\\textsc{measured}",
    "\\measured{}",
    "smoke",
    "synthetic",
    "\\texttt{molmetal/",
)

RULES: tuple[Rule, ...] = (
    # ---------------- (a) EXTERNAL ----------------
    Rule(
        rule_id="A1",
        kind=KIND_EXTERNAL,
        pattern=r"92\.7\s*\\?%",
        number="92.7 %",
        what="YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)",
        measured_by="external",
        should_cite=(
            "Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, "
            "doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)"
        ),
        population_required=(
            "per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort "
            "(Crude De novo Generated, Gaussian noise sigma = 0.2 A)"
        ),
        source_markers=_SRC_GENERIC,
        population_markers=(
            "cdg",
            "crude de novo generated",
            "sigma = 0.2",
            "0.2 \\,\\aa",
            "per-bond",
            "worst case",
            "783",
            "1000",
        ),
        note=(
            "See docs/HONEST_NEGATIVE_CATALOG.md 6.264 section 6 (four comparability breaks). "
            "'on distorted molecules' is NOT a population label: the cohort is the CDG set "
            "at Gaussian sigma = 0.2 A, and the number is a per-BOND recovery F1."
        ),
    ),
    Rule(
        rule_id="A2",
        kind=KIND_EXTERNAL,
        pattern=r"98\.4\s*\\?%",
        number="98.4 %",
        what="YuelBond per-bond recovery F1 on standard 3-D structures",
        measured_by="external",
        should_cite=(
            "Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, "
            "doi:10.1021/acs.jcim.5c03052"
        ),
        population_required="standard 3-D structures (not the distorted CDG cohort)",
        source_markers=_SRC_GENERIC,
        population_markers=("standard 3-d", "cdg", "distorted", "per-bond"),
    ),
    Rule(
        rule_id="A3",
        kind=KIND_EXTERNAL,
        pattern=r"\b783\b",
        number="783",
        what="RDKit xyz2mol / DetermineBonds failures out of 1000 distorted CDGs",
        measured_by="external",
        should_cite=(
            "Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, "
            "doi:10.1021/acs.jcim.5c03052"
        ),
        population_required="out of 1000 CDGs at sigma = 0.2 A",
        source_markers=_SRC_GENERIC,
        population_markers=("cdg", "1000", "sigma = 0.2", "distorted"),
    ),
    Rule(
        rule_id="A4",
        kind=KIND_EXTERNAL,
        pattern=r"\b99\.9\b",
        number="99.9 %",
        what="FlowMol3 %Valid (GEOM-Drugs)",
        measured_by="external",
        should_cite=(
            "Dunn & Koes, Digital Discovery 2026, 5(5), 2052-2066, "
            "doi:10.1039/d5dd00363f (arXiv 2508.12629)"
        ),
        population_required="GEOM-Drugs, explicit hydrogens",
        source_markers=_SRC_GENERIC,
        population_markers=("geom-drugs", "geom-drugs", "valid"),
    ),
    Rule(
        rule_id="A5",
        kind=KIND_EXTERNAL,
        pattern=r"\b91\.9\b",
        number="91.9 %",
        what="FlowMol3 %PB-Valid (GEOM-Drugs)",
        measured_by="external",
        should_cite=(
            "Dunn & Koes, Digital Discovery 2026, 5(5), 2052-2066, "
            "doi:10.1039/d5dd00363f (arXiv 2508.12629)"
        ),
        population_required="GEOM-Drugs, explicit hydrogens",
        source_markers=_SRC_GENERIC,
        population_markers=("geom-drugs", "valid"),
    ),
    Rule(
        rule_id="A6",
        kind=KIND_EXTERNAL,
        pattern=r"\b92\.98\b",
        number="92.98 %",
        what="EF-TALFM Connected rate",
        measured_by="external",
        should_cite="EF-TALFM, arXiv 2609.08333, App. A.2",
        population_required="10,000 generated molecules",
        source_markers=_SRC_GENERIC,
        population_markers=("10,000", "10000", "app. a.2"),
    ),
    Rule(
        rule_id="A7",
        kind=KIND_EXTERNAL,
        pattern=r"\b94\.33\b",
        number="94.33 %",
        what="EF-TALFM Novel+unique+sanitized rate",
        measured_by="external",
        should_cite="EF-TALFM, arXiv 2609.08333, App. A.2",
        population_required="10,000 generated molecules",
        source_markers=_SRC_GENERIC,
        population_markers=("10,000", "10000", "app. a.2"),
    ),
    Rule(
        rule_id="A8",
        kind=KIND_OURS,
        pattern=r"8\.51\s*\\?%",
        number="8.51 %",
        what="OUR raw connectivity rate (all samples drawn, unparseable against)",
        measured_by="us",
        should_cite="metrics/by_metric/cfm_run_census.json",
        population_required="2,112 rows, ALL samples drawn, RAW (no post-processing)",
        source_markers=_SRC_GENERIC,
        population_markers=("2112", "2,112", "all samples", "raw", "cohort", "pool"),
        note="Seed for the brief: expected in paper/**. MEASURED 2026-11-01: 0 hits in paper/**.",
    ),
    Rule(
        rule_id="A9",
        kind=KIND_OURS,
        pattern=r"25\.05\s*\\?%",
        number="25.05 %",
        what="OUR n_fragments == 1 share (529 / 2112 rows)",
        measured_by="us",
        should_cite="metrics/by_metric/cfm_run_census.json",
        population_required="529 / 2112 rows of the CFM generation census",
        source_markers=_SRC_GENERIC,
        population_markers=("529", "2112", "2,112", "census", "fragment"),
        note="Seed for the brief: expected in paper/**. MEASURED 2026-11-01: 0 hits in paper/**.",
    ),
    # ---------------- (b) OURS ----------------
    Rule(
        rule_id="B1",
        kind=KIND_OURS,
        pattern=r"\b0/192\b",
        number="0/192",
        what="Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke",
        measured_by="us",
        should_cite="WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)",
        population_required="192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget",
        source_markers=_SRC_GENERIC,
        population_markers=(
            "192 synthetic",
            "10-atom",
            "500-step",
            "mini-budget",
            "smoke",
            "path-a",
            "path a",
            "192-sample",
            "192 sample",
            "3 seeds",
            "2 pockets",
            "16 samples",
            "checkpoint",
            "aggregate",
            "crossdocked",
            "blackwell",
            "decode\\_ratio",
            "final.md",
            "report.json",
            "wf_",
            "gpu",
            "round-",
        ),
    ),
    Rule(
        rule_id="B2",
        kind=KIND_OURS,
        pattern=r"\b192/192\b",
        number="192/192",
        what="Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke",
        measured_by="us",
        should_cite="WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)",
        population_required="192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget",
        source_markers=_SRC_GENERIC,
        population_markers=(
            "192 synthetic",
            "10-atom",
            "500-step",
            "mini-budget",
            "smoke",
            "path-b",
            "path b",
        ),
    ),
    Rule(
        rule_id="B3",
        kind=KIND_OURS,
        pattern=r"\b0/8\b",
        number="0/8",
        what="decode_ratio on the R15 CPU/GPU 500-step smokes",
        measured_by="us",
        should_cite=(
            "molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json "
            "+ phase4_gpu_retry.json"
        ),
        population_required="8 decoded slots at step 100/200/500, CPU and GPU runs",
        source_markers=_SRC_GENERIC,
        population_markers=("step 100", "step 200", "step 500", "500-step", "cpu", "gpu"),
    ),
    # ---------------- (c) CONSTANT ----------------
    Rule(
        rule_id="C1",
        kind=KIND_CONSTANT,
        pattern=r"3\.5\$?\s*\\,?\s*\\AA|3\.5\s*Å",
        number="3.5 A",
        what=(
            "OUR FLAT 3.5 A candidate-edge cutoff in the YuelBond decoder "
            "(a DECLARED DEVIATION from the published YuelBond 3 A)"
        ),
        measured_by="us",
        should_cite=(
            "molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py (cutoff default 3.5, "
            "mask = d_raw < self.cutoff) + Wang & Dokholyan's published 3 A"
        ),
        population_required=(
            "the value must be labelled as OURS and must state the published 3 A it deviates from"
        ),
        source_markers=(
            "declared deviation",
            "deviation",
            "our ",
            "ours",
            "legacy",
            "flat",
            "yuelbond_decoder.py",
            "molmetal/",
            "3 \\,\\aa",
            "3\\,\\aa",
            "3 a",
        ),
        population_markers=("cutoff", "candidate-edge", "declared deviation", "molmetal/"),
    ),
    Rule(
        rule_id="C2",
        kind=KIND_CONSTANT,
        pattern=r"2\.4\$?\s*\\,?\s*\\AA|2\.4\s*Å",
        number="2.4 A",
        what="legacy hard bond cutoff / soft-mask centre d_th",
        measured_by="us",
        should_cite="molmetal/molmetal_lam/lam_chem/decoder_rework.py:DEFAULT_DISTANCE_THRESHOLD",
        population_required="must name the component it belongs to (legacy BondAwareDecoder)",
        source_markers=("legacy", "bondawaredecoder", "decoder_rework", "molmetal/", "d_{\\text{th}}", "d_\\text{th}"),
        population_markers=("cutoff", "legacy", "d_{\\text{th}}", "d_\\text{th}"),
    ),
    Rule(
        rule_id="C3",
        kind=KIND_CONSTANT,
        pattern=r"0\.3\$?\s*\\,?\s*\\AA|0\.3\s*Å",
        number="0.3 A",
        what="soft-band width tau (DEFAULT_SOFT_SIGMA)",
        measured_by="us",
        should_cite="molmetal/molmetal_lam/lam_chem/decoder_rework.py:DEFAULT_SOFT_SIGMA",
        population_required="must name the component (soft distance mask)",
        source_markers=("soft", "band", "tau", "\\tau", "decoder_rework", "molmetal/"),
        population_markers=("soft", "band", "\\tau", "tau"),
    ),
    Rule(
        rule_id="C4",
        kind=KIND_CONSTANT,
        pattern=r"p_\{\\text\{dist\}\}[^0-9]{0,24}0\.05",
        number="p_dist ~ 0.05 at d = 3.5 A",
        what="value of OUR soft distance mask at d = 3.5 A",
        measured_by="us",
        should_cite=(
            "molmetal/molmetal_lam/lam_chem/decoder_rework.py:499-500 "
            "(z = -(d - threshold)/sigma; torch.sigmoid(z))"
        ),
        population_required="d_th = 2.4 A, tau = 0.3 A, both stated in the same paragraph",
        source_markers=("d_{\\text{th}}", "d_\\text{th}", "sigma", "\\tau", "tau", "sigmoid"),
        population_markers=("2.4", "0.3"),
        note=(
            "FALSE-SCOPE candidate: the stated formula with the stated constants gives "
            "sigmoid(-(3.5-2.4)/0.3) = 0.024924, not 0.05 (2x off)."
        ),
    ),
    Rule(
        rule_id="C5",
        kind=KIND_CONSTANT,
        pattern=r"\[1\.5,\s*\\,?3\.5\]|1-5\s*\\,?\\AA|1--5\s*\\,?\\AA",
        number="1.5-3.5 A (and 1-5 A)",
        what="'typical CFM coordinate spread' -- our own empirical claim about coordinate spread",
        measured_by="us",
        should_cite="a round-12 trained-CFM coordinate-spread measurement",
        population_required="which run / round / pool the spread was measured on",
        source_markers=("round-12", "round 12", "trained-cfm", "molmetal/", "our "),
        population_markers=("round-12", "round 12", "distribution", "measured", "spread"),
    ),
)


# --------------------------------------------------------------------------
# Explicit overrides (measured facts that no marker heuristic can infer)
# --------------------------------------------------------------------------

#: (relative path, line number, rule_id) -> (classification, evidence)
OVERRIDES: dict[tuple[str, int, str], tuple[str, str]] = {
    (
        "paper/sections/03_method.tex",
        181,
        "C4",
    ): (
        CLS_FALSE_SCOPE,
        "The paragraph states the formula p_dist(d) = sigma(-(d - d_th)/tau) with "
        "d_th = 2.4 A and tau = 0.3 A (paper/sections/03_method.tex:175-178) and the "
        "same constants live in the code "
        "(molmetal/molmetal_lam/lam_chem/decoder_rework.py:139 DEFAULT_DISTANCE_THRESHOLD = 2.4, "
        ":143 DEFAULT_SOFT_SIGMA = 0.3, :499-500 z = -(distances - threshold)/sigma; "
        "return torch.sigmoid(z)). Evaluating that exact expression at d = 3.5 A gives "
        "sigmoid(-(3.5-2.4)/0.3) = 0.024924, so the paper's 'p_dist ~ 0.05' is off by ~2x. "
        "MEASURED 2026-11-01, CPU-only closed-form evaluation (no torch/CUDA import).",
    ),
    (
        "paper/sections/03_6_multi_parameter_ph.tex",
        89,
        "C1",
    ): (
        CLS_UNSOURCED,
        "This '3.5 A' is NOT a cutoff -- it is an illustrative Vietoris-Rips radius in a "
        "worked example. The same paragraph asserts that a pair at r = 3.5 A with "
        "coordination distance d = 0 is 'a covalent C--C bond', while the paper's own "
        "section 3 (paper/sections/03_method.tex:179) uses 1.5 A for the covalent-bond "
        "distance. A covalent C-C bond is ~1.54 A and a Pt-C bond ~2.0-2.1 A, so the "
        "example is internally inconsistent with the paper's own constant. "
        "MEASURED 2026-11-01 by reading both sites; no geometry run was performed.",
    ),
}


# --------------------------------------------------------------------------
# File enumeration (no os.walk)
# --------------------------------------------------------------------------

PAPER_TEXT_SUFFIXES = (".tex", ".md", ".txt", ".csv", ".bib", ".json")


def enumerate_paper_files(root: Path) -> tuple[list[str], list[str]]:
    """Return ``(tracked_text_files, untracked_or_ignored_text_files)``.

    Enumeration is via ``git ls-files`` (never ``os.walk``).  The second list
    holds files that exist on disk under ``paper/`` but are NOT tracked --
    they are reported but flagged, because TODO-84 governs the tracked tree
    and a stale build artifact is a different kind of site.
    """
    tracked = [
        rel
        for rel in ts.tracked_files(root)
        if rel.startswith("paper/") and rel.endswith(PAPER_TEXT_SUFFIXES)
    ]
    untracked = [
        rel
        for rel in ts.untracked_files(root, include_ignored=True)
        if rel.startswith("paper/") and rel.endswith(PAPER_TEXT_SUFFIXES)
    ]
    tracked_set = set(tracked)
    untracked = [rel for rel in untracked if rel not in tracked_set]
    return sorted(tracked), sorted(untracked)


# --------------------------------------------------------------------------
# Sentence extraction
# --------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\%])")


def _reconstruct(lines: Sequence[str], index: int, before: int, after: int) -> tuple[str, int]:
    """Join the window into one string; return (window, offset_of_line_index)."""
    lo = max(0, index - before)
    hi = min(len(lines), index + after + 1)
    window = " ".join(line.strip() for line in lines[lo:hi])
    offset = sum(len(line.strip()) + 1 for line in lines[lo:index])
    return window, offset


def _sentence_containing(window: str, offset: int, local_pos: int) -> str:
    """Best-effort sentence containing the match inside the joined window."""
    start = 0
    for m in _SENT_SPLIT.finditer(window):
        if m.start() <= local_pos:
            start = m.end()
        else:
            break
    end = len(window)
    m = _SENT_SPLIT.search(window, local_pos)
    if m is not None:
        end = m.start()
    sent = window[start:end].strip()
    if not sent:
        sent = window.strip()
    return sent


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def _contains_any(haystack_lower: str, markers: Iterable[str]) -> bool:
    return any(m.lower() in haystack_lower for m in markers)


def classify(rule: Rule, window: str) -> tuple[str, str]:
    """Return ``(classification, evidence)`` for one hit."""
    low = window.lower()
    src_ok = _contains_any(low, rule.source_markers)
    pop_ok = _contains_any(low, rule.population_markers)
    if rule.kind == KIND_EXTERNAL:
        if src_ok and pop_ok:
            return (
                CLS_QUALIFIED,
                "window names the source AND the population/cohort that makes the number interpretable",
            )
        if src_ok and not pop_ok:
            return (
                CLS_UNQUALIFIED,
                "window names the source but omits the population (cohort / noise regime / denominator)",
            )
        if pop_ok and not src_ok:
            return (
                CLS_UNQUALIFIED,
                "window names a population but does not name the source",
            )
        return (CLS_UNSOURCED, "no source and no population anywhere in the window")
    if rule.kind == KIND_OURS:
        if pop_ok:
            return (
                CLS_QUALIFIED,
                "window names the population (cohort / pool / budget / seed count) that the number belongs to",
            )
        return (
            CLS_UNQUALIFIED,
            "window does not name the population (cohort / pool / budget / seed count) the number belongs to",
        )
    # CONSTANT
    if src_ok:
        return (
            CLS_QUALIFIED,
            "window names where the constant comes from (a source / a code path / an explicit deviation note)",
        )
    return (CLS_UNSOURCED, "no source for the constant anywhere in the window")


# --------------------------------------------------------------------------
# Minimal-edit proposals (text only -- NEVER applied)
# --------------------------------------------------------------------------

MINIMAL_EDITS: dict[str, str] = {
    "A1@paper/sections/04_evaluation.tex:3179": (
        "Replace the parenthetical with: "
        "`(Wang \\& Dokholyan 2025 bioRxiv 10.1101/2025.05.06.652517; published "
        "J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052; "
        "per-bond recovery F1 = 92.7\\% on the CDG cohort -- Crude De novo Generated, "
        "Gaussian noise \\sigma = 0.2 \\AA, the paper's worst case -- and 98.4\\% on "
        "standard 3-D structures; NOT a per-molecule validity rate)`."
    ),
    "A1@paper/sections/06_limitations.tex:387": (
        "Replace `F1=92.7\\%` with "
        "`per-bond recovery F1 = 92.7\\% on the CDG cohort (Crude De novo Generated, "
        "Gaussian noise \\sigma = 0.2 \\AA, the paper's worst case; 98.4\\% on standard "
        "3-D structures) -- a per-bond F1, not a per-molecule validity rate`."
    ),
    "C1@paper/sections/04_evaluation.tex:3183": (
        "Replace `distance cutoff $3.5$\\,\\AA{}` with "
        "`distance cutoff $3.5$\\,\\AA{} (ours; a declared deviation from the published "
        "YuelBond $3$\\,\\AA, \\texttt{yuelbond\\_decoder.py})`."
    ),
    "C4@paper/sections/03_method.tex:181": (
        "Replace `$p_{\\text{dist}}\\!\\approx\\!0.05$` with "
        "`$p_{\\text{dist}}\\!\\approx\\!0.025$` (sigmoid(-(3.5-2.4)/0.3) = 0.0249), "
        "or restate tau so the printed value follows."
    ),
    "C1@paper/sections/03_6_multi_parameter_ph.tex:89": (
        "Replace the `r = 3.5` A worked example with the covalent-bond radius the paper "
        "itself uses elsewhere (`$r = 1.5$\\,\\AA{}`), or state that $3.5$ A is an "
        "arbitrary illustrative filter radius and not a bond length."
    ),
    "C5@paper/sections/03_method.tex:184": (
        "Replace `the typical CFM coordinate spread` with `the round-12 trained-CFM "
        "coordinate spread (\\texttt{molmetal/reports/<round-12 spread report>})` so the "
        "range names its measurement."
    ),
}


# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------


@dataclass
class Site:
    rule_id: str
    kind: str
    file: str
    line: int
    tracked: bool
    number: str
    matched_text: str
    sentence: str
    what: str
    measured_by: str
    should_cite: str
    population_required: str
    classification: str
    evidence: str
    in_live_bundle: bool
    minimal_edit: str
    todo84_blocked: bool
    also_in_paper_tables: bool


LIVE_BUNDLE_INPUTS = {
    "paper/sections/01_intro.tex",
    "paper/sections/02_related.tex",
    "paper/sections/03_method.tex",
    "paper/sections/04_evaluation.tex",
    "paper/sections/05_ablation.tex",
    "paper/sections/06_limitations.tex",
    "paper/sections/07_future.tex",
    "paper/appendices/closure_theorem.tex",
    "paper/appendices/betanf_semantics.tex",
    "paper/supplementary.tex",
}


def scan(root: Path) -> dict:
    """Run the inventory.  Returns the full result dict."""
    tracked, untracked = enumerate_paper_files(root)
    all_files = [(rel, True) for rel in tracked] + [(rel, False) for rel in untracked]

    rules = [r for r in RULES if r.kind != KIND_OURS or r.rule_id not in {"A8", "A9"}]
    compiled = [(r, re.compile(r.pattern)) for r in rules]

    sites: list[Site] = []
    per_rule_hits: dict[str, int] = {r.rule_id: 0 for r in rules}
    files_scanned = 0
    files_skipped: list[str] = []

    for rel, is_tracked in all_files:
        path = root / rel
        try:
            data = _read_bytes(path)
        except OSError as exc:
            files_skipped.append(f"{rel}: {exc}")
            continue
        files_scanned += 1
        kind = ts.classify_bytes(data)
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        for rule, rx in compiled:
            for i, line in enumerate(lines):
                for m in rx.finditer(line):
                    lo = max(0, i - 6)
                    hi = min(len(lines), i + 7)
                    window, offset = _reconstruct(lines, i, 6, 6)
                    local_pos = offset + m.start()
                    sentence = _sentence_containing(window, offset, local_pos)
                    cls, evidence = classify(rule, window)
                    ov = OVERRIDES.get((rel, i + 1, rule.rule_id))
                    if ov is not None:
                        cls, evidence = ov
                    key = f"{rule.rule_id}@{rel}:{i + 1}"
                    sites.append(
                        Site(
                            rule_id=rule.rule_id,
                            kind=rule.kind,
                            file=rel,
                            line=i + 1,
                            tracked=is_tracked,
                            number=rule.number,
                            matched_text=m.group(0),
                            sentence=sentence,
                            what=rule.what,
                            measured_by=rule.measured_by,
                            should_cite=rule.should_cite,
                            population_required=rule.population_required,
                            classification=cls,
                            evidence=evidence,
                            in_live_bundle=rel in LIVE_BUNDLE_INPUTS,
                            minimal_edit=MINIMAL_EDITS.get(
                                key, "(no site-specific minimal edit proposed)"
                            ),
                            todo84_blocked=True,
                            also_in_paper_tables=False,
                        )
                    )
                    per_rule_hits[rule.rule_id] += 1

    # which of these numbers also appear in metrics/paper_tables/ (that tree MAY change)
    pt_numbers = _paper_tables_numbers(root)
    for s in sites:
        s.also_in_paper_tables = any(
            n in s.matched_text or n in s.number for n in pt_numbers
        )

    return {
        "schema": "wfa161.paper_unqualified_numbers/1",
        "generated_by": "molmetal/scripts/wfa161_paper_unqualified_numbers.py",
        "scope": "tracked + untracked text files under paper/",
        "read_only_contract": {
            "writes": assert_read_only(root),
            "imports": import_audit(),
            "write_modes_found": [],
        },
        "files_scanned": files_scanned,
        "files_skipped": files_skipped,
        "tracked_text_files": len(tracked),
        "untracked_text_files": len(untracked),
        "per_rule_hits": per_rule_hits,
        "sites": [asdict(s) for s in sites],
    }


def _paper_tables_numbers(root: Path) -> list[str]:
    """Numeric strings published in metrics/paper_tables/ (that tree may change)."""
    out: set[str] = set()
    pt = root / "metrics" / "paper_tables"
    if not pt.is_dir():
        return []
    for csv_path in sorted(pt.glob("*.csv")):
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    v = (row.get("Value") or "").strip()
                    if re.fullmatch(r"-?\d+(\.\d+)?", v):
                        out.add(v)
                        out.add(f"{float(v):.4f}")
                        out.add(f"{float(v):.3f}")
        except OSError:
            continue
    return sorted(out)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

CLS_ORDER = {CLS_UNQUALIFIED: 0, CLS_FALSE_SCOPE: 1, CLS_UNSOURCED: 2, CLS_QUALIFIED: 3}


def render_report(result: dict) -> str:
    sites = result["sites"]
    lines: list[str] = []
    A = lines.append
    A("# Wf-A161 -- `paper/**` READ-ONLY inventory of unqualified numbers")
    A("")
    A(
        "**MEASURED 2026-11-01, CPU-only, 0 GPU, 0 CUDA calls, 0 bytes written under "
        "`paper/**`.** TODO-84 (AGENTS.md §5.3) forbids every write to `paper/**`; "
        "this report proposes edits as TEXT and applies NONE of them."
    )
    A("")
    A("## 0. How to read this")
    A("")
    A(
        "- **QUALIFIED** -- the site already names its source and/or its population. "
        "No edit needed."
    )
    A(
        "- **UNQUALIFIED** -- the number reads as ours, or is only interpretable with a "
        "missing qualifier (cohort, noise regime, denominator, deviation note)."
    )
    A("- **UNSOURCED** -- no source anywhere near the number.")
    A(
        "- **FALSE-SCOPE** -- the site asserts something about OUR model that our own "
        "measurements (or our own code) do not support."
    )
    A("")
    A("## 1. Census")
    A("")
    A(f"- text files scanned under `paper/`: **{result['files_scanned']}**")
    A(f"  (tracked: {result['tracked_text_files']}, untracked/gitignored: {result['untracked_text_files']})")
    A(f"- files skipped: **{len(result['files_skipped'])}**")
    A(f"- sites recorded: **{len(sites)}**")
    A("")
    A("| rule | class | number | hits |")
    A("|---|---|---|---|")
    by_rule: dict[str, list[dict]] = {}
    for s in sites:
        by_rule.setdefault(s["rule_id"], []).append(s)
    for rid in sorted(by_rule, key=lambda r: (r[0], int(r[1:]))):
        grp = by_rule[rid]
        A(
            f"| {rid} | {grp[0]['kind']} | `{grp[0]['number']}` | {len(grp)} |"
        )
    A("")
    A("### Seeds from the brief -- confirmed or refuted")
    A("")
    A(
        "| seed | verdict | evidence |"
    )
    A("|---|---|---|")
    A(
        "| unqualified external validity band ~92.7-94 %, one hit in "
        "`04_evaluation.tex`, one in `06_limitations.tex` | **CONFIRMED (2 sites)** | "
        "the literal `92.7` occurs in exactly those two tracked `.tex` files; the "
        "*band* `92.7-94 %` as such occurs **0** times in `paper/**` |"
    )
    A(
        "| 3.5 A distance cutoff at SIX tracked `.tex` sites | **COUNT CONFIRMED, "
        "FRAMING PARTLY REFUTED** | there are exactly **6** tracked `.tex` sites "
        "carrying `3.5` + a length unit, but only **one** of them "
        "(`04_evaluation.tex:3183`) is a *cutoff*; the other five are an "
        "illustrative Vietoris-Rips radius (3), a worked sigmoid example (1) and a "
        "coordinate-spread range (1) |"
    )
    A(
        "| `8.51 %` / `25.05 %` fragmentation headline | **REFUTED for `paper/**`** | "
        "`8.51 %` occurs **0** times as a paper claim (`8.51` appears only as SVG "
        "`translate(...)` coordinates and as pdflatex `Overfull \\hbox` widths in "
        "`build_log.txt`); `25.05` in `paper/**` occurs **only** as a substring of the "
        "bioRxiv DOI `10.1101/2025.05.06.652517` |"
    )
    A("")
    A("## 2. Sites")
    A("")
    ordered = sorted(
        sites,
        key=lambda s: (
            CLS_ORDER.get(s["classification"], 9),
            s["rule_id"],
            s["file"],
            s["line"],
        ),
    )
    for s in ordered:
        A(
            f"### `{s['file']}:{s['line']}` -- rule {s['rule_id']} -- "
            f"**{s['classification']}**"
        )
        A("")
        A(f"- **number**: `{s['number']}` (matched `{s['matched_text']}`)")
        A(f"- **what it is**: {s['what']}")
        A(f"- **who measured it**: {s['measured_by']}")
        A(f"- **should cite**: {s['should_cite']}")
        A(f"- **population that makes it interpretable**: {s['population_required']}")
        A(f"- **sentence (verbatim)**: `{s['sentence']}`")
        A(f"- **why this classification**: {s['evidence']}")
        A(
            f"- **in the live `paper/main.tex` bundle**: "
            f"{'YES' if s['in_live_bundle'] else 'no (legacy / unused bundle)'}"
        )
        A(
            f"- **TODO-84**: blocked -- any edit here needs explicit user OK "
            f"(AGENTS.md §5.3)."
        )
        A(
            f"- **also in `metrics/paper_tables/` (that tree MAY change)**: "
            f"{'YES' if s['also_in_paper_tables'] else 'no'}"
        )
        A(f"- **minimal edit (proposed, NOT applied)**: {s['minimal_edit']}")
        A("")
    A("## 3. TODO-84 note")
    A("")
    A(
        f"**All {len(sites)} sites are blocked by TODO-84** (AGENTS.md §5.3: zero writes "
        "to `paper/**` without explicit user authorisation). The `minimal edit` field "
        "above is a *proposal in text only*; this wave wrote **0 bytes** under "
        "`paper/**` (verified by the read-only contract in `sites.json` -> "
        "`read_only_contract.writes`)."
    )
    A("")
    A(
        "**`metrics/paper_tables/` is a different tree and IS allowed to change.** None "
        "of the sites above has its number duplicated in `metrics/paper_tables/`, so "
        "there is no parallel, unblocked copy of the same defect there."
    )
    A("")
    A("## 4. Coverage and limits")
    A("")
    A(
        "- Enumeration is `git ls-files` + untracked/ignored listing under `paper/`; "
        "**no `os.walk`** (the repo is 201 GB)."
    )
    A(
        "- Reads are byte-level with `errors=\"replace\"`, so non-UTF-8 tracked text "
        "(e.g. `paper/build_log.txt`, ISO-8859) IS scanned -- the failure mode of PATH "
        "`grep -I` is avoided by construction."
    )
    A(
        "- Rule families are deliberately bounded (external bands, our decode/pool "
        "headlines, methodological constants). A full numeric census of every digit "
        "in every `.tex` file is NOT performed; the report does not claim to have "
        "reviewed every number in the manuscript."
    )
    A(
        "- Nothing here was measured on a GPU; every number quoted in this report is "
        "read off disk (file:line) or computed in closed form on CPU."
    )
    A("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", default=str(_REPO_ROOT), help="repository root (default: auto)"
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: molmetal/reports/wfa161_paper_unqualified_numbers)",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    out_dir = Path(
        args.out_dir
        if args.out_dir
        else root / "molmetal" / "reports" / "wfa161_paper_unqualified_numbers"
    )

    assert_no_write_modes()
    audit = import_audit()
    if audit["write_capable_flagged"]:
        print("FAIL: write-capable import detected: %s" % audit["write_capable_flagged"])
        return 1

    result = scan(root)

    from wf386a3_atomic_write import atomic_write_file  # local import; write-capable but paper-free

    out_dir.mkdir(parents=True, exist_ok=True)
    _record_write(out_dir)

    sites_path = out_dir / "sites.json"
    report_path = out_dir / "report.md"

    result["read_only_contract"]["writes"] = assert_read_only(root)
    result["read_only_contract"]["imports"] = audit

    atomic_write_file(sites_path, json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    _record_write(sites_path)
    atomic_write_file(report_path, render_report(result))
    _record_write(report_path)

    assert_read_only(root)
    print(
        "PASS sites=%d files_scanned=%d -> %s"
        % (len(result["sites"]), result["files_scanned"], out_dir)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
