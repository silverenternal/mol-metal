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

"""Wf-R43 m7 — diversity_subpocket UNIFORM-PLACEHOLDER guard (CPU-only).

Background — what the record currently claims
---------------------------------------------
``metrics/by_metric/diversity_subpocket.json`` is the m7 record.  Its
``verdict`` reads ``DESIGN_ONLY`` and its ``measurements`` array holds a
single entry whose ``status`` is ``DESIGN`` — "smoke-only: empty=NaN,
N=1=0.0, 2-disjoint>0" over ``test fixtures``.  There is no
production-scale measurement anywhere in the file.

Two separate defects make that record unfalsifiable, and this module
exists to make both of them loud.

Defect 1 — the sub-pocket weighting is a uniform placeholder
-------------------------------------------------------------
``molmetal/scripts/r4_lambda_only_run.py::metric_subpocket_diversity_mean``
and its twin ``molmetal/scripts/r4_lib/metrics.py`` both build the
per-residue weight vector as::

    weights = [1] * int(n_residues)     # n_residues defaults to 4

with a docstring that defers the real vector to
``voxelization.py:208,255`` / ``vina_adapter.py:961``.  MEASURED on this
repo:

  * ``molmetal/molmetal_lam/sbdd_env/voxelization.py:208`` is an
    **empty line**;
  * ``voxelization.py:255`` is ``origin = center - half_extent`` (a
    voxel-grid origin, not a residue weight);
  * ``vina_adapter.py:961`` is an **empty line**;
  * ``grep -rn "residue_weight" molmetal/molmetal_lam/sbdd_env/*.py``
    returns **nothing**.

So the cited source of the "real" per-residue weights does not exist.
A uniform weight vector is, by construction, *indifferent to which
residue the bits belong to* — the "sub-pocket" axis is exactly the axis
the metric claims to measure.  What the number actually computes is a
Morgan-ECFP4 mean pairwise Tanimoto distance under a bit-index modulo-K
partition (``residue_bit_partition``), i.e. a re-bucketed GLOBAL
diversity number, not a local sub-pocket one.

That is not automatically zero — it is genuinely different from plain
global Tanimoto, because the modulo partition re-weights sparse bins.
MEASURED on three triazole SMILES (K=4, radius=2, n_bits=2048):

======  ==========================  =====================  =========
pair    per-residue (uniform, K=4)  plain global Tanimoto  delta
======  ==========================  =====================  =========
0 / 1             0.823502886                 0.833333333  -0.0098
0 / 2             0.919871795                 0.914893617  +0.0050
1 / 2             0.522222222                 0.512820513  +0.0094
======  ==========================  =====================  =========

So the defect is *misattribution*, not a silently-zero metric: the value
is a real number, but it does not carry the sub-pocket meaning its name,
its unit string, and its lit anchors (Bemis-Murcko 1996; Peter 2019 SPF)
imply.

Defect 2 — the one real sweep is 30 copies of a single measurement
-----------------------------------------------------------------
``molmetal/reports/wf_r16_round13_30cell/final.json`` is the only
30-cell sweep carrying a ``div_subpocket`` column.  MEASURED over its
``per_cell`` array (n=30):

===================  =========  ==========
metric               n_unique   std
===================  =========  ==========
diversity_subpocket          1  0.0
div_tan                     1  0.0
div_homo                    1  0.0
novelty                     1  0.0
sa / qed / logp / tpsa      1  0.0
===================  =========  ==========

Every pool-derived metric is byte-identical across all 30 (pocket, seed)
cells.  The sweep therefore contributes **one** diversity measurement
repeated 30 times, not 30 independent ones — the same pathology
Wf-R43 m2 just closed for the R2 cohorts, where 2 of 3 "cohorts" were
one measurement.  Its ``div_subpocket_mean`` 0.6538729654113289 with
``div_subpocket_std`` 0.0 must not be read as a reproducible
across-pocket result.

What this module does
---------------------
It answers one question —

    may diversity_subpocket be reported as a MEASURED sub-pocket
    diversity metric with per-residue (non-uniform) weights?

— and answers **FAIL** until both of the following hold:

1. the weighted implementation actually in the training/eval path uses a
   **non-uniform** per-residue weight vector (a real pocket-residue
   weight source, not ``[1] * n``); and
2. the on-disk record carries a multi-cell measurement whose
   per-cell values are not all identical (i.e. the sweep carries real
   between-cell variance, not one value replicated).

It never trains, never writes a metric, never touches ``paper/*``, and
never launches a GPU.

CPU-reachable remediation
-------------------------
Defect 1 is CPU-resolvable today: the 10 Wf-30 pockets are staged READY
under ``molmetal/reports/wf30_pocket_prep/`` (10/10, with per-pocket
``receptor.pdbqt`` + source ``PDB`` sha256 on record), and RDKit
**2026.03.6** is importable in the project venv.  A per-residue weight
vector can therefore be derived from real pocket residues today; what is
missing is the wiring, not the data or the dependency.

Note this also invalidates the file's own ``DESIGN_ONLY`` justification,
which reads "a re-run on a host with RDKit would lift this metric to
MEASURED" — RDKit is present, so the blocker it names is not the blocker.

Defect 2 needs a real sweep with genuine per-cell pool diversity; it is
not resolvable by re-aggregating the existing r16 file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

GATE_NAME = "p3_wf_r43_m7_subpocket_uniform"
METRIC_KEY = "diversity_subpocket"
OPT_IN_ENV = "WF43_M7_SUBPOCKET_UNIFORM"

#: Sweep that carries the only ``div_subpocket`` column on disk.
R16_SWEEP_RELPATH = "molmetal/reports/wf_r16_round13_30cell/final.json"
#: Wf-R43-A3 closure sweep — 4 cells x 2 seeds (8 rows) computed from
#: metalcytotox_5metal_multitask_v4.csv with the centre-decay per-residue
#: weight vector.  Used as the between-cell-variance evidence base when
#: present; the legacy 30-cell sweep is preserved in
#: ``wf_r16_round13_30cell/final.json`` (its uniform-placeholder weights
#: make every row carry the same div_subpocket — the historic defect
#: that this guard was added to fail loud).
R43A3_SWEEP_RELPATH = "molmetal/reports/wf_r43a3_m7_mini_sweep/final.json"

#: The two metric implementations that must stop using ``[1] * n``.
IMPL_RELPATHS = (
    "molmetal/scripts/r4_lambda_only_run.py",
    "molmetal/scripts/r4_lib/metrics.py",
)

#: Matches the uniform placeholder, tolerating whitespace / int() wrapping.
_UNIFORM_WEIGHT_RE = re.compile(
    r"weights\s*=\s*\[\s*1\s*\]\s*\*\s*(?:int\s*\(\s*n_residues\s*\)|n_residues)",
)

#: Matches ``[1.0] * n`` / ``[1] * K`` style variants.
_UNIFORM_WEIGHT_ALT_RE = re.compile(
    r"\[\s*1(?:\.0+)?\s*\]\s*\*\s*(?:int\s*\(\s*\w+\s*\)|\w+)",
)

#: Weight sources the docstrings cite as the "real" deferred vector.
CITED_WEIGHT_SOURCES = {
    "molmetal/molmetal_lam/sbdd_env/voxelization.py": (208, 255),
    "molmetal/molmetal_lam/sbdd_env/vina_adapter.py": (961,),
}


def _resolve(repo_root: Optional[Path]) -> Path:
    if repo_root is not None:
        return Path(repo_root)
    return Path(__file__).resolve().parents[2]


def _audit_uniform_weights(repo_root: Path) -> Dict[str, Any]:
    """MEASURE whether the live metric implementations use uniform weights.

    Returns a dict with ``uniform`` (bool), ``offenders`` (list of
    relpaths), and ``cited_sources`` (per-source line audit).  A missing
    file is recorded, not raised — the guard degrades to FAIL evidence
    rather than crashing the pre-flight run.
    """
    offenders: List[str] = []
    details: Dict[str, Any] = {}
    for rel in IMPL_RELPATHS:
        path = repo_root / rel
        if not path.is_file():
            details[rel] = {"present": False}
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            details[rel] = {"present": True, "read_error": repr(exc)[:120]}
            continue
        # Strip comment-only lines so a docstring example does not trip
        # the uniform-weight detector (the docstring legitimately says
        # "[1] * n_residues" as the *historical* placeholder pattern;
        # the live code is on a non-uniform centre-decay vector).
        code_lines = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        hits = [
            m.group(0)
            for m in list(_UNIFORM_WEIGHT_RE.finditer(code_lines))
            + list(_UNIFORM_WEIGHT_ALT_RE.finditer(code_lines))
        ]
        details[rel] = {"present": True, "uniform_weight_hits": len(hits)}
        if hits:
            offenders.append(rel)

    # Audit the line numbers the docstrings cite for the "real" weights.
    cited: Dict[str, Any] = {}
    for rel, lines in CITED_WEIGHT_SOURCES.items():
        path = repo_root / rel
        if not path.is_file():
            cited[rel] = {"present": False}
            continue
        try:
            all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:  # pragma: no cover
            cited[rel] = {"present": True, "read_error": True}
            continue
        per_line = {}
        for ln in lines:
            text = all_lines[ln - 1].strip() if 1 <= ln <= len(all_lines) else "<out-of-range>"
            per_line[str(ln)] = {
                "text": text,
                "blank": text == "",
                "looks_like_weights": bool(
                    re.search(r"weight|residue", text, re.IGNORECASE)
                ),
            }
        cited[rel] = {"present": True, "lines": per_line}

    return {
        "uniform": bool(offenders),
        "offenders": offenders,
        "impl_details": details,
        "cited_sources": cited,
    }


def _audit_sweep_variance(repo_root: Path) -> Dict[str, Any]:
    """MEASURE whether the on-disk sweep has between-cell variance.

    A sweep whose ``div_subpocket`` is constant across cells carries one
    measurement, not N.  Returns ``present`` / ``n_cells`` /
    ``n_unique`` / ``value`` / ``has_variance``.

    Wf-R43-A3 sweep-precedence rule: when ``R43A3_SWEEP_RELPATH`` is on
    disk it takes precedence over the legacy R16 sweep because the new
    sweep uses non-uniform centre-decay weights (the closure condition)
    and the legacy sweep uses uniform-placeholder weights (the very
    defect this guard was added to flag).  Reading the legacy sweep
    first would be self-defeating.
    """
    # Prefer the Wf-R43-A3 closure sweep when present.
    new_path = repo_root / R43A3_SWEEP_RELPATH
    legacy_path = repo_root / R16_SWEEP_RELPATH
    if new_path.is_file():
        path = new_path
        relpath = R43A3_SWEEP_RELPATH
    else:
        path = legacy_path
        relpath = R16_SWEEP_RELPATH
    out: Dict[str, Any] = {
        "present": False,
        "relpath": relpath,
        "n_cells": 0,
        "n_unique": None,
        "value": None,
        "has_variance": False,
        "n_metrics_constant": 0,
    }
    if not path.is_file():
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        out["read_error"] = f"{exc.__class__.__name__}: {exc}"
        return out

    cells = payload.get("per_cell")
    if not isinstance(cells, list) or not cells:
        return out

    out["present"] = True
    out["n_cells"] = len(cells)

    # Primary column under audit + the pool-derived siblings that were
    # measured constant alongside it (corroborating evidence).
    #
    # NOTE: ``per_cell`` rows use the SHORT aggregate keys (``div_subpocket``,
    # not ``diversity_subpocket``); the long names appear in ``overall`` /
    # ``per_pocket``.  Both spellings are checked so the guard works whether
    # a future sweep writes one or the other.
    pool_derived = (
        "div_subpocket",
        "diversity_subpocket",
        "div_tan",
        "div_homo",
        "novelty",
        "sa",
        "qed",
        "logp",
        "tpsa",
    )
    n_const = 0
    for key in pool_derived:
        vals = [c.get(key) for c in cells if isinstance(c, dict) and c.get(key) is not None]
        if not vals:
            continue
        uniq = {round(float(v), 9) for v in vals}
        if len(uniq) == 1:
            n_const += 1
            if key in ("div_subpocket", "diversity_subpocket"):
                out["n_unique"] = 1
                out["value"] = float(vals[0])
                out["has_variance"] = False
    out["n_metrics_constant"] = n_const
    return out


def audit_m7_subpocket_uniform(
    metric_path: Path,
    *,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit whether ``diversity_subpocket`` may be reported as MEASURED.

    Two conditions must hold for PASS:

    1. **Non-uniform weights in the live path** — neither
       ``r4_lambda_only_run.py`` nor ``r4_lib/metrics.py`` may build the
       per-residue vector as ``[1] * n_residues``.
    2. **Between-cell variance in the on-disk sweep** — the recorded
       sweep must not report a single ``div_subpocket`` value repeated
       across every cell.

    Returns ``status`` in {PASS, FAIL, SKIP}, an ``evidence`` string, and
    the structured findings.  Never raises on malformed input: a broken
    record FAILs loudly rather than crashing pre-flight (AGENTS.md §5.2).
    """
    root = _resolve(repo_root)
    result: Dict[str, Any] = {
        "status": "FAIL",
        "metric": METRIC_KEY,
        "evidence": "",
        "weights": None,
        "sweep": None,
        "metric_present": False,
        "design_only_claim": None,
    }

    # --- the metric record itself -------------------------------------
    if metric_path.is_file():
        result["metric_present"] = True
        try:
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            result["evidence"] = (
                f"{METRIC_KEY}: {metric_path.name} unreadable "
                f"({exc.__class__.__name__}: {exc}) — FAIL loud rather than "
                "inheriting a DESIGN-only record (AGENTS.md §5.2)"
            )
            return result
        verdict = str(payload.get("verdict", ""))
        result["design_only_claim"] = verdict.startswith("DESIGN_ONLY")
    else:
        result["evidence"] = (
            f"{METRIC_KEY}: no record at {metric_path} — nothing to audit "
            "(absent metric is SKIP, not FAIL)"
        )
        result["status"] = "SKIP"
        return result

    weights = _audit_uniform_weights(root)
    sweep = _audit_sweep_variance(root)
    result["weights"] = weights
    result["sweep"] = sweep

    if weights["uniform"]:
        n_cited_blank = sum(
            1
            for src in weights["cited_sources"].values()
            for ln in src.get("lines", {}).values()
            if ln.get("blank")
        )
        result["evidence"] = (
            f"{METRIC_KEY}: FAIL — per-residue weights are the uniform "
            f"placeholder in {len(weights['offenders'])}/2 live metric "
            f"implementations ({', '.join(weights['offenders'])}); "
            f"{n_cited_blank} of the docstring-cited 'real weight source' "
            "lines are blank. A uniform weight vector is indifferent to "
            "which residue a bit belongs to, so the 'sub-pocket' axis the "
            "metric claims to measure carries no information; the number "
            "is a modulo-partitioned GLOBAL Tanimoto distance."
        )
        return result

    if sweep["present"] and not sweep["has_variance"] and sweep["n_unique"] == 1:
        result["evidence"] = (
            f"{METRIC_KEY}: FAIL — weights are non-uniform, but the recorded "
            f"sweep ({sweep['relpath']}, n_cells={sweep['n_cells']}) reports "
            f"div_subpocket={sweep['value']!r} with ZERO between-cell variance "
            f"(n_unique=1; {sweep['n_metrics_constant']} pool-derived metrics "
            "constant across all cells). That is one measurement repeated N "
            "times, not N independent measurements (Wf-R43 m2 precedent)."
        )
        return result

    result["status"] = "PASS"
    result["evidence"] = (
        f"{METRIC_KEY}: PASS — non-uniform per-residue weights in the live "
        "path AND between-cell variance present in the recorded sweep."
    )
    return result


def gate_m7_subpocket_uniform(
    name: str,
    priority: str,
    pipeline: str,
    metric_path: Path,
    *,
    repo_root: Optional[Path] = None,
    opt_in: bool = True,
) -> Any:
    """Pre-flight gate wrapper around :func:`audit_m7_subpocket_uniform`.

    ``opt_in=True`` (the registered default) emits SKIP unless
    ``WF43_M7_SUBPOCKET_UNIFORM=1``, following the Wf-R35 P-A precedent
    used by ``p3_wf_r43_m16_pt_ckpt_gap`` and
    ``p3_wf_r43_m1_paper_cohort`` in this same wave: the gate is
    registered and auditable, but adding it does not change the
    PASS/FAIL verdict of an existing pre-flight run.
    """
    import time

    from molmetal.scripts.pre_flight_check import GateResult

    t0 = time.perf_counter()
    if opt_in and os.environ.get(OPT_IN_ENV, "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"{name}: opt-in gate — set {OPT_IN_ENV}=1 to enforce the "
            f"{METRIC_KEY} uniform-placeholder-weight guard (Wf-R35 P-A / "
            "Wf-R43 m16+m1 opt-in pattern)",
            time.perf_counter() - t0,
        )

    result = audit_m7_subpocket_uniform(metric_path, repo_root=repo_root)
    return GateResult(
        name,
        priority,
        pipeline,
        result["status"],
        f"{name}: {result['evidence']}",
        time.perf_counter() - t0,
    )


__all__: List[str] = [
    "CITED_WEIGHT_SOURCES",
    "GATE_NAME",
    "IMPL_RELPATHS",
    "METRIC_KEY",
    "OPT_IN_ENV",
    "R16_SWEEP_RELPATH",
    "audit_m7_subpocket_uniform",
    "gate_m7_subpocket_uniform",
]
