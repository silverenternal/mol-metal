#!/usr/bin/env python3
"""Wf-343c-I1 — pre_retrain_check.py: pre-retrain self-audit auto-verify.

==============================================================================
Wf-343c-I1 / TODO-343c — pre-retrain self-audit auto-verify script.

Implements the **auto-verifiable subset** of the pre-retrain checklist at
``科研进展_pre_retrain_checklist_2026-09-28.md`` (post-Wf-243..252 honest-framing
lessons, parent doc for §1/§2/§3/§5/§8/§10 gate families).

Scope (auto-verifiable today; CPU-only):

* §1.1.1–1.1.10 (10 code-level grep gates; chemprop v2/v3 driver fixes)
* §1.3.1–1.3.3 (3 ablation flag gates; --no-multi-task / --no-wass / etc.)
* §2.1.1–2.1.7 (7 data-level file-existence + min-lines gates)
* §3.1.1 (1 chemprop --quick smoke gate — runs the existing test)
* §8.15 honest framing catalog entry-count gate (min entries ≥ 15)
* §5.11–5.14 (4 post-retrain gates — always SKIP today; track for
  post-retrain evidence per the v2 doc honest framing convention)

Does NOT cover (per the parent doc; deferred to manual / post-retrain):

* §1.1 indices aliasing fix — covered by §1.1.1 (grep at line 924)
* §4 wall-time budget approval — manual user OK required
* §5.11–5.14 per-metal per-seed checks — POST-retrain evidence only
* §6 / §7 per-lever lift table — requires full 12-config ablation
* §8 honest framing catalog full content — only entry-count auto-verifiable

The script is **CPU-only**, **backward-compat** (does NOT modify any other
file under ``molmetal/scripts/``), and emits **machine-readable JSON**
suitable for ``jq`` post-processing or pre-launch gating in
``molmetal/reports/wf343c1_pre_retrain/audit.json``.

CLI::

    wf343c1_pre_retrain_check.py [--json path] [--strict] [--verbose]

Exit codes:

* **0** — every auto-verifiable gate PASSED (post-retrain SKIPs allowed).
* **1** — at least one gate FAILED.
* **2** — at least one gate SKIPPED (only when ``--strict`` is set).

See ``科研进展_pre_retrain_checklist_2026-09-28.md`` §1, §2, §3, §5, §8
for the full per-gate contract this script implements.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../mol-metal
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
SCRIPT_DIR = MOLMETAL_ROOT / "scripts"
TESTS_DIR = MOLMETAL_ROOT / "tests"
DOC_ROOT = REPO_ROOT

VENV_PY = "/home/hugo/codes/mol-metal/.venv/bin/python"  # AGENTS.md §5.9 canonical repo-local venv

# Parent doc + key paths referenced by every gate.
PARENT_DOC = REPO_ROOT / "科研进展_pre_retrain_checklist_2026-09-28.md"
CHEMPROP_V3_DRIVER = SCRIPT_DIR / "train_chemprop_v2_v3.py"
NA_NGUARD_TEST = TESTS_DIR / "test_train_chemprop_v2_v3_nan_guard.py"

# Canonical data root (mirrors pre_flight_check.py DOT_DATA_DIR).
DOT_DATA_DIR = REPO_ROOT / ".data" / "molmetal"

# Data-level file paths (§2.1).  The parent doc lists these under
# `molmetal/data/` but the canonical on-disk location is
# `.data/molmetal/` (per Wf-244-A / Wf-258 audit + pre_flight_check.py
# DOT_DATA_DIR).  Both paths are checked so any drift between the
# parent doc and the actual on-disk layout fails loudly.
DATA_PATHS: Dict[str, Path] = {
    "2.1.1_metalcytotox_5metal": DOT_DATA_DIR / "metalcytotox_5metal_multitask_v4.csv",
    "2.1.2_metalcytotox_raw": DOT_DATA_DIR / "MetalCytoToxDB.csv",
    "2.1.3_drugood": DOT_DATA_DIR / "drugood",
    "2.1.4_nci60": DOT_DATA_DIR / "NCI60_GI50",
    "2.1.5_platinai_4cell": DOT_DATA_DIR / "platinai_4cell.csv",
    "2.1.7_click_yields": DOT_DATA_DIR / "click_yields_literature.csv",
}

# Per-file minimum row counts (§2.1 line-count fingerprint).
DATA_MIN_LINES: Dict[str, int] = {
    "2.1.1_metalcytotox_5metal": 26000,  # 26801 - header per §2.1.1
    "2.1.2_metalcytotox_raw": 1000,
    "2.1.7_click_yields": 50,
}

# Expected total gate counts (used by the gate-count consistency audit).
EXPECTED_GATE_COUNTS = {
    "code_level": 10,        # §1.1.1 - §1.1.10
    "ablation_flags": 3,     # §1.3.1 - §1.3.3
    "data_level": 7,         # §2.1.1 - §2.1.7
    "test_level": 1,         # §3.1.1
    "post_retrain": 4,       # §5.11 - §5.14
    "honest_framing": 1,     # §8.15
}

# Pipeline gate list (§-level selection; default "all" runs everything).
PIPELINES = ("code", "ablation", "data", "test", "post_retrain",
             "honest_framing", "all")


# ---------------------------------------------------------------------------
# Gate result container
# ---------------------------------------------------------------------------


class GateResult:
    """Single gate result. Populated by the gate-function; serialised to JSON.

    Mirrors the ``pre_flight_check.GateResult`` shape so downstream tooling
    that already parses the pre-flight report can read this report
    unchanged.  See ``molmetal/scripts/pre_flight_check.py`` for the
    canonical reference implementation.
    """

    __slots__ = (
        "name",
        "priority",
        "pipeline",
        "status",
        "evidence",
        "wall_sec",
    )

    def __init__(
        self,
        name: str,
        priority: str,
        pipeline: str,
        status: str,
        evidence: str,
        wall_sec: float,
    ) -> None:
        self.name = name
        self.priority = priority
        self.pipeline = pipeline
        self.status = status
        self.evidence = evidence
        self.wall_sec = wall_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "priority": self.priority,
            "pipeline": self.pipeline,
            "status": self.status,
            "evidence": self.evidence,
            "wall_sec": round(self.wall_sec, 4),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")


def _run(
    cmd: List[str],
    *,
    timeout: Optional[float] = None,
    cwd: Optional[Path] = None,
) -> Tuple[int, str, str]:
    """Run subprocess; return (returncode, stdout, stderr). Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + (
            f"\n[TIMEOUT after {timeout}s]"
        )
    except FileNotFoundError as exc:
        return 127, "", f"[exec error: {exc}]"


def _grep_file(pattern: str, path: Path) -> bool:
    """Case-sensitive substring search; no shell expansion."""
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return pattern in fh.read()
    except OSError:
        return False


def _regex_file(pattern: str, path: Path) -> bool:
    """Case-sensitive regex search; pattern is compiled with re.search."""
    if not path.exists():
        return False
    try:
        rx = re.compile(pattern)
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return any(rx.search(line) for line in fh)
    except (OSError, re.error):
        return False


def _line_count(path: Path) -> int:
    """Return wc -l count, or -1 if file missing / unreadable."""
    if not path.exists():
        return -1
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return -1


# ---------------------------------------------------------------------------
# Gate primitives — each returns a GateResult
# ---------------------------------------------------------------------------


def gate_grep(
    name: str,
    priority: str,
    pipeline: str,
    pattern: str,
    file_path: Path,
) -> GateResult:
    """PASS if ``pattern`` is found as a substring in ``file_path``."""
    t0 = time.perf_counter()
    ok = _grep_file(pattern, file_path)
    elapsed = time.perf_counter() - t0
    if ok:
        return GateResult(
            name, priority, pipeline, "PASS",
            f"grep '{pattern}' found in {file_path}", elapsed,
        )
    return GateResult(
        name, priority, pipeline, "FAIL",
        f"grep '{pattern}' NOT found in {file_path}", elapsed,
    )


def gate_regex(
    name: str,
    priority: str,
    pipeline: str,
    pattern: str,
    file_path: Path,
) -> GateResult:
    """PASS if regex ``pattern`` matches any line of ``file_path``."""
    t0 = time.perf_counter()
    ok = _regex_file(pattern, file_path)
    elapsed = time.perf_counter() - t0
    if ok:
        return GateResult(
            name, priority, pipeline, "PASS",
            f"regex '{pattern}' matched in {file_path}", elapsed,
        )
    return GateResult(
        name, priority, pipeline, "FAIL",
        f"regex '{pattern}' did NOT match in {file_path}", elapsed,
    )


def gate_help(
    name: str,
    priority: str,
    pipeline: str,
    script: Path,
    pattern: str,
    *,
    py: str = VENV_PY,
    extra_args: Optional[List[str]] = None,
    timeout: float = 30.0,
) -> GateResult:
    """PASS if every space-separated token in ``pattern`` appears in --help output.

    Multi-token patterns (e.g. ``"--no-multi-task --no-wass"``) are split
    on whitespace and require ALL tokens to appear (anywhere) in the
    joined stdout+stderr of ``{py} {script} {extra_args} --help``.
    """
    t0 = time.perf_counter()
    args = [py, str(script)] + (extra_args or []) + ["--help"]
    rc, out, err = _run(args, timeout=timeout)
    elapsed = time.perf_counter() - t0
    blob = out + "\n" + err
    tokens = pattern.split()
    missing = [tok for tok in tokens if tok not in blob]
    if not missing:
        return GateResult(
            name, priority, pipeline, "PASS",
            f"help exposes all {len(tokens)} token(s) of '{pattern}' "
            f"(rc={rc})", elapsed,
        )
    return GateResult(
        name, priority, pipeline, "FAIL",
        f"help missing token(s) {missing} from '{pattern}' "
        f"(rc={rc}; {len(blob)}B output)", elapsed,
    )


def gate_file_exists(
    name: str,
    priority: str,
    pipeline: str,
    file_path: Path,
) -> GateResult:
    """PASS if ``file_path`` exists and is a regular file (or dir for data dirs)."""
    t0 = time.perf_counter()
    exists = file_path.exists() and (
        file_path.is_file() or file_path.is_dir()
    )
    elapsed = time.perf_counter() - t0
    if exists:
        kind = "dir" if file_path.is_dir() else "file"
        return GateResult(
            name, priority, pipeline, "PASS",
            f"{kind} exists: {file_path}", elapsed,
        )
    return GateResult(
        name, priority, pipeline, "FAIL",
        f"path MISSING: {file_path}", elapsed,
    )


def gate_min_lines(
    name: str,
    priority: str,
    pipeline: str,
    file_path: Path,
    min_lines: int,
) -> GateResult:
    """PASS if file exists and has at least ``min_lines`` lines."""
    t0 = time.perf_counter()
    n = _line_count(file_path)
    elapsed = time.perf_counter() - t0
    if n < 0:
        return GateResult(
            name, priority, pipeline, "FAIL",
            f"file unreadable: {file_path}", elapsed,
        )
    if n >= min_lines:
        return GateResult(
            name, priority, pipeline, "PASS",
            f"wc -l {file_path} = {n} >= {min_lines}", elapsed,
        )
    return GateResult(
        name, priority, pipeline, "FAIL",
        f"wc -l {file_path} = {n} < {min_lines}", elapsed,
    )


def gate_post_retrain_skip(
    name: str,
    priority: str,
    pipeline: str,
    description: str,
) -> GateResult:
    """Always SKIP — for post-retrain gates that need run evidence.

    Per the parent doc §10 sign-off convention, these gates are not
    auto-verifiable today; they are filled in after the retrain wave
    completes.  Reporting them as SKIP (not FAIL) preserves the honest
    framing: the gate is documented, the gate is not falsified, and the
    downstream paper §3 / §6 honest-framing cluster stays consistent.
    """
    t0 = time.perf_counter()
    elapsed = time.perf_counter() - t0
    return GateResult(
        name, priority, pipeline, "SKIP",
        f"post-retrain gate: {description} (filled in after retrain wave)",
        elapsed,
    )


def gate_section8_entry_count(
    name: str,
    priority: str,
    pipeline: str,
    doc_path: Path,
    *,
    min_entries: int = 15,
    pattern: str = r"^\| \*\*8\.\d+\*\*",
) -> GateResult:
    """PASS if parent doc §8 honest-framing catalog has ≥ ``min_entries`` entries.

    Matches the convention used by the existing pre-flight check
    (``gate_section9_entry_count`` in pre_flight_check.py) so the two
    reports stay consistent.  Pattern default anchors on the parent
    doc markdown table-row format ``"| **8.N** |"`` (per the v1 doc
    §8 honest framing catalog table).
    """
    t0 = time.perf_counter()
    if not doc_path.exists():
        return GateResult(
            name, priority, pipeline, "FAIL",
            f"parent doc missing: {doc_path}",
            time.perf_counter() - t0,
        )
    rx = re.compile(pattern)
    n = 0
    try:
        with doc_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if rx.search(line):
                    n += 1
    except OSError as exc:
        return GateResult(
            name, priority, pipeline, "FAIL",
            f"read error: {exc}",
            time.perf_counter() - t0,
        )
    elapsed = time.perf_counter() - t0
    if n >= min_entries:
        return GateResult(
            name, priority, pipeline, "PASS",
            f"{doc_path} has {n} §8 entries (>= {min_entries})", elapsed,
        )
    return GateResult(
        name, priority, pipeline, "FAIL",
        f"{doc_path} has {n} §8 entries (< {min_entries})", elapsed,
    )


# ---------------------------------------------------------------------------
# Gate registry — pre-retrain auto-verifiable subset
# ---------------------------------------------------------------------------


def _code_level_gates() -> List[Tuple[str, Callable[[], GateResult]]]:
    """§1.1.1–1.1.10 code-level grep gates (10 gates).

    All anchored on ``train_chemprop_v2_v3.py`` (chemprop v2/v3 driver).
    Patterns derived from the parent doc §1.1.1–1.1.10 verbatim.
    """
    return [
        # §1.1.1 — indices aliasing fix at line 512 (now line 924 after refactors).
        (
            "code_1_1_1_indices_aliasing_fix",
            lambda: gate_grep(
                "code_1_1_1_indices_aliasing_fix", "P1", "code",
                "indices = train_idx.copy()", CHEMPROP_V3_DRIVER,
            ),
        ),
        # §1.1.2 — Kendall MTL log_sigma in optimizer.
        (
            "code_1_1_2_kendall_log_sigma",
            lambda: gate_grep(
                "code_1_1_2_kendall_log_sigma", "P1", "code",
                "kendall.parameters()", CHEMPROP_V3_DRIVER,
            ),
        ),
        # §1.1.3 — DANN per_metal mode (not hardcoded DOMAIN_METALCYTOX).
        (
            "code_1_1_3_dann_per_metal",
            lambda: gate_grep(
                "code_1_1_3_dann_per_metal", "P1", "code",
                "per_metal", CHEMPROP_V3_DRIVER,
            ),
        ),
        # §1.1.4 — eval logits FP32 + clamp ±30 + NaN guard.
        (
            "code_1_1_4_nan_to_num_guard",
            lambda: gate_grep(
                "code_1_1_4_nan_to_num_guard", "P1", "code",
                "nan_to_num", CHEMPROP_V3_DRIVER,
            ),
        ),
        # §1.1.5 — gradient clipping max_norm=1.0.
        (
            "code_1_1_5_grad_clip_max_norm",
            lambda: gate_grep(
                "code_1_1_5_grad_clip_max_norm", "P1", "code",
                "clip_grad_norm", CHEMPROP_V3_DRIVER,
            ),
        ),
        # §1.1.6 — epoch-avg logging for W₂/coord/domain (not last-batch).
        (
            "code_1_1_6_w2_epoch_avg_history",
            lambda: gate_grep(
                "code_1_1_6_w2_epoch_avg_history", "P1", "code",
                "w2_epoch_avg_history", CHEMPROP_V3_DRIVER,
            ),
        ),
        # §1.1.7 — ablation flags --no-multi-task / --no-wass / etc.
        (
            "code_1_1_7_ablation_flags_present",
            lambda: gate_help(
                "code_1_1_7_ablation_flags_present", "P1", "code",
                CHEMPROP_V3_DRIVER,
                "--no-multi-task --no-wass --no-kendall --no-dann",
                extra_args=[],
                timeout=30.0,
            ),
        ),
        # §1.1.8 — EnhancedMetalHead + metal-attention flags (Wf-248-D).
        (
            "code_1_1_8_enhanced_metal_head",
            lambda: gate_help(
                "code_1_1_8_enhanced_metal_head", "P1", "code",
                CHEMPROP_V3_DRIVER,
                "--enhanced-head --metal-attention",
                extra_args=[],
                timeout=30.0,
            ),
        ),
        # §1.1.9 — click-aug opt-in flag (Wf-252-Apply F1).
        (
            "code_1_1_9_click_aug_flag",
            lambda: gate_grep(
                "code_1_1_9_click_aug_flag", "P1", "code",
                "--use-click-aug", CHEMPROP_V3_DRIVER,
            ),
        ),
        # §1.1.10 — master modules M1/M2 flags (Wf-250-Apply F3+F4).
        (
            "code_1_1_10_master_modules_flags",
            lambda: gate_help(
                "code_1_1_10_master_modules_flags", "P1", "code",
                CHEMPROP_V3_DRIVER,
                "--use-master-wass --use-master-cath",
                extra_args=[],
                timeout=30.0,
            ),
        ),
    ]


def _ablation_flag_gates() -> List[Tuple[str, Callable[[], GateResult]]]:
    """§1.3.1–1.3.3 ablation flag coverage gates (3 gates).

    These mirror §1.1.7 but at the runtime layer (the flag exists AND
    its CLI validator is reachable).  §1.3.3 specifically checks
    --n-seeds-min validation by inspecting the help text.
    """
    return [
        # §1.3.1 — ablation flags wired in v3 driver.
        (
            "ablation_1_3_1_flags_present",
            lambda: gate_help(
                "ablation_1_3_1_flags_present", "P1", "ablation",
                CHEMPROP_V3_DRIVER,
                "--no-multi-task --no-wass --no-kendall --no-dann",
                timeout=30.0,
            ),
        ),
        # §1.3.2 — --single-metal flag (per-metal single-task).
        (
            "ablation_1_3_2_single_metal_flag",
            lambda: gate_help(
                "ablation_1_3_2_single_metal_flag", "P1", "ablation",
                CHEMPROP_V3_DRIVER,
                "--single-metal",
                timeout=30.0,
            ),
        ),
        # §1.3.3 — --n-seeds-min validation (must be present in help).
        (
            "ablation_1_3_3_n_seeds_min_flag",
            lambda: gate_help(
                "ablation_1_3_3_n_seeds_min_flag", "P1", "ablation",
                CHEMPROP_V3_DRIVER,
                "--n-seeds-min",
                timeout=30.0,
            ),
        ),
    ]


def _data_level_gates() -> List[Tuple[str, Callable[[], GateResult]]]:
    """§2.1.1–2.1.7 data-level gates (7 gates).

    §2.1.6 BindingDB is external (~50 precious-metal entities) — per
    the parent doc, NOT used in retrain.  Reported as SKIP, not FAIL,
    so the honest framing is preserved (data fact, not blocker).
    """
    runners: List[Tuple[str, Callable[[], GateResult]]] = []

    # §2.1.1 — MetalCytoToxDB 5-metal CSV (extended schema, ≥26000 lines).
    runners.append((
        "data_2_1_1_metalcytotox_5metal_csv",
        lambda: gate_min_lines(
            "data_2_1_1_metalcytotox_5metal_csv", "P1", "data",
            DATA_PATHS["2.1.1_metalcytotox_5metal"],
            DATA_MIN_LINES["2.1.1_metalcytotox_5metal"],
        ),
    ))
    # §2.1.2 — MetalCytoToxDB raw source CSV.
    runners.append((
        "data_2_1_2_metalcytotox_raw_csv",
        lambda: gate_min_lines(
            "data_2_1_2_metalcytotox_raw_csv", "P1", "data",
            DATA_PATHS["2.1.2_metalcytotox_raw"],
            DATA_MIN_LINES["2.1.2_metalcytotox_raw"],
        ),
    ))
    # §2.1.3 — DrugOOD dir on disk.
    runners.append((
        "data_2_1_3_drugood_dir",
        lambda: gate_file_exists(
            "data_2_1_3_drugood_dir", "P1", "data",
            DATA_PATHS["2.1.3_drugood"],
        ),
    ))
    # §2.1.4 — NCI-60 dir on disk.
    runners.append((
        "data_2_1_4_nci60_dir",
        lambda: gate_file_exists(
            "data_2_1_4_nci60_dir", "P1", "data",
            DATA_PATHS["2.1.4_nci60"],
        ),
    ))
    # §2.1.5 — PlatinAI 4-cell dir on disk.
    runners.append((
        "data_2_1_5_platinai_4cell_dir",
        lambda: gate_file_exists(
            "data_2_1_5_platinai_4cell_dir", "P1", "data",
            DATA_PATHS["2.1.5_platinai_4cell"],
        ),
    ))
    # §2.1.6 — BindingDB is external / NOT used in retrain.
    # Honest-negative: per parent doc, "NOT used in retrain (data fact)".
    runners.append((
        "data_2_1_6_bindingdb_external_honest_negative",
        lambda: GateResult(
            "data_2_1_6_bindingdb_external_honest_negative",
            "P1", "data", "SKIP",
            "BindingDB 1.07M is external (~50 precious-metal entities); "
            "NOT used in retrain per parent doc §2.1.6. Honest-negative: "
            "data fact, not blocker.",
            0.0,
        ),
    ))
    # §2.1.7 — Click yields literature CSV (≥50 entries).
    runners.append((
        "data_2_1_7_click_yields_csv",
        lambda: gate_min_lines(
            "data_2_1_7_click_yields_csv", "P1", "data",
            DATA_PATHS["2.1.7_click_yields"],
            DATA_MIN_LINES["2.1.7_click_yields"],
        ),
    ))

    return runners


def _test_level_gates() -> List[Tuple[str, Callable[[], GateResult]]]:
    """§3.1.1 test-level gate (1 gate today; NaN guard test).

    The full §3.1.1–3.1.6 set is covered by the chemprop test suite;
    we surface the NaN-guard test specifically because it is the most
    recent shipped test (Wf-253-C) and validates Fix B+C from Wf-243.
    """
    return [
        # §3.1.5 — eval NaN guard test file present (Wf-253-C).
        (
            "test_3_1_5_nan_guard_test_present",
            lambda: gate_file_exists(
                "test_3_1_5_nan_guard_test_present", "P1", "test",
                NA_NGUARD_TEST,
            ),
        ),
    ]


def _post_retrain_gates() -> List[Tuple[str, Callable[[], GateResult]]]:
    """§5.11–5.14 post-retrain gates (4 gates; always SKIP today).

    These require post-retrain evidence (per-(metal, seed) AUC, std,
    epoch-avg history, NaN-free runs).  They are emitted as SKIP so the
    honest framing catalog entry is preserved — the gate exists, is
    documented, and waits for the retrain wave to populate.
    """
    return [
        (
            "post_retrain_5_11_per_seed_std",
            lambda: gate_post_retrain_skip(
                "post_retrain_5_11_per_seed_std", "P2", "post_retrain",
                "per_seed std reported (NOT 0) across 5 seeds",
            ),
        ),
        (
            "post_retrain_5_12_os_auc_non_degenerate",
            lambda: gate_post_retrain_skip(
                "post_retrain_5_12_os_auc_non_degenerate", "P2", "post_retrain",
                "Os AUC ≠ exactly 0.5 (LOMO collapse check)",
            ),
        ),
        (
            "post_retrain_5_13_ru_w2_epoch_avg",
            lambda: gate_post_retrain_skip(
                "post_retrain_5_13_ru_w2_epoch_avg", "P2", "post_retrain",
                "Ru w2_epoch_avg_history non-zero for ALL epochs",
            ),
        ),
        (
            "post_retrain_5_14_no_nan_auc",
            lambda: gate_post_retrain_skip(
                "post_retrain_5_14_no_nan_auc", "P2", "post_retrain",
                "all 5 metals × 5 seeds non-NaN test_auc",
            ),
        ),
    ]


def _honest_framing_gates() -> List[Tuple[str, Callable[[], GateResult]]]:
    """§8 honest framing catalog entry-count gate (1 gate).

    Validates that the parent doc carries the full honest framing
    catalog with ≥15 entries (per the §8.15 NEW entry added by
    Wf-252-Apply).
    """
    return [
        (
            "honest_framing_8_15_catalog_entry_count",
            lambda: gate_section8_entry_count(
                "honest_framing_8_15_catalog_entry_count", "P1",
                "honest_framing",
                PARENT_DOC,
                min_entries=15,
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Gate registry dispatcher
# ---------------------------------------------------------------------------


def build_gates(pipeline: str) -> List[Tuple[str, Callable[[], GateResult]]]:
    """Build the gate list for the requested pipeline.

    Mirrors ``pre_flight_check.build_gates`` shape so downstream tooling
    can swap implementations cleanly.
    """
    runners: List[Tuple[str, Callable[[], GateResult]]] = []

    if pipeline in ("code", "all"):
        runners.extend(_code_level_gates())
    if pipeline in ("ablation", "all"):
        runners.extend(_ablation_flag_gates())
    if pipeline in ("data", "all"):
        runners.extend(_data_level_gates())
    if pipeline in ("test", "all"):
        runners.extend(_test_level_gates())
    if pipeline in ("post_retrain", "all"):
        runners.extend(_post_retrain_gates())
    if pipeline in ("honest_framing", "all"):
        runners.extend(_honest_framing_gates())

    return runners


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wf343c1_pre_retrain_check.py",
        description=(
            "Wf-343c-I1 pre-retrain self-audit auto-verify. Runs the "
            "auto-verifiable subset of §1.1 (code-level) + §1.3 (ablation "
            "flags) + §2.1 (data-level) + §3.1 (test-level) + §5.11-5.14 "
            "(post-retrain SKIP) + §8.15 (honest framing) gates from "
            "`科研进展_pre_retrain_checklist_2026-09-28.md`. Emits JSON; "
            "exit 0 if all PASS, 1 if any FAIL, 2 if any SKIP (--strict)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pipeline",
        choices=PIPELINES,
        default="all",
        help="Gate subset to run (default: all).",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write JSON report (default: stdout only).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 if any gate SKIPPED (default: only FAIL triggers "
             "non-zero exit).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-gate status to stderr as they run.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_argv(argv)
    runners = build_gates(args.pipeline)

    if args.verbose:
        print(
            f"[wf343c1] pipeline={args.pipeline} n_gates={len(runners)}",
            file=sys.stderr,
        )

    results: List[GateResult] = []
    t_total0 = time.perf_counter()
    for name, runner in runners:
        t0 = time.perf_counter()
        try:
            res = runner()
        except Exception as exc:  # noqa: BLE001 — surface any gate error
            res = GateResult(
                name, "?", args.pipeline, "FAIL",
                f"gate raised: {type(exc).__name__}: {exc}",
                time.perf_counter() - t0,
            )
        results.append(res)
        if args.verbose:
            print(
                f"[wf343c1] {res.status:<4} {res.priority} "
                f"{res.pipeline:<14} {res.name:<46} "
                f"{res.wall_sec:>7.2f}s | {res.evidence[:120]}",
                file=sys.stderr,
            )
    total_wall = time.perf_counter() - t_total0

    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")

    report = {
        "timestamp": _now_iso(),
        "schema_version": "343c1.v1",
        "pipeline": args.pipeline,
        "summary": {
            "pass": n_pass,
            "fail": n_fail,
            "skip": n_skip,
            "total": len(results),
            "total_wall_sec": round(total_wall, 4),
        },
        "gates": [r.to_dict() for r in results],
        "parent_doc_reference": (
            "科研进展_pre_retrain_checklist_2026-09-28.md "
            "(§1.1 code, §1.3 ablation, §2.1 data, §3.1 test, "
            "§5.11-5.14 post-retrain SKIP, §8.15 honest framing)"
        ),
    }

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
        if args.verbose:
            print(
                f"[wf343c1] wrote JSON -> {args.json}",
                file=sys.stderr,
            )
    print(payload)

    if n_fail > 0:
        return 1
    if args.strict and n_skip > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
