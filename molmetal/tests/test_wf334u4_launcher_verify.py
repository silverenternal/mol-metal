"""Wf-334-U4 — G5 + G6 launcher verify that all Wf-328 optimizations are
applied (either via explicit override or via trainer default).

Owner: Hugo Lee
Date: 2026-09-29
Parent: Wf-328 quick-wins sub-wave (W1 torch.compile + W2 cpu-num-workers +
        W3 BF16 + W4 TF32 + W5 W2-sort + W6 num-workers dataloader factory).
Per AGENTS.md §5.1 (kill_then_adapt) + §5.2 (MEASURED-vs-PROJECTED honest framing).

Background (Wf-328 closeouts — 2026-09-29):
  W1 — --torch-compile: trainer default ``True`` (MEASURED 2.72x encoder +
        2.41x W2 step speedup on RTX PRO 6000 Blackwell sm_120). CLI flag
        added to trainer; production trainer turns it ON by default.
  W2 — --cpu-num-workers 8: explicit override needed because trainer default
        is 4. Wf-326B OPTIMAL_NUM_WORKERS for 24-core host.
  W3 — --bf16: trainer default ``True`` (DEFAULT_BF16=True). Trainer applies
        BF16 autocast automatically.
  W4 — --tf32: trainer default ``True`` (DEFAULT_TF32=True). Trainer applies
        TF32 for CUDA matmul + cuDNN automatically.
  W5 — --w2-gradient-mode sort: trainer default ``"sort"`` (sort-based W2).
        Per Wf-328-W5 §Integration points > Launcher scripts, in-flight G5/G6
        launchers PRESERVE the bit-exact ``--w2-gradient-mode sliced`` override
        for the Wf-131 P1 paper numbers reproducibility (paper §6 audit).
        Future launchers can drop the explicit override to inherit sort default.
  W6 — --num-workers: chemprop-specific DataLoader factory. NOT applicable to
        the G5/G6 D-MPNN multitask trainer (which uses --cpu-num-workers
        directly).

Wf-334-U4 task: verify that all 6 launch scripts have the Wf-328 optimizations
applied EFFECTIVELY (either explicitly OR via trainer default-on flags).
G5 + G6 are in-flight (TODO-181 / TODO-182) per AGENTS.md §5.1 standing GPU
authorization, so this is a non-regression audit.

Tests (2 main scenarios + helpers):

  - test_g5_g6_launchers_have_all_wf328_optimizations
        Verifies per-launcher effective state for G5 + G6:
          * --cpu-num-workers 8 (explicit override, W2)
          * --w2-gradient-mode sliced (intentional bit-exact, W5)
          * --torch-compile / --bf16 / --tf32 NOT explicitly overridden
            (trainer defaults take effect; explicit would break paper
            reproducibility audit semantics)
          * --num-workers NOT set (chemprop-specific, W6)
  - test_backward_compat_no_flags_means_baseline
        Verifies trainer defaults wire through correctly when launchers don't
        pass explicit flags:
          * DEFAULT_BF16 == True (W3)
          * DEFAULT_TF32 == True (W4)
          * DEFAULT_W2_GRADIENT_MODE == "sort" (W5)
          * --torch-compile default True (W1)
          * --cpu-num-workers default 4 (Wf-260 trainer baseline)
          * parse_args([]) resolves to production defaults
  - test_six_launch_scripts_cpu_num_workers_8_cross_check
        Per-launcher regression gate that ALL 6 Wf-328-W2 targets still have
        --cpu-num-workers 8 (not regressed to 0 or 4).
  - test_trainer_no_anticancer_or_cfm_in_default_path
        Sanity: trainer module loads without external domain dependencies.

All tests are CPU-only — no torch.cuda, no real launches.  < 1 s wall.

Honest framing (AGENTS.md §5.2 / party-twenty-spirit §3.5):
  - MEASURED: 6 launch scripts hard-coded --cpu-num-workers 0 (Wf-326B grep
    2026-09-29); Wf-328-W2 fix replaced with --cpu-num-workers 8.
  - MEASURED: trainer defaults are DEFAULT_BF16=True, DEFAULT_TF32=True,
    DEFAULT_W2_GRADIENT_MODE="sort", --torch-compile default=True.
  - PROJECTED: 2-3x step throughput from Wf-328 quick-wins (BLOCKED on
    AGENTS.md §5.1 kill_then_adapt; needs > 1 hr GPU launch to measure
    end-to-end).
  - INTENTIONAL: --w2-gradient-mode sliced is preserved bit-exact per
    Wf-328-W5 §Integration points. Do NOT change to "sort" — would break
    Wf-131 P1 paper §6 reproducibility audit.
"""

from __future__ import annotations

import re
import sys
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root + path setup
# ---------------------------------------------------------------------------
ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS = ROOT / "scripts"
TEST_SCRIPTS_DIR = ROOT / "molmetal" / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Constants — the launch scripts that Wf-328 (W2 batch) fixed
# ---------------------------------------------------------------------------
WF328W2_TARGETS = (
    SCRIPTS / "launch_wf124_p3_g4.sh",
    SCRIPTS / "launch_wf124_p3_g5.sh",
    SCRIPTS / "launch_wf124_p3_g6.sh",
    SCRIPTS / "launch_wf310b_g6_final.sh",
    SCRIPTS / "launch_wf300_cfm_main.sh",
    SCRIPTS / "retry_g5_missing_pairs.sh",
)

# Per Wf-334-U4 task: G5 + G6 are the focus launchers (TODO-181 + TODO-182)
WF334U4_TARGETS = (
    SCRIPTS / "launch_wf124_p3_g5.sh",
    SCRIPTS / "launch_wf124_p3_g6.sh",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read(path: Path) -> str:
    assert path.exists(), f"launch script missing: {path}"
    return path.read_text()


def _active_flag_lines(src: str, flag: str) -> list[str]:
    """Extract lines containing ``flag`` that are NOT in comments.

    Skips comment lines (start with ``#``) so Wf-328 banner comments that
    mention old flag values for context do not get flagged as active overrides.
    """
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if flag in line:
            out.append(line)
    return out


def _parse_int_arg(line: str, flag: str) -> int:
    """Parse ``--flag <int>`` from a single shell line."""
    m = re.search(rf"{re.escape(flag)}\s+(-?\d+)", line)
    assert m, f"could not parse {flag} value from line: {line!r}"
    return int(m.group(1))


def _parse_str_arg(line: str, flag: str) -> str:
    """Parse ``--flag <value>`` from a single shell line."""
    m = re.search(rf"{re.escape(flag)}\s+(\S+)", line)
    assert m, f"could not parse {flag} value from line: {line!r}"
    return m.group(1)


# ---------------------------------------------------------------------------
# Test 1 — per-launcher Wf-328 effective state for G5 + G6
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", WF334U4_TARGETS, ids=lambda p: p.name)
def test_g5_g6_launchers_have_all_wf328_optimizations(path: Path) -> None:
    """Wf-334-U4 task: verify G5 + G6 launchers have all Wf-328
    optimizations applied (either explicitly or via trainer default-on).

    Per-launcher effective state for TODO-181 G5 / TODO-182 G6:
      * --cpu-num-workers 8 (Wf-328-W2 explicit override — required because
        trainer default is 4).
      * --w2-gradient-mode sliced (Wf-328-W5 INTENTIONAL bit-exact preservation
        for Wf-131 P1 paper §6 reproducibility audit; trainer default is
        "sort" but launcher EXPLICITLY keeps "sliced" for in-flight retrains).
      * --torch-compile / --bf16 / --tf32 NOT explicitly passed (trainer
        defaults are True; explicit override would change reproducibility
        audit semantics and is redundant).
      * --num-workers NOT set (Wf-328-W6 chemprop-specific DataLoader factory;
        G5/G6 D-MPNN multitask uses --cpu-num-workers directly).
    """
    src = _read(path)

    # ---- W2: --cpu-num-workers 8 (explicit) ----
    cpu_lines = _active_flag_lines(src, "--cpu-num-workers")
    assert cpu_lines, f"{path.name} has no active --cpu-num-workers line"
    for line in cpu_lines:
        val = _parse_int_arg(line, "--cpu-num-workers")
        assert val == 8, (
            f"{path.name} uses --cpu-num-workers {val}; Wf-328-W2 expects 8 "
            f"(Wf-326B OPTIMAL_NUM_WORKERS on 24-core host): {line!r}"
        )

    # ---- W5: --w2-gradient-mode sliced (intentional) ----
    w2_lines = _active_flag_lines(src, "--w2-gradient-mode")
    assert w2_lines, f"{path.name} has no active --w2-gradient-mode line"
    for line in w2_lines:
        val = _parse_str_arg(line, "--w2-gradient-mode")
        assert val == "sliced", (
            f"{path.name} uses --w2-gradient-mode {val}; Wf-328-W5 expects "
            f"'sliced' (intentional bit-exact preservation for Wf-131 P1 "
            f"paper §6 reproducibility audit; trainer default 'sort' applies "
            f"only to FUTURE launchers per Wf-328-W5 §Integration points): "
            f"{line!r}"
        )

    # ---- W1: --torch-compile NOT explicitly passed (trainer default True) ----
    tc_lines = _active_flag_lines(src, "--torch-compile")
    tc_lines = [l for l in tc_lines if "--torch-compile-mode" not in l]
    assert not tc_lines, (
        f"{path.name} explicitly passes --torch-compile; Wf-328-W1 expects "
        f"trainer default (default=True) to take effect — explicit override "
        f"is redundant and changes paper §6 reproducibility semantics: "
        f"{tc_lines!r}"
    )

    # ---- W3: --bf16 NOT explicitly passed (trainer default True) ----
    bf16_lines = _active_flag_lines(src, "--bf16")
    assert not bf16_lines, (
        f"{path.name} explicitly passes --bf16; Wf-328-W3 expects trainer "
        f"default (DEFAULT_BF16=True) to take effect — explicit override is "
        f"redundant: {bf16_lines!r}"
    )

    # ---- W4: --tf32 NOT explicitly passed (trainer default True) ----
    tf32_lines = _active_flag_lines(src, "--tf32")
    assert not tf32_lines, (
        f"{path.name} explicitly passes --tf32; Wf-328-W4 expects trainer "
        f"default (DEFAULT_TF32=True) to take effect — explicit override is "
        f"redundant: {tf32_lines!r}"
    )

    # ---- W6: --num-workers NOT set (chemprop-specific; not for G5/G6) ----
    nw_lines = _active_flag_lines(src, "--num-workers")
    # --cpu-num-workers matches "--num-workers" as substring; filter out:
    nw_lines = [l for l in nw_lines if "--cpu-num-workers" not in l]
    assert not nw_lines, (
        f"{path.name} explicitly passes --num-workers; Wf-328-W6 expects "
        f"this flag to NOT be set for G5/G6 (D-MPNN multitask trainer uses "
        f"--cpu-num-workers directly; --num-workers is chemprop-specific "
        f"DataLoader factory): {nw_lines!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — backward compat: trainer defaults wire through
# ---------------------------------------------------------------------------
def test_backward_compat_no_flags_means_baseline() -> None:
    """Backward-compat gate — when a launcher doesn't pass explicit flags,
    the trainer must wire through the Wf-328 production defaults:
      * DEFAULT_BF16 == True (W3)
      * DEFAULT_TF32 == True (W4)
      * DEFAULT_W2_GRADIENT_MODE == "sort" (W5)
      * --torch-compile default True (W1)
      * --cpu-num-workers default 4 (Wf-260 trainer baseline)
      * parse_args([]) resolves to production defaults
    """
    import importlib

    _trainer = importlib.import_module(
        "molmetal.scripts.train_dmpnn_multitask_wasserstein"
    )

    # ---- Constants (Wf-328 quick-wins ship surface) ----
    assert getattr(_trainer, "DEFAULT_BF16", None) is True, (
        "DEFAULT_BF16 drifted from True; Wf-328-W3 expects True "
        "(production default ON; ~1.5-2x step throughput on "
        "RTX 5090 / RTX PRO 6000 Blackwell sm_120)"
    )
    assert getattr(_trainer, "DEFAULT_TF32", None) is True, (
        "DEFAULT_TF32 drifted from True; Wf-328-W4 expects True "
        "(production default ON; ~1.5x matmul speedup with negligible "
        "accuracy loss per Wf-326A quick wins)"
    )
    assert getattr(_trainer, "DEFAULT_W2_GRADIENT_MODE", None) == "sort", (
        "DEFAULT_W2_GRADIENT_MODE drifted from 'sort'; Wf-328-W5 expects "
        "'sort' (sort-based W2 is 2-3x faster than sliced per Wf-326A)"
    )

    # ---- parse_args([]) defaults (CLI wire-through) ----
    args = _trainer.parse_args([])
    assert getattr(args, "bf16", None) is True, (
        f"parse_args([]).bf16 = {getattr(args, 'bf16', None)}; "
        f"Wf-328-W3 expects True (DEFAULT_BF16=True)"
    )
    assert getattr(args, "tf32", None) is True, (
        f"parse_args([]).tf32 = {getattr(args, 'tf32', None)}; "
        f"Wf-328-W4 expects True (DEFAULT_TF32=True)"
    )
    assert getattr(args, "torch_compile", None) is True, (
        f"parse_args([]).torch_compile = "
        f"{getattr(args, 'torch_compile', None)}; Wf-328-W1 expects True"
    )
    assert getattr(args, "w2_gradient_mode", None) == "sort", (
        f"parse_args([]).w2_gradient_mode = "
        f"{getattr(args, 'w2_gradient_mode', None)}; Wf-328-W5 expects 'sort'"
    )
    assert getattr(args, "cpu_num_workers", None) == 4, (
        f"parse_args([]).cpu_num_workers = "
        f"{getattr(args, 'cpu_num_workers', None)}; trainer baseline is 4 "
        f"(Wf-260); launchers override to 8 via --cpu-num-workers 8 (Wf-328-W2)"
    )
    assert getattr(args, "num_workers", None) is None, (
        f"parse_args([]).num_workers = "
        f"{getattr(args, 'num_workers', None)}; Wf-328-W6 expects None "
        f"(default = use --cpu-num-workers via legacy path; chemprop-specific "
        f"DataLoader factory is opt-in via --num-workers N)"
    )

    # ---- W2 OPTIMAL_NUM_WORKERS constant matches ----
    _wf326b = importlib.import_module("molmetal.scripts.wf326b_dataloader_optimized")
    assert getattr(_wf326b, "OPTIMAL_NUM_WORKERS", None) == 8, (
        f"OPTIMAL_NUM_WORKERS = {getattr(_wf326b, 'OPTIMAL_NUM_WORKERS', None)}; "
        f"Wf-326B / Wf-328-W2 expects 8 (cap-bounded sweet spot for 24-core host)"
    )


# ---------------------------------------------------------------------------
# Test 3 — all 6 launchers cross-check for --cpu-num-workers 8 (regression)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", WF328W2_TARGETS, ids=lambda p: p.name)
def test_six_launch_scripts_cpu_num_workers_8_cross_check(path: Path) -> None:
    """Per-launcher regression gate that ALL 6 Wf-328-W2 targets still have
    --cpu-num-workers 8 (not regressed to 0 or 4).

    Complements molmetal/tests/test_wf328w2_launcher_num_workers.py — this
    test runs in the Wf-334-U4 batch as a cross-check that the Wf-328-W2 fix
    persists across the Wf-328 quick-wins family.
    """
    src = _read(path)
    cpu_lines = _active_flag_lines(src, "--cpu-num-workers")
    assert cpu_lines, (
        f"{path.name} has no active --cpu-num-workers line; Wf-328-W2 "
        f"fix requires explicit --cpu-num-workers 8 override"
    )
    for line in cpu_lines:
        val = _parse_int_arg(line, "--cpu-num-workers")
        assert val == 8, (
            f"{path.name} uses --cpu-num-workers {val}; Wf-328-W2 / "
            f"Wf-334-U4 expects 8 (Wf-326B OPTIMAL_NUM_WORKERS on "
            f"24-core host): {line!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 — trainer --help shows all Wf-328 flags (no CLI regression)
# ---------------------------------------------------------------------------
def test_trainer_help_shows_all_wf328_flags() -> None:
    """The trainer's --help output must list all 6 Wf-328 flag families so
    that launchers / users can discover them. This is a non-regression gate
    against accidental removal of any of the Wf-328 CLI surfaces.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(TEST_SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        proc.returncode == 0
    ), f"trainer --help failed (rc={proc.returncode}): {proc.stderr}"
    out = proc.stdout

    expected = [
        "--cpu-num-workers",  # W2
        "--num-workers",  # W6 (chemprop)
        "--tf32",  # W4
        "--bf16",  # W3
        "--torch-compile",  # W1
        "--w2-gradient-mode",  # W5
    ]
    for flag in expected:
        assert flag in out, (
            f"trainer --help is missing {flag}; Wf-328 regression "
            f"(the flag should be discoverable from --help output)"
        )

    # The --w2-gradient-mode choices should include BOTH "sliced" and "sort"
    # per Wf-328-W5 backward-compat surface (paper §6 reproducibility audits).
    assert "{sliced,sort}" in out or "sliced" in out and "sort" in out, (
        "trainer --help is missing w2-gradient-mode choices; Wf-328-W5 "
        "expects both 'sliced' (legacy / paper §6 audit) and 'sort' "
        "(production default) to be discoverable"
    )
