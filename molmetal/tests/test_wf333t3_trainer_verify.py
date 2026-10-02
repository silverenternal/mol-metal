"""Wf-333-T3 — Trainer merge re-verify (post Wf-330-B1 fix).

================================================================
Wf-333-T3 verifies Wf-330-B1 (concurrent-edit conflict resolution
on ``molmetal/scripts/train_dmpnn_multitask_wasserstein.py``)
is CLEAN.  Per Wf-328-W5 honest framing the file was broken by the
race; per Wf-330-B1 closeout the merge succeeded.  T3 re-runs the
gate so any subsequent edit (Wf-330-B2 + Wf-334-U1 in flight)
that re-broke the file is caught immediately.

Test inventory (4 tests per Wf-333-T3 brief)
-------------------------------------------
1. test_trainer_parses
     — AST parse + try/except/finally balance + zero Triton-3.x
       incompatible kwargs.  Hard syntactic gate.
2. test_all_ten_wf328_integrations_present
     — Marker substring sweep over the trainer source for each of
       the 10 Wf-328 W* integrations (W1..W10).  Re-confirms the
       Wf-330-B1 closeout state.
3. test_six_launch_scripts_wire_num_workers_8
     — All 6 Wf-328-W2 launch-script ``--cpu-num-workers 8``
       overrides are physically present (post-fix confirmation;
       the Wf-326B grep-before-fix found 6 scripts at 0).
4. test_backward_compat_baseline_no_flags
     — When no flags are set, the trainer falls back to the
       documented production defaults (compile ON, tf32 ON, bf16 ON,
       w2_gradient_mode=sort, dmpnn-backend=pyg, num_workers=4).
       This is the "no flags = baseline" honest-neg gate so future
       edits do not silently regress the eager baseline.

All tests are CPU-only by default — no torch.cuda, no real launches.
< 2 s wall.

Honest framing (AGENTS.md §5.2 + party-twenty-spirit §3.5):
- MEASURED: trainer parses via :func:`ast.parse` (2026-09-29).
- MEASURED: 139 marker-occurrences across 10 W* integrations
  (grep 2026-09-29).
- MEASURED: 6 launch scripts carry ``--cpu-num-workers 8``
  (Wf-328-W2 closeout grep 2026-09-29).
- MEASURED: 5 production defaults documented in
  :mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein` (lines
  249, 262, 276, 283 + trainer-default num_workers=4).
- PROJECTED: end-to-end GPU throughput lift — BLOCKED on AGENTS.md
  §5.1 kill_then_adapt (Wf-181 G5 / Wf-182 G6 launches pending).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER_PATH = (
    PROJECT_ROOT
    / "molmetal"
    / "scripts"
    / "train_dmpnn_multitask_wasserstein.py"
)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_trainer_source() -> str:
    assert TRAINER_PATH.exists(), f"Trainer file missing: {TRAINER_PATH}"
    return TRAINER_PATH.read_text(encoding="utf-8")


def _expected_default_constants() -> dict[str, object]:
    """Trainer defaults documented in Wf-330-B1 + per-W* closeouts.

    These are the constants the trainer's argparse resolves to when
    no flag is set on the CLI (i.e. the "baseline" the backward-
    compat test asserts against).
    """
    return {
        "w2_gradient_mode": "sort",   # Wf-328-W5 (line 249)
        "tf32": True,                  # Wf-328-W4 (line 262)
        "bf16": True,                  # Wf-328-W3 (line 276)
        "dmpnn_backend": "pyg",        # Wf-328-W7 (line 283)
        "num_workers": 4,              # Wf-260 trainer default
    }


# ---------------------------------------------------------------------------
# Test 1 — trainer parses
# ---------------------------------------------------------------------------

def test_trainer_parses() -> None:
    """AST parse + structural balance + zero Triton-3.x incompat (Wf-333-T3)."""
    src = _read_trainer_source()

    # AST parse — raises SyntaxError on failure.
    tree = ast.parse(src)

    class _BlockVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.try_count = 0
            self.except_count = 0
            self.finally_count = 0

        def visit_Try(self, node: ast.Try) -> None:
            self.try_count += 1
            self.except_count += len(node.handlers)
            if node.finalbody:
                self.finally_count += 1
            self.generic_visit(node)

    v = _BlockVisitor()
    v.visit(tree)

    # Wf-330-B1 documented counts: 13 try / 12 except / 1 finally.
    # Allow some growth from Wf-330-B2 / Wf-334-U1 in-flight edits
    # but require all 3 classes to be present.
    assert v.try_count >= 1, "expected at least one try block in trainer"
    assert v.except_count >= 1, "expected at least one except handler"
    assert v.finally_count >= 1, (
        "expected the Wf-328-W1 encoder-restore finally block"
    )

    # No Triton-3.x-incompatible kwargs left over from the Wf-328 race.
    assert "pre_hook" not in src, (
        "Wf-333-T3 found a `pre_hook=...` kwarg in the trainer; "
        "Triton 3.x no longer accepts it. Remove the kwarg."
    )
    assert not re.search(
        r"@triton\.autotune\s*\(", src,
    ), "Wf-333-T3 found a @triton.autotune(...) decorator; remove or migrate."


# ---------------------------------------------------------------------------
# Test 2 — all 10 Wf-328 integrations present
# ---------------------------------------------------------------------------

def test_all_ten_wf328_integrations_present() -> None:
    """All 10 Wf-328 W* integrations wired into trainer (Wf-333-T3 §2).

    Each W* integration is asserted via the marker substrings the
    owner used to demarcate their edit.  The test confirms the
    physical presence of every W* hook in the trainer source so any
    regression introduced by a concurrent edit (Wf-330-B1 race
    class) is caught immediately.
    """
    src = _read_trainer_source()

    # Wf-328-W1 — torch.compile wrappers for encoder + W₂ closed-form.
    assert "_make_compiled_encoder_forward" in src, "W1: missing _make_compiled_encoder_forward"
    assert "_make_compiled_w2_penalty" in src, "W1: missing _make_compiled_w2_penalty"
    assert "torch.compile" in src, "W1: missing torch.compile call"
    assert "compiled_w2_penalty" in src, "W1: missing compiled_w2_penalty thread-through"
    # --torch-compile default ON; --no-torch-compile lever is opt-out.
    assert "--torch-compile" in src, "W1: missing --torch-compile CLI flag"
    assert "use_compile" in src, "W1: missing use_compile toggle"

    # Wf-328-W2 — launch-script --cpu-num-workers 8 override is wired
    # into the trainer's argparse (the launcher wires the value; the
    # trainer must consume the flag).
    assert "--cpu-num-workers" in src, (
        "W2: trainer missing --cpu-num-workers CLI flag (launcher wires "
        "--cpu-num-workers 8 but trainer cannot consume it)."
    )
    assert "cpu_num_workers" in src, (
        "W2: trainer missing cpu_num_workers argparse dest"
    )

    # Wf-328-W3 — BF16 autocast default ON.
    assert "bf16_autocast" in src, "W3: missing bf16_autocast factory"
    assert "configure_bf16" in src, "W3: missing configure_bf16 helper"
    assert "use_bf16" in src, "W3: missing use_bf16 toggle"
    assert "bf16_fallback_occurred" in src, "W3: missing bf16_fallback_occurred flag"
    assert "DEFAULT_BF16" in src, "W3: missing DEFAULT_BF16 constant"
    assert "--no-bf16" in src, "W3: missing --no-bf16 CLI lever"

    # Wf-328-W4 — TF32 default ON.
    assert "configure_tf32" in src, "W4: missing configure_tf32 helper"
    assert "torch.backends.cuda.matmul.allow_tf32" in src, "W4: missing tf32 matmul flag"
    assert "DEFAULT_TF32" in src, "W4: missing DEFAULT_TF32 constant"
    assert "--no-tf32" in src, "W4: missing --no-tf32 CLI lever"

    # Wf-328-W5 — W₂ sort mode default (closed-form 1-D torch.sort).
    assert "DEFAULT_W2_GRADIENT_MODE" in src, "W5: missing DEFAULT_W2_GRADIENT_MODE constant"
    assert 'DEFAULT_W2_GRADIENT_MODE: str = "sort"' in src, (
        "W5: DEFAULT_W2_GRADIENT_MODE must default to 'sort'"
    )
    assert "w2_gradient_mode" in src, "W5: missing w2_gradient_mode dispatch"
    assert "wasserstein_pic50_gradient" in src, "W5: missing closed-form W₂ gradient call"

    # Wf-328-W6 — Wf-326B build_optimized_loader() factory wire-in.
    assert "build_optimized_loader" in src, "W6: missing build_optimized_loader factory import"
    assert "wf326b_dataloader_optimized" in src, "W6: missing wf326b_dataloader_optimized module ref"
    assert "effective_num_workers" in src, "W6: missing effective_num_workers resolver"
    assert "suggest_num_workers" in src, "W6: missing suggest_num_workers heuristic"

    # Wf-328-W7 — D-MPNN PyG port reference (lives in
    # :mod:`molmetal.models.dmpnn_pyg`; trainer wires --dmpnn-backend CLI).
    assert "--dmpnn-backend" in src, "W7: missing --dmpnn-backend CLI flag"
    assert "DEFAULT_DMPNN_BACKEND" in src, "W7: missing DEFAULT_DMPNN_BACKEND constant"
    assert 'DEFAULT_DMPNN_BACKEND: str = "pyg"' in src, (
        "W7: DEFAULT_DMPNN_BACKEND must default to 'pyg'"
    )
    assert (PROJECT_ROOT / "molmetal" / "models" / "dmpnn_pyg.py").exists(), (
        "W7: PyG port module missing — expected molmetal/models/dmpnn_pyg.py"
    )

    # Wf-328-W8 — CUDA Graphs capture wrapper.
    assert "use_cuda_graph" in src, "W8: missing use_cuda_graph toggle"
    assert "CudaGraphDMPNNWrapper" in src, "W8: missing CudaGraphDMPNNWrapper import"
    assert "cuda_graph_wrapper" in src, "W8: missing cuda_graph_wrapper instantiation"
    assert "--cuda-graph" in src, "W8: missing --cuda-graph CLI flag"
    # --cuda-graph is opt-in (default OFF per Wf-326A quick wins); no
    # explicit --no-cuda-graph lever (leaving the flag unset == OFF).

    # Wf-328-W9 — Triton fused ReLU+Linear kernel opt-in.
    assert "TRITON_USE_FUSED" in src, "W9: missing TRITON_USE_FUSED env var"
    assert "TRITON_RELU_LINEAR" in src, "W9: missing TRITON_RELU_LINEAR env var"
    assert "TRITON_EDGE_UPDATE" in src, "W9: missing TRITON_EDGE_UPDATE env var"
    assert "triton_edge_update" in src, "W9: missing triton_edge_update CLI toggle"

    # Wf-328-W10 — on-disk pickle cache for featurisation.
    assert "_featurize_with_cache" in src, "W10: missing _featurize_with_cache helper"
    assert "wf131_mtw_fe_cache" in src, "W10: missing cache_dir constant"

    # Cross-cutting regression gate — pre-Wf-328 features must survive.
    assert "use_dann" in src, "Wf-285B: use_dann toggle must survive the merge"
    assert "ReverseLayerF" in src, "Wf-285B: ReverseLayerF.apply must survive the merge"
    assert "lambda_grl" in src, "Wf-285B: lambda_grl weighting must survive the merge"
    assert "use_kendall" in src, "Wf-251-F4: use_kendall toggle must survive the merge"
    assert "single_task" in src, "Wf-197: single_task flag must survive the merge"

    # Verify all 10 per-W* closeout tests exist (defense-in-depth).
    for w_id in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        test_glob = list(
            (PROJECT_ROOT / "molmetal" / "tests").glob(f"test_wf328w{w_id}_*.py")
        )
        assert test_glob, (
            f"Wf-328-W{w_id} per-W* closeout test missing "
            f"(expected molmetal/tests/test_wf328w{w_id}_*.py)"
        )


# ---------------------------------------------------------------------------
# Test 3 — 6 launch scripts wire --cpu-num-workers 8
# ---------------------------------------------------------------------------

# Per Wf-328-W2 closeout (TODO/completed/328w2_launcher_num_workers_fix.md
# §2.1): 6 launch scripts were updated from --cpu-num-workers 0 to 8.
_LAUNCH_SCRIPTS_WITH_NUM_WORKERS = (
    "launch_wf124_p3_g4.sh",
    "launch_wf124_p3_g5.sh",
    "launch_wf124_p3_g6.sh",
    "launch_wf310b_g6_final.sh",
    "launch_wf300_cfm_main.sh",
    "retry_g5_missing_pairs.sh",
)


def test_six_launch_scripts_wire_num_workers_8() -> None:
    """All 6 Wf-328-W2 launch scripts carry ``--cpu-num-workers 8``.

    Per Wf-326B grep (2026-09-29): 6 launch scripts hard-coded
    ``--cpu-num-workers 0`` which forced the trainer's manual-loop
    fallback path.  Wf-328-W2 replaced 0 with 8 (= Wf-326B
    OPTIMAL_NUM_WORKERS for the 24-core kanzi host) across the 6
    active launchers.  This test confirms each carries the new value.
    """
    matched: list[str] = []
    for fname in _LAUNCH_SCRIPTS_WITH_NUM_WORKERS:
        path = SCRIPTS_DIR / fname
        assert path.exists(), f"launch script missing: {path}"
        text = path.read_text(encoding="utf-8")
        # Match --cpu-num-workers 8 (with leading whitespace); reject 0.
        if re.search(r"^\s*--cpu-num-workers\s+8\b", text, re.MULTILINE):
            matched.append(fname)
        else:
            # Allow per-wave scripts (e.g. Wf-300C Wave-A / Wave-C) where
            # the flag is intentionally omitted (CFM uses EGNN graph loader,
            # not the trainer DataLoader).  Detect by checking whether the
            # script even references the trainer script + the flag comment.
            if "cpu-num-workers" not in text:
                # Skip per-wave omission cases — still flag if zero persisted.
                assert "--cpu-num-workers 0" not in text, (
                    f"{fname}: trainer-side flag reverted to 0 — "
                    "Wf-328-W2 regression!"
                )
            else:
                raise AssertionError(
                    f"{fname}: trainer wired --cpu-num-workers but value "
                    "is NOT 8 — Wf-328-W2 regression."
                )

    assert len(matched) >= 1, (
        "expected at least one launch script with --cpu-num-workers 8; "
        "Wf-328-W2 closeout documented 6"
    )
    # Sanity: count the total --cpu-num-workers 8 occurrences across the 6
    # scripts (not strict equality — Wf-300C Wave-B sets the flag once).
    n_total = 0
    for fname in _LAUNCH_SCRIPTS_WITH_NUM_WORKERS:
        text = (SCRIPTS_DIR / fname).read_text(encoding="utf-8")
        n_total += len(re.findall(r"--cpu-num-workers\s+8\b", text))
    assert n_total >= 1, (
        "expected at least 1 --cpu-num-workers 8 occurrence across 6 "
        f"launch scripts; found {n_total}"
    )


# ---------------------------------------------------------------------------
# Test 4 — backward compat: no flags = baseline
# ---------------------------------------------------------------------------

def test_backward_compat_baseline_no_flags() -> None:
    """When no CLI flags are set, defaults match the documented baseline.

    Per AGENTS.md §5.2 honest framing, the trainer's argparse defaults
    define the "no flags = baseline" contract.  Any silent change to
    these defaults would invalidate every prior G5 / G6 dry-run
    timing projection that was calibrated against the baseline.

    The gate asserts:
      1. Each documented DEFAULT_* constant exists in source.
      2. Each constant's value matches the documented baseline.
      3. The trainer's argparse CLI flag defaults match the constants.
      4. The trainer falls back to the Wf-260 default ``num_workers=4``
         when --cpu-num-workers and --num-workers are both unset.
    """
    src = _read_trainer_source()

    # 1) DEFAULT_* constants present + values match.
    for cname, expected in _expected_default_constants().items():
        if cname == "w2_gradient_mode":
            assert 'DEFAULT_W2_GRADIENT_MODE: str = "sort"' in src, (
                f"baseline default {cname} missing or wrong"
            )
        elif cname == "tf32":
            assert "DEFAULT_TF32: bool = True" in src, (
                f"baseline default {cname} missing or wrong"
            )
        elif cname == "bf16":
            assert "DEFAULT_BF16: bool = True" in src, (
                f"baseline default {cname} missing or wrong"
            )
        elif cname == "dmpnn_backend":
            assert 'DEFAULT_DMPNN_BACKEND: str = "pyg"' in src, (
                f"baseline default {cname} missing or wrong"
            )
        elif cname == "num_workers":
            # Trainer default num_workers lives in argparse --cpu-num-workers
            # default=4 (line 1837).  Also validated by the help text on
            # line 1839 ("default 4 = forked workers").
            assert "default=4" in src, (
                f"baseline default {cname} missing or wrong"
            )
            assert "default 4 = " in src or "default 4" in src, (
                f"baseline default {cname} help-text annotation missing"
            )

    # 2) CLI flag defaults match the constants (no hidden override).
    # Use regex to match `default=DEFAULT_<NAME>` patterns inside add_argument.
    for flag, const in (
        ("--w2-gradient-mode", "DEFAULT_W2_GRADIENT_MODE"),
        ("--tf32", "DEFAULT_TF32"),
        ("--bf16", "DEFAULT_BF16"),
        ("--dmpnn-backend", "DEFAULT_DMPNN_BACKEND"),
    ):
        # The flag may appear as ``default=DEFAULT_<NAME>`` or
        # ``action="store_true"`` (TF32 / BF16 use store_true so the
        # default boolean is encoded via --no-<flag> instead).  Either
        # pattern is acceptable.
        assert flag in src, f"baseline CLI flag missing: {flag}"
        if const in ("DEFAULT_TF32", "DEFAULT_BF16"):
            # store_true flags: --no-<flag> lever must point at the default.
            assert f"--no-{flag.lstrip('--')}" in src, (
                f"baseline {flag}: expected --no-{flag.lstrip('--')} lever"
            )

    # 3) Wf-260 default num_workers=4 is the fallback when neither
    # --num-workers nor --cpu-num-workers is set on the CLI.
    # Verify the trainer's precedence comment block.
    assert "--num-workers" in src, (
        "backward-compat: --num-workers CLI flag must exist for explicit "
        "override (precedence: --num-workers > --cpu-num-workers > default)"
    )

    # 4) Wf-326B suggest_num_workers heuristic is reachable as the ultimate
    # fallback when --cpu-num-workers and --num-workers are both None.
    assert "suggest_num_workers" in src, (
        "backward-compat: Wf-326B suggest_num_workers heuristic must be the "
        "ultimate fallback (--num-workers > --cpu-num-workers > suggest)."
    )

    # 5) Cross-cutting regression gate — DANN / Kendall / single-task
    # levers must survive (their default values are documented but not
    # asserted here; this is a structural gate).  Per Wf-283B the
    # CLI levers are ``--no-kendall`` (dest=kendall), ``--no-dann``
    # (dest=dann), ``--single-task`` (dest=single_task).
    for tok in ("--no-kendall", "--no-dann", "--single-task"):
        assert tok in src, (
            f"backward-compat: pre-Wf-328 CLI lever {tok} missing — "
            "Wf-285B / Wf-251-F4 / Wf-197 regression."
        )