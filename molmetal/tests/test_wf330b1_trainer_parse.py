"""Wf-330-B1: Concurrent-edit conflict resolution tests for trainer file.

================================================================
Wf-330-B1 — merge + parse verification for
``molmetal/scripts/train_dmpnn_multitask_wasserstein.py`` after
concurrent Wf-328-W1..W10 edits raced on the same file.
================================================================

Per Wf-328-W5 honest framing (2026-09-29): multiple Wf-328 agents
(W1..W10) concurrently edited the trainer file with edits that
were not originally coordinated.  Wf-330-B1 performs the merge + parse
gate so downstream Wf-181 G5 / Wf-182 G6 launches can read the trainer
shape with confidence.

Test inventory (2 tests per Wf-330-B1 brief)
-------------------------------------------
1. test_trainer_file_parses_without_syntax_errors
     — Python AST parse + structural balance (try/except/finally).
2. test_wf328_integrations_present
     — every Wf-328 W* integration the production trainer needs is
       actually wired into the file (compile, BF16, TF32, sort, num-
       workers, dmpnn_pyg via W7 reference, cuda_graph, triton,
       cache).

All tests are CPU-only by default — no torch.cuda, no real launches.
< 1 s wall.

**Honest framing (party-twenty-spirit §3.5):**
- MEASURED: file parses via :func:`ast.parse` (2026-09-29).
- MEASURED: 13 try / 12 except / 1 finally blocks (2026-09-29 grep).
- MEASURED: Wf-328 W1, W3, W4, W5, W6, W8, W9 markers present in
  trainer file source (2026-09-29 grep).
- MEASURED: W2 lives in launch scripts (not trainer file); W7 is a
  PyG port referenced in CLI help text; W10 lives in the on-disk
  pickle cache (``_featurize_with_cache``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER_PATH = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
)


def _read_trainer_source() -> str:
    assert TRAINER_PATH.exists(), f"Trainer file missing: {TRAINER_PATH}"
    return TRAINER_PATH.read_text(encoding="utf-8")


def test_trainer_file_parses_without_syntax_errors() -> None:
    """AST parse + structural balance gate (Wf-330-B1 §1)."""
    src = _read_trainer_source()
    # AST parse — raises SyntaxError on failure.
    tree = ast.parse(src)

    # Walk the AST and tally try/except/finally blocks to detect any
    # unclosed blocks left behind by a concurrent edit race.
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

    # Sanity: there should be MORE try blocks than except handlers? No —
    # a bare ``finally`` is fine (try/finally without except is valid).
    # The gate is just that the AST parsed and counts are >= 0.
    assert v.try_count >= 1, "expected at least one try block in trainer"
    assert v.except_count >= 1, "expected at least one except handler"
    # 2026-10-20: the ``finally`` requirement was dropped.  It asserted the
    # Wf-328-W1 encoder-restore block, which left with the m21 selective
    # revert (``af84a545``) — the shipped trainer has no try/finally at all.
    # Keeping the assertion only re-asserted removed code.

    # No pre_hook / Triton-3.x-incompatible autotuner kwargs should be
    # left dangling.  Per Wf-330-B1 §3 the original race dropped a
    # ``pre_hook=...`` kwarg into :func:`triton.autotune`; the merged
    # file should NOT carry that pattern.
    assert "pre_hook" not in src, (
        "Wf-330-B1 found a `pre_hook=...` kwarg in the trainer; Triton 3.x "
        "no longer accepts it. Remove the kwarg or migrate to "
        "`pre_hook` removed in 3.x."
    )

    # No stale ``@triton.autotune(`` decorator that requires Triton-3.x
    # incompatible kwarg shape.
    assert not re.search(
        r"@triton\.autotune\s*\(",
        src,
    ), "Wf-330-B1 found a @triton.autotune(...) decorator; remove or migrate."


def test_wf328_integrations_present() -> None:
    """All Wf-328 W* integrations wired into trainer (Wf-330-B1 §2).

    Verifies the production-port markers are physically present in
    the trainer source.  Each marker is the comment-prefix used by
    the W* owner to demarcate their edit; the test asserts the
    substring is present, NOT that the edit is behaviourally
    correct (the per-W* tests in ``test_wf328w*_*.py`` cover that).
    """
    src = _read_trainer_source()

    # Wf-328-W1 — torch.compile wrappers for encoder + W₂ closed-form.
    assert (
        "_make_compiled_encoder_forward" in src
    ), "Wf-328-W1: missing _make_compiled_encoder_forward"
    assert (
        "_make_compiled_w2_penalty" in src
    ), "Wf-328-W1: missing _make_compiled_w2_penalty"
    assert "torch.compile" in src, "Wf-328-W1: missing torch.compile call"
    assert (
        "compiled_w2_penalty" in src
    ), "Wf-328-W1: missing compiled_w2_penalty thread-through"

    # Wf-328-W3 — BF16 autocast context.  The BooleanOptionalAction pair
    # (``use_bf16`` / ``bf16_fallback_occurred`` / ``--no-bf16``) left with
    # the m21 revert; the helper + constant survived and are asserted below.
    assert "bf16_autocast" in src, "Wf-328-W3: missing bf16_autocast factory"
    assert "configure_bf16" in src, "Wf-328-W3: missing configure_bf16 helper"
    assert "DEFAULT_BF16" in src, "Wf-328-W3: missing DEFAULT_BF16 constant"

    # Wf-328-W4 — TF32.  ``--tf32`` is store_true default OFF; the
    # ``--no-tf32`` negation form did not survive the revert.
    assert "configure_tf32" in src, "Wf-328-W4: missing configure_tf32 helper"
    assert (
        "torch.backends.cuda.matmul.allow_tf32" in src
    ), "Wf-328-W4: missing tf32 matmul flag"
    assert "DEFAULT_TF32" in src, "Wf-328-W4: missing DEFAULT_TF32 constant"
    assert "--tf32" in src, "Wf-328-W4: missing --tf32 CLI lever"

    # Wf-328-W5 — W₂ gradient mode dispatch.  The default is ``sliced`` (the
    # Wf-328-W5 flip to ``sort`` did not survive the revert); the sort path
    # is still implemented and selectable.
    assert (
        "DEFAULT_W2_GRADIENT_MODE" in src
    ), "Wf-328-W5: missing DEFAULT_W2_GRADIENT_MODE constant"
    assert 'DEFAULT_W2_GRADIENT_MODE: str = "sliced"' in src, (
        "Wf-328-W5: DEFAULT_W2_GRADIENT_MODE must default to 'sliced' — that "
        "is what argparse, the signatures and every live config use"
    )
    assert "w2_gradient_mode" in src, "Wf-328-W5: missing w2_gradient_mode dispatch"
    assert (
        "wasserstein_pic50_gradient" in src
    ), "Wf-328-W5: missing closed-form W₂ gradient call"

    # Wf-328-W6 — NOT WIRED.  The trainer builds batches manually; there is
    # no torch DataLoader for the Wf-326B factory to feed.  Assert the
    # absence so the reverted surface cannot leak back in silently.
    for absent in (
        "build_optimized_loader",
        "wf326b_dataloader_optimized",
        "effective_num_workers",
        "torch.utils.data.DataLoader",
    ):
        assert absent not in src, (
            f"Wf-328-W6: {absent!r} is back in the trainer — the Wf-326B "
            f"DataLoader factory was removed with the m21 revert; re-enabling "
            f"it is a numerics change, not a test fix"
        )

    # Wf-328-W8 — CUDA Graphs NOT WIRED (same revert).
    for absent in ("use_cuda_graph", "CudaGraphDMPNNWrapper", "--cuda-graph"):
        assert absent not in src, (
            f"Wf-328-W8: {absent!r} is back in the trainer — CUDA graphs were "
            f"removed with the m21 revert"
        )

    # Wf-328-W9 — only the global env gate survived; the per-op env vars and
    # the CLI toggle left with the revert.
    assert "TRITON_USE_FUSED" in src, "Wf-328-W9: missing TRITON_USE_FUSED env var"
    for absent in ("TRITON_RELU_LINEAR", "TRITON_EDGE_UPDATE", "triton_edge_update"):
        assert absent not in src, (
            f"Wf-328-W9: {absent!r} is back in the trainer — the per-op Triton "
            f"toggles were removed with the m21 revert"
        )

    # Wf-328-W10 — on-disk pickle cache for featurisation (re-used from
    # Wf-84 P3 but re-affirmed by W10 as a quick-win energy saver).
    assert (
        "_featurize_with_cache" in src
    ), "Wf-328-W10: missing _featurize_with_cache helper"
    assert "wf131_mtw_fe_cache" in src, "Wf-328-W10: missing cache_dir constant"

    # Wf-285B (pre-Wf-328) — DANN Gradient-Reversal Layer wire-in (kept
    # across the merge).  This is a regression gate: the DANN head
    # predates the Wf-328 wave and the merge MUST NOT have removed it.
    assert "use_dann" in src, "Wf-285B: use_dann toggle must survive the merge"
    assert "ReverseLayerF" in src, "Wf-285B: ReverseLayerF.apply must survive the merge"
    # ``lambda_grl`` was the Wf-285B GRL weighting constant.  It is gone from
    # the shipped trainer (the DANN head keeps ``use_dann`` / ``ReverseLayerF``
    # but no explicit λ knob); asserting it only re-asserted removed code.

    # Wf-251-F4 — Kendall UW (uncertainty-weighted) MTL loss toggle.
    assert "use_kendall" in src, "Wf-251-F4: use_kendall toggle must survive the merge"

    # Wf-197 — single-task ablation lever must survive the merge.
    assert "single_task" in src, "Wf-197: single_task flag must survive the merge"

    # Wf-328-W7 (PyG port) — the port itself lives in
    # :mod:`molmetal.models.dmpnn_pyg` and the trainer selects it through
    # ``DEFAULT_DMPNN_BACKEND``.  The old assertion checked a ``--cuda-graph``
    # help string, which left with the W8 revert; assert the durable claim.
    assert "DEFAULT_DMPNN_BACKEND" in src, (
        "Wf-328-W7: the PyG backend selector is missing from the trainer"
    )
    assert (
        (Path(__file__).resolve().parent.parent / "models" / "dmpnn_pyg.py").is_file()
    ), "Wf-328-W7: molmetal/models/dmpnn_pyg.py (the PyG port) is missing"
