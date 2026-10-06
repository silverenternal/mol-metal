"""
Regression tests for 4 CONFIRMED defects in the D-MPNN line.

Per adversarial verifier report (all 4 CONFIRMED, structurally real):
1. Lines 1624-1632 — bare except RuntimeError silently drops a pool metal;
   n_metals_pool (line 1620) still uses len(metals_in_pool) so it reports
   "5-metal" even when one or more silently dropped.
2. Lines 2092-2097 — best_state fallback uses last epoch weights; downstream
   JSON saves best_epoch=epochs and NaN val_auc without any warning.
3. Lines 1426-1431 + 1459-1461 — invalid SMILES with active=1 becomes
   score=0.0, biasing ROC down (guaranteed false-negative).
4. Lines 782-789 — bare except Exception: pass on cache load; corrupt pickle
   silently triggers full re-featurisation with no warning.

Per AGENTS.md §5.2 honest framing — every test FAILS on unfixed code,
PASSES on fixed code.  If you cannot make a test fail first, delete it.
"""
from __future__ import annotations

import textwrap
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
    _featurize_with_cache,
    predict_joint,
)


# ============================================================================
# Finding 4: cache load silent failure (lines 782-789)
# ============================================================================
def test_finding4_cache_load_failure_emits_warning(tmp_path, capsys):
    """Corrupt pickle cache must NOT be silently swallowed.

    Unfixed: ``except Exception: pass`` at line 789 — no stderr message.
    Fixed: must emit a warning mentioning cache/pickle/failure.
    """
    cache_dir = tmp_path / "cache"
    cache_key = "wf_ultracode_f4"
    smiles = ["CCO", "CCC"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.pkl"
    cache_path.write_bytes(b"definitely not a valid pickle at all")

    _featurize_with_cache(smiles, cache_key, cache_dir)
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    # Unfixed: bare `except Exception: pass` → no warning about the failure.
    # The featurisation message at line 799 does mention "cache" (in
    # "(cache: <path>)") but does NOT warn that the cache load FAILED.
    # Fixed: must emit a warning like "WARNING: cache <path> load failed".
    combined_lower = combined.lower()
    warned_failure = (
        "load failed" in combined_lower
        or "corrupt" in combined_lower
        or "invalid pickle" in combined_lower
        or "cache load" in combined_lower
        or "failed to load" in combined_lower
    )
    assert warned_failure, (
        f"Cache load failure was silent — no warning about the failure itself.\n"
        f"Unfixed: bare `except Exception: pass` at line 789.\n"
        f"(The featurisation message mentions 'cache' in '(cache: <path>)',\n"
        f"but does NOT warn that the cache load FAILED.)\n"
        f"Combined output: {combined!r}"
    )


# ============================================================================
# Finding 3: invalid SMILES biased ROC (lines 1426-1431 + 1459-1461)
# ============================================================================
def test_finding3_predict_joint_excludes_invalid_rows():
    """Invalid SMILES (None graphs) must NOT be appended with (label, 0.0, 0.0).

    Unfixed: invalid rows appended at lines 1428-1430 (collate-fail branch)
    and 1468-1470 (per-row invalid branch) — active=1 with score=0.0 biases
    ROC down (guaranteed false-negative).
    Fixed: invalid rows excluded entirely from y_true/y_score/y_pic50.
    """

    class _MockTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def clone(self):
            return self

    class MockModel:
        def eval(self):
            pass

        def __call__(self, atom_f, bond_f, e_src, e_dst, mids_t):
            n = atom_f.shape[0]
            return (
                torch.zeros(n, dtype=torch.float32),
                torch.zeros((n, 2), dtype=torch.float32),
            )

    valid_graph = (
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
    )

    graphs = [valid_graph, None, valid_graph]
    metal_ids = [0, 0, 0]
    pic50 = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    active = np.array([1, 1, 1], dtype=np.int64)

    y_true, y_score, y_pic50 = predict_joint(
        MockModel(),
        graphs,
        metal_ids,
        pic50,
        active,
        batch_size=3,
        device=torch.device("cpu"),
    )

    assert len(y_true) == 2, (
        f"predict_joint returned {len(y_true)} rows for 3 inputs (1 invalid).\n"
        f"Unfixed: invalid rows appended at lines 1428-1430 (collate-fail branch).\n"
        f"Fixed: invalid rows must be excluded — expected 2 valid rows.\n"
        f"y_true={y_true.tolist()}, y_score={y_score.tolist()}"
    )
    assert 0.0 not in y_score, (
        f"y_score contains 0.0 entries — these are from the invalid-row fallback\n"
        f"(score=0.0 appended at lines 1469-1470).  Excluded after fix.\n"
        f"y_score={y_score.tolist()}"
    )


# ============================================================================
# Finding 1: n_metals_pool not recomputed after silent drop (line 1620 + 1624-1632)
# ============================================================================
def test_finding1_n_metals_pool_recomputed_after_drop():
    """When a pool metal fails with RuntimeError, n_metals_pool must reflect the
    actual successful count, not the requested pool size.

    Unfixed: n_metals_pool is set at line 1620 to ``len(metals_in_pool)`` (5)
    and never recomputed; the bare ``except RuntimeError`` at 1627 silently
    substitutes empty arrays but leaves n_metals_pool=5.
    Fixed: after the loop, n_metals_pool is recomputed to count only metals
    with non-empty train splits.
    """
    src_path = Path(
        "molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
    )
    src = src_path.read_text()

    # Extract the pool-construction block by pattern: from the
    # ``if single_task:`` guard through the ``if len(train_pool_idx) < 50:``
    # check (which is after the for-loop AND any post-loop recomputation
    # of n_metals_pool).  Pattern matching keeps it robust to line shifts
    # when the fix adds/removes lines.
    start = src.index("    if single_task:")
    end = src.index("    if len(train_pool_idx) < 50:", start)
    block = textwrap.dedent(src[start:end])

    def mock_build(split_df, ds, m):
        if m == "Pt":
            raise RuntimeError("simulated Pt failure")
        return (
            np.array([0, 1, 2], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([4], dtype=np.int64),
        )

    ns = {
        "FIVE_METALS": ("Pt", "Ru", "Ir", "Au", "Pd"),
        "single_task": False,
        "metals_pool": ("Pt", "Ru", "Ir", "Au", "Pd"),
        "build_metal_split_indices": mock_build,
        "split_df": None,
        "ds": None,
        "np": np,
        "sys": sys,
    }

    exec(  # noqa: S102 — intentional exec for line-range regression test
        block, ns
    )

    assert ns["n_metals_pool"] == 4, (
        f"n_metals_pool={ns['n_metals_pool']} but Pt was silently dropped.\n"
        f"Unfixed: n_metals_pool set at line 1620 to len(metals_in_pool)=5\n"
        f"and never recomputed after the bare-except drop at 1624-1632.\n"
        f"Fixed: should be 4 (Pt excluded from successful pool)."
    )


# ============================================================================
# Finding 2: best_state fallback silently saves best_epoch=epochs + NaN val_auc
# ============================================================================
def test_finding2_best_state_fallback_marks_no_best():
    """When best_state is None (no NaN val_auc ever won), the fallback must NOT
    silently set best_epoch=epochs.

    Unfixed: lines 2092-2097 set ``best_epoch = epochs`` (e.g., 50) when
    best_state is None.  Downstream JSON at line 2154 saves this + NaN val_auc
    with no warning — honest-negative bug.
    Fixed: the fallback marks ``best_epoch`` with a sentinel (e.g., -1) so the
    downstream JSON surfaces the no-best state honestly.
    """
    src_path = Path(
        "molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
    )
    src = src_path.read_text()

    # Extract the fallback block by pattern: from ``if best_state is None:``
    # to the end of the fallback (just before ``train_seconds =``).
    # Robust to line shifts when the fix adds/removes lines.
    start = src.index("    if best_state is None:")
    end = src.index("    train_seconds = time.time()", start)
    fallback_block = textwrap.dedent(src[start:end])

    class _MockTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def clone(self):
            return self

    class MockModel:
        def state_dict(self):
            return {"weight": _MockTensor()}

    ns = {
        "best_state": None,
        "model": MockModel(),
        "best_epoch": 0,
        "epochs": 50,
        "metal": "Ru",
        "seed": 0,
        "sys": sys,
    }

    exec(  # noqa: S102 — intentional exec for line-range regression test
        fallback_block, ns
    )

    assert ns["best_epoch"] != ns["epochs"], (
        f"best_epoch={ns['best_epoch']} equals epochs={ns['epochs']}.\n"
        f"Unfixed: silent fallback at lines 2092-2097 sets best_epoch=epochs.\n"
        f"Fixed: must use a sentinel (e.g., -1) to mark 'no best AUC found'."
    )