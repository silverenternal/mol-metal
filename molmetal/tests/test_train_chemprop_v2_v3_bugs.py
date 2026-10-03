"""Tests for Wf-244 Fix 1 + Fix 2 in train_chemprop_v2_v3.py.

Wf-244-B driver bugs:

  1. **Ru seed=0 w2=0 across all epochs** — root cause was an
     ``indices`` aliasing bug: ``np.arange(n_train)`` was used to index
     into ``metals_all[batch_idx]`` instead of ``train_idx`` itself.
     When the corpus is laid out ``[Ru, Ir, Os]`` and Ru is held out,
     ``train_idx`` points to positions 19135..24798 in ``metals_all``,
     but ``np.arange(n_train)`` = ``[0, 1, ..., 5663]`` always pointed
     to the FIRST 5664 rows of ``metals_all`` (all Ru).  Every batch
     was Ru-only → ``len(w2_per_metal) == 1`` → ``w2_loss = 0``.

  2. **Held-out metal prediction collapse** — when the held-out metal
     has no rows in ``train_idx`` (LOMO), the multi-task BCE trains
     that head on label=0 only and the head collapses to a constant
     output, yielding AUC=0.5.  Detected by a guard in the eval loop
     that logs when ``np.ptp(scores) == 0``.

These tests are CPU-only and use synthetic data; they do NOT touch
the chemprop encoder (which would require the heavy chemprop_v2 import
chain).  They validate the *indexing* and *guard* logic in isolation.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# G1: Wf-244 Fix 1 — indices must equal train_idx (global positions),
#      NOT np.arange(n_train) (local positions).
# ---------------------------------------------------------------------------
def test_wf244_fix1_indices_match_train_idx_for_held_out_metal() -> None:
    """Wf-244 Fix 1.

    Simulate the Ru LOMO scenario: corpus is laid out
    ``[Ru (19135 rows), Ir (4546 rows), Os (1118 rows)]`` and Ru is the
    held-out metal.  ``train_idx`` should point to positions
    19135..24798.  Pre-fix: ``indices = np.arange(n_train)`` would
    point to positions 0..5663 (all Ru).  Post-fix: ``indices`` MUST
    equal ``train_idx.copy()`` so that ``metals_all[batch_idx]``
    yields Ir+Os rows, not Ru rows.

    This is the unit-test version of the bug; the integration test
    ``test_wf244_fix1_w2_fires_for_held_out_metal`` checks end-to-end.
    """
    # Synthetic corpus laid out [Ru, Ir, Os].
    metals_all = np.array(
        ["Ru"] * 19135 + ["Ir"] * 4546 + ["Os"] * 1118,
        dtype=object,
    )
    n_total = len(metals_all)
    target_metal = "Ru"

    train_mask = metals_all != target_metal
    train_idx = np.where(train_mask)[0]
    n_train = len(train_idx)
    assert n_train == 4546 + 1118 == 5664

    # ---- POST-FIX (expected) ----
    indices = train_idx.copy()
    np.random.seed(0)
    np.random.shuffle(indices)
    batch_size = 64
    metal_counts = {"Ru": 0, "Ir": 0, "Os": 0}
    for start in range(0, n_train, batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_metals = metals_all[batch_idx]
        for m in metal_counts:
            metal_counts[m] += int((batch_metals == m).sum())
    # POST-FIX batches should be mostly Ir+Os; Ru count = 0.
    assert (
        metal_counts["Ru"] == 0
    ), f"post-fix batch composition leaked Ru rows: {metal_counts}"
    assert (
        metal_counts["Ir"] > 0 and metal_counts["Os"] > 0
    ), f"post-fix batches should have BOTH Ir and Os: {metal_counts}"

    # ---- PRE-FIX (the bug) — same logic, broken indices ----
    indices_buggy = np.arange(n_train)
    np.random.seed(0)
    np.random.shuffle(indices_buggy)
    metal_counts_buggy = {"Ru": 0, "Ir": 0, "Os": 0}
    for start in range(0, n_train, batch_size):
        batch_idx = indices_buggy[start : start + batch_size]
        batch_metals = metals_all[batch_idx]
        for m in metal_counts_buggy:
            metal_counts_buggy[m] += int((batch_metals == m).sum())
    # PRE-FIX bug signature: ALL rows are Ru (no Ir, no Os).
    assert metal_counts_buggy["Ru"] == n_train
    assert metal_counts_buggy["Ir"] == 0 and metal_counts_buggy["Os"] == 0


def test_wf244_fix1_batch_metal_count_ge_2_for_wasserstein() -> None:
    """Wf-244 Fix 1 (companion).

    After the fix, batches for the held-out metal contain >= 2 distinct
    metals → ``wasserstein_transfer_loss`` is computable (it requires
    ``len(per_metal_embeds) >= 2``).  We assert that for every batch
    in a 5-epoch simulation of the Ru LOMO scenario, the number of
    distinct metals in the batch is >= 2.
    """
    metals_all = np.array(
        ["Ru"] * 19135 + ["Ir"] * 4546 + ["Os"] * 1118,
        dtype=object,
    )
    train_idx = np.where(metals_all != "Ru")[0]
    n_train = len(train_idx)
    batch_size = 64
    indices = train_idx.copy()
    n_below_2 = 0
    n_total = 0
    for epoch in range(5):
        np.random.seed(epoch)
        np.random.shuffle(indices)
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_metals = metals_all[batch_idx]
            unique = set(batch_metals.tolist())
            n_total += 1
            if len(unique) < 2:
                n_below_2 += 1
    assert n_below_2 == 0, (
        f"after Fix 1, batches with <2 metals should be 0, got "
        f"{n_below_2}/{n_total}"
    )


# ---------------------------------------------------------------------------
# G2: Wf-244 Fix 2 — held-out metal prediction-constant guard.
# ---------------------------------------------------------------------------
def test_wf244_fix2_pred_constant_guard_detects_collapse() -> None:
    """Wf-244 Fix 2.

    When the held-out metal's predictions are all the same value
    (e.g. all 0.0 or all 0.5), ``np.ptp(scores) == 0`` and the guard
    should fire.  This test verifies the detector (NOT the driver
    itself — we don't run the full chemprop pipeline).
    """
    # Simulated collapsed predictions: all 0.5 (random).
    collapsed = np.full(100, 0.5, dtype=np.float32)
    assert float(np.ptp(collapsed)) == 0.0

    # Simulated healthy predictions.
    healthy = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    assert float(np.ptp(healthy)) > 0.0

    # Edge case: 1 score only (degenerate).
    single = np.array([0.42], dtype=np.float32)
    # Guard logic must NOT fire on len(scores) == 0 (would IndexError
    # accessing scores[0]).  Driver guards this with `if scores and ...`.
    assert len(single) > 0
    assert float(np.ptp(single)) == 0.0  # would fire if `scores` non-empty


# ---------------------------------------------------------------------------
# G3: Backward compat — the script still has the Fix 1 marker in the
#     docstring/code so future readers can trace it.
# ---------------------------------------------------------------------------
def test_wf244_fix1_marker_in_source() -> None:
    """Backward compat — the Wf-244 Fix 1 marker must remain in the
    train_chemprop_v2_v3.py source so future readers / reviewers can
    trace the bug fix.
    """
    src = (_SCRIPT_DIR / "train_chemprop_v2_v3.py").read_text()
    assert "Wf-244 Fix 1" in src, (
        "Wf-244 Fix 1 marker missing — fix may have been silently " "reverted."
    )
    # The fix must use ``train_idx.copy()`` (NOT ``np.arange(n_train)``).
    assert "indices = train_idx.copy()" in src, (
        "Fix 1 implementation drifted: indices must equal "
        "train_idx.copy() to index metals_all correctly under LOMO."
    )


def test_wf244_fix2_marker_in_source() -> None:
    """Backward compat — the Wf-244 Fix 2 (held-out guard) marker
    must remain in train_chemprop_v2_v3.py.
    """
    src = (_SCRIPT_DIR / "train_chemprop_v2_v3.py").read_text()
    assert (
        "Wf-244 Fix 2" in src
    ), "Wf-244 Fix 2 marker missing — guard may have been removed."
    # The guard must use np.ptp(scores) == 0 to detect constant preds.
    assert "np.ptp(scores)" in src, (
        "Fix 2 guard drifted: must use np.ptp(scores) to detect the "
        "held-out metal prediction collapse."
    )


# ---------------------------------------------------------------------------
# G4: Smoke — the driver imports without ImportError.
# ---------------------------------------------------------------------------
def test_wf244_driver_imports_cleanly() -> None:
    """The driver must still import cleanly after Wf-244 edits."""
    # If the import side-effects fail (e.g. syntax error), this errors.
    mod = importlib.import_module("train_chemprop_v2_v3")
    assert hasattr(
        mod, "_run_one"
    ), "Wf-244 edits broke driver import: _run_one missing"
    assert hasattr(mod, "main"), "Wf-244 edits broke driver import: main missing"
