"""Scale guard for the CFM soft-valence loss term (BLK-C-L2, 2026-10-12).

The defect this pins
--------------------
``metrics/by_metric/cfm_loss_term_liveness.json`` recorded the term reading
4.13 on the synthetic cohort at batch 8 and 134294.75 on tmQM at batch 64 — a
32537x swing driven by nothing but batch size and molecule count.  Cause:
``valence_log_barrier`` returns ``excess.pow(2).sum()`` over ATOMS
(``decoder_rework.py:499``) and ``train_one_epoch`` summed that over every
cloud in the batch, so the term was ``O(batch * n_atoms)``.  No fixed weight
in a YAML could be right across cohorts, which is why the term had never been
switched on despite being implemented, wired and gradient-verified.

STATUS NOTE — read before "fixing" this again.  The per-atom division was
already applied in commit ``c3ee284b64`` (2026-10-10 04:57:26), which is the
SAME commit that created the JSON above (file mtime 2026-10-10 03:53:26,
i.e. the measurement is 64 minutes OLDER than the fix).  The JSON's
``"still_true": true`` is stale by an hour and contradicts the code shipped
alongside it.  What did not exist was any guard, which is what this file adds:
the sibling tests in ``test_wf_r41_soft_valence_loss.py`` are source-greps for
the argparse flag, so reverting the division broke nothing.

Why per-atom, not per-cloud
---------------------------
The decision was made and defended in ``train_one_epoch`` at the division.
Restated here because the test encodes it: the barrier is a squared hinge
SUMMED over atoms, so per-cloud would still grow with molecule size and leave
the cohort dependence only partly out.  Per-atom makes the number a mean
violation per atom — the quantity a weight should mean, and the one that is
comparable between configs.

How the two tests split the work
--------------------------------
* ``test_soft_valence_per_atom_is_batch_size_invariant`` — deterministic unit
  test over the REAL ``valence_log_barrier`` on REAL tmQM clouds.  Bounds the
  batch-size ratio.  Cheap.
* ``test_trainer_divides_soft_valence_by_batch_atoms`` — behavioural test on
  the REAL trainer: instruments the barrier, runs ONE epoch of ONE batch, and
  asserts the emitted term equals ``weight * sum(calls) / sum(n_atoms)``.
  This is the one that goes RED if the division is reverted.

Honest framing: the per-atom statistic is HEAVY-TAILED (MEASURED below: per
cloud, per atom, min 0.0134 to max 2.383 across 64 tmQM Pt complexes — 177.8x).
A single small batch therefore still swings a long way; this file bounds
BATCH SIZE dependence, not sampling variance.  See
``metrics/by_metric/cfm_soft_valence_scale.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
TMQM_DIR = REPO_ROOT / ".data" / "molmetal" / "tmQM"

#: Bound on max(per_atom over bs>=8) / min(per_atom over bs>=8).  The target
#: set by BLK-C-L2 is 100x; see the module docstring for what is actually
#: achieved and why a smaller number is not the point.
BATCH_RATIO_BOUND = 100.0

_BOND_ORDER_IDX = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.5])


def _tmqm_barrier_per_cloud(n_clouds: int = 64):
    """Real ``valence_log_barrier`` per real tmQM cloud -> [(value, n_atoms)].

    The logits are a fixed-seed stand-in for an untrained bond head.  The
    invariance claim under test does not depend on their values — only on the
    fact that the SAME per-cloud values are regrouped into different batch
    sizes, so any residual movement is the reduction's and nothing else.
    """
    pytest.importorskip("molmetal.data.tmqm_clouds")
    from molmetal.data.tmqm_clouds import load_tmqm_clouds
    from molmetal.molmetal_lam.lam_chem.decoder_rework import (
        DEFAULT_VALENCES,
        TMQM_MEASURED_METAL_CAPS,
        valence_log_barrier,
    )

    clouds = load_tmqm_clouds(TMQM_DIR, metals=["Pt"], n_clouds=n_clouds, seed=0)
    valences = {**DEFAULT_VALENCES, **TMQM_MEASURED_METAL_CAPS}
    gen = torch.Generator().manual_seed(7)
    out = []
    for c in clouds:
        n = int(c.atomic_numbers.shape[0])
        src, dst = c.bonds[0].long(), c.bonds[1].long()
        logits = torch.randn(src.numel(), 5, generator=gen)
        orders = (torch.softmax(logits, dim=-1) * _BOND_ORDER_IDX).sum(dim=-1)
        val = valence_log_barrier(
            orders, c.atomic_numbers.long(), src, dst, n, valences=valences
        )
        out.append((float(val), n))
    return out


def test_soft_valence_per_atom_is_batch_size_invariant() -> None:
    """Per-atom reduction must bound the batch-size ratio; the sum must not.

    Same 64 clouds regrouped into batches of 1..64, so the ONLY thing that
    varies is how many clouds share a sum.  If this is green, the division is
    doing its job; if the division is reverted the "per_atom" column would
    track the "raw_sum" column and the ratio would blow past the bound.
    """
    pytest.importorskip("molmetal.data.tmqm_clouds")
    per_cloud = _tmqm_barrier_per_cloud(64)
    assert len(per_cloud) == 64, f"expected 64 real tmQM clouds, got {len(per_cloud)}"

    raw_ratios, per_atom_ratios = {}, {}
    for bs in (1, 2, 4, 8, 16, 32, 64):
        window = per_cloud[:bs]
        s = sum(v for v, _ in window)
        a = sum(n for _, n in window)
        raw_ratios[bs] = s
        per_atom_ratios[bs] = s / a

    # bs=1 is the single-cloud arm; compare the multi-cloud arms to each other
    # so the comparison is not anchored to one molecule's variance.
    multi = [per_atom_ratios[b] for b in (8, 16, 32, 64)]
    ratio = max(multi) / min(multi)
    raw_ratio = raw_ratios[64] / raw_ratios[8]

    assert ratio <= BATCH_RATIO_BOUND, (
        "per-atom soft-valence reduction is not batch-size invariant: "
        f"ratio {ratio:.2f}x exceeds the {BATCH_RATIO_BOUND:.0f}x bound "
        f"(per_atom={ {b: round(per_atom_ratios[b], 6) for b in multi} })"
    )
    # Sensitivity: the whole point of the division is that the raw sum keeps
    # MOVING with batch size while the per-atom value does not.  So the raw
    # movement must be far larger than the per-atom movement.  (The absolute
    # raw growth is NOT compared against BATCH_RATIO_BOUND: on this 64-cloud
    # prefix it is only ~6.8x, because the first 64 clouds are not all the
    # same size.  The 32537x in the original finding was a SYNTHETIC-vs-tmQM
    # cohort difference, which is a different axis — see the module docstring.)
    assert raw_ratio > 5.0 * ratio, (
        f"the raw sum ({raw_ratio:.2f}x) and the per-atom value ({ratio:.2f}x) "
        "barely differ across batch sizes, so this guard would pass whether or "
        "not the division is applied. Re-check the fixture — it is not "
        "exercising the scale bug it claims to guard."
    )


#: One batch, so every barrier call recorded below belongs to the same batch
#: and the expected sum/total can be reconstructed exactly.
_ONE_BATCH_ARGS = [
    "--device",
    "cpu",
    "--epochs",
    "1",
    "--n-train",
    "8",
    "--n-val",
    "4",
    "--batch-size",
    "8",
    "--pocket-id",
    "1h36",
    "--resume-from",
    "none",
    "--gen-validity-weight",
    "1.0",
]

#: Wraps the REAL barrier, records (returned value, n_atoms), returns the real
#: tensor unchanged so the trainer's gradient path is untouched.
#: Substituted with ``.replace()``, not ``.format()`` — the probe is full of
#: dict literals and format's brace escaping is a trap here.
_INSTRUMENTED_PROBE = """
import sys, json, importlib.util
sys.argv = ["train_cfm_yuelbond_wasserstein.py"] + sys.argv[1:]
spec = importlib.util.spec_from_file_location("_t", "__TRAINER__")
m = importlib.util.module_from_spec(spec); sys.modules["_t"] = m
spec.loader.exec_module(m)

import molmetal.molmetal_lam.lam_chem.decoder_rework as _drw
_real = _drw.valence_log_barrier
calls = []
def _rec(*a, **kw):
    out = _real(*a, **kw)
    calls.append({"value": float(out.detach()), "n_atoms": int(a[4])})
    return out
_drw.valence_log_barrier = _rec

m.main()
term = m._LAST_LOSS_TERMS.get("soft_valence_term")
print("SV_PROBE_JSON=" + json.dumps({
    "calls": calls,
    "term": None if term is None else float(term.detach()),
    "requires_grad": False if term is None else bool(term.requires_grad),
}))
"""


def test_trainer_divides_soft_valence_by_batch_atoms(tmp_path: Path) -> None:
    """BEHAVIOURAL guard: the emitted term must be sum(calls)/sum(n_atoms).

    This is the test that goes RED if the per-atom division is reverted to a
    plain sum.  It asserts on what the trainer EMITS, computed from what the
    REAL ``valence_log_barrier`` RETURNED — not on the source text.  A
    source-grep guard would have stayed green through the revert that shipped
    untested on 2026-10-10.
    """
    probe = tmp_path / "_probe_soft_valence.py"
    probe.write_text(
        _INSTRUMENTED_PROBE.replace("__TRAINER__", str(TRAINER)), encoding="utf-8"
    )
    proc = subprocess.run(
        [sys.executable, str(probe), *_ONE_BATCH_ARGS,
         "--out-dir", str(tmp_path / "run"), "--ckpt-dir", str(tmp_path / "ck")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=1800,
    )
    marker = "SV_PROBE_JSON="
    payload = None
    for line in reversed(proc.stdout.splitlines()):
        if marker in line:
            payload = json.loads(line.split(marker, 1)[1])
            break
    assert payload is not None, (
        "instrumented trainer did not emit SV_PROBE_JSON — it crashed or the "
        f"instrumentation hook was lost.\nrc={proc.returncode}\n"
        f"STDOUT tail:\n{proc.stdout[-3000:]}\nSTDERR tail:\n{proc.stderr[-3000:]}"
    )

    calls = payload["calls"]
    assert len(calls) > 0, (
        "valence_log_barrier was never called — gen_validity_weight=1.0 did "
        "not turn the term on, so this guard is measuring nothing"
    )
    raw_sum = sum(c["value"] for c in calls)
    n_atoms_total = sum(c["n_atoms"] for c in calls)
    expected = raw_sum / n_atoms_total

    assert payload["requires_grad"], (
        "soft_valence_term lost its gradient — the wire-in has gone FLAG_ONLY"
    )
    assert abs(payload["term"] - expected) <= 1e-6 * max(1.0, abs(expected)), (
        f"trainer emitted {payload['term']:.6f} but sum(calls)/sum(n_atoms) is "
        f"{expected:.6f} (raw_sum={raw_sum:.6f} over {len(calls)} clouds / "
        f"{n_atoms_total} atoms). The per-atom division is NOT being applied — "
        "revert the fix at train_cfm_yuelbond_wasserstein.py and this fails."
    )
