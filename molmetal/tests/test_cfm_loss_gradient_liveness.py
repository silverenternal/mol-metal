"""Gradient-liveness guard for the CFM trainer's loss cascade.

Why this test exists
--------------------
On 2026-10-04 an audit of ``train_cfm_yuelbond_wasserstein.py`` found that 11
of the 13 terms inside the ``loss = (...)`` sum could not train anything.  The
audit was prompted by a narrower question ("is the yaml config real?") and the
answer kept getting worse the closer we looked.

Three failure shapes were found, and **none of them is visible** to either of
the project's existing checks:

===========================================  ==============  ==================
shape                                        example term    caught by
===========================================  ==============  ==================
``x * torch.tensor(0.0)``                    shortcut/cons.  neither
every operand ``.detach()``'d                so3, edm,       neither
                                             diversity
``torch.tensor(float(v))`` cuts the path     sa_loss         neither
exception swallowed by ``except: continue``  pll_term        neither
===========================================  ==============  ==================

* A "does anything consume this flag?" grep reports the flag as consumed —
  the value really is computed and really does reach the sum.
* The AGENTS.md section 5.7 smoke checks only ``decode_ratio > 0`` and
  ``loss[0] > loss[49]``.  Both hold perfectly while 11 terms sit at zero.
* Reading the code is not enough either: the parent audit read
  ``diversity_term`` as live (it multiplies real MMD/repulsion tensors) and
  only the measurement below showed ``requires_grad=False``.

So this test asserts the property that actually matters, **on a real forward
pass**: every tensor that enters the loss either carries a gradient, or is on
an explicit allowlist of intentional, documented diagnostics.

How to read a failure
---------------------
* ``UNDECLARED`` — a term reached the loss sum that this file does not know
  about.  Add it to ``EXPECTED_LIVE`` or to ``ALLOWED_DEAD`` with a reason.
  This is the guard against silently adding term #14.
* ``REGRESSED`` — a term that used to carry a gradient no longer does.  This
  is the regression the test is named for.
* ``REMOVED`` — a term is listed here but no longer reaches the sum.  Move it
  out of this file's tables so the tables keep matching reality.

The allowlist is meant to SHRINK.  Each entry names the work item that will
retire it; when that work lands, move the term to ``EXPECTED_LIVE``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"

#: Every term currently inside the ``loss = (...)`` sum, as emitted by the
#: trainer's ``_LAST_LOSS_TERMS`` side-channel.
ALL_TERMS = frozenset(
    {
        "bond_ce",
        "wasserstein_weight*w2_val",
        "0.05*cb_loss",
        "diversity_term",
        "shortcut_distill_loss",
        "consistency_term",
        "so3_term",
        "edm_term",
        "fm_term",
    }
)

#: Terms that MUST carry a gradient.  A term is moved here once it really
#: trains something — verified by the measurement below, never by inspection.
EXPECTED_LIVE = frozenset(
    {
        "bond_ce",
        "wasserstein_weight*w2_val",
        # Fixed 2026-10-04: the prediction operand was ``all_logits.detach()``,
        # so the term was a constant.  ``closure_bounded: true`` is set in
        # cfm_full_retrain.yaml + cfm_real_decode_lift_p1.3.yaml, so it was
        # live-but-inert in 2 of the 3 CFM configs.
        "0.05*cb_loss",
        # NEW 2026-10-04: the flow-matching objective.  Unlike every other
        # entry in this test, this one was never a placeholder -- it did not
        # exist before today.  It is the (x0, x1, x_t, t) path that the
        # shortcut/consistency/EDM/SO(3)-FM regularisers need in order to be
        # wireable at all.  Its inputs are resampled every batch (x1 = batch
        # coords, x0 ~ N(0,I), t ~ uniform or log-normal), so it cannot
        # collapse into a constant the way the fixed-reference regression it
        # sits beside can.
        "fm_term",
        # Wired 2026-10-04 onto that path.  Each of these was previously
        # unusable for want of an (x0, x1, x_t, t) input, not for want of care:
        # shortcut/consistency were `weight * torch.tensor(0.0)`; so3/edm
        # computed on fully-detached tensors (so3 measured nonzero at 0.007
        # with zero gradient).  They now consume the per-batch x_t/t/target_dx.
        "shortcut_distill_loss",
        "consistency_term",
        "so3_term",
        "edm_term",
        # Fixed 2026-10-04: `mmd_loss(vel.detach(), ...)` and
        # `repulsion_loss(vel.detach(), ...)` detached the MODEL's own velocity
        # output, so the regulariser was a constant -- measured
        # requires_grad=False at value 0.17 even with mmd_weight=1.0.  Only the
        # target side stays detached now.
        "diversity_term",
    }
)

#: Terms permitted to be gradient-dead.
#:
#: EMPTY as of 2026-10-04, and that is the point.  It reached 11 entries when
#: this test was written; every one has since been either fixed (cb_loss,
#: diversity_term) or removed from the loss sum for being structurally
#: incapable of carrying a gradient (sa_penalty_weight*sa_loss,
#: per_cohort_term, pll_term, chemprop_v3_cache_term, log_normal_t_term).
#:
#: Keep it that way: an entry here means a term is inside the objective while
#: training nothing, and each one has to earn its place with a specific
#: reason.  Prefer fixing the term or taking it out of the sum.
ALLOWED_DEAD: dict = {}

#: The trainer invocation used to sample a real loss.  Every feature flag is
#: ON so the ACTIVE branch of each term runs — with flags off, a dead term and
#: a merely-disabled one are indistinguishable, which is how this survived.
_SMOKE_ARGS = [
    "--device",
    "cpu",
    "--epochs",
    "1",
    "--n-train",
    "32",
    "--n-val",
    "16",
    "--batch-size",
    "8",
    "--pocket-id",
    "1h36",
    "--mmd-weight",
    "1.0",
    "--repulsion-weight",
    "1.0",
    "--sa-penalty",
    "1.0",
    "--shortcut-distill",
    "--consistency-weight",
    "1.0",
    "--per-cohort-features",
    "--cfm-multiph-pll",
    "--cfm-multiph-pll-weight",
    "1.0",
    "--so3-average",
    "--use-log-normal-t",
    "--edm-preconditioning",
    "--chemprop-v3-cache",
    "--use-fm-objective",
    # 2026-10-04 — hermetic.  Since --resume-from became real, the default
    # 'auto' picks up ckpt_dir/cfm_yuelbond_wasserstein_seed0.pt if one is
    # lying around, and the probe then measures the resumed state instead of
    # a clean 1-epoch run (and, before the empty-per_epoch guard, crashed the
    # report).  The probe asserts which loss terms carry gradients in a fresh
    # step, so it must not inherit a previous run's weights.
    "--resume-from",
    "none",
]

_PROBE = """
import sys, json, importlib.util
sys.argv = ["train_cfm_yuelbond_wasserstein.py"] + sys.argv[1:]
spec = importlib.util.spec_from_file_location("_t", {trainer!r})
m = importlib.util.module_from_spec(spec); sys.modules["_t"] = m
spec.loader.exec_module(m)
m.main()
out = {{k: {{"requires_grad": bool(v.requires_grad),
            "value": float(v.detach())}}
       for k, v in m._LAST_LOSS_TERMS.items()}}
print("LOSS_TERMS_JSON=" + json.dumps(out))
"""


#: Module-level cache so the two tests below share ONE trainer run.  Spawning
#: torch in a fresh subprocess and running an all-flags-on epoch costs ~50 s;
#: paying it twice for the same measurement would be pure waste.
_MEASURED_CACHE: dict = {}


def _measure_terms(tmp_path: Path) -> dict:
    """Run one real epoch and return ``{term: {requires_grad, value}}``.

    Cached across calls — see ``_MEASURED_CACHE``.
    """
    if _MEASURED_CACHE:
        return _MEASURED_CACHE
    return _measure_terms_uncached(tmp_path)


def _measure_terms_uncached(tmp_path: Path) -> dict:
    """Run one real epoch and return ``{term: {requires_grad, value}}``."""
    probe = tmp_path / "_probe_loss_terms.py"
    probe.write_text(_PROBE.format(trainer=str(TRAINER)), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(probe),
            *_SMOKE_ARGS,
            "--out-dir",
            str(tmp_path / "run"),
            # 2026-10-04 — the OTHER half of the hermeticity this file already
            # started on.  `--resume-from none` fixed the READ side; without
            # `--ckpt-dir` the WRITE side still went to the repo-global
            # `molmetal/checkpoints` default, and this probe wrote a
            # 1-epoch/32-sample checkpoint over the shipped
            # `cfm_yuelbond_wasserstein.pt` / `..._seed0.pt` on every run.
            # MEASURED, both files on disk carried
            # `out_dir=/tmp/pytest-of-hugo/...` in their saved args.  The
            # trainer now derives its default from --out-dir, so this is belt
            # AND braces — but a test that executes real training must never
            # depend on another component's default for its isolation.
            "--ckpt-dir",
            str(tmp_path / "ck"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=900,
    )
    marker = "LOSS_TERMS_JSON="
    for line in reversed(proc.stdout.splitlines()):
        if marker in line:
            _MEASURED_CACHE.update(json.loads(line.split(marker, 1)[1]))
            return _MEASURED_CACHE
    raise AssertionError(
        "trainer did not emit LOSS_TERMS_JSON — it crashed or the "
        "_LAST_LOSS_TERMS side-channel was removed.\n"
        f"rc={proc.returncode}\nSTDOUT tail:\n{proc.stdout[-3000:]}\n"
        f"STDERR tail:\n{proc.stderr[-3000:]}"
    )


def test_no_undeclared_loss_terms(tmp_path: Path) -> None:
    """The loss sum must match this file's tables exactly.

    Guards against silently adding term #14: a new term that this file has
    never heard of fails here rather than shipping as another placeholder.
    """
    measured = set(_measure_terms(tmp_path))
    undeclared = measured - ALL_TERMS
    removed = ALL_TERMS - measured
    assert not undeclared, (
        "UNDECLARED loss term(s) reached the loss sum: "
        f"{sorted(undeclared)}. Add each to EXPECTED_LIVE (if it trains "
        "something) or to ALLOWED_DEAD with a specific reason."
    )
    assert not removed, (
        f"REMOVED loss term(s) no longer reach the sum: {sorted(removed)}. "
        "Delete them from ALL_TERMS/EXPECTED_LIVE/ALLOWED_DEAD so these "
        "tables keep describing reality."
    )


def test_loss_terms_carry_gradients(tmp_path: Path) -> None:
    """Every term in the sum either has a gradient or is a declared exception."""
    measured = _measure_terms(tmp_path)
    assert measured, "no loss terms measured — the side-channel is empty"

    regressed = sorted(
        t for t in EXPECTED_LIVE if not measured.get(t, {}).get("requires_grad")
    )
    assert not regressed, (
        "REGRESSED — these terms are expected to train something but their "
        f"tensor does not require grad: {regressed}. Measured: "
        f"{ {t: measured.get(t) for t in regressed} }. A term that reaches the "
        "loss sum while carrying no gradient is a constant: it inflates the "
        "reported train_loss and trains nothing."
    )

    undeclared_dead = sorted(
        t
        for t, v in measured.items()
        if not v.get("requires_grad") and t not in ALLOWED_DEAD
    )
    assert not undeclared_dead, (
        f"UNDECLARED dead term(s): {undeclared_dead}. Each is inside the loss "
        "sum but cannot train anything. Either fix it and list it in "
        "EXPECTED_LIVE, or declare it in ALLOWED_DEAD with a specific reason "
        "and the work item that retires it."
    )
