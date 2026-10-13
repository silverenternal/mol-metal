# SPDX-License-Identifier: Apache-2.0
"""McNemar on the pad_groups A/B -- the statistics must be checkable by hand.

WHY THIS EXISTS
---------------
The first pad_groups comparison crashed (CUDA OOM, 04:57:04) and the surviving
grouped arm reported aggregates only. Aggregates cannot answer the question:
the A/B is PAIRED (same clouds, same noise seed, only the batching differs),
so the answer lives in the discordant cells, and ``n_out 37 vs 38`` could be
one cloud flipping or thirty flipping in opposite directions.

The replacement therefore records per-cloud vectors and applies McNemar's
exact test. A statistical test nobody can check by hand is a test that will
be misread, so the arithmetic is pinned against values worked out
independently here.

THE p-VALUE, DERIVED
--------------------
Exact two-sided McNemar: under H0 each discordant pair is a fair coin, so

    p = 2 * P(X <= min(b, c)),  X ~ Binom(b + c, 1/2)

  b=5,  c=3 -> n=8, min=3: 2 * (C(8,0)+C(8,1)+C(8,2)+C(8,3)) / 256
                     = 2 * (1 + 8 + 28 + 56) / 256 = 2 * 93/256 = 0.7265625
  b=0,  c=0 -> no discordant pairs; p is undefined, reported as 1.0
  b=10, c=0 -> n=10, min=0: 2 * 1/1024 = 0.001953125

RED CHECK: replace the binomial sum with a normal approximation and
``test_known_values`` fails; make the length check a warning and
``test_length_mismatch_is_fatal`` fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "cfm_pad_groups_mcnemar.py"


def _mod():
    if "_pad_groups_mcnemar_test" in sys.modules:
        return sys.modules["_pad_groups_mcnemar_test"]
    spec = importlib.util.spec_from_file_location(
        "_pad_groups_mcnemar_test", str(SCRIPT)
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["_pad_groups_mcnemar_test"] = m
    spec.loader.exec_module(m)
    return m


def test_repo_paths_resolve():
    """REBUG 2026-10-12: this file lives in <repo>/scripts/, so
    ``parent.parent`` is the repo root.

    It was written as ``parent.parent.parent`` -- the value correct for
    modules inside ``<repo>/molmetal/scripts/`` -- which resolved to
    ``/home/hugo/codes``. Every path built from it raised FileNotFoundError, so
    BOTH arms died in under a second and the aggregate then failed on a
    missing file. The launcher reported "SWEEP DONE" having never loaded a
    model: the same silent-success shape as the failures this whole file
    exists to prevent.
    """
    assert _mod().REPO.name == "mol-metal", (
        f"REPO resolved to {_mod().REPO}; this script lives in <repo>/scripts/ "
        "so parent.parent is the root and parent.parent.parent overshoots"
    )
    for rel in (
        "molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
        "configs/cfm_tmqm_all6.yaml",
        "molmetal/reports/cfm_fullcorpus_2026_10_10/checkpoints/"
        "cfm_yuelbond_wasserstein_seed0.pt",
    ):
        p = _mod().REPO / rel
        assert p.exists(), f"path does not resolve: {p}"


def test_known_values():
    p = _mod().mcnemar_exact_p
    assert abs(p(5, 3) - 0.7265625) < 1e-12, f"p(5,3) = {p(5,3)}"
    assert p(0, 0) == 1.0, "no discordant pairs means no evidence"
    assert abs(p(10, 0) - 0.001953125) < 1e-12, f"p(10,0) = {p(10,0)}"
    # Symmetric: swapping the arms must not move the p-value.
    assert abs(p(3, 5) - p(5, 3)) < 1e-15


def test_identical_arms_have_zero_discordant():
    v = [True, False, True, True, False] * 8
    t = _mod().paired_table(v, list(v))
    assert t["contingency"]["grouped_only"] == 0
    assert t["contingency"]["padded_only"] == 0
    assert t["mcnemar"]["n_discordant"] == 0
    assert t["mcnemar"]["verdict"].startswith("SUB-FLOOR")


def test_one_flip_is_not_significant():
    v = [True] * 40
    w = list(v)
    w[7] = False
    t = _mod().paired_table(v, w)
    assert t["contingency"]["grouped_only"] == 1
    assert t["mcnemar"]["exact_two_sided_p"] == 1.0, (
        "a single discordant pair cannot clear the test"
    )


def test_symmetric_split_carries_no_evidence():
    """b == c is the NULL configuration, not a strong signal.

    An earlier version of this test asserted p < 1e-15 for a 32-vs-32 split.
    That is arithmetically impossible: with b = c = 32 the smaller count is 32,
    exactly half of n = 64, so ``P(X <= 32)`` is already > 0.5 and the
    two-sided p caps at 1.0. A perfectly symmetric disagreement is what H0
    predicts, so reporting it as a significant difference would be the test
    backwards. This pins the correct reading.
    """
    v = [True] * 32 + [False] * 32
    w = [False] * 32 + [True] * 32
    t = _mod().paired_table(v, w)
    assert t["contingency"]["grouped_only"] == 32
    assert t["contingency"]["padded_only"] == 32
    assert t["mcnemar"]["exact_two_sided_p"] == 1.0, (
        f"a symmetric split must read as NO EVIDENCE; got "
        f"{t['mcnemar']['exact_two_sided_p']}"
    )
    assert t["mcnemar"]["verdict"].startswith("SUB-FLOOR")


def test_asymmetric_difference_is_detected():
    """The real signal shape: many clouds flip ONE way."""
    # Disjoint halves, so every cloud is discordant and one direction only.
    # (Overlapping cohorts would put clouds in `both_pass` and make
    # `padded_only` non-zero -- the fixture, not the test, would be wrong.)
    v = [True] * 64 + [False] * 36
    w = [False] * 64 + [False] * 36
    t = _mod().paired_table(v, w)
    assert t["contingency"]["grouped_only"] == 64
    assert t["contingency"]["padded_only"] == 0
    # 2 * (1/2)^64 = 1.08e-19. Asserted against the exact value, not a
    # hand-picked threshold: 2**-63 is the last normal double, so anything
    # past n=64 denormalises and a loose "< 1e-15" would quietly stop
    # testing the arithmetic at all.
    assert t["mcnemar"]["exact_two_sided_p"] == 2.0 ** -63, (
        f"64 one-directional flips must give exactly 2**-63 = 2^-64 * 2; got "
        f"{t['mcnemar']['exact_two_sided_p']}"
    )
    assert t["mcnemar"]["verdict"] == "CLEAR"


def test_length_mismatch_is_fatal():
    """A short vector is the alignment bug; McNemar must refuse to run."""
    try:
        _mod().paired_table([True] * 10, [True] * 9)
    except ValueError as exc:
        assert "unpaired" in str(exc).lower()
        assert "per_cloud" in str(exc)
    else:
        raise AssertionError(
            "a length mismatch must raise, not silently truncate to the "
            "shorter vector -- truncating would pair cloud i with cloud i "
            "for the first 9 and drop the rest, which is a different test"
        )


def test_wilson_interval_brackets_the_estimate():
    t = _mod().paired_table([True] * 30 + [False] * 10, [True] * 30 + [False] * 10)
    g = t["wilson_independently"]["grouped"]
    assert g["lo"] < g["mean"] < g["hi"], f"Wilson must bracket the mean: {g}"
    assert g["n"] == 40


def test_rates_reported_for_both_arms():
    t = _mod().paired_table([True] * 25 + [False] * 15, [True] * 20 + [False] * 20)
    assert t["wilson_independently"]["grouped"]["mean"] == 0.625
    assert t["wilson_independently"]["padded"]["mean"] == 0.5