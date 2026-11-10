"""Pytest: Wf-A126 — the ablation LEVER IDENTITY + TRAINER-BUILD FINGERPRINT.

Wf-A124's critic found that the root causes of two catalog findings live in
the trainer, not in the docs:

* **§6.220** — "the artifacts do not record which lever was disabled: 3 of
  the 4 G6 cohorts are indistinguishable by their own config".  The G6
  launcher maps each arm to a trainer flag, but the trainer wrote only
  ``wasserstein_weight``, so three of four arms produced byte-identical
  summaries and the cohort DIRECTORY NAME was the sole record of the lever.

* **§6.219** — "the G5 baseline carries no trainer fingerprint".  G5 ran
  2026-10-02 (before commit ``8d139d23`` added the fingerprint), G6 ran
  2026-10-08/09 (with it), and commit ``af84a545`` (2026-10-15) removed the
  fingerprint from the trainer again — so a run launched today would once more
  be unversion-matchable against G5, which is exactly the confound that made
  the published per-lever Δ table irreproducible (§6.221).

These tests pin four contracts:

1. ``lever_and_build_fingerprint`` reports the three previously-unrecorded
   levers (``single_task`` / ``kendall`` / ``dann``) from the parsed args.
2. The G6 launcher's four ablation arms produce **pairwise distinct**
   fingerprints — the §6.220 root cause is closed.
3. The fingerprint carries the build fields the current trainer knows.
4. The trainer's ``training_summary.json`` really merges the fingerprint in
   (a source-level lock, so a later edit cannot silently drop it).

All tests are CPU-only and finish in <5 s.  No GPU, no training, no data.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
LAUNCHER = PROJECT_ROOT / "scripts" / "launch_wf124_p3_g6.sh"
TRAINER = SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_dmpnn_multitask_wasserstein as mtw  # noqa: E402


# The four G6 arms and the trainer flag each one maps to
# (scripts/launch_wf124_p3_g6.sh :: CONFIRMED_FLAGS).
G6_ARMS: dict[str, list[str]] = {
    "no_multi_task": ["--single-task"],
    "no_wass": ["--wasserstein-weight", "0.0"],
    "no_kendall": ["--no-kendall"],
    "no_dann": ["--no-dann"],
}

# Which recorded summary key each launcher flag lands on.
FLAG_TO_RECORDED_KEY = {
    "--single-task": "single_task",
    "--wasserstein-weight": "wasserstein_weight",
    "--no-kendall": "kendall",
    "--no-dann": "dann",
}


def _arm_fingerprint(flags: list[str]) -> dict:
    """The full recorded lever signature of one arm.

    ``wasserstein_weight`` is written by the caller (it predates Wf-A126 and
    is already in the summary), so it is added here to model what a reader of
    ``training_summary.json`` would actually see.
    """
    args = mtw.parse_args(flags)
    fp = dict(mtw.lever_and_build_fingerprint(args))
    fp["wasserstein_weight"] = float(args.wasserstein_weight)
    return fp


# ===========================================================================
# 1. the three previously-unrecorded levers
# ===========================================================================
def test_fingerprint_records_the_three_previously_unrecorded_levers() -> None:
    """single_task / kendall / dann all reach the fingerprint."""
    fp = mtw.lever_and_build_fingerprint(mtw.parse_args([]))
    assert fp["single_task"] is False
    assert fp["kendall"] is True
    assert fp["dann"] is True


def test_each_lever_flag_flips_its_own_recorded_key() -> None:
    """Each of the three levers is recorded as toggled, and only its own."""
    fp = mtw.lever_and_build_fingerprint(mtw.parse_args(["--single-task"]))
    assert fp["single_task"] is True and fp["kendall"] is True and fp["dann"] is True

    fp = mtw.lever_and_build_fingerprint(mtw.parse_args(["--no-kendall"]))
    assert fp["kendall"] is False and fp["single_task"] is False and fp["dann"] is True

    fp = mtw.lever_and_build_fingerprint(mtw.parse_args(["--no-dann"]))
    assert fp["dann"] is False and fp["single_task"] is False and fp["kendall"] is True


# ===========================================================================
# 2. the §6.220 root cause: four arms, four distinct signatures
# ===========================================================================
def test_the_four_g6_arms_are_pairwise_distinguishable() -> None:
    """No two G6 arms may produce the same recorded lever signature.

    This is the regression test for catalog §6.220: before Wf-A126 the
    ``no_multi_task`` / ``no_kendall`` / ``no_dann`` arms were
    indistinguishable from their own config.
    """
    sigs = {arm: _arm_fingerprint(flags) for arm, flags in G6_ARMS.items()}
    seen: dict[tuple, str] = {}
    for arm, sig in sigs.items():
        key = tuple(sorted(sig.items()))
        assert key not in seen, (
            f"arms {arm!r} and {seen.get(key)!r} produce an IDENTICAL recorded "
            f"signature — the §6.220 root cause is back"
        )
        seen[key] = arm
    assert len(seen) == len(G6_ARMS)


def test_every_launcher_flag_reaches_a_recorded_summary_key() -> None:
    """Each G6 launcher flag must land on a key the summary actually writes."""
    recorded = set(mtw.lever_and_build_fingerprint(mtw.parse_args([])))
    recorded.add("wasserstein_weight")
    for arm, flags in G6_ARMS.items():
        for flag in flags:
            if flag not in FLAG_TO_RECORDED_KEY:
                continue
            key = FLAG_TO_RECORDED_KEY[flag]
            assert key in recorded, (
                f"arm {arm!r} sets {flag!r} -> dest {key!r}, which the summary "
                f"does NOT record — that lever would be lost"
            )


def test_the_launcher_still_declares_the_four_arms_this_test_knows() -> None:
    """Guard against the launcher gaining an arm this test does not cover."""
    text = LAUNCHER.read_text()
    declared = set(re.findall(r'\["(no_[a-z_]+)"\]=', text))
    assert declared == set(G6_ARMS), (
        f"launcher arms {sorted(declared)} != test arms {sorted(G6_ARMS)}"
    )


# ===========================================================================
# 3. the build fingerprint
# ===========================================================================
def test_the_fingerprint_carries_the_build_fields_the_trainer_knows() -> None:
    """tf32 / torch_compile / w2_gradient_mode are recorded."""
    fp = mtw.lever_and_build_fingerprint(mtw.parse_args([]))
    for key in ("tf32", "torch_compile", "w2_gradient_mode"):
        assert key in fp, f"build field {key!r} is not recorded"
    assert isinstance(fp["tf32"], bool)
    assert isinstance(fp["torch_compile"], bool)
    assert isinstance(fp["w2_gradient_mode"], str)


def test_w2_gradient_mode_is_recorded_from_the_parsed_args() -> None:
    """The build fingerprint reflects the run, not a hard-coded constant."""
    fp = mtw.lever_and_build_fingerprint(mtw.parse_args(["--w2-gradient-mode", "sort"]))
    assert fp["w2_gradient_mode"] == "sort"
    fp = mtw.lever_and_build_fingerprint(mtw.parse_args([]))
    assert fp["w2_gradient_mode"] == mtw.DEFAULT_W2_GRADIENT_MODE


# ===========================================================================
# 4. the summary really merges the fingerprint in
# ===========================================================================
def test_the_training_summary_merges_the_fingerprint() -> None:
    """Source-level lock: the summary must merge the fingerprint.

    A behavioural test for this would need a full training run; the trainer's
    summary dict is built inline in ``main()``, so the merge is asserted at
    the source level (the same pattern as
    ``test_no_kendall_no_dann_flags.py``).
    """
    src = TRAINER.read_text()
    assert src.count("**lever_and_build_fingerprint(args),") == 1, (
        "training_summary.json no longer merges lever_and_build_fingerprint"
    )
    assert src.count("def lever_and_build_fingerprint(") == 1
    # the pre-existing key must survive — the merge is additive
    assert '"wasserstein_weight": args.wasserstein_weight,' in src
