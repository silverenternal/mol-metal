# SPDX-License-Identifier: Apache-2.0
"""The sweep aggregator must refuse to over-read, and must not compare across checkpoints.

WHY THIS EXISTS
---------------
``wf_blkc_l3_guidance_ab.py`` is pairwise and geometry-only. The sweep is
four-arm and decode-enabled, so this file aggregates N arms, adds the decoded
molecule's properties, and — the part that matters — refuses to present a
geometry win as a molecule win.

Three specific failure modes are guarded:

1. **No control.** A sweep handed only guided arms has nothing to be different
   from. Returning a table anyway invites reading a level as an effect.
2. **Geometry winner != molecule winner.** If guidance improves coordinates
   while leaving or worsening what the bond head makes of them, the table
   hides that and a reader scanning for the biggest geometry delta concludes
   the sweep worked. The contradiction is stated in the HEADLINE.
3. **Cross-checkpoint comparison.** This sweep decodes from
   ``cfm_patience10_seed0``; ``cfm_checkpoint_property_benchmark.json`` was
   measured on ``cfm_fullcorpus_2026_10_10`` epoch 10. The control arm here
   scores QED 0.2358 where that panel recorded 0.0944. Read as an improvement
   it is a 2.5x jump that did not happen. The two must never be compared.

RED CHECK: delete the contradiction branch from ``headline`` and
``test_geometry_winner_differing_from_molecule_winner_is_called_out`` fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _mod():
    name = "wf_blkc_l3_sweep_aggregate"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(REPO / "molmetal" / "scripts" / f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _arm(strength: float, min_d: float, decode_rate: float, n_ok: int):
    return {
        "arm": f"strength{strength}",
        "strength": strength,
        "geometry": {
            "min_pair_distance": {
                "generated": min_d, "real_tmqm": 1.172,
                "direction_wanted": "up",
            },
            "frac_pairs_hard_clash": {
                "generated": 0.01 * strength, "real_tmqm": 0.0,
                "direction_wanted": "down",
            },
        },
        "decode": {
            "n_attempted": 64, "n_with_smiles": n_ok,
            "decode_rate": decode_rate,
            "single_fragment_ratio": 0.6, "n_errors": 0,
            "all_clouds_errored": False,
        },
    }


def test_control_only_is_reported_as_no_comparison():
    m = _mod()
    h = m.headline([_arm(0.0, 0.53, 0.34, 22)], _arm(0.0, 0.53, 0.34, 22))
    assert "NO GUIDED ARM" in h, (
        f"a sweep with only the control produced a comparison anyway: {h!r}. "
        "A level would then be read as an effect."
    )


def test_geometry_winner_differing_from_molecule_winner_is_called_out():
    """The exact shape of a false positive: geometry improves, decode does not."""
    m = _mod()
    control = _arm(0.0, 0.53, 0.34, 22)
    # 0.05 gives the BEST geometry but a WORSE decode rate than 0.01.
    arms = [control, _arm(0.01, 0.90, 0.40, 26), _arm(0.05, 1.10, 0.30, 19)]
    h = m.headline(arms, control)
    assert "CONTRADICTION" in h, (
        "the best-geometry arm (0.05) is not the best-decoding arm (0.01), "
        f"but the headline did not say so:\n{h}"
    )
    assert "0.05" in h and "0.01" in h, (
        f"the contradiction must name BOTH strengths, not gesture at it:\n{h}"
    )


def test_agreement_is_not_reported_as_a_contradiction():
    m = _mod()
    control = _arm(0.0, 0.53, 0.34, 22)
    arms = [control, _arm(0.01, 0.90, 0.40, 26), _arm(0.05, 1.10, 0.45, 29)]
    h = m.headline(arms, control)
    assert "CONTRADICTION" not in h, (
        f"geometry and decode agree here, so no contradiction should be "
        f"claimed:\n{h}"
    )


def test_decode_denominator_is_always_reported():
    """The earlier Vina-style survivor count (n=51 over an unstated attempted
    set) conditioned the mean on success. Both numbers must always appear."""
    m = _mod()
    src = (REPO / "molmetal" / "scripts" / f"{m.__name__}.py").read_text()
    assert "n_attempted" in src and "n_with_smiles" in src, (
        "the aggregator dropped one of the two decode counts"
    )
    h = m.headline([_arm(0.0, 0.53, 0.34, 22), _arm(0.05, 1.1, 0.45, 29)],
                   _arm(0.0, 0.53, 0.34, 22))
    assert "22" in h and "29" in h, (
        f"the headline must quote successes per arm, not just a rate:\n{h}"
    )


def test_cross_checkpoint_comparison_is_forbidden_in_the_output():
    m = _mod()
    src = (REPO / "molmetal" / "scripts" / f"{m.__name__}.py").read_text()
    assert "DO_NOT_COMPARE_ACROSS_CHECKPOINTS" in src, (
        "the output carries no warning that these properties are measured on a "
        "different checkpoint and cohort than "
        "cfm_checkpoint_property_benchmark.json, so the control's QED of 0.2358 "
        "reads as a 2.5x improvement over the recorded 0.0944"
    )


def test_missing_arms_abort_rather_than_produce_a_partial_table():
    """A partial sweep must not look like a complete one.

    Three of four strengths landing is not the same finding as two of four,
    and the reader of the JSON cannot tell which happened unless it is recorded.
    """
    import subprocess

    mod = _mod()
    r = subprocess.run(
        [sys.executable,
         str(REPO / "molmetal" / "scripts" / f"{mod.__name__}.py"),
         "--arms", str(REPO / "does_not_exist_arm.json"),
         "--out", "/tmp/should_not_exist_agg.json"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode != 0, (
        "a missing arm file was accepted; the aggregator would emit a table "
        "that reads like a finished sweep"
    )
    assert "MISSING" in (r.stdout + r.stderr), (
        "the failure did not say which file was missing: " + (r.stdout + r.stderr)[-300:]
    )

def test_best_geometry_is_closest_to_real_not_largest_value(monkeypatch):
    """Ranking geometry by RAW value always picks the strongest arm.

    That is not a tie-break, it is a blind spot. At strength 0.15 in the
    measured sweep, min_pair_distance reaches 1.368 A against a real-tmqm 1.229
    -- an OVERSHOOT -- while p01_pair_distance (0.390 error) and
    radius_of_gyration (0.767 error) are both worse than the control. Ranking
    by raw min_pair_distance selects 0.15 and reports it as "best geometry";
    ranking by total absolute error against real selects 0.05, which is the
    arm that actually looks most like a real complex.

    RED CHECK: revert `_geo_err` to a raw-value max and this fails.
    """
    m = _mod()

    def arm(strength, md, p01, clash, rg, decode, n_ok):
        return {
            "arm": f"s{strength}", "strength": strength,
            "geometry": {
                "min_pair_distance": {"generated": md, "real_tmqm": 1.2293},
                "p01_pair_distance": {"generated": p01, "real_tmqm": 1.3370},
                "frac_pairs_hard_clash": {"generated": clash, "real_tmqm": 0.0},
                "radius_of_gyration": {"generated": rg, "real_tmqm": 3.9598},
            },
            "decode": {"n_attempted": 64, "n_with_smiles": n_ok,
                       "decode_rate": decode, "single_fragment_ratio": 0.7,
                       "n_errors": 0, "all_clouds_errored": False},
        }

    # Values lifted from the MEASURED sweep (metrics/by_metric/
    # cfm_l3_guidance_sweep.json), so this pins the real shape of the data and
    # not a shape invented to make the test pass.
    control = arm(0.0, 0.5338, 1.1311, 0.01449, 3.6967, 0.34375, 22)
    s05 = arm(0.05, 1.1354, 1.5125, 0.00264, 4.0190, 0.5, 32)
    s15 = arm(0.15, 1.3679, 1.7270, 0.00009, 4.7267, 0.828125, 53)

    h = m.headline([control, s05, s15], control)
    assert "strength 0.05" in h.split("best DECODED")[0], (
        "the best-geometry arm must be 0.05 (closest to real on total error), "
        f"not 0.15 (largest raw min_pair_distance, i.e. an overshoot).\n{h}"
    )
    assert "OVERSHOOT" in h, (
        "the arm that wins on decode must be flagged as overshooting the real "
        f"band; the headline did not:\n{h}"
    )
    assert "CONTRADICTION" in h, (
        "geometry wants 0.05 and decode wants 0.15; that is the whole point of "
        f"the sweep and it was not called out:\n{h}"
    )


def test_property_columns_are_reported_with_their_n(monkeypatch):
    """The selection effect is only visible if each arm's n is stated."""
    m = _mod()
    src = (REPO / "molmetal" / "scripts" / f"{m.__name__}.py").read_text()
    assert "SELECTION EFFECT" in src, (
        "the aggregator compares QED/SA across arms whose cohorts differ in "
        "SIZE (22/23/32/53 decoded molecules) without saying so; that reads as "
        "a like-for-like property comparison and it is not one"
    )
    arm = {
        "arm": "s", "strength": 0.15,
        "geometry": {"min_pair_distance": {"generated": 1.37, "real_tmqm": 1.23}},
        "decode": {"n_attempted": 64, "n_with_smiles": 53, "decode_rate": 0.83,
                   "single_fragment_ratio": 0.81, "n_errors": 0},
        "properties_on_decoded": {"qed": {"mean": 0.1327, "n": 53}, "n_parsed": 53},
    }
    control = {
        "arm": "c", "strength": 0.0,
        "geometry": {"min_pair_distance": {"generated": 0.53, "real_tmqm": 1.23}},
        "decode": {"n_attempted": 64, "n_with_smiles": 22, "decode_rate": 0.34,
                   "single_fragment_ratio": 0.64, "n_errors": 0},
        "properties_on_decoded": {"qed": {"mean": 0.2358, "n": 22}, "n_parsed": 22},
    }
    h = m.headline([control, arm], control)
    assert "0.1327" in h and "0.2358" in h, (
        f"the QED direction must be stated with both values:\n{h}"
    )
    assert "53" in h and "22" in h, (
        f"both cohort sizes must appear so the selection effect is readable:\n{h}"
    )
