# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Wf-A142 (2026-10-31) — the one-sided element-pair distance UPPER BOUND:
is it REACHABLE, and does it behave like the TABASCO bound it stands for?

ROW 2 OF THE FRONTIER
----------------------
``/tmp/wfa140/frontier-lit.md`` row 2: TABASCO (Vonessen, Harris, Cretu,
Liò, "TABASCO: A Fast, Simplified Model for Molecular Generation with
Improved Physical Quality", arXiv:2507.00899, ICML 2025 GenBio Workshop)
§3.5 "physically-constrained guidance" — pre-tabulated element-pair
distance BOUNDS, penalised ONE-SIDEDLY, parameter-free.

The repo already implements the "above the bound" half of that idea, behind
three DEFAULT-OFF kwargs on ``_BondLengthGuidance``
(``molmetal/adapters/flow_matching_lipman/__init__.py`` :1305-1307):

    upper_bound          (False)   -- the switch
    upper_bound_reach    (3.5 A)   -- the bond head's own candidate cutoff
    upper_bound_strength (None)    -- None means "same as the spring constant"

Wf-A142's job (this file) is the REACHABILITY + SIGN half of the question:

  1. the flag-OFF path must still be BIT-EXACT with the pre-Wf-A128
     algorithm, and with the HARD-CODED legacy expectations;
  2. the flag-ON path must produce a RESTORING force exactly on the pairs
     the spring's inner cutoff abandoned (the 2.5-3.5 A dead zone) and
     nowhere else — i.e. the implied potential is the one-sided quadratic
     TABASCO uses;
  3. the switch must be REACHABLE from a YAML config under AGENTS.md §5.11,
     with the right TYPE on the round trip;
  4. the switch must be SAMPLING-ONLY — it cannot move a training gradient.

WHAT THIS FILE DELIBERATELY DOES NOT DO
---------------------------------------
It does not assert any decode-level connectivity/fragmentation number.  Those
live in ``molmetal/reports/wfa142_upper_bound/probe.json``, produced by
``molmetal/scripts/wfa142_upper_bound_probe.py`` on CPU; a unit test that
re-ran a decode would be a slow test with a stochastic denominator.

CPU-ONLY
--------
Every torch op here is on the CPU by construction (the tensors are tiny).
Run with ``CUDA_VISIBLE_DEVICES=""`` — the repo's full pytest suite
auto-selects CUDA and burns VRAM.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_REL = "molmetal/adapters/flow_matching_lipman/__init__.py"
ADAPTER = PROJECT_ROOT / ADAPTER_REL
TRAINER_REL = "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
WFA128_TEST_REL = "molmetal/tests/test_wfa128_bond_length_reach.py"

#: the config that actually carries the switch (the ONLY one in the repo).
WFA128_CONFIG_REL = "configs/wfa128_bond_reach_upper_bound_2026_10_31.yaml"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_AD = _load_module(ADAPTER, "_ad_wfa142")
_BLG = _AD._BondLengthGuidance

# The Wf-A128 regression test carries a SELF-CONTAINED transcription of the
# pre-Wf-A128 algorithm (``_legacy_force``) plus the reference pair helper
# (``_pair``).  We import it rather than re-transcribing it a second time:
# two independent transcriptions would be two places to drift.
_WFA128 = _load_module(PROJECT_ROOT / WFA128_TEST_REL, "_t_wfa128")
_legacy_force = _WFA128._legacy_force
_pair = _WFA128._pair


def _cc_guidance(**kw):
    """A C-C guidance with the Wf-A128 test's settings (cutoff 5.0 A so the
    guidance's radius graph reaches every distance under test)."""
    return _BLG(
        atom_types=torch.tensor([6, 6], dtype=torch.long),
        cutoff=5.0,
        guidance_weight=1.0,
        **kw,
    )


#: ``radius_edge_index`` builds a DIRECTED graph: one unordered pair appears
#: twice (i->j and j->i) and each directed edge contributes its own force,
#: so a pair receives TWICE ``strength * (d - expected)``.  MEASURED on the
#: pre-Wf-A128 code: a C-C pair at 2.4 A got +90.0 = 2 * 50.0 * (2.4 - 1.5).
_DOUBLE = 2.0
_SPRING_K = 50.0
#: the C-C expected bond length used by ``_expected_length_table``.
_CC_TARGET = 1.50


# ---------------------------------------------------------------------------
# 1 — the flag-OFF path is bit-exact with the pre-Wf-A128 algorithm
# ---------------------------------------------------------------------------
def test_flag_off_is_bit_exact_with_the_wfa128_reference() -> None:
    """The reference is the Wf-A128 test's own transcription of the
    pre-Wf-A128 ``__call__`` — the algorithm that ran before the upper bound
    existed.  ``torch.equal`` (bitwise), not ``allclose``."""
    torch.manual_seed(20261031)
    cases = [
        # (a) a random batch
        (
            torch.randint(1, 80, (3, 6), dtype=torch.long),
            torch.randn(3, 6, 3, dtype=torch.float32) * 2.5,
            None,
        ),
        # (b) a 3-atom chain inside the inner cutoff
        (
            torch.tensor([6, 6, 6], dtype=torch.long),
            torch.tensor(
                [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
            None,
        ),
        # (c) a padded batch
        (
            torch.tensor([[6, 6, 7, 8, 0]], dtype=torch.long),
            torch.randn(1, 5, 3, dtype=torch.float32) * 2.0,
            None,
        ),
        # (d) no edges at all
        (
            torch.tensor([6, 6], dtype=torch.long),
            torch.tensor(
                [[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]], dtype=torch.float32
            ),
            None,
        ),
    ]
    for idx, (z, x, mask) in enumerate(cases):
        legacy_g = _BLG(
            atom_types=z, cutoff=5.0, guidance_weight=1.0, node_mask=mask
        )
        new_g = _BLG(
            atom_types=z,
            cutoff=5.0,
            guidance_weight=1.0,
            node_mask=mask,
            upper_bound=False,
        )
        ref = _legacy_force(legacy_g, x)
        got = new_g(x)
        assert torch.equal(ref, got), (
            f"case {idx}: the flag-OFF path must be BIT-EXACT with the "
            f"pre-Wf-A128 algorithm; max |diff| = "
            f"{(ref - got).abs().max().item():.3e}"
        )


def test_flag_off_matches_the_hardcoded_legacy_expectations() -> None:
    """The legacy behaviour, written down as NUMBERS so the claim does not
    rest on "whatever the code does now":

      * a C-C pair at 2.8 A -> EXACTLY 0.0 (the dead zone the bound fills);
      * a C-C pair at 2.4 A -> EXACTLY +90.0 on atom 0 (attractive), which is
        2 * 50.0 * (2.4 - 1.50).

    These are the values the Wf-A128 test's own docstring records as
    "MEASURED before the fix"."""
    off = _cc_guidance(upper_bound=False)

    f28 = off(_pair(2.8))
    assert torch.equal(f28, torch.zeros_like(f28)), (
        f"legacy: a C-C pair at 2.8 A must get EXACTLY zero force; got "
        f"max abs = {f28.abs().max().item():.3e}"
    )

    f24 = off(_pair(2.4))
    assert float(f24[0, 0, 0].item()) == pytest.approx(90.0, rel=1e-6), (
        f"legacy: a C-C pair at 2.4 A must get +90.0 on atom 0; got "
        f"{float(f24[0, 0, 0].item()):.6f}"
    )


# ---------------------------------------------------------------------------
# 2 — the DEAD ZONE: exactly the pairs the spring abandoned
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", (2.5, 2.6, 2.8, 3.0, 3.4))
def test_bound_off_is_exactly_zero_across_the_dead_zone(d: float) -> None:
    """2.5 A is the spring's inner cutoff
    (``_INNER_BOND_CUTOFF``, adapter :1403) and 3.5 A is the YuelBond head's
    candidate cutoff (``yuelbond_decoder.py`` :531/:696/:903).  Between the
    two the head could have bonded a pair that the guidance never touched."""
    f = _cc_guidance(upper_bound=False)(_pair(d))
    assert torch.equal(f, torch.zeros_like(f)), (
        f"bound OFF: a C-C pair at {d} A must receive exactly zero force; "
        f"got max abs = {f.abs().max().item():.3e}"
    )


@pytest.mark.parametrize("d", (2.6, 2.8, 3.0, 3.4))
def test_bound_on_restores_the_dead_zone(d: float) -> None:
    """Bound ON: the pair is pulled TOGETHER (atom 0 pushed +x, atom 1 -x)
    and the subsystem's net force is zero (Newton's 3rd law)."""
    f = _cc_guidance(upper_bound=True)(_pair(d))
    f0_x = float(f[0, 0, 0].item())
    f1_x = float(f[0, 1, 0].item())
    assert f0_x > 0.0, (
        f"bound ON: a stretched C-C pair at {d} A must be pulled together; "
        f"got f0_x = {f0_x:.4f}"
    )
    assert f1_x < 0.0, (
        f"bound ON: the partner must be pulled the other way; got "
        f"f1_x = {f1_x:.4f}"
    )
    assert abs(f0_x + f1_x) < 1e-5, (
        f"bound ON: the net force on the 2-atom subsystem must be zero; got "
        f"{f0_x + f1_x:.3e}"
    )


def _bound_contribution(d: float) -> float:
    """The force the BOUND contributes, isolated from the spring.

    The bound is ADDITIVE to the spring (the Wf-A128 test pins that), so
    the bound's own contribution is ``F(flag ON) - F(flag OFF)``.  Reading
    the raw flag-ON force would mix the two: below the spring's 2.5 A
    inner cutoff the spring fires too, and the total is ``spring + bound``."""
    on = _cc_guidance(upper_bound=True)(_pair(d))
    off = _cc_guidance(upper_bound=False)(_pair(d))
    return float((on - off)[0, 0, 0].item())


def test_bound_force_is_affine_in_d_with_a_zero_at_the_target() -> None:
    """The bound's own contribution is ``2 * k * (d - d_target)`` on
    ``(d_target, reach]`` and EXACTLY 0 below ``d_target``.

    That is the one-sided quadratic potential TABASCO uses:
    ``U(d) = 1/2 * (2k) * max(0, d - d_target)**2`` — zero on the allowed
    side, rising quadratically on the forbidden side."""
    for d in (1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4):
        got = _bound_contribution(d)
        want = _DOUBLE * _SPRING_K * (d - _CC_TARGET)
        assert got == pytest.approx(want, rel=1e-5), (
            f"bound ON: the bound's contribution at d={d} must be "
            f"2k(d - d_target) = {want}; got {got}"
        )
    for d in (0.9, 1.1, 1.3, 1.5):
        got = _bound_contribution(d)
        assert got == 0.0, (
            f"bound ON: below the target length the bound must contribute "
            f"EXACTLY nothing (one-sided); at d={d} got {got}"
        )


def test_implied_potential_is_monotone_decreasing_to_the_target() -> None:
    """MEASURE the implied potential by integrating the MEASURED bound
    force, and check the shape TABASCO relies on.

    ``U(d) = int_{d_target}^{d} F_bound(s) ds`` with F_bound the force this
    repo actually produces (so the constant of integration is fixed at
    ``U(d_target) = 0``).

    U is strictly INCREASING in d above the target — a stretched pair costs
    energy, so the potential drives d DOWN toward the target, and U falls to
    exactly 0 as d reaches it.  U is identically 0 at and below the target
    (one-sided)."""
    d_target = _CC_TARGET
    grid = [1.5 + 0.1 * i for i in range(1, 21)]  # 1.6 .. 3.5
    forces = [_bound_contribution(d) for d in grid]

    # trapezoid integration of F_bound over [d_target, d]
    us: list[float] = []
    acc = 0.0
    prev_d, prev_f = d_target, 0.0
    for d, f in zip(grid, forces):
        acc += 0.5 * (prev_f + f) * (d - prev_d)
        us.append(acc)
        prev_d, prev_f = d, f

    assert us[0] > 0.0, (
        f"the implied potential must be POSITIVE just above the target "
        f"(the pair is over-stretched); got U(1.6) = {us[0]}"
    )
    for i in range(1, len(us)):
        assert us[i] > us[i - 1], (
            f"the implied potential must be STRICTLY INCREASING in d above "
            f"the target (a stretched pair must cost energy, so the "
            f"potential can drive it back down); U({grid[i]:.2f}) = "
            f"{us[i]:.3f} is not above U({grid[i - 1]:.2f}) = {us[i - 1]:.3f}"
        )
    # the closed form: 1/2 * (2k) * (d - d_target)^2
    for d, u in zip(grid, us):
        want = 0.5 * (_DOUBLE * _SPRING_K) * (d - d_target) ** 2
        assert u == pytest.approx(want, rel=2e-3), (
            f"the implied potential must be the one-sided quadratic; "
            f"U({d:.2f}) = {u:.4f}, expected {want:.4f}"
        )


def test_bound_is_one_sided_below_the_target() -> None:
    """A COMPRESSED pair (1.2 A < 1.50 A) must get the same force with the
    switch on and off: the bound may never push atoms apart."""
    off = _cc_guidance(upper_bound=False)(_pair(1.2))
    on = _cc_guidance(upper_bound=True)(_pair(1.2))
    assert torch.equal(off, on), (
        f"the upper bound must be ONE-SIDED; max |diff| = "
        f"{(off - on).abs().max().item():.3e}"
    )


def test_bound_respects_its_reach() -> None:
    """A pair at 3.6 A and at 4.0 A is outside the 3.5 A reach and must get
    nothing from the bound."""
    for d in (3.6, 4.0):
        off = _cc_guidance(upper_bound=False)(_pair(d))
        on = _cc_guidance(upper_bound=True)(_pair(d))
        assert torch.equal(off, on), (
            f"a pair at {d} A is beyond the 3.5 A reach and must be "
            f"untouched by the bound; max |diff| = "
            f"{(off - on).abs().max().item():.3e}"
        )


# ---------------------------------------------------------------------------
# 3 — YAML reachability (AGENTS.md §5.11)
# ---------------------------------------------------------------------------
def test_the_switch_is_reachable_from_a_yaml_config() -> None:
    """The switch must be settable from a YAML config, and the round trip
    must preserve the TYPE (bool / float), not merely the value.

    The YAML key PATH is ``flow.bond_length_guidance_upper_bound``; the
    loader flattens it to ``flow_bond_length_guidance_upper_bound``
    (section prefix + leaf), which is the name the config-binding gate
    resolves against the trainer's argparse dest.

    MEASURED GOTCHA (AGENTS.md §5.11): a relative ``--config`` path that
    does not resolve from the CWD silently leaves every dest at its argparse
    default — no warning, no error.  The round trip below therefore uses an
    ABSOLUTE path, which is the only form that can fail loudly."""
    from molmetal.scripts.wfA2_config_loader import load_config

    cfg_path = PROJECT_ROOT / WFA128_CONFIG_REL
    assert cfg_path.is_file(), f"missing config: {cfg_path}"
    flat = load_config(str(cfg_path))
    for leaf in (
        "bond_length_guidance_weight",
        "bond_length_guidance_upper_bound",
        "bond_length_guidance_upper_bound_reach",
        "bond_length_guidance_upper_bound_strength",
    ):
        hits = [k for k in flat if k.endswith(leaf)]
        assert hits, (
            f"{WFA128_CONFIG_REL} must carry a key ending in {leaf!r}; "
            f"the loaded keys are {sorted(flat)}"
        )

    from molmetal.scripts import train_cfm_yuelbond_wasserstein as trainer

    old_argv = sys.argv
    sys.argv = ["train_cfm_yuelbond_wasserstein.py", "--config", str(cfg_path)]
    try:
        args = trainer.parse_args()
    finally:
        sys.argv = old_argv

    assert args.bond_length_guidance_upper_bound is True, (
        f"the switch must round-trip as the bool True; got "
        f"{args.bond_length_guidance_upper_bound!r} "
        f"({type(args.bond_length_guidance_upper_bound).__name__})"
    )
    assert isinstance(args.bond_length_guidance_upper_bound, bool)
    for dest, want in (
        ("bond_length_guidance_weight", 0.1),
        ("bond_length_guidance_upper_bound_reach", 3.5),
        ("bond_length_guidance_upper_bound_strength", 25.0),
    ):
        got = getattr(args, dest)
        assert isinstance(got, float), (
            f"{dest} must round-trip as a float; got {got!r} "
            f"({type(got).__name__})"
        )
        assert got == pytest.approx(want, rel=1e-9), (
            f"{dest} must round-trip as {want}; got {got}"
        )


def test_the_switch_is_absent_from_every_other_config() -> None:
    """The switch is reachable, and from exactly ONE config.  If a second
    config starts carrying it, this guard must be re-read — the point is to
    make the surface auditable, not to forbid it forever."""
    from molmetal.scripts.wfA2_config_loader import load_config

    hits = []
    for cfg in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        try:
            flat = load_config(str(cfg))
        except Exception:  # noqa: BLE001 — a broken config is another gate's job
            continue
        if any(k.endswith("bond_length_guidance_upper_bound") for k in flat):
            hits.append(cfg.name)
    assert hits == [Path(WFA128_CONFIG_REL).name], (
        f"only {Path(WFA128_CONFIG_REL).name} may carry the switch; "
        f"found {hits}"
    )


# ---------------------------------------------------------------------------
# 4 — the switch is SAMPLING-ONLY
# ---------------------------------------------------------------------------
def test_the_switch_cannot_move_a_training_gradient() -> None:
    """``bond_length_guidance_weight`` (and therefore the switch that lives
    inside the guidance object) must be consumed ONLY by the sampling /
    generation-evaluation path.  If a training-loss function ever reads it,
    the switch would silently become a training hyper-parameter and every
    archived run would need re-reading.

    TWO independent AST guards (a grep for the identifier misses
    ``getattr(args, "bond_length_guidance_weight")``, which the repo has
    been bitten by before):

      1. the guidance WRAPPER ``_with_bond_length_guidance`` may only be
         CALLED from ``generate_clouds_from_noise``;
      2. ``_BondLengthGuidance`` may only be CONSTRUCTED inside that
         wrapper.

    Together they say: the guidance object does not exist during a
    training step, so the switch cannot reach a gradient."""
    src = (PROJECT_ROOT / TRAINER_REL).read_text(encoding="utf-8")
    tree = ast.parse(src)

    allowed_readers = {
        "generate_clouds_from_noise",
        "evaluate_generation_validity",
        "_with_bond_length_guidance",
        "parse_args",
        "build_parser",
        "main",  # forwards the args into the gen-eval call
    }
    readers: set[str] = set()
    wrapper_callers: set[str] = set()
    guidance_ctor_callers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                names.add(sub.attr)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                names.add(sub.value)
        if "bond_length_guidance_weight" in names:
            readers.add(node.name)
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            fn = sub.func
            fname = (
                fn.id
                if isinstance(fn, ast.Name)
                else (fn.attr if isinstance(fn, ast.Attribute) else "")
            )
            if fname == "_with_bond_length_guidance":
                wrapper_callers.add(node.name)
            if fname == "_BondLengthGuidance":
                guidance_ctor_callers.add(node.name)

    assert readers <= allowed_readers, (
        f"bond_length_guidance_weight must be read ONLY by the sampling path "
        f"({sorted(allowed_readers)}); it also appears in "
        f"{sorted(readers - allowed_readers)}"
    )
    assert wrapper_callers <= {"generate_clouds_from_noise"}, (
        f"the guidance wrapper must be called ONLY from the sampling path; "
        f"it is also called from {sorted(wrapper_callers - {'generate_clouds_from_noise'})}"
    )
    assert guidance_ctor_callers <= {"_with_bond_length_guidance"}, (
        f"_BondLengthGuidance must be constructed ONLY inside the guidance "
        f"wrapper; it is also constructed in "
        f"{sorted(guidance_ctor_callers - {'_with_bond_length_guidance'})}"
    )


def test_the_switch_defaults_are_off() -> None:
    """The argparse default must stay OFF: the bound is opt-in."""
    from molmetal.ports import GenerationConfig

    cfg = GenerationConfig()
    assert cfg.bond_length_guidance_upper_bound is False
    assert cfg.bond_length_guidance_upper_bound_reach == 3.5
    assert cfg.bond_length_guidance_upper_bound_strength is None
    assert cfg.bond_length_guidance_weight == 0.0


# ---------------------------------------------------------------------------
# 5 — the probe's own geometry helper (a regression guard on a MEASURED trap)
# ---------------------------------------------------------------------------
def test_geometry_band_stats_excludes_the_self_pair() -> None:
    """MEASURED (Wf-A142, 2026-10-31): ``torch.cdist(p, p)`` does NOT have an
    exact zero diagonal in float32 — the matmul path cancels catastrophically
    and leaves ~1e-4 A there.  A naive ``d.min(dim=1).values`` then returns
    the SELF-distance for every atom and reports a mean nearest-neighbour of
    ~1e-4 A for a perfectly healthy geometry, which reads as a total
    collapse.  The probe computes the distances in float64 and fills the
    diagonal with +inf; this test pins that behaviour."""
    from molmetal.scripts.wfa142_upper_bound_probe import geometry_band_stats

    class _Cloud:
        def __init__(self, positions, atomic_numbers):
            self.positions = positions
            self.atomic_numbers = atomic_numbers

    cloud = _Cloud(
        torch.tensor([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([6, 6], dtype=torch.long),
    )
    stats = geometry_band_stats([cloud])
    assert stats["n_clouds"] == 1
    assert stats["mean_nn_distance"] == pytest.approx(2.8, rel=1e-6), (
        f"the nearest-neighbour distance must EXCLUDE the self-pair; got "
        f"{stats['mean_nn_distance']}"
    )
    # 2.8 A sits in the 2.5-3.5 A dead zone.  The pair count is DIRECTED
    # (each unordered pair is counted twice, matching ``radius_edge_index``),
    # so one unordered pair in the band reads as 2.
    assert stats["mean_pairs_in_band_per_cloud"] == 2.0
    assert stats["frac_atoms_nn_in_band"] == pytest.approx(1.0, rel=1e-9)
