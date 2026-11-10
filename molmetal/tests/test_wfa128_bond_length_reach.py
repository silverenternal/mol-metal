"""Wf-A128 (2026-10-31) — the bond-length guidance's REACH gap, and the
one-sided upper-bound companion switch.

THE DEFECT
----------
``_BondLengthGuidance.__call__`` applied an inner-bond cutoff of 2.5 Å::

    _INNER_BOND_CUTOFF = 2.5
    inner_mask = d_bi < _INNER_BOND_CUTOFF

so a pair between 2.5 Å and the YuelBond head's own candidate cutoff
(3.5 Å — ``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py``,
``distance_cutoff: float = 3.5``) received NO restoring force at all.
Such a pair is INSIDE the head's candidate set — the head could have
bonded it — but the guidance could only ever TIGHTEN a near-bond, never
pull an atom into the head's candidate set.

MEASURED before the fix (Wf-A128, CPU): the force on a C-C pair is
exactly 0.0 at d = 2.5, 2.6, 2.8, 3.0 and 3.4 Å, while the same pair at
d = 2.4 Å gets +90.0 (attractive).

THE FIX
-------
A ONE-SIDED UPPER BOUND behind a NEW switch (``upper_bound``), DEFAULT
OFF.  It fires only when the distance EXCEEDS the expected bond length
and has NO inner truncation: it reaches out to ``upper_bound_reach``
(default 3.5 Å = the bond head's candidate cutoff).

THE BIT-EXACTNESS PROOF
------------------------
Two independent comparisons, both bitwise (``torch.equal``):

  1. against a self-contained transcription of the PRE-Wf-A128
     ``__call__`` (``_legacy_force`` below) — always meaningful;
  2. against the module as it was at ``git HEAD`` — a genuine
     old-vs-new comparison while the fix is uncommitted (it degenerates
     to new-vs-new after the fix is committed, which is why (1) exists).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_REL = "molmetal/adapters/flow_matching_lipman/__init__.py"
ADAPTER = PROJECT_ROOT / ADAPTER_REL


def _load_adapter():
    spec = importlib.util.spec_from_file_location("_ad_wfa128", str(ADAPTER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ad_wfa128"] = mod
    spec.loader.exec_module(mod)
    return mod


_AD = _load_adapter()
_BLG = _AD._BondLengthGuidance
_radius_edge_index = _AD.radius_edge_index


# ---------------------------------------------------------------------------
# The pre-Wf-A128 algorithm, transcribed.  Used as the reference for the
# bit-exactness proof.
# ---------------------------------------------------------------------------
def _legacy_force(guidance, x: torch.Tensor) -> torch.Tensor:
    """Exact transcription of ``_BondLengthGuidance.__call__`` as it was
    BEFORE Wf-A128 (no upper-bound block)."""
    if guidance._guidance_weight <= 0.0:
        return torch.zeros_like(x)
    edge_index, edge_mask = _radius_edge_index(
        x, guidance._node_mask, guidance._cutoff
    )
    if edge_index.shape[-1] == 0:
        return torch.zeros_like(x)
    b, n, _ = x.shape
    flat = x.reshape(b * n, 3)
    flat_force = torch.zeros_like(flat)
    _INNER_BOND_CUTOFF = 2.5
    for bi in range(b):
        src_bi = edge_index[bi, 0, :]
        dst_bi = edge_index[bi, 1, :]
        mask_bi = edge_mask[bi, :]
        if not bool(mask_bi.any().item()):
            continue
        diff_bi = x[bi, dst_bi[mask_bi]] - x[bi, src_bi[mask_bi]]
        d_bi = diff_bi.norm(dim=-1).clamp(min=1e-3)
        inner_mask = d_bi < _INNER_BOND_CUTOFF
        if not bool(inner_mask.any().item()):
            continue
        diff_bi = diff_bi[inner_mask]
        d_bi = d_bi[inner_mask]
        src_bi_eff = src_bi[mask_bi][inner_mask]
        dst_bi_eff = dst_bi[mask_bi][inner_mask]
        table = guidance._expected_length_table().to(x.device).to(x.dtype)
        expected_bi = table[
            guidance._atom_types[bi, src_bi_eff],
            guidance._atom_types[bi, dst_bi_eff],
        ].clamp(min=0.5)
        residual_bi = (d_bi - expected_bi)
        force_mag_bi = guidance._spring_constant * residual_bi
        unit_bi = diff_bi / d_bi.unsqueeze(-1)
        src_idx = src_bi_eff + bi * n
        dst_idx = dst_bi_eff + bi * n
        flat_force.index_add_(
            0, src_idx, force_mag_bi.unsqueeze(-1) * unit_bi
        )
        flat_force.index_add_(
            0, dst_idx, -force_mag_bi.unsqueeze(-1) * unit_bi
        )
    force = flat_force.reshape(b, n, 3)
    return guidance._guidance_weight * force


def _pair(d: float, z_i: int = 6, z_j: int = 6) -> torch.Tensor:
    """Two atoms on the x axis at distance ``d`` (Angstrom)."""
    return torch.tensor(
        [[[0.0, 0.0, 0.0], [float(d), 0.0, 0.0]]], dtype=torch.float32
    )


def _cc_guidance(**kw):
    return _BLG(
        atom_types=torch.tensor([6, 6], dtype=torch.long),
        cutoff=5.0,
        guidance_weight=1.0,
        **kw,
    )


#: ``radius_edge_index`` builds a DIRECTED graph: every unordered pair
#: appears twice (i->j and j->i) and each directed edge contributes its
#: own force, so one pair receives TWICE the naive
#: ``strength * (d - expected)``.  MEASURED on the pre-fix code: a C-C
#: pair at d = 2.4 A got +90.0 = 2 * 50.0 * (2.4 - 1.5).  This factor is
#: pre-existing behaviour and is preserved bit-exactly by the flag-off
#: path; it is spelled out here so the expectations below are readable.
_DOUBLE = 2.0


# ---------------------------------------------------------------------------
# 1 — the flag-off path is bit-exact with the pre-Wf-A128 algorithm
# ---------------------------------------------------------------------------
def test_flag_off_is_bit_exact_with_legacy_algorithm() -> None:
    """``torch.equal`` (bitwise, not approximate) against the transcribed
    pre-Wf-A128 implementation, over inputs that exercise every branch:
    a random cloud, a chain, a padded batch, and the empty-edge case."""
    torch.manual_seed(20261031)
    cases = []

    # (a) random batch of 3 clouds x 6 atoms
    cases.append(
        (
            torch.randint(1, 80, (3, 6), dtype=torch.long),
            torch.randn(3, 6, 3, dtype=torch.float32) * 2.5,
            None,
        )
    )
    # (b) a 3-atom chain inside the inner cutoff
    cases.append(
        (
            torch.tensor([6, 6, 6], dtype=torch.long),
            torch.tensor(
                [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
            None,
        )
    )
    # (c) a padded batch: the 5th atom is padding
    mask = torch.ones(1, 5, dtype=torch.bool)
    mask[0, 4] = False
    cases.append(
        (
            torch.tensor([[6, 6, 7, 8, 0]], dtype=torch.long),
            torch.randn(1, 5, 3, dtype=torch.float32) * 2.0,
            mask,
        )
    )
    # (d) no edges at all (two atoms 20 A apart)
    cases.append(
        (
            torch.tensor([6, 6], dtype=torch.long),
            torch.tensor(
                [[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]], dtype=torch.float32
            ),
            None,
        )
    )

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
            f"case {idx}: the flag-off path must be BIT-EXACT with the "
            f"pre-Wf-A128 algorithm; max |diff| = "
            f"{(ref - got).abs().max().item():.3e}"
        )


def test_flag_off_is_bit_exact_against_git_head_module() -> None:
    """Old path vs new path with the flag off, as a bitwise comparison of
    two independently loaded modules: the working tree and ``git HEAD``.

    While the fix is uncommitted this is a real old-vs-new comparison.
    After the fix lands, HEAD == the working tree and the comparison
    degenerates to new-vs-new — the transcribed reference above is what
    keeps the claim honest in that case.
    """
    proc = subprocess.run(
        ["git", "show", f"HEAD:{ADAPTER_REL}"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip("git HEAD blob unavailable (not a git checkout?)")
    # The adapter is a PACKAGE (``__init__.py``) and does a module-level
    # relative import (``from .amp import CFMAMPContext``), so the old
    # blob has to be loaded as a package directory, not as a bare file.
    pkg_src = PROJECT_ROOT / "molmetal" / "adapters" / "flow_matching_lipman"
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "wfa128_head_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(proc.stdout, encoding="utf-8")
        (pkg / "amp.py").write_text(
            (pkg_src / "amp.py").read_text(encoding="utf-8"), encoding="utf-8"
        )
        spec = importlib.util.spec_from_file_location(
            "wfa128_head_pkg", str(pkg / "__init__.py")
        )
        old_mod = importlib.util.module_from_spec(spec)
        sys.modules["wfa128_head_pkg"] = old_mod
        spec.loader.exec_module(old_mod)

    torch.manual_seed(20261031)
    z = torch.randint(1, 80, (3, 6), dtype=torch.long)
    x = torch.randn(3, 6, 3, dtype=torch.float32) * 2.5
    old_g = old_mod._BondLengthGuidance(
        atom_types=z, cutoff=5.0, guidance_weight=1.0
    )
    new_g = _BLG(
        atom_types=z, cutoff=5.0, guidance_weight=1.0, upper_bound=False
    )
    ref = old_g(x)
    got = new_g(x)
    assert torch.equal(ref, got), (
        f"the flag-off path must be BIT-EXACT with the pre-Wf-A128 module; "
        f"max |diff| = {(ref - got).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# 2 — the legacy gap is preserved when the switch is off
# ---------------------------------------------------------------------------
def test_flag_off_preserves_the_legacy_reach_gap() -> None:
    """A C-C pair at 2.8 A is inside the bond head's 3.5 A candidate set
    but outside the spring's 2.5 A inner cutoff.  With the switch OFF the
    force must be EXACTLY zero there — the fix must not silently change
    the default behaviour."""
    g = _cc_guidance(upper_bound=False)
    f = g(_pair(2.8))
    assert torch.equal(f, torch.zeros_like(f)), (
        f"flag off: a C-C pair at 2.8 A must receive exactly zero force; "
        f"got max abs = {f.abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# 3 — the switch reaches beyond the inner cutoff
# ---------------------------------------------------------------------------
def test_upper_bound_reaches_beyond_inner_cutoff() -> None:
    """With the switch ON, the 2.8 A pair finally gets an attractive
    force: atom 0 pushed +x (toward atom 1), atom 1 pushed -x."""
    g = _cc_guidance(upper_bound=True)
    f = g(_pair(2.8))
    f0_x = float(f[0, 0, 0].item())
    f1_x = float(f[0, 1, 0].item())
    assert f0_x > 0.0, (
        f"flag on: a stretched C-C pair at 2.8 A must be pulled together; "
        f"got f0_x = {f0_x:.4f}"
    )
    assert f1_x < 0.0, (
        f"flag on: the partner must be pulled the other way; "
        f"got f1_x = {f1_x:.4f}"
    )
    assert abs(f0_x + f1_x) < 1e-5, (
        f"flag on: net force on the 2-atom subsystem must be zero "
        f"(Newton's 3rd law); got {f0_x + f1_x:.3e}"
    )


def test_upper_bound_magnitude_matches_the_formula() -> None:
    """F = strength * (d - expected), with expected = 1.50 A for C-C and
    strength = the spring constant (50.0) by default — times the
    directed-graph factor of 2 (see ``_DOUBLE``)."""
    g = _cc_guidance(upper_bound=True)
    f = g(_pair(2.8))
    expected_mag = _DOUBLE * 50.0 * (2.8 - 1.50)
    assert float(f[0, 0, 0].item()) == pytest.approx(expected_mag, rel=1e-6), (
        f"flag on: |F| must be strength * (d - d_target) = {expected_mag}; "
        f"got {float(f[0, 0, 0].item()):.6f}"
    )


def test_upper_bound_strength_is_independently_settable() -> None:
    g = _cc_guidance(upper_bound=True, upper_bound_strength=10.0)
    f = g(_pair(2.8))
    assert float(f[0, 0, 0].item()) == pytest.approx(
        _DOUBLE * 10.0 * (2.8 - 1.50), rel=1e-6
    )


# ---------------------------------------------------------------------------
# 4 — the bound is ONE-SIDED: it never pushes atoms apart
# ---------------------------------------------------------------------------
def test_upper_bound_is_one_sided_on_a_compressed_pair() -> None:
    """A pair BELOW its expected length must be unaffected by the bound:
    the flag-on force equals the flag-off force exactly (the spring's
    repulsion only)."""
    off = _cc_guidance(upper_bound=False)(_pair(1.2))
    on = _cc_guidance(upper_bound=True)(_pair(1.2))
    assert torch.equal(off, on), (
        f"the upper bound must be one-sided: a compressed pair (1.2 A < "
        f"1.50 A) must get the same force with the switch on and off; "
        f"max |diff| = {(off - on).abs().max().item():.3e}"
    )
    # ... and the force is repulsive (pushing them apart).
    assert float(on[0, 0, 0].item()) < 0.0, (
        "a compressed pair must still be pushed apart by the spring"
    )


# ---------------------------------------------------------------------------
# 5 — the reach is bounded by the bond head's candidate cutoff
# ---------------------------------------------------------------------------
def test_upper_bound_respects_its_reach() -> None:
    """A pair at 4.0 A is inside the guidance's radius graph (cutoff
    5.0 A) but OUTSIDE the upper bound's reach (3.5 A = the YuelBond
    head's candidate cutoff).  It must get nothing from the bound."""
    off = _cc_guidance(upper_bound=False)(_pair(4.0))
    on = _cc_guidance(upper_bound=True)(_pair(4.0))
    assert torch.equal(off, on), (
        f"a pair beyond the reach must be untouched by the bound; "
        f"max |diff| = {(off - on).abs().max().item():.3e}"
    )


def test_upper_bound_covers_the_head_candidate_cutoff() -> None:
    """A pair at 3.4 A (inside the head's 3.5 A candidate set) IS reached
    by the bound; a pair at 3.6 A is not."""
    reached = _cc_guidance(upper_bound=True)(_pair(3.4))
    assert float(reached[0, 0, 0].item()) > 0.0, (
        "a pair at 3.4 A is inside the bond head's candidate set and must "
        "be reached by the bound"
    )
    beyond = _cc_guidance(upper_bound=True)(_pair(3.6))
    assert torch.equal(beyond, torch.zeros_like(beyond)), (
        "a pair at 3.6 A is outside the 3.5 A reach and must get nothing"
    )


# ---------------------------------------------------------------------------
# 6 — the bound is ADDITIVE to the spring (documented overlap)
# ---------------------------------------------------------------------------
def test_upper_bound_is_additive_to_the_spring() -> None:
    """Inside the 2.5 A inner cutoff the spring ALSO fires, so with the
    switch on the total is spring + bound.  This is the documented
    consequence of the bound being additive rather than a replacement."""
    off = _cc_guidance(upper_bound=False)(_pair(2.0))
    on = _cc_guidance(upper_bound=True)(_pair(2.0))
    spring = _DOUBLE * 50.0 * (2.0 - 1.50)
    bound = _DOUBLE * 50.0 * (2.0 - 1.50)
    assert float(off[0, 0, 0].item()) == pytest.approx(spring, rel=1e-6)
    assert float(on[0, 0, 0].item()) == pytest.approx(
        spring + bound, rel=1e-6
    )


def test_upper_bound_keeps_newtons_third_law_on_a_three_atom_chain() -> None:
    z = torch.tensor([6, 6, 6], dtype=torch.long)
    x = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.8, 0.0, 0.0], [5.6, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    g = _BLG(
        atom_types=z, cutoff=5.0, guidance_weight=1.0, upper_bound=True
    )
    f = g(x)
    net = float(f[0, :, 0].sum().item())
    assert abs(net) < 1e-4, (
        f"the net force on the chain must be zero; got {net:.3e}"
    )


# ---------------------------------------------------------------------------
# 7 — the config surface
# ---------------------------------------------------------------------------
def test_generation_config_default_is_off() -> None:
    from molmetal.ports import GenerationConfig

    cfg = GenerationConfig()
    assert cfg.bond_length_guidance_upper_bound is False, (
        "the companion switch must default to OFF"
    )
    assert cfg.bond_length_guidance_upper_bound_reach == 3.5, (
        "the default reach must be the YuelBond head's candidate cutoff"
    )
    assert cfg.bond_length_guidance_upper_bound_strength is None, (
        "None must mean 'same as the spring constant'"
    )
    # The pre-existing switch is untouched.
    assert cfg.bond_length_guidance_weight == 0.0


def test_adapter_passes_the_switch_to_the_guidance() -> None:
    """Source-level wiring guard: the adapter must read the switch off
    the GenerationConfig and hand it to ``_BondLengthGuidance``."""
    text = (PROJECT_ROOT / ADAPTER_REL).read_text(encoding="utf-8")
    assert "bond_length_guidance_upper_bound" in text, (
        "the adapter must read bond_length_guidance_upper_bound off the "
        "GenerationConfig"
    )
    assert "upper_bound=bool(" in text, (
        "the adapter must forward the switch to _BondLengthGuidance"
    )


def test_the_switch_is_additive_not_a_repurpose() -> None:
    """The pre-existing ``bond_length_guidance_weight`` must still drive
    the spring; the new switch must be a separate field."""
    from molmetal.ports import GenerationConfig

    assert "bond_length_guidance_weight" in GenerationConfig.__dataclass_fields__
    assert (
        "bond_length_guidance_upper_bound"
        in GenerationConfig.__dataclass_fields__
    )


# ---------------------------------------------------------------------------
# 8 — the honest production finding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cfg",
    (
        "configs/cfm_frag_assemble.yaml",
        "configs/c8_fullgen_b64_2026_10_24.yaml",
        "configs/c7_fullgen_2026_10_20.yaml",
    ),
)
def test_guidance_is_off_in_the_production_configs(cfg: str) -> None:
    """MEASURED 2026-10-31 (Wf-A128): none of the three production CFM
    configs carries ``bond_length_guidance_weight`` at all, so the
    argparse default (0.0) applies and the guidance — Wf-A75's spring or
    Wf-A128's upper bound — is OFF in production.  The Wf-A128 fix is
    therefore a no-op until someone enables the guidance."""
    from molmetal.scripts.wfA2_config_loader import load_config

    flat = load_config(str(PROJECT_ROOT / cfg))
    hits = {
        k: v for k, v in flat.items() if k.endswith("bond_length_guidance_weight")
    }
    assert not hits, (
        f"{cfg}: the production configs must not carry the guidance key "
        f"(found {hits}); if that changes, this guard must be re-read."
    )
