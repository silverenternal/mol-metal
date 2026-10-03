"""Wf-318F F2(a) MetalLigandExchange SMARTS runtime wiring tests (TODO-29 follow-up).

Background
----------
Wf-290 + Wf-318F identified the runtime-wiring gap: the F2(a)
MetalLigandExchange SMARTS library is defined on disk (5 curated +
84 L-R17-B2 matrix + 17 NEW SMARTS + 82-metal Wf-223 library = up to
188 patterns before dedup) but MCTS dispatch in
:mod:`molmetal_lam.search_alg.proof_search` only invokes the
:class:`MetalLigandExchange` ReactionRule class (one canonical
Pt_Cl_NH3 SMARTS). The 84+17 matrix + the 82-metal library are
*registerable* but the MCTS never queries them.

The :class:`F2MetalLigandExchangeRuntime` class in
:mod:`molmetal_lam.lam_chem.metal_ligand_exchange_runtime` closes
the gap with a lightweight dispatcher that MCTS reward can call.

This file locks down the dispatcher contract (12 tests total):

1. **84-rule load + dispatch** — 4 tests cover the basic
   load + dispatch surface.
2. **Per-metal coverage** — 3 tests cover Ru/Ir/Os/Rh/Re coverage.
3. **Integration with pt_metal_ligand_exchange** — 2 tests cover
   that the runtime composes with the existing
   :func:`register_metal_ligand_exchange_rules` registration helper.
4. **Edge cases** — 3 tests cover empty SMILES, invalid metal
   symbols, malformed RDKit parse.

Run with::

    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        molmetal/tests/test_wf318f_f2a_metal_ligand_runtime.py -v

Honest framing
--------------
* This is **CPU-only** integration work. The dispatcher is a thin
  wrapper around the existing SMARTS — no new chemistry is encoded.
* Phase-3G 10×3 rotation + r13_100x3 sweep (the BLOCKED-on-TODO-29
  follow-up) still need GPU cycles. After this ships the runtime
  is wired but the actual rotation runs on Blackwell / RTX 5090.
"""

from __future__ import annotations

import pytest

# RDKit — required for the dispatcher.
from rdkit import Chem  # type: ignore

# The runtime under test.
from molmetal_lam.lam_chem.metal_ligand_exchange_runtime import (
    F2DispatchResult,
    F2MetalLigandExchangeRuntime,
    F2RuntimeConfig,
    render_runtime_summary,
)

# Integration partners (existing on-disk modules).
from molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
    METAL_LIGAND_EXCHANGE_SMARTS,
    register_metal_ligand_exchange_rules,
)


# ===========================================================================
# Section 1 — 84-rule load + dispatch (4 tests).
# ===========================================================================


def test_runtime_default_config_loads_at_least_170_patterns() -> None:
    """Wf-318F acceptance: default config loads ≥ 170 patterns.

    Default config = curated F2(a) (5) + matrix (84) + NEW SMARTS (17)
    + 82-metal library minus p-block / synthetic filter (64) = 170.
    The dispatcher must surface these on construction.
    """
    rt = F2MetalLigandExchangeRuntime()
    assert rt.n_patterns_loaded >= 170, (
        f"expected ≥ 170 patterns loaded by default runtime; "
        f"got {rt.n_patterns_loaded} (5 curated + 84 matrix + 17 NEW "
        f"+ 64 82-metal-filtered = 170)"
    )


def test_runtime_dispatch_returns_bool_for_pt_nh3() -> None:
    """Wf-318F acceptance: Pt + NH3 donor matches F2(a) canonical.

    The canonical F2(a) ``Pt_Cl_NH3`` pattern is ``[Pt]([Cl])[*].[NH3]>>
    [Pt]([NH3])[*].[Cl]``. The dispatcher must return ``True`` for
    ``("Pt", "[NH3]")`` since the canonical pattern (and its matrix
    superset ``Pt_II_Cl_NH3`` + the 82-metal bare ``"Pt"`` entry) all
    carry the NH3 donor.
    """
    rt = F2MetalLigandExchangeRuntime()
    assert rt.dispatch("Pt", "[NH3]") is True


def test_runtime_dispatch_sub_libraries_are_reachable() -> None:
    """Wf-318F acceptance: each of the 4 sub-libraries is in ``patterns``.

    Spot-check by name: the curated ``Pt_Cl_NH3``, the matrix
    ``Pt_II_Cl_NH3``, a NEW SMARTS transmetalation entry, and the
    82-metal bare ``"Ru"`` entry must all be present after a default
    load.
    """
    rt = F2MetalLigandExchangeRuntime()
    loaded_names = set(rt.list_loaded_pattern_names())
    # Curated F2(a) — ``Pt_Cl_NH3`` is the canonical.
    assert "Pt_Cl_NH3" in loaded_names, f"curated F2(a) Pt_Cl_NH3 not in loaded library"
    # Matrix — ``Pt_II_Cl_NH3`` is the matrix form of the same reaction.
    assert "Pt_II_Cl_NH3" in loaded_names, f"matrix Pt_II_Cl_NH3 not in loaded library"
    # NEW SMARTS — the transmetalation prefix ``transmet_Pt_II_to_Pd_II``
    # should be in the loaded library.
    assert (
        "transmet_Pt_II_to_Pd_II" in loaded_names
    ), f"NEW SMARTS transmet_Pt_II_to_Pd_II not in loaded library"
    # 82-metal library — bare metal names like ``"Ru"`` survive the
    # dedup race because no curated / matrix / NEW entry uses a bare
    # metal as the pattern name.
    assert "Ru" in loaded_names, f"82-metal library bare Ru entry not in loaded library"


def test_runtime_render_summary_produces_markdown_table() -> None:
    """Wf-318F acceptance: ``render_runtime_summary`` produces markdown."""
    rt = F2MetalLigandExchangeRuntime()
    summary = render_runtime_summary(rt)
    assert "| metric | value |" in summary
    assert "| n_patterns_loaded |" in summary
    assert "| n_metal_symbols |" in summary
    assert str(rt.n_patterns_loaded) in summary


# ===========================================================================
# Section 2 — per-metal coverage (3 tests).
# ===========================================================================


@pytest.mark.parametrize(
    "metal",
    ["Ru", "Ir", "Os", "Rh", "Re"],
)
def test_runtime_per_metal_dispatch_matches_nh3(metal: str) -> None:
    """Wf-318F acceptance: Ru/Ir/Os/Rh/Re dispatch against NH3 donor.

    For each metal symbol, the runtime must return ``True`` when
    queried with the canonical NH3 donor ``[NH3]``. The matrix
    entries (``Ru_II_Cl_NH3`` etc.) + the 82-metal bare entries
    (``"Ru"`` / ``"Ir"`` etc.) jointly cover the metal.
    """
    rt = F2MetalLigandExchangeRuntime()
    matched = rt.dispatch(metal, "[NH3]")
    assert matched is True, (
        f"dispatch({metal!r}, '[NH3]') returned False — "
        f"metal {metal} should match at least one F2(a) pattern"
    )


def test_runtime_integration_point_returns_structured_result() -> None:
    """Wf-318F acceptance: integration_point returns F2DispatchResult."""
    rt = F2MetalLigandExchangeRuntime()
    result = rt.integration_point("Pt", "[NH3]")
    assert isinstance(result, F2DispatchResult)
    assert result.matched is True
    assert result.pattern_name is not None
    assert result.n_patterns_evaluated >= 1


def test_runtime_covers_at_least_60_metal_symbols() -> None:
    """Wf-318F acceptance: runtime exposes ≥ 60 distinct metal symbols.

    Per :func:`register_metal_ligand_exchange_rules` (101 matrix+NEW
    patterns) + the 64-metal filtered 82-metal library, the dedup'd
    dispatcher covers at least 60 distinct metal symbols (some overlap).
    """
    rt = F2MetalLigandExchangeRuntime()
    assert (
        rt.n_metal_symbols() >= 60
    ), f"expected ≥ 60 distinct metal symbols; got {rt.n_metal_symbols()}"


# ===========================================================================
# Section 3 — integration with pt_metal_ligand_exchange (2 tests).
# ===========================================================================


def test_runtime_loads_all_curated_f2a_entries() -> None:
    """Wf-318F acceptance: curated ``METAL_LIGAND_EXCHANGE_SMARTS`` are in runtime.

    Direct spot-check that the runtime surfaces every curated F2(a)
    entry from :data:`METAL_LIGAND_EXCHANGE_SMARTS`.
    """
    rt = F2MetalLigandExchangeRuntime()
    curated_names = {p[0] for p in METAL_LIGAND_EXCHANGE_SMARTS}
    loaded = set(rt.list_loaded_pattern_names())
    missing = curated_names - loaded
    assert not missing, f"runtime is missing curated F2(a) entries: {missing}"


def test_runtime_count_matches_register_helper_sum() -> None:
    """Wf-318F acceptance: runtime + register helper agree on the library.

    :func:`register_metal_ligand_exchange_rules` returns 101 patterns
    (84 matrix + 17 NEW SMARTS); every one must appear in the runtime's
    patterns list.
    """
    rt = F2MetalLigandExchangeRuntime()
    registered = register_metal_ligand_exchange_rules()
    registered_names = {p[0] for p in registered}
    loaded_names = set(rt.list_loaded_pattern_names())
    missing = registered_names - loaded_names
    assert not missing, f"runtime is missing entries from register_metal_ligand_exchange_rules(): {missing}"


# ===========================================================================
# Section 4 — edge cases (3 tests).
# ===========================================================================


def test_runtime_invalid_smiles_returns_false() -> None:
    """Wf-318F acceptance: bogus SMILES returns ``False`` (RDKit parse fails)."""
    rt = F2MetalLigandExchangeRuntime()
    assert rt.dispatch("Pt", "bogus_smiles") is False


def test_runtime_empty_metal_returns_false() -> None:
    """Wf-318F acceptance: empty metal symbol returns ``False`` (no pattern)."""
    rt = F2MetalLigandExchangeRuntime()
    assert rt.dispatch("", "[NH3]") is False


def test_runtime_empty_smiles_returns_false() -> None:
    """Wf-318F acceptance: empty SMILES returns ``False`` (RDKit can't parse)."""
    rt = F2MetalLigandExchangeRuntime()
    assert rt.dispatch("Pt", "") is False
