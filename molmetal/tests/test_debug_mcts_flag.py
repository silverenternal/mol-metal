"""Tests for the TODO-IMPROVEMENT_BACKLOG B3 ``debug=True`` flag on
:class:`MCTSProofSearch` and its ``--debug-mcts`` CLI wrapper.

The test suite verifies:

1. **The flag exists** — :class:`MCTSProofSearch` accepts ``debug=True``
   in its constructor without raising.
2. **Per-step trace is printed** — running with ``debug=True`` writes
   lines to stdout that contain the PUCT formula components
   (``PUCT``, ``Q=``, ``P=``, ``N_child=``, ``sqrt(N_parent)=``,
   ``cpuct=``) and the final ``LEAF`` line with ``smi=``, ``rule=``,
   ``reward=``.
3. **Default (debug=False) is silent** — running without the flag
   produces zero PUCT-trace lines on stdout.
4. **CLI wrapper works** — the
   :func:`molmetal.scripts.debug_mcts.main` entrypoint accepts
   ``--debug-mcts`` and prints the expected per-step trace.
5. **Structured ``debug_trace`` is populated** — the dataclass field
   :attr:`MCTSProofSearch.debug_trace` is appended to each iteration
   when ``debug=True``.

Mirrors the mock-chemistry fixtures in
:file:`tests/test_proof_search_strengthened.py` so the suite runs on a
fresh checkout without RDKit / D-MPNN / DiffDock.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from typing import List

import pytest

from molmetal_lam.binding.types import BindingSite
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.reactions.beta_reductions import ReactionRule
from molmetal_lam.search_alg.proof_search import MCTSProofSearch


# ---------------------------------------------------------------------------
# Mock chemistry — same fixture as
# :file:`tests/test_proof_search_strengthened.py`.
# ---------------------------------------------------------------------------


class _IdentityRule(ReactionRule):
    """Trivial reaction: concatenates atoms of ``state`` and ``tile``."""

    def __init__(self, name: str = "identity") -> None:
        self.name = name

    def reduce(self, pair):  # type: ignore[override]
        state, tile = pair
        try:
            from molmetal_lam.molecules.closed_term import (
                MoleculeClosedTerm as _MCT,
            )
            from molmetal_lam.bonds.application import FreeSiteLedger

            new_atoms = list(state.atoms) + list(tile.atoms)
            return [
                _MCT(
                    atoms=new_atoms,
                    bonds=list(state.bonds) + list(tile.bonds),
                    ledger=FreeSiteLedger(),
                    source_smiles=None,
                )
            ]
        except Exception:
            return []

    def __repr__(self) -> str:  # pragma: no cover
        return f"_IdentityRule({self.name!r})"


def _tiles() -> List[MoleculeClosedTerm]:
    return [
        MoleculeClosedTerm.from_smiles(smi) for smi in ("CCO", "CCN", "CCCl", "CCS")
    ]


def _binding_site() -> BindingSite:
    return BindingSite(name="debug_site", constraints=[], geometry_hints={})


def _rules() -> dict:
    return {"identity": _IdentityRule()}


def _build_search(debug: bool = True) -> MCTSProofSearch:
    return MCTSProofSearch(
        tile_library=_tiles(),
        rules=_rules(),
        target_predicates=[],
        binding_site=_binding_site(),
        scorer=lambda s: float(s.n_atoms) / 10.0,
        n_simulations=10,
        c_puct=1.4,
        top_k=3,
        rng=random.Random(0),
        dirichlet_alpha=0.0,  # disable noise for deterministic trace
        early_stop=False,  # honour the full budget
        debug=bool(debug),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDebugMCTSFlag:
    """TODO-IMPROVEMENT_BACKLOG B3 — verify the ``debug=True`` flag and
    its per-step PUCT trace output."""

    def test_debug_field_default_false(self) -> None:
        """Default :attr:`MCTSProofSearch.debug` is ``False`` — preserves
        bit-for-bit silence on stdout in production."""
        mcts = MCTSProofSearch(
            tile_library=_tiles(),
            rules=_rules(),
            target_predicates=[],
            binding_site=_binding_site(),
            scorer=lambda s: 0.5,
            n_simulations=1,
            rng=random.Random(0),
        )
        assert mcts.debug is False
        assert mcts.debug_trace == []

    def test_debug_field_accepted_in_constructor(self) -> None:
        """``debug=True`` is a valid kwarg and survives construction."""
        mcts = _build_search(debug=True)
        assert mcts.debug is True
        # debug_trace is a list (default_factory=list)
        assert isinstance(mcts.debug_trace, list)

    def test_run_with_debug_true_emits_puct_lines(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Running with ``debug=True`` writes per-step PUCT lines to
        stdout containing ``PUCT``, ``Q=``, and ``P=`` substrings."""
        mcts = _build_search(debug=True)
        initial = MoleculeClosedTerm.from_smiles("CCO")
        mcts.search(initial, max_depth=2)
        captured = capsys.readouterr().out
        # Must contain at least one PUCT line per PUCT selection, plus
        # one LEAF summary line per iteration.
        assert "PUCT" in captured, f"expected 'PUCT' in stdout, got:\n{captured[:1000]}"
        assert "Q=" in captured, f"expected 'Q=' in stdout, got:\n{captured[:1000]}"
        assert "P=" in captured, f"expected 'P=' in stdout, got:\n{captured[:1000]}"
        assert (
            "LEAF" in captured
        ), f"expected 'LEAF' summary line, got:\n{captured[:1000]}"

    def test_run_with_debug_false_emits_no_puct_lines(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Default (debug=False) is silent — no PUCT trace on stdout."""
        mcts = _build_search(debug=False)
        initial = MoleculeClosedTerm.from_smiles("CCO")
        mcts.search(initial, max_depth=2)
        captured = capsys.readouterr().out
        # The wrapper prints "[step=...]" only when debug=True.
        assert (
            "PUCT" not in captured
        ), "debug=False must not emit PUCT lines; got:\n{captured}"
        assert (
            "LEAF" not in captured
        ), "debug=False must not emit LEAF summary; got:\n{captured}"

    def test_debug_trace_populated_after_run(self) -> None:
        """The structured :attr:`debug_trace` list is populated once
        per iteration when ``debug=True``."""
        mcts = _build_search(debug=True)
        initial = MoleculeClosedTerm.from_smiles("CCO")
        mcts.search(initial, max_depth=2)
        # 10 simulations × 1 entry each.
        assert len(mcts.debug_trace) == 10
        # Each entry has the expected keys.
        for entry in mcts.debug_trace:
            assert "step" in entry
            assert "path_len" in entry
            assert "leaf_smiles" in entry
            assert "best_action" in entry
            assert "reward" in entry
            assert "puct_components" in entry
            assert isinstance(entry["puct_components"], list)
            # Per-depth PUCT components must have the documented fields.
            for puct in entry["puct_components"]:
                assert "depth" in puct
                assert "Q" in puct
                assert "P" in puct
                assert "N_child" in puct
                assert "sqrt_N_parent" in puct
                assert "cpuct" in puct
                assert "U" in puct
                assert "parent_N" in puct
                assert "child_smiles" in puct

    def test_debug_trace_is_reset_per_search_call(self) -> None:
        """Re-calling :meth:`search` resets the debug_trace buffer so a
        previous run's entries do not leak into the new run."""
        mcts = _build_search(debug=True)
        initial = MoleculeClosedTerm.from_smiles("CCO")
        mcts.search(initial, max_depth=2)
        first_len = len(mcts.debug_trace)
        assert first_len == 10
        mcts.search(initial, max_depth=2)
        # Re-run must produce the same length (10) — not 20.
        assert len(mcts.debug_trace) == 10

    def test_run_10_iterations_step_lines_have_required_substrings(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Canonical requirement from B3: 10 iterations, capture
        stdout, assert step lines contain ``PUCT``, ``Q=``, ``P=``
        substrings."""
        mcts = _build_search(debug=True)
        initial = MoleculeClosedTerm.from_smiles("CCO")
        # 10 iterations.
        mcts.search(initial, max_depth=3)
        captured = capsys.readouterr().out
        # Assert required substrings — explicit per B3 spec.
        for needle in ("PUCT", "Q=", "P="):
            assert (
                needle in captured
            ), f"required substring {needle!r} not found in stdout"

    def test_puct_lines_have_all_components(self) -> None:
        """Each PUCT line contains every component from the formula:
        ``Q=``, ``P=``, ``N_child=``, ``sqrt(N_parent)=``, ``cpuct=``."""
        import io as _io
        import contextlib as _cl

        buf = _io.StringIO()
        mcts = _build_search(debug=True)
        initial = MoleculeClosedTerm.from_smiles("CCO")
        with _cl.redirect_stdout(buf):
            mcts.search(initial, max_depth=2)
        out = buf.getvalue()
        # Pick out the per-depth PUCT lines (not the LEAF summary).
        puct_lines = [
            line for line in out.splitlines() if "PUCT" in line and "LEAF" not in line
        ]
        assert puct_lines, "no PUCT lines were emitted"
        for line in puct_lines:
            for needle in (
                "Q=",
                "P=",
                "N_child=",
                "sqrt(N_parent)=",
                "cpuct=",
                "U=",
                "child=",
            ):
                assert needle in line, f"missing {needle!r} in PUCT line: {line!r}"

    def test_leaf_line_has_smiles_rule_reward(self) -> None:
        """The ``LEAF`` summary line exposes ``smi=``, ``rule=``,
        ``reward=``, ``path_len=`` fields."""
        import io as _io
        import contextlib as _cl

        buf = _io.StringIO()
        mcts = _build_search(debug=True)
        initial = MoleculeClosedTerm.from_smiles("CCO")
        with _cl.redirect_stdout(buf):
            mcts.search(initial, max_depth=2)
        out = buf.getvalue()
        leaf_lines = [line for line in out.splitlines() if "LEAF" in line]
        assert len(leaf_lines) == 10  # one per iteration
        for line in leaf_lines:
            for needle in ("smi=", "rule=", "reward=", "path_len="):
                assert needle in line, f"missing {needle!r} in LEAF line: {line!r}"


class TestDebugMCTSCLI:
    """End-to-end CLI test: ``python -m molmetal.scripts.debug_mcts``
    with ``--debug-mcts`` produces the same PUCT trace."""

    def test_cli_emits_puct_lines(self, tmp_path) -> None:
        """Invoke the wrapper script and assert the PUCT trace is
        emitted."""
        # Find the project root (one level above this test file).
        test_path = os.path.abspath(__file__)
        # .../molmetal/tests/test_debug_mcts_flag.py
        # .../molmetal/tests/  = dirname(test_path)
        # .../molmetal/        = dirname(dirname(test_path))
        # .../mol-metal/       = dirname(dirname(dirname(test_path)))
        tests_dir = os.path.dirname(test_path)
        molmetal_pkg = os.path.dirname(tests_dir)  # molmetal/
        project_root = os.path.dirname(molmetal_pkg)  # mol-metal/
        script_path = os.path.join(
            molmetal_pkg,
            "scripts",
            "debug_mcts.py",
        )
        assert os.path.exists(script_path), f"debug_mcts.py not found at {script_path}"
        # Run the script as a module so the relative imports resolve.
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "molmetal.scripts.debug_mcts",
                "--seed",
                "0",
                "--depth",
                "2",
                "--sims",
                "5",
                "--debug-mcts",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"CLI exited non-zero: rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        out = result.stdout
        for needle in ("PUCT", "Q=", "P=", "LEAF"):
            assert needle in out, (
                f"required substring {needle!r} not found in CLI "
                f"stdout:\n{out[:2000]}"
            )

    def test_cli_without_debug_flag_is_silent(self, tmp_path) -> None:
        """Without ``--debug-mcts`` the CLI is silent on the PUCT
        trace (matches the production contract)."""
        test_path = os.path.abspath(__file__)
        tests_dir = os.path.dirname(test_path)
        molmetal_pkg = os.path.dirname(tests_dir)
        project_root = os.path.dirname(molmetal_pkg)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "molmetal.scripts.debug_mcts",
                "--seed",
                "0",
                "--depth",
                "2",
                "--sims",
                "5",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"CLI exited non-zero: rc={result.returncode}\n" f"stderr:\n{result.stderr}"
        )
        out = result.stdout
        # No PUCT trace on stdout when --debug-mcts is omitted.
        assert "PUCT" not in out
        assert "LEAF" not in out
        # But with --print-summary, we'd still see the summary line —
        # not requested here, so absence is expected.
