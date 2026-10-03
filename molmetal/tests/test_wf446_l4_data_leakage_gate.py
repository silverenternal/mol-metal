"""Wf-446 L4 — data-leakage diagnostic gate tests (``p2_data_leakage_diag``).

CPU-only, no network, no GPU.  Covers the L4 fix: ``check_data_leakage.py``
was unwired; the gate ``gate_p2_data_leakage_diag`` now runs it with
explicit inputs and requires scaffold-split strictness for both datasets +
shipped-split strictness for metalcytotox v4 (random-split overlap is the
expected honest-negative, recorded but not gated).

1. FAIL on a synthetic shipped-split leak (constructed bad input), PASS on
   the same synthetic pair without the leak.
2. FAIL when the metalcytotox CSV loses its ``split`` column.
3. FAIL on missing script / missing tracked platinai CSV; SKIP (not FAIL)
   when the gitignored ``.data/**`` v4 CSV is absent.
4. Registration contract: exactly one ``p2_data_leakage_diag`` entry, in
   the ``all`` pipeline only, count consistent with EXPECTED_GATE_COUNTS.
5. (integration, ~22 s) the healthy tree PASSes with the honest-negative
   random-split markers in evidence.

See TODO/completed/h4_data_leakage_gate.md for the ship surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molmetal.scripts.pre_flight_check import (
    DATA_DIR,
    DOT_DATA_DIR,
    SCRIPT_DIR,
    build_gates,
    gate_p2_data_leakage_diag,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATINAI_CSV = REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
CYTO_V4_CSV = REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"


def _write_platinai(path: Path) -> None:
    path.write_text(
        "smiles\nCCO\nCCN\nCCC\nCCCl\nCO\nCN\nCS\nCCBr\n",
        encoding="utf-8",
    )


def _write_cyto(path: Path, *, leak: bool = False, with_split: bool = True) -> None:
    rows = [
        ("CCO", "s1", "train"),
        ("CCN", "s2", "train"),
        ("CCC", "s3", "val"),
        ("CCCl", "s4", "test"),
        ("CO", "s5", "train"),
        ("CN", "s6", "val"),
        ("CS", "s7", "test"),
        # leak=True injects scaffold s1 into the TEST split (s1 also train)
        ("CCBr", "s1" if leak else "s8", "test"),
    ]
    if with_split:
        body = "smiles,scaffold_id,split\n" + "".join(
            f"{s},{sc},{sp}\n" for s, sc, sp in rows
        )
    else:
        body = "smiles,scaffold_id\n" + "".join(f"{s},{sc}\n" for s, sc, _sp in rows)
    path.write_text(body, encoding="utf-8")


def _run_gate(tmp_path: Path, *, leak: bool = False, with_split: bool = True):
    pytest.importorskip("numpy")
    plat = tmp_path / "platinai.csv"
    cyto = tmp_path / "cyto.csv"
    _write_platinai(plat)
    _write_cyto(cyto, leak=leak, with_split=with_split)
    return gate_p2_data_leakage_diag(
        "t",
        "P2",
        "data",
        SCRIPT_DIR / "check_data_leakage.py",
        platinai_csv=plat,
        metalcytotox_csv=cyto,
        out_path=tmp_path / "leak" / "leakage_check.json",
    )


class TestBothWaysSynthetic:
    def test_fails_on_injected_shipped_leak(self, tmp_path: Path) -> None:
        r = _run_gate(tmp_path, leak=True)
        assert r.status == "FAIL", r.evidence
        assert "shipped split NOT strict" in r.evidence

    def test_passes_on_clean_synthetic_pair(self, tmp_path: Path) -> None:
        r = _run_gate(tmp_path)
        assert r.status == "PASS", r.evidence
        assert "shipped strict=PASS" in r.evidence
        assert "HONEST-NEGATIVE" in r.evidence

    def test_fails_when_shipped_split_column_absent(self, tmp_path: Path) -> None:
        r = _run_gate(tmp_path, with_split=False)
        assert r.status == "FAIL", r.evidence
        assert "split column" in r.evidence or "absent from the report" in r.evidence


class TestEnvAndPathContract:
    def test_fails_on_missing_script(self, tmp_path: Path) -> None:
        r = gate_p2_data_leakage_diag(
            "t",
            "P2",
            "data",
            tmp_path / "nope.py",
            platinai_csv=PLATINAI_CSV,
            metalcytotox_csv=CYTO_V4_CSV,
            out_path=tmp_path / "leak.json",
        )
        assert r.status == "FAIL", r.evidence
        assert "check_data_leakage.py missing" in r.evidence

    def test_fails_on_missing_platinai_csv(self, tmp_path: Path) -> None:
        r = gate_p2_data_leakage_diag(
            "t",
            "P2",
            "data",
            SCRIPT_DIR / "check_data_leakage.py",
            platinai_csv=tmp_path / "nope.csv",
            metalcytotox_csv=CYTO_V4_CSV,
            out_path=tmp_path / "leak.json",
        )
        assert r.status == "FAIL", r.evidence
        assert "platinai CSV missing" in r.evidence

    def test_skips_when_gitignored_v4_absent(self, tmp_path: Path) -> None:
        r = gate_p2_data_leakage_diag(
            "t",
            "P2",
            "data",
            SCRIPT_DIR / "check_data_leakage.py",
            platinai_csv=PLATINAI_CSV,
            metalcytotox_csv=tmp_path / "nope.csv",
            out_path=tmp_path / "leak.json",
        )
        assert r.status == "SKIP", r.evidence
        assert ".data/** is gitignored" in r.evidence


class TestRegistration:
    def test_registered_all_only_and_count_consistent(self) -> None:
        names_all = [n for n, _ in build_gates("all")]
        assert names_all.count("p2_data_leakage_diag") == 1
        for p in ("frontier", "chemprop_v2_v3", "cfm"):
            assert "p2_data_leakage_diag" not in [
                n for n, _ in build_gates(p)
            ], f"gate must be 'all'-only, found in {p!r}"


class TestHealthyTreeIntegration:
    def test_healthy_tree_passes_with_honest_negative(self, tmp_path: Path) -> None:
        """Integration (~22 s): the REAL v4 + platinai inputs PASS and the
        expected random-split overlap is reported as honest-negative.
        """
        pytest.importorskip("numpy")
        pytest.importorskip("rdkit")
        if not CYTO_V4_CSV.is_file():
            pytest.skip(f"v4 CSV not on disk (gitignored): {CYTO_V4_CSV}")
        r = gate_p2_data_leakage_diag(
            "t",
            "P2",
            "data",
            SCRIPT_DIR / "check_data_leakage.py",
            platinai_csv=PLATINAI_CSV,
            metalcytotox_csv=CYTO_V4_CSV,
            out_path=tmp_path / "leak.json",
        )
        assert r.status == "PASS", r.evidence
        assert "shipped strict=PASS" in r.evidence
        assert "HONEST-NEGATIVE" in r.evidence
        assert "1931 shared SMILES" in r.evidence
