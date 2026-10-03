"""T2-FIX — NCI-60 random-split fallback visibility tests.

The scaffold-OOD helper ``_scaffold_split_indices_nci60`` had a SILENT
``None`` fallback: at the default ``n_test_min=50`` all nine NCI-60 cells
hit it, so ``--split scaffold`` runs produced random 80/10/20 splits while
metadata said "scaffold". These tests pin the fix:

  * the fallback now emits BOTH a ``logging.WARNING`` and a
    ``RuntimeWarning`` (belt-and-braces);
  * the success path stays quiet;
  * ``write_split_manifest`` emits a machine-readable JSON manifest with
    ``path_taken`` per (cell, seed) so a gate can assert scaffold-OOD;
  * the real CSV regression guard: 9/9 cells return ``None`` (fallback)
    at ``n_test_min=50`` — if a future data/helper change starts
    "succeeding" there, this test fails loudly (deliberate tripwire).

The split construction itself is UNTOUCHED — the m21 byte-identical
comparability is preserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit.Chem.Scaffolds.MurckoScaffold")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import _nci60_scaffold_helpers as H  # noqa: E402

CSV_DEFAULT = PROJECT_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"

# Synthetic SMILES whose Bemis-Murcko groups are all size 1.
FALLBACK_SMILES = ["CCO", "c1ccccc1", "c1ccncc1", "CC(=O)O", "C1CCCCC1"]
# 15 copies of benzene (one group of 15) + 5 acyclics (group of 5).
SUCCESS_SMILES = ["c1ccccc1"] * 15 + ["CCO"] * 5


def _active(n: int) -> np.ndarray:
    return np.asarray([i % 2 for i in range(n)], dtype=int)


class TestFallbackIsLoud:
    def test_fallback_warns_on_both_channels(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger=H.__name__):
            with pytest.warns(RuntimeWarning, match="FALLBACK"):
                out = H._scaffold_split_indices_nci60(
                    smiles_subset=FALLBACK_SMILES,
                    active_subset=_active(len(FALLBACK_SMILES)),
                    seed=7,
                    n_test_min=10,
                )
        assert out is None
        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("FALLBACK" in m for m in msgs), msgs
        assert any("seed=7" in m and "n_test_min=10" in m for m in msgs), msgs
        assert any("NOT scaffold-OOD" in m for m in msgs), msgs

    def test_success_path_is_quiet(self, caplog, recwarn) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger=H.__name__):
            out = H._scaffold_split_indices_nci60(
                smiles_subset=SUCCESS_SMILES,
                active_subset=_active(len(SUCCESS_SMILES)),
                seed=0,
                n_test_min=10,
            )
        assert out is not None, "expected the size-15 benzene group to qualify"
        assert len(out[2]) >= 10
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestSplitManifest:
    def test_manifest_roundtrip_and_gate_assertion(self, tmp_path) -> None:
        p = tmp_path / "sub" / "x.split_manifest.json"
        records = [
            {
                "cell": "A549",
                "seed": 0,
                "path_taken": H.SPLIT_PATH_SCAFFOLD_OOD,
                "split_label": "scaffold_ood_seed0",
                "status": "ok",
                "n_train": 100,
                "n_val": 20,
                "n_test": 33,
            },
            {
                "cell": "DU145",
                "seed": 0,
                "path_taken": H.SPLIT_PATH_RANDOM_FALLBACK,
                "split_label": "random_80_10_20_fallback_no_scaffold_group",
                "status": "ok",
                "n_train": 128,
                "n_val": 16,
                "n_test": 17,
            },
        ]
        written = H.write_split_manifest(
            p, records, extra={"requested_split": "scaffold"}
        )
        assert written == p and p.exists()
        payload = json.loads(p.read_text())
        assert payload["schema"] == "nci60_split_manifest_v1"
        assert payload["n_records"] == 2
        assert payload["extra"] == {"requested_split": "scaffold"}
        assert [r["path_taken"] for r in payload["records"]] == [
            "scaffold_ood",
            "random_fallback",
        ]
        # The gate use-case: assert scaffold-OOD actually happened.
        assert (
            all(r["path_taken"] == "scaffold_ood" for r in payload["records"]) is False
        )

    def test_manifest_constants_exported(self) -> None:
        for name in (
            "SPLIT_PATH_SCAFFOLD_OOD",
            "SPLIT_PATH_RANDOM_FALLBACK",
            "SPLIT_PATH_RANDOM_TEMPORAL",
            "SPLIT_PATH_UNKNOWN",
        ):
            assert name in H.__all__, name
            assert isinstance(getattr(H, name), str)
        assert H.SPLIT_PATH_SCAFFOLD_OOD == "scaffold_ood"
        assert H.SPLIT_PATH_RANDOM_FALLBACK == "random_fallback"
        assert H.SPLIT_PATH_RANDOM_TEMPORAL == "random_temporal"
        assert H.SPLIT_PATH_UNKNOWN == "unknown"


@pytest.mark.skipif(not CSV_DEFAULT.exists(), reason=f"missing {CSV_DEFAULT}")
class TestRealCsvNineCellPaths:
    CELLS = [
        "A549",
        "MCF7",
        "A2780",
        "HCT116",
        "A2780cis",
        "PC3",
        "HL60",
        "SKOV3",
        "DU145",
    ]

    def test_all_nine_cells_fall_back_at_default_threshold(self, caplog) -> None:
        """MEASURED T2-FIX 2026-10-03: path@n_test_min=50 = random_fallback 9/9.

        This is the silent-random tripwire: these nine runs used to be
        labelled scaffold-OOD while actually being random 80/10/20.
        """
        import logging

        import pandas as pd

        df = pd.read_csv(CSV_DEFAULT, low_memory=False)
        paths = {}
        with caplog.at_level(logging.WARNING, logger=H.__name__):
            for cell in self.CELLS:
                sub = df[df["cell_line"] == cell].reset_index(drop=True)
                out = H._scaffold_split_indices_nci60(
                    smiles_subset=sub["smiles"].astype(str).tolist(),
                    active_subset=sub["active"].astype(int).to_numpy(),
                    seed=0,
                    n_test_min=50,
                )
                paths[cell] = (
                    H.SPLIT_PATH_RANDOM_FALLBACK
                    if out is None
                    else H.SPLIT_PATH_SCAFFOLD_OOD
                )
        assert paths == {c: H.SPLIT_PATH_RANDOM_FALLBACK for c in self.CELLS}, paths
        warns = [r.getMessage() for r in caplog.records if "FALLBACK" in r.getMessage()]
        assert len(warns) == 9, f"expected 9 fallback warnings, got {len(warns)}"


class TestDriverExecuteSweepVisibility:
    """T2-FIX caller check: the driver itself must state the path taken."""

    # 20 distinct-scaffold SMILES x 3 copies = 60 rows; max scaffold group 6.
    BASE = [
        "c1ccccc1",
        "c1ccncc1",
        "c1ccc2ccccc2c1",
        "C1CCCCC1",
        "C1CCNCC1",
        "C1CCOCC1",
        "CC(=O)O",
        "CC(=O)N",
        "CCS",
        "CCBr",
        "CCI",
        "CCF",
        "CCCl",
        "CCN",
        "CCC",
        "CCO",
        "c1ccsc1",
        "c1ccoc1",
        "c1cnc2ccccc2n1",
        "c1ccc(O)cc1",
    ]

    def test_execute_sweep_records_fallback_path_and_manifest(
        self, tmp_path, caplog
    ) -> None:
        import logging

        try:
            from molmetal.scripts import nci60_9cell_chemprop_driver as mod
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"nci60 driver import failed: {exc}")

        rows = ["smiles,cell_line,active"] + [
            f"{s},T2SMOKE,{i % 2}" for i, s in enumerate(self.BASE * 3)
        ]
        csv_path = tmp_path / "tiny.csv"
        csv_path.write_text("\n".join(rows) + "\n")
        out = tmp_path / "sweep.json"
        args = mod.build_argparser().parse_args(
            [
                "--csv",
                str(csv_path),
                "--cells",
                "T2SMOKE",
                "--split",
                "scaffold",
                "--n-test-min",
                "50",
                "--epochs",
                "1",
                "--batch-size",
                "16",
                "--device",
                "cpu",
                "--seeds",
                "0",
                "--time-budget-s",
                "600",
                "--out",
                str(out),
                "--execute",
            ]
        )
        with caplog.at_level(logging.WARNING):
            result = mod.execute_sweep(args)

        # Plan said scaffold; the outcome must now say random_fallback loudly.
        assert result["per_cell_split_choice"]["T2SMOKE"] == "scaffold"
        assert result["per_cell_split_actual"]["T2SMOKE"] == "random_fallback"
        assert [r["path_taken"] for r in result["per_seed_records"]] == [
            "random_fallback"
        ]
        assert result["per_cell_summary"]["T2SMOKE"]["path_taken"] == "random_fallback"
        msgs = [r.getMessage() for r in caplog.records]
        assert any("SPLIT FALLBACK" in m for m in msgs), msgs

        # Machine-readable trace + the gate use-case.
        man_path = Path(result["split_manifest_path"])
        assert man_path.exists(), man_path
        man = json.loads(man_path.read_text())
        assert man["schema"] == "nci60_split_manifest_v1"
        assert [r["path_taken"] for r in man["records"]] == ["random_fallback"]
        assert all(r["path_taken"] == "scaffold_ood" for r in man["records"]) is False
