# SPDX-License-Identifier: Apache-2.0
"""Red-check test: generated_aromatic_frac must skip the CSV header.

WHY THIS EXISTS
---------------
The QED-lever script ``/tmp/wf_blkc_qed_lever/wf_blkc_qed_lever.py`` scored
the per-complex aromatic bond fraction of the generated cohort by reading
the CFM generation CSV line-by-line and skipping only lines starting with
"smiles". The actual header is
``epoch,seed,smiles,sanitized,n_fragments,largest_fragment_frac,metal_z`` --
it starts with "epoch", not "smiles" -- so the header was NOT skipped and
the first data row was parsed as a SMILES of "10" (Chem.MolFromSmiles
returns None). The function silently returned ``n_molecules = 0`` and the
verdict in /tmp/wf_blkc_qed_lever/qed_lever.json was drawn from an empty
sample. Independent recompute against the same CSV (in
metrics/by_metric/cfm_objective_inventory.json) gives 0.001229 aromatic
bond fraction across 66 generated molecules.

This test pins the fix: the function MUST read the CSV with
csv.DictReader (or an equivalent that auto-detects the header), MUST
return ``n_molecules = 66`` against the production CSV, and MUST raise
on a CSV with no ``smiles`` column (rather than silently returning
n_molecules = 0).

RED-CHECK PROCEDURE
-------------------
::

    # expect GREEN
    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        /home/hugo/codes/mol-metal/molmetal/tests/test_wf_blkc_qed_lever_generated_frac.py -v

    # expect RED -- revert the bugfix (use line-based "smiles" skip)
    sed -i 's/csv.DictReader(fh)/fh.readlines()/' \\
        /tmp/wf_blkc_qed_lever/wf_blkc_qed_lever.py
    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        /home/hugo/codes/mol-metal/molmetal/tests/test_wf_blkc_qed_lever_generated_frac.py -v
    # restore from the bugfix
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path("/home/hugo/codes/mol-metal")
_SCRIPT = Path("/tmp/wf_blkc_qed_lever/wf_blkc_qed_lever.py")


def _load_script():
    """Import the script's ``generated_aromatic_frac`` function."""
    if not _SCRIPT.is_file():
        pytest.skip(f"qed_lever script not found at {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("wf_blkc_qed_lever", _SCRIPT)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not load {_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"qed_lever script failed to import: {exc!r}")
    return mod


def test_function_skips_header_in_cfm_generation_csv():
    """The production CSV's header is
    ``epoch,seed,smiles,sanitized,n_fragments,largest_fragment_frac,metal_z``.
    The fixed function must return n_molecules=66 (matching the CFM
    pipeline's emission), not 0. The pre-fix line-based skip
    ("startswith('smiles')") misses the "epoch" prefix and the first
    row's "10" parses to None, so n_molecules comes back 0.
    """
    mod = _load_script()
    csv_path = (
        _REPO_ROOT
        / "molmetal"
        / "reports"
        / "cfm_fullcorpus_2026_10_10"
        / "generated_smiles.csv"
    )
    if not csv_path.is_file():
        pytest.skip(f"production CSV not found: {csv_path}")
    out = mod.generated_aromatic_frac(csv_path)
    assert out["n_molecules"] == 66, (
        f"n_molecules={out['n_molecules']}, expected 66. If this is 0, "
        "the function is back to the pre-fix 'skip lines starting with "
        "smiles' behaviour and the CSV header (which starts with "
        "'epoch') is not being skipped."
    )
    # And the value should match the independent recompute in
    # metrics/by_metric/cfm_objective_inventory.json
    # (aromatic_bond_fraction_generated: 0.001229).
    assert out["mean_aromatic_bond_fraction"] == pytest.approx(0.001229, abs=1e-3), (
        f"mean_aromatic_bond_fraction={out['mean_aromatic_bond_fraction']}, "
        "expected ~0.001229 (the value independently recomputed in "
        "cfm_objective_inventory.json::D6)."
    )


def test_function_skips_header_with_fake_epoch_prefixed_csv(tmp_path):
    """Synthetic CSV: 5 data rows, header
    ``epoch,seed,smiles,sanitized`` (mirrors the CFM pipeline). The
    fixed function must read all 5 rows, not 0 or 1.
    """
    mod = _load_script()
    csv_path = tmp_path / "fake_generated.csv"
    # Header explicitly does NOT start with "smiles" (it starts with
    # "epoch") so the pre-fix line-based skip mis-fires.
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "seed", "smiles", "sanitized"])
        # 5 valid SMILES (benzene, pyridine, cyclohexane, naphthalene,
        # cyclopropane).  Aromatic-bond fractions will be 0.5, 0.4, 0,
        # 0.43, 0 respectively (rough); the exact numbers don't matter
        # for this test -- only n_molecules does.
        w.writerow(["10", "0", "c1ccccc1", "1"])
        w.writerow(["10", "0", "c1ccncc1", "1"])
        w.writerow(["10", "0", "C1CCCCC1", "1"])
        w.writerow(["10", "0", "c1ccc2ccccc2c1", "1"])
        w.writerow(["10", "0", "C1CC1", "1"])
    out = mod.generated_aromatic_frac(csv_path)
    assert out["n_molecules"] == 5, (
        f"n_molecules={out['n_molecules']}, expected 5. If this is 0, "
        "the pre-fix 'skip lines starting with smiles' behaviour has "
        "returned -- the 'epoch' header was not skipped."
    )
    assert out["n_with_any_aromatic"] >= 3, (
        f"n_with_any_aromatic={out['n_with_any_aromatic']}, expected >=3 "
        "(benzene, pyridine, naphthalene all have aromatic bonds)."
    )


def test_function_raises_on_csv_without_smiles_column(tmp_path):
    """If a CSV has no 'smiles' column at all, the function must raise
    -- not silently return n_molecules=0. The pre-fix behaviour was to
    fall through and return 0, which is exactly the bug that hid the
    n=66 result in /tmp/wf_blkc_qed_lever/qed_lever.json.
    """
    mod = _load_script()
    csv_path = tmp_path / "no_smiles_col.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "seed", "mol_id"])
        w.writerow(["10", "0", "12345"])
    with pytest.raises(ValueError, match="smiles"):
        mod.generated_aromatic_frac(csv_path)
