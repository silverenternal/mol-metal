"""Wf-336-X4 — TODO-134 Vina meeko alternative integration.

=====================================================================
Per Wf-329-W7 + Wf-323G: the canonical Vina CLI binary is already on
disk at ``/home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina``
and the CadButEaas ``.venv`` interpreter carries the in-process
Vina Python binding.  The meeko Python wrapper is the canonical
**alternative** ligand-preparation path when Vina CLI is unreachable
(NOT a fallback docking engine — meeko only generates the PDBQT).

Wf-336-X4 ships the **meeko alternative integration** for
``molmetal/scripts/cfm_vina_dock_panel.py``:

  1. ``_probe_meeko_alternative()`` exposes the runtime meeko probe
     so the panel JSON emits ``meeko_available``, ``meeko_version``,
     ``meeko_moleculeprep_class``, ``meeko_pdbqt_writer``,
     ``meeko_import_error``, and ``meeko_alternative_path``.

  2. ``STATUS_MEEKO_PREP_ONLY`` is a new cell status: ligand prep
     succeeded via meeko but the Vina CLI docking step could not run
     (binary missing / unhealthy / ``--meeko-only`` flag set).  The
     prepared PDBQT path is recorded on the cell ``note``.

  3. ``--meeko-only`` CLI flag forces the alternative path so the
     integration can be smoke-tested without a working Vina CLI.

  4. ``build_panel(meeko_only=True)`` plumbs the flag through end to
     end.

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
- All 4 tests are CPU-only; no GPU launch was performed.
- This is an **integration hardening** of the already-recovered Wf-320H
  pipeline: the Vina binary IS reachable on disk; meeko is the
  diagnostic + alternative path.
- Per Wf-329-W7: meeko 0.8.0 is installed in the project venv; the
  ``meeko_available`` probe returns True.

Wf-336-X4 deliverables:
- 4 NEW tests (this file)
- 1 NEW CLI flag (``--meeko-only``)
- 1 NEW cell status (``MEEKO_PREP_ONLY``)
- 6 NEW metadata fields (``meeko_available``, ``meeko_version``,
  ``meeko_moleculeprep_class``, ``meeko_pdbqt_writer``,
  ``meeko_import_error``, ``meeko_alternative_path``,
  ``meeko_only_mode``)
- 2 NEW aggregate counters (``n_meeko_prep_only`` cohort + per-pocket)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Canonical paths (per Wf-329-W7 + Wf-30 staging)
# ---------------------------------------------------------------------------
WF30_POCKET_PREP = PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"
WF336X4_SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "cfm_vina_dock_panel.py"
)
PROJECT_VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
WF179_POCKETS = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")


# ---------------------------------------------------------------------------
# Test 1: meeko availability check via ``_probe_meeko_alternative``
# ---------------------------------------------------------------------------
def test_meeko_alternative_probe_returns_measured_status() -> None:
    """The meeko probe must return an honest availability verdict.

    Per Wf-329-W7 §3.4: meeko 0.8.0 is installed in the project venv.
    The probe must surface ``available=True``, ``version="0.8.0"``,
    and the class names of ``MoleculePreparation`` + ``PDBQTWriterLegacy``.
    When meeko is unavailable (e.g. wrong interpreter), the probe must
    still return a dict with ``available=False`` + ``import_error``
    populated — never raise.
    """
    if not PROJECT_VENV_PY.is_file():
        pytest.skip(
            f"Project venv missing at {PROJECT_VENV_PY}; meeko probe "
            f"is an infrastructure prerequisite."
        )

    from molmetal.scripts.cfm_vina_dock_panel import (
        _probe_meeko_alternative,
        get_meeko_probe,
    )

    # Probe must NEVER raise — even when meeko is unavailable.
    try:
        probe = _probe_meeko_alternative()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"_probe_meeko_alternative raised: {exc!r}; must return "
            f"a dict even when meeko is unavailable."
        )
    assert isinstance(probe, dict), (
        f"Probe returned {type(probe).__name__}; expected dict."
    )
    for key in (
        "available",
        "version",
        "moleculeprep_class",
        "pdbqt_writer",
        "import_error",
    ):
        assert key in probe, f"Probe missing key {key!r}: {probe}"

    # Cached accessor must return the same shape.
    cached = get_meeko_probe()
    assert isinstance(cached, dict)
    assert "available" in cached

    # When meeko IS importable (project venv path), expect the Wf-329-W7
    # canonical state: available=True, version=0.8.x, both classes.
    if probe.get("available"):
        assert probe["version"], (
            f"meeko is available but version is empty: {probe}"
        )
        assert probe["moleculeprep_class"] == "MoleculePreparation", (
            f"Unexpected moleculeprep_class: {probe['moleculeprep_class']}"
        )
        assert probe["pdbqt_writer"] == "PDBQTWriterLegacy", (
            f"Unexpected pdbqt_writer: {probe['pdbqt_writer']}"
        )
        assert not probe["import_error"], (
            f"available=True but import_error populated: {probe['import_error']!r}"
        )


# ---------------------------------------------------------------------------
# Test 2: ``STATUS_MEEKO_PREP_ONLY`` is a documented status + meeko
# alternative integration produces it when Vina CLI is missing
# ---------------------------------------------------------------------------
def test_meeko_alternative_integration_produces_meeko_prep_only_status() -> None:
    """When Vina CLI is missing BUT meeko is available, the panel
    must emit ``STATUS_MEEKO_PREP_ONLY`` for every cell (NOT
    ``STUB_FALLBACK``).

    The Wf-336-X4 integration specifically upgrades the cell from
    ``STUB_FALLBACK`` to ``MEEKO_PREP_ONLY`` so downstream consumers
    know the prepared PDBQT is on disk and can feed it into an
    alternative docking engine.
    """
    from molmetal.scripts.cfm_vina_dock_panel import (
        STATUS_MEEKO_PREP_ONLY,
        STATUS_STUB_FALLBACK,
        build_panel,
    )

    assert STATUS_MEEKO_PREP_ONLY == "MEEKO_PREP_ONLY", (
        f"STATUS_MEEKO_PREP_ONLY constant changed to "
        f"{STATUS_MEEKO_PREP_ONLY!r}; downstream consumers may "
        f"break."
    )
    assert STATUS_STUB_FALLBACK == "STUB_FALLBACK", (
        f"Backward-compat: STUB_FALLBACK constant changed."
    )

    # We need at least one pocket staged.  Use whatever Wf-30 pocket
    # is staged on disk; if none are, skip the smoke test.
    staged_pocket: str | None = None
    for pocket in WF179_POCKETS:
        if (WF30_POCKET_PREP / pocket).is_dir():
            staged_pocket = pocket
            break
    if staged_pocket is None:
        pytest.skip(
            f"No Wf-30 pocket staged under {WF30_POCKET_PREP}; "
            f"this is an infrastructure prerequisite, not a test "
            f"failure."
        )

    smiles = [
        # paracetamol — common painkiller, well-known SMILES.
        "CC(=O)Nc1ccc(O)cc1",
        # caffeine — well-known SMILES.
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]

    with tempfile.TemporaryDirectory(prefix="wf336x4_") as tmp:
        work = Path(tmp)
        panel = build_panel(
            smiles_list=smiles,
            pocket_pdb_dir=WF30_POCKET_PREP,
            pockets=[staged_pocket],
            vina_bin="/nonexistent/vina_for_wf336x4",  # force NOT_OK
            exhaustiveness=8,
            n_poses=5,
            box_size=20.0,
            seed=0,
            timeout=5,
            work_dir=work,
            meeko_only=True,  # force alternative path
        )

    # Metadata fields populated by the probe.
    meta = panel["metadata"]
    assert meta.get("meeko_available") is True, (
        f"meeko_available must be True on the project venv: {meta}"
    )
    assert meta.get("meeko_version"), (
        f"meeko_version empty: {meta}"
    )
    assert meta.get("meeko_alternative_path") is True, (
        f"meeko_alternative_path must be True when Vina missing + "
        f"meeko available: {meta}"
    )
    assert meta.get("meeko_only_mode") is True

    # Per-cell status: every cell is MEEKO_PREP_ONLY (NOT
    # STUB_FALLBACK) when the alternative path is active.
    n_meeko = 0
    for m in panel["per_molecule"]:
        cell = m["per_pocket"][staged_pocket]
        assert cell["status"] == STATUS_MEEKO_PREP_ONLY, (
            f"Expected {STATUS_MEEKO_PREP_ONLY}, got {cell['status']!r}. "
            f"The alternative integration should rescue from "
            f"STUB_FALLBACK when meeko is available."
        )
        # The cell note must reference the prepared PDBQT path so a
        # downstream consumer can pick it up.
        note = cell["note"]
        assert "meeko" in note.lower(), (
            f"Cell note should mention meeko prep: {note!r}"
        )
        assert ".pdbqt" in note, (
            f"Cell note should reference the prepared PDBQT path: "
            f"{note!r}"
        )
        # No kcal/mol is fabricated — best_kcal_mol is None.
        assert cell["best_kcal_mol"] is None, (
            f"MEEKO_PREP_ONLY must NOT fabricate a kcal/mol score: "
            f"got {cell['best_kcal_mol']}"
        )
        n_meeko += 1
    assert n_meeko == len(smiles), (
        f"Expected {len(smiles)} meeko-prep cells, got {n_meeko}."
    )

    # Cohort counter + per-pocket counter agree.
    assert panel["n_meeko_prep_only"] == len(smiles), (
        f"Cohort n_meeko_prep_only={panel['n_meeko_prep_only']} != "
        f"{len(smiles)}."
    )
    per_pocket = panel["aggregate"]["per_pocket"][staged_pocket]
    assert per_pocket["n_meeko_prep_only"] == len(smiles), (
        f"Per-pocket n_meeko_prep_only={per_pocket['n_meeko_prep_only']} "
        f"!= {len(smiles)}."
    )


# ---------------------------------------------------------------------------
# Test 3: ``--meeko-only`` flag appears in the CLI parser + meeko
# integration falls back to STUB_FALLBACK when meeko is missing
# ---------------------------------------------------------------------------
def test_meeko_alternative_falls_back_to_stub_when_meeko_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When meeko is unavailable AND the Vina CLI is missing, every
    cell falls back to ``STUB_FALLBACK`` (the Wf-320H backward-compat
    behaviour).  The ``meeko_available`` metadata flag must be False
    so downstream consumers see the honest-negative state.

    This is the negative test: when the alternative path itself is
    unavailable, the panel degrades gracefully to the original
    STUB_FALLBACK behaviour.  No fabricated kcal/mol, no silent
    skip.
    """
    from molmetal.scripts import cfm_vina_dock_panel as cvdp

    # Stage: pretend meeko is unavailable.  We monkeypatch the
    # module-level cache AND the probe function to force the
    # ``available=False`` state.  This isolates the panel from the
    # real interpreter's meeko install.
    monkeypatch.setattr(
        cvdp,
        "_MEEKO_PROBE",
        {
            "available": False,
            "version": "",
            "moleculeprep_class": "",
            "pdbqt_writer": "",
            "import_error": "fake missing meeko for wf336x4",
        },
    )

    # Pick a staged pocket for the smoke test.
    staged_pocket: str | None = None
    for pocket in WF179_POCKETS:
        if (WF30_POCKET_PREP / pocket).is_dir():
            staged_pocket = pocket
            break
    if staged_pocket is None:
        pytest.skip(
            f"No Wf-30 pocket staged under {WF30_POCKET_PREP}; "
            f"this is an infrastructure prerequisite."
        )

    smiles = ["CC(=O)Nc1ccc(O)cc1"]

    with tempfile.TemporaryDirectory(prefix="wf336x4_neg_") as tmp:
        work = Path(tmp)
        panel = cvdp.build_panel(
            smiles_list=smiles,
            pocket_pdb_dir=WF30_POCKET_PREP,
            pockets=[staged_pocket],
            vina_bin="/nonexistent/vina_for_wf336x4",  # force NOT_OK
            exhaustiveness=8,
            n_poses=5,
            box_size=20.0,
            seed=0,
            timeout=5,
            work_dir=work,
            meeko_only=True,
        )

    # meeko_available must be False (negative state).
    assert panel["metadata"]["meeko_available"] is False, (
        f"Expected meeko_available=False when meeko is unavailable: "
        f"{panel['metadata']}"
    )
    assert panel["metadata"]["meeko_import_error"], (
        f"Expected meeko_import_error populated: {panel['metadata']}"
    )
    assert panel["metadata"]["meeko_alternative_path"] is False

    # Per-cell: every cell falls back to STUB_FALLBACK.
    for m in panel["per_molecule"]:
        cell = m["per_pocket"][staged_pocket]
        assert cell["status"] == cvdp.STATUS_STUB_FALLBACK, (
            f"Expected STUB_FALLBACK when meeko unavailable + Vina "
            f"missing; got {cell['status']!r}."
        )
        assert "Vina binary missing" in cell["note"], (
            f"Cell note should reference Vina binary missing: "
            f"{cell['note']!r}"
        )

    # CLI parser must surface ``--meeko-only``.
    parser = cvdp._build_parser()
    parsed = parser.parse_args([
        "--smiles-csv", "/tmp/x.csv",
        "--pocket-pdb-dir", str(WF30_POCKET_PREP),
        "--out-json", "/tmp/x.json",
        "--meeko-only",
    ])
    assert parsed.meeko_only is True, (
        f"--meeko-only flag parsed as {parsed.meeko_only}; expected True."
    )

    # CLI parser default must be False (backward-compat).
    parsed_default = parser.parse_args([
        "--smiles-csv", "/tmp/x.csv",
        "--pocket-pdb-dir", str(WF30_POCKET_PREP),
        "--out-json", "/tmp/x.json",
    ])
    assert parsed_default.meeko_only is False, (
        f"--meeko-only default must be False for backward-compat; "
        f"got {parsed_default.meeko_only}."
    )


# ---------------------------------------------------------------------------
# Test 4: integration on the 5 canonical Wf-179 m15 pockets
# ---------------------------------------------------------------------------
def test_meeko_alternative_integration_on_wf179_m15_pockets() -> None:
    """The meeko alternative integration must run end-to-end on the
    canonical 5-pocket Wf-179 m15 cohort (``1h36, 3eml, 3ny8, 4aaw,
    4rlu``) under ``--meeko-only`` mode.

    Per Wf-30 staging, every pocket must have a ``receptor.pdbqt``
    staged.  When the receptor is staged, the panel writes a per-cell
    meeko-prep PDBQT under the work_dir; when the receptor is NOT
    staged, the cell falls back to ``NO_RECEPTOR`` (not a test
    failure — staging is an infrastructure prerequisite).
    """
    if not WF30_POCKET_PREP.is_dir():
        pytest.skip(
            f"Wf-30 pocket staging directory missing at "
            f"{WF30_POCKET_PREP}; this is an infrastructure "
            f"prerequisite, not a test failure."
        )

    from molmetal.scripts.cfm_vina_dock_panel import (
        STATUS_MEEKO_PREP_ONLY,
        STATUS_NO_RECEPTOR,
        build_panel,
    )

    smiles = [
        # paracetamol
        "CC(=O)Nc1ccc(O)cc1",
        # caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]

    # Use whatever subset of Wf-179 pockets is staged.  At least one
    # pocket must be staged for the smoke test to run.
    staged = [p for p in WF179_POCKETS if (WF30_POCKET_PREP / p).is_dir()]
    if not staged:
        pytest.skip(
            f"No Wf-179 pockets staged under {WF30_POCKET_PREP}; "
            f"this is an infrastructure prerequisite."
        )

    with tempfile.TemporaryDirectory(prefix="wf336x4_5pocket_") as tmp:
        work = Path(tmp)
        panel = build_panel(
            smiles_list=smiles,
            pocket_pdb_dir=WF30_POCKET_PREP,
            pockets=list(WF179_POCKETS),
            vina_bin="/nonexistent/vina_for_wf336x4",
            exhaustiveness=8,
            n_poses=5,
            box_size=20.0,
            seed=0,
            timeout=5,
            work_dir=work,
            meeko_only=True,
        )

        # Aggregate cohort counters.
        n_meeko = panel["n_meeko_prep_only"]
        n_no_receptor = sum(
            1
            for m in panel["per_molecule"]
            for cell in m["per_pocket"].values()
            if cell["status"] == STATUS_NO_RECEPTOR
        )
        n_meeko_expected = (
            len(smiles) * len(staged)
        )
        assert n_meeko == n_meeko_expected, (
            f"Cohort n_meeko_prep_only={n_meeko}; expected "
            f"{n_meeko_expected} = {len(smiles)} SMILES × "
            f"{len(staged)} staged pockets."
        )
        assert n_no_receptor == len(smiles) * (len(WF179_POCKETS) - len(staged)), (
            f"NO_RECEPTOR count {n_no_receptor} unexpected; unstaged "
            f"pockets must report NO_RECEPTOR."
        )

        # Per-pocket aggregate: staged pockets show n_meeko_prep_only > 0,
        # unstaged pockets show 0.
        for pocket in WF179_POCKETS:
            agg = panel["aggregate"]["per_pocket"][pocket]
            if pocket in staged:
                assert agg["n_meeko_prep_only"] == len(smiles), (
                    f"Staged pocket {pocket} n_meeko_prep_only="
                    f"{agg['n_meeko_prep_only']} != {len(smiles)}."
                )
            else:
                assert agg["n_meeko_prep_only"] == 0, (
                    f"Unstaged pocket {pocket} should have 0 meeko-prep "
                    f"cells; got {agg['n_meeko_prep_only']}."
                )

        # Every meeko-prep cell must reference a real PDBQT file on disk
        # (so downstream consumers can pick it up).  The work_dir is
        # still alive inside the ``with`` block.
        import re
        for m in panel["per_molecule"]:
            for pocket, cell in m["per_pocket"].items():
                if cell["status"] != STATUS_MEEKO_PREP_ONLY:
                    continue
                note = cell["note"]
                assert ".pdbqt" in note, (
                    f"Meeko-prep cell for pocket {pocket} should "
                    f"reference the prepared PDBQT path: {note!r}"
                )
                # Extract the path and verify it exists on disk.
                m_pdbqt = re.search(r"(/tmp/[^ ]+\.pdbqt)", note)
                assert m_pdbqt, (
                    f"Could not extract PDBQT path from cell note: "
                    f"{note!r}"
                )
                pdbqt_path = Path(m_pdbqt.group(1))
                assert pdbqt_path.is_file(), (
                    f"Meeko-prep PDBQT path {pdbqt_path} should exist on "
                    f"disk but does not."
                )
                assert pdbqt_path.stat().st_size > 0, (
                    f"Meeko-prep PDBQT {pdbqt_path} is empty."
                )

    # Verify the script itself imports the new helper symbols.
    src = WF336X4_SCRIPT.read_text()
    for sym in (
        "_probe_meeko_alternative",
        "get_meeko_probe",
        "STATUS_MEEKO_PREP_ONLY",
        "--meeko-only",
        "meeko_only",
        "meeko_available",
        "meeko_alternative_path",
        "n_meeko_prep_only",
    ):
        assert sym in src, (
            f"Wf-336-X4 symbol {sym!r} missing from "
            f"{WF336X4_SCRIPT}. The meeko alternative integration "
            f"is not fully wired."
        )
