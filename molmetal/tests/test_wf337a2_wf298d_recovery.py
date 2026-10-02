"""Wf-337-A2 unit tests — Wf-264 sub-panel recovery paths.

Per Wf-334-U3: Wf-264 Vina sub-panel MEASURED 0/100 docked ok on 1h36
when SMILES contain Pt coordination centres — meeko 0.8.0 returns
``is_ok=False`` from ``PDBQTWriterLegacy.write_string`` because RDKit
UFF cannot parameterise Pt+2 (the meeko metal-atom limit).  This
test module verifies the Wf-337-A2 recovery module:
``molmetal/scripts/wf337a2_vina_recovery.py``.

4 NEW tests verify:

  1. ``test_smina_availability`` — Smina probe returns the expected
     dict shape; ``available=False`` in this environment (smina is
     NOT installed); honest-negative reason populated.
  2. ``test_quickvina_availability`` — QuickVina probe returns the
     expected dict shape; ``available=False`` in this environment
     (neither ``q.vina`` nor ``quickvina`` on PATH); honest-negative
     reason populated.
  3. ``test_custom_metal_loop_detector`` — ``detect_metal_loop``
     correctly identifies Pt coordination centres in cisplatin,
     ruthenium complexes, and iridium complexes; recommends
     ``pt_strip_largest_fragment`` for cisplatin (≥3 non-H
     neighbours) and ``pt_to_zn_substitution`` for a hypothetical
     dangling-Pt SMILES (≤2 non-H neighbours); non-metal SMILES
     return ``has_metal_loop=False``.
  4. ``test_pocket_sweep_diagnostic_5_wf325b_pockets`` — the
     pocket sweep diagnostic returns the correct per-pocket
     structure for the 5 Wf-325B pockets, with honest-negative
     recovery verdict recorded when neither Smina nor QuickVina
     is available (the canonical Wf-334-U3 MEASURED scenario).

All tests CPU-only.  No real Vina engine invoked.  No GPU launch
per AGENTS.md §5.1.

Honest framing (party-twenty-spirit §3.5):
Per Wf-334-U3 MEASURED: 0/100 docked ok on 1h36 due to meeko
metal-atom limit.  Recovery verdict when neither Smina nor
QuickVina is installed + SMILES has a Pt centre:
HONEST_NEG_METAL_LOOP_OK (the metal-loop detector IS the recovery;
the Wf-264 panel does not yet dispatch to it but the detector is
the documented next-step wire-up).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Local import of the Wf-337-A2 recovery module under test.
RECOVERY_SCRIPT = ROOT / "molmetal" / "scripts" / "wf337a2_vina_recovery.py"

# Canonical 5 Wf-325B pockets (per Wf-179 m15 + Wf-30 staging).
WF325B_POCKETS = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")

# Canonical Wf-325B test SMILES: cisplatin.
CISPLATIN_SMILES = "[Pt](N)(N)(Cl)Cl"

# Optional staging dir (used when receptor.pdbqt is available).
WF30_POCKET_DIR = ROOT / "molmetal" / "reports" / "wf30_pocket_prep"


# ---------------------------------------------------------------------------
# Helpers — import recovery module
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def recovery_mod():
    """Import the Wf-337-A2 recovery module under test.

    Skips the fixture if the module cannot be imported (e.g. RDKit
    missing in the test venv).
    """
    if not RECOVERY_SCRIPT.exists():
        pytest.skip(
            f"Recovery script not found at {RECOVERY_SCRIPT}"
        )
    # The module uses @dataclass, which requires the module to be
    # registered in ``sys.modules`` BEFORE exec_module runs (Python
    # 3.12 dataclasses look up cls.__module__ in sys.modules).  The
    # standard ``importlib.util.module_from_spec`` + exec_module
    # pattern fails on dataclasses — register explicitly.
    import importlib.util  # noqa: WPS433
    spec = importlib.util.spec_from_file_location(
        "wf337a2_vina_recovery", str(RECOVERY_SCRIPT)
    )
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot import recovery module from {RECOVERY_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass decorator needs this.
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(spec.name, None)
        pytest.skip(f"Recovery module import failed: {exc!r}")
    return mod


# ===========================================================================
# Test 1 — Smina availability probe
# ===========================================================================
def test_smina_availability(recovery_mod):
    """Smina probe returns the expected dict shape with the right
    honest-negative markers when smina is NOT installed.

    Wf-337-A2 §1: Smina is the Vina fork that supports custom atom
    types; when it is NOT installed, the probe must surface
    ``available=False`` + a populated ``honest_negative_reason`` so
    downstream consumers know the recovery branch is BLOCKED.
    """
    probe = recovery_mod.probe_smina_availability()

    # Dict shape contract.
    assert isinstance(probe, dict)
    for k in (
        "available", "binary_path", "version",
        "probe_error", "honest_negative_reason",
    ):
        assert k in probe, f"probe missing key: {k}"
    assert isinstance(probe["available"], bool)

    # In this environment, smina is NOT installed.  The probe must
    # honestly report that.  We do not assert False unconditionally
    # because a future test env with smina installed should also
    # work; we assert the dict contract + the truthful population
    # of the honest-negative fields when available=False.
    if not probe["available"]:
        assert probe["binary_path"] == ""
        assert probe["probe_error"] != ""
        assert probe["honest_negative_reason"] != ""
        # The honest-negative reason must mention "smina" (the
        # engine we tried to find) and a recovery hint.
        reason = probe["honest_negative_reason"].lower()
        assert "smina" in reason
        assert "install" in reason or "not available" in reason
    else:
        # Smina IS installed — sanity check the path is real.
        p = Path(probe["binary_path"])
        assert p.is_file()
        assert probe["version"] != ""


# ===========================================================================
# Test 2 — QuickVina availability probe
# ===========================================================================
def test_quickvina_availability(recovery_mod):
    """QuickVina probe returns the expected dict shape with the right
    honest-negative markers when neither ``q.vina`` nor ``quickvina``
    is installed.

    Wf-337-A2 §2: QuickVina is a faster Vina fork (Handoko et al.
    2012); it shares Vina's PDBQT-parser limitations so it does
    NOT solve the Pt UFF parameterisation problem.  The probe still
    surfaces availability so a downstream consumer can opt for the
    speed lift on organic SMILES.
    """
    probe = recovery_mod.probe_quickvina_availability()

    # Dict shape contract.
    assert isinstance(probe, dict)
    for k in (
        "available", "binary_path", "version",
        "probe_error", "honest_negative_reason",
    ):
        assert k in probe, f"probe missing key: {k}"
    assert isinstance(probe["available"], bool)

    if not probe["available"]:
        assert probe["binary_path"] == ""
        assert probe["probe_error"] != ""
        assert probe["honest_negative_reason"] != ""
        # The honest-negative reason must mention "quickvina" and a
        # recovery hint.
        reason = probe["honest_negative_reason"].lower()
        assert "quickvina" in reason or "q.vina" in reason
    else:
        p = Path(probe["binary_path"])
        assert p.is_file()
        assert probe["version"] != ""


# ===========================================================================
# Test 3 — Custom metal-loop detector
# ===========================================================================
def test_custom_metal_loop_detector(recovery_mod):
    """Custom metal-loop detector correctly classifies metal centres.

    Wf-337-A2 §3: the detector is the MEASURED recovery path.  It
    runs RDKit-only (no meeko); output is the recommended recovery
    strategy (``pt_to_zn_substitution`` or ``pt_strip_largest_fragment``).

    Test cases:
      a. Cisplatin ``[Pt](N)(N)(Cl)Cl`` — has_metal_loop=True,
         recovery_strategy=``pt_strip_largest_fragment`` (Pt has 4
         non-H neighbours; coordination complex).
      b. ``[Pt](Cl)(Cl)`` — has_metal_loop=True,
         recovery_strategy=``pt_to_zn_substitution`` (Pt has only 2
         non-H neighbours; dangling centre).
      c. Ruthenium ``[Ru](N)(N)(N)(N)(N)N`` — has_metal_loop=True,
         recovery_strategy=``pt_strip_largest_fragment`` (Ru has 6
         non-H neighbours; octahedral coordination complex).
      d. Iridium ``[Ir](C)(C)(C)(C)`` — has_metal_loop=True,
         recovery_strategy=``pt_strip_largest_fragment``.
      e. Organic ``CC(=O)Oc1ccccc1C(=O)O`` (aspirin) — has_metal_loop=False,
         recovery_strategy=``none``.
    """
    # (a) Cisplatin — coordination complex → strip.
    rep_a = recovery_mod.detect_metal_loop(CISPLATIN_SMILES)
    assert rep_a.has_metal_loop is True
    assert rep_a.n_metal_centres == 1
    assert 78 in rep_a.metal_atomic_nums  # Pt atomic number.
    assert rep_a.recovery_strategy == "pt_strip_largest_fragment"
    assert rep_a.honest_negative_reason != ""
    assert "pt-strip" in rep_a.honest_negative_reason.lower() or "largest" in rep_a.honest_negative_reason.lower()

    # (b) Dangling Pt — substitute.
    rep_b = recovery_mod.detect_metal_loop("[Pt](Cl)(Cl)")
    assert rep_b.has_metal_loop is True
    assert rep_b.n_metal_centres == 1
    assert rep_b.recovery_strategy == "pt_to_zn_substitution"
    assert "zn" in rep_b.honest_negative_reason.lower()

    # (c) Octahedral Ru — strip (6 non-H neighbours).
    rep_c = recovery_mod.detect_metal_loop("[Ru](N)(N)(N)(N)(N)N")
    assert rep_c.has_metal_loop is True
    assert 44 in rep_c.metal_atomic_nums  # Ru atomic number.
    assert rep_c.recovery_strategy == "pt_strip_largest_fragment"

    # (d) Ir — strip.
    rep_d = recovery_mod.detect_metal_loop("[Ir](C)(C)(C)(C)")
    assert rep_d.has_metal_loop is True
    assert 77 in rep_d.metal_atomic_nums  # Ir atomic number.
    assert rep_d.recovery_strategy == "pt_strip_largest_fragment"

    # (e) Aspirin — no metal loop.
    rep_e = recovery_mod.detect_metal_loop("CC(=O)Oc1ccccc1C(=O)O")
    assert rep_e.has_metal_loop is False
    assert rep_e.n_metal_centres == 0
    assert rep_e.metal_atomic_nums == []
    assert rep_e.recovery_strategy == "none"


# ===========================================================================
# Test 4 — Pocket sweep diagnostic on 5 Wf-325B pockets
# ===========================================================================
def test_pocket_sweep_diagnostic_5_wf325b_pockets(recovery_mod):
    """The pocket sweep diagnostic returns the correct structure for
    the 5 Wf-325B pockets.

    Wf-337-A2 §3: ``pocket_sweep_diagnostic`` runs all 3 recovery
    branches against the 5 Wf-325B pockets.  The output must:

    a. List the 5 pockets (1h36, 3eml, 3ny8, 4aaw, 4rlu).
    b. Report per-pocket recovery_branch + verdict + note.
    c. When neither Smina nor QuickVina is installed AND the SMILES
       has a Pt centre (cisplatin), the aggregate verdict is
       HONEST_NEG_METAL_LOOP_OK — the metal-loop detector is the
       recommended recovery path, NOT HONEST_NEGATIVE_WF264_SUBPANEL_PERSISTENT.
    d. ``n_pockets_docked = 0`` in this environment (no real Vina
       fork reachable); ``n_pockets_metal_loop_ok = 5`` (the
       detector recommends a recovery for every pocket).
    e. The ``metal_loop`` sub-dict carries the cisplatin
       classification: n_metal_centres=1, atomic_nums=[78],
       recovery_strategy=``pt_strip_largest_fragment``.
    """
    diag = recovery_mod.pocket_sweep_diagnostic(
        smiles=CISPLATIN_SMILES,
        pockets=WF325B_POCKETS,
        pocket_pdb_dir=(WF30_POCKET_DIR if WF30_POCKET_DIR.exists() else None),
    )

    # (a) Pockets enumeration.
    assert isinstance(diag, dict)
    assert diag["pockets"] == list(WF325B_POCKETS)
    assert diag["n_pockets_total"] == 5

    # (b) Per-pocket structure.
    per_pocket = diag["per_pocket"]
    assert len(per_pocket) == 5
    seen_pocket_ids = set()
    for entry in per_pocket:
        assert "pocket_id" in entry
        assert "recovery_branch" in entry
        assert "verdict" in entry
        assert "receptor_available" in entry
        assert "note" in entry
        seen_pocket_ids.add(entry["pocket_id"])
    assert seen_pocket_ids == set(WF325B_POCKETS)

    # (c) Aggregate verdict — depends on runtime probes.
    smina_avail = diag["smina_probe"]["available"]
    qvina_avail = diag["quickvina_probe"]["available"]
    if not smina_avail and not qvina_avail:
        # Canonical Wf-334-U3 scenario: neither Smina nor QuickVina
        # is installed.  Cisplatin HAS a Pt centre, so the metal-
        # loop detector fires.  Verdict must be HONEST_NEG_METAL_LOOP_OK.
        assert diag["aggregate_verdict"] == recovery_mod.VERDICT_METAL_LOOP_OK
        for entry in per_pocket:
            assert entry["recovery_branch"] == "metal_loop"
            assert entry["verdict"] == recovery_mod.VERDICT_METAL_LOOP_OK
    elif smina_avail:
        assert diag["aggregate_verdict"] == recovery_mod.VERDICT_RECOVERED
        for entry in per_pocket:
            assert entry["recovery_branch"] == "smina"
    else:
        assert diag["aggregate_verdict"] == recovery_mod.VERDICT_RECOVERED
        for entry in per_pocket:
            assert entry["recovery_branch"] == "quickvina"

    # (d) n_pockets_docked / n_pockets_metal_loop_ok counters.
    if not smina_avail and not qvina_avail:
        # No real engine — n_pockets_docked MUST be 0.
        assert diag["n_pockets_docked"] == 0
        # Metal-loop detector fires for cisplatin (5 pockets all
        # recommend a recovery strategy).
        assert diag["n_pockets_metal_loop_ok"] == 5
    else:
        # When a real engine is available, n_pockets_docked =
        # number of pockets with a receptor available (or 5 when
        # pocket_pdb_dir was not provided).
        assert diag["n_pockets_docked"] >= 0

    # (e) metal_loop sub-dict — cisplatin classification.
    ml = diag["metal_loop"]
    assert ml["has_metal_loop"] is True
    assert ml["n_metal_centres"] == 1
    assert 78 in ml["metal_atomic_nums"]
    assert ml["recovery_strategy"] == "pt_strip_largest_fragment"
    assert ml["honest_negative_reason"] != ""

    # Round-trip JSON serialisation (downstream consumers need this).
    text = json.dumps(diag, indent=2, default=str)
    assert "1h36" in text
    assert "pt_strip_largest_fragment" in text
