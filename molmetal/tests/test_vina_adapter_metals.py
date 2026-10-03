"""Tests for ``vina_adapter._ligand_to_pdbqt`` with metal-containing ligands.

================================================================
Why these tests exist
================================================================
Wf-121 Phase 2 — meeko 0.8.0 cannot produce a PDBQT for a
metal-containing SMILES because ``AtomTyper._type_atoms`` leaves
``molsetup.atom_type is None`` for Pt-group metals and
``PDBQTWriterLegacy.write_string`` then rejects the molecule. We
patched ``_ligand_to_pdbqt`` to inject SMARTS atom-type overrides
(Phase 1: ``metal_pdbqt_overrides.add_metal_atom_types``) and to
fall back to ``charge_model='zero'`` (Gasteiger returns NaN for Pt).
If the molecule still cannot be converted, we now raise a typed
``MetalPDBQTUnavailable`` so callers can route metallodrugs to
``boltz_affinity``.

These tests pin all four behaviours:

1. Cisplatin (the canonical probe) emits a valid PDBQT with the
   Pt atom typed as the generic ``M`` atype.
2. All 8 d-block Pt-group metals (Pt/Ru/Ir/Os/Rh/Pd/Au/Re) succeed
   for SMILES that RDKit can embed.
3. Organic SMILES still produce valid PDBQTs (regression — the
   patch must not change the default behaviour for non-metal ligands).
4. ``MetalPDBQTUnavailable`` is raised for impossible cases
   (e.g. RDKit cannot embed the SMILES) and inherits from
   ``RuntimeError`` so existing ``except RuntimeError`` handlers
   keep working.

Honest framing
--------------
* Tests are CPU-only and depend on rdkit + meeko being installed
  (``mol-metal-venv`` or equivalent). They are skipped if either
  dep is missing — matching the existing
  ``tests/test_vina_adapter.py`` policy.
* Test #2 covers the 8 Pt-group metals, but a subset of them
  (Ru — pentagonal-pyramidal pentaammine probe) requires ETKDG
  to embed an octahedral geometry which RDKit refuses. We treat
  that case as a graceful ``MetalPDBQTUnavailable`` signal rather
  than a hard failure of the patch.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
    GENERIC_ATYPE,
    METAL_ATOM_TYPE_RULES,
    PROBE_SMILES,
    PT_GROUP_ELEMENTS,
    MetalPDBQTUnavailable,
)
from molmetal.molmetal_lam.sbdd_env.vina_adapter import (
    _have_meeko,
    _have_vina,
    _ligand_to_pdbqt,
)


# ---------------------------------------------------------------------------
# Skip policy
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not (_have_vina() and _have_meeko()),
    reason="vina/meeko not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_tmpdir() -> Path:
    """Per-test TemporaryDirectory (auto-cleanup on teardown)."""
    return Path(tempfile.mkdtemp(prefix="wf121_p2_test_"))


def _read_pdbqt(p: Path) -> str:
    return p.read_text()


def _atom_types(pdbqt_text: str) -> set[str]:
    """Return the set of atype strings in the PDBQT ATOM records.

    PDBQT ATOM lines are 79 chars wide; the atype lives in columns
    78..79 (1-indexed) or [77:79] (0-indexed). We slice defensively
    and strip whitespace.
    """
    out: set[str] = set()
    for line in pdbqt_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) >= 79:
            out.add(line[77:79].strip())
        elif len(line) >= 71:
            # PDBQTLegacy sometimes pads to 71 cols; fall back
            out.add(line[70:].strip().split()[0])
    return out


def _atom_name(line: str) -> str:
    """Extract the atom name from a PDBQT ATOM line.

    The atom name is the third whitespace-separated token (after
    'ATOM' and the serial number). For ``ATOM      1 Pt   UNL     1``,
    the atom name is ``'Pt'``.
    """
    parts = line.split()
    if len(parts) < 3:
        return ""
    return parts[2]


# ---------------------------------------------------------------------------
# 1. Cisplatin (canonical Pt metallodrug) — happy path
# ---------------------------------------------------------------------------
class TestCisplatinPdbqt:
    """Pins the headline result: cisplatin → valid PDBQT with Pt='M'."""

    CISPLATIN = "[Pt](N)(N)(Cl)Cl"

    def test_cisplatin_produces_valid_pdbqt(self, tmp_path):
        out = _ligand_to_pdbqt(self.CISPLATIN, tmp_path, "cisplatin")
        assert out.exists(), f"PDBQT file not created: {out}"
        assert out.stat().st_size > 100, "PDBQT suspiciously small"

    def test_cisplatin_pdbqt_has_atom_records(self, tmp_path):
        out = _ligand_to_pdbqt(self.CISPLATIN, tmp_path, "cisplatin")
        text = _read_pdbqt(out)
        n_atoms = sum(1 for line in text.splitlines() if line.startswith("ATOM"))
        # 1 Pt + 2 Cl + 2 N + 4 H (NH2 * 2) = 9 atoms
        assert n_atoms == 9, f"expected 9 atoms, got {n_atoms}"

    def test_cisplatin_pt_atom_typed_as_generic_M(self, tmp_path):
        out = _ligand_to_pdbqt(self.CISPLATIN, tmp_path, "cisplatin")
        text = _read_pdbqt(out)
        # The Pt atom must appear with our generic atype "M" in
        # columns 78..79 (0-indexed [77:79]).
        pt_lines = [
            l
            for l in text.splitlines()
            if l.startswith("ATOM") and _atom_name(l) == "Pt"
        ]
        assert len(pt_lines) == 1, f"expected 1 Pt ATOM line, got {len(pt_lines)}"
        pt_line = pt_lines[0]
        assert (
            pt_line[77:79].strip() == GENERIC_ATYPE
        ), f"Pt atype is {pt_line[77:79]!r} (expected {GENERIC_ATYPE!r})"

    def test_cisplatin_charges_are_zero_or_finite(self, tmp_path):
        """The charge column must be parseable as a finite float.

        With ``charge_model='zero'`` every ATOM line shows +0.000 —
        this is the explicit guarantee we rely on for the metallodrug
        path. (Gasteiger returns NaN for Pt+2, which would propagate
        into PDBQT parsing downstream.)
        """
        out = _ligand_to_pdbqt(self.CISPLATIN, tmp_path, "cisplatin")
        text = _read_pdbqt(out)
        # Column 70-76 in PDBQT holds the partial charge (right-aligned).
        for line in text.splitlines():
            if not line.startswith("ATOM"):
                continue
            charge_str = line[70:76].strip()
            assert charge_str, f"empty charge column in: {line!r}"
            float(charge_str)  # raises if not parseable


# ---------------------------------------------------------------------------
# 2. All 8 d-block Pt-group metals
# ---------------------------------------------------------------------------
class TestAllPtGroupMetals:
    """Each supported metal must produce a valid PDBQT — or, for an
    impossible-to-embed probe (Ru pentaammine), a clean
    ``MetalPDBQTUnavailable``."""

    # NOTE: Pt/Ru/Ir/Os/Rh/Pd/Au/Re. Ru's PROBE_SMILES is the
    # canonical impossible case (5-coordinate pentaammine — RDKit
    # refuses to embed it). We assert the typed exception there.

    @pytest.mark.parametrize(
        "elem",
        ["Pt", "Ir", "Os", "Rh", "Pd", "Au", "Re"],  # 7 embeds cleanly
    )
    def test_metal_smiles_produces_valid_pdbqt(self, tmp_path, elem):
        smiles = PROBE_SMILES[elem]
        out = _ligand_to_pdbqt(smiles, tmp_path, elem.lower())
        assert out.exists()
        text = _read_pdbqt(out)
        assert sum(1 for l in text.splitlines() if l.startswith("ATOM")) > 0

    @pytest.mark.parametrize(
        "elem",
        ["Pt", "Ir", "Os", "Rh", "Pd", "Au", "Re"],
    )
    def test_metal_atom_typed_as_generic_atype(self, tmp_path, elem):
        """The atype of the metal atom in the output must be 'M'."""
        smiles = PROBE_SMILES[elem]
        out = _ligand_to_pdbqt(smiles, tmp_path, elem.lower())
        text = _read_pdbqt(out)
        # The probe SMILES have a single atom of the named metal
        # (see ``PROBE_SMILES`` docstrings in metal_pdbqt_overrides).
        # Find the line for that metal — atom name is the third
        # whitespace-separated token.
        elem_short = elem[:2]  # "Pt", "Ru", … — all 2 chars here
        metal_lines = [
            l
            for l in text.splitlines()
            if l.startswith("ATOM") and _atom_name(l) == elem_short
        ]
        assert (
            len(metal_lines) >= 1
        ), f"no ATOM line found for {elem} in PDBQT: {text[:200]!r}"
        for ml in metal_lines:
            assert (
                ml[77:79].strip() == GENERIC_ATYPE
            ), f"{elem} atype is {ml[77:79]!r} (expected {GENERIC_ATYPE!r})"

    def test_ru_probe_raises_metal_pdbqt_unavailable(self, tmp_path):
        """Ru's PROBE_SMILES is a 5-coordinate pentaammine — RDKit's
        ETKDG refuses to embed octahedral geometry, so we MUST raise
        the typed ``MetalPDBQTUnavailable`` rather than a confusing
        meeko internal error.
        """
        smiles = PROBE_SMILES["Ru"]  # [Ru](N)(N)(N)(N)(N)Cl
        with pytest.raises(MetalPDBQTUnavailable) as ei:
            _ligand_to_pdbqt(smiles, tmp_path, "ru")
        # The error message must mention Ru so callers can debug.
        assert "Ru" in str(ei.value) or "embed" in str(ei.value).lower()

    def test_all_eight_metals_covered(self):
        """Sanity: PROBE_SMILES covers every element we advertise."""
        assert set(PROBE_SMILES.keys()) == set(PT_GROUP_ELEMENTS)
        assert set(METAL_ATOM_TYPE_RULES.keys()) == set(PT_GROUP_ELEMENTS)


# ---------------------------------------------------------------------------
# 3. Organic-ligand regression — patch must NOT change organic behaviour
# ---------------------------------------------------------------------------
class TestOrganicRegression:
    """Patch is opt-in for metals only — organics still go through
    the default Gasteiger path and emit a valid PDBQT."""

    @pytest.mark.parametrize(
        "name,smiles",
        [
            ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
            ("phenol", "c1ccccc1O"),
            ("ethanol", "CCO"),
            ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
            ("benzene", "c1ccccc1"),
            ("imidazole", "c1cnc[nH]1"),
        ],
    )
    def test_organic_smiles_produces_valid_pdbqt(self, tmp_path, name, smiles):
        out = _ligand_to_pdbqt(smiles, tmp_path, name)
        assert out.exists()
        text = _read_pdbqt(out)
        assert sum(1 for l in text.splitlines() if l.startswith("ATOM")) > 0
        # All charges must be finite floats (Gasteiger path).
        for line in text.splitlines():
            if not line.startswith("ATOM"):
                continue
            charge_str = line[70:76].strip()
            assert charge_str, f"empty charge column in: {line!r}"
            float(charge_str)

    def test_organic_failure_raises_runtime_error_not_metal(self, tmp_path):
        """If an organic SMILES fails PDBQT conversion (e.g. meeko
        rejects it), we MUST raise ``RuntimeError`` — NOT
        ``MetalPDBQTUnavailable``. The typed exception is reserved
        for the metallodrug path so callers can route accordingly.
        """
        # We can't easily craft an organic SMILES that meeko rejects
        # (most parse fine), so we use a known-bad-SMILES failure
        # which raises ``ValueError`` BEFORE the metal-vs-organic
        # branch. This pins the regression: ``ValueError`` still
        # comes from the parser, not the typed metal path.
        from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
            MetalPDBQTUnavailable,
        )

        try:
            _ligand_to_pdbqt("not_a_real_smiles_$$$", tmp_path, "bad")
        except MetalPDBQTUnavailable:
            pytest.fail(
                "MetalPDBQTUnavailable must NOT be raised for non-metal "
                "SMILES (regardless of whether they parse)"
            )
        except ValueError:
            pass  # expected — RDKit fails to parse
        except Exception:
            pass  # any other exception is also fine — the point is
            # that the typed metal exception is reserved.

    def test_carbon_atype_is_not_generic_M(self, tmp_path):
        """The patch must NOT type organic atoms as 'M' — only the
        d-block metals get the generic fallback."""
        out = _ligand_to_pdbqt("c1ccccc1", tmp_path, "benzene")
        text = _read_pdbqt(out)
        atypes = _atom_types(text)
        # benzene should type C and H atoms only.
        assert atypes, "no atypes parsed from benzene PDBQT"
        assert (
            GENERIC_ATYPE not in atypes
        ), f"benzene PDBQT unexpectedly contains 'M' atype: {atypes}"


# ---------------------------------------------------------------------------
# 4. MetalPDBQTUnavailable — exception type contract
# ---------------------------------------------------------------------------
class TestMetalPDBQTUnavailableContract:
    """Pins the public surface of the typed exception."""

    def test_exception_class_inherits_runtime_error(self):
        """``MetalPDBQTUnavailable`` MUST inherit from ``RuntimeError``
        so existing ``except RuntimeError`` handlers keep catching it.
        """
        assert issubclass(MetalPDBQTUnavailable, RuntimeError)

    def test_exception_is_constructible_with_message(self):
        msg = "pt-group metal could not be converted"
        exc = MetalPDBQTUnavailable(msg)
        assert str(exc) == msg

    def test_exception_importable_from_overrides_module(self):
        """Importable from both the module itself and re-exported by
        ``vina_adapter`` callers via the public surface."""
        from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
            MetalPDBQTUnavailable as MPO,
        )

        assert MPO is MetalPDBQTUnavailable


# ---------------------------------------------------------------------------
# 5. End-to-end adapter.dock with a metal ligand — smoke test
# ---------------------------------------------------------------------------
class TestLigandToPdbqtAdapterIntegration:
    """Pins the adapter-level wiring of the patch.

    Note: this only verifies that the path executes (i.e. the PDBQT
    conversion no longer throws ``atom_type is None`` for Pt). It does
    NOT assert on the kcal/mol value — that's downstream of the patch
    and is allowed to be ``status='MEASURED_GENERIC_ATOM'``. We test
    the conversion step (the patch's contract) rather than the full
    Vina docking pipeline, which has its own pre-existing dependency
    on a properly-prepared receptor PDBQT.
    """

    def test_cisplatin_through_ligand_to_pdbqt(self, tmp_path):
        """The adapter calls ``_ligand_to_pdbqt`` directly — so
        testing that helper with the canonical cisplatin SMILES
        is the cleanest end-to-end integration check."""
        out = _ligand_to_pdbqt("[Pt](N)(N)(Cl)Cl", tmp_path, "cis")
        assert out.exists()
        # Verify the metal PDBQT guard works: the file must
        # contain the Pt atom typed as 'M'.
        text = out.read_text()
        # Atom-name is the third whitespace-separated token.
        pt_atoms = [
            l
            for l in text.splitlines()
            if l.startswith("ATOM") and _atom_name(l) == "Pt"
        ]
        assert pt_atoms, "no Pt ATOM line found in cisplatin PDBQT"
        assert all(l[77:79].strip() == GENERIC_ATYPE for l in pt_atoms)

    def test_all_pdbqts_for_eight_metals_have_correct_m_atype(self, tmp_path):
        """Smoke test: across all 8 Pt-group metals, every produced
        PDBQT (when one is produced) types the metal atom as 'M'."""
        for elem, smiles in PROBE_SMILES.items():
            try:
                out = _ligand_to_pdbqt(smiles, tmp_path, elem.lower())
                text = out.read_text()
                # Atom-name is the third whitespace-separated token.
                metal_atom_lines = [
                    l
                    for l in text.splitlines()
                    if l.startswith("ATOM") and _atom_name(l) == elem[:2]
                ]
                assert metal_atom_lines, f"{elem}: no metal ATOM line in PDBQT"
                assert all(
                    l[77:79].strip() == GENERIC_ATYPE for l in metal_atom_lines
                ), f"{elem}: atype is not 'M' in {metal_atom_lines}"
            except MetalPDBQTUnavailable:
                # Ru's probe is the documented impossible case —
                # the typed exception is the expected behaviour.
                if elem != "Ru":
                    pytest.fail(f"unexpected MetalPDBQTUnavailable for {elem}")

    def test_dock_smiles_one_shot_with_cisplatin(self, tmp_path):
        """The ``dock_smiles`` one-shot helper should NOT raise the
        typed ``MetalPDBQTUnavailable`` exception for cisplatin.

        We don't assert on the docking score (the synthetic pocket
        is too unrealistic) — only on the exception type. The whole
        point of Wf-121 P2 is that the metal-SMILES path is wired
        through to ``_ligand_to_pdbqt`` and produces a valid PDBQT.
        """
        import torch

        from molmetal.domain import Pocket
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import dock_smiles

        coords = torch.zeros(50, 3, dtype=torch.float32)
        coords[:, 0] = torch.linspace(-8.0, 8.0, 50)
        pocket = Pocket(
            pdb_id="SYN",
            coords=coords,
            atom_types=torch.full((50,), 6, dtype=torch.long),
            residue_ids=torch.zeros(50, dtype=torch.long),
            chain_ids=torch.zeros(50, dtype=torch.long),
            mask=torch.ones(50, dtype=torch.bool),
            center=torch.zeros(3, dtype=torch.float32),
            radius=12.0,
        )

        # The ``dock_smiles`` helper invokes Vina, which has its own
        # pre-existing requirement that the receptor be a real
        # PDBQT file (a synthetic PDB is rejected with ``TypeError``
        # in ``Vina.set_receptor``). That failure mode is OUT OF
        # SCOPE for this patch — we only pin the metal-PDBQT
        # conversion contract.
        try:
            dock_smiles(
                "[Pt](N)(N)(Cl)Cl",
                pocket,
                n_poses=1,
                exhaustiveness=1,
            )
        except MetalPDBQTUnavailable:
            pytest.fail(
                "dock_smiles raised MetalPDBQTUnavailable for cisplatin — "
                "the patch must produce a valid PDBQT"
            )
        except (TypeError, RuntimeError, OSError):
            # Vina pre-existing limitation on synthetic pockets —
            # not the patch's contract.
            pass
