"""Tests for ``molmetal.molmetal_lam.lam_chem.ood_protocol``.

Pitfall Rank 14 — TODO-30 P6.3 — Tanimoto<0.4 + MMseqs2 OOD protocol.

Coverage (24 tests across 4 sections)
-------------------------------------
1. **compute_tanimoto scalar helper** (6 tests) — parity,
   self-similarity, parse-failure semantics, identity convention,
   cross-molecule distinctness.
2. **TanimotoOODProtocol** (8 tests) — threshold gating, empty
   train pool, self-vs-train, cisplatin canonical case,
   threshold validation, max_tanimoto, fingerprint-radius config,
   parameter defaults.
3. **MMseqs2OODProtocol** (4 tests) — stub raises NotImplemented
   for both ``is_ood`` and ``max_seq_identity``; threshold validation
   still works; param defaults.
5. **CombinedOODProtocol** (6 tests) — type validation, ligand-only
   path (mmseqs2=None), combined path raises NotImplementedError,
   AND semantics, train-list empty handling.

All tests are CPU-only (RDKit + dataclasses only — no GPU, no
MMseqs2 binary).  When RDKit is missing the tests skip with a clear
message so the rest of the suite can still pass.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# RDKit availability probe + lazy module loader
# ---------------------------------------------------------------------------
def _have_rdkit() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


requires_rdkit = pytest.mark.skipif(
    not _have_rdkit(),
    reason="RDKit not installed; OOD protocol tests require Morgan FP",
)


def _load_ood_protocol():
    """Load the ood_protocol module without going through
    ``lam_chem.__init__`` (which imports pysr_wrapper).

    Mirrors the lazy-load pattern used in test_buyable_gate.py so the
    tests stay robust against environments where ``lam_chem``
    package-level imports fail (numpy missing, pysr missing, etc.).
    """
    module_name = "molmetal.molmetal_lam.lam_chem.ood_protocol"
    if module_name in sys.modules:
        return sys.modules[module_name]
    pkg_init = ROOT / "molmetal" / "molmetal_lam" / "lam_chem" / "__init__.py"
    if pkg_init.is_file():
        try:
            spec = importlib.util.spec_from_file_location(
                "molmetal.molmetal_lam.lam_chem",
                pkg_init,
                submodule_search_locations=[str(pkg_init.parent)],
            )
            if spec is not None and spec.loader is not None:
                pkg = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = pkg
                try:
                    spec.loader.exec_module(pkg)
                except Exception:
                    sys.modules.pop(spec.name, None)
        except Exception:
            pass
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "molmetal" / "molmetal_lam" / "lam_chem" / "ood_protocol.py",
    )
    if spec is None or spec.loader is None:
        pytest.skip("ood_protocol.py is not importable")
    module = importlib.util.module_from_spec(spec)
    # IMPORTANT: register the module in sys.modules BEFORE exec'ing.
    # The ``@dataclass`` decorator calls ``sys.modules[cls.__module__]``
    # to introspect class defaults; without this registration it sees
    # ``None`` and raises AttributeError.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ood():
    """Pytest fixture: the loaded ood_protocol module."""
    return _load_ood_protocol()


# Canonical Pt drug SMILES (same as test_pt_metal_ligand_exchange_known.py).
CISPLATIN = "N.N.[Cl][Pt][Cl]"
CARBOPLATIN = "N.N.[Pt]12(OC(=O)CCC1)OC(=O)CCC2"
SATRAPLATIN = (
    "CC1=C(C(=O)O[Pt]([NH3])([NH3])(OC(=O)c2ccccc2)C(C)(C)C)"
    "C(C)(C)C=C1"
)

# Other small drug-like ligands for cross-checks.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
PHENOL = "c1ccccc1O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


# ---------------------------------------------------------------------------
# 1. compute_tanimoto scalar helper
# ---------------------------------------------------------------------------
class TestComputeTanimoto:
    """Scalar Tanimoto similarity semantics."""

    @requires_rdkit
    def test_self_similarity_is_one(self, ood):
        """Tanimoto of any SMILES with itself = 1.0."""
        for smi in (CISPLATIN, ASPIRIN, PHENOL, CAFFEINE):
            assert ood.compute_tanimoto(smi, smi) == pytest.approx(1.0, abs=1e-6)

    @requires_rdkit
    def test_distinct_molecules_below_one(self, ood):
        """Different molecules should have Tanimoto < 1.0 (strict)."""
        # Aspirin vs phenol — both benzene-bearing but chemically distinct.
        assert ood.compute_tanimoto(ASPIRIN, PHENOL) < 1.0

    @requires_rdkit
    def test_unparseable_returns_zero(self, ood):
        """Unparseable query SMILES → Tanimoto 0.0 (fail-closed OOD miss)."""
        # "not-a-smiles" cannot be parsed by RDKit — should return 0.0.
        result = ood.compute_tanimoto("not-a-smiles", ASPIRIN)
        assert result == pytest.approx(0.0, abs=1e-6)

    @requires_rdkit
    def test_both_unparseable_returns_one(self, ood):
        """Empty-zero convention: both-empty Tanimoto = 1.0 (symmetric)."""
        # Both invalid → convention returns 1.0 to preserve the
        # "empty = self-similar" invariant used in data_diversity.
        result = ood.compute_tanimoto("invalid-a", "invalid-b")
        assert result == pytest.approx(1.0, abs=1e-6)

    @requires_rdkit
    def test_cisplatin_carboplatin_below_0_4(self, ood):
        """Cisplatin vs carboplatin (both small Pt(II) complexes) Tanimoto < 0.4.

        Honest framing: both are acyclic Pt_II d⁸ complexes but the
        carboplatin CBDCA chelate adds 4 heavy atoms that are not in
        cisplatin, so the Morgan-ECFP4 fingerprint is structurally
        distinct — well below the 0.4 OOD threshold.
        """
        s = ood.compute_tanimoto(CISPLATIN, CARBOPLATIN)
        assert s < 0.4, f"cisplatin/carboplatin should be Tanimoto<0.4; got {s:.3f}"

    @requires_rdkit
    def test_aspirin_carboplatin_below_0_4(self, ood):
        """Aspirin (organic) vs carboplatin (Pt complex) — completely disjoint chemistry."""
        s = ood.compute_tanimoto(ASPIRIN, CARBOPLATIN)
        assert s < 0.4, f"aspirin/carboplatin should be Tanimoto<0.4; got {s:.3f}"


# ---------------------------------------------------------------------------
# 2. TanimotoOODProtocol — primary OOD classifier
# ---------------------------------------------------------------------------
class TestTanimotoOODProtocol:
    """CPU Tanimoto<0.4 OOD classifier semantics."""

    @requires_rdkit
    def test_threshold_below_is_ood(self, ood):
        """A query with Tanimoto < 0.4 to all train mols is OOD."""
        p = ood.TanimotoOODProtocol(threshold=0.4)
        # Phenol vs cisplatin — completely different chemistry.
        assert p.is_ood(PHENOL, [CISPLATIN]) is True

    @requires_rdkit
    def test_same_molecule_not_ood(self, ood):
        """Self-similarity (Tanimoto = 1.0) is NOT OOD by definition."""
        p = ood.TanimotoOODProtocol(threshold=0.4)
        assert p.is_ood(CISPLATIN, [CISPLATIN]) is False

    @requires_rdkit
    def test_cisplatin_vs_train_not_ood(self, ood):
        """Cisplatin vs cisplatin-in-train is NOT OOD (Tanimoto = 1.0)."""
        p = ood.TanimotoOODProtocol(threshold=0.4)
        train = [CISPLATIN, CARBOPLATIN, ASPIRIN]
        assert p.is_ood(CISPLATIN, train) is False

    @requires_rdkit
    def test_empty_train_pool_is_ood(self, ood):
        """An empty training pool → every query is OOD (no chem to compare)."""
        p = ood.TanimotoOODProtocol(threshold=0.4)
        assert p.is_ood(CISPLATIN, []) is True

    @requires_rdkit
    def test_threshold_validation(self, ood):
        """Threshold outside [0.0, 1.0] must raise ValueError."""
        with pytest.raises(ValueError):
            ood.TanimotoOODProtocol(threshold=-0.1)
        with pytest.raises(ValueError):
            ood.TanimotoOODProtocol(threshold=1.1)
        with pytest.raises(ValueError):
            ood.TanimotoOODProtocol(threshold=2.0)

    @requires_rdkit
    def test_max_tanimoto_helper(self, ood):
        """max_tanimoto returns the maximum similarity across the train list."""
        p = ood.TanimotoOODProtocol(threshold=0.4)
        train = [ASPIRIN, CISPLATIN, CARBOPLATIN]
        # Phenol vs aspirin > 0 (shared benzene), vs cisplatin < 0.4,
        # vs carboplatin < 0.4.  The max should be the aspirin sim.
        m = p.max_tanimoto(PHENOL, train)
        aspirin_sim = p.compute_tanimoto(PHENOL, ASPIRIN)
        assert m == pytest.approx(aspirin_sim, abs=1e-6)
        assert m > 0.0

    @requires_rdkit
    def test_compute_tanimoto_instance_method(self, ood):
        """TanimotoOODProtocol.compute_tanimoto delegates to the scalar helper."""
        p = ood.TanimotoOODProtocol(threshold=0.4)
        # The instance method must return the same value as the module
        # function (delegation semantics).
        direct = ood.compute_tanimoto(PHENOL, CISPLATIN)
        instance = p.compute_tanimoto(PHENOL, CISPLATIN)
        assert direct == pytest.approx(instance, abs=1e-9)

    @requires_rdkit
    def test_radius_and_nbits_config(self, ood):
        """Non-default radius / n_bits are accepted and used."""
        # Just check the constructor accepts them and produces a
        # valid protocol.  We don't pin the numerical output because
        # the Morgan FP at different (radius, n_bits) settings is
        # not bit-for-bit identical to the ECFP4 default.
        p = ood.TanimotoOODProtocol(threshold=0.4, radius=3, n_bits=1024)
        assert p.radius == 3
        assert p.n_bits == 1024
        # The protocol still works (returns a bool).
        result = p.is_ood(PHENOL, [CISPLATIN])
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 3. MMseqs2OODProtocol — stub (NotImplementedError)
# ---------------------------------------------------------------------------
class TestMMseqs2Stub:
    """MMseqs2 OOD protocol is a deferred stub."""

    def test_is_ood_raises_not_implemented(self, ood):
        """MMseqs2 protocol always raises NotImplementedError (deferred)."""
        proto = ood.MMseqs2OODProtocol(threshold_seq_id=0.30)
        with pytest.raises(NotImplementedError):
            proto.is_ood("ARNDCEQ", ["MKTAY"])

    def test_max_seq_identity_raises_not_implemented(self, ood):
        """max_seq_identity also raises NotImplementedError (deferred)."""
        proto = ood.MMseqs2OODProtocol(threshold_seq_id=0.30)
        with pytest.raises(NotImplementedError):
            proto.max_seq_identity("ARNDCEQ", ["MKTAY"])

    def test_threshold_validation(self, ood):
        """MMseqs2 threshold outside [0.0, 1.0] must raise ValueError."""
        with pytest.raises(ValueError):
            ood.MMseqs2OODProtocol(threshold_seq_id=-0.1)
        with pytest.raises(ValueError):
            ood.MMseqs2OODProtocol(threshold_seq_id=1.5)

    def test_default_threshold_is_0_30(self, ood):
        """Default MMseqs2 sequence-identity threshold = 0.30."""
        proto = ood.MMseqs2OODProtocol()
        assert proto.threshold_seq_id == pytest.approx(0.30, abs=1e-9)
        assert proto.mmseqs_binary is None


# ---------------------------------------------------------------------------
# 4. CombinedOODProtocol — OR-combined (logical AND actually)
# ---------------------------------------------------------------------------
class TestCombinedOODProtocol:
    """Combined ligand + pocket OOD protocol."""

    @requires_rdkit
    def test_tanimoto_only_works(self, ood):
        """With mmseqs2=None, only the ligand axis is consulted."""
        p = ood.CombinedOODProtocol(
            tanimoto=ood.TanimotoOODProtocol(threshold=0.4),
            mmseqs2=None,
        )
        # Phenol vs cisplatin — should be OOD on the ligand axis alone.
        result = p.is_ood(
            query_smi=PHENOL,
            query_seq="ARNDCEQGHILK",  # pocket sequence (ignored here)
            train_smi=[CISPLATIN],
            train_seq=["MKTAYIAKQRQ"],  # ignored
        )
        assert result is True

    @requires_rdkit
    def test_type_validation(self, ood):
        """Constructor must reject wrong types for `tanimoto` / `mmseqs2`."""
        with pytest.raises(TypeError):
            # tanimoto wrong type.
            ood.CombinedOODProtocol(tanimoto="not a protocol")
        with pytest.raises(TypeError):
            # mmseqs2 wrong type (not MMseqs2OODProtocol and not None).
            ood.CombinedOODProtocol(
                tanimoto=ood.TanimotoOODProtocol(threshold=0.4),
                mmseqs2="not a protocol",
            )

    @requires_rdkit
    def test_tanimoto_and_pocket_axis_raises(self, ood):
        """With mmseqs2 set, is_ood raises NotImplementedError (deferred)."""
        p = ood.CombinedOODProtocol(
            tanimoto=ood.TanimotoOODProtocol(threshold=0.4),
            mmseqs2=ood.MMseqs2OODProtocol(threshold_seq_id=0.30),
        )
        with pytest.raises(NotImplementedError):
            p.is_ood(
                query_smi=PHENOL,
                query_seq="ARNDCEQGHILK",
                train_smi=[CISPLATIN],
                train_seq=["MKTAYIAKQRQ"],
            )

    @requires_rdkit
    def test_self_ligand_not_ood(self, ood):
        """When the query ligand IS in the train list, it must not be OOD."""
        p = ood.CombinedOODProtocol(
            tanimoto=ood.TanimotoOODProtocol(threshold=0.4),
            mmseqs2=None,
        )
        # Cisplatin querying against cisplatin-train → not OOD.
        assert p.is_ood(
            query_smi=CISPLATIN,
            query_seq="ARND",
            train_smi=[CISPLATIN, ASPIRIN],
            train_seq=["MKTA"],
        ) is False

    @requires_rdkit
    def test_empty_train_smi_is_ood(self, ood):
        """Empty train_smi → OOD on the ligand axis (no chem to compare)."""
        p = ood.CombinedOODProtocol(
            tanimoto=ood.TanimotoOODProtocol(threshold=0.4),
            mmseqs2=None,
        )
        assert p.is_ood(
            query_smi=CISPLATIN,
            query_seq="ARND",
            train_smi=[],
            train_seq=["MKTA"],
        ) is True

    @requires_rdkit
    def test_threshold_strictness(self, ood):
        """Strictly below threshold → OOD; at-or-above → not OOD."""
        # Find a pair whose Tanimoto sits between 0 and 1 so the
        # boundary semantics matter.  We use phenol vs aspirin (both
        # have a benzene ring, so the sim is non-zero) — their sim
        # is typically around 0.3-0.5.
        sim = ood.compute_tanimoto(PHENOL, ASPIRIN)
        # Set threshold below the actual sim → query is NOT OOD.
        below = ood.TanimotoOODProtocol(threshold=max(0.0, sim - 0.05))
        # Set threshold above the actual sim → query IS OOD.
        above = ood.TanimotoOODProtocol(threshold=min(1.0, sim + 0.05))
        assert below.is_ood(PHENOL, [ASPIRIN]) is False
        assert above.is_ood(PHENOL, [ASPIRIN]) is True
