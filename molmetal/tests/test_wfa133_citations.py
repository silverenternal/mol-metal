# SPDX-License-Identifier: Apache-2.0
"""Wf-A133 — the citation block must survive an audit.

Three defects, all in the repo's own documentation of where its numbers
come from.  Two are CITATION errors and one is a SILENT-DROP defect.

DEFECT 1 — the "Lit anchors" docstring of
``molmetal/eval/scoring/decode_gate.py`` cited two papers by identifiers
that do not resolve to them:

* ``arXiv:1811.01457`` was given for *MOSES*.  That identifier is Innes
  et al., "Fashionable Modelling with Flux" (a Julia
  differentiable-programming framework paper).  *MOSES* is
  **arXiv:1811.12823** (Polykovskiy et al.).
* *GuacaMol* was given as "J. Cheminform. 11:49".  It is
  **J. Chem. Inf. Model. 2019, 59(3), 1096-1108**, DOI
  ``10.1021/acs.jcim.8b00839** (Brown, Fiscato, Segler, Vaucher).

Both were re-verified against the primary sources on 2026-10-31 (arXiv
abstract page + the papers' own PDF/HTML text + the benchmarks' released
code).  The corrected block also states, per anchor, what the source
DOES say — so a future reader can check the claim instead of trusting it.

DEFECT 2 — the same docstring called the ``largest_frac_min = 0.80``
default "the standard MOSES / GuacaMol acceptance floor".  It is not:
MOSES ships no largest-fragment ratio at all (its released
``fraction_valid`` is parse+sanitise; ``GetMolFrags`` appears nowhere in
the MOSES repository) and GuacaMol's ``is_valid`` is parse-only.  0.80
is a REPO-LOCAL choice.  The VALUE is a behaviour constant and is NOT
changed — only its provenance is corrected.  ``test_the_080_behaviour_is
_unchanged`` pins the behaviour.

  NOTE (a divergence from the Wf-A133 brief, flagged to the main thread):
  the brief asserted that "MOSES validity = sanitizable AND a single
  connected fragment" is the correct substantive claim.  That could NOT be
  reproduced from the primary sources — the released MOSES code has no
  fragment term, and the journal version's own words are "We define
  validity using RDKit's molecular structure parser that checks atoms'
  valency and consistency of bonds in aromatic rings" (Front. Pharmacol.
  2020, 11, 565644).  The docstring therefore states what MOSES DOES say,
  and the one-fragment rule is attributed to END (where it is a separate
  "Connectivity" metric, not validity).

DEFECT 3 — the SECOND RDKit bond-drop site,
``molmetal/models/bond_head.py`` ``BondAwareDecoder._assemble_mol``,
dropped a refused bond with a bare ``continue`` and a comment that named
the wrong mechanism ("e.g. valence overflow" — RDKit does not raise on
valence overflow at ``AddBond`` time; valence is checked at SANITISE
time, which is a different branch).  A dropped bond is a MOLECULAR
SPLIT, so it now carries a counter + a reason histogram, exactly like
its twin at ``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py``
(Wf-A128).  The decode behaviour is unchanged: the bond is still
dropped, the molecule is still split — what changed is that the split is
now COUNTED.

DEFECT 4 — the SAME false claim survives OUTSIDE ``decode_gate.py``.  The
correction is only real if it reaches every site that made the claim:

* ``configs/cfm_patience10_seed0_2026_10_12.yaml`` described the 0.80
  default as "standard MOSES / GuacaMol acceptance floor" and the
  ``min_atoms_per_mol: 2`` default as the "standard MOSES floor".
* ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py`` called the
  per-atom gate's thresholds "the standard MOSES floor".
* ``tests/test_track_e_decoding.py`` called 0.80 "the standard MOSES 0.80
  floor".

All three are now labelled as this repo's OWN project-strict choices,
with the two primary sources named so the claim stays checkable.  NO
VALUE CHANGED — the defaults are still 0.80 and 2, and the gate still
behaves identically.

CPU-only.  No GPU, no network, no corpus on disk.
"""

from __future__ import annotations

from pathlib import Path

import torch

from molmetal.eval.scoring.decode_gate import gate_ratio, gate_smiles
from molmetal.eval.scoring import decode_gate
from molmetal.models.bond_head import (
    BOND_DOUBLE,
    BOND_SINGLE,
    AtomCloud,
    BondAwareDecoder,
    BondOrderHead,
)

#: The whole "Lit anchors" block, straight from the module docstring.
_LIT_BLOCK = decode_gate.__doc__ or ""

#: Repo root — this file lives at ``<repo>/molmetal/tests/``.
_REPO = Path(__file__).resolve().parents[2]

#: The same block with all whitespace collapsed, so that a docstring REWRAP
#: (which does not change what the block says) cannot silently break a
#: citation assertion.  Citation strings are compared against this.
_LIT_FLAT = " ".join(_LIT_BLOCK.split())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decoder(seed: int = 1234) -> BondAwareDecoder:
    """A deterministic decoder (same seed -> same head weights)."""
    torch.manual_seed(seed)
    return BondAwareDecoder(bond_head=BondOrderHead(hidden_dim=8, n_layers=2))


def _cloud(atomic_numbers=(6, 6, 8)) -> AtomCloud:
    """Three atoms in a line; positions are irrelevant to the assembler."""
    n = len(atomic_numbers)
    return AtomCloud(
        positions=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) * 1.5,
        atomic_numbers=torch.tensor(list(atomic_numbers), dtype=torch.long),
    )


def _assert_only_negated(wrong: str) -> None:
    """A wrong identifier may still be MENTIONED — to record the correction —
    but only ever next to an explicit ``NOT``."""
    idx = _LIT_FLAT.find(wrong)
    assert idx != -1, f"{wrong!r} is not mentioned at all (the correction was dropped)"
    while idx != -1:
        window = _LIT_FLAT[max(0, idx - 400) : idx]
        assert "NOT" in window, (
            f"{wrong!r} may only appear as an explicit correction, but it "
            f"appears without a nearby NOT: ...{window[-120:]}[{wrong}]"
        )
        idx = _LIT_FLAT.find(wrong, idx + 1)


# ===========================================================================
# 1. THE CITATION CORRECTIONS
# ===========================================================================


def test_moses_is_cited_by_the_identifier_that_resolves_to_it():
    """arXiv:1811.12823 is MOSES; arXiv:1811.01457 is a Julia paper."""
    assert "1811.12823" in _LIT_FLAT, "MOSES must be cited as arXiv:1811.12823"
    # The wrong identifier may still be MENTIONED (to record the correction),
    # but only ever as an explicit negation.
    _assert_only_negated("1811.01457")


def test_the_moses_journal_version_is_cited_too():
    """MOSES was published in Front. Pharmacol. 2020, 11, 565644 (verified
    against the publisher's own metadata record on 2026-10-31)."""
    assert "10.3389/fphar.2020.565644" in _LIT_FLAT, "the MOSES DOI is missing"
    assert "Front. Pharmacol." in _LIT_FLAT, "the MOSES journal is missing"


def test_guacamol_is_cited_with_its_real_venue():
    """GuacaMol is JCIM 2019, 59(3), 1096-1108 — not J. Cheminform. 11:49."""
    assert (
        "J. Chem. Inf. Model. 2019, 59(3), 1096-1108" in _LIT_FLAT
    ), "GuacaMol's venue is missing or wrong"
    assert "10.1021/acs.jcim.8b00839" in _LIT_FLAT, "GuacaMol's DOI is missing"
    _assert_only_negated("11:49")


def test_the_wrong_guacamol_venue_is_identified():
    """"J. Cheminform. 11:49" is the Open Babel coordinate-generation paper
    (DOI 10.1186/s13321-019-0372-5) — naming it makes the correction
    checkable instead of merely deleting the wrong string."""
    assert "10.1186/s13321-019-0372-5" in _LIT_FLAT


def test_jtvae_anchor_keeps_its_own_identifier():
    """JTVAE is arXiv:1802.04364 (Jin, Barzilay, Jaakkola, ICML 2018)."""
    assert "1802.04364" in _LIT_FLAT


def test_the_published_criteria_that_the_080_is_not_are_named():
    """The corrected block must name the criteria that DO exist."""
    # CBGBench: largest fragment > 85% of the atoms (arXiv:2406.10840).
    assert "2406.10840" in _LIT_FLAT
    assert "85%" in _LIT_FLAT
    # END: connectivity is a SEPARATE metric requiring exactly one fragment.
    assert "2506.10532" in _LIT_FLAT
    assert "exactly one fragment" in _LIT_FLAT


# ===========================================================================
# 2. THE 0.80 IS REPO-LOCAL, NOT A PUBLISHED STANDARD
# ===========================================================================


def test_080_is_not_called_a_published_standard_anywhere():
    """The phrase "standard MOSES / GuacaMol" must be gone from both
    docstrings that carried it."""
    gate_doc = " ".join((gate_smiles.__doc__ or "").split())
    for blob, where in ((_LIT_FLAT, "module docstring"), (gate_doc, "gate_smiles")):
        lowered = blob.lower()
        assert (
            "standard moses" not in lowered
        ), f"the {where} still calls 0.80 a MOSES standard"
        assert (
            "standard guacamol" not in lowered
        ), f"the {where} still calls 0.80 a GuacaMol standard"
        assert (
            "acceptance floor" not in lowered
        ), f"the {where} still calls 0.80 an acceptance floor"


def test_080_is_explicitly_named_repo_local():
    """The replacement wording must say REPO-LOCAL, not just drop the claim."""
    assert "REPO-LOCAL" in _LIT_FLAT
    assert "REPO-LOCAL" in " ".join((gate_smiles.__doc__ or "").split())


def test_the_080_behaviour_is_unchanged():
    """The VALUE is a behaviour constant — this fix changed the provenance,
    not the number.  Pinned here so a future 'fix' cannot quietly move it."""
    # A 4-fragment SMILES: 1/4 of the atoms are in the largest fragment.
    r = gate_smiles("C.C.C.C", largest_frac_min=0.80, min_atoms=2)
    assert r.ok is False and r.reason == "multi_fragment", r
    # A clean molecule passes.
    r2 = gate_smiles("CCO", largest_frac_min=0.80, min_atoms=2)
    assert r2.ok is True, r2
    # The legacy (threshold-free) path is still bit-exact RDKit-strict.
    r3 = gate_smiles("C.C.C.C", largest_frac_min=0.0, min_atoms=0)
    assert r3.ok is True, r3
    # And the aggregate arithmetic is unchanged.
    gr = gate_ratio(
        ["CCO", "C.C.C.C", "nonsense", "[He]", "CC"],
        largest_frac_min=0.80,
        min_atoms=2,
    )
    assert gr["n_total"] == 5 and gr["n_ok"] == 2, gr
    assert abs(gr["per_atom_validity"] - 0.4) < 1e-9, gr
    assert gr["multi_frag_rejects"] == 1, gr
    assert gr["min_atoms_rejects"] == 1, gr
    assert gr["parse_failures"] == 1, gr


# ===========================================================================
# 3. THE SECOND RDKit BOND-DROP SITE IS COUNTED
# ===========================================================================


def test_a_dropped_bond_is_counted_with_its_reason():
    """RDKit refuses the SECOND (0, 1) bond; the decoder now says so."""
    dec = _decoder()
    d = dec._assemble_mol(
        _cloud(),
        [(0, 1, BOND_SINGLE), (0, 1, BOND_DOUBLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert d.mol is not None
    assert dec._last_bonds_dropped == 1, "the refused bond was not counted"
    assert dec._last_bonds_dropped_reasons == {"RuntimeError": 1}
    report = dec.bond_drop_report()
    assert report["bonds_dropped"] == 1, report
    assert report["reasons"] == {"RuntimeError": 1}, report


def test_the_drop_still_splits_the_molecule():
    """The FIX IS INSTRUMENTATION ONLY: the bond is still dropped and the
    molecule is still split — the decode now merely says so."""
    dec = _decoder()
    d = dec._assemble_mol(
        _cloud(),
        [(0, 1, BOND_SINGLE), (0, 1, BOND_DOUBLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert d.mol.GetNumBonds() == 2, "RDKit kept 2 of the 3 requested bonds"
    assert d.n_bonds == 3, "the decoded mol still reports the 3 requested bonds"


def test_the_count_reaches_the_per_molecule_record():
    """``DecodedMol.warnings`` is the repo's existing caveat channel, and the
    YuelBond decoder already uses the ``bonds_dropped_rdkit:<n>`` token on
    it.  The second drop site now uses the same token."""
    dec = _decoder()
    d = dec._assemble_mol(
        _cloud(),
        [(0, 1, BOND_SINGLE), (0, 1, BOND_DOUBLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert "bonds_dropped_rdkit:1" in d.warnings, d.warnings


def test_a_clean_decode_reports_zero_drops_and_no_warning():
    """0 is the state in which the fragment count can be read as the bond
    head's intent."""
    dec = _decoder()
    d = dec._assemble_mol(
        _cloud(),
        [(0, 1, BOND_SINGLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec.bond_drop_report() == {"bonds_dropped": 0, "reasons": {}}
    assert "bonds_dropped_rdkit" not in " ".join(d.warnings), d.warnings


def test_the_counter_resets_between_decodes():
    """A stale counter is worse than no counter: the next decode would
    inherit the previous decode's splits.  The reset is at the TOP of the
    assembler."""
    dec = _decoder()
    dec._assemble_mol(
        _cloud(),
        [(0, 1, BOND_SINGLE), (0, 1, BOND_DOUBLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 1
    dec._assemble_mol(
        _cloud(),
        [(0, 1, BOND_SINGLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 0, "the counter inherited a stale value"
    assert dec._last_bonds_dropped_reasons == {}


def test_the_histogram_separates_the_exception_types():
    """MEASURED on RDKit 2026.03.6: an out-of-range atom index and a
    self-bond raise ``RuntimeError``; a NEGATIVE atom index raises
    ``OverflowError`` (Python's int -> unsigned conversion, not RDKit)."""
    dec = _decoder()
    dec._assemble_mol(
        _cloud(),
        [
            (0, 1, BOND_SINGLE),
            (0, 9, BOND_SINGLE),
            (1, 1, BOND_SINGLE),
            (-1, 2, BOND_SINGLE),
        ],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 3, dec._last_bonds_dropped
    assert dec._last_bonds_dropped_reasons == {"RuntimeError": 2, "OverflowError": 1}


def test_the_wide_catch_is_retained():
    """Narrowing the catch to ``(RuntimeError, OverflowError)`` would be a
    BEHAVIOUR change: ``int(src)`` on a non-numeric index raises
    ``ValueError``/``TypeError``, which the wide catch swallows (bond
    dropped) and a narrow catch would PROPAGATE (the whole decode raises).
    The wide catch stays; the exception TYPE is recorded instead."""
    dec = _decoder()
    d = dec._assemble_mol(
        _cloud(),
        [(0, 1, BOND_SINGLE), ("x", 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert d is not None, "a non-numeric index must not abort the decode"
    assert dec._last_bonds_dropped == 1, dec._last_bonds_dropped
    assert dec._last_bonds_dropped_reasons == {"ValueError": 1}


# ===========================================================================
# 4. THE SAME FALSE CLAIM OUTSIDE decode_gate.py
# ===========================================================================

#: The three files that repeated the false "standard MOSES / GuacaMol"
#: claim outside the module docstring.  Each is (repo-relative path, the
#: substring that must NOT survive).
_RESIDUAL_SITES = (
    ("configs/cfm_patience10_seed0_2026_10_12.yaml", "standard MOSES"),
    ("molmetal/scripts/train_cfm_yuelbond_wasserstein.py", "standard MOSES"),
    ("tests/test_track_e_decoding.py", "standard MOSES"),
)


def _read(rel: str) -> str:
    p = _REPO / rel
    assert p.is_file(), f"the file the claim lived in is gone: {rel}"
    return p.read_text(encoding="utf-8")


def test_the_false_claim_is_gone_from_every_residual_site():
    """A correction that only lands in ONE file is not a correction."""
    for rel, forbidden in _RESIDUAL_SITES:
        blob = _read(rel)
        assert forbidden not in blob, (
            f"{rel} still calls the repo-local thresholds a MOSES standard"
        )
        assert "acceptance floor" not in blob, (
            f"{rel} still calls the repo-local thresholds an acceptance floor"
        )


def test_the_residual_sites_name_the_real_sources():
    """Deleting the claim is not enough — the replacement must say WHAT the
    sources actually constrain, so a reader can check it."""
    yaml_blob = _read("configs/cfm_patience10_seed0_2026_10_12.yaml")
    assert "1811.12823" in yaml_blob, "the YAML must name the MOSES identifier"
    assert "not a published standard" in yaml_blob, (
        "the YAML must say the 0.80 default is not a published standard"
    )
    trainer_blob = _read("molmetal/scripts/train_cfm_yuelbond_wasserstein.py")
    assert "1811.12823" in trainer_blob, (
        "the trainer comment must name the MOSES identifier"
    )
    test_blob = _read("tests/test_track_e_decoding.py")
    assert "MOSES has no fragment term" in test_blob, (
        "the test docstring must say what MOSES actually constrains"
    )


def test_no_residual_site_reintroduces_a_published_standard():
    """The guard itself must not be the only place the truth is written —
    a reader who never opens this file must still see the correction."""
    for rel, _forbidden in _RESIDUAL_SITES:
        blob = _read(rel).lower()
        assert "standard moses" not in blob, rel
        assert "standard guacamol" not in blob, rel
        assert "moses floor" not in blob, rel
        assert "guacamol acceptance floor" not in blob, rel


# ===========================================================================
# 5. THE ADAPTER'S OWN SELF-CONTRADICTION
# ===========================================================================

#: The CFM adapter.  Two stale claims lived here.
_ADAPTER = "molmetal/adapters/flow_matching_lipman/__init__.py"


def test_the_moldiff_venue_is_not_unverified_anymore():
    """The adapter said "venue unverified" at the sampler call site while
    its OWN class docstring (40 lines up) said "venue VERIFIED".  The
    file contradicted itself; the venue is ICML 2023, PMLR v202 peng23b."""
    blob = _read(_ADAPTER)
    assert "venue unverified" not in blob, (
        "the adapter still calls the MolDiff venue unverified"
    )
    idx = blob.find('"MolDiff"')
    assert idx != -1, "the MolDiff citation block is gone entirely"
    window = blob[max(0, idx - 400) : idx + 400]
    assert "peng23b" in window, (
        "the MolDiff citation at the call site must name the verified venue"
    )


def test_the_adapters_cutoff_comment_names_the_real_candidacy_radius():
    """The comment claimed "the decoder uses 2.4 Å for bond candidacy".
    That is `BondAwareDecoder.bond_cutoff` — a DIFFERENT decoder.  The
    bond head this guidance feeds is `YuelBondDecoderHead`, whose
    candidate cutoff is 3.5 Å (`yuelbond_decoder.py:531`, applied at
    :903; `edge_cutoff: 3.5` in the CFM configs)."""
    blob = _read(_ADAPTER)
    assert "2.4 Å for bond candidacy" not in blob, (
        "the adapter still attributes a 2.4 Å candidacy radius to the "
        "decoder that this guidance serves"
    )
    assert "3.5 Å candidate cutoff" in blob, (
        "the corrected comment must name the head's real 3.5 Å cutoff"
    )


def test_the_adapters_reach_gap_is_stated_not_hidden():
    """With the Wf-A128 upper bound OFF (the default), a pair at
    2.5-3.5 Å IS a bond candidate for the head but receives NO restoring
    force.  The old comment called that band "not real bonds", which is
    the reach gap stated backwards."""
    blob = _read(_ADAPTER)
    assert "reach gap" in blob, (
        "the adapter must name the reach gap it has (2.5-3.5 Å)"
    )


