#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A155 BRANCH COORD — the metal's COORDINATION SPHERE over the generation
pool named by ``--pool``.

WHAT THIS MEASURES
------------------
Frontier option (D) from the Wf-A153 frontier scan: switch the analysis from
the *ligand* to the *metal*.  For every row of the generation pool passed by
``--pool`` (the default is the published C8 pool) this computes the
coordination sphere of the metal centre:

* the metal's **coordination number** (RDKit ``GetDegree()`` — the number of
  atoms covalently bonded to the metal, i.e. heavy-atom neighbours; the
  pool's hydrogen convention is MEASURED at run time, not assumed — see
  ``definitions.coordination_number``);
* the **neighbour element multiset** (the coordination motif, e.g.
  ``N,N,N,Cl``) and its multiplicity;
* the **donor-class breakdown** of that multiset — N donor / O donor /
  C donor / halide / S / P / Se / other — because those are the classical
  platinum(II) / ruthenium(II) / iridium(III) donor sets;
* the metal's **formal oxidation state** under three explicitly-labelled
  layers (see ``OXIDATION_STATE_LAYERS`` below), each of which is a
  RULE-BASED INFERENCE and never a measurement;
* the **coordination-number distribution**, the **donor-motif distribution**,
  and the coordination-number **plausibility** against the repo's own
  ``coordination_capacity`` table;
* a **cross-tab** of coordination against the ligand-only connectivity (the
  option-A question): does a fragmented ligand correlate with an under- or
  over-coordinated metal?

Why a NEW metric and not a re-quote
-----------------------------------
This is a NEW denominator over a NEW population and it is NOT either of the two
headline numbers that already exist:

* the end-to-end single-molecule rate published in
  ``metrics/by_metric/cfm_fragmentation_audit.json``
  (``/summary/generation_side/single_molecule_rate_end_to_end``) is a MEDIAN
  over RUNS; the run count is read back out of that file AT RUN TIME
  (:func:`_end_to_end_headline`);
* the ``n_frags == 1`` fraction of the SAME pool is RE-MEASURED here at run
  time (:func:`build_what_this_is_not`), never quoted from a constant.

The coordination sphere is a per-atom property of ONE pool; it has no
"rate" denominator of either kind.  A restatement of either headline number
would be a FAILURE.

Comparability to the published frontier band
---------------------------------------------
The published band is a FRONTIER CONSTANT -- an external, published number,
never one of ours -- and its PRIMARY-RECORD provenance (arXiv id, locator,
verbatim quote, retrieval sha256) lives in :data:`FRONTIER_BAND_CITATION`.
It is separated from anything measured here by FOUR independent comparability
breaks (DENOMINATOR / METRIC / POST-PROCESSING / HYDROGENS).  The break
texts are COMPUTED at run time from the pool actually being measured
(:func:`build_comparability_breaks`): the row count, the pool name and the
break count all come from the run, so the statement is TRUE for ANY pool.
Never present a number from this script as comparable to the frontier band.

The metal-alone rows
--------------------
The rows whose metal sits ALONE in its RDKit fragment have an EMPTY
coordination sphere **by construction** (not by a modelling failure).  HOW
MANY there are is a MEASUREMENT (:func:`summarize`), not a constant: they
are reported separately, never averaged away, and the pool-wide coordination
statistics are reported BOTH over all rows and over the rows with a
non-empty sphere.

CPU-only
--------
This script never imports torch and never touches a GPU: it is a pure
RDKit-over-a-CSV pass.  ``CUDA_VISIBLE_DEVICES`` is forced to ``""`` before
RDKit is imported.  (Note: ``molmetal/data/__init__.py`` transitively
imports torch for the whole repo, so a torch module may be present in the
interpreter after importing this file — it is never USED here, and the
module's own import graph contains no torch.  See
``molmetal/tests/test_wfa155_metal_coordination_sphere.py`` for the static
import check.)

Revision history (this file is NEW in Wf-A155 — there is no pre-existing
rendering to strike; the entries below record the design changes made
while the branch was in flight, dated, before publication)
----------------------------------------------------------------------
* 2026-11-01 — the oxidation-state rule was split into three labelled
  layers (A: formal charge; B: ionic approximation with a monatomic-X
  scope; C: the repo prior) after the first draft collapsed to a single
  silent rule that returned 0 for every row.
* 2026-11-01 — the PRIMARY coordination classification was moved to the
  PRIOR-based capacity (the metal's USUAL oxidation state) and the
  structure-derived variant demoted to a clearly-labelled SECONDARY
  check, because Layer B under-estimates the oxidation state of a complex
  carrying a single X-type donor and so spuriously flagged octahedral
  Ir(III)/Rh(III) centres as over-coordinated.
* 2026-11-01 — Layer B's scope was tightened: a row is now OUT OF SCOPE
  whenever the metal's fragment holds a charged atom that is not one of the
  counted monatomic anionic donors (a polyatomic anionic ligand such as a
  carboxylate written ``[O-]``), instead of silently returning a number.
* 2026-11-01 — **Wf-A164 REPAIR (the pool-specific-claim defect class).**
  Wf-A163's census flagged this file for embedding POOL-SPECIFIC CLAIMS as
  string literals inside a ``tag: MEASURED`` artifact: true for the default
  pool, FALSE for any other.  Every such literal is now COMPUTED from the
  run (row count, pool name, break count, the pool's own single-fragment
  fraction, the cross-referenced published metric), the "92.7-94 %" band is
  labelled a FRONTIER CONSTANT and cited to its PRIMARY RECORD
  (:data:`FRONTIER_BAND_CITATION`), and a NARRATIVE SELF-CONSISTENCY GUARD
  (:func:`_narrative_self_consistency`) now FAILS the run when a number
  rendered in a narrative string is not backed by a machine field of the
  same artifact.  The tool's own docstring, ``purpose``,
  ``population.pool_kind``, the ``findings`` statements and the
  "implicit H / dative bond / explicit hydrogens" claims were in the same
  class and were repaired with them.

Run:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \\
        molmetal/scripts/wf_a155_metal_coordination_sphere.py \\
        --pool molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv \\
        --out metrics/by_metric/wfa155_metal_coordination_sphere.json

Exit codes: 0 = ran, 1 = usage/IO error OR the narrative self-consistency
guard FAILED (the artifact is then written with ``status: FAIL`` and the
unbacked numbers listed, and no claim is published).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# CPU-only guard (AGENTS.md §5.1 standing grant is SUSPENDED for Wf-A155).
# Set BEFORE RDKit is imported so nothing can pick a device up.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402

from molmetal.data.coord_geometry import classify_coord_geometry  # noqa: E402
from molmetal.data.metal_smiles import coordination_capacity  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The published pool this branch measures (post-hoc single decode).
DEFAULT_POOL = (
    PROJECT_ROOT / "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"
)

#: The artifact this branch publishes.
DEFAULT_OUT = PROJECT_ROOT / "metrics/by_metric/wfa155_metal_coordination_sphere.json"

#: Metal symbol -> atomic number.  Verbatim metal subset of the trainer's
#: ``_METAL_SYMBOL_TO_Z`` (``molmetal/scripts/train_cfm_yuelbond_wasserstein.py:11384``).
#: NOTE: oxygen is NOT a metal; an early exploratory pass of this script that
#: included ``O: 8`` counted 3231 spurious "oxygen metals" and is fixed here.
METAL_Z_BY_SYMBOL: Dict[str, int] = {
    "Pt": 78,
    "Pd": 46,
    "Au": 79,
    "Ag": 47,
    "Cu": 29,
    "Ni": 28,
    "Ru": 44,
    "Rh": 45,
    "Ir": 77,
    "Os": 76,
    "Re": 75,
    "Fe": 26,
    "Co": 27,
    "Mn": 25,
    "Cr": 24,
    "Mo": 42,
    "W": 74,
    "Ti": 22,
    "V": 23,
    "Zr": 40,
    "Hf": 72,
    "Nb": 41,
    "Ta": 73,
    "Sc": 21,
    "Y": 39,
    "La": 57,
    "Zn": 30,
    "Cd": 48,
    "Hg": 80,
}
_SYMBOL_BY_Z: Dict[int, str] = {z: s for s, z in METAL_Z_BY_SYMBOL.items()}

#: Per-metal DEFAULT oxidation state — the repo's own PRIOR, copied verbatim
#: from ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py:11399``
#: (``_DEFAULT_OXIDATION_STATE``).  This is a PRIOR, NOT a measurement and NOT
#: derived from any SMILES.
DEFAULT_OXIDATION_STATE: Dict[str, int] = {
    "Pt": 2,
    "Pd": 2,
    "Au": 3,
    "Ru": 2,
    "Rh": 3,
    "Ir": 3,
    "Os": 2,
    "Re": 3,
    "Fe": 2,
    "Cu": 2,
    "Ni": 2,
    "Zn": 2,
}

#: Donor class per element symbol.  The brief asks for O / N / C / halide;
#: S, P, Se and "other" are kept because they are real donor atoms in this
#: pool (S 138, P 521, Se 13 metal-adjacent) and dropping them would hide
#: them.
DONOR_CLASS_BY_SYMBOL: Dict[str, str] = {
    "N": "N_donor",
    "O": "O_donor",
    "C": "C_donor",
    "F": "halide",
    "Cl": "halide",
    "Br": "halide",
    "I": "halide",
    "S": "S_donor",
    "P": "P_donor",
    "Se": "Se_donor",
}

#: Donor classes that are NOT carbon (the "classical" donor set of the
#: Pt(II) / Ru(II) / Ir(III) metallodrug literature).
NON_CARBON_DONOR_CLASSES: Tuple[str, ...] = (
    "N_donor",
    "O_donor",
    "halide",
    "S_donor",
    "P_donor",
    "Se_donor",
)

#: Halide symbols (X-type donors).  A monatomic halide ligand has
#: oxidation state -1 in the ionic approximation.
HALIDE_SYMBOLS: frozenset = frozenset({"F", "Cl", "Br", "I"})

#: Every element that can carry a lone pair into the metal's sphere.
#: Used to count donors that got split OFF into another fragment.
DONOR_SYMBOLS: frozenset = frozenset({"N", "O", "S", "P", "Se", "F", "Cl", "Br", "I"})

#: The three oxidation-state layers.  ALL THREE ARE RULE-BASED INFERENCE.
OXIDATION_STATE_LAYERS: Dict[str, str] = {
    "os_formal_charge": (
        "Layer A — ionic-approximation identity. In a complex written in the "
        "IONIC convention the metal's formal charge in the SMILES IS its "
        "oxidation state, because every ligand's ionic charge already sits on "
        "the ligand. OS_A(M) = q(M). Under the COVALENT convention (every atom "
        "written neutral) this returns 0, which is NOT evidence of a metal(0) "
        "centre — it is the ABSENCE of charge information. ASSUMED RULE, not a "
        "measurement of the oxidation state."
    ),
    "os_charge_balanced": (
        "Layer B — ionic-approximation with a MONATOMIC-X scope. "
        "OS_B(M) = Q_row + sum(donor contributions), where Q_row is the "
        "total formal charge of the row's molecule and each donor "
        "contributes the negation of its LIGAND's oxidation state: a "
        "monatomic halide donor (X-type) contributes +1 (OS = -1), a "
        "neutral L-type donor (amine / pyridine / water / phosphine) "
        "contributes 0, and an anionic donor carrying formal charge -1 "
        "that is the ONLY charged atom in the metal's fragment contributes "
        "+1 (a monatomic anionic X-type ligand, OS = -1). This is exact for "
        "the classic covalent-written cis-[M(NH3)2X2] case (two halides -> "
        "+2) and for a metal carbanion. It is OUT OF SCOPE for any row "
        "whose metal's fragment holds a charged atom that is not one of "
        "those monatomic anionic donors (e.g. a carboxylate written [O-], "
        "whose ligand oxidation state is NOT the sum of its atoms' formal "
        "charges) — such a row reports os_charge_balanced = null rather "
        "than a number. When the computed value is not positive it is "
        "reported but NOT usable as an oxidation state "
        "(os_charge_balanced_usable = false). ASSUMED RULE, not a "
        "measurement."
    ),
    "os_prior_default": (
        "Layer C — the repo's own DEFAULT oxidation state per metal, copied "
        "verbatim from molmetal/scripts/train_cfm_yuelbond_wasserstein.py:11399 "
        "(_DEFAULT_OXIDATION_STATE). A PRIOR chosen by the repo; NOT derived "
        "from any SMILES and NOT a measurement. Used only as the fallback "
        "when Layers A and B are silent."
    ),
}

# ---------------------------------------------------------------------------
# The published frontier band — a FRONTIER CONSTANT, never one of our numbers
# ---------------------------------------------------------------------------

#: The published band this metric is NOT comparable to.  It is an EXTERNAL,
#: PUBLISHED quantity: it is quoted, labelled as a frontier constant, cited to
#: its primary record, and never blended with a number measured here.
FRONTIER_BAND = "92.7-94 %"

#: The denominator of the two quoted endpoints, in the primary record's own
#: words ("10,000 unconditional QM9 samples").
FRONTIER_DENOMINATOR = "10,000 unconditional QM9 samples"

#: PRIMARY-RECORD provenance for the band.  Every field is either a pointer
#: to a record on disk or a quote read out of that record.
#:
#: Wf-A164 re-fetched the primary PDF live and re-verified both quotes; the
#: ``retrieval_sha256`` below is the sha256 of the file fetched in THAT
#: session, so the quotes can be re-checked without trusting this file.
#: The CANONICAL reading-count clause shipped by Wf-A196 (2026-11-02),
#: copied CHARACTER-FOR-CHARACTER from the Wf-A196 record
#: `molmetal/reports/wfa175_synthesis/_write_md.py` ->
#: `_X5_COUNT_NOT_SETTLED`.
#: WHY it is reused here: a live leaf that renders "TWO competing
#: readings" / "under BOTH readings" asserts a SETTLED COUNT of
#: competing readings, while the repo's own primary-source
#: verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` ->
#: `invariants`.`competing_readings`) records a FURTHER reading that
#: does not reproduce the quoted numerals.  The NUMBER of competing
#: readings is itself NOT settled on disk, so no live leaf may assert
#: a count.  The character-for-character copy is VERIFIED by
#: molmetal/tests/test_wfa202_sphere_reading_closure.py ::
#: test_the_producer_clause_is_character_for_character_the_wfa196_clause
WFA196_COUNT_NOT_SETTLED = (
    "The NUMBER of competing readings is itself NOT settled on disk: this record enumerates "
    "two, and the repo's primary-source verification "
    "(`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> "
    "`invariants.competing_readings`) records a further reading that likewise does not reproduce "
    "the quoted numerals."
)


FRONTIER_BAND_CITATION: Dict[str, object] = {
    "band_as_quoted_in_the_repo": FRONTIER_BAND,
    "denominator": FRONTIER_DENOMINATOR,
    # Wf-A177 (2026-11-01) CORRECTION 2 — the superseded text, PRESERVED
    # VERBATIM per AGENTS.md §5.5 rule #6, and itself SUPERSEDED:
    #     # Wf-A202 (2026-11-02) X5 REPAIR -- the live rendering asserted a
    # SETTLED COUNT of competing readings ("TWO COMPETING ...", "under
    # BOTH readings"), while the repo's own primary-source verification
    # records a FURTHER reading (see WFA196_COUNT_NOT_SETTLED above,
    # copied CHARACTER-FOR-CHARACTER from the Wf-A196 record).  The live
    # rendering is now READING-CONDITIONAL: it names the competing
    # readings by ROBUST JSON PATHS and states only what holds under
    # EVERY reading.  The prior rendering is preserved VERBATIM in the
    # dated sibling WHAT_THE_BAND_IS_SUPERSEDED_2026_11_02_WFA202
    # (AGENTS.md §5.5 rule #6).
    "what_the_band_is": (
        "COMPETING PROVENANCE READINGS OF WHICH FRONTIER CELLS THE BAND "
        "NAMES, NOT ONE HOMOGENEOUS QUANTITY: the frontier post-processing "
        "matrix record (`frontier_matrix.json`) holds COMPETING readings of "
        "which frontier cells the published band names -- among them the "
        "readings listed at the JSON path "
        "`band_provenance`.`competing_reading_and_what_this_does_not_settle`."
        "`readings` and the reading at the JSON path `sources`.`EF-TALFM` -- "
        "and they do not even agree on the numerals, so the band's source "
        "cells are NOT settled on disk. "
        + WFA196_COUNT_NOT_SETTLED
        + " What holds under EVERY reading: the band is EXTERNAL, is NOT one "
        "homogeneous quantity, and is NOT on our scale (OUR number is a "
        "connected-component count on the RAW emitted SMILES, and nothing "
        "else), so it is not comparable to a coordination-sphere profile."
    ),
    "primary_record": {
        "arxiv_id": "2603.27113",
        "version": "v1",
        "title": (
            "Hierarchy-Guided Topology Latent Flow for Molecular Graph "
            "Generation"
        ),
        "url": "https://arxiv.org/abs/2603.27113v1",
        "locator_low_endpoint": (
            "Table 1(b), row 'Val&Uniq (%)', column 'GeoLDM' — the endpoint is "
            "a BASELINE row, not HLTF's own number"
        ),
        "quote_low_endpoint": (
            "Val&Uniq (%) | 92.7 ± 0.5 | 93.3 ± 0.0 * | 92.9 ± 0.2"
        ),
        "locator_high_endpoint": (
            "Table 1(b), row 'PB-Valid (%)', column 'HLTF (ours)'"
        ),
        "quote_high_endpoint": (
            "PB-Valid (%) | 93.1 ± 0.4 | 91.9 ± 0.5 | 94 ± 0.5 *"
        ),
        "denominator_locator": "Section 4.1 'Benchmarks, protocols, and metrics'",
        "abstract_own_text": (
            "On QM9, HLTF achieves 98.8% atom stability and 92.9% "
            "valid-and-unique, improving PoseBusters validity to 94.0% (+0.9 "
            "over the strongest reported baseline)."
        ),
        "arms": (
            "the two quoted endpoints are the QM9 Table 1(b) numbers; the "
            "record's own words for the QM9 protocol are 'On QM9: we report "
            "atom stability (AS), molecule stability (MS), RDKit validity "
            "(Val), and Val&Uniq on 10,000 unconditional samples.' The "
            "post-processed arm is published for GEOM-DRUGS (Table 2, the "
            "'+PP' rows): 'both raw and after standardized post-processing "
            "(largest fragment, explicit hydrogens, UFF relaxation)'."
        ),
        "retrieved_at": "2026-11-01",
        "retrieval_url": "https://arxiv.org/pdf/2603.27113v1",
        "retrieval_sha256": (
            "03cc2f6c93c2603ec1cbe0c977c0b54d7ffc32922287bdc993441477763b1da2"
        ),
        "verification": (
            "RE-VERIFIED LIVE in Wf-A164: both quotes above were read out of "
            "Table 1(b) of the fetched primary PDF (sha256 above). The 92.7 % "
            "endpoint is NOT in the abstract — the abstract's own numbers are "
            "98.8 / 92.9 / 94.0 — it is the GeoLDM column of Table 1(b)."
        ),
    },
    # Wf-A202 (2026-11-02) D2 REPAIR -- the live rendering asserted a
    # SETTLED NON-ATTRIBUTION ("EF-TALFM must NOT be cited for this
    # band"), while the SAME artefact's live `what_the_band_is` and the
    # frontier-matrix record hold EF-TALFM as ONE OF THE COMPETING
    # PROVENANCE READINGS for the band.  The non-attribution's premise
    # (EF-TALFM's ABSTRACT says 87.9 %) is true only of the ABSTRACT,
    # while the competing reading names EF-TALFM's OWN Table 3(a) cells.
    # The live rendering is now READING-CONDITIONAL: it states what holds
    # under EVERY reading and names the competing readings by ROBUST JSON
    # PATHS.  The prior rendering is preserved VERBATIM in the dated
    # sibling NOT_ATTRIBUTED_TO_EF_TALFM_SUPERSEDED_2026_11_02_WFA202
    # (AGENTS.md §5.5 rule #6).
    "not_attributed_to_ef_talfm": (
        "READING-CONDITIONAL (Wf-A202, 2026-11-02): EF-TALFM's ABSTRACT is "
        "NOT a source of this band -- the frontier post-processing matrix "
        "record carries EF-TALFM's abstract wording (the 87.9 % "
        "unique-and-novel figure on PCQM4Mv2) at the JSON path "
        "`retrieval_log`[1].`extracted`, and 87.9 % is not one of this band's "
        "figures. BUT the band's source cells are NOT settled on disk, and "
        "EF-TALFM is ONE OF THE COMPETING PROVENANCE READINGS for them: the "
        "competing reading at the JSON path `sources`.`EF-TALFM` names "
        "EF-TALFM's OWN Table 3(a) cells -- recorded at the JSON paths "
        "`sources`.`EF-TALFM`.`reported_band` and "
        "`sources`.`EF-TALFM`.`reported_numbers` -- NOT its abstract. It is "
        "therefore NOT settled that EF-TALFM must not be cited for this band; "
        "only the claims that hold under EVERY reading may be published. What "
        "holds under EVERY reading: EF-TALFM's ABSTRACT does not state this "
        "band; the band's source cells are NOT settled on disk; the band is "
        "EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale."
    ),
    "repo_provenance": (
        "the frontier post-processing matrix record "
                "(`frontier_matrix.json`)"
    ),
}


# ---------------------------------------------------------------------------
# The comparability statement and the "what this is NOT" block — COMPUTED
# ---------------------------------------------------------------------------


#: Wf-A189 (2026-11-01) -- the dated rule-#6 records this
#: artefact carries.  AGENTS.md §5.5 rule #6 requires every
#: superseded rendering to survive VERBATIM; emitting them
#: HERE is what makes a re-run REPRODUCE the published
#: artefact instead of silently DROPPING the dated history.

BREAK_3_POST_PROCESSING_SUPERSEDED_2026_11_01_WFA176: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A176',
    'rule': 'AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'supersedes_break_id': 'break_3_post_processing',
    'field': 'break_3_post_processing (the whole leaf)',
    'rendering': 'POST-PROCESSING — the published band is measured on the published POST-PROCESSED stream; the pool measured here is RAW (no post-processing, no repair).',
    'why_superseded': 'the pre-Wf-A177 live break_3 rendering asserted that the published band is measured on the published POST-PROCESSED stream.  That rendering was replaced during Wf-A176/Wf-A177 and rule #6 requires the superseded text to survive VERBATIM in the artefact; it was NOT preserved here - it survived only in the historical report molmetal/reports/wfa175_band_provenance/report.md.  This key closes that gap.',
    'source_of_this_rendering': '/tmp/wfa177_old_sphere.json (md5 2ffa508b) -- the OLDEST of the two /tmp baselines of this wave; molmetal/reports/wfa175_band_provenance/report.md carries the same text',
    'baseline_disagreement': 'the two /tmp baselines of this wave DISAGREE: /tmp/wfa177_old_sphere.json (md5 2ffa508b) carries the pre-Wf-A177 rendering kept here, while /tmp/wfa179_old_sphere.json (md5 08e37ab6) carries the later Wf-A177 rendering.  The OLDEST was used, per the instruction of record.',
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}

BREAK_3_POST_PROCESSING_SUPERSEDED_2026_11_01_WFA183: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A183',
    'rule': 'AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'supersedes_break_id': 'break_3_post_processing',
    'field': 'frontier',
    'rendering': "the frontier publishes the post-processing axis in ONE table -- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW and POST-PROCESSED side by side -- so 'the frontier number' is one of two numbers depending on the arm; the other frontier papers print a single arm. The published 92.7-94 % band's two endpoints are both RAW (post_processed: false); which frontier cells the band names is NOT settled on disk (two competing provenance readings, both in molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -- the first under `band_provenance`, the second under `sources`.`EF-TALFM` -- that do not even agree on the numerals), but under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale.",
    'why_superseded': "the superseded rendering asserted the published band's whole-band RAW-ness UNCONDITIONALLY.  That holds only under the `band_provenance` reading of molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json; the competing `sources`.`EF-TALFM` reading of the same record pairs a raw connectivity cell with a post-processed companion cell, so whether the band as a whole is RAW is NOT settled on disk.  This is the MIRROR of the blanket POST-PROCESSED over-claim Wf-A175 already refuted.  The live rendering is now READING-CONDITIONAL; it introduces NO numeral and keeps every numeral the superseded rendering carried.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.  The emitter was re-run on the SAME published pool and its output matched this artefact in every numeric leaf; the only differences are the narrative strings repaired here and the `reproduce` block's own --out path, which records the temporary re-run location (the published path is retained).",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}

#: Wf-A189 (2026-11-01) -- the rule-#6 RESTORATION of the
#: Wf-A179-era renderings Wf-A183 replaced but preserved
#: NOWHERE.  Read out of the ephemeral /tmp baseline BY
#: THEIR JSON PATHS (never retyped); see each block's
#: `source_of_this_rendering` field.

BREAK_3_POST_PROCESSING_SUPERSEDED_2026_11_01_WFA189: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A189',
    'rule': 'AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'supersedes': 'comparability_statement.break_3_post_processing',
    'field': 'break_3_post_processing (the whole leaf)',
    'rendering': "POST-PROCESSING — the frontier publishes the post-processing axis in ONE table -- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW and POST-PROCESSED side by side -- so 'the frontier number' is one of two numbers depending on the arm; the other frontier papers print a single arm. The published 92.7-94 % band's two endpoints are both RAW (post_processed: false); which frontier cells the band names is NOT settled on disk (two competing provenance readings, both in molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -- under `band_provenance` and under `frontier_axis_mapping` -- that do not even agree on the numerals), but under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale. the pool measured here is RAW (no post-processing, no repair), so it is comparable only to a RAW frontier arm, never to a post-processed one.",
    'why_superseded': 'Wf-A183 (2026-11-01) replaced the LIVE rendering of `comparability_statement.break_3_post_processing` and preserved the Wf-A176-era rendering in the sibling key `comparability_statement.break_3_post_processing_superseded_2026_11_01_wfa176` and the Wf-A177-era rendering in the sibling key `comparability_statement.break_3_post_processing_superseded_2026_11_01_wfa183`.  The IMMEDIATELY-PRECEDING rendering - the Wf-A179-era wording that Wf-A183 itself replaced - survived NOWHERE on disk, so the rule-#6 chain had a hole: a re-run of the producer would have dropped the whole dated record and the published artefact would have become hand-patched and non-reproducible.  This key closes the hole by restoring that rendering VERBATIM.  The restored rendering cites the locator `frontier_axis_mapping` for the second provenance reading; per the Wf-A183 record that token is NOT a JSON key of the frontier post-processing matrix record (it survives only as a string value), which is exactly why the LIVE rendering cites JSON PATHS.  The rendering is restored VERBATIM anyway: rule #6 preserves what WAS published, and the correction lives in the dated live rendering, never in a silent edit of history.',
    'source_of_this_rendering': "/tmp/wfa179_old_sphere.json (md5 08e37ab616e02088a7a5a7548a4ddaa9), read out at the JSON path `comparability_statement.break_3_post_processing` - the Wf-A179-era state of this artefact.  That file is EPHEMERAL (/tmp is not durable; a reboot erases it) and this rendering is NOT recoverable from ANY tracked file: a verbatim containment search over the repo's tracked + untracked + gitignored text files found ZERO other holders of this string.  The repo's rule-#6 chain therefore rested on a hand-copied /tmp baseline, which is why a re-run could have erased the history with nobody noticing.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}

BREAK_3_FRONTIER_SUPERSEDED_2026_11_01_WFA189: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A189',
    'rule': 'AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'supersedes': 'comparability_statement.breaks[2].frontier',
    'field': "frontier (the third comparability break's `frontier` field)",
    'rendering': "the frontier publishes the post-processing axis in ONE table -- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW and POST-PROCESSED side by side -- so 'the frontier number' is one of two numbers depending on the arm; the other frontier papers print a single arm. The published 92.7-94 % band's two endpoints are both RAW (post_processed: false); which frontier cells the band names is NOT settled on disk (two competing provenance readings, both in molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -- under `band_provenance` and under `frontier_axis_mapping` -- that do not even agree on the numerals), but under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale.",
    'why_superseded': "Wf-A183 (2026-11-01) replaced the LIVE rendering of `comparability_statement.breaks[2].frontier` and preserved the Wf-A177-era wording in the sibling key `comparability_statement.break_3_post_processing_superseded_2026_11_01_wfa183`.  The IMMEDIATELY-PRECEDING rendering - the Wf-A179-era wording that Wf-A183 itself replaced - survived NOWHERE on disk, so the rule-#6 chain had a hole: a re-run of the producer would have dropped the whole dated record and the published artefact would have become hand-patched and non-reproducible.  This key closes the hole by restoring that rendering VERBATIM.  It is carried as a SIBLING key of `breaks` (not as a fourth field of the third break) because the repo's own regression test molmetal/tests/test_wfa162_narrative_self_consistency.py asserts that `comparability.breaks` resolves to exactly four breaks x (break, frontier, ours).  The restored rendering cites the locator `frontier_axis_mapping` for the second provenance reading; per the Wf-A183 record that token is NOT a JSON key of the frontier post-processing matrix record, which is exactly why the LIVE rendering cites JSON PATHS.  The rendering is restored VERBATIM anyway: rule #6 preserves what WAS published, and the correction lives in the dated live rendering, never in a silent edit of history.",
    'source_of_this_rendering': "/tmp/wfa179_old_sphere.json (md5 08e37ab616e02088a7a5a7548a4ddaa9), read out at the JSON path `comparability_statement.breaks[2].frontier` - the Wf-A179-era state of this artefact.  That file is EPHEMERAL (/tmp is not durable; a reboot erases it) and this rendering is NOT recoverable from ANY tracked file: a verbatim containment search over the repo's tracked + untracked + gitignored text files found ZERO other holders of this string.  The repo's rule-#6 chain therefore rested on a hand-copied /tmp baseline, which is why a re-run could have erased the history with nobody noticing.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}

WHAT_THE_BAND_IS_SUPERSEDED_2026_11_01_WFA189: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A189',
    'rule': 'AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'supersedes': 'frontier_reference.what_the_band_is',
    'field': 'what_the_band_is',
    'rendering': "TWO COMPETING PROVENANCE READINGS, NOT ONE HOMOGENEOUS QUANTITY: both are recorded in the frontier post-processing matrix record (`frontier_matrix.json`) -- one under `band_provenance`, one under `frontier_axis_mapping` -- and they do not even agree on the numerals, so the band's source cells are NOT settled on disk. What holds under BOTH readings: the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale (OUR number is a connected-component count on the RAW emitted SMILES, and nothing else), so it is not comparable to a coordination-sphere profile.",
    'why_superseded': 'Wf-A183 (2026-11-01) replaced the LIVE rendering of `frontier_reference.what_the_band_is` and preserved no dated record of its own predecessor at that path.  The IMMEDIATELY-PRECEDING rendering - the Wf-A179-era wording that Wf-A183 itself replaced - survived NOWHERE on disk, so the rule-#6 chain had a hole: a re-run of the producer would have dropped the whole dated record and the published artefact would have become hand-patched and non-reproducible.  This key closes the hole by restoring that rendering VERBATIM.  It is carried as a SIBLING key of `frontier_reference` so that the live `what_the_band_is` field keeps its own shape.  The restored rendering cites the locator `frontier_axis_mapping` for the second provenance reading; per the Wf-A183 record that token is NOT a JSON key of the frontier post-processing matrix record, which is exactly why the LIVE rendering cites JSON PATHS.  The rendering is restored VERBATIM anyway: rule #6 preserves what WAS published, and the correction lives in the dated live rendering, never in a silent edit of history.',
    'source_of_this_rendering': "/tmp/wfa179_old_sphere.json (md5 08e37ab616e02088a7a5a7548a4ddaa9), read out at the JSON path `frontier_reference.what_the_band_is` - the Wf-A179-era state of this artefact.  That file is EPHEMERAL (/tmp is not durable; a reboot erases it) and this rendering is NOT recoverable from ANY tracked file: a verbatim containment search over the repo's tracked + untracked + gitignored text files found ZERO other holders of this string.  The repo's rule-#6 chain therefore rested on a hand-copied /tmp baseline, which is why a re-run could have erased the history with nobody noticing.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}


#: Wf-A202 (2026-11-02) -- the dated rule-#6 records this wave adds to
#: `frontier_reference`.  Each block preserves the PRIOR rendering
#: VERBATIM and states WHY it was superseded (AGENTS.md §5.5 rule #6).

WHAT_THE_BAND_IS_SUPERSEDED_2026_11_02_WFA202: Dict[str, str] = {
    'date': '2026-11-02',
    'wave': 'Wf-A202',
    'rule': "AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated",
    'supersedes': "frontier_reference.what_the_band_is",
    'field': "what_the_band_is",
    'rendering': "TWO COMPETING PROVENANCE READINGS, NOT ONE HOMOGENEOUS QUANTITY: both are recorded in the frontier post-processing matrix record (`frontier_matrix.json`) -- the first under `band_provenance`, the second under `sources`.`EF-TALFM` -- and they do not even agree on the numerals, so the band's source cells are NOT settled on disk. What holds under BOTH readings: the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale (OUR number is a connected-component count on the RAW emitted SMILES, and nothing else), so it is not comparable to a coordination-sphere profile.",
    'why_superseded': "the live rendering asserted a SETTLED COUNT of competing provenance readings -- \"TWO COMPETING PROVENANCE READINGS\" and \"What holds under BOTH readings\" -- while the repo's own primary-source verification (molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json -> `invariants`.`competing_readings`) records a FURTHER reading that likewise does not reproduce the quoted numerals.  The NUMBER of competing readings is itself NOT settled on disk, so a live leaf may not assert a count.  The live rendering is now READING-CONDITIONAL: it names the competing readings by ROBUST JSON PATHS, carries the Wf-A196 canonical reading-count clause CHARACTER-FOR-CHARACTER, and states only what holds under EVERY reading.",
    'source_of_this_rendering': "the published artefact metrics/by_metric/wfa155_metal_coordination_sphere.json at sha256 55034ba3e6376bb887c9543f5ea2550208f5294f332c3c76e1e4727ca5eeb94b (the state on disk before the Wf-A202 repair), read out at the JSON path `frontier_reference.what_the_band_is`.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}

NOT_ATTRIBUTED_TO_EF_TALFM_SUPERSEDED_2026_11_02_WFA202: Dict[str, str] = {
    'date': '2026-11-02',
    'wave': 'Wf-A202',
    'rule': "AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated",
    'supersedes': "frontier_reference.not_attributed_to_ef_talfm",
    'field': "not_attributed_to_ef_talfm",
    'rendering': "EF-TALFM (arXiv:2609.08333v4) must NOT be cited for this band: its own abstract says 'unconditional generation yields 87.9% unique and novel molecules passing sanitization and PoseBusters checks' on PCQM4Mv2 — 87.9 %, not 92.7-94 %.",
    'why_superseded': "the live rendering asserted a SETTLED NON-ATTRIBUTION -- EF-TALFM \"must NOT be cited for this band\" -- while the SAME artefact's live `what_the_band_is` and the frontier-matrix record hold EF-TALFM as ONE OF THE COMPETING PROVENANCE READINGS for the band.  The non-attribution's premise (EF-TALFM's ABSTRACT says 87.9 %) is true only of the ABSTRACT; the competing reading named at the JSON path `sources`.`EF-TALFM` names EF-TALFM's OWN Table 3(a) cells, not its abstract, so the exclusion of EF-TALFM is itself NOT settled on disk.  The live rendering is now READING-CONDITIONAL: it states what holds under EVERY reading and names the competing readings by ROBUST JSON PATHS.",
    'source_of_this_rendering': "the published artefact metrics/by_metric/wfa155_metal_coordination_sphere.json at sha256 55034ba3e6376bb887c9543f5ea2550208f5294f332c3c76e1e4727ca5eeb94b (the state on disk before the Wf-A202 repair), read out at the JSON path `frontier_reference.not_attributed_to_ef_talfm`.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}

PRIMARY_RECORD_READING_CONDITIONAL_2026_11_02_WFA202: Dict[str, str] = {
    'date': '2026-11-02',
    'wave': 'Wf-A202',
    'rule': 'AGENTS.md §5.5 rule #6 - the correction is additive + dated; the existing leaves are NOT edited',
    'qualifies': 'frontier_reference.primary_record',
    'field': 'primary_record',
    'annotation': "READING-CONDITIONAL (Wf-A202, 2026-11-02): the cells quoted by `frontier_reference.primary_record` -- Table 1(b) row 'Val&Uniq (%)', column 'GeoLDM', and Table 1(b) row 'PB-Valid (%)', column 'HLTF (ours)' of arXiv:2603.27113v1 -- are MEASURED facts about THAT paper's Table 1(b) (retrieval_sha256 above) and they are NOT edited here. WHETHER THOSE CELLS ARE THE CELLS THE PUBLISHED BAND NAMES IS NOT SETTLED ON DISK: the competing provenance readings are recorded in molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json at the JSON paths `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings` and `sources`.`EF-TALFM`; a further candidate for the same numeral is recorded at the JSON path `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings`[0].`endpoints`.`third_candidate_for_the_same_numeral`; and the repo's primary-source verification records further competing readings at the JSON path molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json -> `invariants`.`competing_readings`. What holds under EVERY reading: the band's source cells are NOT settled on disk, so the quotes above MUST NOT be presented as the band's settled source cells; the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale.",
    'what_is_unchanged': "the MEASURED leaves of `primary_record` -- `quote_low_endpoint`, `quote_high_endpoint`, `verification` and the rest -- survive BYTE-IDENTICAL: they are facts about Table 1(b) of arXiv:2603.27113v1 (retrieval_sha256 03cc2f6c93c2603ec1cbe0c977c0b54d7ffc32922287bdc993441477763b1da2), and this key qualifies only WHETHER THOSE CELLS ARE THE CELLS THE PUBLISHED BAND NAMES.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}


#: Wf-A202 (2026-11-02) -- the dated rule-#6 records for the
#: comparability statement's third break.  Both prior renderings are
#: preserved VERBATIM; both asserted a SETTLED COUNT of competing
#: provenance readings ("two competing provenance readings", "under
#: BOTH readings").

BREAK_3_POST_PROCESSING_SUPERSEDED_2026_11_02_WFA202: Dict[str, str] = {
    'date': '2026-11-02',
    'wave': 'Wf-A202',
    'rule': "AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated",
    'supersedes': "comparability_statement.break_3_post_processing",
    'field': "break_3_post_processing (the whole leaf)",
    'rendering': "POST-PROCESSING — the frontier publishes the post-processing axis in ONE table -- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW and POST-PROCESSED side by side -- so 'the frontier number' is one of two numbers depending on the arm; the other frontier papers print a single arm. The published 92.7-94 % band's two endpoints are RAW under the `band_provenance` reading (post_processed: false in that reading), while the competing `sources`.`EF-TALFM` reading pairs a raw connectivity cell with a post-processed companion cell -- so whether the band as a whole is RAW is NOT settled on disk; which frontier cells the band names is NOT settled on disk (two competing provenance readings, both in molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -- the first under `band_provenance`, the second under `sources`.`EF-TALFM` -- that do not even agree on the numerals), but under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale. the pool measured here is RAW (no post-processing, no repair), so it is comparable only to a RAW frontier arm, never to a post-processed one.",
    'why_superseded': "the live rendering asserted a SETTLED COUNT of competing provenance readings -- \"two competing provenance readings\" and \"but under BOTH readings\" -- while the repo's own primary-source verification records a FURTHER reading that likewise does not reproduce the quoted numerals.  The NUMBER of competing readings is itself NOT settled on disk.  The live rendering is now READING-CONDITIONAL: it names the competing readings by ROBUST JSON PATHS and states only what holds under EVERY reading.  The repair introduces NO numeral and keeps every numeral the superseded rendering carried.",
    'source_of_this_rendering': "the published artefact metrics/by_metric/wfa155_metal_coordination_sphere.json at sha256 55034ba3e6376bb887c9543f5ea2550208f5294f332c3c76e1e4727ca5eeb94b (the state on disk before the Wf-A202 repair), read out at the JSON path `comparability_statement.break_3_post_processing`.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}

BREAKS_2_FRONTIER_SUPERSEDED_2026_11_02_WFA202: Dict[str, str] = {
    'date': '2026-11-02',
    'wave': 'Wf-A202',
    'rule': "AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated",
    'supersedes': "comparability_statement.breaks[2].frontier",
    'field': "frontier (the third comparability break's `frontier` field)",
    'rendering': "the frontier publishes the post-processing axis in ONE table -- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW and POST-PROCESSED side by side -- so 'the frontier number' is one of two numbers depending on the arm; the other frontier papers print a single arm. The published 92.7-94 % band's two endpoints are RAW under the `band_provenance` reading (post_processed: false in that reading), while the competing `sources`.`EF-TALFM` reading pairs a raw connectivity cell with a post-processed companion cell -- so whether the band as a whole is RAW is NOT settled on disk; which frontier cells the band names is NOT settled on disk (two competing provenance readings, both in molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -- the first under `band_provenance`, the second under `sources`.`EF-TALFM` -- that do not even agree on the numerals), but under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale.",
    'why_superseded': "the live rendering asserted a SETTLED COUNT of competing provenance readings -- \"two competing provenance readings\" and \"but under BOTH readings\" -- while the repo's own primary-source verification records a FURTHER reading that likewise does not reproduce the quoted numerals.  The NUMBER of competing readings is itself NOT settled on disk.  The live rendering is now READING-CONDITIONAL: it names the competing readings by ROBUST JSON PATHS and states only what holds under EVERY reading.  The repair introduces NO numeral and keeps every numeral the superseded rendering carried.",
    'source_of_this_rendering': "the published artefact metrics/by_metric/wfa155_metal_coordination_sphere.json at sha256 55034ba3e6376bb887c9543f5ea2550208f5294f332c3c76e1e4727ca5eeb94b (the state on disk before the Wf-A202 repair), read out at the JSON path `comparability_statement.breaks[2].frontier`.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}

def build_comparability_breaks(
    pool_name: str,
    n_rows: int,
    *,
    n_rows_with_explicit_h: int = 0,
) -> List[Dict[str, str]]:
    """The FOUR comparability breaks, with OUR side COMPUTED from the run.

    Every number that describes OUR population (the row count, the explicit-H
    row count) is passed in from the run; the frontier side is a quoted
    constant.  The list form matters for the narrative guard: a LIST-valued
    narrative path has to resolve to its ELEMENTS, or every number inside it
    would go unchecked.
    """
    if n_rows_with_explicit_h == 0:
        hydrogen_ours = (
            f"this pool carries NO explicit H atom in any of its {n_rows} "
            "rows — measured here, not assumed — so it is measured with "
            "implicit H, which changes both sanitisation and geometry, and "
            "therefore the coordination count."
        )
    else:
        hydrogen_ours = (
            f"this pool carries an explicit H atom in {n_rows_with_explicit_h} "
            f"of its {n_rows} rows — measured here, not assumed — so its "
            "hydrogen convention is NOT uniform across the pool, which "
            "changes both sanitisation and geometry, and therefore the "
            "coordination count."
        )
    return [
        {
            "id": "break_1_denominator",
            "axis": "DENOMINATOR",
            "frontier": (
                f"the frontier band is over {FRONTIER_DENOMINATOR}, sampled "
                "from the training-set distribution."
            ),
            "ours": (
                f"ours is ONE pool of {n_rows} rows from ONE post-hoc decode "
                f"(pool directory: {pool_name})."
            ),
        },
        {
            "id": "break_2_metric",
            "axis": "METRIC",
            "frontier": (
                "'connected' in the frontier band means the DECODER's own "
                "connectivity claim; their 'Valid' means RDKit sanitisation "
                "and their '% PB-Valid' means PoseBusters geometry."
            ),
            "ours": (
                "here 'connected' means an RDKit fragment count of the "
                "emitted SMILES, and the coordination sphere is the metal's "
                "RDKit bond neighbourhood — a quantity the frontier band does "
                "not report at all. Neither frontier quantity is a "
                "coordination quantity."
            ),
        },
        {
            # Wf-A176 (2026-11-01) CORRECTION — the superseded text, PRESERVED
            # VERBATIM per AGENTS.md §5.5 rule #6:
            #     "frontier": (
            #         "the frontier publishes the post-processing axis itself: the "
            #         "same paper reports every model twice, once RAW and once "
            #         "POST-PROCESSED, so 'the frontier number' is one of two "
            #         "numbers depending on the arm."
            #     ),
            #
            # WHY: TWO defects.  (a) OVER-GENERAL — the raw/post pair axis
            # exists in ONE table, not across the frontier: MEASURED,
            # `frontier_matrix.json`
            # `band_provenance.post_processing_evidence_from_the_frontier_itself`
            # = "THE FRONTIER PUBLISHES THE POST-PROCESSING AXIS ITSELF. Six
            # models, one table, raw and post-processed side by side, under
            # one pipeline."  (b) The 92.7-94 % band's two endpoints carry
            # `post_processed: false` in the same file's
            # `band_provenance.endpoint_candidates`, and NEITHER is a
            # connected-component count — so the post-processing axis is a
            # THIRD, separate axis on top of the METRIC-TYPE break, not the
            # break itself.
            #
            # Wf-A177 (2026-11-01) CORRECTION 2 — the correction-1 text below,
            # PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, and itself
            # SUPERSEDED:
            #     "frontier": (
            #         "the frontier publishes the post-processing axis in ONE table "
            #         "-- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW "
            #         "and POST-PROCESSED side by side -- so 'the frontier number' "
            #         "is one of two numbers depending on the arm; the other "
            #         "frontier papers print a single arm. The published 92.7-94 % "
            #         "band's two endpoints are both RAW (post_processed: false) "
            #         "and neither is a connected-component count."
            #     ),
            #
            # WHY correction 1 was SUPERSEDED (adversarial verify, Wf-A176):
            # it asserted "neither is a connected-component count" as if
            # settled.  The repo holds TWO competing provenance readings for
            # the band (frontier_matrix.json -> `band_provenance` :51-89 vs
            # the same file's `frontier_axis_mapping` :360/:378-379,
            # EF-TALFM arXiv:2609.08333v4) and under the SECOND the lower
            # endpoint IS a connectivity count (EF-TALFM's own raw `C.` cell,
            # 92.98 %).  The two readings do not even agree on the numerals
            # (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT
            # settled on disk.  Only the invariants that hold under BOTH
            # readings may be published.
            "id": "break_3_post_processing",
            "axis": "POST-PROCESSING",
            # Wf-A177 (2026-11-01) CORRECTION 3 — the correction-2 rendering
            # below, PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, and
            # itself SUPERSEDED:
            #     "frontier": (
            #         "the frontier publishes the post-processing axis in ONE table "
            #         "-- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW "
            #         "and POST-PROCESSED side by side -- so 'the frontier number' "
            #         "is one of two numbers depending on the arm; the other "
            #         "frontier papers print a single arm. The published 92.7-94 % "
            #         "band's two endpoints are both RAW (post_processed: false); "
            #         "which frontier cells the band names is NOT settled on disk "
            #         "(two competing provenance readings, frontier_matrix.json "
            #         "`band_provenance` :51-89 vs the same file's "
            #         "`frontier_axis_mapping` :360/:378-379, EF-TALFM "
            #         "arXiv:2609.08333v4), but under BOTH readings the band is "
            #         "external, is NOT one homogeneous quantity, and is NOT on our "
            #         "scale."
            #     ),
            #
            # WHY correction 2 was SUPERSEDED: the emitters' OWN narrative
            # self-consistency guard (the Wf-A162 class-level guard) REJECTED
            # it — the EF-TALFM arXiv id and the frontier_matrix.json line
            # numbers were backed by NO machine field of this artifact.  The
            # guard is right: a `tag: MEASURED` artifact must not render a
            # number it cannot back.  The fix is to keep the CLAIM and drop
            # the numerals from the PROSE — the concrete readings live in the
            # SOURCE RECORD named below, at the two named key paths.  This
            # keeps the correction TEXT-ONLY: 0 numeric leaves change and 0
            # keys are added.
            #
            # Wf-A180 (2026-11-01) — POINTER REPAIR.  The rendering below
            # pointed the reader at `frontier_axis_mapping`, which is NOT a
            # JSON key of molmetal/reports/wfa158_frontier_postproc_matrix/
            # frontier_matrix.json (a recursive key-walk finds that token
            # only as a STRING VALUE, twice).  The record's OWN
            # band_provenance.competing_reading_and_what_this_does_not_settle
            # .line_number_note states: "The ROBUST locators are the JSON
            # PATHS -- `band_provenance` and `sources`.`EF-TALFM` -- not the
            # line numbers."  The SUPERSEDED rendering is preserved VERBATIM
            # below (AGENTS.md §5.5 rule #6); the live string names the two
            # REAL key paths.  TEXT-ONLY: 0 numeric leaves change.
            #
            #     "-- under `band_provenance` and under `frontier_axis_mapping` "
            #
            # Wf-A183 (2026-11-01) CORRECTION 4 — the Wf-A180 pointer-repair
            # rendering below, PRESERVED VERBATIM per AGENTS.md §5.5 rule #6,
            # and itself SUPERSEDED:
            #     "frontier": (
            #         "the frontier publishes the post-processing axis in ONE table "
            #         "-- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW "
            #         "and POST-PROCESSED side by side -- so 'the frontier number' "
            #         "is one of two numbers depending on the arm; the other "
            #         "frontier papers print a single arm. The published 92.7-94 % "
            #         "band's two endpoints are both RAW (post_processed: false); "
            #         "which frontier cells the band names is NOT settled on disk "
            #         "(two competing provenance readings, both in molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json "
            #         "-- the first under `band_provenance`, the second under "
            #         "`sources`.`EF-TALFM` "
            #         "-- that do not even agree on the numerals), but under BOTH "
            #         "readings the band is external, is NOT one homogeneous "
            #         "quantity, and is NOT on our scale."
            #     ),
            #
            # WHY the Wf-A180 rendering was SUPERSEDED (Wf-A183, 2026-11-01): it
            # asserted the band's whole-band RAW-ness UNCONDITIONALLY, which
            # holds ONLY under the `band_provenance` reading.  The repo's OWN
            # record -- molmetal/reports/wfa158_frontier_postproc_matrix/
            # frontier_matrix.json -- holds TWO competing readings, and under
            # the second (`sources`.`EF-TALFM`) the frontier arm is MIXED: a
            # raw connectivity cell paired with a post-processed companion
            # cell.  So whether the band as a whole is RAW is NOT settled on
            # disk.  This is the MIRROR of the blanket "the band is
            # POST-PROCESSED" over-claim Wf-A175 already refuted.  The repair
            # is TEXT-ONLY: it introduces NO numeral, keeps every numeral that
            # was already in the string, and leaves the invariants that hold
            # under BOTH readings (external, not one homogeneous quantity, not
            # on our scale) untouched.
            # Wf-A202 (2026-11-02) X5 REPAIR -- the live rendering
            # asserted a SETTLED COUNT of competing provenance readings
            # ("two competing provenance readings", "but under BOTH
            # readings").  It is now READING-CONDITIONAL: it names the
            # competing readings by ROBUST JSON PATHS and states only
            # what holds under EVERY reading.  TEXT-ONLY: it introduces
            # NO numeral and keeps every numeral the superseded rendering
            # carried, so the artefact's narrative self-consistency
            # counters do not move.  The prior rendering is preserved
            # VERBATIM in BREAKS_2_FRONTIER_SUPERSEDED_2026_11_02_WFA202
            # (AGENTS.md §5.5 rule #6).
            "frontier": (
                "the frontier publishes the post-processing axis in ONE table "
                "-- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW "
                "and POST-PROCESSED side by side -- so 'the frontier number' "
                "is one of two numbers depending on the arm; the other "
                "frontier papers print a single arm. The published 92.7-94 % "
                "band's two endpoints are both RAW (post_processed: false); "
                "which frontier cells the band names is NOT settled on disk "
                "(COMPETING provenance readings, all in "
                "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json "
                "-- among them the readings listed at the JSON path "
                "`band_provenance`.`competing_reading_and_what_this_does_not_settle`."
                "`readings` and the reading at the JSON path `sources`.`EF-TALFM` "
                "-- that do not even agree on the numerals), and the NUMBER of "
                "competing readings is itself NOT settled on disk. What holds "
                "under EVERY reading: the band is external, is NOT one homogeneous "
                "quantity, and is NOT on our scale."
            ),
            "ours": (
                "the pool measured here is RAW (no post-processing, no "
                "repair), so it is comparable only to a RAW frontier arm, "
                "never to a post-processed one."
            ),
        },
        {
            "id": "break_4_hydrogens",
            "axis": "HYDROGENS",
            "frontier": (
                "the frontier's POST-PROCESSED arm adds explicit hydrogens; "
                "the raw arm's hydrogen convention is not stated by the "
                "record."
            ),
            "ours": hydrogen_ours,
        },
    ]


def build_comparability_statement(
    pool_name: str,
    n_rows: int,
    *,
    n_rows_with_explicit_h: int = 0,
) -> Dict[str, object]:
    """The comparability statement, with the break count and the row count
    COMPUTED from the run."""
    breaks = build_comparability_breaks(
        pool_name,
        n_rows,
        n_rows_with_explicit_h=n_rows_with_explicit_h,
    )
    out: Dict[str, object] = {
        "summary": (
            "This metric is NOT comparable to the published "
            f"{FRONTIER_BAND} frontier band, and no number in this artifact "
            "may be presented as comparable to it. "
            f"{len(breaks)} independent breaks separate the two."
        ),
        "n_breaks": len(breaks),
        "breaks": breaks,
    }
    for brk in breaks:
        out[brk["id"]] = (
            f"{brk['axis']} — {brk['frontier']} {brk['ours']}"
        )
    # Wf-A189 (2026-11-01): the dated rule-#6 records are emitted HERE, in the
    # SAME POSITION they occupy in the published artefact, so a re-run
    # REPRODUCES the published artefact instead of silently DROPPING the dated
    # history (AGENTS.md §5.5 rule #6).  They are SIBLING keys of the break
    # they supersede -- NOT a fourth field of a break -- because the repo's own
    # regression test molmetal/tests/test_wfa162_narrative_self_consistency.py
    # asserts that `comparability.breaks` resolves to exactly four breaks x
    # (break, frontier, ours).
    ordered: Dict[str, object] = {}
    for key, value in out.items():
        ordered[key] = value
        if key == "break_3_post_processing":
            ordered["break_3_post_processing_superseded_2026_11_01_wfa183"] = dict(
                BREAK_3_POST_PROCESSING_SUPERSEDED_2026_11_01_WFA183
            )
            ordered["break_3_post_processing_superseded_2026_11_01_wfa176"] = dict(
                BREAK_3_POST_PROCESSING_SUPERSEDED_2026_11_01_WFA176
            )
            ordered["break_3_post_processing_superseded_2026_11_01_wfa189"] = dict(
                BREAK_3_POST_PROCESSING_SUPERSEDED_2026_11_01_WFA189
            )
            ordered["breaks_2_frontier_superseded_2026_11_01_wfa189"] = dict(
                BREAK_3_FRONTIER_SUPERSEDED_2026_11_01_WFA189
            )
            # Wf-A202 (2026-11-02): the X5 supersessions of the third break.
            ordered["break_3_post_processing_superseded_2026_11_02_wfa202"] = dict(
                BREAK_3_POST_PROCESSING_SUPERSEDED_2026_11_02_WFA202
            )
            ordered["breaks_2_frontier_superseded_2026_11_02_wfa202"] = dict(
                BREAKS_2_FRONTIER_SUPERSEDED_2026_11_02_WFA202
            )
    return ordered


def build_what_this_is_not(
    *,
    n_rows: int,
    single_fragment_rate: Optional[float],
    end_to_end_n_runs: Optional[int],
    n_breaks: int,
) -> Dict[str, str]:
    """The "what this metric is NOT" block, with OUR numbers COMPUTED.

    ``single_fragment_rate`` is the pool's OWN re-measured single-fragment
    fraction (measured here, not quoted); ``end_to_end_n_runs`` is read out of
    the published end-to-end artifact at run time (never typed in as a
    constant).
    """
    if end_to_end_n_runs is not None:
        end_to_end_text = (
            "This is NOT the end-to-end single-molecule rate published in "
            "metrics/by_metric/cfm_fragmentation_audit.json "
            "(/summary/generation_side/single_molecule_rate_end_to_end): a "
            f"MEDIAN over {end_to_end_n_runs} RUNS (its median RATE is "
            "carried verbatim in this artifact's end_to_end_headline.median "
            "field, read from that artifact at run time). A run-level "
            "end-to-end rate is over a DIFFERENT population "
            f"({end_to_end_n_runs} runs, not ONE pool of {n_rows} rows)."
        )
    else:
        end_to_end_text = (
            "This is NOT the end-to-end single-molecule rate published in "
            "metrics/by_metric/cfm_fragmentation_audit.json "
            "(/summary/generation_side/single_molecule_rate_end_to_end): a "
            "MEDIAN over RUNS. That published artifact was not readable at "
            f"run time, so its run count is not asserted here. A run-level "
            "end-to-end rate is over a DIFFERENT population (runs, not ONE "
            f"pool of {n_rows} rows)."
        )

    if single_fragment_rate is None:
        single_fragment_text = (
            "This is NOT the SINGLE-FRAGMENT fraction of this pool. The "
            "fraction could not be measured at run time, so no number is "
            "asserted here."
        )
    else:
        single_fragment_text = (
            "This is NOT the "
            f"{round(single_fragment_rate * 100.0, 2)} % SINGLE-FRAGMENT "
            "fraction of THIS pool (re-measured here, not quoted). A "
            "fragment count is not a coordination number: the coordination "
            "number is a per-atom property of the metal's neighbourhood. A "
            "row can be a single fragment AND hold a metal with an empty or "
            "an over-full sphere, so the two quantities are not a "
            "restatement of one another."
        )

    return {
        "not_8_51_percent": end_to_end_text,
        "not_25_05_percent": single_fragment_text,
        "not_a_frontier_comparison": (
            "This is NOT comparable to the "
            f"{FRONTIER_BAND} frontier band — see comparability_statement "
            f"for the {n_breaks} independent breaks."
        ),
        "oxidation_state_is_inferred": (
            "The oxidation state is a RULE-BASED INFERENCE (three labelled "
            "layers), never a measurement. Only Layer B, and only inside its "
            "declared scope, is derived from the SMILES; Layer C is a repo "
            "PRIOR."
        ),
    }


# ---------------------------------------------------------------------------
# Small statistics helpers (stdlib only, no scipy)
# ---------------------------------------------------------------------------


def _median(values: Sequence[float]) -> Optional[float]:
    """Median of a sequence, or ``None`` for an empty sequence."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson r, or ``None`` when it is undefined."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _pearson_p(r: Optional[float], n: int) -> Optional[float]:
    """Two-sided p for a Pearson r via a normal approximation to t.

    The t-distribution is approximated by the normal because n is in the
    thousands here (t and normal quantiles agree to <0.1 % beyond n = 100).
    """
    if r is None or n < 4:
        return None
    r = max(-1.0, min(1.0, float(r)))
    if abs(r) >= 1.0:
        return 0.0
    t = abs(r) * math.sqrt((n - 2.0) / (1.0 - r * r))
    return math.erfc(t / math.sqrt(2.0))


def _two_proportion_z(
    k1: int, n1: int, k2: int, n2: int
) -> Tuple[Optional[float], Optional[float]]:
    """Two-proportion z statistic and its two-sided normal p-value."""
    if n1 <= 0 or n2 <= 0:
        return None, None
    p1 = k1 / n1
    p2 = k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n2))
    if se <= 0.0:
        return None, None
    z = (p1 - p2) / se
    return z, math.erfc(abs(z) / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Published-artifact cross-checks (MEASURED from disk, never asserted)
# ---------------------------------------------------------------------------

#: The published artifacts this tool may reproduce or differ from.  A
#: published artifact is a CANDIDATE only when its own ``source_pool`` names
#: THE POOL BEING MEASURED — otherwise the comparison would be between two
#: different populations.
PUBLISHED_POOLED_SOURCES: Tuple[
    Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...
] = (
    (
        "wfa155_ligand_connected_rate",
        ("metal_inclusive", "single_fragment_rate"),
        ("metal_inclusive", "n_single_fragment"),
    ),
    (
        "cfm_fragmentation_postproc_arm",
        ("raw_axis", "single_fragment_rate_over_parsed"),
        ("raw_axis", "n_single"),
    ),
)

#: The published end-to-end headline (a run-level rate over several runs).
END_TO_END_HEADLINE_METRIC = "cfm_fragmentation_audit"
END_TO_END_HEADLINE_PATH: Tuple[str, ...] = (
    "summary",
    "generation_side",
    "single_molecule_rate_end_to_end",
)


def _dig(doc: Any, path: Sequence[str]) -> Any:
    """Walk ``path`` through nested mappings; ``None`` when the path is absent."""
    cur: Any = doc
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def _load_metric(metric: str) -> Optional[Dict[str, Any]]:
    """Load ``metrics/by_metric/<metric>.json`` (``None`` when not on disk)."""
    path = PROJECT_ROOT / "metrics" / "by_metric" / f"{metric}.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _end_to_end_headline() -> Optional[Dict[str, Any]]:
    """The published end-to-end single-molecule rate, read back from disk.

    The run count is a number ABOUT the published artifact, so it is
    measured here rather than typed in as a constant.  ``None`` when the
    published artifact is not readable.
    """
    doc = _load_metric(END_TO_END_HEADLINE_METRIC)
    if doc is None:
        return None
    node = _dig(doc, END_TO_END_HEADLINE_PATH)
    if not isinstance(node, dict):
        return None
    median = node.get("median")
    n_runs = node.get("n")
    return {
        "source": "metrics/by_metric/cfm_fragmentation_audit.json",
        "jsonpath": "/" + "/".join(END_TO_END_HEADLINE_PATH),
        "median": median,
        "n_runs": n_runs,
        "note": (
            "the median is carried as the PUBLISHED RATE, verbatim; the "
            "percent form is deliberately not rendered anywhere, because a "
            "percent with two decimals is exactly the kind of rounded "
            "literal that this repair exists to remove"
        ),
    }


def _published_cross_check(
    pool_name: str,
    rate: Optional[float],
    n_single: Optional[int],
) -> Dict[str, Any]:
    """COMPUTE, do not assert, whether this run reproduces a published number.

    A published artifact is a candidate ONLY when its own ``source_pool``
    names THIS pool; the comparison is then made on the numbers themselves
    (rate and numerator), and the verdict says which.
    """
    considered: List[Dict[str, Any]] = []
    matched = 0
    for metric, rate_path, count_path in PUBLISHED_POOLED_SOURCES:
        doc = _load_metric(metric)
        if doc is None:
            considered.append({"metric": metric, "present_on_disk": False})
            continue
        source_pool = doc.get("source_pool")
        names_this_pool = bool(
            source_pool and Path(str(source_pool)).parent.name == pool_name
        )
        pub_rate = _dig(doc, rate_path)
        pub_count = _dig(doc, count_path)
        entry: Dict[str, Any] = {
            "metric": metric,
            "present_on_disk": True,
            "names_this_pool": names_this_pool,
            "published_rate": pub_rate,
            "published_n": pub_count,
        }
        if names_this_pool:
            entry["source_pool"] = source_pool
        else:
            # The published artifact names a DIFFERENT pool.  Its path is NOT
            # reproduced here: repeating another pool's identity inside a
            # MEASURED artifact measured on THIS pool is the same defect
            # class this repair is closing (a pool-specific string literal in
            # an artifact that does not describe it).  The verdict and the
            # metric name carry the provenance; the path stays on disk.
            entry["source_pool"] = (
                "A DIFFERENT POOL (not this pool); see "
                f"metrics/by_metric/{metric}.json for that artifact's own "
                "declared source_pool"
            )
        if (
            names_this_pool
            and isinstance(pub_rate, (int, float))
            and not isinstance(pub_rate, bool)
            and rate is not None
        ):
            delta = abs(float(pub_rate) - float(rate))
            entry["abs_delta"] = delta
            entry["rate_identical"] = delta <= 1e-12
            entry["numerator_identical"] = (
                isinstance(pub_count, (int, float))
                and not isinstance(pub_count, bool)
                and int(pub_count) == int(n_single if n_single is not None else -1)
            )
            entry["verdict"] = (
                "THIS RUN REPRODUCES THE PUBLISHED NUMBER FOR THIS POOL "
                "(bit-for-bit)"
                if entry["rate_identical"]
                else "THIS RUN DIFFERS FROM THE PUBLISHED NUMBER FOR THIS POOL"
            )
            matched += 1
        else:
            entry["verdict"] = "NOT A CANDIDATE FOR THIS POOL"
        considered.append(entry)

    return {
        "rule": (
            "a published artifact is a candidate ONLY when its own "
            "source_pool names THIS pool; the comparison is computed from "
            "the two numbers, never asserted"
        ),
        "this_pool": pool_name,
        "this_run_rate": rate,
        "this_run_n_single_fragment": n_single,
        "verdict": (
            "A PUBLISHED ARTIFACT NAMES THIS POOL"
            if matched
            else "NO PUBLISHED ARTIFACT NAMES THIS POOL"
        ),
        "n_matched": matched,
        "candidates": considered,
    }


# ---------------------------------------------------------------------------
# Per-metal analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetalSphere:
    """The coordination sphere of ONE metal atom in ONE pool row.

    Every numeric field is MEASURED from the parsed SMILES except the
    oxidation-state fields, which are RULE-BASED INFERENCE (see
    :data:`OXIDATION_STATE_LAYERS`) and the prior fallback.
    """

    smiles: str
    metal_symbol: str
    metal_z: int
    metal_formal_charge: int

    # --- coordination sphere -------------------------------------------
    coordination_number: int
    total_degree: int
    n_dative_bonds: int
    neighbour_symbols: Tuple[str, ...]
    neighbour_motif: str
    donor_class_counts: Dict[str, int]
    n_classical_donors: int

    # --- oxidation state (rule-based inference / prior) -----------------
    os_formal_charge: int
    os_charge_balanced: Optional[int]
    os_charge_balanced_status: str
    os_charge_balanced_usable: bool
    os_prior_default: Optional[int]
    os_used: Optional[int]
    os_source: str

    # --- plausibility ----------------------------------------------------
    expected_cn: Optional[int]
    coordination_class: str
    expected_cn_os_derived: Optional[int]
    coordination_class_os_derived: str
    geometry_class: Optional[str]

    # --- row context -----------------------------------------------------
    row_n_frags: int
    row_total_charge: int
    metal_fragment_atoms: int
    metal_fragment_is_largest: bool
    n_metal_free_fragments: int
    disconnected_donor_atoms: int
    disconnected_halide_fragments: int

    @property
    def metal_alone_in_fragment(self) -> bool:
        """True when the metal has no bonded neighbour at all."""
        return self.coordination_number == 0


def donor_class(symbol: str) -> str:
    """Donor class of an element symbol (``C_donor`` / ``N_donor`` / ...)."""
    return DONOR_CLASS_BY_SYMBOL.get(symbol, "other")


def _coordination_class(coordination_number: int, expected_cn: Optional[int]) -> str:
    """Classify a coordination number against the expected capacity."""
    if coordination_number == 0:
        return "empty"
    if expected_cn is None:
        return "unknown_expected"
    if coordination_number < expected_cn:
        return "under"
    if coordination_number == expected_cn:
        return "at_capacity"
    return "over"


def analyze_smiles(
    smiles: str, *, row_n_frags: Optional[int] = None
) -> List[MetalSphere]:
    """Coordination sphere of every metal atom in ``smiles``.

    Parameters
    ----------
    smiles
        The pool SMILES (may be multi-component, i.e. contain ``.``).
    row_n_frags
        Optional pre-measured fragment count (the pool's ``n_frags_raw``
        column).  When supplied it is only used for a consistency check; the
        fragment count used in the metric is always re-measured here.

    Returns
    -------
    list of :class:`MetalSphere`
        One entry per metal atom found, in atom order.  Empty when the SMILES
        does not parse or carries no metal.
    """
    if not smiles:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    frag_atom_ids = Chem.GetMolFrags(mol)
    n_frags = len(frag_atom_ids)
    frag_size = [len(ids) for ids in frag_atom_ids]
    largest_frag_size = max(frag_size) if frag_size else 0
    atom_to_frag: Dict[int, int] = {}
    for fi, ids in enumerate(frag_atom_ids):
        for aid in ids:
            atom_to_frag[aid] = fi

    row_total_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())

    metal_indices = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _SYMBOL_BY_Z
    ]
    if not metal_indices:
        return []

    metal_free_fragments = [
        fi
        for fi in range(n_frags)
        if not any(
            mol.GetAtomWithIdx(int(aid)).GetAtomicNum() in _SYMBOL_BY_Z
            for aid in frag_atom_ids[fi]
        )
    ]
    disconnected_halide_fragments = 0
    for fi in metal_free_fragments:
        ids = frag_atom_ids[fi]
        if len(ids) != 1:
            continue
        if mol.GetAtomWithIdx(int(ids[0])).GetSymbol() in HALIDE_SYMBOLS:
            disconnected_halide_fragments += 1

    spheres: List[MetalSphere] = []
    for midx in metal_indices:
        metal = mol.GetAtomWithIdx(midx)
        symbol = _SYMBOL_BY_Z[metal.GetAtomicNum()]
        neighbours = list(metal.GetNeighbors())
        nbr_symbols = tuple(sorted(n.GetSymbol() for n in neighbours))
        motif = ",".join(nbr_symbols)

        class_counts: Dict[str, int] = {}
        for sym in nbr_symbols:
            cls = donor_class(sym)
            class_counts[cls] = class_counts.get(cls, 0) + 1
        n_classical = sum(class_counts.get(cls, 0) for cls in NON_CARBON_DONOR_CLASSES)

        # --- oxidation state, three labelled layers -------------------
        os_a = int(metal.GetFormalCharge())
        own_frag = atom_to_frag[midx]
        own_frag_ids = frag_atom_ids[own_frag]
        charged_in_own_frag = [
            int(aid)
            for aid in own_frag_ids
            if int(aid) != midx and mol.GetAtomWithIdx(int(aid)).GetFormalCharge() != 0
        ]
        n_halide_donors = 0
        n_anionic_donors = 0
        scope_ok = True
        for n in neighbours:
            sym = n.GetSymbol()
            q = int(n.GetFormalCharge())
            if sym in HALIDE_SYMBOLS:
                n_halide_donors += 1
            elif q == 0:
                continue  # neutral L-type donor, OS 0
            elif (
                q == -1
                and len(charged_in_own_frag) == 1
                and (charged_in_own_frag[0] == n.GetIdx())
            ):
                n_anionic_donors += 1
            else:
                scope_ok = False
                break
        # A charged atom in the metal's fragment that is NOT one of the
        # counted monatomic anionic donors means some ligand carries a charge
        # that is not recoverable from formal charges alone (a polyatomic
        # anionic ligand such as a carboxylate written [O-]) -> the row is
        # OUT OF SCOPE for Layer B rather than silently producing a number.
        if scope_ok and charged_in_own_frag:
            counted = {
                n.GetIdx()
                for n in neighbours
                if int(n.GetFormalCharge()) == -1
                and n.GetSymbol() not in HALIDE_SYMBOLS
            }
            if not set(charged_in_own_frag).issubset(counted):
                scope_ok = False
        if scope_ok:
            os_b_value = int(row_total_charge) + n_halide_donors + n_anionic_donors
            os_b: Optional[int] = os_b_value
            os_b_status = "determined_in_scope"
        else:
            os_b = None
            os_b_status = "out_of_scope_charged_polyatomic_donor"
        os_c = DEFAULT_OXIDATION_STATE.get(symbol)

        if os_b is not None and os_b > 0:
            os_used: Optional[int] = os_b
            os_source = "os_charge_balanced (rule-based inference, in scope)"
        elif os_a != 0:
            os_used = os_a
            os_source = "os_formal_charge (rule-based inference)"
        else:
            os_used = os_c
            os_source = (
                "os_prior_default (ASSUMED prior; the SMILES carries no "
                "oxidation-state information)"
            )

        expected_cn = (
            int(coordination_capacity(symbol, os_c)) if os_c is not None else None
        )
        expected_cn_os = (
            int(coordination_capacity(symbol, os_used)) if os_used is not None else None
        )
        cn = int(metal.GetDegree())
        geometry = (
            classify_coord_geometry(cn, metal.GetAtomicNum(), os_c) if cn > 0 else None
        )

        metal_fragment_atoms = frag_size[own_frag]
        disconnected_donor_atoms = 0
        for fi, ids in enumerate(frag_atom_ids):
            if fi == own_frag:
                continue
            for aid in ids:
                if mol.GetAtomWithIdx(int(aid)).GetSymbol() in DONOR_SYMBOLS:
                    disconnected_donor_atoms += 1

        spheres.append(
            MetalSphere(
                smiles=smiles,
                metal_symbol=symbol,
                metal_z=int(metal.GetAtomicNum()),
                metal_formal_charge=os_a,
                coordination_number=cn,
                total_degree=int(metal.GetTotalDegree()),
                n_dative_bonds=sum(
                    1
                    for b in metal.GetBonds()
                    if b.GetBondType() == Chem.BondType.DATIVE
                ),
                neighbour_symbols=nbr_symbols,
                neighbour_motif=motif,
                donor_class_counts=class_counts,
                n_classical_donors=n_classical,
                os_formal_charge=os_a,
                os_charge_balanced=os_b,
                os_charge_balanced_status=os_b_status,
                os_charge_balanced_usable=bool(os_b is not None and os_b > 0),
                os_prior_default=os_c,
                os_used=os_used,
                os_source=os_source,
                expected_cn=expected_cn,
                coordination_class=_coordination_class(cn, expected_cn),
                expected_cn_os_derived=expected_cn_os,
                coordination_class_os_derived=_coordination_class(cn, expected_cn_os),
                geometry_class=geometry,
                row_n_frags=n_frags,
                row_total_charge=row_total_charge,
                metal_fragment_atoms=metal_fragment_atoms,
                metal_fragment_is_largest=(metal_fragment_atoms == largest_frag_size),
                n_metal_free_fragments=len(metal_free_fragments),
                disconnected_donor_atoms=disconnected_donor_atoms,
                disconnected_halide_fragments=disconnected_halide_fragments,
            )
        )
    return spheres


# ---------------------------------------------------------------------------
# Pool-level aggregation
# ---------------------------------------------------------------------------


def _distribution(values: Sequence[int]) -> Dict[str, int]:
    """Integer histogram with string keys, ascending."""
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def _coordination_class_counts(spheres: Sequence[MetalSphere]) -> Dict[str, int]:
    counts = Counter(s.coordination_class for s in spheres)
    out = {
        k: int(counts.get(k, 0))
        for k in ("empty", "under", "at_capacity", "over", "unknown_expected")
    }
    return out


def _coordination_class_counts_os(spheres: Sequence[MetalSphere]) -> Dict[str, int]:
    """Counts for the SECONDARY (structure-derived-OS) classification."""
    counts = Counter(s.coordination_class_os_derived for s in spheres)
    out = {
        k: int(counts.get(k, 0))
        for k in ("empty", "under", "at_capacity", "over", "unknown_expected")
    }
    return out


def summarize(
    spheres: Sequence[MetalSphere],
    *,
    n_rows: int,
    n_rows_parsed: int,
    n_rows_with_metal: int,
    n_rows_multi_metal: int,
    n_rows_unparseable: int,
    n_frags_column_mismatches: int,
    n_rows_empty_string: int = 0,
    n_rows_with_explicit_h: int = 0,
    n_single_fragment_rows: int = 0,
) -> Dict[str, object]:
    """Pool-level summary of the coordination-sphere profile."""
    cn_all = [s.coordination_number for s in spheres]
    cn_nonempty = [c for c in cn_all if c > 0]
    alone = [s for s in spheres if s.metal_alone_in_fragment]

    donor_totals: Counter = Counter()
    for s in spheres:
        for cls, n in s.donor_class_counts.items():
            donor_totals[cls] += n

    motif_counter = Counter(s.neighbour_motif for s in spheres)
    motif_distribution = [
        {"motif": m, "n": int(n), "fraction": float(n) / len(spheres)}
        for m, n in motif_counter.most_common()
    ]

    per_metal: Dict[str, Dict[str, object]] = {}
    for metal in sorted({s.metal_symbol for s in spheres}):
        sub = [s for s in spheres if s.metal_symbol == metal]
        sub_cn = [s.coordination_number for s in sub]
        per_metal[metal] = {
            "n_metal_atoms": len(sub),
            "coordination_number_distribution": _distribution(sub_cn),
            "coordination_number_mean": _mean(sub_cn),
            "coordination_number_median": _median(sub_cn),
            "coordination_class_counts": _coordination_class_counts(sub),
            "expected_cn_prior": DEFAULT_OXIDATION_STATE.get(metal),
            "coordination_capacity_for_prior": (
                coordination_capacity(metal, DEFAULT_OXIDATION_STATE[metal])
                if metal in DEFAULT_OXIDATION_STATE
                else None
            ),
            "os_charge_balanced_status_counts": dict(
                Counter(s.os_charge_balanced_status for s in sub)
            ),
            "top_neighbour_motifs": [
                {"motif": m, "n": int(n)}
                for m, n in Counter(s.neighbour_motif for s in sub).most_common(8)
            ],
        }

    os_status = Counter(
        (s.os_source.split(" ")[0], s.os_charge_balanced_status) for s in spheres
    )
    os_used_counter = Counter(s.os_used for s in spheres)

    geometry_counter = Counter(s.geometry_class for s in spheres)

    n_analyzed = n_rows_parsed - n_rows_empty_string
    n_implicit_h = sum(1 for s in spheres if s.total_degree > s.coordination_number)
    n_dative = int(sum(s.n_dative_bonds for s in spheres))

    if n_implicit_h == 0:
        cn_definition = (
            "RDKit GetDegree() of the metal atom: the number of atoms "
            "covalently bonded to it (heavy-atom neighbours; the pool's "
            "metals carry no implicit H, so GetDegree() == "
            "GetTotalDegree() in every row)."
        )
    else:
        cn_definition = (
            "RDKit GetDegree() of the metal atom: the number of atoms "
            "covalently bonded to it (heavy-atom neighbours; "
            f"{n_implicit_h} metal atoms in this pool carry an implicit H, "
            "so GetDegree() != GetTotalDegree() for those atoms)."
        )

    return {
        "population": {
            "n_rows": n_rows,
            "n_rows_parsed": n_rows_parsed,
            "n_rows_empty_string": n_rows_empty_string,
            "n_rows_unparseable": n_rows_unparseable,
            "n_rows_analyzed": n_analyzed,
            "n_rows_with_at_least_one_metal": n_rows_with_metal,
            "n_rows_multi_metal": n_rows_multi_metal,
            "n_metal_atoms": len(spheres),
            "n_single_fragment_rows": n_single_fragment_rows,
            "single_fragment_rate_over_analyzed": (
                (n_single_fragment_rows / n_analyzed) if n_analyzed else None
            ),
            "n_rows_with_explicit_hydrogen_atoms": n_rows_with_explicit_h,
            "n_metal_atoms_with_an_implicit_h": n_implicit_h,
            "n_dative_metal_incident_bonds": n_dative,
            "n_frags_column_mismatches": n_frags_column_mismatches,
            "n_frags_column_mismatch_note": (
                "rows where the pool's n_frags_raw column differs from the "
                "RDKit fragment count re-measured here"
            ),
        },
        "coordination_number": {
            "definition": cn_definition,
            "distribution_all_rows": _distribution(cn_all),
            "mean_all_rows": _mean(cn_all),
            "median_all_rows": _median(cn_all),
            "mean_excluding_empty_spheres": _mean(cn_nonempty),
            "median_excluding_empty_spheres": _median(cn_nonempty),
            "n_rows_with_empty_sphere": len(alone),
            "empty_sphere_note": (
                "the empty-sphere rows are the metal-ALONE rows; the "
                "excluding-empty statistics are reported next to the "
                "all-rows ones so neither is silently dropped"
            ),
        },
        "coordination_class": {
            "definition": (
                "empty = CN 0; under/at_capacity/over compare the measured CN "
                "against expected_cn = coordination_capacity(metal, "
                "DEFAULT_OXIDATION_STATE[metal]) — the metal's USUAL "
                "oxidation state, i.e. the repo PRIOR. The metal is "
                "over-coordinated when CN > expected. This is the PRIMARY "
                "classification; the structure-derived-OS variant is under "
                "coordination_class_secondary_os_derived."
            ),
            "counts": _coordination_class_counts(spheres),
            "fractions": {
                k: (v / len(spheres) if spheres else None)
                for k, v in _coordination_class_counts(spheres).items()
            },
        },
        "coordination_class_secondary_os_derived": {
            "definition": (
                "the SAME classification but against the capacity implied by "
                "the STRUCTURE-DERIVED oxidation state (Layer B) where it is "
                "in scope, and by the prior otherwise. This is a SECONDARY "
                "check and is reported separately because Layer B can "
                "UNDER-estimate the metal's true oxidation state: a complex "
                "with exactly ONE X-type (halide) donor and five neutral "
                "donors yields OS = +1, whose capacity is smaller than the "
                "metal's real one, so an octahedral Ir(III)/Rh(III) centre "
                "is spuriously flagged over-coordinated. The PRIMARY check "
                "uses the metal's USUAL oxidation state (the repo prior)."
            ),
            "counts": _coordination_class_counts_os(spheres),
            "fractions": {
                k: (v / len(spheres) if spheres else None)
                for k, v in _coordination_class_counts_os(spheres).items()
            },
        },
        "donor_classes": {
            "definition": (
                "the element class of each atom covalently bonded to the "
                "metal: N_donor / O_donor / C_donor / halide / S_donor / "
                "P_donor / Se_donor / other"
            ),
            "totals_across_all_metal_atoms": dict(sorted(donor_totals.items())),
            "n_classical_donors_mean": _mean([s.n_classical_donors for s in spheres]),
            "n_rows_with_zero_classical_donor": sum(
                1 for s in spheres if s.n_classical_donors == 0
            ),
            "n_classical_donor_only_carbon_rows": sum(
                1
                for s in spheres
                if s.n_classical_donors == 0 and s.coordination_number > 0
            ),
        },
        "donor_motif_distribution": {
            "definition": (
                "the sorted multiset of the metal's neighbour elements, e.g. "
                "'N,N,N,Cl'; this is the coordination motif"
            ),
            "n_distinct_motifs": len(motif_counter),
            "top_motifs": motif_distribution[:40],
            "full_motif_counts": {m: int(n) for m, n in motif_counter.items()},
        },
        "oxidation_state": {
            "status": "RULE-BASED INFERENCE — not a measurement",
            "layers": dict(OXIDATION_STATE_LAYERS),
            "layer_status_counts": {
                f"{a}|{b}": int(n) for (a, b), n in sorted(os_status.items())
            },
            "os_used_distribution": {
                str(k): int(v)
                for k, v in sorted(
                    os_used_counter.items(), key=lambda kv: (kv[0] is None, kv[0])
                )
            },
            "os_charge_balanced_value_distribution": {
                str(k): int(v)
                for k, v in sorted(
                    Counter(
                        s.os_charge_balanced
                        for s in spheres
                        if s.os_charge_balanced is not None
                    ).items()
                )
            },
            "n_rows_where_the_smiles_carries_no_charge_information": sum(
                1
                for s in spheres
                if s.metal_formal_charge == 0 and s.row_total_charge == 0
            ),
            "n_metal_atoms_with_a_nonzero_formal_charge": sum(
                1 for s in spheres if s.metal_formal_charge != 0
            ),
            "n_metal_atoms_with_no_charge_information": sum(
                1
                for s in spheres
                if s.metal_formal_charge == 0 and s.row_total_charge == 0
            ),
        },
        "geometry_class": {
            "definition": (
                "molmetal.data.coord_geometry.classify_coord_geometry(CN, Z, "
                "DEFAULT_OXIDATION_STATE[metal]) — the geometry class implied "
                "by the measured CN under the metal's USUAL oxidation state "
                "(the repo prior)"
            ),
            "counts": {
                str(k): int(v)
                for k, v in sorted(
                    geometry_counter.items(), key=lambda kv: (kv[0] is None, kv[0])
                )
            },
        },
        "per_metal": per_metal,
        "metal_alone_in_fragment": {
            "definition": (
                "rows whose metal has NO bonded neighbour, i.e. the metal is "
                "alone in its own RDKit fragment. The coordination sphere is "
                "EMPTY BY CONSTRUCTION for these rows — they are reported "
                "separately and never averaged away."
            ),
            "n_rows": len(alone),
            "fraction": (len(alone) / len(spheres)) if spheres else None,
            "metal_breakdown": dict(Counter(s.metal_symbol for s in alone)),
            "n_frags_breakdown": dict(Counter(str(s.row_n_frags) for s in alone)),
            "disconnected_donor_atoms_mean": _mean(
                [s.disconnected_donor_atoms for s in alone]
            ),
            "disconnected_donor_atoms_total": int(
                sum(s.disconnected_donor_atoms for s in alone)
            ),
        },
    }


def cross_tab(
    spheres: Sequence[MetalSphere],
    *,
    largest_frag_frac: Sequence[float],
) -> Dict[str, object]:
    """Cross-tab the coordination sphere against the LIGAND-ONLY connectivity.

    This is the option-A question: does a fragmented ligand correlate with an
    under- or an over-coordinated metal?
    """
    buckets: Dict[str, List[MetalSphere]] = {}
    for s in spheres:
        key = str(s.row_n_frags) if s.row_n_frags <= 4 else "5+"
        buckets.setdefault(key, []).append(s)

    by_n_frags: Dict[str, Dict[str, object]] = {}
    for key in sorted(buckets, key=lambda k: (k == "5+", k)):
        sub = buckets[key]
        cns = [s.coordination_number for s in sub]
        counts = _coordination_class_counts(sub)
        by_n_frags[key] = {
            "n_rows": len(sub),
            "coordination_number_mean": _mean(cns),
            "coordination_number_median": _median(cns),
            "coordination_class_counts": counts,
            "fraction_empty": counts["empty"] / len(sub) if sub else None,
            "fraction_under_or_empty": (
                (counts["empty"] + counts["under"]) / len(sub) if sub else None
            ),
            "fraction_at_capacity": (counts["at_capacity"] / len(sub) if sub else None),
            "fraction_over": counts["over"] / len(sub) if sub else None,
            "disconnected_donor_atoms_mean": _mean(
                [s.disconnected_donor_atoms for s in sub]
            ),
        }

    single = [s for s in spheres if s.row_n_frags == 1]
    frag = [s for s in spheres if s.row_n_frags > 1]

    def _under_or_empty(sub: Sequence[MetalSphere]) -> int:
        return sum(1 for s in sub if s.coordination_class in ("under", "empty"))

    z, p = _two_proportion_z(
        _under_or_empty(frag), len(frag), _under_or_empty(single), len(single)
    )

    cn_values = [s.coordination_number for s in spheres]
    nfrag_values = [float(s.row_n_frags) for s in spheres]
    r_frags = _pearson(nfrag_values, cn_values)
    r_frac = _pearson(list(largest_frag_frac), cn_values)

    return {
        "question": (
            "does a fragmented ligand correlate with an under- or an "
            "over-coordinated metal?"
        ),
        "by_row_n_frags": by_n_frags,
        "fragmented_vs_single_fragment": {
            "definition": (
                "fragmented = row_n_frags > 1; single = row_n_frags == 1. "
                "'under_or_empty' = measured CN < expected_cn (the empty "
                "sphere counted as under-coordinated, since an empty sphere "
                "is the limiting case of a lost donor)."
            ),
            "single_fragment": {
                "n_rows": len(single),
                "n_under_or_empty": _under_or_empty(single),
                "rate": (_under_or_empty(single) / len(single) if single else None),
            },
            "fragmented": {
                "n_rows": len(frag),
                "n_under_or_empty": _under_or_empty(frag),
                "rate": _under_or_empty(frag) / len(frag) if frag else None,
            },
            "two_proportion_z": z,
            "two_proportion_p_two_sided": p,
        },
        "pearson_n_frags_vs_cn": r_frags,
        "pearson_n_frags_vs_cn_p_two_sided": _pearson_p(r_frags, len(spheres)),
        "pearson_largest_frag_frac_vs_cn": r_frac,
        "pearson_largest_frag_frac_vs_cn_p_two_sided": _pearson_p(r_frac, len(spheres)),
        "pearson_note": (
            "p-values use a normal approximation to the t distribution; this "
            f"run has n = {len(spheres)} metal atoms, so the approximation "
            "is "
            + (
                "accurate (n is in the thousands)."
                if len(spheres) >= 1000
                else "ROUGH (n is NOT in the thousands)."
            )
        ),
    }


def candidate_mechanism(spheres: Sequence[MetalSphere]) -> Dict[str, object]:
    """Test the 'the donors were split off' mechanism for fragmentation.

    Hypothesis: a fragmented ligand leaves the metal UNDER-coordinated
    because its donor atoms were split into other fragments, leaving the
    metal's own fragment with too few neighbours.  The direct evidence is the
    count of donor-type atoms that are DISCONNECTED from the metal (they sit
    in another fragment of the SAME row).
    """
    by_class: Dict[str, Dict[str, object]] = {}
    for cls in ("empty", "under", "at_capacity", "over", "unknown_expected"):
        sub = [s for s in spheres if s.coordination_class == cls]
        if not sub:
            by_class[cls] = {"n_rows": 0}
            continue
        by_class[cls] = {
            "n_rows": len(sub),
            "disconnected_donor_atoms_mean": _mean(
                [s.disconnected_donor_atoms for s in sub]
            ),
            "disconnected_donor_atoms_total": int(
                sum(s.disconnected_donor_atoms for s in sub)
            ),
            "rows_with_at_least_one_disconnected_donor": sum(
                1 for s in sub if s.disconnected_donor_atoms > 0
            ),
            "coordination_number_mean": _mean([s.coordination_number for s in sub]),
        }

    under_empty = [
        s
        for s in spheres
        if s.coordination_class in ("under", "empty") and s.row_n_frags > 1
    ]
    at_cap_frag = [
        s
        for s in spheres
        if s.coordination_class == "at_capacity" and s.row_n_frags > 1
    ]
    n_under_with_donor = sum(1 for s in under_empty if s.disconnected_donor_atoms > 0)
    n_atcap_with_donor = sum(1 for s in at_cap_frag if s.disconnected_donor_atoms > 0)

    # A donor split off can only EXPLAIN the missing coordination if the
    # missing count is at most the number of donors that are actually
    # sitting in another fragment.
    explained = 0
    for s in under_empty:
        expected = s.expected_cn
        if expected is None:
            continue
        missing = expected - s.coordination_number
        if missing > 0 and s.disconnected_donor_atoms >= missing:
            explained += 1

    return {
        "hypothesis": (
            "A fragmented ligand leaves the metal under-coordinated because "
            "its donor atoms were split into separate fragments, leaving the "
            "metal's own fragment with too few neighbours."
        ),
        "test": (
            "count donor-type atoms (N/O/S/P/Se/halide) that are DISCONNECTED "
            "from the metal, i.e. sitting in another fragment of the same row"
        ),
        "by_coordination_class": by_class,
        "fragmented_under_or_empty": {
            "n_rows": len(under_empty),
            "n_with_at_least_one_disconnected_donor": n_under_with_donor,
            "disconnected_donor_atoms_mean": _mean(
                [s.disconnected_donor_atoms for s in under_empty]
            ),
        },
        "fragmented_at_capacity": {
            "n_rows": len(at_cap_frag),
            "n_with_at_least_one_disconnected_donor": n_atcap_with_donor,
            "disconnected_donor_atoms_mean": _mean(
                [s.disconnected_donor_atoms for s in at_cap_frag]
            ),
        },
        "donor_split_fully_explains_the_missing_coordination": {
            "n_rows": explained,
            "denominator": len(under_empty),
            "fraction": (explained / len(under_empty)) if under_empty else None,
            "definition": (
                "rows whose (expected_cn - measured CN) missing neighbours "
                "are covered by at least that many donor atoms found "
                "disconnected from the metal in the same row"
            ),
        },
        "counter_evidence": {
            "n_rows_with_disconnected_donor_at_capacity": n_atcap_with_donor,
            "n_rows_with_disconnected_donor_at_capacity_denominator": len(at_cap_frag),
            "note": (
                "donors split off also occur in rows whose metal is already "
                "at capacity, so a split donor is not by itself evidence of "
                "the fragmentation cause"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Pool driver
# ---------------------------------------------------------------------------


def analyze_pool(pool_path: Path) -> Dict[str, object]:
    """Run the coordination-sphere profile over a generation pool CSV."""
    with pool_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    spheres: List[MetalSphere] = []
    largest_frag_frac: List[float] = []
    n_parsed = 0
    n_with_metal = 0
    n_multi_metal = 0
    n_unparseable = 0
    n_frags_mismatch = 0
    n_empty_string = 0
    n_rows_with_explicit_h = 0
    n_single_fragment_rows = 0
    n_rows = len(rows)

    for row in rows:
        smiles = (row.get("smiles") or "").strip()
        raw_frags = row.get("n_frags_raw")
        try:
            declared_frags = int(raw_frags) if raw_frags not in (None, "") else None
        except (TypeError, ValueError):
            declared_frags = None
        try:
            frac = float(row.get("largest_frag_frac") or 0.0)
        except (TypeError, ValueError):
            frac = 0.0
        largest_frag_frac.append(frac)

        if not smiles:
            n_empty_string += 1
            continue

        parsed = Chem.MolFromSmiles(smiles)
        if parsed is None:
            n_unparseable += 1
            continue
        n_parsed += 1

        # --- pool-level facts that the narrative used to ASSERT ---------
        # explicit hydrogens: H atoms that are NODES in the graph (an
        # implicit H is not an atom, so this counts them exactly).
        if any(a.GetAtomicNum() == 1 for a in parsed.GetAtoms()):
            n_rows_with_explicit_h += 1
        # the pool's OWN single-fragment fraction, re-measured here (this is
        # the number the published ligand-connected metric reports for the
        # same pool, so the two can be compared).
        frags = Chem.GetMolFrags(parsed)
        n_heavy = sum(1 for a in parsed.GetAtoms() if a.GetAtomicNum() > 1)
        if len(frags) == 1 and n_heavy > 0:
            n_single_fragment_rows += 1

        found = analyze_smiles(smiles)
        if (
            declared_frags is not None
            and len(Chem.GetMolFrags(parsed)) != declared_frags
        ):
            n_frags_mismatch += 1
        if not found:
            continue
        n_with_metal += 1
        if len(found) > 1:
            n_multi_metal += 1
        spheres.extend(found)

    summary = summarize(
        spheres,
        n_rows=n_rows,
        n_rows_parsed=n_parsed,
        n_rows_with_metal=n_with_metal,
        n_rows_multi_metal=n_multi_metal,
        n_rows_unparseable=n_unparseable,
        n_frags_column_mismatches=n_frags_mismatch,
        n_rows_empty_string=n_empty_string,
        n_rows_with_explicit_h=n_rows_with_explicit_h,
        n_single_fragment_rows=n_single_fragment_rows,
    )
    return {
        "summary": summary,
        "cross_tab": cross_tab(spheres, largest_frag_frac=largest_frag_frac),
        "candidate_mechanism": candidate_mechanism(spheres),
        "spheres": spheres,
    }


#: Column order of the columnar per-row dump in the emitted artifact.
PER_ROW_COLUMNS: Tuple[str, ...] = (
    "row_index",
    "smiles",
    "metal_symbol",
    "metal_z",
    "metal_formal_charge",
    "coordination_number",
    "total_degree",
    "neighbour_motif",
    "donor_class_counts",
    "n_classical_donors",
    "os_formal_charge",
    "os_charge_balanced",
    "os_charge_balanced_status",
    "os_charge_balanced_usable",
    "os_prior_default",
    "os_used",
    "expected_cn",
    "coordination_class",
    "expected_cn_os_derived",
    "coordination_class_os_derived",
    "geometry_class",
    "row_n_frags",
    "row_total_charge",
    "metal_fragment_atoms",
    "metal_fragment_is_largest",
    "n_metal_free_fragments",
    "disconnected_donor_atoms",
    "disconnected_halide_fragments",
    "metal_alone_in_fragment",
)


def _per_row_records(spheres: Sequence[MetalSphere]) -> List[List[object]]:
    """Columnar per-row records (see :data:`PER_ROW_COLUMNS`)."""
    out: List[List[object]] = []
    for i, s in enumerate(spheres):
        out.append(
            [
                i,
                s.smiles,
                s.metal_symbol,
                s.metal_z,
                s.metal_formal_charge,
                s.coordination_number,
                s.total_degree,
                s.neighbour_motif,
                dict(s.donor_class_counts),
                s.n_classical_donors,
                s.os_formal_charge,
                s.os_charge_balanced,
                s.os_charge_balanced_status,
                s.os_charge_balanced_usable,
                s.os_prior_default,
                s.os_used,
                s.expected_cn,
                s.coordination_class,
                s.expected_cn_os_derived,
                s.coordination_class_os_derived,
                s.geometry_class,
                s.row_n_frags,
                s.row_total_charge,
                s.metal_fragment_atoms,
                s.metal_fragment_is_largest,
                s.n_metal_free_fragments,
                s.disconnected_donor_atoms,
                s.disconnected_halide_fragments,
                s.metal_alone_in_fragment,
            ]
        )
    return out


def build_findings(result: Dict[str, object]) -> Dict[str, object]:
    """The honest verdicts, COMPUTED from the measured values.

    Nothing here is hard-coded: every number is read back out of the
    summary / cross-tab / mechanism blocks, so the verdicts cannot drift
    away from the measurements they describe.

    Wf-A164: the SENTENCES are computed too, not just the numbers.  A
    sentence that is true for the default pool and false for another one is
    the defect class this branch repairs, so every conditional statement
    here is selected by the measurement and the machine ``verdict`` field
    says which branch was taken.  The six finding KEYS are legacy schema
    labels from Wf-A155 and are kept for compatibility.
    """
    summary = result["summary"]
    xt = result["cross_tab"]
    mech = result["candidate_mechanism"]
    cls = summary["coordination_class"]["counts"]
    alone = summary["metal_alone_in_fragment"]
    frag = xt["fragmented_vs_single_fragment"]
    explained = mech["donor_split_fully_explains_the_missing_coordination"]
    n_metal_atoms = summary["population"]["n_metal_atoms"]
    n_rows = summary["population"]["n_rows"]

    # --- finding 1: is any metal over-coordinated? -----------------------
    over = cls["over"]
    if over == 0:
        f1_statement = (
            "NO metal in the pool is over-coordinated relative to its "
            "usual oxidation state's coordination capacity. Every "
            "departure is UNDER-coordination or an EMPTY sphere."
        )
        f1_verdict = "no_metal_is_over_coordinated"
    else:
        f1_statement = (
            f"{over} of the {n_metal_atoms} metal atoms in the pool ARE "
            "over-coordinated relative to the usual oxidation state's "
            "coordination capacity; every other departure is "
            "UNDER-coordination or an EMPTY sphere."
        )
        f1_verdict = "some_metals_are_over_coordinated"

    # --- finding 2: does the SMILES carry any oxidation-state info? -----
    n_nonzero_charge = summary["oxidation_state"][
        "n_metal_atoms_with_a_nonzero_formal_charge"
    ]
    n_no_charge_info = summary["oxidation_state"][
        "n_metal_atoms_with_no_charge_information"
    ]
    if n_nonzero_charge == 0 and n_no_charge_info == n_metal_atoms:
        f2_statement = (
            "Every metal atom and every donor atom in the pool is "
            "written with formal charge 0, so the emitted SMILES "
            "carries NO oxidation-state information at all. The "
            "oxidation state used for the capacity check is therefore a "
            "PRIOR for most rows, and even the structure-derived Layer "
            "B can only fire on the rows that carry a halide donor."
        )
        f2_verdict = "no_charge_information_in_the_pool"
    else:
        f2_statement = (
            f"{n_nonzero_charge} of the {n_metal_atoms} metal atoms carry a "
            "NONZERO formal charge, so the emitted SMILES DOES carry some "
            "charge information; the oxidation state used for the capacity "
            f"check is still a PRIOR for the other rows ({n_no_charge_info} "
            "metal atoms carry no charge information at all)."
        )
        f2_verdict = "some_charge_information_in_the_pool"

    # --- finding 3: does fragmentation predict under-coordination? -------
    p_two_prop = frag["two_proportion_p_two_sided"]
    r_frags = xt["pearson_n_frags_vs_cn"]
    r_frac = xt["pearson_largest_frag_frac_vs_cn"]
    r_values = [
        abs(float(r)) for r in (r_frags, r_frac) if isinstance(r, (int, float))
    ]
    max_abs_r = max(r_values) if r_values else None
    p_rounded = (
        round(float(p_two_prop), 4)
        if isinstance(p_two_prop, (int, float)) and not isinstance(p_two_prop, bool)
        else None
    )
    max_abs_r_rounded = round(max_abs_r, 4) if max_abs_r is not None else None
    if p_rounded is not None and p_rounded < 0.05:
        f3_verdict = "fragmentation_DOES_predict_under_coordination"
        f3_statement = (
            "The binary cross-tab DOES support the 'a fragmented ligand "
            "leaves the metal under-coordinated' story in this pool: the "
            "under-or-empty rate differs between single-fragment and "
            f"fragmented rows (two-proportion p = {p_rounded})."
            + (
                f" The continuous correlations are not negligible "
                f"(|r| <= {max_abs_r_rounded})."
                if max_abs_r_rounded is not None
                else ""
            )
        )
    else:
        f3_verdict = "fragmentation_does_NOT_predict_under_coordination"
        f3_statement = (
            "HONEST NEGATIVE. The binary cross-tab does NOT support the "
            "'a fragmented ligand leaves the metal under-coordinated' "
            "story: the under-or-empty rate is statistically "
            "indistinguishable between single-fragment and fragmented rows."
            + (
                f" The continuous correlations are WEAK (|r| <= "
                f"{max_abs_r_rounded})."
                if max_abs_r_rounded is not None
                else " The continuous correlations could not be computed."
            )
            + " Fragmentation and coordination loss are largely INDEPENDENT "
            "in this pool."
        )

    # --- finding 4: is the donor-split mechanism the general cause? -----
    n_explained = explained["n_rows"]
    denom = explained["denominator"]
    frac = explained["fraction"]
    frac_pct = (
        round(float(frac) * 100.0, 2)
        if isinstance(frac, (int, float)) and not isinstance(frac, bool)
        else None
    )
    if frac_pct is None:
        f4_verdict = "no_rows_to_test"
        f4_statement = (
            "NO fragmented, under-coordinated rows were measured in this "
            "pool, so the donor-split mechanism cannot be tested here."
        )
    elif frac_pct < 50.0:
        f4_verdict = "donor_split_mechanism_is_REFUTED_as_the_general_cause"
        f4_statement = (
            "HONEST NEGATIVE. For most fragmented, under-coordinated rows "
            "the donors were NOT split off: the missing coordination is "
            "fully covered by split-off donors in only "
            f"{n_explained} of {denom} rows ({frac_pct} %). A split-off "
            "donor is therefore NOT a general explanation of the "
            "fragmentation."
        )
    else:
        f4_verdict = "donor_split_mechanism_is_NOT_refuted"
        f4_statement = (
            "HONEST NEGATIVE, BUT NOT A REFUTATION. The missing coordination "
            "is fully covered by split-off donors in "
            f"{n_explained} of {denom} rows ({frac_pct} %) — a MAJORITY, so "
            "this run does NOT refute the donor-split mechanism as the "
            "general cause; it only says the mechanism is not universal."
        )

    # --- finding 5: the metal-alone rows --------------------------------
    n_alone = alone["n_rows"]
    n_alone_with_donor = mech["by_coordination_class"]["empty"].get(
        "rows_with_at_least_one_disconnected_donor", 0
    )
    class_means = {
        cls_name: blk.get("disconnected_donor_atoms_mean")
        for cls_name, blk in mech["by_coordination_class"].items()
        if isinstance(blk.get("disconnected_donor_atoms_mean"), (int, float))
    }
    alone_mean = alone.get("disconnected_donor_atoms_mean")
    alone_is_max = bool(
        class_means
        and isinstance(alone_mean, (int, float))
        and alone_mean >= max(float(v) for v in class_means.values())
    )
    if n_alone == 0:
        f5_verdict = "no_metal_alone_rows"
        f5_statement = (
            "NO metal-alone row was measured in this pool, so there is no "
            "case in which the donor split mechanism holds outright."
        )
    elif n_alone_with_donor == n_alone:
        f5_statement = (
            f"The {n_alone} metal-ALONE rows are the one case where the "
            "donor split mechanism holds outright: every one of them has at "
            "least one donor atom sitting in another fragment"
            + (
                ", and they carry the highest disconnected-donor counts of "
                "any coordination class."
                if alone_is_max
                else "."
            )
            + " Their coordination sphere is EMPTY BY CONSTRUCTION, not by a "
            "modelling failure."
        )
        f5_verdict = "metal_alone_rows_ARE_the_donor_split_mechanism"
    else:
        f5_statement = (
            f"The {n_alone} metal-ALONE rows are the one case where the "
            "donor split mechanism holds outright: "
            f"{n_alone_with_donor} of them have at least one donor atom "
            "sitting in another fragment. Their coordination sphere is EMPTY "
            "BY CONSTRUCTION, not by a modelling failure."
        )
        f5_verdict = "metal_alone_rows_PARTLY_the_donor_split_mechanism"

    # --- finding 6: which donor class dominates? ------------------------
    totals = summary["donor_classes"]["totals_across_all_metal_atoms"]
    if totals:
        top_class, top_n = max(totals.items(), key=lambda kv: kv[1])
    else:
        top_class, top_n = "none", 0
    if top_class == "C_donor":
        f6_statement = (
            "The emitted coordination spheres are dominated by CARBON "
            "donors, not by the classical N/O/halide donor sets of the "
            "Pt(II)/Ru(II)/Ir(III) metallodrug literature. That is a "
            "property of the emitted SMILES, and it is a chemistry "
            "observation, not a defect verdict."
        )
        f6_verdict = "carbon_dominant"
    else:
        f6_statement = (
            "The emitted coordination spheres are dominated by "
            f"{top_class} donors ({top_n} donor atoms), not by the "
            "classical N/O/halide donor sets of the Pt(II)/Ru(II)/Ir(III) "
            "metallodrug literature. That is a property of the emitted "
            "SMILES, and it is a chemistry observation, not a defect verdict."
        )
        f6_verdict = f"dominated_by_{top_class}"

    return {
        "finding_1_no_metal_is_over_coordinated": {
            "statement": f1_statement,
            "verdict": f1_verdict,
            "measured": {
                "over": cls["over"],
                "at_capacity": cls["at_capacity"],
                "under": cls["under"],
                "empty": cls["empty"],
                "n_metal_atoms": n_metal_atoms,
            },
        },
        "finding_2_the_oxidation_state_is_not_in_the_smiles": {
            "statement": f2_statement,
            "verdict": f2_verdict,
            "measured": {
                "n_metal_atoms_with_a_nonzero_formal_charge": n_nonzero_charge,
                "n_metal_atoms_with_no_charge_information": n_no_charge_info,
                "n_rows_where_the_smiles_carries_no_charge_information": (
                    summary["oxidation_state"][
                        "n_rows_where_the_smiles_carries_no_charge_information"
                    ]
                ),
                "os_charge_balanced_value_distribution": summary["oxidation_state"][
                    "os_charge_balanced_value_distribution"
                ],
            },
        },
        "finding_3_fragmentation_does_NOT_significantly_predict_under_coordination": {
            "statement": f3_statement,
            "verdict": f3_verdict,
            "measured": {
                "single_fragment_rate": frag["single_fragment"]["rate"],
                "fragmented_rate": frag["fragmented"]["rate"],
                "two_proportion_z": frag["two_proportion_z"],
                "two_proportion_p_two_sided": frag["two_proportion_p_two_sided"],
                "two_proportion_p_two_sided_rounded": p_rounded,
                "pearson_n_frags_vs_cn": xt["pearson_n_frags_vs_cn"],
                "pearson_n_frags_vs_cn_p_two_sided": xt[
                    "pearson_n_frags_vs_cn_p_two_sided"
                ],
                "pearson_largest_frag_frac_vs_cn": xt[
                    "pearson_largest_frag_frac_vs_cn"
                ],
                "pearson_largest_frag_frac_vs_cn_p_two_sided": xt[
                    "pearson_largest_frag_frac_vs_cn_p_two_sided"
                ],
                "max_abs_pearson_r": max_abs_r,
                "max_abs_pearson_r_rounded": max_abs_r_rounded,
            },
        },
        "finding_4_the_donor_split_mechanism_is_REFUTED_as_the_general_cause": {
            "statement": f4_statement,
            "verdict": f4_verdict,
            "measured": {
                "n_rows": n_explained,
                "denominator": denom,
                "fraction": frac,
                "fraction_percent_rounded": frac_pct,
                "fragmented_at_capacity_rows_with_a_split_donor": mech[
                    "counter_evidence"
                ]["n_rows_with_disconnected_donor_at_capacity"],
                "fragmented_at_capacity_denominator": mech["counter_evidence"][
                    "n_rows_with_disconnected_donor_at_capacity_denominator"
                ],
            },
        },
        "finding_5_the_metal_alone_rows_ARE_the_donor_split_mechanism": {
            "statement": f5_statement,
            "verdict": f5_verdict,
            "measured": {
                "n_rows": alone["n_rows"],
                "fraction": alone["fraction"],
                "metal_breakdown": alone["metal_breakdown"],
                "disconnected_donor_atoms_mean": alone["disconnected_donor_atoms_mean"],
                "disconnected_donor_atoms_total": alone[
                    "disconnected_donor_atoms_total"
                ],
                "rows_with_at_least_one_disconnected_donor": n_alone_with_donor,
                "alone_class_mean_is_the_max": alone_is_max,
            },
        },
        "finding_6_the_donor_sets_are_CARBON_DOMINANT": {
            "statement": f6_statement,
            "verdict": f6_verdict,
            "measured": {
                "donor_class_totals": summary["donor_classes"][
                    "totals_across_all_metal_atoms"
                ],
                "top_donor_class": top_class,
                "top_donor_class_n": top_n,
                "n_rows_with_zero_classical_donor": summary["donor_classes"][
                    "n_rows_with_zero_classical_donor"
                ],
                "n_classical_donor_only_carbon_rows": summary["donor_classes"][
                    "n_classical_donor_only_carbon_rows"
                ],
            },
        },
    }


def _frontier_reference_block() -> Dict[str, object]:
    """The ``frontier_reference`` block, with the dated rule-#6 record in place.

    Wf-A189 (2026-11-01): the dated record for the ``what_the_band_is``
    rendering that Wf-A183 replaced is emitted as a SIBLING of that field, in
    the SAME POSITION it occupies in the published artefact, so a re-run
    REPRODUCES the published artefact instead of silently DROPPING the dated
    history (AGENTS.md §5.5 rule #6).
    """
    out: Dict[str, object] = {}
    for key, value in FRONTIER_BAND_CITATION.items():
        out[key] = value
        if key == "what_the_band_is":
            out["what_the_band_is_superseded_2026_11_01_wfa189"] = dict(
                WHAT_THE_BAND_IS_SUPERSEDED_2026_11_01_WFA189
            )
            # Wf-A202 (2026-11-02): the X5 supersession.
            out["what_the_band_is_superseded_2026_11_02_wfa202"] = dict(
                WHAT_THE_BAND_IS_SUPERSEDED_2026_11_02_WFA202
            )
        elif key == "primary_record":
            # Wf-A202 (2026-11-02): the D1 additive annotation.
            out["primary_record_reading_conditional_2026_11_02_wfa202"] = dict(
                PRIMARY_RECORD_READING_CONDITIONAL_2026_11_02_WFA202
            )
        elif key == "not_attributed_to_ef_talfm":
            # Wf-A202 (2026-11-02): the D2 supersession.
            out["not_attributed_to_ef_talfm_superseded_2026_11_02_wfa202"] = dict(
                NOT_ATTRIBUTED_TO_EF_TALFM_SUPERSEDED_2026_11_02_WFA202
            )
    return out


def build_artifact(
    pool_path: Path,
    result: Dict[str, object],
    out_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Assemble the JSON artifact.

    Every number that describes OUR population is COMPUTED here from the
    run; the only external numbers are (a) the frontier band, which is a
    quoted and CITED frontier constant, and (b) the published artifacts this
    run is compared against, which are read back from disk.
    """
    spheres: List[MetalSphere] = result["spheres"]  # type: ignore[assignment]
    summary = result["summary"]
    pop = summary["population"]
    pool_dir = pool_path.parent.name
    n_rows = pop["n_rows"]
    out = out_path if out_path is not None else DEFAULT_OUT

    single_fragment_rate = pop["single_fragment_rate_over_analyzed"]
    n_single = pop["n_single_fragment_rows"]
    end_to_end = _end_to_end_headline()
    end_to_end_n_runs = end_to_end.get("n_runs") if end_to_end else None
    cross_check = _published_cross_check(pool_dir, single_fragment_rate, n_single)

    comparability = build_comparability_statement(
        pool_dir,
        n_rows,
        n_rows_with_explicit_h=pop["n_rows_with_explicit_hydrogen_atoms"],
    )
    what_this_is_not = build_what_this_is_not(
        n_rows=n_rows,
        single_fragment_rate=single_fragment_rate,
        end_to_end_n_runs=end_to_end_n_runs,
        n_breaks=comparability["n_breaks"],
    )

    n_implicit_h = pop["n_metal_atoms_with_an_implicit_h"]
    n_dative = pop["n_dative_metal_incident_bonds"]
    if n_implicit_h == 0:
        hydrogen_clause = (
            "The pool's metals carry no implicit H (GetDegree() == "
            "GetTotalDegree() in every row), so the count is a heavy-atom "
            "neighbour count."
        )
    else:
        hydrogen_clause = (
            f"{n_implicit_h} metal atoms in this pool carry an implicit H "
            "(GetDegree() != GetTotalDegree() for those atoms), so the count "
            "is a heavy-atom neighbour count."
        )
    if n_dative == 0:
        dative_clause = (
            "The pool carries no dative ('->') bond, so every metal-incident "
            "bond is a covalent SINGLE or DOUBLE; a DOUBLE bond still "
            "contributes ONE neighbour (a metal-oxo is one donor)."
        )
    else:
        dative_clause = (
            f"The pool carries {n_dative} dative ('->') metal-incident "
            "bond(s), so NOT every metal-incident bond is a covalent SINGLE "
            "or DOUBLE; a dative bond contributes ONE neighbour too."
        )

    return {
        "metric": "wfa155_metal_coordination_sphere",
        "tag": "MEASURED",
        "date": "2026-11-01",
        "wave": "Wf-A155",
        "branch": "COORD",
        "provenance": {
            "metric_wave": "Wf-A155",
            "frontier_scan_wave": "Wf-A153",
            "narrative_repair_wave": "Wf-A164",
            "guard": "_narrative_self_consistency",
            "note": (
                "the wave ids are MACHINE fields so that a narrative may "
                "name a wave without the narrative inventing a number the "
                "artifact does not carry"
            ),
        },
        "purpose": (
            "Frontier option (D) from the Wf-A153 frontier scan: profile the "
            "METAL's coordination sphere (coordination number, neighbour "
            "element multiset, donor-class breakdown, rule-based oxidation "
            "state) for every row of the generation pool named in "
            "population.pool_path, and cross-tab it against the ligand-only "
            "connectivity to test whether a fragmented ligand leaves the "
            "metal under-coordinated."
        ),
        "population": {
            "pool_path": str(pool_path),
            "pool_directory": pool_dir,
            "pool_kind": (
                "post-hoc single decode of ONE checkpoint — NOT a "
                "training-set-distribution sampling"
            ),
            "columns": ["epoch", "smiles", "n_frags_raw", "largest_frag_frac"],
            "n_rows": n_rows,
            "n_rows_analyzed": pop["n_rows_analyzed"],
            "n_single_fragment_rows": n_single,
            "single_fragment_rate_over_analyzed": single_fragment_rate,
            "single_fragment_rate_percent": (
                round(float(single_fragment_rate) * 100.0, 2)
                if single_fragment_rate is not None
                else None
            ),
        },
        "what_this_is_not": what_this_is_not,
        "comparability_statement": comparability,
        "frontier_reference": _frontier_reference_block(),
        "end_to_end_headline": end_to_end,
        "published_cross_check": cross_check,
        "definitions": {
            "coordination_number": (
                "RDKit GetDegree() of the metal atom = the number of atoms "
                "covalently bonded to it. "
                + hydrogen_clause
                + " "
                + dative_clause
            ),
            "coordination_sphere": (
                "the metal's RDKit bond neighbourhood — the atoms reachable "
                "in ONE bond from the metal WITHIN the metal's own fragment. "
                "A donor sitting in ANOTHER fragment of the same row is NOT "
                "in the sphere; it is counted separately as a "
                "disconnected_donor."
            ),
            "donor_class": dict(DONOR_CLASS_BY_SYMBOL),
            "coordination_class": (
                "empty (CN 0) / under (CN < expected_cn) / at_capacity "
                "(CN == expected_cn) / over (CN > expected_cn), where "
                "expected_cn = coordination_capacity(metal, "
                "DEFAULT_OXIDATION_STATE[metal]) from "
                "molmetal.data.metal_smiles — the metal's USUAL oxidation "
                "state (the repo prior). The secondary classification "
                "coordination_class_os_derived instead uses the capacity "
                "implied by the structure-derived oxidation state."
            ),
            "oxidation_state_layers": dict(OXIDATION_STATE_LAYERS),
        },
        "summary": result["summary"],
        "findings": build_findings(result),
        "findings_note": (
            "the six finding KEYS are legacy schema labels from Wf-A155 and "
            "are kept for compatibility; what THIS run found is in each "
            "finding's 'verdict' field and in its 'statement'"
        ),
        "cross_tab_fragmentation_vs_coordination": result["cross_tab"],
        "candidate_mechanism": result["candidate_mechanism"],
        "per_row_sphere": {
            "format": (
                "columnar: one list per metal atom, fields in the order "
                "given by per_row_columns. row_index is the index into the "
                "pool CSV's data rows (0-based, header excluded)."
            ),
            "per_row_columns": list(PER_ROW_COLUMNS),
            "per_row": _per_row_records(spheres),
        },
        "reproduce": {
            "command": (
                'CUDA_VISIBLE_DEVICES="" .venv/bin/python '
                "molmetal/scripts/wf_a155_metal_coordination_sphere.py "
                f"--pool {pool_path} --out {out}"
            ),
            "cpu_only": True,
            "gpu_used": False,
            "reads": [str(pool_path)],
            "writes": [str(out)],
        },
    }


# ---------------------------------------------------------------------------
# The narrative self-consistency guard (Wf-A164)
# ---------------------------------------------------------------------------

#: The narrative paths whose numbers must be backed by a machine field of the
#: SAME artifact.  A path may designate a leaf, a mapping or a LIST; the
#: list form is the one that matters (see :func:`_resolve`).
NARRATIVE_PATHS: Tuple[str, ...] = (
    "purpose",
    "findings_note",
    "population.pool_kind",
    "what_this_is_not",
    "comparability_statement.summary",
    "comparability_statement.break_1_denominator",
    "comparability_statement.break_2_metric",
    "comparability_statement.break_3_post_processing",
    "comparability_statement.break_4_hydrogens",
    "comparability_statement.breaks",
    "definitions.coordination_number",
    "definitions.coordination_sphere",
    "summary.coordination_number.definition",
    "summary.coordination_number.empty_sphere_note",
    "summary.coordination_class.definition",
    "summary.donor_classes.definition",
    "summary.donor_motif_distribution.definition",
    "summary.oxidation_state.status",
    "summary.geometry_class.definition",
    "summary.metal_alone_in_fragment.definition",
    "summary.population.n_frags_column_mismatch_note",
    "cross_tab_fragmentation_vs_coordination.question",
    "cross_tab_fragmentation_vs_coordination.pearson_note",
    "candidate_mechanism.hypothesis",
    "candidate_mechanism.test",
    "candidate_mechanism.donor_split_fully_explains_the_missing_coordination.definition",
    "candidate_mechanism.counter_evidence.note",
    "reproduce.command",
)

#: STATIC rule text, excluded from the narrative set on purpose: these
#: strings are byte-identical for every pool, so they carry no pool-specific
#: claim, and their rule constants (a coordination-number threshold, an
#: oxidation-state offset) would make the guard FAIL on a pool that happens
#: not to contain a row with that coordination number.  The exclusion is
#: recorded in the guard output so it stays auditable.
EXCLUDED_STATIC_PATHS: Tuple[str, ...] = (
    "definitions.coordination_class",
    "definitions.oxidation_state_layers",
    "summary.coordination_class_secondary_os_derived.definition",
    "summary.oxidation_state.layers",
    "cross_tab_fragmentation_vs_coordination.fragmented_vs_single_fragment.definition",
)


class NarrativeSelfConsistencyError(Exception):
    """Raised when a narrative string carries a number no machine field backs."""

    def __init__(self, artifact: Dict[str, Any], guard: Dict[str, Any]) -> None:
        super().__init__(
            "narrative self-consistency FAILED: "
            + json.dumps(guard.get("unbacked_numbers", {}), ensure_ascii=False)
        )
        self.artifact = artifact
        self.guard = guard


def _leaf_map(artifact: Any) -> Dict[str, Any]:
    """Flatten an artifact to ``dotted.path -> leaf``."""
    out: Dict[str, Any] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                walk(val, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for idx, val in enumerate(node):
                walk(val, f"{path}[{idx}]")
        else:
            out[path] = node

    walk(artifact, "")
    return out


def _resolve(artifact: Any, path: str) -> List[str]:
    """The leaf paths under a dotted narrative path.

    A path may designate a LEAF (``comparability_statement.summary``), a
    MAPPING (``what_this_is_not`` -> ``what_this_is_not.<key>``) or a LIST
    (``comparability_statement.breaks`` -> ``...breaks[0].frontier``, ...).
    The list form matters: with a naive ``path + "."`` prefix the list
    elements would never match, they would be silently classified as
    MACHINE fields, and every number inside them would go unchecked.
    """
    leaves = _leaf_map(artifact)
    if path in leaves:
        return [path]
    prefixes = (path + ".", path + "[")
    return sorted(p for p in leaves if p.startswith(prefixes))


def _numeric_tokens(text: str) -> List[str]:
    """Every numeric literal in ``text``.

    Hand-rolled instead of ``re``: this module's import graph is frozen by
    a static test that allows only a fixed set of stdlib roots, and ``re`` is
    not among them.  The grammar mirrors the reference guard's — digits with
    optional thousands separators (groups of exactly three) and an optional
    decimal part.  The token is returned WITH its separators, so that a prose
    token and a machine string are compared as written.
    """
    tokens: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] not in "0123456789":
            i += 1
            continue
        j = i
        while j < n and text[j] in "0123456789":
            j += 1
        # thousands separators: a comma is part of the number only when it is
        # followed by exactly three digits.
        while (
            j < n
            and text[j] == ","
            and j + 4 <= n
            and text[j + 1] in "0123456789"
            and text[j + 2] in "0123456789"
            and text[j + 3] in "0123456789"
            and (j + 4 >= n or text[j + 4] not in "0123456789")
        ):
            j += 4
        # optional decimal part
        if j < n and text[j] == "." and j + 1 < n and text[j + 1] in "0123456789":
            j += 1
            while j < n and text[j] in "0123456789":
                j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def _is_numeric_char(ch: str) -> bool:
    """True for a digit or a decimal point (the guard's boundary chars)."""
    return bool(ch) and ch in "0123456789."


def _standalone_in_any(token: str, strings: List[str]) -> bool:
    """True when ``token`` appears in one of ``strings`` as a STANDALONE
    number — delimited by characters that are neither digits nor a decimal
    point.

    Deliberately stricter than a substring test: the token ``1`` must NOT
    count as "backed" by the ``155`` in a file name; with a plain substring
    test it would, and the guard would be measurably looser than it looks.
    """
    for text in strings:
        start = 0
        while True:
            idx = text.find(token, start)
            if idx < 0:
                break
            before = text[idx - 1] if idx > 0 else ""
            after = text[idx + len(token)] if idx + len(token) < len(text) else ""
            if not (_is_numeric_char(before) or _is_numeric_char(after)):
                return True
            start = idx + 1
    return False


def _narrative_paths(artifact: Dict[str, Any]) -> List[str]:
    """The narrative paths of this artifact (the static set + the findings)."""
    paths = list(NARRATIVE_PATHS)
    findings = artifact.get("findings")
    if isinstance(findings, dict):
        for key in findings:
            paths.append(f"findings.{key}.statement")
    return paths


def _narrative_self_consistency(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """THE GUARD for the class of defect, not just this instance.

    Wf-A162 (2026-11-01) and Wf-A164 (2026-11-01): a measurement tool that
    embeds POOL-SPECIFIC CLAIMS as string literals inside a ``tag: MEASURED``
    artifact emits TRUE statements for the default pool and FALSE ones for
    any other pool.  The class is stopped here: every number rendered in a
    NARRATIVE string must be backed by a MACHINE field of the same
    artifact, either as a standalone number inside a machine string, or
    equal (to the token's own decimal count) to a machine number.

    A token with no backing is a DEFECT and fails the run.
    """
    leaves = _leaf_map(artifact)
    prose_paths: set = set()
    # A narrative path that resolves to NO leaf would silently disable its own
    # coverage (a renamed field would leave the guard green while checking
    # nothing).  Such a path is a DEFECT, not a skip.
    unresolved: List[str] = []
    for path in _narrative_paths(artifact):
        resolved = _resolve(artifact, path)
        if not resolved:
            unresolved.append(path)
            continue
        prose_paths.update(resolved)
    machine = {p: v for p, v in leaves.items() if p not in prose_paths}

    machine_numbers: List[float] = []
    machine_strings: List[str] = []
    for value in machine.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            machine_numbers.append(float(value))
        elif isinstance(value, str):
            machine_strings.append(value)

    unbacked: Dict[str, List[str]] = {}
    n_prose_numbers = 0
    for path in sorted(prose_paths):
        text = leaves.get(path)
        if not isinstance(text, str):
            continue
        for token in _numeric_tokens(text):
            n_prose_numbers += 1
            norm = token.replace(",", "")
            try:
                value = float(norm)
            except ValueError:  # pragma: no cover - defensive
                continue
            decimals = len(norm.split(".")[1]) if "." in norm else 0
            tol = 0.5 * (10.0 ** (-decimals)) if decimals else 0.0
            backed = _standalone_in_any(token, machine_strings)
            if not backed:
                backed = any(
                    abs(mv - value) <= tol + 1e-12 for mv in machine_numbers
                )
            if not backed:
                unbacked.setdefault(token, []).append(path)
    return {
        "rule": (
            "every number rendered in a narrative string must be backed by a "
            "machine field of the same artifact -- as a standalone number "
            "inside a machine string, or equal (to the token's own decimal "
            "count) to a machine number"
        ),
        "narrative_paths": _narrative_paths(artifact),
        "excluded_static_paths": list(EXCLUDED_STATIC_PATHS),
        "unresolved_narrative_paths": sorted(unresolved),
        "n_prose_leaves": len(prose_paths),
        "n_prose_numbers": n_prose_numbers,
        "n_machine_leaves": len(machine),
        "n_machine_numbers": len(machine_numbers),
        "n_machine_strings": len(machine_strings),
        "unbacked_numbers": {
            tok: sorted(set(paths)) for tok, paths in sorted(unbacked.items())
        },
        "pass": not unbacked and not unresolved and n_prose_numbers > 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    pool = args.pool if args.pool.is_absolute() else PROJECT_ROOT / args.pool
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    if not pool.is_file():
        print(f"ERROR: pool not found: {pool}", file=sys.stderr)
        return 1

    result = analyze_pool(pool)
    artifact = build_artifact(pool, result, out_path=out)
    guard = _narrative_self_consistency(artifact)
    artifact["narrative_self_consistency"] = guard

    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        detect_truncation,
        safe_overwrite_check,
    )

    if not guard["pass"]:
        artifact["status"] = "FAIL"
        artifact["failure"] = {
            "reason": "narrative self-consistency FAILED",
            "unbacked_numbers": guard["unbacked_numbers"],
            "unresolved_narrative_paths": guard["unresolved_narrative_paths"],
            "note": (
                "a number rendered in a narrative string is not backed by any "
                "machine field of this artifact -- the narrative would state "
                "a fact the artifact does not carry"
            ),
        }
        text = json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=False)
        atomic_write_file(out, text)
        print(
            "FAIL: narrative self-consistency; unbacked numbers: "
            + json.dumps(guard["unbacked_numbers"], ensure_ascii=False)
            + "; unresolved narrative paths: "
            + json.dumps(guard["unresolved_narrative_paths"], ensure_ascii=False),
            file=sys.stderr,
        )
        return 1

    artifact["status"] = "PASS"
    text = json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=False)

    risk = safe_overwrite_check(out, text)
    if risk is not None:
        print(
            f"ERROR: refusing to overwrite {out} (diff_ratio={risk:.3f})",
            file=sys.stderr,
        )
        return 1
    atomic_write_file(out, text)
    trunc = detect_truncation(text, expected_min_length=200)
    if trunc is not None:
        print(f"ERROR: artifact looks truncated: {trunc}", file=sys.stderr)
        return 1

    print(f"wrote {out} ({len(text)} chars)")
    print(
        "narrative self-consistency: PASS "
        f"({guard['n_prose_numbers']} prose numbers, "
        f"{guard['n_machine_numbers']} machine numbers)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
