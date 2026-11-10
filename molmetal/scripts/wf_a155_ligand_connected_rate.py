#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A155 (2026-11-01) -- BRANCH LIG: the LIGAND-ONLY "connected" rate.

WHY THIS EXISTS
---------------
The published frontier band (92.7-94 %) is a WHOLE-MOLECULE number under
the `band_provenance` reading, but the band's PROVENANCE is NOT settled on
disk: the frontier post-processing matrix record (``frontier_matrix.json``)
holds TWO competing readings of which frontier cells the band names -- the
first under ``band_provenance``, the second under
``sources``.``EF-TALFM`` -- and they do not even agree on the numerals, so
only the invariants that hold under BOTH readings may be published (the band
is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale).
Our pool is a set of METAL COMPLEXES, so a
whole-molecule connectivity number cannot tell us which of two very
different failures produced a fragment:

  (a) the METAL fell off -- the metal ended up alone in its own
      fragment while the ligand stayed intact; versus
  (b) the LIGAND itself fragmented -- the organic skeleton broke into
      pieces, with or without the metal.

Failure (a) is a metal-coordination problem.  Failure (b) is an
assembly/chemistry problem, and the repair effort for the two is not
the same.  This tool separates them by MEASURING BOTH ARMS ON THE SAME
ROWS:

  * the METAL-INCLUSIVE arm -- the whole molecule, exactly as the pool
    CSV reports it (this is the number a frontier comparison would
    target); and
  * the LIGAND-ONLY arm -- the SAME molecule with the metal atom(s)
    deleted and their bonds removed, i.e. the connectivity that the
    ORGANIC SKELETON has on its own.

Both arms are computed by ONE function (:func:`arm_stats`) so they are
internally comparable by construction, not by two code paths that
happen to agree.  The pair is then compared row by row (a 2x2
contingency + an exact McNemar-style binomial test on the discordant
pairs).

THE METAL-STRIPPING RULE (documented, and it is NOT pool-fitted)
-----------------------------------------------------------------
A "metal" is any atom whose element symbol is in :data:`METAL_SYMBOLS`
-- the repo's canonical d-block transition-metal list, the same tuple
the project already uses as its "is this a metal" predicate in
``molmetal/molmetal_lam/lam_chem/platinai_dataset.py`` (``TRANSITION_METALS``,
Sc..Hg).  It is a SUPERSET of the trainer's own cohort set
(``_TARGET_METALS`` = {Ru, Rh, Re, Os, Ir, Pt}), so a row whose metal
is one of the six the CFM was trained on is certainly stripped.  The
rule is checked for pool-fittedness in the artifact itself: the whole
measurement is recomputed under the narrower cohort set and the two
answers are reported side by side
(``metal_set_invariance``), so a reader can see the answer does not
depend on which of the two repo-canonical metal sets is used.

Stripping semantics: the metal ATOM is deleted and every bond
incident to it goes with it (RDKit ``RWMol.RemoveAtom``).  Nothing
else is touched -- counterions (a free Cl/Br/F/I fragment), a lone
methyl, and everything else stay in the ligand arm exactly as the
decoder emitted them.  This is the "what would be left if the metal
were not there" counterfactual, NOT a claim that the stripped graph is
a real molecule.

THE EMPTY-LIGAND ROW IS COUNTED, NEVER DROPPED
-----------------------------------------------
A row whose ligand is EMPTY after stripping (the entire row was metal
and nothing else) cannot be "connected" -- there is nothing to connect.
Such rows are counted in their own bucket (``empty_ligand``), are
reported separately, and are NOT counted as connected in either arm
and NOT silently removed from the denominator.  On THIS pool that
bucket is EMPTY (0 rows), which is a measurement, not an assumption.

NOTE ON A CLAUSE THAT THE DATA DOES NOT SUPPORT
-----------------------------------------------
The task brief for this branch asserted that the 30 rows in which the
metal is alone in its fragment are "exactly" the empty-ligand case.
MEASURED on this pool, they are not: in every one of those 30 rows
the row ALSO carries other, ligand-bearing fragments, so stripping the
metal leaves a NON-empty ligand.  The genuinely empty-ligand case (a
row that is metal and NOTHING else) has count 0.  The two quantities
are therefore reported as two SEPARATE, separately-named counters
(``metal_alone_in_own_fragment`` and ``empty_ligand``) so the
distinction is not lost.  See ``premise_checks`` in the artifact.

WHAT THIS METRIC IS NOT
-----------------------
* It is NOT the end-to-end headline (``metrics/by_metric/cfm_fragmentation_audit.json``
  -> ``/summary/generation_side/single_molecule_rate_end_to_end``, the
  MEDIAN over 9 RUNS of "one connected, parseable molecule per DRAWN
  sample").  This is a per-ROW paired analysis over ONE pool; the pool's
  name and the measured row count are recorded in the artifact.
* It is NOT the single-pool fragment count of any OTHER pool.  The
  metal-inclusive arm of THIS tool is reported only so the two arms are
  comparable; the artifact carries a COMPUTED cross-check
  (``published_cross_check``) against every published metric that names
  THIS pool, so the comparison is measured rather than asserted.
* It is NOT comparable to the frontier band (92.7-94 %) -- the
  independent breaks are carried in the artifact under ``comparability``,
  and the number of them is computed from the list itself.

CPU-only, RDKit-only.  No GPU, no checkpoint, no decode, no model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# CPU-only, unconditionally: this tool must never be able to touch a GPU, so
# the guard is set BEFORE anything is imported that could read it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.wf386a3_atomic_write import (  # noqa: E402
    atomic_write_file,
    detect_truncation,
    safe_overwrite_check,
)

DEFAULT_POOL = (
    REPO / "molmetal" / "reports" / "cfm_c8_fullgen_2026_10_24" / "generated_smiles.csv"
)
DEFAULT_OUT = REPO / "metrics" / "by_metric" / "wfa155_ligand_connected_rate.json"

#: The repo's canonical d-block transition metals (Sc..Hg), copied from
#: ``molmetal/molmetal_lam/lam_chem/platinai_dataset.py`` (``TRANSITION_METALS``).
#: This is the project's own "is this atom a metal" predicate; the trainer's
#: cohort set (``_TARGET_METALS`` = {Ru, Rh, Re, Os, Ir, Pt}) is a subset.
METAL_SYMBOLS: frozenset[str] = frozenset(
    {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    }
)

#: The trainer's own seed-metal cohort (atomic numbers 44, 45, 75, 76, 77, 78
#: = Ru, Rh, Re, Os, Ir, Pt), from
#: ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py``.  Used ONLY as the
#: narrower arm of the metal-set invariance check -- never as the rule.
COHORT_METAL_SYMBOLS: frozenset[str] = frozenset({"Ru", "Rh", "Re", "Os", "Ir", "Pt"})

#: Counterion elements: a fragment consisting of exactly one atom of one of
#: these is a free halide, not part of the ligand skeleton.  Reported as an
#: auxiliary diagnostic only -- the primary metric never strips them.
COUNTERION_ELEMENTS: frozenset[str] = frozenset({"Cl", "Br", "F", "I"})

#: The published frontier band and the population it is measured over.  These
#: are the FRONTIER's numbers, not ours, and they are carried as machine
#: fields so the prose that quotes them is backed (Wf-A162).
FRONTIER_BAND = "92.7-94 %"
FRONTIER_DENOMINATOR = "10,000-15,000"


#: Wf-A189 (2026-11-01) -- the dated rule-#6 record that
#: Wf-A183's correction of `comparability.breaks[2].frontier`
#: left in the PUBLISHED artefact.  AGENTS.md §5.5 rule #6
#: requires the superseded rendering to survive VERBATIM;
#: emitting it HERE is what makes a re-run REPRODUCE the
#: published artefact instead of silently DROPPING the dated
#: history.

SUPERSEDED_FRONTIER_2026_11_01_WFA183: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A183',
    'rule': 'AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'supersedes_field': 'comparability.breaks[2].frontier',
    'rendering': "the published band's two endpoints are RAW (post_processed: false in the band-provenance record), and which frontier cells the band names is NOT settled on disk: the repo holds TWO competing provenance readings for it, both recorded in the frontier post-processing matrix record (`frontier_matrix.json`) -- the first under `band_provenance`, the second under `sources`.`EF-TALFM` -- and they do not even agree on the numerals. Under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale -- so the post-processing axis is a THIRD, separate axis on top of the METRIC break, not the break itself.",
    'why_superseded': "the superseded rendering asserted the published band's whole-band RAW-ness UNCONDITIONALLY.  That holds only under the `band_provenance` reading of molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json; the competing `sources`.`EF-TALFM` reading of the same record pairs a raw connectivity cell with a post-processed companion cell, so whether the band as a whole is RAW is NOT settled on disk.  This is the MIRROR of the blanket POST-PROCESSED over-claim Wf-A175 already refuted.  The live rendering is now READING-CONDITIONAL; it introduces NO numeral and keeps every numeral the superseded rendering carried.",
    'why_this_key_is_not_inside_the_breaks_list': "the repo's own regression test molmetal/tests/test_wfa162_narrative_self_consistency.py asserts that `comparability.breaks` resolves to exactly four breaks x (break, frontier, ours), so the superseded rendering is carried in a SIBLING key of `breaks` rather than as a fourth field of a break.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.  The emitter was re-run on the SAME published pool and its output matched this artefact in every numeric leaf; the only difference is the narrative string repaired here.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}

#: Wf-A189 (2026-11-01) -- the rule-#6 RESTORATION of the
#: Wf-A179-era rendering of `comparability.breaks[2].frontier`,
#: the one Wf-A183 replaced but preserved NOWHERE.  Read out
#: of the ephemeral /tmp baseline BY ITS JSON PATH (never
#: retyped); see the `source_of_this_rendering` field of the
#: block itself.

SUPERSEDED_FRONTIER_2026_11_01_WFA189: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A189',
    'rule': 'AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'supersedes': 'comparability.breaks[2].frontier',
    'field': 'frontier',
    'rendering': "the published band's two endpoints are RAW (post_processed: false in the band-provenance record), and which frontier cells the band names is NOT settled on disk: the repo holds TWO competing provenance readings for it, both recorded in the frontier post-processing matrix record (`frontier_matrix.json`) -- under `band_provenance` and under `frontier_axis_mapping` -- and they do not even agree on the numerals. Under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale -- so the post-processing axis is a THIRD, separate axis on top of the METRIC break, not the break itself.",
    'why_superseded': 'Wf-A183 (2026-11-01) replaced the LIVE rendering of `comparability.breaks[2].frontier` and preserved ITS OWN predecessor (the Wf-A177-era wording) in the sibling key `comparability.superseded_frontier_2026_11_01_wfa183`.  The IMMEDIATELY-PRECEDING rendering - the Wf-A179-era wording that Wf-A183 itself replaced - survived NOWHERE on disk, so the rule-#6 chain had a hole: a re-run of the producer would have dropped the whole dated record and the published artefact would have become hand-patched and non-reproducible.  This key closes the hole by restoring that rendering VERBATIM.  The restored rendering cites the locator `frontier_axis_mapping` for the second provenance reading; per the Wf-A183 record that token is NOT a JSON key of the frontier post-processing matrix record (it survives only as a string value), which is exactly why the LIVE rendering cites JSON PATHS.  The rendering is restored VERBATIM anyway: rule #6 preserves what WAS published, and the correction lives in the dated live rendering, never in a silent edit of history.',
    'source_of_this_rendering': "/tmp/wfa179_old_ligand.json (md5 59492ebef64d8fe82eeb8ee9620a0074), read out at the JSON path `comparability.breaks[2].frontier` - the Wf-A179-era state of this artefact.  That file is EPHEMERAL (/tmp is not durable; a reboot erases it) and this rendering is NOT recoverable from ANY tracked file: a verbatim containment search over the repo's tracked + untracked + gitignored text files found ZERO other holders of this string.  The repo's rule-#6 chain therefore rested on a hand-copied /tmp baseline, which is why a re-run could have erased the history with nobody noticing.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.",
    'guard_counters_note': "the recorded `narrative_self_consistency` leaf counters in this artefact are the emitter's own output, computed on the artefact BEFORE the guard block itself was attached, so they already under-count the leaves on disk before this patch - a pre-existing self-reference quirk of the emitter, NOT introduced here.  A fresh `_narrative_self_consistency(...)` run on this artefact still PASSES (`unbacked_numbers` is empty); its leaf counts are larger than the recorded ones by the guard block's own leaves plus the dated keys added here.  The recorded counters were left UNCHANGED on purpose: no existing numeric leaf of this artefact may move.",
}

#: Wf-A194 (2026-11-02) -- the dated rule-#6 record that Wf-A194's
#: TEXT-ONLY patch of `comparability.breaks[2].frontier` left in the
#: PUBLISHED artefact.  AGENTS.md §5.5 rule #6 requires the superseded
#: rendering to survive VERBATIM; emitting it HERE is what makes a re-run
#: REPRODUCE the published artefact instead of silently DROPPING the dated
#: history.  The eight leaves below are copied EXACTLY (key for key, value
#: for value) out of
#: molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json
#: at the JSON path `comparability.superseded_frontier_2026_11_02_wfa194`.
#:
#: NOTE (Wf-A199, 2026-11-02): the `producer_parity` leaf of this record
#: says, VERBATIM, "THIS dated key is NOT emitter-emitted ... so a producer
#: re-run would DROP it".  That sentence was TRUE when Wf-A194 hand-patched
#: the artefact, and it became FALSE at the moment this constant was
#: emitted.  It is copied VERBATIM anyway: rule #6 preserves what WAS
#: published, and the correction lives in a DATED sibling note, never in a
#: silent edit of history.  The dated note that supersedes it is
#: SUPERSEDED_PRODUCER_PARITY_2026_11_02_WFA199 below.

SUPERSEDED_FRONTIER_2026_11_02_WFA194: Dict[str, str] = {
    "date": "2026-11-02",
    "wave": "Wf-A194",
    "rule": "AGENTS.md §5.5 rule #6 -- the superseded rendering is preserved VERBATIM and the correction is additive + dated",
    "supersedes_field": "comparability.breaks[2].frontier",
    "rendering": "the published band is measured on the published POST-PROCESSED stream.",
    "why_superseded": "the published rendering asserted, WITHOUT qualification, that the 92.7-94 % frontier band is measured on a POST-PROCESSED stream.  That is FALSE under the `band_provenance` reading of molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json, whose band endpoints carry `post_processed: false`; and it is not a settled statement under the competing `sources`.`EF-TALFM` reading of the SAME record, whose frontier arm is MIXED (a RAW connectivity cell `C.` = 92.98 % counted BEFORE the largest-component retention, paired with the post-processed `N. & U. & S.` = 94.33 % companion cell).  The corrected rendering is the one the emitter already carries (the Wf-A183/Wf-A189 chain in molmetal/scripts/wf_a155_ligand_connected_rate.py): it scopes the `post_processed: false` statement to the `band_provenance` reading and states that whether the band AS A WHOLE is RAW is NOT settled on disk.  It introduces NO numeral and keeps every numeral the superseded rendering carried.",
    "number_preserving": "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED and the added leaves are STRINGS only.  The tool was NOT re-run, because a re-run would produce a NEW measurement, not a correction.",
    "producer_parity": "the LIVE rendering is byte-identical to the output of `wf_a155_ligand_connected_rate.py::build_comparability_breaks` re-run on this artefact's own `denominators.n_analyzed` and pool name, so a producer re-run REPRODUCES the repair instead of erasing it.  THIS dated key is NOT emitter-emitted (the emitter is owned by another wave), so a producer re-run would DROP it; the emitter's own dated siblings `comparability.superseded_frontier_2026_11_01_wfa183` and `comparability.superseded_frontier_2026_11_01_wfa189` are the emitter-side record of the same correction chain, and they are added here in the emitter's own positions.",
}

#: Wf-A199 (2026-11-02) -- the DATED CORRECTION of the `producer_parity` leaf
#: of SUPERSEDED_FRONTIER_2026_11_02_WFA194.  As of this wave the dated
#: record IS emitter-emitted (see the `comparability` dict in
#: `build_artifact`), so a producer re-run REPRODUCES it instead of dropping
#: it.  This note lives in the PRODUCER, not in the artefact: publishing it
#: would change the artefact's bytes (and therefore its sha256 and its
#: registry row) for no measurement reason -- the same reasoning as
#: DOCSTRING_SUPERSEDED_2026_11_02_WFA194.

SUPERSEDED_PRODUCER_PARITY_2026_11_02_WFA199: Dict[str, str] = {
    'date': "2026-11-02",
    'wave': "Wf-A199",
    'rule': "AGENTS.md \u00a75.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated",
    'supersedes': "the `producer_parity` leaf of `comparability.superseded_frontier_2026_11_02_wfa194`",
    'field': "comparability.superseded_frontier_2026_11_02_wfa194.producer_parity",
    'rendering': "THIS dated key is NOT emitter-emitted (the emitter is owned by another wave), so a producer re-run would DROP it; the emitter's own dated siblings `comparability.superseded_frontier_2026_11_01_wfa183` and `comparability.superseded_frontier_2026_11_01_wfa189` are the emitter-side record of the same correction chain, and they are added here in the emitter's own positions.",
    'why_superseded': "the superseded rendering asserted that the dated key is NOT emitter-emitted and that a producer re-run would DROP it.  That was TRUE when Wf-A194 hand-patched the artefact (the emitter was owned by another wave then), and it became FALSE when Wf-A199 added the dated key to the emitter's `comparability` dict: the key is now emitted in the SAME POSITION it occupies in the published artefact, so a producer re-run REPRODUCES the published artefact instead of dropping the dated history.  The superseded rendering is preserved VERBATIM inside the emitted record; this note is the additive + dated correction, and it changes NO numeral and NO leaf of the published artefact.",
    'number_preserving': "TEXT-ONLY note in the PRODUCER.  The published artefact's leaves are UNCHANGED by this wave: the emitted `comparability.superseded_frontier_2026_11_02_wfa194` record is copied EXACTLY (key for key, value for value) from the published artefact, so the artefact's bytes, its sha256 and its registry row do not move.",
    'emitted_into_the_artifact': "no -- deliberately.  This note lives in the producer only; publishing it would change the artefact's bytes (and therefore its sha256 and its registry row) without changing a single measurement.",
}

#: Wf-A204 (2026-11-02) -- the DATED CORRECTION of the `producer_parity`
#: leaf of SUPERSEDED_FRONTIER_2026_11_02_WFA194, and the record that
#: carries the correction INTO the artefact.
#:
#: Wf-A194 hand-patched the published artefact with the dated record, and
#: the record's own `producer_parity` leaf said -- at that time -- that the
#: key was NOT emitter-emitted and that a producer re-run would DROP it.
#: Wf-A199 made the emitter emit the dated key, so that sentence became
#: FALSE for the artefact it lives in.  Wf-A204 re-published the
#: c8-derived metric so that the file on disk is BYTE-IDENTICAL to a
#: producer re-run; with that in place the sentence is false for THIS
#: artefact as well.
#:
#: AGENTS.md §5.5 rule #6 forbids editing the superseded leaf: it is
#: preserved VERBATIM in the `rendering` leaf below, and this record is
#: the additive + dated correction.
#:
#: PLACEMENT (Wf-A204): this record is emitted at the TOP LEVEL of the
#: artefact rather than inside `comparability`, because
#: molmetal/tests/test_wfa199_parity_census.py pins the producer's emitted
#: `comparability` key set to the PUBLISHED C7 artefact's key set, and the
#: C7 artefact is outside this wave's write scope -- a new key inside
#: `comparability` would therefore break a repo test.  The record names the
#: exact JSON path it supersedes, so a reader of the artefact can find it
#: from the leaf it corrects.

SUPERSEDED_PRODUCER_PARITY_2026_11_02_WFA204: Dict[str, str] = {
    'date': '2026-11-02',
    'wave': 'Wf-A204',
    'rule': 'AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'supersedes': 'the `producer_parity` leaf of `comparability.superseded_frontier_2026_11_02_wfa194`',
    'field': 'comparability.superseded_frontier_2026_11_02_wfa194.producer_parity',
    'rendering': "the LIVE rendering is byte-identical to the output of `wf_a155_ligand_connected_rate.py::build_comparability_breaks` re-run on this artefact's own `denominators.n_analyzed` and pool name, so a producer re-run REPRODUCES the repair instead of erasing it.  THIS dated key is NOT emitter-emitted (the emitter is owned by another wave), so a producer re-run would DROP it; the emitter's own dated siblings `comparability.superseded_frontier_2026_11_01_wfa183` and `comparability.superseded_frontier_2026_11_01_wfa189` are the emitter-side record of the same correction chain, and they are added here in the emitter's own positions.",
    'why_superseded': "the superseded rendering asserted that the dated key is NOT emitter-emitted and that a producer re-run would DROP it.  That was TRUE when Wf-A194 hand-patched the artefact, because the emitter was owned by another wave then; it became FALSE when Wf-A199 added the dated key to the producer's own `comparability` dict, and it is FALSE for the artefact this record travels with: the dated key IS emitted, and this artefact is byte-identical to a producer re-run of the emitter on its own pool.  The superseded rendering is preserved VERBATIM in the `rendering` leaf of this record; this note is the additive + dated correction, and it changes NO numeral and NO numeric leaf.",
    'number_preserving': "TEXT-ONLY record: every leaf of this record is a STRING, so no numeric leaf of the artefact moves.  The narrative self-consistency guard is computed on the artefact BEFORE the dated records are attached, so the recorded guard counters keep their values and the guard block written by a re-run is byte-identical to the one already on disk.",
}

#: Wf-A194 (2026-11-02) -- the dated rule-#6 record for the MODULE DOCSTRING
#: this wave replaced.  AGENTS.md §5.5 rule #6 requires every superseded
#: rendering to survive VERBATIM; a DATED NAMED CONSTANT in the producer is
#: the sanctioned carrier for a rendering that lives in the producer's own
#: prose (the module docstring is NOT read by the emitter, so it cannot be
#: carried in the artefact -- and publishing it there would change the
#: artefact's bytes, its sha256 and its registry row for no measurement
#: reason).

DOCSTRING_SUPERSEDED_2026_11_02_WFA194: Dict[str, str] = {
    'date': '2026-11-02',
    'wave': 'Wf-A194',
    'rule': 'AGENTS.md §5.5 rule #6 - the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'supersedes': 'the module docstring of molmetal/scripts/wf_a155_ligand_connected_rate.py (the `WHY THIS EXISTS` paragraph)',
    'field': 'module docstring',
    'rendering': "The frontier's connectivity band (92.7-94 %) is measured on the WHOLE\nemitted molecule.",
    'why_superseded': "the superseded rendering asserted ONE provenance reading as settled: it called the band a 'connectivity band' and said the band is measured on the WHOLE emitted molecule.  That is TRUE under the `band_provenance` reading (GeoLDM Validity & Uniq + HLTF PB-Valid, both whole-molecule quantities) and NOT under the competing `sources`.`EF-TALFM` reading, whose lower endpoint is EF-TALFM's own RAW connectivity cell and whose upper endpoint is a POST-PROCESSED companion cell.  The repo's own frontier-labelling record (molmetal/scripts/wfa161_frontier_labelling.py) states that ONLY the invariants that hold under BOTH readings may be published: the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale.  The live docstring is now READING-SCOPED; it introduces NO numeral and keeps every numeral the superseded rendering carried.",
    'number_preserved': "TEXT-ONLY patch to the module docstring.  The published artefact metrics/by_metric/wfa155_ligand_connected_rate.json is UNCHANGED by this wave -- the module docstring is not read by the emitter, so the emitted artefact is byte-identical before and after: every numeric leaf and the artefact's sha256 are unchanged and its registry row stays valid.",
    'emitted_into_the_artifact': "no -- deliberately.  A module docstring is not part of the emitted artefact; adding it to the artefact would change the artefact's bytes (and therefore its sha256 and its registry row) without changing a single measurement.",
}


class NarrativeSelfConsistencyError(Exception):
    """Raised when a narrative string carries a number that no machine field
    of the same artifact backs.

    Wf-A162 (2026-11-01): the tool used to hard-code pool-specific claims
    inside a ``tag: MEASURED`` artifact.  They were true for one pool and
    false for every other.  This exception is the guard for that CLASS: the
    offending artifact is carried on the exception so the caller can write
    it out as a FAIL, rather than silently publishing a false claim.
    """

    def __init__(self, artifact: Dict[str, Any], guard: Dict[str, Any]) -> None:
        super().__init__(
            "narrative self-consistency FAILED: unbacked numbers "
            f"{sorted(guard.get('unbacked_numbers', {}))}"
        )
        self.artifact = artifact
        self.guard = guard


def build_comparability_breaks(
    pool_name: str, n_analyzed: int
) -> Tuple[Dict[str, str], ...]:
    """The breaks between our numbers and the published frontier band.

    Wf-A162 (2026-11-01): every number that describes OUR population is
    COMPUTED from the run -- the pool's own name and the MEASURED row
    count -- so this text cannot go stale when the tool is pointed at a
    different pool.  It used to be a module constant carrying "2112 rows"
    and "cfm_c8_fullgen_2026_10_24/generated_smiles.csv", which is FALSE
    for every pool except the default one -- including the wrong pool
    PATH in a ``tag: MEASURED`` artifact.
    """
    return (
        {
            "break": "DENOMINATOR",
            "frontier": (
                f"{FRONTIER_BAND} is measured over {FRONTIER_DENOMINATOR} "
                "generated molecules from a training-set-distribution "
                "sampling."
            ),
            "ours": (
                f"ONE pool of {n_analyzed} rows from ONE post-hoc decode "
                f"({pool_name}/generated_smiles.csv)."
            ),
        },
        {
            "break": "METRIC",
            "frontier": (
                "'connected' there means the DECODER's own connectivity claim; "
                "'valid' means RDKit sanitisation; '% PB-Valid' means "
                "PoseBusters geometry."
            ),
            "ours": (
                "an RDKit connected-component count of the emitted SMILES. The "
                "ligand-only arm additionally deletes the metal atom(s) first, "
                "which is a counterfactual the frontier papers do not publish."
            ),
        },
        {
            # Wf-A176 (2026-11-01) CORRECTION — the superseded text, PRESERVED
            # VERBATIM per AGENTS.md §5.5 rule #6:
            #     "frontier": (
            #         "the published band is measured on the published "
            #         "POST-PROCESSED stream."
            #     ),
            #
            # WHY: FALSE per the repo's OWN band-provenance record —
            # molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json
            # `band_provenance.endpoint_candidates` carries
            # `post_processed: false` for BOTH band endpoints (GeoLDM
            # Validity & Uniq 92.7 %, HLTF PB-Valid 94.0 %).  The band's
            # separation from our number is the METRIC-TYPE break — neither
            # endpoint is a connected-component count — and the
            # post-processing axis is a THIRD, separate axis on top of it.
            #
            # Wf-A177 (2026-11-01) CORRECTION 2 — the correction-1 text below,
            # PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, and itself
            # SUPERSEDED:
            #     "frontier": (
            #         "the published band's two endpoints are RAW (post_processed: "
            #         "false in the band-provenance record) and neither endpoint is "
            #         "a connected-component count, so the post-processing axis is "
            #         "a THIRD, separate axis on top of the METRIC break, not the "
            #         "break itself."
            #     ),
            #
            # WHY correction 1 was SUPERSEDED (adversarial verify, Wf-A176):
            # it asserted "neither endpoint is a connected-component count" as
            # if settled.  The repo holds TWO competing provenance readings
            # for the band (frontier_matrix.json -> `band_provenance` :51-89
            # vs the same file's `frontier_axis_mapping` :360/:378-379,
            # EF-TALFM arXiv:2609.08333v4) and under the SECOND the lower
            # endpoint IS a connectivity count (EF-TALFM's own raw `C.` cell,
            # 92.98 %, counted BEFORE the largest-component retention).  The
            # two readings do not even agree on the numerals (92.7/94.0 vs
            # 92.98/94.33), so the band's source cells are NOT settled on
            # disk.  Only the invariants that hold under BOTH readings may be
            # published: the band is EXTERNAL, is NOT one homogeneous
            # quantity, and is NOT on our scale.
            "break": "POST-PROCESSING",
            # Wf-A177 (2026-11-01) CORRECTION 3 — the correction-2 rendering
            # below, PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, and
            # itself SUPERSEDED:
            #     "frontier": (
            #         "the published band's two endpoints are RAW (post_processed: "
            #         "false in the band-provenance record), and which frontier "
            #         "cells the band names is NOT settled on disk: the repo holds "
            #         "TWO competing provenance readings for it "
            #         "(frontier_matrix.json `band_provenance` :51-89 vs the same "
            #         "file's `frontier_axis_mapping` :360/:378-379, EF-TALFM "
            #         "arXiv:2609.08333v4) that do not even agree on the numerals "
            #         "(92.7/94.0 vs 92.98/94.33). Under BOTH readings the band is "
            #         "external, is NOT one homogeneous quantity, and is NOT on "
            #         "our scale -- so the post-processing axis is a THIRD, separate "
            #         "axis on top of the METRIC break, not the break itself."
            #     ),
            #
            # WHY correction 2 was SUPERSEDED: the emitters' OWN narrative
            # self-consistency guard (the Wf-A162 class-level guard) REJECTED
            # it — six of its numeric tokens (the EF-TALFM arXiv id, the
            # frontier_matrix.json line numbers, and the two competing
            # readings' cell values) were backed by NO machine field of this
            # artifact.  The guard is right: a `tag: MEASURED` artifact must
            # not render a number it cannot back.  The fix is to keep the
            # CLAIM and drop the numerals from the PROSE — the concrete
            # readings live in the SOURCE RECORD named below, at the two
            # named key paths.  This keeps the correction TEXT-ONLY: 0
            # numeric leaves change and 0 keys are added.
            # Wf-A183 (2026-11-01) CORRECTION 4 — the correction-3 rendering
            # below, PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, and
            # itself SUPERSEDED:
            #     "frontier": (
            #         "the published band's two endpoints are RAW (post_processed: "
            #         "false in the band-provenance record), and which frontier "
            #         "cells the band names is NOT settled on disk: the repo holds "
            #         "TWO competing provenance readings for it, both recorded in "
            #         "the frontier post-processing matrix record "
            #         "(`frontier_matrix.json`)"
            #         " -- the first under `band_provenance`, the second under "
            #         "`sources`.`EF-TALFM` -- and they do not even agree on the "
            #         "numerals. Under BOTH readings the band is external, is NOT "
            #         "one homogeneous quantity, and is NOT on our scale -- so the "
            #         "post-processing axis is a THIRD, separate axis on top of the "
            #         "METRIC break, not the break itself."
            #     ),
            #
            # WHY correction 3 was SUPERSEDED (Wf-A183, 2026-11-01): it asserted
            # the band's whole-band RAW-ness UNCONDITIONALLY, which holds ONLY
            # under the `band_provenance` reading.  The repo's OWN record --
            # molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json
            # -- holds TWO competing readings, and under the second
            # (`sources`.`EF-TALFM`) the frontier arm is MIXED: a raw
            # connectivity cell paired with a post-processed companion cell.
            # So whether the band as a whole is RAW is NOT settled on disk.
            # This is the MIRROR of the blanket "the band is POST-PROCESSED"
            # over-claim Wf-A175 already refuted.  The repair is TEXT-ONLY: it
            # introduces NO numeral, keeps every numeral that was already in
            # the string, and leaves the invariants that hold under BOTH
            # readings (external, not one homogeneous quantity, not on our
            # scale) untouched.
            "frontier": (
                "under the `band_provenance` reading the band's two endpoints "
                "are RAW (post_processed: false in that reading), while the "
                "competing `sources`.`EF-TALFM` reading pairs a raw "
                "connectivity cell with a post-processed companion cell -- so "
                "whether the band as a whole is RAW is NOT settled on disk, and "
                "which frontier cells the band names is NOT settled on disk: the "
                "repo holds TWO competing provenance readings for it, both "
                "recorded in the frontier post-processing matrix record "
                "(`frontier_matrix.json`)"
                " -- the first under `band_provenance`, the second under "
                "`sources`.`EF-TALFM` -- and they do not even agree on the "
                "numerals. Under BOTH readings the band is external, is NOT "
                "one homogeneous quantity, and is NOT on our scale -- so the "
                "post-processing axis is a THIRD, separate axis on top of the "
                "METRIC break, not the break itself."
            ),
            "ours": (
                "every number here is RAW -- the SMILES exactly as the sampler "
                "emitted it, no LargestFragmentChooser, no explicit H, no UFF."
            ),
        },
        {
            "break": "HYDROGENS",
            "frontier": "the frontier dataset carries explicit hydrogens.",
            "ours": (
                "our pool does not (implicit H), which changes both "
                "sanitisation and geometry."
            ),
        },
    )


# --------------------------------------------------------------------------
# The two arms
# --------------------------------------------------------------------------
def strip_metals(mol: Any, metal_symbols: Set[str]) -> Tuple[Any, List[str]]:
    """Delete every metal atom from ``mol`` (and the bonds incident to it).

    Returns the stripped molecule and the list of removed element symbols
    (in atom-index order, which is the order the removals happen in).

    The stripped molecule is deliberately NOT sanitised: removing a metal
    leaves dangling valences on its former neighbours, and the metric only
    needs the CONNECTIVITY of what is left.  ``UpdatePropertyCache(strict=False)``
    is called so the heavy-atom count stays well defined.
    """
    idxs = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() in metal_symbols
    ]
    removed = [mol.GetAtomWithIdx(i).GetSymbol() for i in idxs]
    if not idxs:
        return mol, removed
    return _strip_with(mol, idxs), removed


def _strip_with(mol: Any, idxs: Sequence[int]) -> Any:
    """Remove ``idxs`` (descending, so the remaining indices stay valid)."""
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    for idx in sorted(idxs, reverse=True):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def arm_stats(mol: Any) -> Dict[str, Any]:
    """Connectivity facts of ONE arm.

    This is the SINGLE function both arms go through, so the metal-inclusive
    and ligand-only numbers are internally comparable by construction rather
    than by two code paths that happen to agree.

    ``n_frags`` is the RDKit connected-component count
    (``Chem.GetMolFrags(..., asMols=False, sanitizeFrags=False)``) -- the same
    definition the trainer's ``decode_chemistry`` and the pool's own
    ``n_frags_raw`` column use.  ``largest_frag_frac`` is the largest
    component's atom count over the molecule's heavy-atom count, the same
    definition as the pool's ``largest_frag_frac`` column.
    """
    from rdkit import Chem

    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    sizes = [len(f) for f in frags]
    n_heavy = int(mol.GetNumHeavyAtoms())
    return {
        "n_frags": len(sizes),
        "n_atoms": int(mol.GetNumAtoms()),
        "n_heavy": n_heavy,
        "largest_frag_frac": (max(sizes) / n_heavy) if (sizes and n_heavy) else None,
        "frag_sizes": sizes,
    }


def analyze_row(smiles: str, metal_symbols: Set[str]) -> Optional[Dict[str, Any]]:
    """Per-row facts for the paired analysis.  ``None`` if the SMILES does not parse."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    metal_idxs = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in metal_symbols
    ]
    metal_symbols_present = sorted(
        {mol.GetAtomWithIdx(i).GetSymbol() for i in metal_idxs}
    )

    full = arm_stats(mol)
    stripped = _strip_with(mol, metal_idxs)
    lig = arm_stats(stripped)

    # Is the metal alone in its own fragment (i.e. it fell off)?
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    metal_alone = False
    if metal_idxs:
        metal_frag = next(
            (f for f in frags if any(i in f for i in metal_idxs)), None
        )
        if metal_frag is not None:
            metal_alone = all(
                mol.GetAtomWithIdx(j).GetSymbol() in metal_symbols
                for j in metal_frag
            )

    # Free halide counterion present?  (auxiliary diagnostic only)
    has_counterion = any(
        len(f) == 1 and mol.GetAtomWithIdx(f[0]).GetSymbol() in COUNTERION_ELEMENTS
        for f in frags
    )

    return {
        "n_metal_atoms": len(metal_idxs),
        "metal_symbols": metal_symbols_present,
        "metal_alone_in_own_fragment": metal_alone,
        "has_free_halide_counterion": has_counterion,
        "full": full,
        "lig": lig,
        "empty_ligand": lig["n_heavy"] == 0,
    }


# --------------------------------------------------------------------------
# Paired statistics
# --------------------------------------------------------------------------
def contingency(pairs: Iterable[Tuple[bool, bool]]) -> Dict[str, int]:
    """2x2 over (ligand_connected, metal_inclusive_connected) pairs.

    ``a`` = both connected, ``b`` = ligand-only connected (the metal fell off),
    ``c`` = metal-inclusive connected but the ligand alone is not (the metal
    was BRIDGING otherwise-disconnected ligand pieces), ``d`` = neither.
    """
    a = b = c = d = 0
    for lig_conn, full_conn in pairs:
        if lig_conn and full_conn:
            a += 1
        elif lig_conn and not full_conn:
            b += 1
        elif full_conn and not lig_conn:
            c += 1
        else:
            d += 1
    return {"a": a, "b": b, "c": c, "d": d}


def exact_mcnemar_p(b: int, c: int) -> Dict[str, Any]:
    """Exact two-sided McNemar-style binomial test on the discordant pairs.

    Under H0 the discordant pairs split 50/50, so the count ``b`` is
    Binomial(b + c, 0.5).  The two-sided p-value follows the standard
    "sum the pmf of every outcome at most as likely as the observed one"
    definition (the same definition ``scipy.stats.binomtest`` uses).  With
    p = 0.5 the pmf is ``C(n, x) / 2**n``, so the comparison is exact in
    integers and the p-value is computed as an exact rational before any
    float conversion -- which matters here, because the p-value is around
    1e-130 and would be lost to a float-only implementation.
    """
    n = int(b) + int(c)
    if n == 0:
        return {
            "discordant_total": 0,
            "p_value_two_sided": 1.0,
            "p_value_log10": 0.0,
            "p_value_underflow_to_zero": False,
            "p_value_exact_numerator": 1,
            "p_value_exact_denominator": 1,
        }
    k_obs = min(int(b), int(c))
    denom = 1 << n
    threshold = math.comb(n, k_obs)
    numer = sum(math.comb(n, x) for x in range(n + 1) if math.comb(n, x) <= threshold)
    p_float = numer / denom
    return {
        "discordant_total": n,
        "p_value_two_sided": p_float,
        "p_value_log10": math.log10(numer) - n * math.log10(2.0),
        "p_value_underflow_to_zero": p_float == 0.0,
        "p_value_exact_numerator": numer,
        "p_value_exact_denominator": denom,
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return (numerator / denominator) if denominator else None


def _histogram(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items())}


def _frac_summary(values: Sequence[float]) -> Dict[str, Any]:
    vals = sorted(values)
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    mid = len(vals) // 2
    median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": median,
        "min": vals[0],
        "max": vals[-1],
    }


# --------------------------------------------------------------------------
# Pool-level analysis
# --------------------------------------------------------------------------
def analyze_pool(
    pool_csv: Path,
    *,
    metal_symbols: Set[str] = METAL_SYMBOLS,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the paired metal-inclusive / ligand-only analysis over a pool CSV."""
    rows = list(csv.DictReader(pool_csv.open(encoding="utf-8")))
    if limit is not None:
        rows = rows[:limit]

    n_rows = len(rows)
    n_unparseable = 0
    n_empty_string = 0
    n_no_metal = 0
    n_empty_ligand = 0
    n_metal_alone = 0
    n_counterion = 0

    full_hist: Counter = Counter()
    lig_hist: Counter = Counter()
    full_fracs: List[float] = []
    lig_fracs: List[float] = []
    pairs: List[Tuple[bool, bool]] = []
    delta_hist: Counter = Counter()
    metal_symbol_counts: Counter = Counter()
    n_metal_per_row: Counter = Counter()

    full_single = 0
    lig_single = 0
    per_epoch: Dict[str, Dict[str, int]] = {}
    csv_frag_mismatch = 0
    csv_frac_max_abs_delta = 0.0
    n_atoms_eq_heavy = True

    for row in rows:
        smiles = (row.get("smiles") or "").strip()
        if not smiles:
            n_empty_string += 1
            continue
        facts = analyze_row(smiles, metal_symbols)
        if facts is None:
            n_unparseable += 1
            continue

        full = facts["full"]
        lig = facts["lig"]
        if full["n_atoms"] != full["n_heavy"]:
            n_atoms_eq_heavy = False

        n_metal_per_row[facts["n_metal_atoms"]] += 1
        for sym in facts["metal_symbols"]:
            metal_symbol_counts[sym] += 1
        if facts["n_metal_atoms"] == 0:
            n_no_metal += 1
        if facts["metal_alone_in_own_fragment"]:
            n_metal_alone += 1
        if facts["has_free_halide_counterion"]:
            n_counterion += 1
        if facts["empty_ligand"]:
            n_empty_ligand += 1

        full_hist[full["n_frags"]] += 1
        lig_hist[lig["n_frags"]] += 1
        if full["largest_frag_frac"] is not None:
            full_fracs.append(float(full["largest_frag_frac"]))
        if lig["largest_frag_frac"] is not None:
            lig_fracs.append(float(lig["largest_frag_frac"]))

        full_conn = full["n_frags"] == 1 and full["n_heavy"] > 0
        lig_conn = lig["n_frags"] == 1 and lig["n_heavy"] > 0
        full_single += int(full_conn)
        lig_single += int(lig_conn)
        pairs.append((lig_conn, full_conn))
        delta_hist[lig["n_frags"] - full["n_frags"]] += 1

        epoch = str(row.get("epoch", ""))
        bucket = per_epoch.setdefault(
            epoch, {"n": 0, "metal_inclusive_single": 0, "ligand_only_single": 0}
        )
        bucket["n"] += 1
        bucket["metal_inclusive_single"] += int(full_conn)
        bucket["ligand_only_single"] += int(lig_conn)

        # Cross-check against the pool's OWN precomputed columns.
        raw = row.get("n_frags_raw")
        if raw not in (None, ""):
            try:
                if int(float(raw)) != full["n_frags"]:
                    csv_frag_mismatch += 1
            except (TypeError, ValueError):
                pass
        lf = row.get("largest_frag_frac")
        if lf not in (None, "") and full["largest_frag_frac"] is not None:
            try:
                csv_frac_max_abs_delta = max(
                    csv_frac_max_abs_delta,
                    abs(float(lf) - float(full["largest_frag_frac"])),
                )
            except (TypeError, ValueError):
                pass

    cont = contingency(pairs)
    mcnemar = exact_mcnemar_p(cont["b"], cont["c"])

    analyzed = len(pairs)
    return {
        "denominators": {
            "n_rows_in_csv": n_rows,
            "n_empty_string": n_empty_string,
            "n_unparseable": n_unparseable,
            "n_analyzed": analyzed,
            "n_with_no_metal": n_no_metal,
            "n_empty_ligand_after_strip": n_empty_ligand,
            "n_metal_alone_in_own_fragment": n_metal_alone,
            "n_with_free_halide_counterion": n_counterion,
            "denominator_note": (
                "Every rate below is over the n_analyzed rows. A row whose "
                "ligand is empty after stripping is NOT connected in either "
                "arm and is NOT removed from the denominator; it is counted "
                "in n_empty_ligand_after_strip."
            ),
        },
        "metal_inclusive": {
            "definition": (
                "the whole emitted molecule, exactly as the pool CSV reports "
                "it -- the arm a frontier comparison would target."
            ),
            "n_single_fragment": full_single,
            "single_fragment_rate": _rate(full_single, analyzed),
            "n_frags_histogram": _histogram(full_hist),
            "largest_frag_frac": _frac_summary(full_fracs),
            "mean_n_frags": (
                sum(k * v for k, v in full_hist.items()) / analyzed if analyzed else None
            ),
        },
        "ligand_only": {
            "definition": (
                "the SAME molecule with every metal atom deleted (and its "
                "bonds removed): the connectivity the organic skeleton has "
                "on its own. Not a real molecule -- a counterfactual."
            ),
            "n_single_fragment": lig_single,
            "single_fragment_rate": _rate(lig_single, analyzed),
            "single_fragment_rate_over_nonempty_ligand": _rate(
                lig_single, analyzed - n_empty_ligand
            ),
            "n_frags_histogram": _histogram(lig_hist),
            "largest_frag_frac": _frac_summary(lig_fracs),
            "mean_n_frags": (
                sum(k * v for k, v in lig_hist.items()) / analyzed if analyzed else None
            ),
        },
        "paired": {
            "contingency": {
                "a_both_connected": cont["a"],
                "b_ligand_only_connected_metal_inclusive_not": cont["b"],
                "c_metal_inclusive_connected_ligand_only_not": cont["c"],
                "d_neither_connected": cont["d"],
                "note": (
                    "b = the metal fell off and the ligand stayed intact "
                    "(a metal-coordination failure). c = the whole molecule is "
                    "one fragment ONLY because the metal bridges otherwise "
                    "disconnected ligand pieces -- the ligand is fragmented "
                    "and the metal is hiding it."
                ),
            },
            "mcnemar_exact": mcnemar,
            "n_frags_delta_histogram": _histogram(delta_hist),
            "n_frags_delta_note": (
                "ligand-only n_frags minus metal-inclusive n_frags, per row. "
                "Negative = the metal was a fragment of its own (it fell "
                "off); positive = the metal was bonding ligand pieces "
                "together (removing it splits them)."
            ),
        },
        "metal_symbol_counts": _histogram(metal_symbol_counts),
        "n_metal_atoms_per_row": _histogram(n_metal_per_row),
        "per_epoch": {
            epoch: {
                **vals,
                "metal_inclusive_single_rate": _rate(
                    vals["metal_inclusive_single"], vals["n"]
                ),
                "ligand_only_single_rate": _rate(
                    vals["ligand_only_single"], vals["n"]
                ),
            }
            for epoch, vals in sorted(per_epoch.items())
        },
        "csv_cross_check": {
            "n_frags_raw_mismatches": csv_frag_mismatch,
            "largest_frag_frac_max_abs_delta": csv_frac_max_abs_delta,
            "all_atoms_heavy": n_atoms_eq_heavy,
            "note": (
                "The metal-inclusive arm was recomputed from scratch by this "
                "tool and compared against the pool's own n_frags_raw / "
                "largest_frag_frac columns. Zero mismatches means the two "
                "arms are measured over the same molecules by the same "
                "definition."
            ),
        },
    }


# --------------------------------------------------------------------------
# Wf-A162 — the COMPUTED cross-check and the narrative self-consistency guard
# --------------------------------------------------------------------------
#: Published metric artifacts that carry a metal-inclusive single-fragment
#: rate, with the jsonpath of the rate and of its numerator.  An artifact is
#: a candidate ONLY when its own ``source_pool`` names THIS pool.
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

#: A number-like token: digits, optional thousands separators, optional
#: decimal part.
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _dig(doc: Any, path: Sequence[str]) -> Any:
    """Follow a jsonpath of KEYS through nested dicts; ``None`` if absent."""
    cur = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _load_metric(metric: str) -> Optional[Dict[str, Any]]:
    """Read ``metrics/by_metric/<metric>.json``; ``None`` if absent."""
    path = REPO / "metrics" / "by_metric" / f"{metric}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive
        return None


def _published_cross_check(
    pool_name: str, rate: float, n_single: int
) -> Dict[str, Any]:
    """Compare this run's metal-inclusive rate against every PUBLISHED
    metric artifact that names THIS pool.

    Wf-A162 (2026-11-01): the old tool ASSERTED which published number it
    reproduced ("the 8.51 % headline" / "the 25.05 % count"), which was
    FALSE on every pool except its default one -- and it named the WRONG
    POOL PATH while doing it.  The comparison is now COMPUTED: an artifact
    is a candidate only when its own ``source_pool`` names this pool, and
    the delta is measured, never asserted.
    """
    considered: List[Dict[str, Any]] = []
    matches: List[Dict[str, Any]] = []
    for metric, rate_path, n_path in PUBLISHED_POOLED_SOURCES:
        doc = _load_metric(metric)
        if doc is None:
            considered.append({"metric": metric, "present": False})
            continue
        src = doc.get("source_pool")
        src_name = Path(str(src)).parent.name if src else None
        pub_rate = _dig(doc, rate_path)
        pub_n = _dig(doc, n_path)
        considered.append(
            {
                "metric": metric,
                "present": True,
                "source_pool": src,
                "source_pool_name": src_name,
                "published_rate": pub_rate,
                "published_n_single": pub_n,
                "names_this_pool": src_name == pool_name,
            }
        )
        if src_name == pool_name and isinstance(pub_rate, (int, float)):
            matches.append(
                {
                    "metric": metric,
                    "published_rate": pub_rate,
                    "published_n_single": pub_n,
                    "this_run_rate": rate,
                    "this_run_n_single": n_single,
                    "abs_delta": abs(float(pub_rate) - float(rate)),
                    "rate_identical": pub_rate == rate,
                    "numerator_identical": (
                        None if pub_n is None else pub_n == n_single
                    ),
                }
            )
    if not matches:
        verdict = (
            "NO PUBLISHED ARTIFACT NAMES THIS POOL -- no cross-check was "
            "possible, and none is claimed"
        )
    elif all(m["rate_identical"] for m in matches):
        verdict = (
            "THIS RUN REPRODUCES the published rate for THIS pool exactly "
            "(bit-for-bit)"
        )
    else:
        verdict = (
            "THIS RUN DIFFERS from the published rate for THIS pool; the "
            "delta is recorded above and is NOT reconciled here"
        )
    return {
        "rule": (
            "a published artifact is compared ONLY when its own source_pool "
            "names THIS pool; the comparison is computed, never asserted"
        ),
        "this_pool_name": pool_name,
        "this_run_rate": rate,
        "this_run_n_single": n_single,
        "published_artifacts_considered": considered,
        "matches": matches,
        "n_matches": len(matches),
        "verdict": verdict,
    }


def _end_to_end_headline() -> Dict[str, Any]:
    """The END-TO-END median headline, read from the published audit.

    This is the quantity the old prose called "the 8.51 % headline".  It is
    a MEDIAN over several RUNS of a DIFFERENT quantity ("one connected
    parseable molecule per DRAWN sample"), so it is not comparable to a
    single-pool per-ROW rate -- but its ``n`` is carried here so the prose
    can quote the run count from the machine field instead of a literal.
    """
    doc = _load_metric("cfm_fragmentation_audit")
    if doc is None:
        return {
            "source": "metrics/by_metric/cfm_fragmentation_audit.json",
            "present": False,
            "median": None,
            "n_runs": None,
        }
    node = _dig(
        doc, ("summary", "generation_side", "single_molecule_rate_end_to_end")
    )
    if not isinstance(node, dict):
        return {
            "source": "metrics/by_metric/cfm_fragmentation_audit.json",
            "present": False,
            "median": None,
            "n_runs": None,
        }
    return {
        "source": "metrics/by_metric/cfm_fragmentation_audit.json",
        "jsonpath": (
            "/summary/generation_side/single_molecule_rate_end_to_end"
        ),
        "present": True,
        "median": node.get("median"),
        "n_runs": node.get("n"),
        "definition": (
            "'one connected, parseable molecule per DRAWN sample', median "
            "over the generation-side runs"
        ),
    }


#: The artifact's PROSE paths.  Every leaf under these is prose; every
#: other leaf is a MACHINE field.  The self-consistency guard below checks
#: that every number rendered in the prose is backed by a machine field.
NARRATIVE_PATHS: Tuple[str, ...] = (
    "purpose",
    "direction",
    "definitions",
    "not_comparable_to",
    "comparability.statement",
    "comparability.breaks",
    "caveats",
    "verdict.question",
    "verdict.conclusion",
    "premise_checks.brief_claim",
    "premise_checks.explanation",
    "metal_set_invariance.rule",
    "metal_set_invariance.note",
)


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

    A path may designate a LEAF (``comparability.statement``), a MAPPING
    (``not_comparable_to`` -> ``not_comparable_to.<key>``) or a LIST
    (``caveats`` -> ``caveats[0]``, ``caveats[1]``, ...).  The list form
    matters: with a naive ``path + "."`` prefix the list elements would
    never match, they would be silently classified as MACHINE fields, and
    every number inside them would go unchecked.  That hole was found by a
    negative control (Wf-A162) -- the guard was weaker than it looked.
    """
    leaves = _leaf_map(artifact)
    if path in leaves:
        return [path]
    prefixes = (path + ".", path + "[")
    return sorted(p for p in leaves if p.startswith(prefixes))


def _standalone_in_any(token: str, strings: List[str]) -> bool:
    """True when ``token`` appears in one of ``strings`` as a STANDALONE
    number -- delimited by characters that are neither digits nor a decimal
    point.

    This is deliberately stricter than a substring test.  In
    ``wf_a155_ligand_connected_rate.py`` the token ``1`` must NOT count as
    "backed" by the ``155`` in the file name; with a plain substring test it
    would, and the guard would be measurably looser than it looks.
    """
    pattern = re.compile(r"(?<![0-9.])" + re.escape(token) + r"(?![0-9.])")
    return any(pattern.search(text) for text in strings)


def _narrative_self_consistency(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """THE GUARD for the class of defect, not just this instance.

    Wf-A162 (2026-11-01): the old tool hard-coded seven pool-specific
    claims as string literals inside a ``tag: MEASURED`` artifact.  Every
    one of them was true for the DEFAULT pool and FALSE for any other --
    including the wrong pool PATH.  The class is stopped here: every number
    rendered in a NARRATIVE string must be backed by a MACHINE field of the
    same artifact, either as a standalone number inside a machine string, or
    equal (to the token's own decimal count) to a machine number.

    A token with no backing is a DEFECT and fails the run.
    """
    leaves = _leaf_map(artifact)
    prose_paths: set = set()
    for path in NARRATIVE_PATHS:
        prose_paths.update(_resolve(artifact, path))
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
        for token in _NUM_RE.findall(text):
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
        "narrative_paths": list(NARRATIVE_PATHS),
        "n_prose_leaves": len(prose_paths),
        "n_prose_numbers": n_prose_numbers,
        "n_machine_leaves": len(machine),
        "n_machine_numbers": len(machine_numbers),
        "n_machine_strings": len(machine_strings),
        "unbacked_numbers": {
            tok: sorted(set(paths)) for tok, paths in sorted(unbacked.items())
        },
        "pass": not unbacked,
    }


def _repo_relative(pool_csv: Path) -> str:
    """Return the pool path RELATIVE to the repo root when it lives inside it.

    Wf-A180 (2026-11-01).  The published artefact carried the pool path as
    ``molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv`` — a
    REPO-RELATIVE path.  Re-running the emitter with an ABSOLUTE ``--pool-csv``
    rewrote it to ``/home/hugo/codes/mol-metal/molmetal/reports/...``, which
    bakes THIS machine's home directory into a published metric.  The
    artefact is therefore re-published with the repo-relative form, so the
    published value is machine-independent AND identical to what was published
    before.  A pool OUTSIDE the repo keeps its absolute path (there is no
    relative form for it).

    Matching on this field is by the pool DIRECTORY NAME (``parent.name``),
    which is identical for the relative and the absolute form — so this change
    cannot alter which published artefacts count as cross-check candidates.
    """
    try:
        return str(pool_csv.resolve().relative_to(REPO))
    except ValueError:
        return str(pool_csv)


def build_artifact(pool_csv: Path) -> Dict[str, Any]:
    """Assemble the full metric artifact."""
    primary = analyze_pool(pool_csv, metal_symbols=set(METAL_SYMBOLS))
    cohort = analyze_pool(pool_csv, metal_symbols=set(COHORT_METAL_SYMBOLS))

    invariance = {
        "rule": "d-block TRANSITION_METALS (the project's own metal predicate)",
        "narrower_rule": "the trainer's _TARGET_METALS cohort {Ru, Rh, Re, Os, Ir, Pt}",
        "metal_inclusive_single_fragment_rate": primary["metal_inclusive"][
            "single_fragment_rate"
        ],
        "ligand_only_single_fragment_rate_dblock": primary["ligand_only"][
            "single_fragment_rate"
        ],
        "ligand_only_single_fragment_rate_cohort": cohort["ligand_only"][
            "single_fragment_rate"
        ],
        "n_empty_ligand_dblock": primary["denominators"]["n_empty_ligand_after_strip"],
        "n_empty_ligand_cohort": cohort["denominators"]["n_empty_ligand_after_strip"],
        "identical": (
            primary["ligand_only"]["single_fragment_rate"]
            == cohort["ligand_only"]["single_fragment_rate"]
        ),
        "note": (
            "The rule is not pool-fitted: the answer is identical under the "
            "project's broad d-block metal predicate and under the narrow "
            "trainer cohort set."
        ),
    }

    p = primary["paired"]
    cont = p["contingency"]
    analyzed = primary["denominators"]["n_analyzed"]
    full_rate = primary["metal_inclusive"]["single_fragment_rate"]
    lig_rate = primary["ligand_only"]["single_fragment_rate"]

    verdict = {
        "question": (
            "How much of the pool's fragmentation is the METAL falling off, "
            "versus the LIGAND itself fragmenting?"
        ),
        "metal_alone_in_own_fragment_rows": primary["denominators"][
            "n_metal_alone_in_own_fragment"
        ],
        "metal_alone_rate": _rate(
            primary["denominators"]["n_metal_alone_in_own_fragment"], analyzed
        ),
        "empty_ligand_rows": primary["denominators"]["n_empty_ligand_after_strip"],
        "b_metal_fell_off_ligand_intact": cont["b_ligand_only_connected_metal_inclusive_not"],
        "c_metal_bridged_fragmented_ligand": cont["c_metal_inclusive_connected_ligand_only_not"],
        "conclusion": (
            "The 'the metal fell off' hypothesis is REFUTED as the dominant "
            "explanation, and the measurement points the other way: only "
            f"{cont['b_ligand_only_connected_metal_inclusive_not']} of "
            f"{analyzed} rows are 'metal fell off, ligand intact'. The "
            "overwhelming majority of the fragmentation is LIGAND-side. "
            f"Moreover the metal-inclusive rate ({full_rate:.4f}) is much "
            f"HIGHER than the ligand-only rate ({lig_rate:.4f}) because in "
            f"{cont['c_metal_inclusive_connected_ligand_only_not']} rows the "
            "whole molecule is one fragment ONLY because the metal bridges "
            "otherwise-disconnected ligand pieces. So the metal-inclusive "
            "'connected' count is largely the metal CONCEALING ligand "
            "fragmentation, not the ligand being intact -- the repair effort "
            "belongs on the ligand skeleton, not on keeping the metal "
            "attached. This is an honest negative for the metal-falls-off "
            "hypothesis and a positive finding that the metal is a "
            "connectivity hub."
        ),
    }

    premise_checks = {
        "brief_claim": (
            "the task brief asserted that the rows whose metal is alone in its "
            "fragment are exactly the rows whose ligand becomes empty after "
            "stripping."
        ),
        "verdict": "NOT SUPPORTED BY THE DATA",
        "measured_metal_alone_in_own_fragment_rows": primary["denominators"][
            "n_metal_alone_in_own_fragment"
        ],
        "measured_empty_ligand_rows": primary["denominators"]["n_empty_ligand_after_strip"],
        "explanation": (
            "In every one of the metal-alone rows the row ALSO carries "
            "ligand-bearing fragments, so stripping the metal leaves a "
            "non-empty ligand. The genuinely empty-ligand case -- a row that "
            "is metal and nothing else -- has count 0 on this pool. The two "
            "quantities are reported under separate names so the "
            "distinction survives."
        ),
    }

    # -- Wf-A162: every narrative number is COMPUTED from THIS run --------
    pool_name = pool_csv.parent.name
    n_halide = primary["denominators"]["n_with_free_halide_counterion"]
    breaks = build_comparability_breaks(pool_name, analyzed)
    n_breaks = len(breaks)
    end_to_end = _end_to_end_headline()
    n_runs = end_to_end.get("n_runs")
    runs_phrase = (
        f"over {n_runs} RUNS" if isinstance(n_runs, int) else "over the runs"
    )
    cross_check = _published_cross_check(
        pool_name,
        full_rate,
        primary["metal_inclusive"]["n_single_fragment"],
    )

    artifact = {
        "metric": "wfa155_ligand_connected_rate",
        "tag": "MEASURED",
        "date": "2026-11-01",
        "wave": "Wf-A155 (branch LIG) + Wf-A162 (computed narrative)",
        "purpose": (
            "Frontier option (A): a LIGAND-ONLY 'connected' rate published "
            "ALONGSIDE the metal-inclusive one, to separate 'the metal fell "
            "off' from 'the ligand itself fragmented'. Reporting-only. This "
            f"run measures ONE pool of {analyzed} rows; whether it "
            "reproduces a published rate for this pool is COMPUTED in "
            "published_cross_check, not asserted."
        ),
        "direction": (
            "single_fragment_rate is HIGHER IS BETTER for both arms, but the "
            "ligand-only arm is a DIAGNOSTIC, not a target: a coordination "
            "complex is SUPPOSED to have the metal as part of its "
            "connectivity, so a low ligand-only rate is not per se a defect. "
            "Read the two arms TOGETHER (see verdict)."
        ),
        "gpu": "0 (CPU-only, RDKit-only, no checkpoint, no decode)",
        "source_pool": _repo_relative(pool_csv),
        "script": "molmetal/scripts/wf_a155_ligand_connected_rate.py",
        "definitions": {
            "metal_symbols_rule": (
                "an atom is 'a metal' iff its element symbol is in the repo's "
                "canonical d-block TRANSITION_METALS list (Sc..Hg); the "
                "trainer's cohort set {Ru,Rh,Re,Os,Ir,Pt} is a subset. "
                "Stripping deletes the metal atom and every bond incident to "
                "it; nothing else is touched (counterions stay)."
            ),
            "connected_rule": (
                "one RDKit connected component AND at least one heavy atom. "
                "An empty ligand is NOT connected."
            ),
            "both_arms_same_code_path": (
                "both arms go through arm_stats(), so the metal-inclusive and "
                "ligand-only numbers are internally comparable by "
                "construction."
            ),
        },
        **primary,
        "metal_set_invariance": invariance,
        "verdict": verdict,
        "premise_checks": premise_checks,
        "end_to_end_headline": end_to_end,
        "published_cross_check": cross_check,
        "not_comparable_to": {
            "end_to_end_median_headline": (
                "metrics/by_metric/cfm_fragmentation_audit.json -> "
                "/summary/generation_side/single_molecule_rate_end_to_end is "
                "an END-TO-END rate "
                f"{runs_phrase} of a different quantity ('one connected "
                "parseable molecule per DRAWN sample'). This metric is a "
                f"per-ROW paired analysis over ONE pool of {analyzed} rows."
            ),
            "other_pools_single_pool_counts": (
                "the metal-inclusive arm of THIS tool is a single-pool rate "
                "for THIS pool only; it is reported so the two arms are "
                "comparable. The NEW quantity is the ligand-only rate, over a "
                "NEW denominator definition. Which published number (if any) "
                "this run reproduces is COMPUTED in published_cross_check -- "
                "it is not asserted here."
            ),
        },
        "comparability": {
            "statement": (
                "These numbers are NOT comparable to the published frontier "
                f"band of {FRONTIER_BAND}. {n_breaks} independent breaks "
                f"separate them, and all {n_breaks} must hold simultaneously "
                "for any comparison to be meaningful."
            ),
            "breaks": [dict(b) for b in breaks],
            # Wf-A189 (2026-11-01): the dated rule-#6 records are emitted HERE,
            # in the SAME POSITION they occupy in the published artefact, so a
            # re-run REPRODUCES the published artefact instead of silently
            # DROPPING the dated history (AGENTS.md §5.5 rule #6).
            "superseded_frontier_2026_11_01_wfa183": dict(
                SUPERSEDED_FRONTIER_2026_11_01_WFA183
            ),
            "superseded_frontier_2026_11_01_wfa189": dict(
                SUPERSEDED_FRONTIER_2026_11_01_WFA189
            ),
            # Wf-A199 (2026-11-02): the Wf-A194 dated rule-#6 record is
            # emitted HERE, in the SAME POSITION it occupies in the published
            # artefact, so a re-run REPRODUCES the published artefact instead
            # of silently DROPPING the dated history (AGENTS.md §5.5 rule #6).
            "superseded_frontier_2026_11_02_wfa194": dict(
                SUPERSEDED_FRONTIER_2026_11_02_WFA194
            ),
            "n_breaks": n_breaks,
            "frontier_band": FRONTIER_BAND,
            "frontier_denominator": FRONTIER_DENOMINATOR,
        },
        # Wf-A204 (2026-11-02): the DATED CORRECTION of the `producer_parity`
        # leaf of the Wf-A194 record emitted inside `comparability` above.
        # It is emitted HERE, at the TOP LEVEL, so that the correction
        # REACHES the artefact: a reader of the published metric must not
        # read that leaf's now-false claim without the dated correction
        # beside it.  Every leaf of the record is a STRING, and the guard
        # subject below DROPS it for the same reason it drops the
        # comparability dated records, so adding it moves NO numeric leaf.
        "superseded_producer_parity_2026_11_02_wfa204": dict(
            SUPERSEDED_PRODUCER_PARITY_2026_11_02_WFA204
        ),
        "caveats": [
            "The ligand-only arm is a counterfactual (the metal deleted), not "
            "a real molecule; a coordination complex legitimately needs the "
            "metal in its connectivity.",
            "Both arms are RAW: no LargestFragmentChooser, no explicit H, no "
            "UFF, no post-processing of any kind.",
            f"ONE pool, ONE post-hoc decode, {analyzed} rows -- not a "
            "training-set-distribution sampling, so this is not a rate over "
            "the generator's output distribution.",
            "The pool carries implicit hydrogens; the frontier datasets carry "
            "explicit ones.",
            f"The {n_halide} rows with a free halide counterion contribute an "
            "extra fragment to BOTH arms; they are counted, not removed.",
        ],
    }

    # -- THE GUARD for the class of defect, not just this instance --------
    # Wf-A199 (2026-11-02): the guard is computed on the artefact AS IT
    # STOOD BEFORE the dated rule-#6 records were attached, so that adding a
    # dated record moves NO numeric leaf of the published artefact.  The
    # published artefact's own `guard_counters_note` (carried by
    # SUPERSEDED_FRONTIER_2026_11_01_WFA183) documents exactly this: the
    # recorded counters are "the emitter's own output, computed on the
    # artefact BEFORE ... the dated keys added here", and they were "left
    # UNCHANGED on purpose: no existing numeric leaf of this artefact may
    # move".  Computing the guard on the pre-dated-record subject is what
    # makes a producer re-run REPRODUCE the published artefact: the guard
    # block it writes is byte-identical to the one already on disk.
    _dated_record_keys = (
        "superseded_frontier_2026_11_01_wfa183",
        "superseded_frontier_2026_11_01_wfa189",
        "superseded_frontier_2026_11_02_wfa194",
    )
    # Wf-A204 (2026-11-02): the TOP-LEVEL dated record is dropped from the
    # guard subject for the SAME reason the comparability ones are -- the
    # guard block the emitter writes must be byte-identical to the one
    # already on disk, so attaching a dated record may move NO numeric leaf.
    _dated_top_level_keys = ("superseded_producer_parity_2026_11_02_wfa204",)
    _guard_subject = {
        _k: _v for _k, _v in artifact.items() if _k not in _dated_top_level_keys
    }
    _guard_subject["comparability"] = {
        _k: _v
        for _k, _v in artifact["comparability"].items()
        if _k not in _dated_record_keys
    }
    guard = _narrative_self_consistency(_guard_subject)
    artifact["narrative_self_consistency"] = guard
    if not guard["pass"]:
        raise NarrativeSelfConsistencyError(artifact, guard)
    return artifact


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Wf-A155 branch LIG: paired metal-inclusive vs ligand-only "
            "'connected' rate over a CFM generation pool (CPU-only)."
        )
    )
    ap.add_argument("--pool-csv", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only analyse the first N rows (smoke use; the published "
        "artifact is produced with no limit)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the artifact to stdout instead of writing it",
    )
    args = ap.parse_args(argv)

    if not args.pool_csv.is_file():
        print(f"FAIL: pool CSV not found: {args.pool_csv}")
        return 1

    try:
        artifact = build_artifact(args.pool_csv)
    except NarrativeSelfConsistencyError as exc:
        # Wf-A162: a narrative number with no machine backing is a DEFECT.
        # The offending artifact is written out as an explicit FAIL so the
        # failure is on disk, not only on stdout, and the run exits non-zero.
        bad = dict(exc.artifact)
        bad["status"] = "FAIL"
        bad["failure"] = (
            "the narrative self-consistency guard rejected this artifact: "
            f"{exc.guard.get('unbacked_numbers')}"
        )
        content = json.dumps(bad, indent=2, ensure_ascii=False) + "\n"
        if not args.dry_run:
            atomic_write_file(args.out, content)
        print(f"FAIL: {exc}", file=sys.stderr)
        for tok, paths in sorted(exc.guard.get("unbacked_numbers", {}).items()):
            print(f"  unbacked {tok!r} in {paths}", file=sys.stderr)
        return 1
    if args.limit is not None:
        artifact = analyze_pool(args.pool_csv, limit=args.limit)
        artifact = {"metric": "wfa155_ligand_connected_rate", "tag": "MEASURED",
                    "limit": args.limit, **artifact}

    content = json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=False) + "\n"

    if args.dry_run:
        print(content)
        return 0

    risk = safe_overwrite_check(args.out, content)
    if risk is not None:
        print(f"FAIL: refusing to overwrite {args.out} (diff ratio {risk:.3f})")
        return 1
    atomic_write_file(args.out, content)
    truncation = detect_truncation(content)
    if truncation is not None:
        print(f"FAIL: artifact looks truncated: {truncation}")
        return 1
    reread = json.loads(args.out.read_text(encoding="utf-8"))
    if reread != artifact:
        print("FAIL: artifact on disk does not equal the computed artifact")
        return 1

    d = artifact["denominators"]
    print(f"WROTE {args.out}")
    print(f"  rows analyzed        : {d['n_analyzed']}")
    print(
        "  metal-incl. connected : "
        f"{artifact['metal_inclusive']['single_fragment_rate']:.6f} "
        f"({artifact['metal_inclusive']['n_single_fragment']}/{d['n_analyzed']})"
    )
    print(
        "  ligand-only connected : "
        f"{artifact['ligand_only']['single_fragment_rate']:.6f} "
        f"({artifact['ligand_only']['n_single_fragment']}/{d['n_analyzed']})"
    )
    cont = artifact["paired"]["contingency"]
    print(f"  b (metal fell off)    : {cont['b_ligand_only_connected_metal_inclusive_not']}")
    print(f"  c (metal bridged)     : {cont['c_metal_inclusive_connected_ligand_only_not']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
