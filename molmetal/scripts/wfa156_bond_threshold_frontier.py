#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A156 -- the EF-TALFM covalent-radius bond threshold applied to OUR CFM
decode pool, as a CPU-only MEASUREMENT.

WHAT THIS TOOL IS
-----------------
The branch brief asked for ONE frontier mechanism to be applied to our decoder's
suspected bond-candidacy rule, live and as a measurement:

    EF-TALFM (arXiv 2609.08333) candidate bond rule, Algorithm 4 line 6.
    QUOTED VERBATIM from the arXiv LaTeX source, fetched LIVE on 2026-11-01:

        src/appendix.tex:1043
            \\STATE Compute bond threshold $\\tau_{ij} \\gets \\lambda_\\pi
                   \\big(r^{\\mathrm{cov}}(\\hat z_i) + r^{\\mathrm{cov}}(\\hat z_j)\\big)$

        src/appendix.tex:1039
            \\STATE Select reconstruction policy $\\pi \\gets
                   \\texttt{ChoosePolicy}(\\hat{\\mathbf z}, \\hat{\\mathbf R}, \\hat Q)$

    so the paper's rule is

        tau_ij = lambda_POLICY * ( r_cov(z_i) + r_cov(z_j) )

    where ``lambda_POLICY`` is the tolerance factor OF THE SELECTED POLICY
    ``pi``.  ``pi`` is a SUBSCRIPT on lambda, NOT the constant pi.

    The branch brief (and the repo's own Wf-A155 implementation in
    ``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py::radii_scaled_factor``)
    read it as ``lambda * pi * (...)``.  That is a MISREADING of the subscript,
    and it is a 3.14159x difference in the threshold.  This tool therefore
    reports BOTH conventions side by side, under the names ``paper`` and
    ``repo_pi``, and never silently picks one.

    MEASURED (live fetch, 2026-11-01): the value of ``lambda_POLICY`` is NOT
    recoverable from the paper.  The whole LaTeX source contains exactly TWO
    occurrences of the word "policy" (both inside Algorithm 4) and NO numeric
    value for ``lambda_pi`` anywhere.  So every lambda below is a SWEEP, never
    "the paper's value".

WHAT IS MEASURED, AND WHAT IS NOT
---------------------------------
The DEFAULT pool artefact is::

    molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv

(``--pool`` points the tool at ANY other pool.  Every number that describes
OUR population -- the row count, the pool path, the fragment/metal counts --
is COMPUTED from the pool this run actually read; see the self-consistency
guard in the "GUARD" section below.)

with the header ``epoch,smiles,n_frags_raw,largest_frag_frac`` -- written by
``molmetal/scripts/train_cfm_yuelbond_wasserstein.py`` at line 19250.  It
carries NO coordinates.  The pool's own run directory is LISTED from disk at
run time (the default pool's holds the checkpoints, ``events.jsonl``, the pool
CSV and ``master.log``; another pool's may hold less) -- no per-molecule
geometry dump.  CONSEQUENCE: the decoder's own interatomic distance for each
emitted bond is NOT RECOVERABLE, so the literal question "would the bond the
decoder EMITTED have survived tau_ij in the decoder's OWN geometry" is NOT
MEASURABLE from what is on disk.  This tool says so in its own output rather
than substituting a silent proxy.

Three measurement tiers are reported instead, each labelled:

  T1 EXACT, geometry-free: compare tau_ij against the decoder's OWN flat
     geometric gate (3.5 A, MEASURED from the run's ``master.log``) for every
     emitted bond.  This answers "are our bonds merely MORE or actually
     DIFFERENT from covalent-radius geometry" without any geometry at all.
  T2 EXACT, geometry-free bounds: every emitted bond is either GUARANTEED to
     survive (tau_ij >= the decoder's 3.5 A gate, so the bond's distance is
     <= 3.5 A <= tau_ij), GUARANTEED to fail (tau_ij < the shortest bond of
     that element pair ever observed in the tmQM DFT corpus), or UNDECIDABLE.
  T3 LENGTH-TRANSFERRED estimate: the survival fraction if each emitted bond
     had the tmQM-DFT EMPIRICAL MEAN length for its (Z, Z, order) bin.  This is
     an ESTIMATE for the pool (the decoder's geometry is unknown), and it is
     labelled as such wherever it appears.

The BASELINE population is the tmQM ground truth itself (real DFT geometries,
real Wiberg bond orders, real interatomic distances), restricted to the same
six metals the decode cohort used.  There the survival fractions are EXACT:
no transfer, no estimate.

THE COMPARABILITY CAVEAT IS WRITTEN INTO THE REPORT FILES, not only into a
chat answer.  See :data:`COMPARABILITY_CAVEAT_TEMPLATE` and
:func:`build_comparability_caveat` below.

CONSTRAINTS HONOURED
--------------------
* CPU-only.  No torch import.  No GPU call of any kind, and no GPU probe.
* Writes ONLY under ``molmetal/reports/wfa156_bond_threshold_frontier/``.
  Never ``metrics/by_metric/`` (published tree).
* Never reads ``paper/``.

THE GUARD (Wf-A164)
-------------------
Wf-A162 found that a measurement tool can hard-code POOL-SPECIFIC claims as
string literals inside a ``MEASURED`` artefact: the literals are TRUE for the
tool's DEFAULT pool and FALSE for every other pool, and the tool then emits
those false statements into a fresh artefact.  THIS tool had the same defect
(the population cross-check note carried four counts and a denominator as
literals; the comparability caveat carried the row count, the pool path and
the metal ratio as literals).

The class is stopped here, not just the instance:

1. every number describing OUR population is COMPUTED from the run;
2. every narrative string is checked against the artefact's OWN machine
   fields -- a number with no machine backing is a DEFECT and FAILS the run
   (``status: FAIL``, exit 1), and
3. the artefact may name exactly ONE pool path: the one this run read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rdkit import Chem, RDLogger  # noqa: E402

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Constants -- every one of them traced to a file or a live fetch
# ---------------------------------------------------------------------------

#: The lambda sweep the branch brief asked for.  These are SWEEP VALUES.
#: The paper's own ``lambda_POLICY`` is not recoverable (see the docstring).
LAMBDAS: Tuple[float, ...] = (0.8, 1.0, 1.2, 1.4, 1.6)

#: Threshold-factor conventions.
#:
#: ``paper``   -- tau_ij = lambda * (r_cov_i + r_cov_j)          [the paper]
#: ``repo_pi`` -- tau_ij = lambda * pi * (r_cov_i + r_cov_j)     [branch brief
#:               and the repo's Wf-A155 ``radii_scaled_factor``]
CONVENTIONS: Tuple[str, ...] = ("paper", "repo_pi")

#: The decoder's OWN flat geometric candidate cutoff, in Angstrom.
#:
#: PREMISE, and its exact status:
#: * ``YuelBondDecoderHead.cutoff`` DEFAULTS to 3.5 A
#:   (``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py:799``, stored at
#:   :1014) and is applied as ``mask = d_raw < self.cutoff`` at :1244 and
#:   :1934 -- documented there as a DECLARED DEVIATION from the published
#:   YuelBond 3 A.
#: * ``CFMYuelBondDecoder.distance_cutoff`` also defaults to 3.5 A
#:   (``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:1048``,
#:   assigned at :1172).
#: * The run's ``master.log`` DOES print ``cutoff=3.5 A`` -- but that line is
#:   the EGN VELOCITY FIELD's cutoff, not the bond head's.  The log does NOT
#:   print the decoder's candidate cutoff, so 3.5 A is the CODE DEFAULT, not
#:   an independently observed value for this specific run.
#:
#: CONSEQUENCE: every T1/T2 number below is CONDITIONAL on this run having used
#: the 3.5 A default.  It is the value the code path takes unless the launch
#: overrode it, and no override is recorded in the run's artefacts.
DECODER_FLAT_CUTOFF_A: float = 3.5

#: The decoder's own ACCEPTANCE rule for a candidate pair.  MEASURED from the
#: same master.log: "[cfm-decoder] decision=threshold @ threshold=0.500".
#: The decoder emits a bond iff p(bonded) >= 0.5 on the learned head, subject
#: to valence caps; the geometric gate above is the CANDIDACY half.
DECODER_ACCEPT_THRESHOLD: float = 0.5

#: Dative-bond distance gate.  MEASURED from the same master.log:
#: "dative_max_distance=2.60 A".
DECODER_DATIVE_MAX_DISTANCE_A: float = 2.60

#: The pool this branch is about.
DEFAULT_POOL = REPO / "molmetal" / "reports" / "cfm_c8_fullgen_2026_10_24" / "generated_smiles.csv"

#: The tmQM corpus (real DFT geometries + Wiberg bond orders).
DEFAULT_TMQM_ROOT = REPO / ".data" / "molmetal" / "tmQM"

#: The six metals the decode cohort used.  MEASURED from the run's
#: master.log: "[tmqm-clouds] metals=['Ir', 'Os', 'Pt', 'Re', 'Rh', 'Ru']".
DECODE_COHORT_METALS: Tuple[str, ...] = ("Ir", "Os", "Pt", "Re", "Rh", "Ru")

#: The date the frontier fetches below were performed.  A machine field, so
#: the date rendered in the prose is backed by the artefact that records the
#: fetches.
LIVE_FETCH_DATE: str = "2026-11-01"

#: The atomic numbers of the decode cohort's metals, DERIVED from
#: :data:`DECODE_COHORT_METALS` through RDKit's periodic table rather than
#: transcribed.  The tool used to carry the literal set {44, 45, 75, 76, 77,
#: 78} in two separate places, which could silently drift from the cohort it
#: claims to describe.  MEASURED: {Ir: 77, Os: 76, Pt: 78, Re: 75, Rh: 45,
#: Ru: 44}.
COHORT_METAL_Z: frozenset = frozenset(
    Chem.GetPeriodicTable().GetAtomicNumber(sym) for sym in DECODE_COHORT_METALS
)

#: Default number of tmQM complexes in the baseline population.  Deterministic
#: selection (evenly spaced over the sorted csd_code list), so the baseline is
#: reproducible bit-for-bit.
DEFAULT_BASELINE_COMPLEXES: int = 2000

#: The exact live-fetch log.  Filled in by hand from the fetches performed on
#: 2026-11-01; every entry records the URL, the HTTP outcome and what came
#: back.  NOTHING here is quoted from memory.
LIVE_FETCHES: List[Dict[str, Any]] = [
    {
        "url": "http://export.arxiv.org/abs/2609.08333",
        "how": "WebFetch",
        "resolved": True,
        "what": (
            "arXiv abstract page. Returned title 'Fixed-Dimensional Latent Flow for "
            "Generating Variable-Size 3D Molecules', authors Weichi Yao, Cameron "
            "Gruich, Bryan R. Goldsmith, Yixin Wang. Abstract-only: NO appendix, "
            "NO bond-threshold formula, NO 'Connected'/'Sanitized' percentages."
        ),
    },
    {
        "url": "https://arxiv.org/abs/2609.08333v1",
        "how": "WebFetch",
        "resolved": True,
        "what": (
            "Abstract/listing page only -- no appendix text, no tau_ij formula. "
            "The summariser additionally emitted a '89.4% vs 75.6% UAE-3D / 69.8% "
            "FlowMol' passage that does NOT appear in this paper's HTML and is "
            "therefore treated as SUMMARISER NOISE, not as a quote. Discarded; "
            "every number in this report comes from the raw HTML/LaTeX fetches "
            "below, not from this page."
        ),
    },
    {
        "url": "https://arxiv.org/html/2609.08333v1",
        "how": "curl (HTTP 200, 1,234,045 bytes, saved to /tmp/wfa156_arxiv_html_2609.08333v1.html)",
        "resolved": True,
        "what": (
            "Full LaTeX-rendered HTML. Confirmed IN TEXT: 'We propose candidate "
            "heavy-heavy bonds between heavy atom pairs whose Euclidean distance "
            "falls below a scaled sum of covalent radii for the pair.' Confirmed: "
            "'Connectivity: % of molecules with exactly one connected component, "
            "i.e., one fragment.' Confirmed: 'All rates are computed over 10,000 "
            "generated molecules.' Confirmed: 'For fragmented outputs, SMILES-based "
            "and PoseBusters metrics are computed on the atom-count largest "
            "connected component after RDKit sanitization.' Confirmed: 'Relative to "
            "FlowMol, EF-TALFM nearly matches the sanitization rate (98.25% versus "
            "98.72%) ... while its connectivity is moderately lower (92.98% versus "
            "98.72%).' Confirmed: the PCQM4Mv2 run uses the ALL-ATOM representation, "
            "whose connectivity comes from Open Babel perception, while the "
            "heavy-atom branch (Algorithm 4) is where the tau_ij rule lives."
        ),
    },
    {
        "url": "https://arxiv.org/e-print/2609.08333",
        "how": "curl (HTTP 200, 8,508,562 bytes gzip, extracted to /tmp/wfa156_src/)",
        "resolved": True,
        "what": (
            "LaTeX source. THE AUTHORITATIVE QUOTE: src/appendix.tex:1043 -- "
            "\\STATE Compute bond threshold $\\tau_{ij} \\gets \\lambda_\\pi "
            "\\big(r^{\\mathrm{cov}}(\\hat z_i) + r^{\\mathrm{cov}}(\\hat z_j)\\big)$ ; "
            "src/appendix.tex:1039 -- \\STATE Select reconstruction policy $\\pi \\gets "
            "\\texttt{ChoosePolicy}(...). The word 'policy' occurs EXACTLY TWICE in "
            "the whole source (both in Algorithm 4) and NO numeric value for "
            "\\lambda_\\pi appears anywhere. Confirmed Table 5: C. 92.98, S. 98.25 "
            "for EF-TALFM over 10,000 molecules."
        ),
    },
    {
        "url": "https://arxiv.org/abs/2508.12629",
        "how": "curl (HTTP 200, 42,489 bytes)",
        "resolved": True,
        "what": (
            "FlowMol3: 'FlowMol3: Flow Matching for 3D DeNovo Small-Molecule "
            "Generation', Dunn, Ian; Koes, David R., dated 2025/08/18. Abstract "
            "states 'nearly 100% molecular validity for drug-like molecules with "
            "explicit hydrogens'. NO journal_ref and NO 'Digital Discovery 2026 "
            "5(5):2052-2066' metadata on the arXiv record, and the '% Valid 99.9+-0.1 "
            "/ % PB-Valid 91.9+-0.7' figures were NOT verified from this page."
        ),
    },
    {
        "url": "https://arxiv.org/abs/2411.05676",
        "how": "curl (HTTP 200, 42,606 bytes)",
        "resolved": True,
        "what": (
            "GGFlow ORIGINAL: 'Improving Molecular Graph Generation with Flow "
            "Matching and Optimal Transport', Hou, Zhu, Ren, Bu, Gao, Zhang, Sun, "
            "dated 2024/11/08. A DISCRETE 2D molecular-GRAPH flow model -- no 3D "
            "geometry, hence no covalent-radius bond threshold of any kind. The "
            "'retitled TMLR 2026 / OpenReview K8RlXtMgzo' identity in the branch "
            "brief was NOT verified from this page."
        ),
    },
    {
        "url": "https://arxiv.org/abs/2509.01486",
        "how": "curl (HTTP 200, 41,794 bytes)",
        "resolved": True,
        "what": (
            "PAFlow: 'Prior-Guided Flow Matching for Target-Aware Molecule Design "
            "with Learnable Atom Number', Zhou, Qian, Tu, Xu, dated 2025/09/01. "
            "SBDD model on CrossDocked2020, Avg. Vina Score up to -8.31. NOT the "
            "paper the brief calls 'arXiv 2509.01486 NeurIPS 2025' in a way that "
            "changes any number here -- it contributes NO bond-threshold rule."
        ),
    },
]

#: The frontier band's RESTATEMENT as it was handed to this branch.  These
#: are the FRONTIER's numbers, not ours, and they are carried as a MACHINE
#: field so that the prose quoting them is backed by it (Wf-A162 pattern).
#:
#: PROVENANCE, MEASURED: the string "92.7" occurs ZERO times in the fetched
#: arXiv 2609.08333v1 HTML (1,234,045 bytes, fetched 2026-11-01).  It is the
#: branch's restatement of the band, NOT the paper's own wording; the paper's
#: own numbers are in :data:`FRONTIER_BAND_QUOTE`.
FRONTIER_BAND: str = "92.7-94 %"

#: The paper's OWN numbers for the same family of rates, quoted from the
#: fetched source.  MEASURED live 2026-11-01 (see :data:`LIVE_FETCHES`).
FRONTIER_BAND_QUOTE: str = (
    '"Connected 92.98 %, Sanitized 98.25 %, Unique+sanitized 98.21 %, '
    'Novel+unique+sanitized 94.33 %" -- MEASURED in the fetched '
    'arXiv 2609.08333v1 HTML (HTTP 200, 1,234,045 bytes, fetched '
    '2026-11-01).  The prose reads "Relative to FlowMol, EF-TALFM nearly '
    'matches the sanitization rate (98.25% versus 98.72%) and unique '
    'sanitized yield (98.21% versus 98.27%), while its connectivity is '
    'moderately lower (92.98% versus 98.72%)"; the Table 5 row labels '
    'S. / U. & S. / N. & U. & S. carry 98.25 / 98.21 / 94.33.'
)

#: Where each quoted fragment of the frontier comes from.  MACHINE fields, so
#: the section/table numbers rendered in the prose are backed rather than
#: free-floating.
FRONTIER_CITATIONS: Dict[str, str] = {
    "app_g3": 'App. G.3: "We sample 10,000 molecules"',
    "table5": (
        'Table 5 caption: "All rates are computed over 10,000 generated '
        'molecules"'
    ),
    "algorithm4": (
        "Algorithm 4 -- the heavy-atom branch that contains the tau_ij rule "
        "(src/appendix.tex:1039 and :1043)"
    ),
}

#: The comparability caveat.  Written into BOTH report files.
#:
#: Wf-A164 (2026-11-01): every number that describes OUR population is a
#: PLACEHOLDER, filled from the run.  It used to be a module constant
#: carrying the DEFAULT pool's row count, the DEFAULT pool's path and the
#: DEFAULT pool's metal ratio -- TRUE for the default pool and FALSE for every
#: other pool, inside an artefact that reports itself as a measurement.
#: (The literals themselves are deliberately NOT quoted here: a comment that
#: transcribes them is exactly what the Wf-A163 tool-layer scan is built to
#: flag, and a clean scan result is worth more than the quote.)
#:
#: Wf-A180 (2026-11-01) -- the RAW-vs-post-processed correction.  The template
#: asserted, AS A SETTLED FACT, that EF-TALFM's connectivity cell is "computed
#: AFTER retaining the atom-count largest connected component and after RDKit
#: sanitization".  That is REFUTED ON DISK, twice:
#:   * molmetal/reports/wfa168_eftalfm_axis/report.md -- "connectivity is RAW;
#:     the caption's sentence is ambiguous, not the number."
#:   * molmetal/reports/wfa171_frontier_postproc/report.md -- "92.98 % is a RAW
#:     single-component rate", counted BEFORE the largest-component retention.
#: The NUMBER is RAW; only the paper's CAPTION sentence is ambiguous as to the
#: order.  The superseded rendering is preserved VERBATIM, byte-identical, in
#: :data:`COMPARABILITY_CAVEAT_TEMPLATE_SUPERSEDED_2026_11_01` below and in
#: this dated source comment; every artefact this tool writes carries it under
#: the dated key "comparability_caveat_superseded_2026_11_01" and the
#: correction under "comparability_caveat_corrected_wfa180" (AGENTS.md §5.5
#: rule #6 -- append-only, dated, never a silent edit).
#:
#: The superseded block (2) METRIC paragraph, VERBATIM:
#:
#:     (2) METRIC.  Their "Connectivity" is "% of molecules with exactly one
#:     connected component, i.e., one fragment", computed AFTER retaining the
#:     atom-count largest connected component and after RDKit sanitization.
#:     Ours is an RDKit fragment count over RAW rows.
SUPERSEDED_METRIC_SENTENCE = """\
(2) METRIC.  Their "Connectivity" is "% of molecules with exactly one connected
    component, i.e., one fragment", computed AFTER retaining the atom-count
    largest connected component and after RDKit sanitization.  Ours is an RDKit
    fragment count over RAW rows."""

#: Wf-A180 (2026-11-01): the CORRECTED block (2) METRIC paragraph.
CORRECTED_METRIC_SENTENCE = """\
(2) METRIC.  Their "Connectivity" is "% of molecules with exactly one connected
    component, i.e., one fragment".  THAT CELL IS RAW: it is a single-component
    rate counted BEFORE the atom-count largest-component retention.  The paper's
    CAPTION sentence is AMBIGUOUS as to the order -- it grammatically attaches
    the largest-component clause to connectivity too -- but the NUMBER is RAW:
    a post-retention connectivity count would read 100 % by construction, and
    the paper reports 92.98 %.  Ours is an RDKit fragment count over RAW rows."""

#: Wf-A180 (2026-11-01): the canonical reading-safe band statement, used
#: VERBATIM in the caveat and in the dated correction record.  Its two
#: placeholders are filled from the run; the numeral the brief's canonical
#: wording carries ("OUR 8.51 %") is deliberately NOT used, because it is the
#: c7-pool value and this tool also runs against other pools -- the invariant
#: it states ("OUR number IS a connected-component count on the RAW emitted
#: SMILES") is pool-neutral, and the parenthetical carries this pool's own
#: MEASURED counts instead.
#:
#: Wf-A196 (2026-11-02) -- the X5 reading-COUNT scope repair.  The Wf-A180
#: rendering above asserted, AS A SETTLED COUNT, that the record "holds TWO
#: competing readings".  That count is NOT settled on disk: the repo's
#: primary-source verification
#: (molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json
#: -> `invariants`.`competing_readings`) records a FURTHER reading that also
#: does not reproduce the quoted numerals, so two on-disk records disagree
#: about the SIZE of the contest.  Neither "TWO" nor "THREE" is asserted by
#: the repaired clause: the clause states that the NUMBER of competing
#: readings is itself NOT settled on disk.  The superseded rendering is
#: preserved VERBATIM below (AGENTS.md §5.5 rule #6 -- append-only, dated,
#: never a silent edit).
CANONICAL_READING_SAFE_BAND_CLAUSE_SUPERSEDED_2026_11_01 = """\
the published band's provenance is NOT settled on disk: the frontier
post-processing matrix record (`frontier_matrix.json`) holds TWO competing
readings of which frontier cells the band names -- the first under
`band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94 %, both
`post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 %
as a third candidate for the same numeral) and the second under
`sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity
`C. = 92.98 %`, counted BEFORE the largest-component retention, with the
post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even
agree on the numerals (92.7/94 vs 92.98/94.33), so the band's source cells are
NOT settled on disk.  Under BOTH readings the band is EXTERNAL, is NOT one
homogeneous quantity, and is NOT on our scale -- OUR number IS a
connected-component count on the RAW emitted SMILES, and nothing else.  (In
THIS pool: {n_single_parsed_rows} of {n_rows_parsed} parsed rows have exactly
one connected component.)"""

#: Wf-A196 (2026-11-02): the REPAIRED canonical reading-safe band statement.
#: Same invariants, same numerals, no settled count of the contest.
CANONICAL_READING_SAFE_BAND_CLAUSE = """\
the published band's provenance is NOT settled on disk: the frontier
post-processing matrix record (`frontier_matrix.json`) holds COMPETING readings
of which frontier cells the band names -- the first under `band_provenance`
(GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94 %, both
`post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 %
as a third candidate for the same numeral) and the second under
`sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity
`C. = 92.98 %`, counted BEFORE the largest-component retention, with the
post-processed `N. & U. & S. = 94.33 %` companion cell).  The NUMBER of
competing readings is itself NOT settled on disk: this record enumerates two,
and the repo's primary-source verification
(`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json`
-> `invariants`.`competing_readings`) records a further reading that likewise
does not reproduce the quoted numerals.  They do not even agree on the
numerals (92.7/94 vs 92.98/94.33), so the band's source cells are NOT settled
on disk.  Under EVERY reading the band is EXTERNAL, is NOT one homogeneous
quantity, and is NOT on our scale -- OUR number IS a connected-component count
on the RAW emitted SMILES, and nothing else.  (In THIS pool:
{n_single_parsed_rows} of {n_rows_parsed} parsed rows have exactly one
connected component.)"""

COMPARABILITY_CAVEAT_TEMPLATE_SUPERSEDED_2026_11_01 = """\
COMPARABILITY CAVEAT -- applies to EVERY number in this report
================================================================
The frontier band this branch was told about -- EF-TALFM (arXiv 2609.08333)
"Connected 92.98 %, Sanitized 98.25 %, Unique+sanitized 98.21 %,
Novel+unique+sanitized 94.33 %" -- is NOT our band.  NOTHING in this report may
be read as a comparison against it, and no number here is a reproduction of it.

FOUR INDEPENDENT BREAKS (each against the LIVE fetch of arXiv 2609.08333v1 and
its LaTeX source, fetched {fetch_date}):

(1) DENOMINATOR.  Their rates are computed over 10,000 generated molecules per
    method (App. G.3: "We sample 10,000 molecules"; Table 5 caption: "All
    rates are computed over 10,000 generated molecules").  Ours are over the
    {pool_n_rows}-row raw decode pool
    {pool_path} -- ONE
    decode of ONE checkpoint.
(2) METRIC.  Their "Connectivity" is "% of molecules with exactly one connected
    component, i.e., one fragment", computed AFTER retaining the atom-count
    largest connected component and after RDKit sanitization.  Ours is an RDKit
    fragment count over RAW rows.
(3) POST-PROCESSING.  Their pipeline keeps the largest component, applies
    overlong-multiple-bond correction, adds explicit hydrogens and runs a
    restrained force-field relaxation before the numbers are counted.  Our pool
    rows are RAW: no relaxation, no hydrogen layer, no largest-component
    truncation in the counted population.
(4) HYDROGENS + CHEMISTRY, AND -- WORST -- A DIFFERENT BRANCH.  Their headline
    PCQM4Mv2 run is the ALL-ATOM branch ("we train EF-TALFM on PCQM4Mv2 using an
    all-atom molecular representation"), whose connectivity is recovered by
    OPEN BABEL perception, not by the tau_ij rule.  The tau_ij rule lives in the
    HEAVY-ATOM branch (Appendix D, Algorithm 4).  So THE FRONTIER'S HEADLINE
    NUMBERS WERE NOT PRODUCED BY THE MECHANISM THIS REPORT MEASURES.  On top of
    that: {rows_with_metal} of our {pool_n_rows_plain} pool rows carry a metal
    (metal_ratio {metal_ratio}) and every paper cohort is all-organic.

CONSEQUENCE: adopting tau_ij changes WHICH pairs the decoder MAY bond.  It does
NOT make our numbers comparable to theirs, and it does NOT make any fragment
rate in this repository comparable to the 92.7-94 % frontier band.

A FIFTH, SEPARATE LIMIT OF THIS REPORT, WHICH IS OURS AND NOT THEIRS:
the decoder's own interatomic distances are NOT RECOVERABLE from the pool
artefacts (the CSV header is epoch,smiles,n_frags_raw,largest_frag_frac -- no
coordinates; the run directory has no geometry dump).  So the literal
"would the emitted bond have survived tau_ij in the decoder's OWN geometry" is
NOT MEASURED here.  The report gives (a) exact geometry-free bounds, (b) an
exact rule-vs-rule comparison on REAL tmQM DFT geometry, and (c) a
length-TRANSFERRED estimate for the pool, each labelled as such.

AND THE THRESHOLD ITSELF IS CONTESTED, BY THE LIVE FETCH:
the paper's rule is tau_ij = lambda_POLICY * (r_cov(z_i) + r_cov(z_j)), where
lambda_POLICY is the tolerance factor of the reconstruction policy pi chosen by
ChoosePolicy (src/appendix.tex:1039 and :1043).  The subscript pi is NOT the
constant pi.  The branch brief's "lambda * pi * (...)" -- and the repo's own
Wf-A155 radii_scaled_factor() -- are the misreading, and they differ from the
paper by a factor of 3.14159.  Both conventions are reported below under the
labels "paper" and "repo_pi".  The paper's numeric lambda_POLICY is NOT
recoverable: the whole LaTeX source has no numeric value for it.
"""

#: Wf-A180 (2026-11-01): the CORRECTED caveat template.
COMPARABILITY_CAVEAT_TEMPLATE = """\
COMPARABILITY CAVEAT -- applies to EVERY number in this report
================================================================
The frontier band this branch was told about -- EF-TALFM (arXiv 2609.08333)
"Connected 92.98 %, Sanitized 98.25 %, Unique+sanitized 98.21 %,
Novel+unique+sanitized 94.33 %" -- is NOT our band.  NOTHING in this report may
be read as a comparison against it, and no number here is a reproduction of it.

AND NO ENDPOINT OF THAT BAND MAY BE NAMED AS SETTLED -- the canonical
reading-safe statement is the following, verbatim:
the published band's provenance is NOT settled on disk: the frontier
post-processing matrix record (`frontier_matrix.json`) holds COMPETING readings
of which frontier cells the band names -- the first under `band_provenance`
(GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94 %, both
`post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 %
as a third candidate for the same numeral) and the second under
`sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity
`C. = 92.98 %`, counted BEFORE the largest-component retention, with the
post-processed `N. & U. & S. = 94.33 %` companion cell).  The NUMBER of
competing readings is itself NOT settled on disk: this record enumerates two,
and the repo's primary-source verification
(`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json`
-> `invariants`.`competing_readings`) records a further reading that likewise
does not reproduce the quoted numerals.  They do not even agree on the
numerals (92.7/94 vs 92.98/94.33), so the band's source cells are NOT settled
on disk.  Under EVERY reading the band is EXTERNAL, is NOT one homogeneous
quantity, and is NOT on our scale -- OUR number IS a connected-component count
on the RAW emitted SMILES, and nothing else.  (In THIS pool:
{n_single_parsed_rows} of {n_rows_parsed} parsed rows have exactly one
connected component.)

FOUR INDEPENDENT BREAKS (each against the LIVE fetch of arXiv 2609.08333v1 and
its LaTeX source, fetched {fetch_date}):

(1) DENOMINATOR.  Their rates are computed over 10,000 generated molecules per
    method (App. G.3: "We sample 10,000 molecules"; Table 5 caption: "All
    rates are computed over 10,000 generated molecules").  Ours are over the
    {pool_n_rows}-row raw decode pool
    {pool_path} -- ONE
    decode of ONE checkpoint.
(2) METRIC.  Their "Connectivity" is "% of molecules with exactly one connected
    component, i.e., one fragment".  THAT CELL IS RAW: it is a single-component
    rate counted BEFORE the atom-count largest-component retention.  The paper's
    CAPTION sentence is AMBIGUOUS as to the order -- it grammatically attaches
    the largest-component clause to connectivity too -- but the NUMBER is RAW:
    a post-retention connectivity count would read 100 % by construction, and
    the paper reports 92.98 %.  Ours is an RDKit fragment count over RAW rows.
(3) POST-PROCESSING.  Their pipeline keeps the largest component, applies
    overlong-multiple-bond correction, adds explicit hydrogens and runs a
    restrained force-field relaxation before the COMPANION cells (sanitisation,
    uniqueness, novelty) are counted -- the connectivity cell itself is RAW, see
    (2).  Our pool rows are RAW: no relaxation, no hydrogen layer, no
    largest-component truncation in the counted population.
(4) HYDROGENS + CHEMISTRY, AND -- WORST -- A DIFFERENT BRANCH.  Their headline
    PCQM4Mv2 run is the ALL-ATOM branch ("we train EF-TALFM on PCQM4Mv2 using an
    all-atom molecular representation"), whose connectivity is recovered by
    OPEN BABEL perception, not by the tau_ij rule.  The tau_ij rule lives in the
    HEAVY-ATOM branch (Appendix D, Algorithm 4).  So THE FRONTIER'S HEADLINE
    NUMBERS WERE NOT PRODUCED BY THE MECHANISM THIS REPORT MEASURES.  On top of
    that: {rows_with_metal} of our {pool_n_rows_plain} pool rows carry a metal
    (metal_ratio {metal_ratio}) and every paper cohort is all-organic.

CONSEQUENCE: adopting tau_ij changes WHICH pairs the decoder MAY bond.  It does
NOT make our numbers comparable to theirs, and it does NOT make any fragment
rate in this repository comparable to the 92.7-94 % frontier band.

A FIFTH, SEPARATE LIMIT OF THIS REPORT, WHICH IS OURS AND NOT THEIRS:
the decoder's own interatomic distances are NOT RECOVERABLE from the pool
artefacts (the CSV header is epoch,smiles,n_frags_raw,largest_frag_frac -- no
coordinates; the run directory has no geometry dump).  So the literal
"would the emitted bond have survived tau_ij in the decoder's OWN geometry" is
NOT MEASURED here.  The report gives (a) exact geometry-free bounds, (b) an
exact rule-vs-rule comparison on REAL tmQM DFT geometry, and (c) a
length-TRANSFERRED estimate for the pool, each labelled as such.

AND THE THRESHOLD ITSELF IS CONTESTED, BY THE LIVE FETCH:
the paper's rule is tau_ij = lambda_POLICY * (r_cov(z_i) + r_cov(z_j)), where
lambda_POLICY is the tolerance factor of the reconstruction policy pi chosen by
ChoosePolicy (src/appendix.tex:1039 and :1043).  The subscript pi is NOT the
constant pi.  The branch brief's "lambda * pi * (...)" -- and the repo's own
Wf-A155 radii_scaled_factor() -- are the misreading, and they differ from the
paper by a factor of 3.14159.  Both conventions are reported below under the
labels "paper" and "repo_pi".  The paper's numeric lambda_POLICY is NOT
recoverable: the whole LaTeX source has no numeric value for it.
"""


#: Wf-A180 (2026-11-01): the templates and the standalone paragraph
#: constants must not drift apart -- a silent drift would un-date the
#: correction.  Checked at import time.
for _tpl_name, _tpl_body, _needle in (
    ("COMPARABILITY_CAVEAT_TEMPLATE_SUPERSEDED_2026_11_01",
     COMPARABILITY_CAVEAT_TEMPLATE_SUPERSEDED_2026_11_01,
     SUPERSEDED_METRIC_SENTENCE),
    ("COMPARABILITY_CAVEAT_TEMPLATE", COMPARABILITY_CAVEAT_TEMPLATE,
     CORRECTED_METRIC_SENTENCE),
    ("COMPARABILITY_CAVEAT_TEMPLATE", COMPARABILITY_CAVEAT_TEMPLATE,
     CANONICAL_READING_SAFE_BAND_CLAUSE),
):
    if _tpl_body.count(_needle) != 1:
        raise RuntimeError(
            f"wfa156: {_tpl_name} does not carry {_needle[:48]!r} exactly once"
        )

#: Wf-A196 (2026-11-02): the LIVE caveat template must carry the REPAIRED
#: clause and must NOT carry the superseded one, and the superseded clause
#: must survive VERBATIM as a module-level dated sibling (AGENTS.md §5.5
#: rule #6).  Checked at import time.
if COMPARABILITY_CAVEAT_TEMPLATE.count(
    CANONICAL_READING_SAFE_BAND_CLAUSE_SUPERSEDED_2026_11_01
) != 0:
    raise RuntimeError(
        "wfa156: the live comparability caveat still carries the superseded "
        "reading-count clause"
    )
if CANONICAL_READING_SAFE_BAND_CLAUSE_SUPERSEDED_2026_11_01 == (
    CANONICAL_READING_SAFE_BAND_CLAUSE
):
    raise RuntimeError(
        "wfa156: the superseded reading-count clause is not preserved verbatim "
        "as a dated sibling of the repaired one"
    )


def build_comparability_caveat(
    pool_path: Path,
    n_rows: int,
    rows_with_metal: int,
    pool_path_display: str,
    n_rows_parsed: int = 0,
    rows_with_one_fragment: int = 0,
) -> str:
    """The comparability caveat, with OUR population's numbers COMPUTED.

    Wf-A164 (2026-11-01): this used to be a module constant that carried the
    DEFAULT pool's row count, the DEFAULT pool's path and the DEFAULT pool's
    metal ratio as literals.  Those are TRUE for the default pool and FALSE
    for any other pool, and the tool emitted them into every artefact it
    produced.  Every number that describes OUR population is now taken from
    the run, so the caveat cannot disagree with the machine fields of the same
    artefact.

    Wf-A180 (2026-11-01): the template now states that EF-TALFM's connectivity
    cell is RAW (counted BEFORE the largest-component retention) and carries
    the canonical reading-safe band statement; the superseded rendering is
    preserved verbatim by :func:`build_comparability_caveat_superseded`.
    """
    metal_ratio = (rows_with_metal / n_rows) if n_rows else 0.0
    return COMPARABILITY_CAVEAT_TEMPLATE.format(
        fetch_date=LIVE_FETCH_DATE,
        pool_n_rows=f"{n_rows:,}",
        pool_n_rows_plain=n_rows,
        pool_path=pool_path_display,
        rows_with_metal=rows_with_metal,
        metal_ratio=round(metal_ratio, 6),
        n_single_parsed_rows=rows_with_one_fragment,
        n_rows_parsed=n_rows_parsed,
    )


def build_comparability_caveat_superseded(
    pool_path: Path,
    n_rows: int,
    rows_with_metal: int,
    pool_path_display: str,
) -> str:
    """The SUPERSEDED caveat (Wf-A180, 2026-11-01), rendered VERBATIM.

    Same substitutions as the live template, so the superseded text an
    artefact carries is byte-identical to what this tool used to emit.
    """
    metal_ratio = (rows_with_metal / n_rows) if n_rows else 0.0
    return COMPARABILITY_CAVEAT_TEMPLATE_SUPERSEDED_2026_11_01.format(
        fetch_date=LIVE_FETCH_DATE,
        pool_n_rows=f"{n_rows:,}",
        pool_n_rows_plain=n_rows,
        pool_path=pool_path_display,
        rows_with_metal=rows_with_metal,
        metal_ratio=round(metal_ratio, 6),
    )


def build_comparability_caveat_correction(
    corrected_caveat_text: str,
    superseded_caveat_text: str,
    n_rows_parsed: int,
    rows_with_one_fragment: int,
) -> Dict[str, Any]:
    """The dated Wf-A180 correction record (AGENTS.md §5.5 rule #6).

    ADDITIVE: it records what was wrong, what is right, and where the
    adjudication lives on disk.  It never deletes or silently rewrites the
    superseded rendering, which travels in its own dated key.
    """
    return {
        "canonical_reading_safe_clause":
            CANONICAL_READING_SAFE_BAND_CLAUSE_SUPERSEDED_2026_11_01.format(
                n_single_parsed_rows=rows_with_one_fragment,
                n_rows_parsed=n_rows_parsed,
            ),
        "corrected_caveat_text": corrected_caveat_text,
        "corrected_sentence": CORRECTED_METRIC_SENTENCE,
        "date": "2026-11-01",
        "defect": (
            "X3 -- the emitted text asserted, AS A SETTLED FACT, that "
            "EF-TALFM's connectivity cell (\"% of molecules with exactly one "
            "connected component, i.e., one fragment\") is computed AFTER "
            "the atom-count largest-component retention.  REFUTED ON DISK, "
            "twice: the cell is a RAW single-component rate counted BEFORE "
            "the retention, and only the paper's CAPTION sentence is "
            "ambiguous as to the order."
        ),
        "locators": (
            "molmetal/reports/wfa158_frontier_postproc_matrix/"
            "frontier_matrix.json -> `band_provenance` and `sources`.`EF-TALFM` "
            "(JSON PATHS, not line numbers)"
        ),
        "note": (
            "The live `comparability_caveat` key of a pre-correction artefact "
            "still holds the SUPERSEDED rendering, preserved VERBATIM "
            "(AGENTS.md §5.5 rule #6); `corrected_caveat_text` below is the "
            "rendering this tool emits now, and `superseded_caveat_text` "
            "below is the superseded rendering, byte-identical to what was "
            "emitted.  The band statement quoted "
            "here is the 2026-11-01 (Wf-A180) rendering, preserved VERBATIM; "
            "it is SUPERSEDED by the 2026-11-02 (Wf-A196) record "
            "`comparability_caveat_reading_count_corrected_wfa196`, because it "
            "asserts a SETTLED COUNT of the competing readings.  The band's "
            "provenance is NOT settled on disk, so NO endpoint of it may be "
            "named as settled."
        ),
        "superseded_by": (
            "comparability_caveat_reading_count_corrected_wfa196 -- the "
            "reading-COUNT repair (Wf-A196, 2026-11-02)."
        ),
        "on_disk_adjudications": [
            "molmetal/reports/wfa168_eftalfm_axis/report.md -- \"connectivity "
            "is RAW; the caption's sentence is ambiguous, not the number.\"",
            "molmetal/reports/wfa171_frontier_postproc/report.md -- \"92.98 % "
            "is a RAW single-component rate\", counted BEFORE the "
            "largest-component retention.",
        ],
        "superseded_caveat_text": superseded_caveat_text,
        "superseded_sentence": SUPERSEDED_METRIC_SENTENCE,
        "wave": "Wf-A180",
    }


def render_comparability_correction_section(
    correction: Dict[str, Any],
) -> List[str]:
    """Render §0b: the SUPERSEDED rendering + the dated Wf-A180 correction.

    One implementation, used by :func:`render_markdown` and by the artefact
    repair, so a repaired artefact and a fresh run render the same bytes.
    """
    lines: List[str] = []
    A = lines.append
    A("### 0b. SUPERSEDED RENDERING + Wf-A180 CORRECTION (2026-11-01)")
    A("")
    A("The rendering this tool emitted BEFORE the Wf-A180 correction is")
    A("preserved VERBATIM below (AGENTS.md §5.5 rule #6 -- append-only, dated,")
    A("never a silent edit):")
    A("")
    A("```")
    A(correction["superseded_caveat_text"].rstrip())
    A("```")
    A("")
    A("**SUPERSEDED (2026-11-01, Wf-A180).**  The following sentence was")
    A("emitted as a settled fact and is REFUTED ON DISK:")
    A("")
    for ln in correction["superseded_sentence"].splitlines():
        A("> " + ln)
    A("")
    A("**CORRECTION (2026-11-01, Wf-A180).**")
    A("")
    for ln in correction["corrected_sentence"].splitlines():
        A("> " + ln)
    A("")
    for ln in correction["canonical_reading_safe_clause"].splitlines():
        A("> " + ln)
    A("")
    A("**ON-DISK ADJUDICATIONS (read these before trusting either rendering).**")
    A("")
    for item in correction["on_disk_adjudications"]:
        A(f"- {item}")
    A("")
    A("**LOCATORS (JSON PATHS, not line numbers).**  "
      f"{correction['locators']}")
    A("")
    return lines


def build_reading_count_correction(
    n_rows_parsed: int,
    rows_with_one_fragment: int,
) -> Dict[str, Any]:
    """The dated Wf-A196 reading-COUNT correction record (AGENTS.md §5.5 rule #6).

    ADDITIVE: it records what was wrong, what is right, and where the
    adjudication lives on disk.  It never deletes or silently rewrites the
    superseded rendering, which travels in its own dated key
    (``superseded_reading_safe_clause_2026_11_01``).
    """
    return {
        "canonical_reading_safe_clause":
            CANONICAL_READING_SAFE_BAND_CLAUSE.format(
                n_single_parsed_rows=rows_with_one_fragment,
                n_rows_parsed=n_rows_parsed,
            ),
        "date": "2026-11-02",
        "defect": (
            "X5 -- the emitted text asserted, AS A SETTLED COUNT, that the "
            "frontier post-processing matrix record \"holds TWO competing "
            "readings\".  The count is NOT settled on disk: the repo's "
            "primary-source verification records a FURTHER reading that also "
            "does not reproduce the quoted numerals, so two on-disk records "
            "disagree about the SIZE of the contest.  The repaired clause "
            "asserts NO count -- it states that the NUMBER of competing "
            "readings is itself NOT settled on disk.  It does NOT assert a "
            "count of three either: how many DISTINCT readings there are is "
            "itself not settled on disk."
        ),
        "locators": (
            "molmetal/reports/wfa158_frontier_postproc_matrix/"
            "frontier_matrix.json -> `band_provenance` and `sources`.`EF-TALFM` "
            "(JSON PATHS, not line numbers); "
            "molmetal/reports/wfa186_frontier_primary_source/"
            "primary_source_verification.json -> `invariants`.`competing_readings`"
        ),
        "note": (
            "The four invariants are UNCHANGED by this repair: under EVERY "
            "reading the band is EXTERNAL, is NOT one homogeneous quantity, "
            "and is NOT on our scale, and the band's source cells are NOT "
            "settled on disk.  ONLY the count is repaired.  The 2026-11-01 "
            "rendering of the clause is preserved VERBATIM in "
            "`superseded_reading_safe_clause_2026_11_01` and is rendered "
            "verbatim in section 0b above."
        ),
        "supersedes": (
            "comparability_caveat_corrected_wfa180 -- the Wf-A180 record "
            "(2026-11-01)."
        ),
        "superseded_reading_safe_clause_2026_11_01":
            CANONICAL_READING_SAFE_BAND_CLAUSE_SUPERSEDED_2026_11_01.format(
                n_single_parsed_rows=rows_with_one_fragment,
                n_rows_parsed=n_rows_parsed,
            ),
        "wave": "Wf-A196",
    }


def render_reading_count_correction_section(
    correction: Dict[str, Any],
) -> List[str]:
    """Render §0c: the dated Wf-A196 reading-COUNT correction.

    One implementation, used by :func:`render_markdown`, so a fresh run and
    a repaired artefact render the same bytes.
    """
    lines: List[str] = []
    A = lines.append
    A("### 0c. Wf-A196 CORRECTION (2026-11-02) -- the NUMBER of competing")
    A("readings is itself NOT settled on disk")
    A("")
    A("**DEFECT.**  The block titled `CORRECTION (2026-11-01, Wf-A180)` above")
    A("carries a SETTLED COUNT of the competing readings.  That count is NOT")
    A("settled on disk: the repo's primary-source verification records a")
    A("FURTHER reading that likewise does not reproduce the quoted numerals,")
    A("so two on-disk records disagree about the SIZE of the contest.  The")
    A("count claim in the 2026-11-01 block is therefore SUPERSEDED by this")
    A("dated section; that block is preserved VERBATIM above (AGENTS.md §5.5")
    A("rule #6 -- append-only, dated, never a silent edit).")
    A("")
    A("**CORRECTION (2026-11-02, Wf-A196).**")
    A("")
    for ln in correction["canonical_reading_safe_clause"].splitlines():
        A("> " + ln)
    A("")
    A("**THE FOUR INVARIANTS ARE UNCHANGED.**  Under EVERY reading the band is")
    A("EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale --")
    A("OUR number IS a connected-component count on the RAW emitted SMILES,")
    A("and nothing else.  The band's source cells are NOT settled on disk.")
    A("")
    A("**LOCATORS (JSON PATHS, not line numbers).**  "
      f"{correction['locators']}")
    A("")
    return lines



# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _periodic_table() -> Any:
    return Chem.GetPeriodicTable()


def rcov(z: int) -> Optional[float]:
    """RDKit covalent radius for atomic number ``z``, in Angstrom.

    MEASURED: a lookup in RDKit's periodic table
    (``GetPeriodicTable().GetRcovalent``), which is the radius source the
    branch brief specified.  Returns ``None`` for an atomic number RDKit does
    not know, so an unknown element is never silently given a fake radius.
    """
    try:
        val = float(_periodic_table().GetRcovalent(int(z)))
    except Exception:
        return None
    if not math.isfinite(val) or val <= 0.0:
        return None
    return val


def talfm_tau(
    z_i: int,
    z_j: int,
    lam: float,
    convention: str = "paper",
) -> Optional[float]:
    """``tau_ij`` for one element pair, in Angstrom.

    ``convention="paper"``    -> ``lam * (r_cov(z_i) + r_cov(z_j))``
    ``convention="repo_pi"``  -> ``lam * pi * (r_cov(z_i) + r_cov(z_j))``

    MEASURED: pure arithmetic on RDKit's covalent radii and the caller's
    lambda.  ``None`` when either radius is unavailable.

    The two conventions are reported side by side because the branch brief
    (and the repo's Wf-A155 implementation) read the paper's ``lambda_pi`` as
    ``lambda * pi``.  See the module docstring.
    """
    r_i = rcov(z_i)
    r_j = rcov(z_j)
    if r_i is None or r_j is None:
        return None
    if convention == "paper":
        factor = float(lam)
    elif convention == "repo_pi":
        factor = float(lam) * math.pi
    else:
        raise ValueError(f"unknown convention {convention!r}")
    return factor * (r_i + r_j)


def bond_survives(
    distance_a: float,
    z_i: int,
    z_j: int,
    lam: float,
    convention: str = "paper",
) -> Optional[bool]:
    """Would a bond at ``distance_a`` survive the frontier threshold?

    The paper's own acceptance test is ``d_ij <= tau_ij`` (Algorithm 4
    line 7).  MEASURED: pure comparison.
    """
    tau = talfm_tau(z_i, z_j, lam, convention)
    if tau is None:
        return None
    return bool(distance_a <= tau)


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


def read_pool(path: Path) -> List[Dict[str, str]]:
    """Read the pool CSV.  MEASURED: a straight CSV read, no filtering."""
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(dict(row))
    return rows


def _fragments(mol: Any) -> List[Any]:
    return Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)


def _bond_order_int(bond: Any) -> int:
    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return 1
    if bt == Chem.BondType.DOUBLE:
        return 2
    if bt == Chem.BondType.TRIPLE:
        return 3
    if bt == Chem.BondType.AROMATIC:
        return 4  # a distinct bin: 1.5 orders, tabulated separately
    return 1


def analyse_pool(
    rows: Sequence[Dict[str, str]],
    ref_lengths: Dict[Tuple[int, int, int], Dict[str, Any]],
    lambdas: Sequence[float] = LAMBDAS,
    conventions: Sequence[str] = CONVENTIONS,
) -> Dict[str, Any]:
    """Measure the pool against the frontier threshold.

    Three tiers, exactly as described in the module docstring:

    T1  EXACT geometry-free: tau_ij vs the decoder's own 3.5 A flat gate.
    T2  EXACT geometry-free bounds: guaranteed survive / guaranteed fail /
        undecidable, per emitted bond.
    T3  LENGTH-TRANSFERRED estimate: survival under the tmQM-DFT empirical
        mean length for the bond's (Z, Z, order) bin.

    Returns a dict with an aggregate block and a per-row block.
    """
    per_row: List[Dict[str, Any]] = []
    agg: Dict[str, Any] = {
        "n_rows": len(rows),
        "n_rows_parsed": 0,
        "n_rows_unparsed": 0,
        "n_bonds_total": 0,
        "n_bonds_with_ref_length": 0,
        "n_bonds_without_ref_length": 0,
        "fragments": Counter(),
        "element_pairs": Counter(),
        "bond_orders": Counter(),
        "metal_bonds": 0,
        "rows_with_metal": 0,
        "metal_shares_fragment": 0,
        "metal_alone": 0,
        "metal_atoms_total": 0,
        "rows_with_arrow": 0,
        "t1_stricter_than_decoder": {},
        "t1_looser_than_decoder": {},
        "t2": {},
        "t3": {},
    }

    for conv in conventions:
        for lam in lambdas:
            agg["t1_stricter_than_decoder"][f"{conv}|{lam}"] = 0
            agg["t1_looser_than_decoder"][f"{conv}|{lam}"] = 0
            agg["t2"][f"{conv}|{lam}"] = {"guaranteed_survive": 0, "guaranteed_fail": 0, "undecidable": 0}
            agg["t3"][f"{conv}|{lam}"] = {"survived": 0, "failed": 0, "no_reference_length": 0}

    for row_index, row in enumerate(rows):
        smi = (row.get("smiles") or "").strip()
        rec: Dict[str, Any] = {
            "row_index": row_index,
            "epoch": row.get("epoch"),
            "smiles": smi,
            "n_frags_raw": row.get("n_frags_raw"),
            "largest_frag_frac": row.get("largest_frag_frac"),
            "parsed": False,
            "n_fragments_rdkit": None,
            "n_bonds": None,
            "bonds": [],
            "t2": {},
            "t3": {},
        }
        mol = None
        if smi:
            try:
                mol = Chem.MolFromSmiles(smi)
            except Exception:
                mol = None
        if mol is None:
            agg["n_rows_unparsed"] += 1
            for conv in conventions:
                for lam in lambdas:
                    key = f"{conv}|{lam}"
                    rec["t2"][key] = {"guaranteed_survive": 0, "guaranteed_fail": 0, "undecidable": 0}
                    rec["t3"][key] = {"survived": 0, "failed": 0, "no_reference_length": 0}
            per_row.append(rec)
            continue

        agg["n_rows_parsed"] += 1
        frags = _fragments(mol)
        rec["parsed"] = True
        rec["n_fragments_rdkit"] = len(frags)
        agg["fragments"][len(frags)] += 1

        # ---- population cross-check (the branch brief's own numbers) ----
        # MEASURED from this parse, so a reader can confirm that this tool
        # sees the SAME population the brief describes.
        metal_zs = COHORT_METAL_Z
        n_metal_atoms = 0
        for frag in frags:
            zs = [int(a.GetAtomicNum()) for a in frag.GetAtoms()]
            frag_metal = [z for z in zs if z in metal_zs]
            frag_nonmetal = [z for z in zs if z not in metal_zs]
            for _ in frag_metal:
                n_metal_atoms += 1
                agg["metal_atoms_total"] += 1
                if frag_nonmetal:
                    agg["metal_shares_fragment"] += 1
                else:
                    agg["metal_alone"] += 1
        if n_metal_atoms:
            agg["rows_with_metal"] += 1
        if "->" in smi:
            agg["rows_with_arrow"] += 1

        n_bonds_row = 0
        for conv in conventions:
            for lam in lambdas:
                rec["t2"][f"{conv}|{lam}"] = {"guaranteed_survive": 0, "guaranteed_fail": 0, "undecidable": 0}
                rec["t3"][f"{conv}|{lam}"] = {"survived": 0, "failed": 0, "no_reference_length": 0}

        for bond in mol.GetBonds():
            i = bond.GetBeginAtom()
            j = bond.GetEndAtom()
            z_i = int(i.GetAtomicNum())
            z_j = int(j.GetAtomicNum())
            order = _bond_order_int(bond)
            z_min, z_max = (z_i, z_j) if z_i <= z_j else (z_j, z_i)
            n_bonds_row += 1
            agg["n_bonds_total"] += 1
            agg["element_pairs"][f"{z_min}-{z_max}"] += 1
            agg["bond_orders"][order] += 1
            is_metal = z_i in COHORT_METAL_Z or z_j in COHORT_METAL_Z
            if is_metal:
                agg["metal_bonds"] += 1

            # --- reference length: tmQM-DFT empirical mean for this bin ---
            ref = ref_lengths.get((z_min, z_max, order))
            if ref is None and order == 4:
                # Aromatic bonds are tabulated as BO 1 / BO 2 halves in the
                # tmQM table; fall back to the single-bond bin and say so.
                ref = ref_lengths.get((z_min, z_max, 1))
            if ref is not None:
                agg["n_bonds_with_ref_length"] += 1
                ref_len = float(ref["mean"])
                ref_min = float(ref["min"])
                ref_src = "tmqm_dft_mean"
            else:
                agg["n_bonds_without_ref_length"] += 1
                ref_len = None
                ref_min = None
                ref_src = "none"

            bond_rec: Dict[str, Any] = {
                "z_i": z_i,
                "z_j": z_j,
                "order": order,
                "is_metal_bond": bool(is_metal),
                "rcov_i": rcov(z_i),
                "rcov_j": rcov(z_j),
                "ref_length_mean": ref_len,
                "ref_length_min": ref_min,
                "ref_length_source": ref_src,
                # tau depends ONLY on (z_i, z_j, convention, lambda), so it is
                # stored as flat lists aligned with LAMBDAS rather than a
                # nested dict -- 10 floats per bond instead of 10 keyed pairs.
            }
            tau_lists: Dict[str, List[Optional[float]]] = {c: [] for c in conventions}

            for conv in conventions:
                for lam in lambdas:
                    key = f"{conv}|{lam}"
                    tau = talfm_tau(z_i, z_j, lam, conv)
                    tau_lists[conv].append(tau)

                    # ---- T1: exact, geometry-free ----
                    if tau is None:
                        agg["t1_stricter_than_decoder"][key] += 0
                        agg["t1_looser_than_decoder"][key] += 0
                        rec["t2"][key]["undecidable"] += 1
                    elif tau < DECODER_FLAT_CUTOFF_A:
                        agg["t1_stricter_than_decoder"][key] += 1
                    else:
                        agg["t1_looser_than_decoder"][key] += 1

                    # ---- T2: exact bounds ----
                    # GUARANTEED SURVIVE: the decoder only emits pairs inside
                    # its 3.5 A gate, so d <= 3.5 <= tau implies d <= tau.
                    if tau is not None and tau >= DECODER_FLAT_CUTOFF_A:
                        rec["t2"][key]["guaranteed_survive"] += 1
                        agg["t2"][key]["guaranteed_survive"] += 1
                    elif tau is None:
                        rec["t2"][key]["undecidable"] += 1
                        agg["t2"][key]["undecidable"] += 1
                    elif ref_min is not None and tau < ref_min:
                        # No bond of this element pair has EVER been observed
                        # shorter than ref_min in the tmQM DFT corpus, and
                        # tau is below that floor: the bond cannot survive.
                        rec["t2"][key]["guaranteed_fail"] += 1
                        agg["t2"][key]["guaranteed_fail"] += 1
                    else:
                        rec["t2"][key]["undecidable"] += 1
                        agg["t2"][key]["undecidable"] += 1

                    # ---- T3: length-transferred estimate ----
                    if ref_len is None:
                        rec["t3"][key]["no_reference_length"] += 1
                        agg["t3"][key]["no_reference_length"] += 1
                    elif tau is None:
                        rec["t3"][key]["no_reference_length"] += 1
                        agg["t3"][key]["no_reference_length"] += 1
                    else:
                        ok = ref_len <= tau
                        rec["t3"][key]["survived" if ok else "failed"] += 1
                        agg["t3"][key]["survived" if ok else "failed"] += 1

            bond_rec["tau_paper"] = tau_lists.get("paper")
            bond_rec["tau_repo_pi"] = tau_lists.get("repo_pi")
            rec["bonds"].append(bond_rec)

        rec["n_bonds"] = n_bonds_row
        per_row.append(rec)

    # ---- aggregate rates -------------------------------------------------
    nb = agg["n_bonds_total"]
    rates: Dict[str, Any] = {}
    for conv in conventions:
        for lam in lambdas:
            key = f"{conv}|{lam}"
            t2 = agg["t2"][key]
            t3 = agg["t3"][key]
            t3_den = t3["survived"] + t3["failed"]
            rates[key] = {
                "lambda": lam,
                "convention": conv,
                "tau_formula": (
                    "lambda * (r_cov(z_i) + r_cov(z_j))"
                    if conv == "paper"
                    else "lambda * pi * (r_cov(z_i) + r_cov(z_j))"
                ),
                "t1_frac_stricter_than_decoder_cutoff": (
                    agg["t1_stricter_than_decoder"][key] / nb if nb else None
                ),
                "t1_n_stricter_than_decoder_cutoff": agg["t1_stricter_than_decoder"][key],
                "t1_n_looser_than_decoder_cutoff": agg["t1_looser_than_decoder"][key],
                "t2_frac_guaranteed_survive": (
                    t2["guaranteed_survive"] / nb if nb else None
                ),
                "t2_frac_guaranteed_fail": (
                    t2["guaranteed_fail"] / nb if nb else None
                ),
                "t2_frac_undecidable": (t2["undecidable"] / nb if nb else None),
                "t2_n": dict(t2),
                "t3_frac_survived_ref_length": (
                    t3["survived"] / t3_den if t3_den else None
                ),
                "t3_n": dict(t3),
                "t3_denominator": t3_den,
            }
    agg["rates"] = rates
    agg["_per_row"] = per_row
    return agg


# ---------------------------------------------------------------------------
# tmQM baseline -- REAL DFT geometry, REAL distances
# ---------------------------------------------------------------------------


def load_ref_lengths(tmqm_root: Path) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    """The tmQM-DFT empirical bond-length table, keyed by (z_min, z_max, order).

    MEASURED: the on-disk cache
    ``.data/molmetal/tmQM/bond_length_table.json``, built by
    ``molmetal/data/bond_length_prior.py::build_bond_length_table`` from the
    real tmQM XYZ geometries + Wiberg bond orders (Wiberg BO > 0.1, bins with
    n >= 5).  It carries ``mean / std / n / min / max`` per bin.
    """
    from molmetal.data.bond_length_prior import load_bond_length_table

    table = load_bond_length_table(cache_path=tmqm_root / "bond_length_table.json", root=tmqm_root)
    out: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    if not table:
        return out
    for key, val in table.items():
        out[(int(key[0]), int(key[1]), int(key[2]))] = {
            "mean": float(val["mean"]),
            "std": float(val["std"]),
            "n": int(val["n"]),
            "min": float(val["min"]),
            "max": float(val["max"]),
        }
    return out


def _decode_cohort_codes(tmqm_root: Path, metals: Sequence[str]) -> List[str]:
    """csd_codes of the tmQM complexes whose metal is in ``metals``.

    MEASURED: read from ``tmqm_parsed.csv`` on disk (column ``metal``).
    """
    import csv as _csv

    wanted = {str(m).strip() for m in metals}
    codes: List[str] = []
    path = tmqm_root / "tmqm_parsed.csv"
    if not path.is_file():
        return codes
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            if (row.get("metal") or "").strip() in wanted:
                codes.append((row.get("csd_code") or "").strip())
    return [c for c in codes if c]


def tmqm_baseline(
    tmqm_root: Path,
    codes: Sequence[str],
    lambdas: Sequence[float] = LAMBDAS,
    conventions: Sequence[str] = CONVENTIONS,
    max_complexes: int = DEFAULT_BASELINE_COMPLEXES,
) -> Dict[str, Any]:
    """Measure the EF-TALFM threshold on REAL tmQM DFT geometry.

    For each selected complex we compute, from the real XYZ coordinates:

    * every WIBERG BOND's real distance, and whether it survives tau_ij;
    * every HEAVY-ATOM PAIR inside the decoder's own 3.5 A candidate gate, and
      whether tau_ij would accept it -- i.e. the candidate-set comparison
      between the decoder's flat rule and the frontier's element-aware rule.

    MEASURED: every number here comes from a real distance in a real
    geometry.  There is no transfer and no estimate on this side.
    """
    from molmetal.data.bond_length_prior import (
        _bo_block_lines,
        _element_to_z,
        _xyz_complex_block,
    )

    xyz_files = ("tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz")
    bo_files = ("tmQM_X1.BO.gz", "tmQM_X2.BO.gz", "tmQM_X3.BO.gz")

    wanted = set(codes)
    # Deterministic, evenly spaced subsample over the sorted code list.
    ordered = sorted(wanted)
    if len(ordered) > max_complexes:
        step = len(ordered) / float(max_complexes)
        keep = {ordered[int(i * step)] for i in range(max_complexes)}
    else:
        keep = set(ordered)
    selected = sorted(keep)

    coords_by_code: Dict[str, list] = {}
    for name in xyz_files:
        p = tmqm_root / name
        if not p.is_file():
            continue
        for code, atoms in _xyz_complex_block(p):
            if code in keep:
                coords_by_code[code] = atoms

    stats: Dict[str, Any] = {
        "n_complexes_requested": len(selected),
        "n_complexes_with_coordinates": len(coords_by_code),
        "n_bonds_total": 0,
        "n_candidate_pairs_total": 0,
        "n_heavy_atoms_total": 0,
        "bond_survived": {},
        "bond_failed": {},
        "bond_failed_pairs": {},
        "cand_accepted": {},
        "cand_rejected":{},
        "decoder_flat_accepted": 0,
        "decoder_flat_rejected": 0,
        "bond_distance_stats": {},
    }
    for conv in conventions:
        for lam in lambdas:
            key = f"{conv}|{lam}"
            stats["bond_survived"][key] = 0
            stats["bond_failed"][key] = 0
            stats["bond_failed_pairs"][key] = {}
            stats["cand_accepted"][key] = 0
            stats["cand_rejected"][key] = 0

    distances: List[float] = []
    bond_hist: Counter = Counter()

    for bo_name in bo_files:
        bo_path = tmqm_root / bo_name
        if not bo_path.is_file():
            continue
        for code, atom_rows in _bo_block_lines(bo_path):
            atoms = coords_by_code.get(code)
            if atoms is None:
                continue
            # ---- real bonds (Wiberg BO > 0.1), one per unordered pair ----
            for parts in atom_rows:
                try:
                    self_idx = int(parts[0]) - 1
                    self_z = _element_to_z(parts[1])
                except (ValueError, IndexError):
                    continue
                if self_z is None or self_idx < 0 or self_idx >= len(atoms):
                    continue
                sz, sx, sy, szz = atoms[self_idx]
                if sz != self_z:
                    continue
                for off in range(3, len(parts) - 2, 3):
                    try:
                        nbr_z = _element_to_z(parts[off])
                        nbr_idx = int(parts[off + 1]) - 1
                        bo_val = float(parts[off + 2])
                    except (ValueError, IndexError):
                        continue
                    if nbr_z is None or bo_val <= 0.1:
                        continue
                    if nbr_idx <= self_idx or nbr_idx >= len(atoms):
                        continue
                    nz, nx, ny, nzz = atoms[nbr_idx]
                    if nz != nbr_z:
                        continue
                    d = math.sqrt((sx - nx) ** 2 + (sy - ny) ** 2 + (szz - nzz) ** 2)
                    distances.append(d)
                    bond_hist[(sz, nz)] += 1
                    stats["n_bonds_total"] += 1
                    for conv in conventions:
                        for lam in lambdas:
                            key = f"{conv}|{lam}"
                            tau = talfm_tau(sz, nz, lam, conv)
                            if tau is None:
                                continue
                            if d <= tau:
                                stats["bond_survived"][key] += 1
                            else:
                                stats["bond_failed"][key] += 1
                                pair = f"{min(sz, nz)}-{max(sz, nz)}"
                                stats["bond_failed_pairs"][key][pair] = (
                                    stats["bond_failed_pairs"][key].get(pair, 0) + 1
                                )

            # ---- candidate pairs under the decoder's flat 3.5 A gate ----
            n = len(atoms)
            heavy = [k for k in range(n) if atoms[k][0] != 1]
            stats["n_heavy_atoms_total"] += len(heavy)
            for a in range(len(heavy)):
                ia = heavy[a]
                za, xa, ya, za_ = atoms[ia]
                for b in range(a + 1, len(heavy)):
                    ib = heavy[b]
                    zb, xb, yb, zb_ = atoms[ib]
                    d = math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za_ - zb_) ** 2)
                    if d > DECODER_FLAT_CUTOFF_A:
                        continue
                    stats["n_candidate_pairs_total"] += 1
                    if d <= DECODER_FLAT_CUTOFF_A:
                        stats["decoder_flat_accepted"] += 1
                    else:
                        stats["decoder_flat_rejected"] += 1
                    for conv in conventions:
                        for lam in lambdas:
                            key = f"{conv}|{lam}"
                            tau = talfm_tau(za, zb, lam, conv)
                            if tau is None:
                                continue
                            if d <= tau:
                                stats["cand_accepted"][key] += 1
                            else:
                                stats["cand_rejected"][key] += 1

    if distances:
        srt = sorted(distances)
        stats["bond_distance_stats"] = {
            "n": len(srt),
            "min": srt[0],
            "p01": srt[max(0, int(0.01 * len(srt)))],
            "median": statistics.median(srt),
            "p99": srt[min(len(srt) - 1, int(0.99 * len(srt)))],
            "max": srt[-1],
            "mean": sum(srt) / len(srt),
        }
    stats["bond_distance_hist_by_element_pair"] = {
        f"{a}-{b}": n for (a, b), n in sorted(bond_hist.items())
    }

    rates: Dict[str, Any] = {}
    nb = stats["n_bonds_total"]
    nc = stats["n_candidate_pairs_total"]
    for conv in conventions:
        for lam in lambdas:
            key = f"{conv}|{lam}"
            surv = stats["bond_survived"][key]
            ca = stats["cand_accepted"][key]
            rates[key] = {
                "lambda": lam,
                "convention": conv,
                "baseline_frac_true_bonds_surviving_tau": (surv / nb) if nb else None,
                "baseline_n_true_bonds": nb,
                "baseline_n_true_bonds_surviving": surv,
                "baseline_n_true_bonds_failing": stats["bond_failed"][key],
                "candidate_set_frac_accepted_by_tau": (ca / nc) if nc else None,
                "candidate_set_n_pairs": nc,
                "candidate_set_n_accepted": ca,
                "candidate_set_n_rejected": stats["cand_rejected"][key],
                "top_failing_element_pairs": dict(
                    sorted(
                        stats["bond_failed_pairs"][key].items(),
                        key=lambda kv: -kv[1],
                    )[:10]
                ),
            }
    stats["rates"] = rates
    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def read_run_log_excerpt(pool_path: Path) -> Dict[str, Any]:
    """The decoder's OWN log lines, read from the RUN that produced this pool.

    Wf-A164 (2026-11-01): the report used to HARD-CODE two quoted lines of
    the default run's ``master.log`` ("...ACTIVE (518,276 params, cutoff=3.5
    A, ...)" and "[cfm-decoder] decision=threshold @ threshold=0.500 ...").
    Those were a transcription of ONE run's log -- true for the default pool
    and FALSE for every other pool, in an artefact that reports itself as a
    measurement.  They are now READ from the log sitting next to the pool
    this run actually read.  When that log is absent the excerpt says so and
    the decoder-rule premises fall back to the code defaults.
    """
    log_path = Path(pool_path).parent / "master.log"
    out: Dict[str, Any] = {
        "source": str(log_path),
        "present": False,
        "velocity_field_line": None,
        "decoder_line": None,
        "n_lines_scanned": 0,
    }
    if not log_path.is_file():
        return out
    out["present"] = True
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive
        return out
    lines = text.splitlines()
    out["n_lines_scanned"] = len(lines)
    for line in lines:
        stripped = line.strip()
        if out["velocity_field_line"] is None and stripped.startswith(
            "[cfm-velocity-field]"
        ):
            out["velocity_field_line"] = stripped
        if out["decoder_line"] is None and stripped.startswith("[cfm-decoder]"):
            out["decoder_line"] = stripped
    return out


def read_run_dir_listing(pool_path: Path) -> Dict[str, Any]:
    """What the pool's own run directory actually holds.

    Wf-A164 (2026-11-01): the report used to ASSERT that the run directory
    holds "only ``checkpoints/*.pt``, ``events.jsonl``, ``generated_smiles.csv``
    and ``master.log``" -- a claim about ONE run's directory, true for the
    default pool and not necessarily for another.  It is now LISTED from
    disk.
    """
    run_dir = Path(pool_path).parent
    out: Dict[str, Any] = {
        "path": str(run_dir),
        "present": run_dir.is_dir(),
        "entries": [],
    }
    if out["present"]:
        out["entries"] = sorted(p.name for p in run_dir.iterdir())
    return out


#: A number-like token: digits, optional thousands separators, optional
#: decimal part.
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

#: The artefact's PROSE paths.  Every leaf under these is prose; every other
#: leaf is a MACHINE field.  The self-consistency guard below checks that
#: every number rendered in the prose is backed by a machine field of the
#: SAME artefact.
#:
#: Wf-A164 (2026-11-01): the comparability caveat and the population
#: cross-check note are the two places where a pool-specific claim could be
#: transcribed as a literal.  Both are listed here.
NARRATIVE_PATHS: Tuple[str, ...] = (
    "comparability_caveat",
    "pool.population_cross_check.note",
)

#: A ``*.csv`` path inside a string.  The character before ``.csv`` must be
#: alphanumeric or ``_``, so a bare ``*.csv`` or a ``.csv`` in prose is NOT a
#: path and does not trip the scan.
_CSV_PATH_RE = re.compile(r"[A-Za-z0-9_.\-/]*[A-Za-z0-9_]\.csv")


class NarrativeSelfConsistencyError(RuntimeError):
    """Raised when a narrative number has no machine backing."""

    def __init__(self, artifact: Dict[str, Any], guard: Dict[str, Any]) -> None:
        super().__init__(
            "narrative self-consistency FAILED: unbacked numbers "
            f"{sorted(guard.get('unbacked_numbers', {}))}"
        )
        self.artifact = artifact
        self.guard = guard


def _leaf_map(artifact: Any) -> Dict[str, Any]:
    """Flatten an artefact to ``dotted.path -> leaf``."""
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

    A path may designate a LEAF (``comparability_caveat``), a MAPPING or a
    LIST.  The list form matters: with a naive ``path + "."`` prefix the list
    elements would never match, they would be silently classified as MACHINE
    fields, and every number inside them would go unchecked.  That hole was
    found by a negative control in Wf-A162 -- a guard that is never shown a
    failing case is not known to work.
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
    """
    pattern = re.compile(r"(?<![0-9.])" + re.escape(token) + r"(?![0-9.])")
    return any(pattern.search(text) for text in strings)


def _narrative_self_consistency(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """THE GUARD for the class of defect, not just this instance.

    Wf-A162 (2026-11-01) and Wf-A164 (2026-11-01): a measurement tool can
    hard-code POOL-SPECIFIC claims as string literals inside a ``MEASURED``
    artefact.  Every one of them is TRUE for the tool's DEFAULT pool and
    FALSE for any other.  The class is stopped here: every number rendered
    in a NARRATIVE string must be backed by a MACHINE field of the same
    artefact -- either as a standalone number inside a machine string, or
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


def _pool_path_scan(report: Dict[str, Any], own_pool_path: Path) -> Dict[str, Any]:
    """The artefact may name exactly ONE pool path: the one this run read.

    Wf-A164 (2026-11-01): the comparability caveat used to hard-code the
    DEFAULT pool's path, so an artefact produced against a different pool
    named a pool it never read.  This scan finds every ``*.csv`` path the
    artefact mentions and requires each of them to resolve EITHER to the pool
    this run actually read, or to a file that lives in the SAME run directory
    (the run directory legitimately holds sibling CSVs -- the c7 pool's
    directory holds ``per_epoch_metrics.csv`` next to ``generated_smiles.csv``
    -- and naming a sibling is not naming another pool).

    A path into ANY OTHER run directory is foreign.
    """
    own = Path(own_pool_path).resolve()
    run_dir = own.parent
    leaves = _leaf_map(report)
    foreign: Dict[str, List[str]] = {}
    siblings: Dict[str, List[str]] = {}
    for path, value in sorted(leaves.items()):
        if not isinstance(value, str):
            continue
        for cand in _CSV_PATH_RE.findall(value):
            cand = cand.strip()
            if not cand:
                continue
            p = Path(cand)
            if p.is_absolute():
                candidates = {p.resolve()}
            else:
                # A relative path may be written against the repo root or
                # against the pool's own run directory (a bare file name).
                candidates = {(REPO / p).resolve(), (run_dir / p).resolve()}
            if own in candidates:
                continue
            # a sibling file of the SAME run directory is not another pool
            if any(
                c != own and c.parent == run_dir and c.is_file() for c in candidates
            ):
                siblings.setdefault(cand, []).append(path)
                continue
            foreign.setdefault(cand, []).append(path)
    return {
        "rule": (
            "the artifact may name only the pool path it actually read (or a "
            "sibling file of the same run directory); every other *.csv path "
            "it mentions is foreign"
        ),
        "own_pool_path": str(own),
        "run_dir": str(run_dir),
        "foreign_pool_paths": {
            k: sorted(set(v)) for k, v in sorted(foreign.items())
        },
        "n_foreign_pool_paths": len(foreign),
        "sibling_run_dir_paths": {
            k: sorted(set(v)) for k, v in sorted(siblings.items())
        },
        "pass": not foreign,
    }


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def build_report(
    pool_path: Path,
    pool: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    tmqm_root: Path,
    lam_list: Sequence[float],
    conventions: Sequence[str],
    baseline_codes: Optional[List[str]],
) -> Dict[str, Any]:
    """Assemble the machine-readable report payload.

    Wf-A164 (2026-11-01): every number that describes OUR population is
    COMPUTED from the run, and the two guards are evaluated on the artefact
    this function returns.  The artefact is returned even when a guard
    fails -- the caller writes it out with ``status: FAIL`` and exits 1.
    """
    radii_probe = {}
    for z in (1, 6, 7, 8, 16, 17, 35, 44, 45, 75, 76, 77, 78):
        radii_probe[str(z)] = rcov(z)

    pool_path = Path(pool_path)
    n_rows = int(pool["n_rows"])
    rows_with_metal = int(pool["rows_with_metal"])
    metal_atoms_total = int(pool["metal_atoms_total"])
    rows_with_one_fragment = int(pool["fragments"].get(1, 0))

    # How the pool is NAMED in the prose: relative to the repo root when it
    # lives inside it, absolute otherwise.  Either way it is the pool THIS
    # run read -- never a transcribed default.
    try:
        pool_path_display = str(pool_path.resolve().relative_to(REPO))
    except ValueError:
        pool_path_display = str(pool_path.resolve())

    population_cross_check = {
        "rows_with_metal": rows_with_metal,
        "rows_with_metal_frac": (rows_with_metal / n_rows if n_rows else None),
        "rows_with_one_fragment": rows_with_one_fragment,
        "metal_atoms_total": metal_atoms_total,
        "metal_atoms_sharing_a_fragment_with_a_nonmetal": pool["metal_shares_fragment"],
        "metal_atoms_alone_in_their_fragment": pool["metal_alone"],
        "rows_containing_arrow": pool["rows_with_arrow"],
        "note": (
            "Computed by THIS tool from the same RDKit parse, so a reader can "
            "confirm the population this run actually measured: "
            f"{rows_with_one_fragment}/{n_rows} rows with one fragment, "
            f"{pool['metal_shares_fragment']}/{metal_atoms_total} metal atoms "
            "sharing a fragment with a non-metal, "
            f"{pool['metal_alone']}/{metal_atoms_total} metal atoms alone, "
            f"{pool['rows_with_arrow']}/{n_rows} rows containing '->'."
        ),
    }

    comparability_caveat = build_comparability_caveat(
        pool_path=pool_path,
        n_rows=n_rows,
        rows_with_metal=rows_with_metal,
        pool_path_display=pool_path_display,
        n_rows_parsed=int(pool["n_rows_parsed"]),
        rows_with_one_fragment=rows_with_one_fragment,
    )

    report: Dict[str, Any] = {
        "tool": "wfa156_bond_threshold_frontier",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu_used": False,
        "torch_imported": False,
        "pool_path": str(pool_path),
        "pool_path_display": pool_path_display,
        "pool_name": pool_path.parent.name,
        "pool_n_rows": n_rows,
        "constants": {
            "lambdas": list(lam_list),
            "conventions": list(conventions),
            "decoder_flat_cutoff_a": DECODER_FLAT_CUTOFF_A,
            "decoder_accept_threshold_p_bonded": DECODER_ACCEPT_THRESHOLD,
            "decoder_dative_max_distance_a": DECODER_DATIVE_MAX_DISTANCE_A,
            "covalent_radius_source": "rdkit.Chem.GetPeriodicTable().GetRcovalent",
            "decode_cohort_metals": list(DECODE_COHORT_METALS),
            "decode_cohort_metal_atomic_numbers": sorted(COHORT_METAL_Z),
            "pi": math.pi,
        },
        "rdkit_rcov_probe": radii_probe,
        "live_fetches": LIVE_FETCHES,
        "live_fetch_date": LIVE_FETCH_DATE,
        "related_scripts": {
            "repo_pi_convention": (
                "molmetal/scripts/wf_a155_ligand_connected_rate.py::"
                "radii_scaled_factor -- the repo's own implementation of the "
                "lambda * pi reading that this report measures against the "
                "paper's rule"
            ),
            "this_tool": "molmetal/scripts/wfa156_bond_threshold_frontier.py",
        },
        "frontier_band": FRONTIER_BAND,
        "frontier_band_quote": FRONTIER_BAND_QUOTE,
        "frontier_citations": dict(FRONTIER_CITATIONS),
        "n_comparability_breaks": 4,
        "decoder_run_log": read_run_log_excerpt(pool_path),
        "run_dir_listing": read_run_dir_listing(pool_path),
        "comparability_caveat": comparability_caveat,
        "comparability_caveat_corrected_wfa180":
            build_comparability_caveat_correction(
                corrected_caveat_text=comparability_caveat,
                superseded_caveat_text=build_comparability_caveat_superseded(
                    pool_path=pool_path,
                    n_rows=n_rows,
                    rows_with_metal=rows_with_metal,
                    pool_path_display=pool_path_display,
                ),
                n_rows_parsed=int(pool["n_rows_parsed"]),
                rows_with_one_fragment=rows_with_one_fragment,
            ),
        "comparability_caveat_reading_count_corrected_wfa196":
            build_reading_count_correction(
                n_rows_parsed=int(pool["n_rows_parsed"]),
                rows_with_one_fragment=rows_with_one_fragment,
            ),
        "pool": {
            "n_rows": pool["n_rows"],
            "n_rows_parsed": pool["n_rows_parsed"],
            "n_rows_unparsed": pool["n_rows_unparsed"],
            "n_bonds_total": pool["n_bonds_total"],
            "n_bonds_with_ref_length": pool["n_bonds_with_ref_length"],
            "n_bonds_without_ref_length": pool["n_bonds_without_ref_length"],
            "n_metal_bonds": pool["metal_bonds"],
            "population_cross_check": population_cross_check,
            "fragment_count_histogram": {str(k): v for k, v in sorted(pool["fragments"].items())},
            "element_pair_histogram": dict(pool["element_pairs"]),
            "bond_order_histogram": {str(k): v for k, v in sorted(pool["bond_orders"].items())},
            "rates": pool["rates"],
        },
        "baseline_tmqm": baseline,
        "baseline_tmqm_codes_available": len(baseline_codes) if baseline_codes is not None else None,
        "tmqm_root": str(tmqm_root),
    }

    # ---- the two guards, evaluated on THIS artefact ----------------------
    report["narrative_self_consistency"] = _narrative_self_consistency(report)
    report["pool_path_scan"] = _pool_path_scan(report, pool_path)
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the human-readable report.  The caveat is emitted FIRST."""
    pool = report["pool"]
    lines: List[str] = []
    A = lines.append

    A("# Wf-A156 -- EF-TALFM covalent-radius bond threshold applied to our CFM decode pool")
    A("")
    A(f"*Generated {report['generated_at']} | CPU-only (no torch, no GPU) | "
      f"tool `{report['tool']}`*")
    A("")
    A("## 0. COMPARABILITY CAVEAT (read before any number below)")
    A("")
    A("```")
    A(report["comparability_caveat"].rstrip())
    A("```")
    A("")
    _correction = report.get("comparability_caveat_corrected_wfa180")
    if _correction:
        for _ln in render_comparability_correction_section(_correction):
            A(_ln)
    _reading_count = report.get(
        "comparability_caveat_reading_count_corrected_wfa196"
    )
    if _reading_count:
        for _ln in render_reading_count_correction_section(_reading_count):
            A(_ln)

    A("## 1. The frontier rule, as the live fetch states it")
    A("")
    A(f"Fetched live on {report['live_fetch_date']} from the arXiv LaTeX source of EF-TALFM")
    A("(`https://arxiv.org/e-print/2609.08333`), `src/appendix.tex`:")
    A("")
    A("```latex")
    A(r"\STATE Select reconstruction policy $\pi \gets \texttt{ChoosePolicy}(\hat{\mathbf z}, \hat{\mathbf R}, \hat Q)$")
    A(r"\STATE Compute bond threshold $\tau_{ij} \gets \lambda_\pi \big(r^{\mathrm{cov}}(\hat z_i) + r^{\mathrm{cov}}(\hat z_j)\big)$")
    A("```")
    A("")
    A("So the paper's rule is **tau_ij = lambda_POLICY * (r_cov(z_i) + r_cov(z_j))**.")
    A("`lambda_POLICY` is the tolerance factor OF THE SELECTED POLICY `pi`; the `pi`")
    A("is a SUBSCRIPT, not the constant pi.  The branch brief's `lambda * pi * (...)`")
    A("-- and this repo's `yuelbond_decoder.py::radii_scaled_factor` -- read it as a")
    A("product with 3.14159.  Both are reported below (`paper` / `repo_pi`).")
    A("")
    A("The paper's numeric `lambda_POLICY` is **NOT recoverable**: the whole LaTeX")
    A("source contains the word `policy` exactly twice (both in Algorithm 4) and no")
    A("numeric value for `lambda_pi` anywhere.  The lambdas below are a SWEEP.")
    A("")
    A("## 2. Live fetch log")
    A("")
    A("| URL | tool | resolved | what came back |")
    A("|---|---|---|---|")
    for f in report["live_fetches"]:
        what = f["what"].replace("|", "\\|")
        A(f"| `{f['url']}` | {f['how']} | {'YES' if f['resolved'] else 'NO'} | {what} |")
    A("")

    A("## 3. The decoder's own rule (MEASURED from this run's `master.log`)"
      if (report.get("decoder_run_log", {}) or {}).get("present")
      else "## 3. The decoder's own rule (NO `master.log` next to this pool -- CODE DEFAULTS only)")
    A("")
    A("```")
    run_log = report.get("decoder_run_log", {}) or {}
    if run_log.get("present"):
        A(run_log.get("velocity_field_line") or "(no [cfm-velocity-field] line found)")
        A(run_log.get("decoder_line") or "(no [cfm-decoder] line found)")
    else:
        A(f"(NO master.log next to this pool: {run_log.get('source')})")
    A("```")
    A("")
    A(f"* geometric **candidacy** gate: flat `d_ij <= {DECODER_FLAT_CUTOFF_A} A`")
    A(f"* learned **acceptance**: `p(bonded) >= {DECODER_ACCEPT_THRESHOLD}` on the YuelBond head")
    A(f"* dative bonds additionally capped at `{DECODER_DATIVE_MAX_DISTANCE_A} A`")
    A("* the D2 connectivity prior may ADD bridging bonds, but only between pairs")
    A(f"  already inside the same {DECODER_FLAT_CUTOFF_A} A candidate gate (never invents a long bond)")
    A("")
    A(f"**Status of the {DECODER_FLAT_CUTOFF_A} A premise.**  It is the CODE DEFAULT, not an observed")
    A(f"value for this run: `YuelBondDecoderHead.cutoff` defaults to {DECODER_FLAT_CUTOFF_A} A")
    A("(`yuelbond_decoder.py:799`, applied as `mask = d_raw < self.cutoff` at :1244")
    A(f"and :1934) and `CFMYuelBondDecoder.distance_cutoff` defaults to {DECODER_FLAT_CUTOFF_A} A")
    A("(`cfm_decoder_yuelbond.py:1048`, assigned at :1172).")
    if run_log.get("present"):
        A(f"The run's master.log prints `cutoff={DECODER_FLAT_CUTOFF_A} A` only for the EGN VELOCITY FIELD; it does NOT print")
        A("the bond head's candidate cutoff, and no override is recorded in the run's")
        A("artefacts.")
    else:
        A("NO master.log was found next to this pool, so NOTHING about THIS run's")
        A("decoder configuration was observed: every value above is a CODE DEFAULT")
        A("and nothing more.")
    A("**Every T1/T2 number below is therefore CONDITIONAL on this")
    A(f"run having used the {DECODER_FLAT_CUTOFF_A} A default.**  If a different cutoff was used, the")
    A("T1 comparison point moves with it.")
    A("")
    A("**The decoder's geometry is NOT on disk.**  The pool CSV header is")
    A("`epoch,smiles,n_frags_raw,largest_frag_frac` (writer:")
    A("`molmetal/scripts/train_cfm_yuelbond_wasserstein.py:19250`) and the run")
    run_dir = report.get("run_dir_listing", {}) or {}
    if run_dir.get("present"):
        A(f"directory holds no coordinate dump -- only {', '.join(run_dir.get('entries') or [])}.")
    else:
        A("directory is not present next to this pool at all.")
    A("So the literal \"would the emitted")
    A("bond survive tau_ij in the decoder's OWN geometry\" is **not measurable**")
    A("here.  Three tiers are reported instead, each labelled.")
    A("")

    A("## 4. The pool (MEASURED)")
    A("")
    A(f"* rows read: **{pool['n_rows']}**; parsed by RDKit: **{pool['n_rows_parsed']}**; "
      f"unparsed: **{pool['n_rows_unparsed']}**")
    A(f"* emitted bonds recovered from the SMILES: **{pool['n_bonds_total']}**")
    A(f"* of which metal-incident: **{pool['n_metal_bonds']}**")
    A(f"* bonds with a tmQM-DFT empirical reference length: "
      f"**{pool['n_bonds_with_ref_length']}** (missing: {pool['n_bonds_without_ref_length']})")
    A(f"* fragment-count histogram (RDKit, over parsed rows): `{pool['fragment_count_histogram']}`")
    A(f"* bond-order histogram: `{pool['bond_order_histogram']}`")
    A("")
    pcc = pool["population_cross_check"]
    A("**Population cross-check** -- computed by THIS tool from the same parse, so a")
    A("reader can confirm it sees the population this run actually measured:")
    A("")
    A(f"* rows carrying a metal: **{pcc['rows_with_metal']}/{pool['n_rows']}** "
      f"(= {_fmt(pcc['rows_with_metal_frac'])} metal_ratio)")
    A(f"* metal atoms sharing a fragment with a non-metal: "
      f"**{pcc['metal_atoms_sharing_a_fragment_with_a_nonmetal']}**")
    A(f"* metal atoms ALONE in their fragment: "
      f"**{pcc['metal_atoms_alone_in_their_fragment']}**")
    A(f"* rows containing `->`: **{pcc['rows_containing_arrow']}**")
    A(f"* rows with exactly one fragment (RDKit): "
      f"**{pool['fragment_count_histogram'].get('1', 0)}**")
    A("")
    A("### 4.1 T1 -- EXACT, geometry-free: is tau_ij stricter or looser than the")
    A(f"### decoder's own {DECODER_FLAT_CUTOFF_A} A gate, per emitted bond?")
    A("")
    A(f"| convention | lambda | frac of emitted bonds where tau < {DECODER_FLAT_CUTOFF_A} A (frontier STRICTER) | n stricter | n looser |")
    A("|---|---|---|---|---|")
    for key, r in pool["rates"].items():
        A(f"| {r['convention']} | {r['lambda']} | {_fmt(r['t1_frac_stricter_than_decoder_cutoff'])} "
          f"| {r['t1_n_stricter_than_decoder_cutoff']} | {r['t1_n_looser_than_decoder_cutoff']} |")
    A("")
    A("### 4.2 T2 -- EXACT, geometry-free bounds")
    A("")
    A("| convention | lambda | frac guaranteed SURVIVE | frac guaranteed FAIL | frac undecidable |")
    A("|---|---|---|---|---|")
    for key, r in pool["rates"].items():
        A(f"| {r['convention']} | {r['lambda']} | {_fmt(r['t2_frac_guaranteed_survive'])} "
          f"| {_fmt(r['t2_frac_guaranteed_fail'])} | {_fmt(r['t2_frac_undecidable'])} |")
    A("")
    A("### 4.3 T3 -- LENGTH-TRANSFERRED ESTIMATE (pool bonds scored at the")
    A("### tmQM-DFT empirical MEAN length for their element pair)")
    A("")
    A("**This is an ESTIMATE for the pool**, not a measurement of the decoder's")
    A("geometry.  It assumes each emitted bond has the same length as the average")
    A("real tmQM bond of its element pair and order.")
    A("")
    A("| convention | lambda | frac surviving (transferred) | n survived | n failed | denominator |")
    A("|---|---|---|---|---|---|")
    for key, r in pool["rates"].items():
        A(f"| {r['convention']} | {r['lambda']} | {_fmt(r['t3_frac_survived_ref_length'])} "
          f"| {r['t3_n']['survived']} | {r['t3_n']['failed']} | {r['t3_denominator']} |")
    A("")

    base = report.get("baseline_tmqm")
    A("## 5. BASELINE -- the tmQM ground truth, on REAL DFT geometry")
    A("")
    if not base:
        A("**NOT MEASURED** -- the tmQM baseline could not be built in this run.")
    else:
        A("This side has NO transfer and NO estimate: every distance is a real")
        A("interatomic distance in a real tmQM DFT geometry, and every bond is a")
        A("real Wiberg bond.")
        A("")
        A("**The pool's OWN row-matched ground-truth SMILES are NOT available.**  The")
        A("pool CSV carries no source-complex column, so no pool row can be joined")
        A("back to its tmQM parent.  The baseline below is therefore the tmQM")
        A(f"GROUND-TRUTH POPULATION (same {len(DECODE_COHORT_METALS)} metals as the decode cohort), not a")
        A("row-matched twin of the pool -- a DIFFERENT population, compared by")
        A("chemistry, not by row.")
        A("")
        A("**The baseline's \"bond\" is a Wiberg bond with BO > 0.1** (the")
        A("Housecroft convention, the same one this repo's own")
        A("`molmetal/data/bond_length_prior.py` uses).  That definition includes")
        A("long, weak contacts, which is why a non-trivial slice of them is longer")
        A("than any tau in the sweep (see 5.3).")
        A("")
        A(f"* complexes used: **{base['n_complexes_with_coordinates']}** "
          f"(of {base['n_complexes_requested']} requested)")
        A(f"* real bonds scored: **{base['n_bonds_total']}**")
        A(f"* heavy-atom candidate pairs inside the decoder's {DECODER_FLAT_CUTOFF_A} A gate: "
          f"**{base['n_candidate_pairs_total']}**")
        if base.get("bond_distance_stats"):
            s = base["bond_distance_stats"]
            A(f"* real bond distance distribution: min {s['min']:.3f} A / p01 "
              f"{s['p01']:.3f} A / median {s['median']:.3f} A / p99 {s['p99']:.3f} A / "
              f"max {s['max']:.3f} A")
        A("")
        A("### 5.1 What fraction of REAL ground-truth bonds survives tau_ij?")
        A("")
        A("| convention | lambda | frac of real bonds surviving | n surviving | n failing |")
        A("|---|---|---|---|---|")
        for key, r in base["rates"].items():
            A(f"| {r['convention']} | {r['lambda']} | "
              f"{_fmt(r['baseline_frac_true_bonds_surviving_tau'])} | "
              f"{r['baseline_n_true_bonds_surviving']} | {r['baseline_n_true_bonds_failing']} |")
        A("")
        A("### 5.2 Candidate-set comparison on the SAME real geometry")
        A("")
        A(f"Among all heavy-atom pairs the decoder's flat {DECODER_FLAT_CUTOFF_A} A gate would admit,")
        A("the fraction the frontier's element-aware tau_ij would ALSO admit:")
        A("")
        A("| convention | lambda | frac of decoder-candidate pairs accepted by tau | n accepted | n rejected |")
        A("|---|---|---|---|---|")
        for key, r in base["rates"].items():
            A(f"| {r['convention']} | {r['lambda']} | "
              f"{_fmt(r['candidate_set_frac_accepted_by_tau'])} | "
              f"{r['candidate_set_n_accepted']} | {r['candidate_set_n_rejected']} |")
        A("")
        A("### 5.3 WHICH real bonds the frontier rule rejects (top element pairs)")
        A("")
        A("| convention | lambda | top failing element pairs (count) |")
        A("|---|---|---|")
        for key, r in base["rates"].items():
            tops = r.get("top_failing_element_pairs") or {}
            txt = ", ".join(f"{k}:{v}" for k, v in list(tops.items())[:6]) or "(none)"
            A(f"| {r['convention']} | {r['lambda']} | {txt} |")
        A("")

    A("## 6. What this report does NOT establish")
    A("")
    A("* It does NOT measure the decoder's own geometry: the pool has no")
    A("  coordinates, so the emitted bonds' real distances are unknown.")
    A("* It does NOT reproduce, approximate or claim the frontier's Connected")
    A("  92.98 % / Sanitized 98.25 %; the four comparability breaks in section 0")
    A("  are unrepaired by any number above.")
    A("* It does NOT establish that adopting tau_ij would reduce fragmentation:")
    A("  no decode was run (0-GPU branch).  Every 'would' here is arithmetic on")
    A("  a threshold, not an outcome.")
    A("* It does NOT recover the paper's `lambda_POLICY`; the value is absent from")
    A("  the paper's own source, so the sweep is a sweep.")
    A("* It does NOT resolve the `lambda * pi` misreading in the repo's Wf-A155")
    A("  `radii_scaled_factor`; that file is owned by another wave and was READ")
    A("  ONLY here.  Both conventions are reported so the reader can see the")
    A("  3.14159x gap rather than inheriting one of them silently.")
    A("* It does NOT provide a ROW-MATCHED ground truth for the pool: the pool")
    A("  CSV has no source-complex column, so the baseline is the tmQM")
    A(f"  GROUND-TRUTH POPULATION (same {len(DECODE_COHORT_METALS)} metals), compared by chemistry only.")
    A(f"* It does NOT independently confirm the {DECODER_FLAT_CUTOFF_A} A candidate gate FOR THIS RUN:")
    A("  that is the CODE DEFAULT, and the run's log prints a cutoff only for")
    A("  the EGN velocity field.  The T1/T2 numbers are conditional on it.")
    A("* It does NOT treat the baseline's Wiberg BO > 0.1 bond definition as a")
    A("  chemical bond census: it includes long weak contacts, which is why some")
    A("  real bonds fail every tau in the sweep.")
    A("")
    A("## 7. Reproduction")
    A("")
    A("```bash")
    A("CUDA_VISIBLE_DEVICES=\"\" /home/hugo/codes/mol-metal/.venv/bin/python \\")
    A("    molmetal/scripts/wfa156_bond_threshold_frontier.py")
    A("```")
    A("")
    A("Outputs in this directory:")
    A("")
    A("* `report.json` -- the full machine-readable payload (git-ignored by")
    A("  `.gitignore`'s `molmetal/reports/**/*.json` rule)")
    A("* `report.md` -- this file (tracked)")
    A("* `per_row.csv` -- the compact, TRACKED per-row survival table: one line")
    A("  per pool row with its fragment structure and its own survival fractions")
    A("* `per_row.json` -- the FULL per-row record including every emitted bond's")
    A("  element pair, reference length and tau at every (convention, lambda)")
    A("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _render_per_row_csv(per_row: Sequence[Dict[str, Any]]) -> str:
    """Compact, TRACKED per-row survival table.

    One line per pool row: the row's identity, its fragment structure, and the
    fraction of ITS OWN emitted bonds that survive the frontier threshold at
    every (convention, lambda) cell -- the length-transferred estimate (T3)
    and the geometry-free guaranteed-survival bound (T2).

    ``.csv`` rather than ``.jsonl`` on purpose: ``.gitignore`` covers
    ``molmetal/reports/**/*.json`` but NOT ``*.jsonl``, so a JSONL of this size
    would land in git.  The full per-bond record goes to ``per_row.json``,
    which is ignored.
    """
    import io

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    header = [
        "row_index",
        "epoch",
        "n_frags_raw",
        "largest_frag_frac",
        "parsed",
        "n_fragments_rdkit",
        "n_bonds",
    ]
    cells: List[str] = []
    for conv in CONVENTIONS:
        for lam in LAMBDAS:
            cells.append(f"t3_frac_{conv}_lam{lam}")
            cells.append(f"t2_frac_guaranteed_survive_{conv}_lam{lam}")
    header.extend(cells)
    header.append("smiles")
    writer.writerow(header)
    for rec in per_row:
        line = [
            rec.get("row_index"),
            rec.get("epoch"),
            rec.get("n_frags_raw"),
            rec.get("largest_frag_frac"),
            int(bool(rec.get("parsed"))),
            rec.get("n_fragments_rdkit"),
            rec.get("n_bonds"),
        ]
        nb = rec.get("n_bonds") or 0
        for conv in CONVENTIONS:
            for lam in LAMBDAS:
                key = f"{conv}|{lam}"
                t3 = rec.get("t3", {}).get(key, {})
                t2 = rec.get("t2", {}).get(key, {})
                den = t3.get("survived", 0) + t3.get("failed", 0)
                line.append(
                    "" if den == 0 else f"{t3.get('survived', 0) / den:.6f}"
                )
                line.append(
                    "" if nb == 0
                    else f"{t2.get('guaranteed_survive', 0) / nb:.6f}"
                )
        line.append(rec.get("smiles"))
        writer.writerow(line)
    return buf.getvalue()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        default=str(DEFAULT_POOL),
        help="pool CSV (default: the c8 fullgen pool named in the branch brief)",
    )
    parser.add_argument(
        "--tmqm-root",
        default=str(DEFAULT_TMQM_ROOT),
        help="tmQM corpus root (real DFT geometries + Wiberg bond orders)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO / "molmetal" / "reports" / "wfa156_bond_threshold_frontier"),
        help="report output directory (non-published tree)",
    )
    parser.add_argument(
        "--baseline-complexes",
        type=int,
        default=DEFAULT_BASELINE_COMPLEXES,
        help="number of tmQM complexes in the real-geometry baseline population",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="skip the tmQM baseline (the pool tiers still run)",
    )
    args = parser.parse_args(argv)

    pool_path = Path(args.pool)
    tmqm_root = Path(args.tmqm_root)
    out_dir = Path(args.out_dir)

    if not pool_path.is_file():
        print(f"FAIL: pool CSV not found: {pool_path}")
        return 1

    rows = read_pool(pool_path)
    ref_lengths = load_ref_lengths(tmqm_root)
    pool = analyse_pool(rows, ref_lengths)

    baseline = None
    codes: Optional[List[str]] = None
    if not args.skip_baseline:
        codes = _decode_cohort_codes(tmqm_root, DECODE_COHORT_METALS)
        if codes:
            baseline = tmqm_baseline(
                tmqm_root,
                codes,
                max_complexes=int(args.baseline_complexes),
            )
        else:
            print("WARN: no tmQM codes resolved for the decode-cohort metals; baseline skipped")

    report = build_report(
        pool_path=pool_path,
        pool=pool,
        baseline=baseline,
        tmqm_root=tmqm_root,
        lam_list=LAMBDAS,
        conventions=CONVENTIONS,
        baseline_codes=codes,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Wf-A164: the guards decide whether this artefact may be written --
    guard = report.get("narrative_self_consistency", {}) or {}
    path_scan = report.get("pool_path_scan", {}) or {}
    if not guard.get("pass") or not path_scan.get("pass"):
        bad = dict(report)
        bad["status"] = "FAIL"
        bad["failure"] = (
            "the self-consistency guards rejected this artifact: "
            f"unbacked_numbers={guard.get('unbacked_numbers')}, "
            f"foreign_pool_paths={path_scan.get('foreign_pool_paths')}"
        )
        atomic_write_file(
            out_dir / "report.json", json.dumps(bad, indent=2, sort_keys=True) + "\n"
        )
        print("FAIL: the narrative/pool-path guards rejected this artifact", file=sys.stderr)
        for tok, paths in sorted((guard.get("unbacked_numbers") or {}).items()):
            print(f"  unbacked {tok!r} in {paths}", file=sys.stderr)
        for cand, paths in sorted((path_scan.get("foreign_pool_paths") or {}).items()):
            print(f"  foreign pool path {cand!r} in {paths}", file=sys.stderr)
        return 1

    report["status"] = "PASS"
    atomic_write_file(out_dir / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_write_file(out_dir / "report.md", render_markdown(report))
    per_row = pool.pop("_per_row")
    # ``per_row.json`` carries the FULL per-row record INCLUDING the per-bond
    # tau table.  It is deliberately a ``.json`` because ``.gitignore`` covers
    # ``molmetal/reports/**/*.json`` and NOT ``*.jsonl`` -- a tracked 56 MB
    # JSONL would be a mistake.  ``per_row.csv`` is the compact, TRACKED
    # per-row survival table.
    atomic_write_file(
        out_dir / "per_row.json",
        json.dumps({"rows": per_row}, separators=(",", ":"), sort_keys=True) + "\n",
    )
    atomic_write_file(out_dir / "per_row.csv", _render_per_row_csv(per_row))

    # ---- stdout summary -------------------------------------------------
    print(f"Wf-A156 bond-threshold frontier measurement (CPU-only)")
    print(f"pool: {pool_path}")
    print(f"  rows={pool['n_rows']} parsed={pool['n_rows_parsed']} "
          f"bonds={pool['n_bonds_total']} metal_bonds={pool['metal_bonds']}")
    for key, r in pool["rates"].items():
        print(
            f"  POOL {r['convention']:>8} lambda={r['lambda']}: "
            f"T1_stricter_than_decoder_{DECODER_FLAT_CUTOFF_A}A={_fmt(r['t1_frac_stricter_than_decoder_cutoff'])} "
            f"T2_survive={_fmt(r['t2_frac_guaranteed_survive'])} "
            f"T2_fail={_fmt(r['t2_frac_guaranteed_fail'])} "
            f"T2_undecidable={_fmt(r['t2_frac_undecidable'])} "
            f"T3_transferred={_fmt(r['t3_frac_survived_ref_length'])}"
        )
    print(
        f"  GUARDS: narrative_self_consistency="
        f"{'PASS' if guard.get('pass') else 'FAIL'} "
        f"({guard.get('n_prose_numbers')} prose numbers, "
        f"{len(guard.get('unbacked_numbers') or {})} unbacked) | "
        f"pool_path_scan={'PASS' if path_scan.get('pass') else 'FAIL'} "
        f"({path_scan.get('n_foreign_pool_paths')} foreign pool paths)"
    )
    if baseline:
        print(f"  BASELINE tmQM: complexes={baseline['n_complexes_with_coordinates']} "
              f"bonds={baseline['n_bonds_total']} "
              f"cand_pairs={baseline['n_candidate_pairs_total']}")
        for key, r in baseline["rates"].items():
            print(
                f"  BASE {r['convention']:>8} lambda={r['lambda']}: "
                f"true_bonds_surviving={_fmt(r['baseline_frac_true_bonds_surviving_tau'])} "
                f"cand_pairs_accepted={_fmt(r['candidate_set_frac_accepted_by_tau'])}"
            )
    print(f"wrote: {out_dir / 'report.json'}")
    print(f"wrote: {out_dir / 'report.md'}")
    print(f"wrote: {out_dir / 'per_row.json'}")
    print(f"wrote: {out_dir / 'per_row.csv'}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
