#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A158 BRANCH FRONT -- the frontier post-processing comparability matrix.

WHAT THIS TOOL IS
-----------------
The repo compares our CFM decode's connectivity rate against a published
frontier band of **92.7-94 % connected**.  The main thread MEASURED that four
things break that comparison (DENOMINATOR / METRIC / POST-PROCESSING /
HYDROGENS).  Breaks 1, 2 and 4 were established from our own side.  **Break 3
(POST-PROCESSING) had never been established from the SOURCES** -- nobody had
read what the frontier papers actually DO to a molecule before they count it
as "connected".

This tool is that reading, done from LIVE FETCHES on 2026-11-01 and embedded
here as a literal constant so the script is deterministic and offline
re-runnable.  It emits ONE row per source and ONE column per break; each cell
carries ``{verdict, quote, locator}``:

    BREAKS           the source's protocol differs from ours on this axis
    DOES-NOT-BREAK   the two protocols agree on this axis
    NOT-STATED       the source does not state this at all
    NOT-RETRIEVED    we could not retrieve this (e.g. a bot-challenge page)

WHAT THIS TOOL IS NOT
---------------------
It is **NOT** a scoreboard.  It does not rank our model against the frontier
and it does not conclude that "our model is bad" or that "the frontier is
cheating".  A published validity band is a property of a *protocol*; the only
question this matrix answers is whether the two protocols are the same
measurement.  If a frontier paper does not state its post-processing, that is
a first-class result recorded as ``NOT-STATED``, not a gap to be papered over.

VERDICT SEMANTICS ARE RELATIVE TO OUR SIDE
-------------------------------------------
The anchor is our own RAW measurement, recorded in :data:`ANCHOR`:

    metrics/by_metric/cfm_fragmentation_audit.json
        /blind_validity_signature/3/single_fragment_ratio  =  0.08509630565203663

i.e. RDKit connected-component count on the RAW emitted SMILES, no
post-processing.  Every ``BREAKS`` verdict in this matrix therefore means
"the two protocols are not the same measurement", NOT "the frontier is wrong".

SOURCES
-------
* EF-TALFM  -- arXiv 2609.08333v4 (3D, PCQM4Mv2, unconditional)
* FlowMol3  -- arXiv 2508.12629v1 (3D, GEOM-Drugs, unconditional) + the
               authors' own public evaluation harness (github.com/dunni3/FlowMol)
* GGFlow    -- arXiv 2411.05676v1 (2D molecular GRAPHS, QM9 / ZINC250k)
* PAFlow    -- arXiv 2509.01486v2 (3D, CrossDocked2020 SBDD) -- establishes
               whether a 4th graph flow-matching paper reports such a band at all

CLI
---
::

    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wfa158_frontier_postproc_matrix.py

    # re-verify the embedded quotes against locally saved extractions
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wfa158_frontier_postproc_matrix.py \\
        --verify-quotes-from /tmp/wfa158

Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP (a requested check could not run).
CPU-only by construction: no torch, no GPU, no checkpoint, no network unless
``--verify-quotes-from`` is given (and even then it only reads local files).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETRIEVAL_DATE = "2026-11-01"

BREAKS: tuple = ("DENOMINATOR", "METRIC", "POST-PROCESSING", "HYDROGENS")
VERDICTS: tuple = ("BREAKS", "DOES-NOT-BREAK", "NOT-STATED", "NOT-RETRIEVED")
#: verdicts that ARE the explicit "we did not find it" markers
MARKER_VERDICTS: tuple = ("NOT-STATED", "NOT-RETRIEVED")
#: sub-verdicts for "is the RECIPE itself described?" (the branch's core question)
RECIPE_VERDICTS: tuple = ("STATED", "NOT-STATED", "NOT-APPLICABLE")

DEFAULT_OUT_DIR = REPO / "molmetal" / "reports" / "wfa158_frontier_postproc_matrix"

#: The DATED SIBLING FILE that carries the byte evidence for BOTH
#: artefacts.  It exists because a file cannot contain its own hash: the
#: markdown's own size and md5 cannot be quoted inside the markdown.  It is
#: written BY the producer, from the same matrix, so it cannot drift.
PARITY_EVIDENCE_FILENAME = "frontier_matrix.parity_2026_11_01.json"

# ---------------------------------------------------------------------------
# The dated correction sections of the markdown artefact
# ---------------------------------------------------------------------------
#
# These two sections were APPENDED to `frontier_matrix.md` by Wf-A177 and by
# Wf-A180 (both 2026-11-01).  They are the record of the contest over the
# band's provenance, and AGENTS.md §5.5 rule #6 requires them to be
# PRESERVED VERBATIM.  They are EMITTED here -- EXTRACTED from the artefact
# on disk, never retyped -- so that a re-run of this producer reproduces
# them byte-for-byte instead of deleting them.
#
# Both sections are quoted by the third section (the producer/record
# reconciliation), which is rendered from the record's own dated key.
WF_A177_MD_SECTION = (
    "## Wf-A177 CORRECTION (2026-11-01) — the band reading is CONTESTED on disk\n"
    "\n"
    "This section is APPENDED.  Per AGENTS.md §5.5 rule #6 the superseded renderings\n"
    "above are PRESERVED VERBATIM and this correction is ADDITIVE + DATED: nothing\n"
    "above this line was edited, removed, re-rendered or re-ordered.\n"
    "\n"
    "**Superseded rendering 1 -- `frontier_matrix.md` (section \"Where the repo's band\n"
    "comes from\" -> `band_provenance` -> `finding`), VERBATIM:**\n"
    "\n"
    "> * finding: THE BAND IS NOT A CONNECTIVITY BAND. Its two endpoints are a VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %); a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE CLASSIFICATION F1. None of the three is a connected-component count. Our 8.51 % is a connected-component count on the RAW emitted SMILES. The comparison the band invites is therefore not merely apples-to-oranges -- it is a comparison of two different observables, and the post-processing axis is a THIRD, separate axis on top of it.\n"
    "\n"
    "**Superseded rendering 2 -- `frontier_matrix.md` (section \"HLTF\" -> \"METRIC --\n"
    "BREAKS\" -> note), VERBATIM:**\n"
    "\n"
    "> THE BAND'S UPPER ENDPOINT IS NOT A CONNECTIVITY NUMBER AT ALL. Table 1(b) reports SIX observables -- atom stability (AS), molecule stability (MS), RDKit validity (Val), Validity&Uniq, Novelty and PoseBusters validity (PB-Valid) -- and the 94 +/- 0.5 % figure is PB-Valid, a SIX-CHECK POSE VERDICT (bond lengths/angles, steric clash, aromatic and double-bond flatness, internal energy). The LOWER endpoint (92.7 %) is GeoLDM's Validity&Uniq -- a sanitisation-and-duplication rate. NEITHER is a connected-component count. Ours is a connected-component count and nothing else. This is a total METRIC break, not a partial one.\n"
    "\n"
    "**THE CORRECTION (Wf-A177, 2026-11-01).** The published 92.7-94 % band is EXTERNAL to\n"
    "Mol-Metal (it is a FRONTIER composite) and it is NOT one homogeneous quantity.\n"
    "The repo holds TWO COMPETING provenance readings for which cells it names, and\n"
    "they do not even agree on the numerals: (A) `frontier_matrix.json` ->\n"
    "`band_provenance` (:51-89): GeoLDM's Validity & Uniq 92.7 % and HLTF's PB-Valid\n"
    "94.0 % on QM9, both `post_processed: false` (plus YuelBond's per-bond-type\n"
    "classification F1 92.7 % as a third candidate for the same numeral) -- this\n"
    "reading reproduces the quoted numerals exactly but makes NO endpoint a\n"
    "connected-component count; (B) the same file -> `frontier_axis_mapping` (:360,\n"
    ":378-379), EF-TALFM arXiv:2609.08333v4: the band mixes EF-TALFM's OWN Table 3(a)\n"
    "raw connectivity cell (C. = 92.98 %, counted BEFORE the largest-component\n"
    "retention) with its post-processed N. & U. & S. companion cell (94.33 %) -- this\n"
    "reading has the connectivity+post shape but does NOT reproduce the quoted\n"
    "numerals (92.98 / 94.33). Under BOTH readings the band is external, is not one\n"
    "homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a\n"
    "connected-component count on the RAW emitted SMILES, and nothing else. The exact\n"
    "source cells of the band are NOT settled on disk.\n"
    "\n"
    "**The only claims that may be published (they hold under BOTH readings):**\n"
    "\n"
    "1. The published 92.7-94 % band is **EXTERNAL** to Mol-Metal -- it is a FRONTIER\n"
    "   composite, not our measurement.\n"
    "2. It is **NOT one homogeneous quantity**.\n"
    "3. It is **NOT on our scale** -- OUR 8.51 % IS a connected-component count on\n"
    "   the RAW emitted SMILES, and nothing else.\n"
    "4. The **source cells of the band are NOT settled on disk**.\n"
    "\n"
    "**What this correction does NOT change.** The per-source protocol breaks\n"
    "recorded above (DENOMINATOR / METRIC / POST-PROCESSING / HYDROGENS per paper)\n"
    "are unaffected: they compare the frontier's STATED protocol against ours, not\n"
    "the band's provenance. The definition of OUR anchor (8.51 %, RDKit\n"
    "connected-component count on the RAW emitted SMILES, no post-processing) is\n"
    "unaffected. What is superseded is the reading that the band's source cells are\n"
    "SETTLED, and the rendering that asserted \"NONE of the three is a\n"
    "connected-component count\" / \"NEITHER is a connected-component count\" as fact\n"
    "rather than as one reading among two.\n"
    "\n"
    "**Machine-readable record:** the same contest is recorded, ADDITIVELY, under\n"
    "`band_provenance` -> `competing_reading_and_what_this_does_not_settle` in\n"
    "`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, and in\n"
    "the SOURCE of the two producers\n"
    "(`molmetal/scripts/wfa158_frontier_postproc_matrix.py`).\n"
    "\n"
    "**Locator drift note (added by Wf-A177, 2026-11-01).** The line ranges cited in the\n"
    "canonical statement above -- `:51-89` (Reading A) and `:360, :378-379` (Reading B) --\n"
    "are the PRE-Wf-A177 line numbers of `frontier_matrix.json`.  The Wf-A177 key added at\n"
    "`:52` of that file shifted every line at or after the old `:52` by +48.  Current\n"
    "positions on disk: Reading A -> `band_provenance` `:50`, `band_as_quoted_in_the_repo`\n"
    "`:51`, `endpoint_candidates` `:100-`, `headline_finding` `:137`; Reading B -> `sources`\n"
    "`:380` -> `EF-TALFM` `:381`, the EF-TALFM `POST-PROCESSING` note `:408` (the\n"
    "`'C. = 92.98 %'` cell), `extra_observations` `:426-427` (the\n"
    "`'92.7-94 % band ... mixes the LOWEST connectivity'` sentence).  The ROBUST locators\n"
    "are the JSON PATHS -- `band_provenance` and `sources`.`EF-TALFM` -- not the line\n"
    "numbers.\n"
    "\n"
)

WF_A180_MD_SECTION = (
    "## Wf-A180 CORRECTION (2026-11-01) — the contested renderings are SUPERSEDED (reading-safe)\n"
    "\n"
    "This section is APPENDED.  Per AGENTS.md §5.5 rule #6 the superseded\n"
    "renderings above are PRESERVED VERBATIM and this correction is ADDITIVE +\n"
    "DATED: nothing above this line was edited, removed, re-rendered or re-ordered.\n"
    "The pre-Wf-A180 bytes of this file are a byte-identical PREFIX of the file\n"
    "that now carries this section.\n"
    "\n"
    "**Locator defect, corrected (additively).** The Wf-A177 section above points\n"
    "Reading B at `frontier_axis_mapping`.  That token is **NOT a key of\n"
    "`frontier_matrix.json`** — it survives only inside a string value, so\n"
    "pointing a reader at it is a DEFECT, not a locator.  The ROBUST locators are\n"
    "the JSON PATHS `band_provenance` and `sources`.`EF-TALFM` — as this file's\n"
    "own \"Locator drift note\" above already said.  The two copies of that pointer\n"
    "inside `frontier_matrix.json` are corrected in place, each with the\n"
    "superseded string preserved VERBATIM in a dated sibling key.\n"
    "\n"
    "**Superseded rendering 1 — `band_provenance`.`headline_finding`, VERBATIM:**\n"
    "\n"
    "> THE BAND IS NOT A CONNECTIVITY BAND. Its two endpoints are a VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %); a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE CLASSIFICATION F1. None of the three is a connected-component count. Our 8.51 % is a connected-component count on the RAW emitted SMILES. The comparison the band invites is therefore not merely apples-to-oranges -- it is a comparison of two different observables, and the post-processing axis is a THIRD, separate axis on top of it.\n"
    "\n"
    "**Superseded rendering 2 — `sources`.`HLTF`.`cells`.`METRIC`.`note`, VERBATIM:**\n"
    "\n"
    "> THE BAND'S UPPER ENDPOINT IS NOT A CONNECTIVITY NUMBER AT ALL. Table 1(b) reports SIX observables -- atom stability (AS), molecule stability (MS), RDKit validity (Val), Validity&Uniq, Novelty and PoseBusters validity (PB-Valid) -- and the 94 +/- 0.5 % figure is PB-Valid, a SIX-CHECK POSE VERDICT (bond lengths/angles, steric clash, aromatic and double-bond flatness, internal energy). The LOWER endpoint (92.7 %) is GeoLDM's Validity&Uniq -- a sanitisation-and-duplication rate. NEITHER is a connected-component count. Ours is a connected-component count and nothing else. This is a total METRIC break, not a partial one.\n"
    "\n"
    "**Superseded rendering 3 — `band_provenance`.`what_this_does_not_settle`, VERBATIM:**\n"
    "\n"
    "> This branch did NOT establish which paper the repo's band was assembled from, because the band was assembled by a literature scan whose selection rule is not recorded on disk. It DID establish, from the primary sources, that every candidate endpoint of the band is a validity / uniqueness / pose / bond-classification number rather than a connectivity rate -- and that the frontier's own literature publishes the post-processing effect as being worth up to 95 percentage points.\n"
    "\n"
    "**Why these three are superseded.** Rendering 1 asserts \"THE BAND IS NOT A\n"
    "CONNECTIVITY BAND\" / \"None of the three is a connected-component count\";\n"
    "rendering 2 asserts \"NEITHER is a connected-component count\" / \"This is a\n"
    "total METRIC break, not a partial one\"; rendering 3 asserts that EVERY\n"
    "candidate endpoint of the band is \"rather than a connectivity rate\".  All\n"
    "three are TRUE under Reading A and FALSE under Reading B — under Reading B the\n"
    "band's LOWER endpoint IS EF-TALFM's RAW connectivity cell\n"
    "`C. = 92.98 %` — so none of them may be published as settled.\n"
    "\n"
    "**THE READING-SAFE STATEMENT (replaces all three):**\n"
    "\n"
    "> The published band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.\n"
    "\n"
    "**The only claims that may be published (they hold under BOTH readings):**\n"
    "\n"
    "1. The published 92.7-94 % band is **EXTERNAL** to Mol-Metal — it is a\n"
    "   FRONTIER composite, not our measurement.\n"
    "2. It is **NOT one homogeneous quantity**.\n"
    "3. It is **NOT on our scale** — OUR 8.51 % IS a connected-component count on\n"
    "   the RAW emitted SMILES, and nothing else.\n"
    "4. The **source cells of the band are NOT settled on disk**.\n"
    "\n"
    "**What this correction does NOT change.** The per-source protocol breaks\n"
    "recorded above (DENOMINATOR / METRIC / POST-PROCESSING / HYDROGENS per paper)\n"
    "are unaffected: they compare the frontier's STATED protocol against ours, not\n"
    "the band's provenance.  The definition of OUR anchor (8.51 %, RDKit\n"
    "connected-component count on the RAW emitted SMILES, no post-processing) is\n"
    "unaffected.  The Wf-A177 section above is itself unaffected — it is the\n"
    "record of the contest, and its \"Locator drift note\" already named the JSON\n"
    "PATHS as the robust locators.\n"
    "\n"
    "**Producer state.** The producer\n"
    "(`molmetal/scripts/wfa158_frontier_postproc_matrix.py`) now EMITS the\n"
    "reading-safe rendering for all three sites AND emits\n"
    "`band_provenance`.`competing_reading_and_what_this_does_not_settle` (the\n"
    "canonical statement, the two readings and the four invariants, addressed by\n"
    "JSON PATH and never by line number), so a re-run reproduces this correction\n"
    "instead of dropping it.\n"
)

# ---------------------------------------------------------------------------
# The retrieval log -- every byte this matrix rests on
# ---------------------------------------------------------------------------
#
# ``sha256`` is over the file as downloaded in this session (the raw arXiv
# HTML, or the raw source file from GitHub).  ``bytes`` is the file size.
# ``extracted`` says what was taken out of it.

RETRIEVAL_LOG: List[Dict[str, Any]] = [
    {
        "url": "https://arxiv.org/html/2609.08333v4",
        "kind": "arxiv-html",
        "arxiv_id": "2609.08333",
        "version": "v4",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 1091210,
        "sha256": "d0459ee59efaa4b70aac7295bf1478b6a80824a1fcd11bf650b255dc8d06d887",
        "extracted": (
            "EF-TALFM metrics definitions (A.1.4), Table 3/Table 4 + captions, "
            "the 92.98/98.25/98.21/94.33 cells, the Appendix-D reconstruction "
            "and largest-component-retention text, the heavy-atom-scaffold "
            "statement, the 10,000-sample denominators, the 92,221-connected "
            "sentence, and the bond-threshold rule in Appendix D Algorithm 4."
        ),
    },
    {
        "url": "https://arxiv.org/abs/2609.08333",
        "kind": "arxiv-abs",
        "arxiv_id": "2609.08333",
        "version": "abs (v4 metadata)",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": None,
        "sha256": None,
        "extracted": "title, authors, abstract (87.9 % unique-and-novel wording, PCQM4Mv2).",
    },
    {
        "url": "http://export.arxiv.org/api/query?id_list=2609.08333,2508.12629,2411.05676,2509.01486",
        "kind": "arxiv-api",
        "arxiv_id": "multiple",
        "version": "metadata",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": None,
        "sha256": None,
        "extracted": (
            "canonical ids, titles, published/updated dates, and the "
            "'Accepted by NeurIPS 2025' comment for 2509.01486."
        ),
    },
    {
        "url": "https://arxiv.org/html/2508.12629v1",
        "kind": "arxiv-html",
        "arxiv_id": "2508.12629",
        "version": "v1",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 327575,
        "sha256": "4294599f80729d9e3aca239169203c0cca78e7dbbffbe2c769df8af248675476",
        "extracted": (
            "FlowMol3 Section 3.2.1 validity definitions, Table 1 (99.9+/-0.1 "
            "% Valid, 91.9+/-0.7 % PB-Valid), the Table 1 caption, the "
            "explicit-hydrogen training statements, the 5000-sample / 5-subset "
            "sampling protocol, the kekulization statement, and the caveat "
            "that validity is 'a necessary but insufficient condition'."
        ),
    },
    {
        "url": "https://arxiv.org/html/2411.05676v1",
        "kind": "arxiv-html",
        "arxiv_id": "2411.05676",
        "version": "v1",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 462840,
        "sha256": "d1c28aecb4ca4f57eaaa6de1d709bdc084fa5422535ec21dc1ceeb0d9c80942b",
        "extracted": (
            "GGFlow Table 1 (Val. 100 / Val. w/o corr. 99.91 on QM9; 100 / "
            "99.63 on ZINC250k), the Appendix-E validity metric definitions "
            "(including 'after valency correction or edge resampling' and the "
            "'Validity w/o correction' baseline), the 10,000-sample statement, "
            "the QM9/ZINC250k dataset description, and the table-statistics "
            "row (4 node types)."
        ),
    },
    {
        "url": "https://openreview.net/forum?id=K8RlXtMgzo",
        "kind": "openreview-forum",
        "arxiv_id": "2411.05676",
        "version": "TMLR forum K8RlXtMgzo (retitled GGFlow)",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 403,
        "bytes": None,
        "sha256": None,
        "extracted": (
            "NOTHING -- the page returned a browser-verification challenge "
            "('Verifying your browser'). The api2.openreview.net notes "
            "endpoint returned 403 ChallengeRequiredError. The retitled TMLR "
            "version of GGFlow was therefore NOT retrieved; the arXiv v1 "
            "(title 'Improving Molecular Graph Generation with Flow Matching "
            "and Optimal Transport') is used instead and is flagged as such."
        ),
    },
    {
        "url": "https://arxiv.org/html/2509.01486v2",
        "kind": "arxiv-html",
        "arxiv_id": "2509.01486",
        "version": "v2",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 552796,
        "sha256": "90a9a456ac3653d9983bf8d616e4468f2a1badfe2db360f131068f29ed4a3c90",
        "extracted": (
            "PAFlow title/abstract, Section 4.1 evaluation metrics, the "
            "100-molecules-per-test-protein sampling statement, and the "
            "CrossDocked2020 dataset statement -- and, by exhaustive grep, "
            "the ABSENCE of any connectivity/validity band."
        ),
    },
    {
        "url": "https://raw.githubusercontent.com/dunni3/FlowMol/main/flowmol/analysis/metrics.py",
        "kind": "github-raw",
        "arxiv_id": "2508.12629",
        "version": "main @ 2026-04-26",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 18291,
        "sha256": "db8275c3c69d7b83b991fa221183edc16a2f47ad6fef43c5feec2ded7f69f47e",
        "extracted": (
            "SampleAnalyzer.compute_validity -- the exact largest-fragment + "
            "sanitisation sequence behind '% Valid', the frac_connected "
            "counter, and the PoseBusters bust() call on the RAW molecule."
        ),
    },
    {
        "url": "https://raw.githubusercontent.com/dunni3/FlowMol/main/fm3_evals/baselines/compute_baseline_comparison.py",
        "kind": "github-raw",
        "arxiv_id": "2508.12629",
        "version": "main @ 2026-04-26",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 4954,
        "sha256": "030cdd93f33de80d2ebf2ae43f89be7c2636f708a2bedd41281481142931792e",
        "extracted": (
            "the SDF loader flags (sanitize=False, removeHs=False), the "
            "n_subsets split arithmetic, n_raw_samples, and the "
            "frac_valid_mols denominator rescaling."
        ),
    },
    {
        "url": "https://raw.githubusercontent.com/dunni3/FlowMol/main/flowmol/analysis/molecule_builder.py",
        "kind": "github-raw",
        "arxiv_id": "2508.12629",
        "version": "main @ 2026-04-26",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 12937,
        "sha256": "a4de6f4d117703ab414d8b58b80699eb1aa6366e334887a140884bc06bbfd976",
        "extracted": "build_molecule -- every sampled atom (hydrogens included) becomes an RDKit atom node.",
    },
    {
        "url": "https://raw.githubusercontent.com/dunni3/FlowMol/main/flowmol/analysis/pb_config.yaml",
        "kind": "github-raw",
        "arxiv_id": "2508.12629",
        "version": "main @ 2026-04-26",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 4828,
        "sha256": "8dfec90099ce8f1f46bb88a5547bdf174051fbb385ce9e9befba5385633ccc9b",
        "extracted": "the PoseBusters module list, ignore_hydrogens: True, and loading.add_hs: False.",
    },
    {
        "url": "https://raw.githubusercontent.com/dunni3/FlowMol/main/fm3_evals/readme.md",
        "kind": "github-raw",
        "arxiv_id": "2508.12629",
        "version": "main @ 2026-04-26",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 4373,
        "sha256": "dc2ddc43ba6b72583c0ba11ab07829dd74ee30bbe460c6853eba8707faa0b9c9",
        "extracted": "the author's own evaluation command line (--n_mols=5000 --n_subsets=5).",
    },
    {
        "url": "https://arxiv.org/html/2603.27113v1",
        "kind": "arxiv-html",
        "arxiv_id": "2603.27113",
        "version": "v1 (only version on arXiv; published 2026-03-28)",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 468625,
        "sha256": "41e2a2e0f72df7ae1bc57091c168fa55d5a599f756841388976582e5f9a5ffb5",
        "extracted": (
            "HLTF Table 1(a) (QM9: AS 98.8, Val 93.8, Val&Uniq 92.9), Table 1(b) "
            "(GeoLDM Val&Uniq 92.7 +/- 0.5; HLTF PB-Valid 94 +/- 0.5), Section 4.1 "
            "(the QM9 10,000-sample and GEOM-DRUGS 100,000-sample denominators and "
            "the post-processing recipe 'largest fragment, explicit hydrogens, UFF "
            "relaxation'), Table 2 (the GEOM-DRUGS raw -> +PP pairs for EVERY "
            "model), and the 'What changes with post-processing' paragraph."
        ),
    },
    {
        "url": "https://arxiv.org/abs/2603.27113",
        "kind": "arxiv-abs",
        "arxiv_id": "2603.27113",
        "version": "abs metadata",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": None,
        "sha256": None,
        "extracted": (
            "authors (Urvi Awasthi, Alexander Arjun Lobo, Leonid Zhukov), the "
            "comments field ('22 pages, 2 figures, 6 tables. Accepted to ICLR "
            "2026 AI4Mat Workshop') and the abstract, which restates 98.8 % AS / "
            "92.9 % Val&Uniq / 94.0 % PB-Valid on QM9 and 85.5 % -> 92.2 % on "
            "GEOM-DRUGS."
        ),
    },
    {
        "url": "https://www.biorxiv.org/content/10.1101/2025.05.06.652517v1.full",
        "kind": "biorxiv-full-text",
        "arxiv_id": None,
        "version": "bioRxiv v1 (preprint of J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052)",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 235062,
        "sha256": "1fcfc65a2e3ef12e8b5dc6ad03883df748fa8ca712dfef95274b0e8e708b73ac",
        "extracted": (
            "the abstract (98.4 % F1 on standard 3-D structures; 92.7 % F1 on "
            "distorted CDGs), the CDG bonds-reconstruction results (accuracy "
            "92.7 %, precision 93.6 %, recall 92.7 %, F1 92.7 % over 1,000 noisy "
            "molecules), the per-bond-type breakdown, the RDKit comparison (217 "
            "of 1,000 processed), and the Methods statement that the metrics are "
            "multi-class bond-type classification metrics."
        ),
    },
    # ---------------------------------------------------------------------------
    # 2026-11-01 (the RESIDUAL-2 repair) -- GeoLDM was retrieved as an arXiv
    # e-print SOURCE TARBALL, so that the record's Reading-A lower endpoint
    # is substantiated by the ORIGINATING paper's own bytes rather than by a
    # transcript of somebody else's table.  The e-print sha256 and the
    # extracted .tex sha256 both MATCH the pins recorded by the Wf-A186
    # primary-source artefact.
    {
        "url": "https://arxiv.org/e-print/2305.01140v1",
        "kind": "arxiv-eprint-source-tarball",
        "arxiv_id": "2305.01140",
        "version": "v1 (only version on arXiv; submitted 2023-05-02)",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": 655684,
        "sha256": "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23",
        "extracted": (
            "icml2023.tex (sha256 "
            "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b): "
            "the tab:qm9_results caption (:468), the tab:qm9_results GeoLDM row "
            "(:491 -- 98.9 +/- 0.1 / 89.4 +/- 0.5 / 93.8 +/- 0.4 / 92.7 +/- 0.5 "
            "/ 84.4 / 99.3), the metric definitions (:532), the sample-size "
            "sentence (:549), the QM9 dataset paragraph (:517) and the "
            "disconnected-components remark (:917); the \\method macro "
            "definition (:70) and the ICML 2023 class/keyword lines (:26, "
            ":131) for the venue."
        ),
    },
    {
        "url": "http://export.arxiv.org/api/query?id_list=2305.01140",
        "kind": "arxiv-api",
        "arxiv_id": "2305.01140",
        "version": "metadata",
        "retrieved_at": RETRIEVAL_DATE,
        "http_status": 200,
        "bytes": None,
        "sha256": None,
        "extracted": (
            "title 'Geometric Latent Diffusion Models for 3D Molecule Generation', "
            "authors (Minkai Xu, Alexander Powers, Ron Dror, Stefano Ermon, Jure "
            "Leskovec), v1 published 2023-05-02, and the ABSENCE of a "
            "journal-ref, a comment field and a DOI."
        ),
    },
]

# ---------------------------------------------------------------------------
# The anchor: OUR side of the comparison (read-only, cross-branch)
# ---------------------------------------------------------------------------

ANCHOR: Dict[str, Any] = {
    "label": "mol-metal CFM decode (RAW, no post-processing)",
    "measured_by": "this branch did NOT re-measure it; read from disk in this session",
    "value": 0.08509630565203663,
    "value_display": "8.51 %",
    "path": "metrics/by_metric/cfm_fragmentation_audit.json",
    "jsonpath": "/blind_validity_signature/3/single_fragment_ratio",
    "definition": (
        "n_single / samples drawn, where n_single = the pool rows whose "
        "decoded molecule has exactly ONE RDKit fragment, measured at the RAW "
        "emitted SMILES after RDKit MolFromSmiles parse (sanitisation) and "
        "Chem.GetMolFrags -- NO post-processing."
    ),
    "post_processing": "none (raw)",
    "hydrogen_convention": "implicit (RDKit default after MolFromSmiles; no AddHs step)",
    # 2026-11-01 RESIDUAL REPAIR (RESIDUAL 4 / D3) -- WHICH quantity is the
    # anchor?  It is QUANTITY B: the SINGLE-POOL raw connected-component
    # ratio of ONE decode pool.  It is NOT the median over the 9 recorded
    # runs (QUANTITY A).  Both print as "8.51 %", so the anchor used to
    # name NEITHER.  Every number below is already on disk: the anchor
    # itself is metrics/by_metric/cfm_fragmentation_audit.json ->
    # /blind_validity_signature/3/single_fragment_ratio, the median is the
    # same file -> /summary/generation_side/single_molecule_rate_end_to_end
    # (n = 9), and the numerator/denominator are the catalogue's own
    # "1078 / 12668 parsed" (docs/HONEST_NEGATIVE_CATALOG.md, quoted
    # verbatim in the Wf-A186 primary-source artefact).  NO NEW STATISTIC
    # is introduced: the numerator and the denominator are already in the
    # record (anchor_our_side.postprocessed_counterpart.denominators).
    "which_quantity": (
        "QUANTITY B -- the SINGLE-POOL RAW CONNECTED-COMPONENT RATIO of ONE "
        "decode pool: 1,078 pool rows whose decoded molecule has exactly ONE "
        "RDKit fragment, out of 12,668 parsed rows (12,669 rows declared in "
        "the CSV, 1 unparseable).  It is NOT the median over the 9 recorded "
        "runs (0.08506944444444445 -- QUANTITY A): that is a DIFFERENT "
        "number that prints as the SAME '8.51 %'.  THIS anchor is the "
        "SINGLE-POOL one; its run is cfm_c7_fullgen_2026_10_20."
    ),
    "which_quantity_note_2026_11_01": (
        "2026-11-01 residual repair: the anchor used to be rendered as "
        "'8.51 % (0.08509630565203663)' with NO statement of WHICH quantity "
        "that is.  It is QUANTITY B -- the single-pool raw "
        "connected-component ratio (1,078 / 12,668 parsed rows) -- and NOT "
        "the median over the 9 recorded runs (0.08506944444444445).  Both "
        "quantities print as '8.51 %'.  The previous unnamed rendering is "
        "PRESERVED VERBATIM in `which_quantity_superseded_2026_11_01`.  "
        "MEASURED: metrics/by_metric/cfm_fragmentation_audit.json -> "
        "/blind_validity_signature/3/single_fragment_ratio (the anchor, run "
        "cfm_c7_fullgen_2026_10_20) and -> "
        "/summary/generation_side/single_molecule_rate_end_to_end (the "
        "median over n = 9 runs)."
    ),
    "which_quantity_superseded_2026_11_01": (
        "8.51 % (0.08509630565203663) -- THE ANCHOR, rendered with no "
        "statement of WHICH quantity it is."
    ),
    "cross_branch_provenance": (
        "the population reconciliation and the definition family (F1_pool_e2e / "
        "F2_pool_cond) are documented in "
        "molmetal/reports/wfa158_population_reconciliation/census.md (a "
        "SIBLING Wf-A158 branch artifact, read-only here). This branch did "
        "not re-derive it and does not claim to."
    ),
    "postprocessed_counterpart": {
        "value": 1.0,
        "value_display": "100.00 %",
        "path": "metrics/by_metric/cfm_fragmentation_postproc_arm.json",
        "jsonpath": "/postprocessed_axis/connected_rate_over_parsed",
        "pipeline": [
            "LargestFragmentChooser (rdMolStandardize, fallback: max heavy atoms)",
            "Chem.AddHs (explicit hydrogens)",
            "AllChem.UFFOptimizeMolecule(maxIters=200)",
        ],
        "note": (
            "MEASURED (Wf-A133, published by Wf-A141), re-read from disk in "
            "this session. It is the SAME pool (n_rows_in_csv 12669) run "
            "through the frontier-style post-processing. LargestFragmentChooser "
            "keeps exactly one fragment by construction, so this number is "
            "1.0 BY CONSTRUCTION -- it is not evidence that our molecules are "
            "chemically valid, it is the price of the mass-deleting operation "
            "(the same file records "
            "mass_retention.fraction_of_raw_mass_discarded = 0.22474803194442472)."
        ),
        "denominators": {
            "n_rows_in_csv": 12669,
            "n_parsed": 12668,
            "n_postprocessed": 12668,
            "n_unparseable": 1,
        },
        "like_for_like_retention": {
            "our_value": 0.7395022201805989,
            "our_value_display": "73.95 % of heavy atoms kept (mean)",
            "our_median": 0.926829268292683,
            "our_jsonpath": "/mass_retention/heavy_atom_retention_mean",
            "our_median_jsonpath": "/mass_retention/heavy_atom_retention_median",
            "frontier_analogue": (
                "FlowMol3's harness computes 'avg_frag_frac' = "
                "largest_mol.GetNumAtoms() / mol.num_atoms -- the SAME "
                "quantity as our heavy_atom_retention_mean -- but the "
                "FlowMol3 PAPER DOES NOT REPORT IT (verified by exhaustive "
                "grep of the paper's extracted text: 'avg_frag_frac' and "
                "'avg_num_components' occur only in the code, never in the "
                "paper)."
            ),
            "note": (
                "This is the ONE genuinely like-for-like number available on "
                "both sides: the mean fraction of heavy atoms retained by the "
                "largest-fragment step, under the SAME protocol family "
                "(Buttenschoen et al., arXiv:2505.00518, LargestFragmentChooser "
                "-> explicit H -> UFF, as recorded in the postproc-arm file's "
                "'purpose' field). It says how much of each molecule the "
                "largest-fragment step keeps -- it does NOT say the surviving "
                "molecule is good."
            ),
        },
        "uff_step_applicability": {
            "value": 0.0,
            "value_display": "0.00 % of post-processed molecules",
            "jsonpath": "/postprocessed_axis/uff_ok_rate_over_postprocessed",
            "note": (
                "THE FRONTIER RECIPE'S THIRD STEP DOES NOT APPLY TO OUR "
                "CORPUS. The same file records uff_ok_rate_over_postprocessed "
                "= 0.0 with uff_reasons 'ValueError: Bad Conformer Id = 12668' "
                "and the note that UFF has NO parameters for the metals in "
                "this corpus. So 'we applied the frontier recipe' is true for "
                "steps 1-2 and FALSE for step 3: our connectivity number of "
                "1.0 is reached WITHOUT the hydrogen-assisted restrained "
                "force-field relaxation that EF-TALFM's geometry rows use."
            ),
        },
        "survivor_shape_note": (
            "The survivors are BIMODAL (from the same file): 'a drug-sized "
            "molecule (>= 11 heavy atoms) or a <= 2-heavy-atom speck; "
            "almost nothing in between' -- n_survivors_le_2_heavy 2663, "
            "n_survivors_ge_11_heavy 10003, n_survivors_le_5_heavy 2663. So "
            "the 100 % connected rate is not the rate at which 'reasonable "
            "molecules' are produced."
        ),
    },
}

# ---------------------------------------------------------------------------
# THE MATRIX
# ---------------------------------------------------------------------------

#: Where the repo's "92.7-94 %" band actually comes from.
#:
#: MEASURED by reading the repo's own frontier scan
#: (``molmetal/reports/wf_cfm_frontier_research/research_phase1b.md``,
#: read-only) and then re-verifying every candidate number against the
#: PRIMARY SOURCE retrieved in this session.  The band's endpoints are NOT
#: connectivity rates.
# ---------------------------------------------------------------------------
# 2026-11-02 (Wf-A197) -- NAMED DATED CONSTANTS.
#
# AGENTS.md §5.5 rule #6: "In a .py you add a NAMED DATED CONSTANT and
# reference it; you do not delete the old one."  Each constant below holds a
# SUPERSEDED RENDERING VERBATIM, and the corrected strings reference them, so
# a re-run of the producer reproduces BOTH the correction and the record of
# what it replaced.
#
# The literals are the BYTES of the published renderings, so the preserved
# copy is byte-identical to the rendering it replaces.
# ---------------------------------------------------------------------------

#: D1 -- the wrong-numeral sources census, superseded VERBATIM.
WFA197_D1_SUPERSEDED_SENTENCE_2026_11_02 = "The five pre-existing source entries are UNCHANGED; `sources`.`GeoLDM` was ADDED beside them."

#: D2 -- the 2026-11-01 rendering of the divergence sentence, superseded
#: VERBATIM (it was edited IN PLACE by the 2026-11-01 residual repair).
WFA197_D2_SUPERSEDED_DIVERGENCE_2026_11_02 = "The producer emits a Wf-A180-SHAPED `competing_reading_and_what_this_does_not_settle` (canonical clause + the two readings + the four invariants, JSON PATHS only, NO self-referential line numbers -- they go stale on every edit).  This record keeps the Wf-A177-SHAPED object under the same key (it carries the extra `line_number_note`, `historical_line_numbers_measured_2026_11_01` and `cross_references` fields).  The two objects agree on every CLAIM; they differ only in the extra Wf-A177 fields and in the wording of the canonical statement.  The stale line-number parentheticals elsewhere in the Wf-A177 object stay covered by that object's own `line_number_note`, which names the JSON PATHS as the robust locators.  (2026-11-01 residual repair: the two `current_line_numbers_on_disk` keys were RENAMED to the dated `historical_line_numbers_measured_2026_11_01` keys, with their values PRESERVED VERBATIM -- see `band_provenance`.`frontier_band_residuals_closed_2026_11_01`.)"

#: D2 -- the PRIOR (pre-Wf-A194) rendering of the same sentence, recovered
#: VERBATIM from the pre-Wf-A194 published bytes of this record
#: (2026-11-01 21:16).  It was preserved NOWHERE on disk before this repair.
WFA197_D2_PRIOR_RENDERING_2026_11_02 = "The producer emits a Wf-A180-SHAPED `competing_reading_and_what_this_does_not_settle` (canonical clause + the two readings + the four invariants, JSON PATHS only, NO self-referential line numbers -- they go stale on every edit).  This record keeps the Wf-A177-SHAPED object under the same key (it carries the extra `line_number_note`, `current_line_numbers_on_disk` and `cross_references` fields).  The two objects agree on every CLAIM; they differ only in the extra Wf-A177 fields and in the wording of the canonical statement.  The stale line-number parentheticals elsewhere in the Wf-A177 object stay covered by that object's own `line_number_note`, which names the JSON PATHS as the robust locators."

#: X5 -- the settled-count claim, superseded VERBATIM.
WFA197_X5_SUPERSEDED_WHY_2026_11_02 = "This key exists so that a reader of the band-provenance block never has to guess which provenance reading an emitted string assumes.  The repo holds TWO competing readings, and they do not even agree on the numerals.  A future reader of this file must therefore read every band claim through the lens of the two readings recorded here."

#: R2 -- the dated note that records that the `2026_11_01` identifier in the
#: name of `band_provenance`.`frontier_band_residuals_closed_2026_11_01` is
#: NOT provable from disk.
WFA197_R2_NOTE_KEY = (
    "frontier_band_residuals_closed_2026_11_01_identifier_not_provable_2026_11_02"
)

WFA197_R2_NOTE: Dict[str, Any] = {
    "dated": "2026-11-02",
    "wave": "Wf-A197",
    "what_this_is": (
        "The dated note that records that the `2026_11_01` identifier in the "
        "NAME of `band_provenance`.`frontier_band_residuals_closed_2026_11_01` "
        "is NOT provable from disk."
    ),
    "measurement": {
        "the_key_is_absent_from_the_pre_wf_a194_published_bytes": True,
        "the_pre_wf_a194_published_bytes_were_written_at": "2026-11-01 21:16",
        "the_key_was_first_published_in_the_write_of": "2026-11-02 01:23:34",
        "measured_on": "2026-11-02",
    },
    "overlaps": (
        "A parallel block, `band_provenance`."
        "`locator_class_and_geo_ldm_source_2026_11_02`, covers the SAME TWO "
        "closures as this key -- the live line numbers demoted "
        "(`residual_1_live_line_numbers_demoted`) and GeoLDM as a first-class "
        "source (`residual_2_geoldm_is_now_a_first_class_source`)."
    ),
    "note": (
        "The key is NOT RENAMED: other artefacts cite it by name.  The date "
        "in the key's identifier is NOT asserted as a first-publication date "
        "anywhere in this record, and no reader may infer one from the "
        "identifier alone."
    ),
}

#: Wf-A197 -- the dated key that records the SIX residual defects an
#: adversarial verifier found in this record and in the producer that emits
#: it, and the repair for each.
WFA197_RESIDUALS_KEY = "record_residuals_closed_2026_11_02"

WFA197_RESIDUALS: Dict[str, Any] = {
    "dated": "2026-11-02",
    "wave": "Wf-A197",
    "heading": (
        "Record residuals closed (2026-11-02) -- six defects, six repairs"
    ),
    "what_this_is": (
        "The dated key that records the SIX residual defects an adversarial "
        "verifier found in this record and in the producer that emits it, and "
        "the repair for each.  Every superseded rendering that a repair "
        "replaced is PRESERVED VERBATIM in a dated sibling key, so that a "
        "re-run of the producer reproduces both the correction and the record "
        "of what it replaced (AGENTS.md §5.5 rule #6)."
    ),
    "the_append_assertion": (
        "This section is APPENDED.  Per AGENTS.md §5.5 rule #6 the DATED "
        "SECTIONS above are PRESERVED VERBATIM, and nothing above this line "
        "was deleted, removed or re-ordered.  The BODY above WAS re-rendered "
        "by Wf-A194 -- the passages that were re-rendered are listed below -- "
        "so the 'nothing above this line was ... re-rendered' assertion in "
        "the dated 2026-11-02 section above is PRESERVED VERBATIM there and is "
        "SUPERSEDED by this section.  This repair re-rendered exactly ONE live "
        "body string of its own -- `sources`.`GeoLDM`.`extra_observations` -- "
        "whose superseded rendering is preserved verbatim below."
    ),
    "the_six_residuals": [
        {
            "residual": "D1",
            "short_title": (
                "a wrong numeral and a false 'UNCHANGED' claim in the sources "
                "census"
            ),
            "defect": (
                "The dated 2026-11-02 locator-class key emits, inside a LIVE "
                "string: 'disk.  The five pre-existing source entries are "
                "UNCHANGED; `sources`.`GeoLDM` was ADDED beside them.'  "
                "MEASURED on 2026-11-02 by diffing the pre-Wf-A194 published "
                "bytes of this record (2026-11-01 21:16) against the published "
                "bytes (2026-11-02 01:23:34), leaf by leaf and source entry "
                "by source entry: SIX source entries pre-existed the Wf-A194 "
                "repair, not five, and `sources`.`HLTF` was NOT unchanged -- "
                "its `cells`.`METRIC`.`note` was REWRITTEN and TWO leaves were "
                "ADDED under it (`cells`.`METRIC`."
                "`note_superseded_2026_11_01_residual` and `cells`.`METRIC`."
                "`note_superseded_2026_11_01_residual_note`).  The other five "
                "-- `EF-TALFM`, `FlowMol3`, `GGFlow`, `PAFlow`, `YuelBond` -- "
                "are byte-identical.  So the sentence is wrong on the numeral "
                "AND false on the claim."
            ),
            "repair": (
                "The sentence is LEFT VERBATIM -- it is rendered into the dated "
                "2026-11-02 markdown section above, and a dated section is not "
                "rewritten in place (AGENTS.md §5.5 rule #6).  The CORRECTED "
                "sentence is emitted in `band_provenance`."
                "`locator_class_and_geo_ldm_source_2026_11_02`."
                "`the_sources_census_correction_2026_11_02`."
                "`corrected_sentence`: 'The SIX pre-existing source entries are "
                "UNCHANGED except `sources`.`HLTF`, whose `cells`.`METRIC`."
                "`note` was rewritten and to which TWO leaves were added; the "
                "other five are byte-identical.  `sources`.`GeoLDM` was ADDED "
                "beside them.'"
            ),
            "superseded_rendering_preserved_verbatim_at": (
                "`band_provenance`.`locator_class_and_geo_ldm_source_2026_11_02`."
                "`the_sources_census_correction_2026_11_02`."
                "`superseded_sentence_preserved_verbatim` (NAMED DATED "
                "CONSTANT `WFA197_D1_SUPERSEDED_SENTENCE_2026_11_02` in the "
                "producer)"
            ),
        },
        {
            "residual": "D2",
            "short_title": (
                "a dated 2026-11-01 block edited IN PLACE, with its prior "
                "rendering preserved nowhere"
            ),
            "defect": (
                "The dated key `band_provenance`."
                "`producer_and_record_reconciled_2026_11_01` had its "
                "`divergence_from_the_producer` sentence EDITED IN PLACE by the "
                "2026-11-01 residual repair, and the PRIOR rendering was "
                "preserved NOWHERE on disk -- an AGENTS.md §5.5 rule #6 "
                "violation.  MEASURED on 2026-11-02, the CURRENT text was ALSO "
                "inaccurate on its own terms: it said the producer emits a "
                "'Wf-A180-SHAPED' object and that this record keeps a "
                "'Wf-A177-SHAPED' object, but the producer's emitted object "
                "and this record's object under `band_provenance`."
                "`competing_reading_and_what_this_does_not_settle` are "
                "BYTE-IDENTICAL."
            ),
            "repair": (
                "The sentence now states that the producer emits, under this "
                "key, the SAME object this record keeps, and that there is "
                "therefore NO divergence between the producer and this record "
                "on this key (MEASURED on 2026-11-02).  BOTH prior renderings "
                "are preserved VERBATIM: the 2026-11-01 rendering in "
                "`band_provenance`.`producer_and_record_reconciled_2026_11_01`."
                "`divergence_from_the_producer_superseded_2026_11_02` (NAMED "
                "DATED CONSTANT `WFA197_D2_SUPERSEDED_DIVERGENCE_2026_11_02`), "
                "and the PRIOR pre-Wf-A194 rendering in `band_provenance`."
                "`producer_and_record_reconciled_2026_11_01_prior_rendering_"
                "2026_11_02`.`divergence_from_the_producer` (NAMED DATED "
                "CONSTANT `WFA197_D2_PRIOR_RENDERING_2026_11_02`), recovered "
                "from the pre-Wf-A194 published bytes of this record "
                "(2026-11-01 21:16)."
            ),
            "superseded_rendering_preserved_verbatim_at": (
                "the two dated sibling keys named in the repair above"
            ),
        },
        {
            "residual": "D3",
            "short_title": (
                "a false 'nothing above this line was re-rendered' assertion"
            ),
            "defect": (
                "The dated 2026-11-02 markdown section asserts, verbatim: "
                "'This section is APPENDED.  Per AGENTS.md §5.5 rule #6 nothing "
                "above this line was deleted, edited, removed, re-rendered or "
                "re-ordered: the dated sections above are PRESERVED VERBATIM, "
                "and every superseded rendering they quote is still the record "
                "of what the body said before this repair.'  THAT ASSERTION IS "
                "FALSE on its own terms: MEASURED by diffing the pre-Wf-A194 "
                "published bytes of this record (2026-11-01 21:16) against the "
                "published bytes (2026-11-02 01:23:34), the BODY was "
                "re-rendered -- seven live leaves carry a different value, two "
                "keys were RENAMED (their values PRESERVED VERBATIM), and the "
                "three self-referential byte-evidence counters were recomputed.  "
                "The dated 2026-11-01 section above it carries the WEAKER, "
                "TRUE assertion ('nothing above this line was deleted: the two "
                "dated sections above are PRESERVED VERBATIM')."
            ),
            "repair": (
                "The FALSE assertion is LEFT VERBATIM in the dated 2026-11-02 "
                "section (a dated section is not rewritten in place), and the "
                "corrected assertion -- the one this section carries -- says "
                "exactly what is true: the DATED SECTIONS were preserved "
                "verbatim; the BODY above was re-rendered by Wf-A194; the "
                "re-rendered passages are listed below."
            ),
            "superseded_rendering_preserved_verbatim_at": (
                "the dated 2026-11-02 markdown section above (left VERBATIM)"
            ),
        },
        {
            "residual": "R1",
            "short_title": (
                "live cross-artefact line-number locators in `sources`."
                "`GeoLDM`"
            ),
            "defect": (
                "The live string at `sources`.`GeoLDM`.`extra_observations` "
                "cited the Wf-A175 frontier comparability record by LINE "
                "NUMBER -- report.md:40, report.md:133, report.md:138, "
                "sources.json:170 and sources.json:417.  MEASURED on 2026-11-02: "
                "all five are ACCURATE right now, but that artefact is "
                "regenerated and grows, so a LINE NUMBER is a PERISHABLE "
                "locator there -- the same locator class Wf-A194 closed inside "
                "THIS record."
            ),
            "repair": (
                "The live string now cites that artefact by JSON PATH plus its "
                "sha256, as measured on 2026-11-02: GeoLDM's own 93.8 +- 0.4 "
                "at `/sources/4/metrics/0/value` and "
                "`/sources/4/metrics/0/evidence_quote`, and HLTF's "
                "GeoLDM-column Val 93.6 +- 0.2 at "
                "`/sources/11/metrics/2/evidence_quote`, all in "
                "molmetal/reports/wfa175_frontier_comparability/sources.json "
                "(sha256 "
                "2e665cb217759a4099c83716a8f27829797f2401c6c630adaeb1b4cfbd6e3c78, "
                "as measured on 2026-11-02); and the Wf-A175 report.md (sha256 "
                "170f642846192837c334d45bb6a006414bf1b53a7089c8907f4061ff0d0f10e0, "
                "as measured on 2026-11-02) is cited by its sha256 and by the "
                "verbatim sentence it carries.  The superseded rendering is "
                "preserved VERBATIM in `sources`.`GeoLDM`."
                "`extra_observations_superseded_2026_11_02` (NAMED DATED "
                "CONSTANT `WFA197_R1_SUPERSEDED_OBSERVATION_2026_11_02` in the "
                "producer)."
            ),
            "superseded_rendering_preserved_verbatim_at": (
                "`sources`.`GeoLDM`.`extra_observations_superseded_2026_11_02`"
            ),
        },
        {
            "residual": "R2",
            "short_title": (
                "a date in a key's identifier that is not provable from disk"
            ),
            "defect": (
                "`band_provenance`.`frontier_band_residuals_closed_2026_11_01` "
                "says its date 'is the identifier', yet MEASURED on 2026-11-02 "
                "the key is ABSENT from the pre-Wf-A194 published bytes of this "
                "record (2026-11-01 21:16) and was FIRST PUBLISHED in the "
                "2026-11-02 01:23:34 write.  A parallel block, `band_provenance`."
                "`locator_class_and_geo_ldm_source_2026_11_02`, covers the SAME "
                "TWO closures."
            ),
            "repair": (
                "The key is NOT RENAMED -- other artefacts cite it by name.  A "
                "dated sibling key `band_provenance`."
                "`frontier_band_residuals_closed_2026_11_01_identifier_not_"
                "provable_2026_11_02` records the measurement, the limitation "
                "and the overlap with the dated 2026-11-02 block, and the date "
                "in the original key's identifier is NOT asserted as a "
                "first-publication date anywhere in this record."
            ),
            "superseded_rendering_preserved_verbatim_at": (
                "n/a -- the key is not renamed, so nothing is superseded"
            ),
        },
        {
            "residual": "X5",
            "short_title": (
                "a settled count claim about the number of competing readings"
            ),
            "defect": (
                "`band_provenance`."
                "`competing_reading_and_what_this_does_not_settle`."
                "`why_this_key_exists` said, verbatim: 'The repo holds TWO "
                "competing readings, and they do not even agree on the "
                "numerals.'  That is a claim about THE REPO, not about this "
                "record's own `readings` array (which really does hold 2 "
                "entries, and whose `status` and "
                "`canonical_reading_safe_band_statement` are locally true).  "
                "MEASURED on 2026-11-02: the repo's own primary-source "
                "verification (molmetal/reports/wfa186_frontier_primary_source/"
                "primary_source_verification.json -> `invariants`."
                "`competing_readings`) records a FURTHER reading that likewise "
                "does not reproduce the quoted numerals, and no document on "
                "disk settles how many there are."
            ),
            "repair": (
                "The live string now asserts NO settled count: the record "
                "enumerates two; the repo's primary-source verification "
                "records a further reading that likewise does not reproduce "
                "the quoted numerals; the NUMBER of competing readings is "
                "itself NOT settled on disk.  The superseded rendering is "
                "preserved VERBATIM in `band_provenance`."
                "`competing_reading_and_what_this_does_not_settle`."
                "`why_this_key_exists_superseded_2026_11_02` (NAMED DATED "
                "CONSTANT `WFA197_X5_SUPERSEDED_WHY_2026_11_02` in the "
                "producer).  The INVARIANT-USE of 'Under BOTH readings' is NOT "
                "churned -- the four invariants hold under EVERY reading, "
                "including the third -- and the four invariants themselves are "
                "NOT touched."
            ),
            "superseded_rendering_preserved_verbatim_at": (
                "`band_provenance`.`competing_reading_and_what_this_does_not_"
                "settle`.`why_this_key_exists_superseded_2026_11_02`"
            ),
        },
    ],
    "the_re_rendered_passages": [
        "`sources`.`HLTF`.`cells`.`METRIC`.`note`",
        "`band_provenance`.`competing_reading_and_what_this_does_not_settle`."
        "`line_number_note`",
        "`band_provenance`.`competing_reading_and_what_this_does_not_settle`."
        "`readings[0]`.`endpoints`.`upper`",
        "`band_provenance`.`competing_reading_and_what_this_does_not_settle`."
        "`the_only_claims_that_may_be_published`."
        "`ii_it_is_not_one_homogeneous_quantity`",
        "`band_provenance`.`endpoint_candidates[1]`.`candidate_a`.`what_it_is`",
        "`band_provenance`.`endpoint_candidates[1]`.`note`",
        "`band_provenance`.`producer_and_record_reconciled_2026_11_01`."
        "`divergence_from_the_producer`",
        "RENAMED, with the values PRESERVED VERBATIM: "
        "`band_provenance`.`competing_reading_and_what_this_does_not_settle`."
        "`readings[0]`.`current_line_numbers_on_disk` -> "
        "`readings[0]`.`historical_line_numbers_measured_2026_11_01`, and "
        "`readings[1]`.`current_line_numbers_on_disk` -> "
        "`readings[1]`.`historical_line_numbers_measured_2026_11_01`",
        "RECOMPUTED (the self-referential counters only): `band_provenance`."
        "`producer_reproduces_the_record_2026_11_01`.`byte_evidence`."
        "`md_body_chars`, `.`record_leaf_count` and "
        "`.`record_numeric_leaf_count`",
        "ADDED, not re-rendered: the `sources`.`GeoLDM` entry itself, its table "
        "row, its section, its two retrieval-log rows, and the dated 2026-11-02 "
        "markdown section",
    ],
    "what_this_repair_did_not_do": [
        "It did NOT change the FOUR publishable invariants -- the band is "
        "EXTERNAL to Mol-Metal, it is NOT one homogeneous quantity, it is NOT "
        "on our scale (OUR 8.51 % IS a connected-component count on the RAW "
        "emitted SMILES), and the source cells of the band are NOT settled on "
        "disk.  They are the only publishable claims and they are not touched "
        "here.",
        "It did NOT churn the INVARIANT-USE of 'Under BOTH readings': the four "
        "invariants hold under EVERY reading, including the third that the "
        "repo's primary-source verification records.",
        "It did NOT write a settled COUNT of the competing readings anywhere: "
        "the naive repair ('THREE') is the same defect.",
        "It did NOT rename `band_provenance`."
        "`frontier_band_residuals_closed_2026_11_01`; other artefacts cite it "
        "by name.",
        "It did NOT edit any dated markdown section in place: the false "
        "2026-11-02 assertion and the wrong-numeral census are LEFT VERBATIM "
        "and are superseded by dated siblings.",
        "It did NOT touch any file outside the six this wave owns: "
        "molmetal/scripts/wfa158_frontier_postproc_matrix.py, the three "
        "artefacts under molmetal/reports/wfa158_frontier_postproc_matrix/, "
        "and the two test files.",
        "It did NOT touch the Wf-A175 / Wf-A186 / Wf-A195 trees: they belong "
        "to another wave running at the same time.",
        "It did NOT move any numeric leaf: no numeric leaf was added, removed "
        "or given a new value -- the only leaves that changed value are the "
        "three self-referential counters the producer itself recomputes.",
    ],
}


#: R1 -- the LIVE string of `sources`.`GeoLDM` that cited the Wf-A175
#: frontier comparability record by LINE NUMBER, superseded VERBATIM.
WFA197_R1_SUPERSEDED_OBSERVATION_2026_11_02 = "THE TWO SOURCE TABLES DISAGREE ON THE SAME OBSERVABLE, AND THE DISAGREEMENT IS ANNOTATED HERE. GeoLDM's OWN paper prints Valid = 93.8 +/- 0.4 for its own row (icml2023.tex:491, sha256 b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b); HLTF prints the GeoLDM column's Val as 93.6 +/- 0.2 in its Table 1(a) (tables/qm9_benchmarks_og.tex:38, sha256 1c81ad499093f9d20d512e29787e934ab68c6d32dc5bb3b791970830c05c8f39) and again in its Table 1(b) (:63), and its Table 1 caption states 'Baselines reproduced from GCDM Table 1', i.e. the column is a TRANSCRIPT. NEITHER PAPER FLAGS THE DIFFERENCE. BOTH NUMERALS ARE ON DISK IN THIS REPO: the Wf-A175 frontier comparability record carries GeoLDM's own 93.8 +- 0.4 at molmetal/reports/wfa175_frontier_comparability/report.md:40 and HLTF's GeoLDM-column Val 93.6 +- 0.2 at the same file's :133, with both in molmetal/reports/wfa175_frontier_comparability/sources.json at :170 and :417. What the Wf-A175 record does NOT do is FLAG the two as a disagreement -- at report.md:138 it calls them 'the same GeoLDM row' without noticing the differing Val. NEITHER NUMERAL IS A BAND ENDPOINT: under Reading A the endpoints are 92.7 / 94.0 and under Reading B they are 92.98 / 94.33, so 93.8 and 93.6 must never be published as band endpoints."


BAND_PROVENANCE: Dict[str, Any] = {
    "band_as_quoted_in_the_repo": "92.7-94 %",
    "where_the_repo_states_it": [
        "metrics/by_metric/wfa155_ligand_connected_rate.json -> /comparability/frontier_band",
        "metrics/by_metric/wfa155_metal_coordination_sphere.json -> /comparability_statement",
        "docs/HONEST_NEGATIVE_CATALOG.md §6.264 (read-only for this branch)",
        "molmetal/reports/wf_cfm_frontier_research/research_phase1b.md (read-only for this branch)",
    ],
    # Wf-A177 (2026-11-01) CONTESTED-CLAIM ANNOTATION -- SOURCE COMMENT ONLY.
    # The emitted string below is DELIBERATELY UNCHANGED: the artefact on
    # disk must keep its original rendering (AGENTS.md §5.5 rule #6).  Only
    # the producer is annotated, so that a future reader of THIS SOURCE
    # knows the claim it emits is contested.
    #
    # The claim emitted below ("THE BAND IS NOT A CONNECTIVITY BAND ... None
    # of the three is a connected-component count") is TRUE under ONE of the
    # TWO competing provenance readings the repo holds, and FALSE under the
    # other, so it may NOT be published as settled.  The competing reading is
    # Reading B, at `frontier_matrix.json` -> `sources`.`EF-TALFM` (EF-TALFM
    # arXiv:2609.08333v4): the band mixes
    # EF-TALFM's OWN Table 3(a) RAW connectivity cell (C. = 92.98 %, counted
    # BEFORE the atom-count largest-component retention) with its
    # post-processed N. & U. & S. = 94.33 % companion cell -- under which the
    # LOWER endpoint IS a connectivity count.  The two readings do not even
    # agree on the numerals (Reading A reproduces the quoted 92.7 / 94
    # exactly; Reading B's cells are 92.98 / 94.33, i.e. "93.0-94.3"), so
    # the source cells of the band are NOT settled on disk and any published
    # claim about them must therefore be reading-qualified.
    #
    # Only the invariants that hold under BOTH readings may be published:
    # (i) the band is EXTERNAL to Mol-Metal (a FRONTIER composite);
    # (ii) it is NOT one homogeneous quantity; (iii) it is NOT on our
    # scale -- OUR 8.51 % IS a connected-component count on the RAW emitted
    # SMILES, and nothing else; (iv) the source cells of the band are NOT
    # settled on disk.  The same canonical reading-safe statement is recorded
    # in `molmetal/scripts/wfa161_frontier_labelling.py` (module docstring,
    # "Wf-A176 (2026-11-01) CORRECTION 2") and, from Wf-A177, in this script's
    # emitted artefact under
    # `band_provenance` -> `competing_reading_and_what_this_does_not_settle`.
    # Wf-A180 (2026-11-01) SUPERSEDED RENDERING -- PRESERVED VERBATIM.
    #
    # AGENTS.md §5.5 rule #6 requires the superseded rendering to be
    # PRESERVED, not to stay LIVE.  The Wf-A177 comment above (which
    # deliberately left the emitted string UNCHANGED) is therefore
    # itself superseded: the emitted string below is now the READING-SAFE
    # rendering, and the contest is carried by an EMITTED key
    # (`band_provenance` ->
    # `competing_reading_and_what_this_does_not_settle`) so that a re-run
    # of this producer reproduces the correction instead of dropping it.
    #
    # SUPERSEDED (the Wf-A177-live rendering, verbatim):
    #   THE BAND IS NOT A CONNECTIVITY BAND. Its two endpoints are a VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %); a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE CLASSIFICATION F1. None of the three is a connected-component count. Our 8.51 % is a connected-component count on the RAW emitted SMILES. The comparison the band invites is therefore not merely apples-to-oranges -- it is a comparison of two different observables, and the post-processing axis is a THIRD, separate axis on top of it.
    #
    # WHY IT IS SUPERSEDED: it asserts "THE BAND IS NOT A CONNECTIVITY
    # BAND" / "None of the three is a connected-component count", which
    # is TRUE under Reading A and FALSE under Reading B (under Reading B
    # the band's lower endpoint IS EF-TALFM's RAW connectivity cell
    # C. = 92.98 %).  The claim may not be published as settled.
    "headline_finding": (
        "READING-QUALIFIED: the published 92.7-94 % band's provenance is "
        "NOT settled on disk: the frontier post-processing matrix record "
        "(`frontier_matrix.json`) holds TWO competing readings of which "
        "frontier cells the band names -- the first under "
        "`band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF "
        "PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's "
        "per-bond-type classification F1 92.7 % as a third candidate for "
        "the same numeral) and the second under `sources`.`EF-TALFM` "
        "(EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity "
        "`C. = 92.98 %`, counted BEFORE the largest-component retention, "
        "with the post-processed `N. & U. & S. = 94.33 %` companion cell). "
        "They do not even agree on the numerals (92.7/94.0 vs "
        "92.98/94.33), so the band's source cells are NOT settled on disk. "
        "Under BOTH readings the band is EXTERNAL, is NOT one homogeneous "
        "quantity, and is NOT on our scale -- OUR 8.51 % IS a "
        "connected-component count on the RAW emitted SMILES, and nothing "
        "else. INDEPENDENTLY OF WHICH READING HOLDS, the frontier publishes "
        "the post-processing axis itself -- six models, raw and "
        "post-processed side by side, deltas from 5.6 to 95.0 pp (see "
        "`band_provenance`.`post_processing_evidence_from_the_frontier_"
        "itself`)."
    ),
    # Wf-A180 (2026-11-01) SUPERSEDED RENDERING -- PRESERVED VERBATIM.
    #
    # The Wf-A177 correction lived ONLY in the artefact on disk, so a
    # re-run of this producer dropped it and re-emitted the contested
    # rendering.  These two keys are EMITTED so that a re-run reproduces
    # the correction AND its history (AGENTS.md §5.5 rule #6).
    "headline_finding_superseded_2026_11_01": (
        "THE BAND IS NOT A CONNECTIVITY BAND. Its two endpoints are a "
        "VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate "
        "(94.0 %); a third paper in the same repo's scan quotes 92.7 % as a "
        "PER-BOND-TYPE CLASSIFICATION F1. None of the three is a "
        "connected-component count. Our 8.51 % is a connected-component "
        "count on the RAW emitted SMILES. The comparison the band invites "
        "is therefore not merely apples-to-oranges -- it is a comparison of "
        "two different observables, and the post-processing axis is a "
        "THIRD, separate axis on top of it."
    ),
    "headline_finding_superseded_note_2026_11_01": (
        "Wf-A180 (2026-11-01): the rendering above is SUPERSEDED.  It "
        "asserted 'THE BAND IS NOT A CONNECTIVITY BAND' / 'None of the "
        "three is a connected-component count', which is TRUE under Reading "
        "A and FALSE under Reading B (EF-TALFM's RAW connectivity cell C. = "
        "92.98 %).  It is PRESERVED VERBATIM here per AGENTS.md §5.5 rule "
        "#6.  The live rendering is the reading-safe one above."
    ),
    # Wf-A180 (2026-11-01) -- this key is EMITTED, not just annotated.
    # The Wf-A177 correction lived ONLY in the artefact on disk, so a re-run
    # of this producer dropped it and re-emitted the contested rendering.
    # It is emitted here so that a re-run reproduces the correction.  The
    # locators are JSON PATHS (`band_provenance`, `sources`.`EF-TALFM`), never
    # line numbers: line numbers go stale on every edit.
    "competing_reading_and_what_this_does_not_settle": {
        # The `wave` marker names the wave that CREATED this key, and that
        # fact does not change: Wf-A177 created it, and the Wf-A177-only
        # fields it still carries (`line_number_note`,
        # `readings[*].historical_line_numbers_measured_2026_11_01`,
        # `cross_references`)
        # are its own.  The READING-SAFE shape of the fields below was
        # applied by Wf-A180 and is recorded in the dated sibling keys and in
        # the dated md sections -- not by rewriting this marker (AGENTS.md
        # §5.5 rule #6: nothing is silently rewritten).
        "wave": "Wf-A177",
        "dated": "2026-11-01",
        "status": (
            "READING-QUALIFIED -- the two readings below are BOTH on disk and "
            "NEITHER is settled.  The band as quoted is EXTERNAL to "
            "Mol-Metal, is NOT one homogeneous quantity, is NOT on our scale, "
            "and its source cells are NOT settled on disk."
        ),
        "canonical_reading_safe_band_statement": (
            # The ONE live canonical clause.  MEASURED: this string was NOT
            # among the six leaves where the producer and the record
            # diverged -- the producer and the record already agreed on it --
            # so it is left exactly as it was.  The competing rendering that
            # DOES name the band ('92.7-94 %') is the live
            # `band_provenance`.`headline_finding`; the Wf-A177-era wording of
            # this clause is preserved verbatim in
            # `canonical_reading_safe_band_statement_superseded_2026_11_01`.
            "The published band's provenance is NOT settled on disk: the "
            "frontier post-processing matrix record (`frontier_matrix.json`) "
            "holds TWO competing readings of which frontier cells the band "
            "names -- the first under `band_provenance` (GeoLDM Validity & "
            "Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: "
            "false`, plus YuelBond's per-bond-type classification F1 92.7 % "
            "as a third candidate for the same numeral) and the second under "
            "`sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): "
            "RAW connectivity `C. = 92.98 %`, counted BEFORE the "
            "largest-component retention, with the post-processed "
            "`N. & U. & S. = 94.33 %` companion cell). They do not even agree "
            "on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source "
            "cells are NOT settled on disk. Under BOTH readings the band is "
            "EXTERNAL, is NOT one homogeneous quantity, and is NOT on our "
            "scale -- OUR 8.51 % IS a connected-component count on the RAW "
            "emitted SMILES, and nothing else."
        ),
        # Wf-A180 (2026-11-01) SUPERSEDED RENDERING -- PRESERVED VERBATIM.
        #
        # The string above pointed the reader at a token that is NOT a key
        # of this record, and cited the readings by LINE NUMBER.  It is
        # preserved here verbatim; the live locators are JSON PATHS.
        "canonical_reading_safe_band_statement_superseded_2026_11_01": (
            "The published 92.7-94 % band is EXTERNAL to Mol-Metal (it is a "
            "FRONTIER composite) and it is NOT one homogeneous quantity. The "
            "repo holds TWO COMPETING provenance readings for which cells it "
            "names, and they do not even agree on the numerals: (A) "
            "`frontier_matrix.json` -> `band_provenance` (:51-89): GeoLDM's "
            "Validity & Uniq 92.7 % and HLTF's PB-Valid 94.0 % on QM9, both "
            "`post_processed: false` (plus YuelBond's per-bond-type "
            "classification F1 92.7 % as a third candidate for the same "
            "numeral) -- this reading reproduces the quoted numerals exactly "
            "but makes NO endpoint a connected-component count; (B) the same "
            "file -> `frontier_axis_mapping` (:360, :378-379), EF-TALFM "
            "arXiv:2609.08333v4: the band mixes EF-TALFM's OWN Table 3(a) raw "
            "connectivity cell (C. = 92.98 %, counted BEFORE the "
            "largest-component retention) with its post-processed N. & U. & S. "
            "companion cell (94.33 %) -- this reading has the connectivity+post "
            "shape but does NOT reproduce the quoted numerals (92.98 / 94.33). "
            "Under BOTH readings the band is external, is not one homogeneous "
            "quantity, and is NOT on our scale -- while OUR 8.51 % IS a "
            "connected-component count on the RAW emitted SMILES, and nothing "
            "else. The exact source cells of the band are NOT settled on disk."
        ),
        "canonical_reading_safe_band_statement_superseded_note_2026_11_01": (
            "Wf-A180 (2026-11-01): the rendering above is SUPERSEDED.  It "
            "pointed the reader at `frontier_axis_mapping`, which is NOT a key "
            "of this record -- the token survives only inside a string value, "
            "so pointing a reader at it is a DEFECT.  The ROBUST locators are "
            "the JSON PATHS `band_provenance` and `sources`.`EF-TALFM` (this "
            "record's own `line_number_note` says so). PRESERVED VERBATIM per "
            "AGENTS.md §5.5 rule #6."
        ),
        # Wf-A177 (2026-11-01) -- the cross-references to the sibling
        # records that carry the same contest.  EMITTED so a re-run keeps
        # them.
        #
        # 2026-11-01 (the producer/artefact parity repair): the first two
        # entries carried the Wf-A177-era assertion 'the emitted string is
        # UNCHANGED', which STOPPED being true when Wf-A180 replaced those
        # two strings with the reading-safe renderings; the second also
        # quoted the contested 'NEITHER is a connected-component count'
        # wording in a LIVE (non-dated) rendering, which P4 forbids.  Both
        # are repaired below and their previous values are PRESERVED
        # VERBATIM in `band_provenance`.`producer_reproduces_the_record_2026_11_01`.
        "cross_references": [
            (
                "molmetal/scripts/wfa158_frontier_postproc_matrix.py -> "
                "BAND_PROVENANCE['headline_finding'] (the producer of the contested "
                "rendering; annotated in place by Wf-A177 and REPLACED by Wf-A180 "
                "with the reading-safe rendering; the superseded rendering is "
                "PRESERVED VERBATIM in this record's dated sibling keys)"
            ),
            (
                "molmetal/scripts/wfa158_frontier_postproc_matrix.py -> "
                "sources['HLTF'].cells['METRIC'].note (the sibling rendering that "
                "Wf-A177 annotated in place and Wf-A180 REPLACED with the "
                "reading-safe rendering; the superseded rendering is PRESERVED "
                "VERBATIM in this record's dated sibling keys)"
            ),
            (
                "molmetal/scripts/wfa161_frontier_labelling.py -> module docstring, "
                "Wf-A176 CORRECTION 2 (the same canonical reading-safe statement)"
            ),
            (
                "molmetal/reports/wfa161_frontier_labelling/report.md (Reading 2, "
                "EF-TALFM arXiv:2609.08333v4)"
            ),
            (
                "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md "
                "-> '## Wf-A177 CORRECTION (2026-11-01)' (the appended correction "
                "section)"
            ),
        ],
        # Wf-A177 (2026-11-01) -- the locator-drift note.  It records that
        # the line numbers quoted by the Wf-A177-era renderings are STALE,
        # and that the ROBUST locators are the JSON PATHS.
        #
        # 2026-11-01 RESIDUAL REPAIR (RESIDUAL 1 / L3) -- this note used to
        # assert a set of CURRENT line numbers of THIS record.  That is a
        # PERISHABLE locator: the record keeps growing, so every line number
        # cited INTO it rots, and a note that asserts "current" positions is
        # STALE BY CONSTRUCTION (6 of the 8 positions it named were already
        # wrong when this repair ran).  The LIVE rendering below now carries
        # the DURABLE locator-class rule and NO line number; the superseded
        # rendering is PRESERVED VERBATIM in
        # `line_number_note_superseded_2026_11_01`.
        "line_number_note": (
            "LOCATOR CLASS -- THE DURABLE RULE.  A LINE NUMBER is a "
            "PERISHABLE locator: this record keeps growing, so every line "
            "number cited into it rots, and a note that asserts a set of "
            "current line numbers is STALE BY CONSTRUCTION.  The ROBUST "
            "locator for any part of this record is its JSON PATH -- e.g. "
            "`band_provenance`, `sources`.`EF-TALFM` -- never a line number.  "
            "The line numbers this note used to quote are PRESERVED VERBATIM "
            "as a DATED HISTORICAL record under "
            "`line_number_note_superseded_2026_11_01` and, per reading, under "
            "the dated `historical_line_numbers_measured_2026_11_01` keys: "
            "they are a RECORD OF DRIFT, not locators."
        ),
        "line_number_note_superseded_2026_11_01": (
            "The ':51-89' range cited for Reading A and the ':360, :378-379' "
            "range cited for Reading B are the PRE-Wf-A177 line numbers of THIS "
            "file.  The Wf-A177 key added at :52 shifted every line at or after "
            "the old :52 by +48.  Current positions on disk: Reading A -> "
            "`band_provenance` :50, `band_as_quoted_in_the_repo` :51, "
            "`endpoint_candidates` :100-, `headline_finding` :137; Reading B -> "
            "`sources` :380 -> `EF-TALFM` :381, the EF-TALFM `POST-PROCESSING` "
            "note :408 (which carries the 'C. = 92.98 %' cell), "
            "`extra_observations` :426-427 (which carry the '92.7-94 % band ... "
            "mixes the LOWEST connectivity' sentence).  The ROBUST locators are "
            "the JSON PATHS -- `band_provenance` and `sources`.`EF-TALFM` -- "
            "not the line numbers."
        ),
        "line_number_note_superseded_note_2026_11_01": (
            "2026-11-01 residual repair: the rendering above is SUPERSEDED.  "
            "It asserted the CURRENT line numbers of THIS record -- a "
            "perishable locator that was already stale by construction (6 of "
            "the 8 positions it named were wrong when the repair ran).  The "
            "LIVE rendering is the durable locator-class rule above.  "
            "PRESERVED VERBATIM per AGENTS.md §5.5 rule #6."
        ),
        "readings": [
            {
                "reading": "A",
                "locator": (
                    "molmetal/reports/wfa158_frontier_postproc_matrix/"
                    "frontier_matrix.json -> `band_provenance`"
                ),
                # Wf-A177 (2026-11-01) -- where the reading's keys sat
                # on disk at that date.  It is a RECORD OF DRIFT, not a
                # locator: the robust locator is the JSON PATH above.
                #
                # 2026-11-01 RESIDUAL REPAIR (RESIDUAL 1 / L3) -- the KEY was
                # RENAMED.  `current_line_numbers_on_disk` asserted CURRENCY
                # for line numbers of a file that keeps growing; the name was
                # a lie once the numbers were historical.  The VALUE is
                # PRESERVED VERBATIM; the rename is recorded as a dated
                # supersession in the dated note below.
                "historical_line_numbers_measured_2026_11_01": (
                    "`band_provenance` :50 (object), `band_as_quoted_in_the_repo` :51, "
                    "`endpoint_candidates` :100-, `headline_finding` :137"
                ),
                "historical_line_numbers_measured_note_2026_11_01": (
                    "2026-11-01 residual repair: this key used to be named "
                    "`current_line_numbers_on_disk`.  The KEY was RENAMED to "
                    "`historical_line_numbers_measured_2026_11_01` because the "
                    "name asserted CURRENCY for line numbers of a record that "
                    "keeps growing -- a locator that is STALE BY "
                    "CONSTRUCTION.  The VALUE is PRESERVED VERBATIM (AGENTS.md "
                    "§5.5 rule #6): it is the Wf-A177-era measurement of where "
                    "Reading A's keys sat on disk on 2026-11-01, BEFORE the "
                    "Wf-A182 producer/record reconciliation re-rendered this "
                    "record.  It is a RECORD OF DRIFT, not a locator; the "
                    "ROBUST locator for Reading A is the JSON PATH above."
                ),
                "reproduces_the_quoted_numerals_exactly": True,
                "makes_the_lower_endpoint_a_connected_component_count": False,
                "post_processed": False,
                "endpoints": {
                    "lower": (
                        "GeoLDM's Validity & Uniq on QM9 = 92.7 % "
                        "(arXiv:2603.27113v1, Table 1(b) row 'Val&Uniq (%)', "
                        "column 'GeoLDM') -- a sanitisation-and-duplication rate"
                    ),
                    "third_candidate_for_the_same_numeral": (
                        "YuelBond's multi-class per-bond-type classification "
                        "F1 = 92.7 % (bioRxiv 10.1101/2025.05.06.652517v1)"
                    ),
                    "upper": (
                        "HLTF's PoseBusters validity on QM9 = 94.0 % "
                        "(arXiv:2603.27113v1, Table 1(b) row 'PB-Valid (%)', "
                        "column 'HLTF (ours)') -- a pose verdict; HLTF states "
                        "NO check count for this cell (NOT-STATED-IN-SOURCE)"
                    ),
                },
            },
            {
                "reading": "B",
                "locator": (
                    "molmetal/reports/wfa158_frontier_postproc_matrix/"
                    "frontier_matrix.json -> `sources`.`EF-TALFM`; EF-TALFM "
                    "arXiv:2609.08333v4, Table 3(a)"
                ),
                # Wf-A177 (2026-11-01) -- where the reading's keys sat
                # on disk at that date.  It is a RECORD OF DRIFT, not a
                # locator: the robust locator is the JSON PATH above.
                #
                # 2026-11-01 RESIDUAL REPAIR (RESIDUAL 1 / L3) -- the KEY was
                # RENAMED, value preserved VERBATIM, exactly as for Reading A.
                "historical_line_numbers_measured_2026_11_01": (
                    "`sources` :380 -> `EF-TALFM` :381, "
                    "`cells`.`POST-PROCESSING`.`note` :408, `extra_observations` "
                    ":426-427"
                ),
                "historical_line_numbers_measured_note_2026_11_01": (
                    "2026-11-01 residual repair: this key used to be named "
                    "`current_line_numbers_on_disk`; it was RENAMED to "
                    "`historical_line_numbers_measured_2026_11_01` because the "
                    "name asserted CURRENCY for line numbers of a record that "
                    "keeps growing.  The VALUE is PRESERVED VERBATIM (AGENTS.md "
                    "§5.5 rule #6): the Wf-A177-era measurement of where "
                    "Reading B's keys sat on disk on 2026-11-01, BEFORE the "
                    "Wf-A182 producer/record reconciliation re-rendered this "
                    "record.  It is a RECORD OF DRIFT, not a locator; the "
                    "ROBUST locator for Reading B is the JSON PATH above.  "
                    "The sibling dated key `locator_superseded_2026_11_01` "
                    "(which named a token that is NOT a key of this record) "
                    "is CORRECT and is KEPT."
                ),
                "reproduces_the_quoted_numerals_exactly": False,
                "makes_the_lower_endpoint_a_connected_component_count": True,
                "post_processed": (
                    "MIXED -- the lower cell is a RAW connectivity count, the "
                    "upper cell is a POST-PROCESSED composite"
                ),
                "endpoints": {
                    "lower": (
                        "EF-TALFM's OWN Table 3(a) RAW connectivity cell "
                        "C. = 92.98 %, counted BEFORE the atom-count "
                        "largest-component retention"
                    ),
                    "upper": (
                        "the same table's post-processed companion cell "
                        "N. & U. & S. = 94.33 % (novel + unique + sanitised), "
                        "counted AFTER the retention"
                    ),
                },
                "note": (
                    "92.98 / 94.33 would round to '93.0-94.3', NOT the quoted "
                    "'92.7-94 %'.  This reading has the connectivity+post "
                    "SHAPE but does NOT reproduce the quoted numerals."
                ),
                # Wf-A180 (2026-11-01) SUPERSEDED LOCATOR -- PRESERVED
                # VERBATIM.  It pointed at a token that is NOT a key of
                # this record, so it was not a locator; the JSON PATH
                # above is.
                "locator_superseded_2026_11_01": (
                    "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json "
                    "-> `frontier_axis_mapping` (:360, :378-379); EF-TALFM "
                    "arXiv:2609.08333v4"
                ),
                "locator_superseded_note_2026_11_01": (
                    "Wf-A180 (2026-11-01): the locator above is SUPERSEDED.  "
                    "`frontier_axis_mapping` is NOT a key of this record (it survives "
                    "only inside a string value), so it is not a locator.  The JSON "
                    "PATH `sources`.`EF-TALFM` is.  PRESERVED VERBATIM per AGENTS.md "
                    "§5.5 rule #6."
                ),
            },
        ],
        "the_readings_disagree_on_the_numerals": (
            "Reading A reproduces the quoted '92.7-94 %' exactly (92.7 / 94.0).  "
            "Reading B's cells are 92.98 / 94.33, which round to '93.0-94.3'.  "
            "The two readings therefore cannot both be the source of the "
            "quoted numerals, so the band's source cells are not a single pair "
            "of cells that anyone has established."
        ),
        "the_source_cells_of_the_band_are_not_settled_on_disk": (
            "TRUE.  The band's selection rule -- which paper and which table "
            "cell the repo's literature scan picked -- is NOT recorded on disk "
            "(see this object's own 'what_this_does_not_settle' key).  Neither "
            "reading may be published as the settled provenance of the band."
        ),
        "the_only_claims_that_may_be_published": {
            "i_the_band_is_external": (
                "The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is "
                "a FRONTIER composite, not our measurement."
            ),
            "ii_it_is_not_one_homogeneous_quantity": (
                "It is NOT one homogeneous quantity: even under Reading A its "
                "two endpoints are two different observables (a "
                "sanitisation-and-duplication rate and a PoseBusters validity "
                "verdict -- HLTF states NO check count for the latter, "
                "NOT-STATED-IN-SOURCE), and under Reading B one endpoint is a "
                "raw connectivity count while the other is a post-processed "
                "composite."
            ),
            "iii_it_is_not_on_our_scale": (
                "It is NOT on our scale.  OUR 8.51 % IS a connected-component "
                "count on the RAW emitted SMILES, and nothing else."
            ),
            "iv_the_source_cells_are_not_settled": (
                "The source cells of the band are NOT settled on disk."
            ),
        },
        "why_this_key_exists": (
            "This key exists so that a reader of the band-provenance block "
            "never has to guess which provenance reading an emitted string "
            "assumes.  The repo holds competing provenance readings, and "
            "they do not even agree on the numerals.  The NUMBER of "
            "competing readings is itself NOT settled on disk: this record "
            "enumerates two of them (readings A and B, below), and the "
            "repo's primary-source verification "
            "(molmetal/reports/wfa186_frontier_primary_source/"
            "primary_source_verification.json -> `invariants`."
            "`competing_readings`) records a further reading that likewise "
            "does not reproduce the quoted numerals.  A future reader of "
            "this file must therefore read every band claim through the "
            "lens of the readings recorded here, and must NOT read a "
            "settled count of them into this key."
        ),
        # 2026-11-02 (Wf-A197) -- the superseded rendering is PRESERVED
        # VERBATIM in a dated sibling field (AGENTS.md §5.5 rule #6).  The
        # prior rendering asserted a SETTLED count of the competing
        # readings; the corrected rendering above asserts NO count.
        "why_this_key_exists_superseded_2026_11_02": (
            WFA197_X5_SUPERSEDED_WHY_2026_11_02
        ),
        "why_this_key_exists_superseded_note_2026_11_02": (
            "2026-11-02 residual repair (Wf-A197): the rendering above is "
            "SUPERSEDED.  It asserted a SETTLED count of the competing "
            "readings ('The repo holds TWO competing readings').  The repo "
            "does not settle that count: the repo's own primary-source "
            "verification records a further reading that likewise does not "
            "reproduce the quoted numerals, so the NUMBER of competing "
            "readings is NOT settled on disk.  The corrected rendering "
            "asserts NO count.  The superseded rendering is PRESERVED "
            "VERBATIM in `why_this_key_exists_superseded_2026_11_02`."
        ),
    },
    "endpoint_candidates": [
        {
            "endpoint": "92.7 %",
            "candidate_a": {
                "paper": "GeoLDM, as tabulated by HLTF (arXiv:2603.27113v1)",
                "what_it_is": "Validity & Uniqueness on QM9 -- a per-molecule RDKit-sanitisable AND non-duplicate rate",
                "denominator": "10,000 unconditional QM9 samples",
                "post_processed": False,
                "locator": "arXiv:2603.27113v1, Table 1(b) row 'Val&Uniq (%)', column 'GeoLDM'",
                "quote": "Val&Uniq (%) | 92.7 ± 0.5 | 93.3 ± 0.0 * | 92.9 ± 0.2",
                "status": "VERIFIED against the primary source in this session",
            },
            "candidate_b": {
                "paper": "YuelBond (bioRxiv 10.1101/2025.05.06.652517v1)",
                "what_it_is": "multi-class BOND-TYPE classification F1 on 1,000 coordinate-noised molecules (sigma = 0.2 A)",
                "denominator": "1,000 held-out test molecules with synthetic noise applied by the paper",
                "post_processed": False,
                "locator": "bioRxiv 10.1101/2025.05.06.652517v1, CDG bonds-reconstruction results",
                "quote": "achieving average accuracy, precision, recall, and F1-score of 92.7%, 93.6%, 92.7%, and 92.7%, respectively",
                "status": "VERIFIED against the primary source in this session",
            },
            "note": (
                "TWO PAPERS, TWO COMPLETELY DIFFERENT OBSERVABLES, THE SAME "
                "NUMBER. This branch did NOT establish which one the repo's "
                "band intended, and does not claim to. It established that "
                "BOTH are not connectivity rates."
            ),
        },
        {
            "endpoint": "94 %",
            "candidate_a": {
                "paper": "HLTF (arXiv:2603.27113v1)",
                # 2026-11-01 RESIDUAL REPAIR (RESIDUAL 3 / D8) -- the check
                # COUNT is attributed honestly.  HLTF states NO count for its
                # PB-Valid cell (NOT-STATED-IN-SOURCE); the SIX-check
                # PoseBusters set belongs to EF-TALFM, not to HLTF.  The
                # previous rendering is PRESERVED VERBATIM in
                # `candidate_a_what_it_is_superseded_2026_11_01`.
                "what_it_is": (
                    "PoseBusters validity on QM9 -- a pose verdict over the "
                    "PoseBusters check set.  HLTF states NO CHECK COUNT for "
                    "this cell (NOT-STATED-IN-SOURCE): it cites PoseBusters "
                    "(Buttenschoen et al. 2024) and says only that "
                    "PoseBusters 'applies stricter structural checks than "
                    "RDKit'.  The SIX-check PoseBusters set is EF-TALFM's "
                    "(arXiv:2609.08333v4), not HLTF's."
                ),
                "denominator": "10,000 unconditional QM9 samples",
                "post_processed": False,
                "locator": "arXiv:2603.27113v1, Table 1(b) row 'PB-Valid (%)', column 'HLTF (ours)'",
                "quote": "PB-Valid (%) | 93.1 ± 0.4 | 91.9 ± 0.5 | 94 ± 0.5 *",
                "status": "VERIFIED against the primary source in this session",
            },
            "candidate_a_what_it_is_superseded_2026_11_01": (
                "PoseBusters validity on QM9 -- a SIX-CHECK pose verdict (bond "
                "lengths/angles, steric clash, aromatic and double-bond "
                "flatness, internal energy)"
            ),
            "candidate_a_what_it_is_superseded_note_2026_11_01": (
                "2026-11-01 residual repair: the rendering above is SUPERSEDED.  "
                "It attributed a SIX-check PoseBusters set to HLTF.  HLTF states "
                "NO check count (NOT-STATED-IN-SOURCE); the six-check set is "
                "EF-TALFM's (arXiv:2609.08333v4, Section A.1.4).  PRESERVED "
                "VERBATIM per AGENTS.md §5.5 rule #6."
            ),
            "note": (
                "A pose-verdict rate, NOT a connectivity rate.  The check COUNT "
                "is NOT-STATED-IN-SOURCE for HLTF.  The SIX-check PoseBusters "
                "set that this record previously attributed to HLTF is "
                "EF-TALFM's: arXiv:2609.08333v4 Section A.1.4 counts 'all six "
                "PoseBusters sanity checks used here' and lists them (Reasonable "
                "bond lengths, Reasonable bond angles, Aromatic ring flatness, "
                "Double bond flatness, Reasonable internal energy, No internal "
                "steric clash), with the joint row 'All PoseBusters checks' in "
                "its Table 3(b).  In EF-TALFM's OWN list, Connectivity is a "
                "SEPARATE component metric and NOT one of the six -- so the "
                "94 % is not a connectivity rate; but that list is EF-TALFM's, "
                "not HLTF's."
            ),
            "note_superseded_2026_11_01": (
                "A pose-verdict rate. It contains a connectivity check as ONE "
                "of six checks, but the number is not that check's rate."
            ),
            "note_superseded_note_2026_11_01": (
                "2026-11-01 residual repair: the rendering above is SUPERSEDED.  "
                "Its 'ONE of six checks' phrasing attributed a six-check "
                "PoseBusters set to HLTF and implied the connectivity check is "
                "among them.  HLTF states NO check count (NOT-STATED-IN-SOURCE), "
                "and in EF-TALFM's six-check list Connectivity is NOT one of the "
                "six -- it is a separate component metric.  PRESERVED VERBATIM "
                "per AGENTS.md §5.5 rule #6."
            ),
        },
    ],
    "post_processing_evidence_from_the_frontier_itself": {
        "source": "arXiv:2603.27113v1, Table 2 (GEOM-DRUGS, 100,000 samples)",
        "recipe": "largest fragment + explicit hydrogens + UFF relaxation, cited to Buttenschoen et al. (2025)",
        # 2026-11-01 RESIDUAL REPAIR (RESIDUAL 4 / D3, second half) -- the
        # SCOPE of the recipe, carried with the Wf-A186 primary-source
        # artefact's exact wording and numerals.  The recipe evidence in this
        # block comes from HLTF's Table 2; the band's Reading-A endpoints are
        # QM9 Table 1 cells, and HLTF itself scopes the recipe to GEOM-DRUGS.
        # The wording below is A186's, verbatim; nothing here asserts that the
        # band's provenance is settled.
        "scope_of_the_standardized_recipe": (
            "HLTF's OWN wording scopes the standardized recipe to GEOM-DRUGS "
            "ONLY: the QM9 sentence in Section 4.1 (sections/experiments.tex:8) "
            "carries NO post-processing clause, while the immediately "
            "following GEOM-DRUGS sentence reads 'On GEOM-DRUGS we report "
            "Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on "
            "100,000 samples, both raw and after standardized post-processing "
            "(largest fragment, explicit hydrogens, UFF relaxation)'. The QM9 "
            "Table 1 carries no '+PP' rows at all; the '+PP' rows live in "
            "Table 2 (GEOM-DRUGS, tables/geom_drugs_results.tex, label "
            "tab:geom_main), whose caption declares that 'PP' denotes "
            "standardized post-processing. Both of the band's Reading-A "
            "endpoints -- GeoLDM Val&Uniq and HLTF PB-Valid -- are QM9 Table 1 "
            "cells, so they carry NO post-processing. Under Reading B the "
            "lower endpoint is EF-TALFM's RAW connectivity count and only the "
            "upper companion cell is post-processed, so NEITHER reading's band "
            "endpoint is post-processed in its own right. CAVEAT (not a "
            "refutation): HLTF also applies an OPTIONAL valence repair that "
            "DELETES low-confidence incident bonds; the paper states 'We "
            "report metrics both with and without this repair' and does not "
            "mark which arm Table 1 carries. That repair deletes BONDS, not "
            "components, so it is not the largest-fragment retention the band "
            "question is about."
        ),
        "scope_of_the_standardized_recipe_note_2026_11_01": (
            "2026-11-01 residual repair: this block used to carry the recipe "
            "and the GEOM-DRUGS pairs with NO statement of the recipe's SCOPE, "
            "which left a reader free to assume the recipe applies to the "
            "band's QM9 Table 1 endpoints.  It does not: HLTF scopes it to "
            "GEOM-DRUGS itself.  The wording above is the Wf-A186 "
            "primary-source artefact's, verbatim "
            "(molmetal/reports/wfa186_frontier_primary_source/"
            "primary_source_verification.json -> findings.F2), and it is "
            "ADDITIVE + DATED; nothing above was edited or removed."
        ),
        "raw_to_post_processed_pairs": [
            {"model": "EQGAT-diff", "raw": 59.7, "pp": 84.2, "delta_pp": 24.5},
            {"model": "FlowMol", "raw": 59.8, "pp": 84.2, "delta_pp": 24.4},
            {"model": "GCDM", "raw": 0.2, "pp": 95.2, "delta_pp": 95.0},
            {"model": "GeoLDM", "raw": 2.9, "pp": 69.6, "delta_pp": 66.7},
            {"model": "SemlaFlow", "raw": 87.5, "pp": 93.1, "delta_pp": 5.6},
            {"model": "HLTF", "raw": 85.5, "pp": 92.2, "delta_pp": 6.7},
        ],
        "note": (
            "THE FRONTIER PUBLISHES THE POST-PROCESSING AXIS ITSELF. Six "
            "models, one table, raw and post-processed side by side, under one "
            "pipeline. The published 'validity' number differs from the raw "
            "number by 5.6 to 95.0 points depending on the model. HLTF also "
            "defines the recovery fraction (PP - raw) / (100 - raw) and "
            "reports ~ 46 % for itself. The SAME formula applied to our own "
            "post-processing arm gives (100 - 8.50963) / (100 - 8.50963) = "
            "100 % (i.e. (1 - 0.0851) / (1 - 0.0851) in fractions), because "
            "LargestFragmentChooser keeps exactly one fragment "
            "by construction -- and the SAME file records "
            "uff_ok_rate_over_postprocessed = 0.0, so our 100 % is reached "
            "WITHOUT the UFF third step that the frontier recipe uses."
        ),
        "recovery_fraction_our_side": {
            "formula": (
                "(post_processed - raw) / (1 - raw)  [both terms as FRACTIONS] "
                "== (pp% - raw%) / (100 - raw%)  [the same formula in PERCENT]"
            ),
            "units": "fractions: raw = 0.0851..., post_processed = 1.0 (= 100 %)",
            "raw": 0.08509630565203663,
            "post_processed": 1.0,
            "value": 1.0,
            "caveat": (
                "1.0 BY CONSTRUCTION -- LargestFragmentChooser cannot leave a "
                "multi-fragment molecule. It is a property of the operation, "
                "not a quality claim, and the same file records "
                "fraction_of_raw_mass_discarded = 0.22474803194442472 and a "
                "BIMODAL survivor shape (a drug-sized molecule or a "
                "<= 2-heavy-atom speck)."
            ),
        },
    },
    # Wf-A180 (2026-11-01) SUPERSEDED RENDERING -- PRESERVED VERBATIM.
    #
    # The superseded text asserted that EVERY candidate endpoint of the
    # band is 'rather than a connectivity rate'.  That is TRUE of the
    # candidates listed in THIS block, but it is FALSE as a statement
    # about the band as a whole under Reading B (EF-TALFM's RAW
    # connectivity cell C. = 92.98 %).  The band is not settled, so the
    # statement may not be published as settled.
    #
    # SUPERSEDED (the Wf-A177-live rendering, verbatim):
    #   This branch did NOT establish which paper the repo's band was assembled from, because the band was assembled by a literature scan whose selection rule is not recorded on disk. It DID establish, from the primary sources, that every candidate endpoint of the band is a validity / uniqueness / pose / bond-classification number rather than a connectivity rate -- and that the frontier's own literature publishes the post-processing effect as being worth up to 95 percentage points.
    "what_this_does_not_settle": (
        "This branch did NOT establish which paper the repo's band was "
        "assembled from, because the band was assembled by a literature "
        "scan whose selection rule is not recorded on disk. It DID "
        "establish, from the primary sources, that the candidate "
        "endpoints listed in this block are validity / uniqueness / pose "
        "/ bond-classification numbers rather than connectivity rates -- "
        "and that the frontier's own literature publishes the "
        "post-processing effect as being worth up to 95 percentage "
        "points. It did NOT establish WHICH cells the band's endpoints "
        "name: the band's source cells are NOT settled on disk (see "
        "`band_provenance`.`competing_reading_and_what_this_does_not_"
        "settle`), and under the EF-TALFM reading the band's lower "
        "endpoint IS a raw connectivity count (`C. = 92.98 %`)."
    ),
    # Wf-A180 (2026-11-01) SUPERSEDED RENDERING -- PRESERVED VERBATIM.
    "what_this_does_not_settle_superseded_2026_11_01": (
        "This branch did NOT establish which paper the repo's band was "
        "assembled from, because the band was assembled by a literature "
        "scan whose selection rule is not recorded on disk. It DID "
        "establish, from the primary sources, that every candidate endpoint "
        "of the band is a validity / uniqueness / pose / "
        "bond-classification number rather than a connectivity rate -- and "
        "that the frontier's own literature publishes the post-processing "
        "effect as being worth up to 95 percentage points."
    ),
    "what_this_does_not_settle_superseded_note_2026_11_01": (
        "Wf-A180 (2026-11-01): the rendering above is SUPERSEDED.  It "
        "asserted that EVERY candidate endpoint of the band is 'rather than "
        "a connectivity rate', which is not settled under Reading B.  "
        "PRESERVED VERBATIM per AGENTS.md §5.5 rule #6."
    ),
    # Wf-A180 (2026-11-01) -- the producer/record reconciliation of that
    # wave.  It lived ONLY in the artefact on disk, so a re-run of this
    # producer dropped it.  It is EMITTED here so that a re-run
    # reproduces it (AGENTS.md §5.5 rule #6).
    "producer_and_record_reconciled_2026_11_01": {
        "append_only": (
            "AGENTS.md §5.5 rule #6: every superseded rendering is PRESERVED "
            "VERBATIM in a dated sibling key; nothing was deleted.  This record "
            "was NOT regenerated -- it is patched in place, because a "
            "regeneration would have destroyed the Wf-A177 additive history."
        ),
        "changed_in_this_record": [
            (
                "`band_provenance`.`headline_finding` -- replaced by the "
                "reading-safe rendering; the superseded rendering is PRESERVED "
                "VERBATIM in `headline_finding_superseded_2026_11_01`."
            ),
            (
                "`band_provenance`.`what_this_does_not_settle` -- replaced by the "
                "reading-safe rendering; the superseded rendering is PRESERVED "
                "VERBATIM in `what_this_does_not_settle_superseded_2026_11_01`."
            ),
            (
                "`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement` "
                "-- the `frontier_axis_mapping` pointer (NOT a key of this record) "
                "is replaced by the JSON PATH `sources`.`EF-TALFM`; the superseded "
                "string is PRESERVED VERBATIM in "
                "`canonical_reading_safe_band_statement_superseded_2026_11_01`."
            ),
            (
                "`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings[1]`.`locator` "
                "-- the same pointer repair; the superseded string is PRESERVED "
                "VERBATIM in `readings[1]`.`locator_superseded_2026_11_01`."
            ),
            (
                "`sources`.`HLTF`.`cells`.`METRIC`.`note` -- replaced by the "
                "reading-safe rendering; the superseded rendering is PRESERVED "
                "VERBATIM in `note_superseded_2026_11_01`."
            ),
        ],
        "dated": "2026-11-01",
        "divergence_from_the_producer": (
            "The producer emits, under this key, the SAME object this record "
            "keeps: "
            "`competing_reading_and_what_this_does_not_settle` (canonical "
            "clause + the readings it records + the four invariants + the "
            "extra `line_number_note`, `cross_references` and the two "
            "`historical_line_numbers_measured_2026_11_01` keys), with every "
            "line number DEMOTED into a dated key -- the live locators are "
            "JSON PATHS, and no self-referential line number is asserted as "
            "current.  There is therefore NO divergence between the producer "
            "and this record on this key: the producer's emitted object and "
            "this record's object are BYTE-IDENTICAL (MEASURED on 2026-11-02).  "
            "The two prior renderings of this sentence are PRESERVED "
            "VERBATIM in `divergence_from_the_producer_superseded_2026_11_02` "
            "and in the dated sibling key "
            "`producer_and_record_reconciled_2026_11_01_prior_rendering_"
            "2026_11_02`."
        ),
        # 2026-11-02 (Wf-A197) -- the rendering this repair replaces is
        # preserved VERBATIM in a dated sibling field.
        "divergence_from_the_producer_superseded_2026_11_02": (
            WFA197_D2_SUPERSEDED_DIVERGENCE_2026_11_02
        ),
        "divergence_from_the_producer_superseded_note_2026_11_02": (
            "2026-11-02 residual repair (Wf-A197): the rendering above is "
            "SUPERSEDED and DEMOTED.  It described a divergence between the "
            "producer and this record that does not exist: the producer "
            "emits, under this key, the very object this record keeps (the "
            "two objects are BYTE-IDENTICAL, MEASURED on 2026-11-02).  The "
            "PRIOR rendering of this sentence -- the one the 2026-11-01 "
            "residual repair edited IN PLACE, whose own prior rendering was "
            "preserved NOWHERE -- is recovered VERBATIM in the dated sibling "
            "key `producer_and_record_reconciled_2026_11_01_prior_rendering_"
            "2026_11_02`."
        ),
        "wave": "Wf-A180",
        "what_this_is": (
            "Wf-A180 (2026-11-01) reconciled the PRODUCER "
            "(molmetal/scripts/wfa158_frontier_postproc_matrix.py) with this "
            "RECORD.  Before this wave the producer emitted two contested "
            "renderings as settled fact and did NOT emit "
            "`band_provenance`.`competing_reading_and_what_this_does_not_settle` "
            "at all, so a re-run of the producer would have DROPPED the Wf-A177 "
            "correction and re-emitted the contested claim.  The producer now "
            "emits the reading-safe rendering AND emits the competing-reading "
            "key, so a re-run reproduces this correction."
        ),
    },
    # 2026-11-02 (Wf-A197) -- the PRIOR rendering of the sentence the
    # 2026-11-01 residual repair edited IN PLACE, preserved VERBATIM.
    "producer_and_record_reconciled_2026_11_01_prior_rendering_2026_11_02": {
        "dated": "2026-11-02",
        "wave": "Wf-A197",
        "what_this_is": (
            "The PRIOR rendering of "
            "`band_provenance`.`producer_and_record_reconciled_2026_11_01`."
            "`divergence_from_the_producer`, preserved VERBATIM.  On "
            "2026-11-01 that sentence was edited IN PLACE and the earlier "
            "rendering was preserved NOWHERE -- the AGENTS.md §5.5 rule #6 "
            "violation this key closes."
        ),
        "divergence_from_the_producer": (
            WFA197_D2_PRIOR_RENDERING_2026_11_02
        ),
        "note": (
            "Recovered VERBATIM from the pre-Wf-A194 published bytes of this "
            "record (the Wf-A194 wave's prior-state snapshot, 2026-11-01 "
            "21:16) and re-typed into this producer, so a re-run reproduces "
            "it instead of losing it again.  The rendering that replaced it "
            "is PRESERVED VERBATIM in `producer_and_record_reconciled_"
            "2026_11_01`.`divergence_from_the_producer_superseded_2026_11_02`; "
            "the corrected rendering is in "
            "`producer_and_record_reconciled_2026_11_01`."
            "`divergence_from_the_producer`."
        ),
    },
    # ---------------------------------------------------------------------------
    # 2026-11-01 -- the FOUR REMAINING RESIDUALS of the frontier-band record
    # are closed here.  This key is DATED, so it may carry the superseded
    # renderings VERBATIM (AGENTS.md §5.5 rule #6).  Every rendering this
    # repair replaced is preserved here or in a dated sibling key next to it;
    # nothing was deleted.
    # ---------------------------------------------------------------------------
    "frontier_band_residuals_closed_2026_11_01": {
        "dated": "2026-11-01",
        "wave": (
            "the 2026-11-01 frontier-band residual repair (this record's own "
            "residual close-out; no wave id was assigned, so the DATE is the "
            "identifier)"
        ),
        "append_only": (
            "AGENTS.md §5.5 rule #6: every superseded rendering is PRESERVED "
            "VERBATIM in a DATED key; nothing was deleted and no live "
            "rendering was silently edited.  Each replaced rendering is named "
            "below with the dated key that holds it."
        ),
        "residual_1_live_line_numbers_demoted": {
            "defect": (
                "three LIVE (non-dated) leaves asserted CURRENT line numbers of "
                "THIS record: `band_provenance`."
                "`competing_reading_and_what_this_does_not_settle`."
                "`line_number_note`, and `readings[0]` / `readings[1]`."
                "`current_line_numbers_on_disk`.  A line number cited INTO a "
                "file that keeps growing is a PERISHABLE locator, and a note "
                "asserting 'current' positions is STALE BY CONSTRUCTION -- 6 of "
                "the 8 positions the note named were already wrong when this "
                "repair ran."
            ),
            "repair": (
                "the live `line_number_note` now carries the DURABLE "
                "locator-class rule and NO line number; the two "
                "`current_line_numbers_on_disk` keys were RENAMED to the dated "
                "`historical_line_numbers_measured_2026_11_01` keys, with "
                "their VALUES preserved VERBATIM.  The JSON PATH is now the "
                "live locator for both readings."
            ),
            "superseded_renderings_preserved_verbatim": [
                "`competing_reading_and_what_this_does_not_settle`."
                "`line_number_note_superseded_2026_11_01`",
                "`competing_reading_and_what_this_does_not_settle`."
                "`readings[0]`.`historical_line_numbers_measured_2026_11_01`",
                "`competing_reading_and_what_this_does_not_settle`."
                "`readings[1]`.`historical_line_numbers_measured_2026_11_01`",
            ],
            "kept_unchanged": (
                "`readings[1]`.`locator_superseded_2026_11_01` -- the dated "
                "sibling that names a token which is NOT a key of this record "
                "-- is CORRECT and is KEPT, together with its note."
            ),
        },
        "residual_2_geoldm_is_now_a_first_class_source": {
            "defect": (
                "the record named GeoLDM's cell as one of the band's TWO "
                "endpoints but carried NO GeoLDM source: the 92.7 % was "
                "present in the repo only as a TRANSCRIPT of HLTF's table, "
                "with the ORIGINATING paper absent from `sources`."
            ),
            "repair": (
                "`sources`.`GeoLDM` was ADDED, built the same way as the other "
                "six (same key shape; the same four cells DENOMINATOR / METRIC "
                "/ POST-PROCESSING / HYDROGENS; `reported_numbers` carrying "
                "row, column, value and a sha256-pinned locator), together "
                "with its retrieval-log entries and its pinned quotes.  The "
                "e-print sha256 and the extracted .tex sha256 both MATCH the "
                "pins recorded by the Wf-A186 primary-source artefact."
            ),
            "no_settled_claim_added": (
                "the GeoLDM entry carries NO narrative claim that the band is "
                "settled.  The reading contest is UNCHANGED: the two competing "
                "readings recorded here, plus the catalogue's third, still "
                "stand, and the four publishable invariants are unchanged."
            ),
            "absorbed_findings": {
                "F1_GEOLDM_FIRST_CLASS": (
                    "ESTABLISHED (Wf-A186 findings.F1).  GeoLDM's OWN paper "
                    "(arXiv:2305.01140v1, icml2023.tex:491) reports its own QM9 "
                    "row as Valid & Unique = 92.7 +/- 0.5 over 10,000 samples, "
                    "and HLTF's Table 1(a)/1(b) carry the SAME printed value -- "
                    "HLTF reproduced it, it did not originate it.  ABSORBED "
                    "into `sources`.`GeoLDM`."
                ),
                "F2_HLTF_SCOPES_PP_TO_GEOM_DRUGS": (
                    "ESTABLISHED (Wf-A186 findings.F2).  HLTF's standardized "
                    "post-processing recipe (largest fragment, explicit "
                    "hydrogens, UFF relaxation) is scoped by HLTF ITSELF to "
                    "GEOM-DRUGS ONLY: arXiv:2603.27113v1 "
                    "sections/experiments.tex:8 reads 'On QM9 we report atom "
                    "stability (AS), molecule stability (MS), RDKit validity "
                    "(Val), and Val&Uniq on 10,000 unconditional samples. On "
                    "GEOM-DRUGS we report Valid, Valid&Unique (V&U), and "
                    "Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw "
                    "and after standardized post-processing (largest fragment, "
                    "explicit hydrogens, UFF relaxation)' -- the QM9 Table 1 has "
                    "no '+PP' rows (the '+PP' rows are in Table 2, "
                    "tables/geom_drugs_results.tex, label tab:geom_main).  "
                    "ABSORBED into `band_provenance`."
                    "`post_processing_evidence_from_the_frontier_itself`."
                    "`scope_of_the_standardized_recipe`.  The catalogue row "
                    "docs/HONEST_NEGATIVE_CATALOG.md:20819 -- the class-(a) "
                    "POST-PROCESSING row of the four-breaks table -- asserts "
                    "'the band is measured on the published post-processed "
                    "stream' and is WRONG FOR THIS BAND; it was inherited from "
                    "a DIFFERENT band (catalogue section 6.252, whose band is "
                    "37.4 % to 96.0 % and whose frontier rates ARE "
                    "post-processed).  That row lives in docs/, which this "
                    "repair does NOT touch; the correction is recorded HERE, "
                    "additively and dated, so that this record does not "
                    "repeat the wrong cell.  The row's OUR-SIDE half ('both "
                    "arms are RAW') stands."
                ),
                "F3_GEO_LDM_QM9_CELLS_ARE_RAW": (
                    "PARTIAL (Wf-A186 findings.F3), and the record says so "
                    "explicitly.  ESTABLISHED: GeoLDM's own paper names the "
                    "artefact ('the model occasionally generates disconnected "
                    "components ... can be easily fixed by just filtering the "
                    "smaller components', icml2023.tex:917) and the remedy is "
                    "PROPOSED, not applied.  PARTIAL: the reading that the "
                    "tab:qm9_results cells are RAW is an ARGUMENT FROM "
                    "SILENCE plus that one DRUG-side sentence -- the paper "
                    "never says 'no post-processing', it simply never describes "
                    "a post-processing step, and the sentence says the "
                    "disconnection 'happens more often when trained on the "
                    "large molecule DRUG dataset', i.e. it is a DRUG-side "
                    "observation, not a QM9-side correction.  ABSORBED into "
                    "`sources`.`GeoLDM`.`cells`.`POST-PROCESSING`, which "
                    "labels the reading as an argument from silence."
                ),
                "F4_THE_93_8_VS_93_6_DISAGREEMENT": (
                    "PARTIAL (Wf-A186 findings.F4), and the record says so "
                    "explicitly.  ESTABLISHED: the two source tables "
                    "DISAGREE on the same observable -- GeoLDM's OWN paper "
                    "prints Valid = 93.8 +/- 0.4 (icml2023.tex:491) and HLTF "
                    "prints the GeoLDM column's Val as 93.6 +/- 0.2 "
                    "(tables/qm9_benchmarks_og.tex:38 and :63) -- and neither "
                    "paper flags it; HLTF's Table 1 caption says 'Baselines "
                    "reproduced from GCDM Table 1', so the column is a "
                    "TRANSCRIPT.  PARTIAL, AND THE PART THAT IS REFUTED: A186's "
                    "verdict reads 'ESTABLISHED -- the two source tables "
                    "DISAGREE, and no document on disk records it', and its "
                    "repo-wide scan for '93.6' is reported as returning hits "
                    "ONLY in an unrelated NCI-60 split percentage, 'zero hits "
                    "in the frontier sense'.  THAT SCAN RESULT IS REFUTED: "
                    "both numerals ARE on disk in the frontier sense -- "
                    "molmetal/reports/wfa175_frontier_comparability/report.md"
                    ":40 records GeoLDM's own 93.8 +- 0.4 and the same file's "
                    ":133 records HLTF's GeoLDM-column Val 93.6 +- 0.2, with "
                    "both in molmetal/reports/wfa175_frontier_comparability/"
                    "sources.json at :170 and :417.  What the Wf-A175 record "
                    "does NOT do is FLAG the two as a disagreement -- at "
                    "report.md:138 it calls them 'the same GeoLDM row' without "
                    "noticing the differing Val.  So this record carries the "
                    "ANNOTATION with sha256-pinned locators and does NOT "
                    "repeat the 'no document on disk records it' clause.  "
                    "NEITHER NUMERAL IS A BAND ENDPOINT: under Reading A the "
                    "endpoints are 92.7 / 94.0 and under Reading B they are "
                    "92.98 / 94.33.  ABSORBED into "
                    "`sources`.`GeoLDM`.`extra_observations`."
                ),
            },
        },
        "residual_3_posebusters_check_count_attributed": {
            "defect": (
                "SIX live renderings described HLTF's PB-Valid cell as 'a "
                "SIX-CHECK pose verdict'.  That check count is NOT HLTF's: it "
                "is EF-TALFM's.  The six-check PoseBusters set comes from EF-"
                "TALFM arXiv:2609.08333v4 Section A.1.4 ('all six PoseBusters "
                "sanity checks used here') and its Table 3(b) joint row 'All "
                "PoseBusters checks'; in that list Connectivity is a SEPARATE "
                "component metric and NOT one of the six.  HLTF states NO check "
                "count for its PB-Valid cell: it cites PoseBusters (Buttenschoen "
                "et al. 2024) and says only that PoseBusters 'applies stricter "
                "structural checks than RDKit'."
            ),
            "repair": (
                "the six live renderings now say so, and the count is marked "
                "NOT-STATED-IN-SOURCE for HLTF."
            ),
            "superseded_renderings_preserved_verbatim": [
                "`band_provenance`.`endpoint_candidates`[1]."
                "`candidate_a`.`candidate_a_what_it_is_superseded_2026_11_01`",
                "`band_provenance`.`endpoint_candidates`[1]."
                "`note_superseded_2026_11_01`",
                "`sources`.`HLTF`.`cells`.`METRIC`."
                "`note_superseded_2026_11_01_residual`",
            ],
            "live_renderings_corrected": [
                "`band_provenance`.`endpoint_candidates`[1]."
                "`candidate_a`.`what_it_is`",
                "`band_provenance`.`endpoint_candidates`[1].`note`",
                "`band_provenance`.`competing_reading_and_what_this_does_not_"
                "settle`.`readings`[0].`endpoints`.`upper`",
                "`band_provenance`.`competing_reading_and_what_this_does_not_"
                "settle`.`the_only_claims_that_may_be_published`."
                "`ii_it_is_not_one_homogeneous_quantity`",
                "`sources`.`HLTF`.`cells`.`METRIC`.`note`",
            ],
            "invariant_ii_previous_wording_verbatim": (
                "It is NOT one homogeneous quantity: even under Reading A its "
                "two endpoints are two different observables (a "
                "sanitisation-and-duplication rate and a six-check pose "
                "verdict), and under Reading B one endpoint is a raw "
                "connectivity count while the other is a post-processed "
                "composite."
            ),
            "invariant_ii_note": (
                "the CLAIM of invariant (ii) is UNCHANGED -- the two endpoints "
                "are still two different observables; only the misattributed "
                "check count is corrected, and 'a PoseBusters validity "
                "verdict' replaces 'a six-check pose verdict'.  The four "
                "publishable invariants are otherwise untouched."
            ),
            # 2026-11-02 (Wf-A194) -- APPEND-ONLY close-out of the residue
            # the 2026-11-01 residual repair left behind.  It corrected
            # FIVE live renderings but preserved the superseded rendering
            # of only FOUR of them; the fifth --
            # `band_provenance`.`competing_reading_and_what_this_does_not_
            # settle`.`readings`[0].`endpoints`.`upper` -- had its string
            # replaced with NO dated sibling holding the old string.  The
            # old string is restored here VERBATIM, as it stood in the
            # published artefact on 2026-11-01 (the bytes are the ones
            # measured at
            # molmetal/reports/wfa158_frontier_postproc_matrix/
            # frontier_matrix.json -> `band_provenance` ->
            # `competing_reading_and_what_this_does_not_settle` ->
            # `readings`[0] -> `endpoints` -> `upper`).
            # NOTHING WAS DELETED: the live string keeps its corrected
            # wording, and the superseded string lives on here, dated.
            "endpoints_upper_superseded_rendering_verbatim": (
                "HLTF's PoseBusters validity on QM9 = 94.0 % (arXiv:2603.27113v1, "
                "Table 1(b) row 'PB-Valid (%)', column 'HLTF (ours)') -- a "
                "six-check pose verdict"
            ),
            "endpoints_upper_superseded_note": (
                "2026-11-02 (Wf-A194): the live rendering at "
                "`band_provenance`.`competing_reading_and_what_this_does_not_"
                "settle`.`readings`[0].`endpoints`.`upper` reads 'a pose "
                "verdict; HLTF states NO check count for this cell "
                "(NOT-STATED-IN-SOURCE)'.  The string above is the "
                "SUPERSEDED rendering, PRESERVED VERBATIM (AGENTS.md §5.5 "
                "rule #6): it attributed a six-check PoseBusters set to "
                "HLTF, and that check count is EF-TALFM's, not HLTF's.  "
                "See `residual_3_posebusters_check_count_attributed`."
            ),
        },
        "residual_4_the_anchor_names_its_quantity": {
            "defect": (
                "the anchor was rendered as '8.51 % (0.08509630565203663)' with "
                "NO statement of WHICH quantity that is, although TWO "
                "different quantities in this repo print as '8.51 %'."
            ),
            "repair": (
                "`anchor_our_side`.`which_quantity` names it: QUANTITY B, the "
                "SINGLE-POOL raw connected-component ratio of ONE decode pool "
                "(1,078 pool rows with exactly one RDKit fragment, out of "
                "12,668 parsed rows; 12,669 rows declared in the CSV, 1 "
                "unparseable).  It is NOT the median over the 9 recorded runs "
                "(0.08506944444444445 -- QUANTITY A), which is a DIFFERENT "
                "number that prints as the SAME '8.51 %'.  NO NEW STATISTIC "
                "is introduced: the numerator and the denominator are already "
                "in this record "
                "(`anchor_our_side`.`postprocessed_counterpart`.`denominators`)"
                " and the median is on disk at "
                "metrics/by_metric/cfm_fragmentation_audit.json -> "
                "/summary/generation_side/single_molecule_rate_end_to_end "
                "(n = 9)."
            ),
            "superseded_rendering_preserved_verbatim": (
                "`anchor_our_side`.`which_quantity_superseded_2026_11_01` -- "
                "'8.51 % (0.08509630565203663) -- THE ANCHOR, rendered with no "
                "statement of WHICH quantity it is.'"
            ),
        },
        "the_four_publishable_invariants_are_unchanged": [
            "The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is a FRONTIER composite, not our measurement.",
            "It is NOT one homogeneous quantity: even under Reading A its two endpoints are two different observables (a sanitisation-and-duplication rate and a PoseBusters validity verdict -- HLTF states NO check count for the latter, NOT-STATED-IN-SOURCE), and under Reading B one endpoint is a raw connectivity count while the other is a post-processed composite.",
            "It is NOT on our scale. OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.",
            "The source cells of the band are NOT settled on disk.",
        ],
        "what_this_repair_did_not_do": [
            "It did NOT settle the band's provenance.  The competing readings are exactly as they were.",
            "It did NOT touch docs/, TODO/, paper/, AGENTS.md, registry/ or metrics/.",
            "It did NOT write a corrected catalogue row: the POST-PROCESSING row at docs/HONEST_NEGATIVE_CATALOG.md:20819 is still live and still wrong for THIS band; the correction is recorded HERE, additively and dated.",
            "It did NOT introduce any numeral that was not already on disk or in the Wf-A186 primary-source artefact.",
            "It did NOT add a new statistic for our side: the anchor is still the same published number, now NAMED.",
        ],
    },
}


# ---------------------------------------------------------------------------
# The ONE dated key that records the producer/record reconciliation
# ---------------------------------------------------------------------------
#
# Before this repair the producer and the record on disk had DIVERGED: a
# re-run of this producer emitted 710 leaves where the record carried 739,
# and the markdown renderer emitted NEITHER dated correction section, so a
# re-run would have DELETED the Wf-A177 additive history and both dated
# correction sections.  A producer that cannot reproduce its own artefact is
# a live append-only hazard.
#
# This key is the RECORD of that reconciliation (AGENTS.md §5.5 rule #6).  It
# is emitted INTO the record, and the markdown's third dated section is
# rendered FROM it, so the JSON and the markdown cannot drift apart.
#
# Every number in `byte_evidence` is MEASURED from the artefact this
# producer emits (see :func:`_fill_byte_evidence`).

RECONCILIATION_KEY = "producer_reproduces_the_record_2026_11_01"

RECONCILIATION: Dict[str, Any] = {
    "heading": (
        "Producer/record reconciliation (2026-11-01) -- the producer now "
        "REPRODUCES this record byte-for-byte"
    ),
    "dated": "2026-11-01",
    "wave": (
        "UNRECORDED -- this repair ran outside the repo's numbered waves and "
        "no wave id for it is recorded on disk, so none is invented here "
        "(AGENTS.md §1 rule 1)."
    ),
    "what_this_is": (
        "The producer (`molmetal/scripts/wfa158_frontier_postproc_matrix.py`) "
        "and this record had DIVERGED.  Before this repair the producer "
        "emitted 710 leaves where the record on disk carried 739, so a re-run "
        "would have DELETED the 29 leaves the record carried and the producer "
        "did not emit -- including the whole Wf-A177 additive history -- and "
        "6 live strings differed.  The markdown renderer emitted NEITHER of "
        "the two dated correction sections, so a re-run would have deleted "
        "the entire record of the contest.  The producer now emits every leaf "
        "the record carries, at the same path, with the same value, and "
        "emits the two dated sections verbatim, so a re-run reproduces both "
        "artefacts instead of dropping them.  The repair ALSO closed a "
        "defect the divergence census found in the record itself: the two "
        "`cross_references` entries that named the contested renderings "
        "still asserted, in a LIVE (non-dated) string, that 'the emitted "
        "string is UNCHANGED' -- which stopped being true when Wf-A180 "
        "replaced those strings -- and one of them quoted the contested "
        "'NEITHER is a connected-component count' wording in a live "
        "rendering, which P4 forbids."
    ),
    "append_only": (
        "AGENTS.md §5.5 rule #6: nothing was deleted.  The three contested "
        "claim renderings stay preserved verbatim in the dated sibling keys "
        "this record already carried, and the seven live strings this "
        "repair changed are preserved verbatim in this key's "
        "`superseded_renderings` list.  The eighth changed live string is a "
        "provenance marker, not a rendering: its previous value is quoted "
        "verbatim in this key's `changed_in_this_record`."
    ),
    "what_changed": (
        "Eight live strings changed relative to the producer as it stood "
        "before this repair -- five renderings, two cross-reference "
        "descriptors and one provenance marker -- and nothing else changed: "
        "0 numeric leaf moved and no leaf was deleted.  Seven of the eight "
        "differ from the artefact as it stood before this repair; the "
        "`wave` marker is the exception, because it was made to agree with "
        "the artefact.  The repair writes the producer's own values, after "
        "which the parity check reports 0 string leaves moved and 0 leaves "
        "only on disk.  Leaves were ADDED on both sides -- the 29 the "
        "artefact already carried and the producer did not emit, and the "
        "leaves of this dated key."
    ),
    "why": (
        "A producer that does not reproduce its own artefact silently deletes "
        "the history it cannot reproduce, and the deletion happens on the "
        "next ordinary re-run, with no signal.  Nothing compared the "
        "producer's output against the bytes on disk, so the divergence "
        "survived two waves.  The parity check the producer now runs in "
        "`--check-only` mode is the early warning; this key is the record."
    ),
    "supersedes": (
        "Two statements made by EARLIER dated material become false the moment "
        "this repair regenerates the record and its markdown.  They are "
        "PRESERVED VERBATIM where they were made and are SUPERSEDED here.  "
        "(1) `band_provenance`.`producer_and_record_reconciled_2026_11_01`."
        "`append_only` said 'This record was NOT regenerated -- it is patched "
        "in place, because a regeneration would have destroyed the Wf-A177 "
        "additive history.'  That was TRUE at its date, and it is no longer "
        "true: this repair regenerates the record, from a producer that now "
        "carries the Wf-A177 additive history, so nothing is destroyed.  "
        "(2) The markdown's two dated sections both assert 'nothing above "
        "this line was edited, removed, re-rendered or re-ordered', and the "
        "Wf-A180 section asserts that its own pre-Wf-A180 bytes are a "
        "byte-identical PREFIX.  Those were TRUE at their dates and are "
        "SUPERSEDED: the body above them was re-rendered by this repair.  "
        "The markdown's THIRD dated section -- the one rendered from this key "
        "-- carries the supersession in prose, so the markdown and the "
        "record cannot drift apart."
    ),
    "what_it_does_not_change": (
        "The four invariants that hold under BOTH provenance readings are "
        "unchanged: the published 92.7-94 % band is EXTERNAL to Mol-Metal, it "
        "is NOT one homogeneous quantity, it is NOT on our scale -- OUR 8.51 "
        "% IS a connected-component count on the RAW emitted SMILES, and "
        "nothing else -- and the source cells of the band are NOT settled on "
        "disk.  The per-source protocol breaks, the quotes, the anchor and "
        "the retrieval log are untouched."
    ),
    "producer_now_reproduces_the_artefact": (
        "`molmetal/scripts/wfa158_frontier_postproc_matrix.py` now emits "
        "`frontier_matrix.json` and `frontier_matrix.md` byte-for-byte, and "
        "its `--check-only` mode prints the PARITY report -- byte-identical "
        "or not, numeric leaves moved, leaves only on disk, leaves only in "
        "the producer -- so the next divergence is caught before it can "
        "delete anything."
    ),
    "changed_in_this_record": [
        (
            "`band_provenance`.`headline_finding` -- now names the band "
            "explicitly ('92.7-94 %'); it read 'the published band's "
            "provenance'.  The superseded rendering is PRESERVED VERBATIM in "
            "this key's `superseded_renderings`."
        ),
        (
            "`band_provenance`.`competing_reading_and_what_this_does_not_"
            "settle`.`status` -- now states the four invariants (external / "
            "not one homogeneous quantity / not on our scale / source cells "
            "not settled); it stated none of them.  The superseded rendering "
            "is PRESERVED VERBATIM in this key's `superseded_renderings`."
        ),
        (
            "`band_provenance`.`competing_reading_and_what_this_does_not_"
            "settle`.`why_this_key_exists` -- re-worded: the previous wording "
            "asserted that the sibling keys 'state ONE provenance reading as "
            "if it were settled', which is no longer true after the "
            "reading-safe repair.  The superseded rendering is PRESERVED "
            "VERBATIM in this key's `superseded_renderings`."
        ),
        (
            "`band_provenance`.`competing_reading_and_what_this_does_not_"
            "settle`.`readings[0]`.`locator` -- the stale line-number "
            "parenthetical is dropped; the live locator is the JSON PATH "
            "`band_provenance`, the same rule the reading-B locator already "
            "followed.  The superseded rendering is PRESERVED VERBATIM in this "
            "key's `superseded_renderings`."
        ),
        (
            "`band_provenance`.`competing_reading_and_what_this_does_not_"
            "settle`.`readings[1]`.`note` -- now quotes the band as "
            "'92.7-94 %'; it read '92.7-94'.  The superseded rendering is "
            "PRESERVED VERBATIM in this key's `superseded_renderings`."
        ),
        (
            "`band_provenance`.`competing_reading_and_what_this_does_not_"
            "settle`.`wave` -- now 'Wf-A177'; it read 'Wf-A180'.  The marker "
            "names the wave that CREATED this key, and the record on disk "
            "carries 'Wf-A177'.  The previous value ('Wf-A180') is quoted "
            "verbatim here so that nothing is silently rewritten."
        ),
        (
            "`band_provenance`.`competing_reading_and_what_this_does_not_"
            "settle`.`cross_references[0]` -- the stale assertion 'the "
            "emitted string is UNCHANGED' is corrected: Wf-A180 DID replace "
            "the string it names.  The superseded string is PRESERVED "
            "VERBATIM in this key's `superseded_renderings`."
        ),
        (
            "`band_provenance`.`competing_reading_and_what_this_does_not_"
            "settle`.`cross_references[1]` -- the same stale-assertion "
            "correction, and the quoted contested wording ('NEITHER is a "
            "connected-component count') is removed from the LIVE rendering: "
            "a live rendering may not assert it, only a dated one may "
            "preserve it.  The superseded string is PRESERVED VERBATIM in "
            "this key's `superseded_renderings`."
        ),
    ],
    "superseded_renderings": [
        {
            "path": "band_provenance/headline_finding",
            "rendering": (
                "READING-QUALIFIED: the published band's provenance is NOT settled "
                "on disk: the frontier post-processing matrix record "
                "(`frontier_matrix.json`) holds TWO competing readings of which "
                "frontier cells the band names -- the first under `band_provenance` "
                "(GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both "
                "`post_processed: false`, plus YuelBond's per-bond-type "
                "classification F1 92.7 % as a third candidate for the same "
                "numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM "
                "arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, "
                "counted BEFORE the largest-component retention, with the "
                "post-processed `N. & U. & S. = 94.33 %` companion cell). They do "
                "not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the "
                "band's source cells are NOT settled on disk. Under BOTH readings "
                "the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT "
                "on our scale -- OUR 8.51 % IS a connected-component count on the "
                "RAW emitted SMILES, and nothing else. INDEPENDENTLY OF WHICH "
                "READING HOLDS, the frontier publishes the post-processing axis "
                "itself -- six models, raw and post-processed side by side, deltas "
                "from 5.6 to 95.0 pp (see "
                "`band_provenance`.`post_processing_evidence_from_the_frontier_itself`)."
            ),
            "note": (
                "The live rendering names the band ('92.7-94 %'); this one did not."
            ),
        },
        {
            "path": "band_provenance/competing_reading_and_what_this_does_not_settle/status",
            "rendering": (
                "READING-QUALIFIED -- the two readings below are BOTH on disk and "
                "NEITHER is settled.  This key does not replace, contradict or "
                "amend any sibling key; it records the contest (AGENTS.md §5.5 rule "
                "#6 -- the superseded renderings are PRESERVED VERBATIM and this "
                "record is ADDITIVE)."
            ),
            "note": (
                "The live status states the four invariants (external / not one "
                "homogeneous quantity / not on our scale / source cells not settled "
                "on disk); this one stated none of them."
            ),
        },
        {
            "path": "band_provenance/competing_reading_and_what_this_does_not_settle/why_this_key_exists",
            "rendering": (
                "The sibling keys of this object ('headline_finding', "
                "'endpoint_candidates') state ONE provenance reading as if it were "
                "settled.  The repo holds TWO competing readings, and they do not "
                "even agree on the numerals.  A future reader of this file must "
                "therefore read every band claim through the lens of the two "
                "readings recorded here."
            ),
            "note": (
                "This wording asserted that the sibling keys ('headline_finding', "
                "'endpoint_candidates') 'state ONE provenance reading as if it were "
                "settled'.  That is no longer true after the reading-safe repair, "
                "which is why the live rendering was re-worded."
            ),
        },
        {
            "path": "band_provenance/competing_reading_and_what_this_does_not_settle/readings[0]/locator",
            "rendering": (
                "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json "
                "-> `band_provenance` (:51-89)"
            ),
            "note": (
                "The live locator is the JSON PATH `band_provenance` alone.  The "
                "line-number parenthetical above is STALE -- this object's own "
                "`line_number_note` says so and names the JSON PATHS as the robust "
                "locators."
            ),
        },
        {
            "path": "band_provenance/competing_reading_and_what_this_does_not_settle/readings[1]/note",
            "rendering": (
                "92.98 / 94.33 would round to '93.0-94.3', NOT the quoted "
                "'92.7-94'.  This reading has the connectivity+post SHAPE but does "
                "NOT reproduce the quoted numerals."
            ),
            "note": (
                "The live note quotes the band as '92.7-94 %'; this one read "
                "'92.7-94'."
            ),
        },
        {
            "path": "band_provenance/competing_reading_and_what_this_does_not_settle/cross_references[0]",
            "rendering": (
                "molmetal/scripts/wfa158_frontier_postproc_matrix.py -> "
                "BAND_PROVENANCE['headline_finding'] (the producer of the contested "
                "rendering; annotated in place by Wf-A177, the emitted string is "
                "UNCHANGED)"
            ),
            "note": (
                "The live cross-reference no longer asserts that 'the emitted "
                "string is UNCHANGED': Wf-A180 REPLACED the string it names with "
                "the reading-safe rendering, so that assertion had become false.  "
                "The superseded rendering it named is preserved verbatim in "
                "`band_provenance`.`headline_finding_superseded_2026_11_01`."
            ),
        },
        {
            "path": "band_provenance/competing_reading_and_what_this_does_not_settle/cross_references[1]",
            "rendering": (
                "molmetal/scripts/wfa158_frontier_postproc_matrix.py -> "
                "sources['HLTF'].cells['METRIC'].note (the sibling 'NEITHER is a "
                "connected-component count' rendering; annotated in place by "
                "Wf-A177, the emitted string is UNCHANGED)"
            ),
            "note": (
                "The live cross-reference no longer asserts that 'the emitted "
                "string is UNCHANGED' -- Wf-A180 REPLACED the string it names -- "
                "and it no longer quotes the contested 'NEITHER is a "
                "connected-component count' wording in a LIVE rendering: P4 "
                "forbids that, and only a DATED rendering may preserve it.  The "
                "superseded rendering it named is preserved verbatim in "
                "`sources`.`HLTF`.`cells`.`METRIC`.`note_superseded_2026_11_01`."
            ),
        },
    ],
    "byte_evidence": {
        "md_body_chars": 0,
        "wf_a177_md_section_chars": 0,
        "wf_a180_md_section_chars": 0,
        "record_leaf_count": 0,
        "record_numeric_leaf_count": 0,
        "numeric_leaves_moved_by_this_repair": 0,
    },
}

# ---------------------------------------------------------------------------
# 2026-11-02 (Wf-A194) -- the LOCATOR-CLASS closure and the GeoLDM source
# entry, recorded in ONE dated key (AGENTS.md §5.5 rule #6: nothing is
# deleted; the superseded renderings stay where they are and the corrections
# are ADDITIVE + DATED).
#
# The LOCATOR CLASS: a LINE NUMBER is a PERISHABLE locator into this record,
# because the record keeps growing.  The LIVE locator is a JSON PATH.
#
# The GEO-LDM ENTRY: Reading A's lower endpoint (92.7 %) was present in this
# record only as a TRANSCRIPT of HLTF's table, with the ORIGINATING paper
# absent from `sources`.  `sources`.`GeoLDM` is now a first-class source.
#
# Every fact below is MEASURED and traceable to a path on disk; the source of
# the GeoLDM facts is the Wf-A186 primary-source artefact
# (molmetal/reports/wfa186_frontier_primary_source/
# primary_source_verification.json), which live-fetched arXiv:2305.01140v1
# through the host proxy and recorded its sha256.
# ---------------------------------------------------------------------------
LOCATOR_CLASS_KEY = "locator_class_and_geo_ldm_source_2026_11_02"

LOCATOR_CLASS: Dict[str, Any] = {
    "dated": "2026-11-02",
    "wave": "Wf-A194",
    "heading": (
        "Locator class + GeoLDM source entry (2026-11-02) -- the LIVE "
        "locator is a JSON PATH"
    ),
    "what_this_is": (
        "This DATED key records TWO closures.  (1) LOCATOR CLASS: the LIVE "
        "locators of this record are JSON PATHS; the line numbers that were "
        "presented as current positions are DEMOTED into dated sibling keys "
        "and are a RECORD OF DRIFT, not locators.  (2) The GeoLDM source "
        "entry: Reading A's lower endpoint (92.7 %) was present in this "
        "record only as a TRANSCRIPT of HLTF's table, with the ORIGINATING "
        "paper absent from `sources`; `sources`.`GeoLDM` is now a "
        "first-class source, built the same way as the other six."
    ),
    "the_locator_class_rule": (
        "A LINE NUMBER is a PERISHABLE locator: this record keeps growing, "
        "so every line number cited into it rots, and a note that asserts a "
        "set of CURRENT line numbers is STALE BY CONSTRUCTION.  The LIVE "
        "locator for any part of this record is its JSON PATH -- e.g. "
        "`band_provenance`, `sources`.`EF-TALFM`, `sources`.`GeoLDM` -- "
        "never a line number.  A line number may appear ONLY inside a DATED "
        "historical record that says, in the same place, that it is a dated "
        "historical snapshot."
    ),
    "the_demoted_line_numbers_are_preserved_verbatim_in": [
        "`band_provenance`.`competing_reading_and_what_this_does_not_settle`."
        "`line_number_note_superseded_2026_11_01`",
        "`band_provenance`.`competing_reading_and_what_this_does_not_settle`."
        "`readings[0]`.`historical_line_numbers_measured_2026_11_01`",
        "`band_provenance`.`competing_reading_and_what_this_does_not_settle`."
        "`readings[1]`.`historical_line_numbers_measured_2026_11_01`",
    ],
    "the_drift_measured_on_2026_11_02": {
        "what_this_is": (
            "A DATED HISTORICAL MEASUREMENT, taken on 2026-11-02 against the "
            "artefact as published on 2026-11-01 (the pre-repair bytes).  It "
            "is a RECORD OF DRIFT and is STALE BY CONSTRUCTION: the ACTUAL "
            "positions it names moved again when this repair was published.  "
            "It is NOT a locator -- the LIVE locators are the JSON PATHS "
            "above.  Every position below was MEASURED by reading the file "
            "on disk, not copied from another document."
        ),
        "rows": [
            {
                "cited_position": "`band_provenance`",
                "old_claim": ":50",
                "actual_position_measured_2026_11_02": ":50",
                "verdict": "OK",
            },
            {
                "cited_position": "`band_as_quoted_in_the_repo`",
                "old_claim": ":51",
                "actual_position_measured_2026_11_02": ":51",
                "verdict": "OK",
            },
            {
                "cited_position": "`endpoint_candidates`",
                "old_claim": ":100-",
                "actual_position_measured_2026_11_02": ":107-",
                "verdict": "STALE",
            },
            {
                "cited_position": "`headline_finding`",
                "old_claim": ":137",
                "actual_position_measured_2026_11_02": ":144",
                "verdict": "STALE",
            },
            {
                "cited_position": "`sources`",
                "old_claim": ":380",
                "actual_position_measured_2026_11_02": ":472",
                "verdict": "STALE",
            },
            {
                "cited_position": "`sources`.`EF-TALFM`",
                "old_claim": ":381",
                "actual_position_measured_2026_11_02": ":473",
                "verdict": "STALE",
            },
            {
                "cited_position": (
                    "`sources`.`EF-TALFM`.`cells`.`POST-PROCESSING`.`note`"
                ),
                "old_claim": ":408",
                "actual_position_measured_2026_11_02": ":500",
                "verdict": "STALE",
            },
            {
                "cited_position": "`sources`.`EF-TALFM`.`extra_observations`",
                "old_claim": ":426-427",
                "actual_position_measured_2026_11_02": ":518-519",
                "verdict": "STALE",
            },
        ],
        "summary": (
            "6 of the 8 cited positions were WRONG when the drift was "
            "measured on 2026-11-02.  This is why the line numbers are "
            "DEMOTED rather than corrected: correcting them would only move "
            "the rot."
        ),
    },
    "the_geo_ldm_entry": {
        "what_it_is": (
            "`sources`.`GeoLDM` -- arXiv:2305.01140v1, 'Geometric Latent "
            "Diffusion Models for 3D Molecule Generation'.  It is the "
            "ORIGINATING paper of Reading A's lower endpoint (92.7 %); "
            "before this entry the 92.7 % was present in this record only as "
            "a TRANSCRIPT of HLTF's table, so the record could not say where "
            "the 92.7 % came from."
        ),
        "locator_class": (
            "JSON PATH (`sources`.`GeoLDM`) -- never a line number.  The "
            "locators that point INTO GeoLDM's own source file "
            "(`icml2023.tex:<line>`) name the EXTERNAL primary source and "
            "are pinned by that source file's sha256, not by this record's "
            "line numbers."
        ),
        "why_the_band_is_not_settled_by_it": (
            "Adding the originating paper does NOT settle the band's "
            "provenance: the two competing readings recorded under "
            "`band_provenance`.`competing_reading_and_what_this_does_not_"
            "settle` are UNCHANGED, and neither may be published as the "
            "settled provenance of the band."
        ),
        "the_f1_fact": (
            "ESTABLISHED (Wf-A186, `findings`.`F1`): GeoLDM's OWN paper "
            "reports its own QM9 row as Valid & Unique = 92.7 +/- 0.5 over "
            "10,000 samples, so 92.7 % is GeoLDM's OWN published cell and "
            "HLTF merely reproduces it identically."
        ),
        "the_f2_fact": (
            "ESTABLISHED (Wf-A186, `findings`.`F2`): HLTF's OWN wording "
            "scopes the standardized post-processing recipe to GEOM-DRUGS "
            "ONLY; the QM9 sentence in Section 4.1 "
            "(sections/experiments.tex:8) carries NO post-processing.  Both "
            "of Reading A's endpoints are QM9 Table 1 cells, so neither "
            "carries a post-processing clause."
        ),
        "the_f3_fact_and_why_it_matters_here": (
            "ESTABLISHED (Wf-A186, `findings`.`F3`): GeoLDM's OWN paper DOES "
            "name the artefact -- 'the model occasionally generates "
            "disconnected components ... this phenomenon actually is not a "
            "problem'.  The remedy ('can be easily fixed by just filtering "
            "the smaller components') is PROPOSED, not applied, so GeoLDM's "
            "QM9 numbers are RAW.  THIS ENTRY THEREFORE DOES NOT RESTATE "
            "THE REFUTED 'not a connectivity band' FRAMING: a bare statement "
            "that the band is not a connectivity band would be false in the "
            "way X1 is false, because under Reading B the band's LOWER "
            "endpoint IS EF-TALFM's RAW connectivity cell C. = 92.98 %."
        ),
        "the_f4_disagreement": (
            "ESTABLISHED (Wf-A186, `findings`.`F4`): the two source tables "
            "DISAGREE on the same observable and no document on disk records "
            "it.  GeoLDM's OWN paper reports its QM9 Valid (%) as 93.8 +/- "
            "0.4; HLTF's Table 1(a) / 1(b) tabulate the GeoLDM column's Val "
            "(%) as 93.6 +/- 0.2.  BOTH numerals are recorded in "
            "`sources`.`GeoLDM` and NEITHER is silently picked.  They are NOT "
            "band endpoints under either reading (under Reading A the "
            "endpoints are 92.7 / 94.0, under Reading B they are 92.98 / "
            "94.33), so 93.8 and 93.6 must never be published as band "
            "endpoints."
        ),
        "the_f5_fact": (
            "ESTABLISHED (Wf-A186, `findings`.`F5`): EF-TALFM Table 3(a) is "
            "CONFIRMED VERBATIM -- its columns are UAE-3D / FlowMol / "
            "EF-TALFM, its caption fixes the 10,000 denominator, and "
            "connectivity is reported OF THE RECONSTRUCTED MOLECULE while "
            "the companion rates are computed AFTER the largest-component "
            "retention."
        ),
        "the_retrieval_receipt": {
            "eprint_url": "https://arxiv.org/e-print/2305.01140v1",
            "eprint_sha256": (
                "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23"
            ),
            "eprint_bytes": 655684,
            "retrieved_utc": "2026-11-01T13:05:30Z",
            "title_locator": "icml2023.tex:88",
            "source_file": "icml2023.tex",
            "source_file_sha256": (
                "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b"
            ),
            "source_of_the_receipt": (
                "molmetal/reports/wfa186_frontier_primary_source/"
                "primary_source_verification.json -> `retrieval`"
            ),
        },
    },
    "the_md_section_is_appended": (
        "The same statements are APPENDED to `frontier_matrix.md` as a dated "
        "section, so the human-readable rendering carries the SAME "
        "reading-safe statement as this record."
    ),
    "what_this_repair_does_not_change": (
        "The four invariants that hold under BOTH provenance readings are "
        "unchanged: the published 92.7-94 % band is EXTERNAL to Mol-Metal, it "
        "is NOT one homogeneous quantity, it is NOT on our scale -- OUR 8.51 "
        "% IS a connected-component count on the RAW emitted SMILES, and "
        "nothing else -- and the source cells of the band are NOT settled on "
        "disk.  The five pre-existing source entries are UNCHANGED; "
        "`sources`.`GeoLDM` was ADDED beside them.  The numeric leaves that "
        "existed before this repair did NOT MOVE; the repair ADDED new "
        "numeric leaves under `sources`.`GeoLDM`, so the 'TEXT-ONLY plus NEW "
        "dated keys' wording of the 2026-11-01 reconciliation section "
        "above describes THAT repair and not this one."
    ),
    # 2026-11-02 (Wf-A197) -- the sources census in the paragraph above is
    # WRONG on the numeral and FALSE on the claim.  The paragraph itself is
    # LEFT VERBATIM: it is rendered into the dated 2026-11-02 markdown
    # section, and a dated section is not rewritten in place (AGENTS.md
    # §5.5 rule #6).  The corrected sentence is emitted here, in a dated
    # sibling field, and in the dated key
    # `band_provenance`.`record_residuals_closed_2026_11_02` that this
    # repair appends -- which the markdown carries as its own dated
    # section.
    "the_sources_census_correction_2026_11_02": {
        "what_it_is": (
            "The CORRECTED sources census for the Wf-A194 repair, "
            "MEASURED on 2026-11-02 by diffing the pre-Wf-A194 published "
            "bytes of this record against the published bytes, leaf by "
            "leaf, source entry by source entry."
        ),
        "superseded_sentence_preserved_verbatim": (
            WFA197_D1_SUPERSEDED_SENTENCE_2026_11_02
        ),
        "corrected_sentence": (
            "The SIX pre-existing source entries are UNCHANGED except "
            "`sources`.`HLTF`, whose `cells`.`METRIC`.`note` was "
            "rewritten and to which TWO leaves were added; the other "
            "five are byte-identical.  `sources`.`GeoLDM` was ADDED "
            "beside them."
        ),
        "measured_on": "2026-11-02",
        "note": (
            "The superseded sentence is PRESERVED VERBATIM above and is "
            "NOT a live claim: this key supersedes it on that point.  The "
            "paragraph it came from is left VERBATIM because it is "
            "rendered into the dated 2026-11-02 markdown section."
        ),
    },
}


SOURCES: Dict[str, Dict[str, Any]] = {
    "EF-TALFM": {
        "label": "EF-TALFM -- Equivariant-Free Transformer-Autoencoded Latent Flow Matching",
        "citation": {
            "arxiv_id": "2609.08333",
            "version_used": "v4 (latest; published 2026-09-08, updated 2026-10-21)",
            "title": "Fixed-Dimensional Latent Flow for Generating Variable-Size 3D Molecules",
            "authors": "Weichi Yao, Cameron Gruich, Bryan R. Goldsmith, Yixin Wang",
            "venue": "arXiv preprint (no journal-ref in the arXiv API record)",
            "doi": None,
            "retrieved_at": RETRIEVAL_DATE,
            "retrieval_url": "https://arxiv.org/html/2609.08333v4",
        },
        "representation": "3D heavy-atom scaffold; unconditional generation on PCQM4Mv2",
        "reported_band": {
            "headline": "C. (connected) 92.98 %",
            "companion_rates": "S. 98.25 % | U. & S. 98.21 % | N. & U. & S. 94.33 %",
            "table": "Table 3 (a) (identical numbers in Table 4, Appendix I of the v1 layout)",
            "sample_nature": "training-distribution (unconditional PCQM4Mv2), NOT held-out",
        },
        "reported_numbers": [
            {
                "label": "C. connected (EF-TALFM)",
                "value": 92.98,
                "unit": "percent",
                "locator": "arXiv:2609.08333v4, Table 3 (a) row 'C.', column 'EF-TALFM'",
                "quote": "C. | 99.60 | 100.00 | 92.98",
            },
            {
                "label": "S. strict RDKit sanitisation (EF-TALFM)",
                "value": 98.25,
                "unit": "percent",
                "locator": "arXiv:2609.08333v4, Table 3 (a) row 'S.', column 'EF-TALFM'",
                "quote": "S. | 83.84 | 98.72 | 98.25",
            },
            {
                "label": "U. & S. unique sanitised (EF-TALFM)",
                "value": 98.21,
                "unit": "percent",
                "locator": "arXiv:2609.08333v4, Table 3 (a) row 'U. & S.', column 'EF-TALFM'",
                "quote": "U. & S. | 83.81 | 98.27 | 98.21",
            },
            {
                "label": "N. & U. & S. novel+unique+sanitised (EF-TALFM)",
                "value": 94.33,
                "unit": "percent",
                "locator": "arXiv:2609.08333v4, Table 3 (a) row 'N. & U. & S.', column 'EF-TALFM'",
                "quote": "N. & U. & S. | 76.03 | 69.83 | 94.33",
            },
            {
                "label": "sample size for Table 3 (a)",
                "value": 10000,
                "unit": "molecules",
                "locator": "arXiv:2609.08333v4, Table 3 caption",
                "quote": "Panel (a) rates are computed over 10,000 generated molecules.",
            },
            {
                "label": "C. connected (UAE-3D baseline, same table)",
                "value": 99.6,
                "unit": "percent",
                "locator": "arXiv:2609.08333v4, Table 3 (a) row 'C.', column 'UAE-3D'",
                "quote": "C. | 99.60 | 100.00 | 92.98",
            },
            {
                "label": "C. connected (FlowMol baseline, same table)",
                "value": 100.0,
                "unit": "percent",
                "locator": "arXiv:2609.08333v4, Table 3 (a) row 'C.', column 'FlowMol'",
                "quote": "C. | 99.60 | 100.00 | 92.98",
            },
            {
                "label": "connected before strict sanitisation, property-conditioned population",
                "value": 92221,
                "unit": "molecules (of 100,000 generated)",
                "locator": "arXiv:2609.08333v4, Appendix B.1.2 'Screening, selection and evaluation populations' -- 'The initial population contains 100,000 generated molecular structures, with 10,000 samples at each of the ten conditioning targets. Among the 92,221 connected molecules, 90,807 reconstructed graphs pass strict sanitization before geometry refinement.' NOTE: this is the ALL-ATOM, property-CONDITIONED population (Appendix B.1.1), not the unconditional heavy-atom population of Table 3 (a); it is evidence about the ORDER of operations, not a second reading of the 92.98 % cell.",
                "quote": "Among the 92,221 connected molecules, 90,807 reconstructed graphs pass strict sanitization before geometry refinement.",
            },
        ],
        "cells": {
            "DENOMINATOR": {
                "verdict": "BREAKS",
                "quote": "Panel (a) rates are computed over 10,000 generated molecules.",
                "locator": "arXiv:2609.08333v4, Table 3 caption; also Table 4 caption ('All rates are computed over 10,000 generated molecules.') and Section A.1.4",
                "note": (
                    "10,000 molecules drawn from the TRAINING-DISTRIBUTION model "
                    "(unconditional PCQM4Mv2, 'we model only the heavy-atom "
                    "scaffold'), over a '90%/10% train/validation split after "
                    "preprocessing'. Their denominator is 'all generated "
                    "samples'; ours is one decode. Their evaluation is NOT "
                    "held-out -- it is unconditional generation from the "
                    "training distribution."
                ),
            },
            "METRIC": {
                "verdict": "BREAKS",
                "quote": "Connectivity: % of molecules with exactly one connected component, i.e., one fragment.",
                "locator": "arXiv:2609.08333v4, Section A.1.4 'Evaluation metrics' (same bullet in the v1 layout: Appendix G.3)",
                "note": (
                    "Their 'Connectivity' is a property of the paper's OWN "
                    "rule-based graph reconstruction (Appendix D, Algorithm 4) "
                    "from the decoded coordinates plus the decoder's predicted "
                    "chemical attributes -- 'a deterministic chemistry-aware "
                    "reconstruction procedure that is separate from the learned "
                    "generator'. Ours is an independent RDKit "
                    "connected-component count on the raw emitted SMILES. Two "
                    "different instruments, not two estimates of one quantity."
                ),
            },
            "POST-PROCESSING": {
                "verdict": "BREAKS",
                "quote": "We report connectivity of the reconstructed molecule and sanitization, uniqueness, and novelty rates computed after retaining the atom-count largest connected component of each sampled molecule and applying strict RDKit sanitization.",
                "locator": "arXiv:2609.08333v4, Table 4 caption (Appendix A.3); corroborated by the Table 3 caption sentence 'For fragmented outputs, SMILES-based and PoseBusters metrics are computed on the atom-count largest connected component after RDKit sanitization.' and by Appendix D ('we retain the largest connected component and discard the rest, since the reconstructed graph may contain multiple components.')",
                "recipe_verdict": "STATED",
                "recipe": (
                    "largest-connected-component retention (atom-count) -> strict "
                    "RDKit sanitisation -> (for the geometry rows) a "
                    "hydrogen-assisted restrained force-field refinement in "
                    "which temporary explicit hydrogens are added, all heavy "
                    "atoms are restrained near their decoded positions during "
                    "minimisation, and the temporary hydrogens are discarded "
                    "afterwards. Connectivity (C.) is counted BEFORE the "
                    "retention; every companion cell in the same table is "
                    "counted AFTER it."
                ),
                "note": (
                    "MIXED, and the nuance matters. The C. = 92.98 % cell is "
                    "measured on the reconstructed graph BEFORE the "
                    "largest-component retention and BEFORE strict "
                    "sanitisation -- two independent pieces of evidence: (i) "
                    "the pipeline always retains the largest component, so a "
                    "post-retention count would be 100 % by construction, and "
                    "the paper reports 92.98 %; (ii) the property-conditioned "
                    "population in Appendix B.1.2 counts '92,221 connected "
                    "molecules' whose reconstructed graphs 'pass strict "
                    "sanitization before geometry refinement', i.e. connected "
                    "is counted first. "
                    "BUT every companion cell in the same table (S., U. & S., "
                    "N. & U. & S.) IS post-processed -- largest-component "
                    "retention + strict RDKit sanitisation -- and the "
                    "PoseBusters rows run on the N. & U. & S. subset with a "
                    "different denominator, after a hydrogen-assisted "
                    "restrained force-field refinement stage. So the "
                    "'92.7-94 % band' that mixes 92.98 (raw connectivity) with "
                    "94.33 (post-processed novel+unique+sanitised) is NOT one "
                    "homogeneous quantity. ON OUR SIDE the same operation is "
                    "worth 8.51 % -> 100 % (see ANCHOR.postprocessed_counterpart), "
                    "which is why the post-processing axis is the whole gap."
                ),
            },
            "HYDROGENS": {
                "verdict": "DOES-NOT-BREAK",
                "quote": "Following common practice in 3D small-molecule generation, we model only the heavy-atom scaffold.",
                "locator": "arXiv:2609.08333v4, Section 3.1.1 'Benchmark dataset'; corroborated by Appendix D ('the heavy-atom branch, which returns an implicit-hydrogen molecule') and by 'For heavy-atom generation, RDKit first adds temporary explicit hydrogens ... the temporary hydrogens are discarded'",
                "note": (
                    "Their connectivity count is on an IMPLICIT-hydrogen "
                    "heavy-atom graph, same as ours. Explicit hydrogens are "
                    "added only temporarily for the restrained force-field "
                    "refinement and are discarded again. This is a case where "
                    "the main thread's break-4 does NOT apply to this source."
                ),
            },
        },
        "caveats_stated_by_the_source": [
            "The Table 3 caption reports the per-cell denominators explicitly (10,000 for panel (a); the N. & U. & S. counts 7,603 / 6,983 / 9,433 for panel (b)).",
            "Table 3 caption: 'Missing internal-energy evaluations (45 for UAE-3D, 12 for FlowMol, and 191 for EF-TALFM) are counted as failures in both the internal-energy and joint pass rates.'",
            "The illustrated molecules are selected through a separate chemical screen, not sampled blindly.",
        ],
        "extra_observations": [
            "THE BAND IS NOT HOMOGENEOUS EVEN INSIDE ONE TABLE. In Table 3 (a) "
            "the connectivity (C.) column reads UAE-3D 99.60 | FlowMol 100.00 | "
            "EF-TALFM 92.98 -- so the same 10,000-sample protocol yields "
            "connectivity between 92.98 % and 100 % depending on the model, "
            "and the frontier's OWN best-practice baseline (FlowMol, the "
            "model whose numbers EF-TALFM is compared against) sits at 100.00. "
            "The '92.7-94 %' band the repo quotes mixes the LOWEST "
            "connectivity in that table with a post-processed companion cell "
            "(N. & U. & S. = 94.33).",
            "EF-TALFM is the paper's own contribution and it reports the "
            "LOWEST raw connectivity of the three models it compares -- so "
            "'the frontier reports 92.7-94 %' is not a statement about what "
            "the frontier achieves, it is a statement about which cells of "
            "which tables were selected.",
        ],
    },
    "FlowMol3": {
        "label": "FlowMol3 -- Flow Matching for 3D De Novo Small-Molecule Generation",
        "citation": {
            "arxiv_id": "2508.12629",
            "version_used": "v1 (only version on arXiv; published 2025-08-18)",
            "title": "FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation",
            "authors": "Ian Dunn and David R. Koes",
            "venue": "arXiv:2508.12629; the branch brief also cites Digital Discovery 2026, 5(5):2052-2066 (NOT independently verified in this session -- the arXiv API record carries no journal-ref)",
            "doi": None,
            "retrieved_at": RETRIEVAL_DATE,
            "retrieval_url": "https://arxiv.org/html/2508.12629v1",
        },
        "representation": "3D all-atom (explicit H) conformer; unconditional generation on GEOM-Drugs",
        "reported_band": {
            "headline": "% Valid 99.9 +/- 0.1 | % PB-Valid 91.9 +/- 0.7",
            "table": "Table 1 (row 'FlowMol3')",
            "sample_nature": "training-distribution (unconditional GEOM-Drugs), NOT held-out",
        },
        "reported_numbers": [
            {
                "label": "% Valid (FlowMol3)",
                "value": 99.9,
                "unit": "percent (mean +/- 95 % CI)",
                "locator": "arXiv:2508.12629v1, Table 1 row 'FlowMol3', column '% Valid'",
                "quote": "FlowMol3 | 99.9 ± 0.1 | 91.9 ± 0.7 | 0.27 ± 0.03 | 0.10 ± 0.01 | 3.83 ± 0.08 | 0.39 ± 0.01 | 6",
            },
            {
                "label": "% PB-Valid (FlowMol3)",
                "value": 91.9,
                "unit": "percent (mean +/- 95 % CI)",
                "locator": "arXiv:2508.12629v1, Table 1 row 'FlowMol3', column '% PB-Valid'",
                "quote": "FlowMol3 | 99.9 ± 0.1 | 91.9 ± 0.7 | 0.27 ± 0.03 | 0.10 ± 0.01 | 3.83 ± 0.08 | 0.39 ± 0.01 | 6",
            },
            {
                "label": "% PB-Valid on the TRAINING data (reference row)",
                "value": 93.2,
                "unit": "percent (mean +/- 95 % CI)",
                "locator": "arXiv:2508.12629v1, Table 1 row 'training data', column '% PB-Valid'; caption explains it is the mean over 10 repeated random samplings of 15 thousand molecules",
                "quote": "training data | 100.00 | 93.2 ± 0.1 | 0.00 | 0.05 | 0.00 | 0.00 |",
            },
            {
                "label": "baseline sample size (baselines except ADiT)",
                "value": 5000,
                "unit": "molecules",
                "locator": "arXiv:2508.12629v1, Section 3.5 'Comparison to Existing Methods'",
                "quote": "For all baseline models except ADiT we sampled 5000 molecules from the trained models using default settings provided by the authors.",
            },
            {
                "label": "ADiT sample size (author-provided collection)",
                "value": 10000,
                "unit": "molecules",
                "locator": "arXiv:2508.12629v1, Section 3.5",
                "quote": "For ADiT we used a collection of 10 thousand molecules provided by the authors.",
            },
            {
                "label": "FlowMol3 row sample size (authors' own eval harness command)",
                "value": 5000,
                "unit": "molecules, split into 5 subsets of 1000",
                "locator": "github.com/dunni3/FlowMol :: fm3_evals/readme.md (the author's own command line) and fm3_evals/baselines/compute_baseline_comparison.py ('mols_per_subset = len(rdkit_mols) // args.n_subsets'). NOTE: the paper itself does NOT state the FlowMol3 row's denominator.",
                "quote": "--n_timesteps=250 --metrics --n_mols=5000 --reos_raw --n_subsets=5",
            },
        ],
        "cells": {
            "DENOMINATOR": {
                "verdict": "BREAKS",
                "quote": "For all baseline models except ADiT we sampled 5000 molecules from the trained models using default settings provided by the authors.",
                "locator": "arXiv:2508.12629v1, Section 3.5; plus github.com/dunni3/FlowMol :: fm3_evals/readme.md for the FlowMol3 row's own --n_mols=5000 --n_subsets=5",
                "note": (
                    "5,000 molecules from the TRAINING-DISTRIBUTION model "
                    "(unconditional GEOM-Drugs), split into 5 random subsets of "
                    "1,000 for the confidence interval. The paper states the "
                    "baseline denominators but NOT the FlowMol3 row's own "
                    "denominator -- that number comes from the authors' "
                    "public evaluation harness, and the code makes the "
                    "denominator exact: "
                    "metrics['frac_valid_mols'] = metrics['frac_valid_mols'] "
                    "* len(sampled_mols) / n_raw_samples, i.e. EVERY raw "
                    "sample counts, including those that failed to build a "
                    "molecule at all. Ours is one decode."
                ),
            },
            "METRIC": {
                "verdict": "BREAKS",
                "quote": "We report “% Valid” which is the percent of molecules that can be sanitized by rdkit.",
                "locator": "arXiv:2508.12629v1, Section 3.2.1 'Validity'; the connectivity criterion is the embedded PoseBusters check 'whether the molecule forms one component (not fragmented into separate molecules)'",
                "note": (
                    "Their headline pair is a TOOLKIT-LEVEL sanitisation rate "
                    "and a POSE-GEOMETRY verdict (six PoseBusters checks "
                    "including an internal-energy comparison against an "
                    "ETKDGv3+UFF ensemble). The 'one component' criterion "
                    "inside % PB-Valid is one check among six, on a 3D "
                    "all-atom conformer. Ours is an RDKit connected-component "
                    "count on a SMILES string -- there is no conformer and no "
                    "geometry in our measurement at all. The paper's own "
                    "caveat: 'We argue that producing “valid” molecules is a "
                    "necessary but insufficient condition for a practically "
                    "useful molecular generative model.'"
                ),
            },
            "POST-PROCESSING": {
                "verdict": "BREAKS",
                "quote": "Chem.SanitizeMol(largest_mol)",
                "locator": "github.com/dunni3/FlowMol :: flowmol/analysis/metrics.py :: SampleAnalyzer.compute_validity (the script the paper names: 'Metrics on the sampled molecules were then computed using the same script in the FlowMol repository.')",
                "recipe_verdict": "STATED",
                "recipe": (
                    "Chem.rdmolops.GetMolFrags(rdmol, asMols=True, "
                    "sanitizeFrags=False) -> pick the fragment with the most "
                    "atoms -> Chem.SanitizeMol(largest_mol) -> "
                    "Chem.MolToSmiles(largest_mol); the SDF loader is "
                    "Chem.SDMolSupplier(..., sanitize=False, removeHs=False). "
                    "NO deduplication. The PoseBusters run is on the RAW "
                    "molecule, not on the largest fragment."
                ),
                "note": (
                    "THE CENTRAL FINDING FOR THIS CELL, and it is MEASURED FROM "
                    "THE AUTHORS' OWN CODE. In compute_validity the "
                    "connectedness counter and the validity counter are "
                    "different measurements of the same samples: "
                    "'if len(mol_frags) > 1: error_message['disconnected'] += 1 "
                    "else: n_connected += 1' increments n_connected ONLY for "
                    "single-fragment molecules, while n_valid is incremented "
                    "AFTER 'largest_mol = max(mol_frags, default=rdmol, "
                    "key=lambda m: m.GetNumAtoms())' and 'Chem.SanitizeMol"
                    "(largest_mol)'. So a FRAGMENTED molecule whose largest "
                    "fragment sanitises counts as VALID but NOT connected. "
                    "'% Valid' (99.9 +/- 0.1) IS the post-processed number, "
                    "with denominator 'n_valid / len(sampled_molecules)' = ALL "
                    "raw samples. "
                    "AND THE PAPER NEVER REPORTS THE RAW COUNTER: exhaustive "
                    "grep of the extracted paper text shows the word "
                    "'connect' appears only in 'fully-connected graph' "
                    "descriptions and in the PoseBusters check list; there is "
                    "no connectivity or fragmentation RATE in Table 1, even "
                    "though the authors' harness computes 'frac_connected', "
                    "'avg_frag_frac' and 'avg_num_components'. "
                    "The fragments are taken with 'sanitizeFrags=False', and "
                    "the SDF loader is 'Chem.SDMolSupplier(..., sanitize=False, "
                    "removeHs=False)'. NO deduplication happens before the "
                    "count (unlike EF-TALFM's U. & S. / N. & U. & S.). The "
                    "PoseBusters run is on the RAW molecule "
                    "('rdmols = [sample.rdkit_mol for sample in "
                    "sampled_molecules]'), so the 'one component' check "
                    "inside % PB-Valid is NOT largest-fragment "
                    "post-processed -- it is a raw check. Training-side "
                    "kekulisation is also in the pipeline: 'we kekulize all "
                    "molecules in the dataset and do not explicitly model "
                    "aromatic bonds.'"
                ),
            },
            "HYDROGENS": {
                "verdict": "BREAKS",
                "quote": "We train models on the GEOM-Drugs dataset 57 using explicit hydrogens.",
                "locator": "arXiv:2508.12629v1, Section 3.1 'Training Data'; Table 1 caption ('All models were trained on GEOM-Drugs with explicit hydrogens.'); github.com/dunni3/FlowMol :: flowmol/analysis/molecule_builder.py :: build_molecule (every sampled atom, hydrogens included, becomes an RDKit atom node) and fm3_evals/baselines/compute_baseline_comparison.py ('Chem.SDMolSupplier(..., sanitize=False, removeHs=False)')",
                "note": (
                    "Their molecules carry EXPLICIT hydrogen atoms as graph "
                    "nodes, so the fragment count, the sanitisation and the "
                    "PoseBusters checks are all computed on an explicit-H "
                    "graph. Ours is implicit-H. (Their PoseBusters geometry "
                    "module separately sets 'ignore_hydrogens: True' and the "
                    "loader 'add_hs: False', so no hydrogen is added or removed "
                    "by the evaluation harness -- the explicit hydrogens are "
                    "the model's own output.)"
                ),
            },
        },
        "caveats_stated_by_the_source": [
            "Section 3.2.2: 'We argue that producing “valid” molecules is a necessary but insufficient condition for a practically useful molecular generative model.'",
            "Section 4.1: 'the difference in chemistry between generated molecules and “real” molecules is still practically significant.'",
            "The Table 1 caption flags that the training-data PB-Valid reference is itself a sampled estimate (10 repeated random samplings of 15 thousand molecules) because of computational constraints.",
        ],
        "extra_observations": [
            "THE FRONTIER'S OWN HARNESS COMPUTES THE RAW CONNECTIVITY NUMBER "
            "AND THE PAPER DOES NOT PRINT IT. 'frac_connected' (raw, "
            "single-fragment only) and 'avg_frag_frac' (mean fraction of "
            "atoms retained by the largest-fragment step) are both computed by "
            "SampleAnalyzer.compute_validity, and the FlowMol3 PAPER reports "
            "neither: exhaustive grep of the extracted paper text finds "
            "'connect' only in 'fully-connected graph' (the model's graph "
            "representation) and in the PoseBusters check list, and "
            "'fragment' only in a reference title and in the same PoseBusters "
            "sentence. The published table carries % Valid and % PB-Valid -- "
            "the first post-processed, the second pose-based.",
            "'avg_frag_frac' IS THE ONE LIKE-FOR-LIKE NUMBER ACROSS BOTH "
            "SIDES, and the frontier does not publish its value either: our "
            "own counterpart is the postproc-arm file's "
            "/mass_retention/heavy_atom_retention_mean = 0.7395022201805989 "
            "(73.95 % of heavy atoms kept, mean; median 0.926829268292683). "
            "The FlowMol3 VALUE of avg_frag_frac was NOT retrieved: it is not "
            "in the paper, and this branch did not run their sampler.",
            "The paper's own framing of the headline metric is a caveat, not "
            "a claim: 'We argue that producing \"valid\" molecules is a "
            "necessary but insufficient condition for a practically useful "
            "molecular generative model.'",
        ],
        "not_retrieved": [
            {
                "verdict": "NOT-RETRIEVED",
                "what": "the journal version the branch brief cites -- 'FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation', Digital Discovery 2026, 5(5):2052-2066",
                "url": "the DOI supplied in the branch brief; NOT fetched in this session",
                "http_status": None,
                "reason": (
                    "not fetched: the branch brief supplied the citation, and "
                    "the arXiv API record for 2508.12629 carries NO journal-ref, "
                    "so the published version could not be located from the "
                    "metadata that WAS retrieved"
                ),
                "consequence": (
                    "every FlowMol3 cell above is stated for arXiv v1 plus the "
                    "authors' public evaluation harness at github.com/dunni3/"
                    "FlowMol @ main. The Digital Discovery published protocol "
                    "was NOT verified in this session."
                ),
            }
        ],
    },
    "GGFlow": {
        "label": "GGFlow -- graph flow matching with optimal transport",
        "citation": {
            "arxiv_id": "2411.05676",
            "version_used": "v1 (only version on arXiv; published 2024-11-08)",
            "title": "Improving Molecular Graph Generation with Flow Matching and Optimal Transport",
            "authors": "not extracted in this session",
            "venue": (
                "arXiv:2411.05676. The branch brief describes the retitled "
                "TMLR 2026 version 'GGFlow: A Graph Flow Matching Method with "
                "Efficient Optimal Transport' (OpenReview K8RlXtMgzo) -- THAT "
                "VERSION WAS NOT RETRIEVED (OpenReview returned a "
                "browser-verification challenge / HTTP 403 ChallengeRequiredError "
                "on both the forum page and api2.openreview.net). Everything "
                "below is from the arXiv v1."
            ),
            "doi": None,
            "retrieved_at": RETRIEVAL_DATE,
            "retrieval_url": "https://arxiv.org/html/2411.05676v1",
        },
        "representation": "2D molecular GRAPHS (no 3D coordinates at all); QM9 and ZINC250k",
        "reported_band": {
            "headline": "QM9 Val. 100 / Val. w/o corr. 99.91 ; ZINC250k Val. 100 / Val. w/o corr. 99.63",
            "table": "Table 1 (means of 3 runs)",
            "sample_nature": "training-distribution (unconditional graph generation), NOT held-out",
        },
        "reported_numbers": [
            {
                "label": "Validity (QM9, GGFlow)",
                "value": 100,
                "unit": "percent",
                "locator": "arXiv:2411.05676v1, Table 1 row 'GGFlow', QM9 block, column 'Val.'",
                "quote": "GGFlow | 100 | 99.91 | 0.0002 | 0.148 | 100 | 99.63 | 0.0010 | 1.455 | 500",
            },
            {
                "label": "Validity without correction (QM9, GGFlow)",
                "value": 99.91,
                "unit": "percent",
                "locator": "arXiv:2411.05676v1, Table 1 row 'GGFlow', QM9 block, column 'Val. w/o corr.'",
                "quote": "GGFlow | 100 | 99.91 | 0.0002 | 0.148 | 100 | 99.63 | 0.0010 | 1.455 | 500",
            },
            {
                "label": "Validity (ZINC250k, GGFlow)",
                "value": 100,
                "unit": "percent",
                "locator": "arXiv:2411.05676v1, Table 1 row 'GGFlow', ZINC250k block, column 'Val.'",
                "quote": "GGFlow | 100 | 99.91 | 0.0002 | 0.148 | 100 | 99.63 | 0.0010 | 1.455 | 500",
            },
            {
                "label": "Validity without correction (ZINC250k, GGFlow)",
                "value": 99.63,
                "unit": "percent",
                "locator": "arXiv:2411.05676v1, Table 1 row 'GGFlow', ZINC250k block, column 'Val. w/o corr.'",
                "quote": "GGFlow | 100 | 99.91 | 0.0002 | 0.148 | 100 | 99.63 | 0.0010 | 1.455 | 500",
            },
            {
                "label": "sample size for the metrics",
                "value": 10000,
                "unit": "molecules",
                "locator": "arXiv:2411.05676v1, Section 4.1",
                "quote": "To calculate these metrics, we sampled 10,000 molecules.",
            },
            {
                "label": "largest post-processing effect in the SAME table (GraphEBM, QM9)",
                "value": 91.22,
                "unit": "percentage points (Val. 100 minus Val. w/o corr. 8.78)",
                "locator": "arXiv:2411.05676v1, Table 1 row 'GraphEBM', QM9 block",
                "quote": "GraphEBM | 100 | 8.78 | 0.0287 | 6.402 | 100 | 5.29 | 0.2089 | 35.467 | -",
            },
        ],
        "cells": {
            "DENOMINATOR": {
                "verdict": "BREAKS",
                "quote": "To calculate these metrics, we sampled 10,000 molecules.",
                "locator": "arXiv:2411.05676v1, Section 4.1 'Molecular Graph Generation'; Table 1 caption ('Results are the means of 3 different runs.')",
                "note": (
                    "10,000 sampled 2D graphs, reported as the mean of 3 "
                    "independent runs, from a model trained on the TRAINING "
                    "split (QM9: 'We follow the train/test split from GDSS, "
                    "using 12,000 molecules for training and the remaining "
                    "1,000 for testing.'). Unconditional, not held-out. Ours "
                    "is one decode of one pool."
                ),
            },
            "METRIC": {
                "verdict": "BREAKS",
                "quote": "Validity We permit atoms to exhibit formal charges during valency checks because of the presence of formal charges in the training molecules. It is the fraction of valid molecules after valency correction or edge resampling.",
                "locator": "arXiv:2411.05676v1, Appendix E 'Details of Molecule Datasets' -- the metric-definition list (Validity / Validity w/o correction / FCD / NSPDK)",
                "note": (
                    "Their 'Validity' is a GRAPH-LEVEL valency check on a 2D "
                    "molecular graph (with formal charges permitted), and the "
                    "paper says so itself: 'It is the fraction of valid "
                    "molecules after valency correction or edge resampling.' "
                    "There is no 3D conformer and no fragment count anywhere "
                    "in it. Ours counts RDKit connected components on a 3D "
                    "decode. The two are not the same observable at all -- "
                    "this is a bigger break than the four the main thread "
                    "listed, and it is not one of the four columns."
                ),
            },
            "POST-PROCESSING": {
                "verdict": "BREAKS",
                "quote": "Validity w/o correction This metric explicitly evaluates the quality of molecule generation before any correction phase, providing a baseline for raw generation performance.",
                "locator": "arXiv:2411.05676v1, Appendix E metric definitions; Section 4.1 commentary",
                "recipe_verdict": "NOT-STATED",
                "recipe": (
                    "The paper names a 'correction phase' -- 'after valency "
                    "correction or edge resampling' -- and PUBLISHES its "
                    "effect (Val. 100 vs Val. w/o corr. 99.91 / 99.63), but "
                    "MEASURED: an exhaustive grep of the extracted full text "
                    "for 'correct' / 'resampl' / 'repair' hits exactly FOUR "
                    "lines -- the Section 4.1 metric list, the Section 4.1 "
                    "commentary sentence, and the two Appendix-E metric "
                    "definitions ('charge' appears in ONE of them). The "
                    "recipe itself is NEVER DESCRIBED."
                ),
                "note": (
                    "THE CLEANEST EVIDENCE IN THE WHOLE MATRIX, AND IT IS "
                    "SELF-REPORTED BY THE FRONTIER. GGFlow publishes BOTH the "
                    "post-processed number and its RAW counterpart: Val. 100 "
                    "vs Val. w/o corr. 99.91 (QM9) and 100 vs 99.63 "
                    "(ZINC250k). So the published 'Validity 100' IS "
                    "post-processed by the paper's own metric definition. "
                    "HONEST MAGNITUDE: for GGFlow itself the correction is "
                    "worth only +0.09 pp (QM9) / +0.37 pp (ZINC250k); the "
                    "axis is nonetheless load-bearing for the other rows in "
                    "the SAME table (GraphEBM 8.78 -> 100, GraphAF 67.14 -> "
                    "100, GDSS 96.17 -> 100). AND THE RECIPE ITSELF IS "
                    "NOT-STATED: an exhaustive grep of the extracted full "
                    "text for 'correct'/'resampl'/'repair' hits exactly FOUR "
                    "lines -- the Section 4.1 metric list, the Section 4.1 "
                    "commentary sentence, and the two Appendix-E metric "
                    "definitions ('charge' appears in ONE of them) -- the "
                    "paper never describes what 'valency correction or edge "
                    "resampling' does. A frontier number whose recipe is "
                    "unreadable cannot be compared to ours at the recipe "
                    "level, only at the number level."
                ),
            },
            "HYDROGENS": {
                "verdict": "DOES-NOT-BREAK",
                "quote": "each containing up to 9 heavy atoms: carbon, oxygen, nitrogen, and fluorine",
                "locator": "arXiv:2411.05676v1, Appendix E.3.1 'Dataset'; corroborated by Table S1 ('QM9 | Real | 133,885 | [1, 9] | 4 | 3') -- 4 node types, no hydrogen",
                "note": (
                    "GGFlow generates HEAVY-ATOM graphs only (QM9: C/O/N/F; "
                    "ZINC250k: 9 heavy-atom types), so its molecules are "
                    "implicit-hydrogen, same as ours. The word 'hydrogen' "
                    "appears NOWHERE in the paper's text (verified by "
                    "exhaustive grep of the extracted full text) -- the "
                    "convention is implicit in the node-type table, not "
                    "stated as a protocol."
                ),
            },
        },
        "caveats_stated_by_the_source": [
            "The paper publishes the raw (uncorrected) validity next to the corrected one, which is what makes the post-processing axis visible at all.",
            "No caveat about the validity metric itself is stated beyond the metric definition.",
        ],
        "not_retrieved": [
            {
                "verdict": "NOT-RETRIEVED",
                "what": "the retitled TMLR version of this paper -- 'GGFlow: A Graph Flow Matching Method with Efficient Optimal Transport', OpenReview K8RlXtMgzo -- which the branch brief names",
                "url": "https://openreview.net/forum?id=K8RlXtMgzo",
                "http_status": 403,
                "reason": (
                    "the forum page returned a browser-verification challenge "
                    "('Verifying your browser') and api2.openreview.net "
                    "returned {\"name\":\"ChallengeRequiredError\"} with HTTP 403"
                ),
                "consequence": (
                    "any change of protocol between arXiv v1 and the TMLR "
                    "version is UNKNOWN. Every GGFlow cell above is stated for "
                    "arXiv v1 only, and the retitle is NOT verified in this "
                    "session."
                ),
            }
        ],
        "extra_observations": [
            "The TMLR retitle could not be verified: OpenReview K8RlXtMgzo returned a browser-verification challenge (HTTP 403 ChallengeRequiredError) on both the forum page and the api2 notes endpoint. The arXiv v1 title is 'Improving Molecular Graph Generation with Flow Matching and Optimal Transport'.",
        ],
    },
    # ---------------------------------------------------------------------------
    # GeoLDM -- ADDED 2026-11-01 (the RESIDUAL-2 repair).
    #
    # The record named GeoLDM's cell as one of the band's TWO endpoints but
    # carried NO GeoLDM source: the 92.7 % was present in the repo only as a
    # TRANSCRIPT of HLTF's table, with the ORIGINATING paper absent from the
    # source list.  This entry is built THE SAME WAY as the other six: the
    # same key shape, the same four cell names (DENOMINATOR / METRIC /
    # POST-PROCESSING / HYDROGENS), and `reported_numbers` carrying the row,
    # the column, the value and a sha256-pinned locator.
    #
    # It carries NO claim that the band's provenance is settled: the reading
    # contest is UNCHANGED and still recorded in `band_provenance`.
    #
    # RETRIEVAL (2026-11-01, this repair): the arXiv e-print SOURCE TARBALL,
    # https://arxiv.org/e-print/2305.01140v1 -- HTTP 200, 655684 bytes,
    # sha256 b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23,
    # which MATCHES the sha256 pin recorded by the Wf-A186 primary-source
    # artefact (molmetal/reports/wfa186_frontier_primary_source/
    # primary_source_verification.json).  The extracted source file
    # icml2023.tex has sha256
    # b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, also
    # matching A186's pin.  The other five sources were read as rendered HTML
    # in the ORIGINAL Wf-A158 session.
    "GeoLDM": {
        "label": "GeoLDM -- Geometric Latent Diffusion Models (3D molecule generation; bond types INFERRED from the generated geometry)",
        "citation": {
            "arxiv_id": "2305.01140",
            "version_used": "v1 (only version on arXiv; submitted 2023-05-02)",
            "title": "Geometric Latent Diffusion Models for 3D Molecule Generation",
            "authors": "Minkai Xu, Alexander S. Powers, Ron O. Dror, Stefano Ermon, Jure Leskovec",
            "venue": (
                "ICML 2023 -- the paper's own source uses the ICML 2023 class "
                "with the [accepted] option (icml2023.tex:26) and declares "
                "'\\icmlkeywords{Machine Learning, ICML}' (icml2023.tex:131). "
                "The arXiv record carries NO journal-ref, NO comment and NO DOI, "
                "so the venue is taken from the paper's own source file, not "
                "from an arXiv metadata field."
            ),
            "doi": None,
            "retrieved_at": RETRIEVAL_DATE,
            "retrieval_url": "https://arxiv.org/e-print/2305.01140v1",
        },
        "representation": (
            "3D coordinates + bond types INFERRED post-hoc ('Given molecular "
            "geometries, we first predict bond types (single, double, triple, "
            "or none) by pair-wise atomic distances and atom types'); "
            "unconditional; QM9 and DRUG"
        ),
        "reported_band": {
            "headline": (
                "QM9 Atom-Sta 98.9 +/- 0.1 ; Mol-Sta 89.4 +/- 0.5 ; Valid "
                "93.8 +/- 0.4 ; Valid & Unique 92.7 +/- 0.5 (10,000 samples)"
            ),
            "table": "tab:qm9_results (icml2023.tex:491)",
            # 2026-11-02 (Wf-A194) -- the sample nature now says, in the
            # words the repo's own invariant uses, whether the population is
            # training-distribution or held-out.  The wording is A186's and
            # the paper's own; nothing here asserts a split the paper does
            # not state.
            "sample_nature": (
                "training-distribution (unconditional generation from the "
                "trained model, evaluated on QM9), NOT held-out -- 'Metrics "
                "are calculated with 10000 samples generated from each "
                "model. On QM9, we run the evaluation for 3 times and report "
                "the derivation.' (icml2023.tex:468) and 'We generate "
                "$10,000$ samples from each method to calculate the above "
                "metrics' (icml2023.tex:549).  The QM9 cells are therefore "
                "the model's OWN unconditional output over 10,000 generated "
                "samples and the paper reports NO held-out evaluation split "
                "for them; the table's own 'Data' row reports the QM9 "
                "DATASET's statistics (Valid 97.7 %, Valid & Unique 97.7 %), "
                "i.e. the reference population is QM9 itself."
            ),
        },
        "reported_numbers": [
            {
                "label": (
                    "Valid & Unique on QM9 -- GeoLDM's OWN cell, the 92.7 % "
                    "endpoint under Reading A"
                ),
                "row": "GeoLDM",
                "column": "Valid & Unique (%)",
                "value": 92.7,
                "unit": "percent (mean +/- 95 % CI over 3 evaluation runs)",
                "locator": (
                    "arXiv:2305.01140v1, icml2023.tex:491 (label "
                    "tab:qm9_results); source file sha256 "
                    "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, "
                    "e-print sha256 "
                    "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23"
                ),
                "quote": (
                    "\\rowcolor{lightgray} \\textbf{\\method} & \\textbf{98.9} "
                    "$\\pm$ 0.1 & \\textbf{89.4} $\\pm$ 0.5 & \\textbf{93.8} "
                    "$\\pm$ 0.4 & \\textbf{92.7} $\\pm$ 0.5 & \\textbf{84.4} & "
                    "\\textbf{99.3} \\\\"
                ),
            },
            {
                "label": "Valid on QM9 -- GeoLDM's OWN cell (the 93.8 / 93.6 disagreement)",
                "row": "GeoLDM",
                "column": "Valid (%)",
                "value": 93.8,
                "unit": "percent (mean +/- 95 % CI over 3 evaluation runs)",
                "locator": (
                    "arXiv:2305.01140v1, icml2023.tex:491 (label "
                    "tab:qm9_results); source file sha256 "
                    "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, "
                    "e-print sha256 "
                    "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23"
                ),
                "quote": (
                    "\\rowcolor{lightgray} \\textbf{\\method} & \\textbf{98.9} "
                    "$\\pm$ 0.1 & \\textbf{89.4} $\\pm$ 0.5 & \\textbf{93.8} "
                    "$\\pm$ 0.4 & \\textbf{92.7} $\\pm$ 0.5 & \\textbf{84.4} & "
                    "\\textbf{99.3} \\\\"
                ),
            },
            {
                "label": "Molecule stability on QM9 -- GeoLDM's OWN cell",
                "row": "GeoLDM",
                "column": "Molecule Stability (%)",
                "value": 89.4,
                "unit": "percent (mean +/- 95 % CI over 3 evaluation runs)",
                "locator": (
                    "arXiv:2305.01140v1, icml2023.tex:491 (label "
                    "tab:qm9_results); source file sha256 "
                    "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, "
                    "e-print sha256 "
                    "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23"
                ),
                "quote": (
                    "\\rowcolor{lightgray} \\textbf{\\method} & \\textbf{98.9} "
                    "$\\pm$ 0.1 & \\textbf{89.4} $\\pm$ 0.5 & \\textbf{93.8} "
                    "$\\pm$ 0.4 & \\textbf{92.7} $\\pm$ 0.5 & \\textbf{84.4} & "
                    "\\textbf{99.3} \\\\"
                ),
            },
            {
                "label": "Atom stability on QM9 -- GeoLDM's OWN cell",
                "row": "GeoLDM",
                "column": "Atom Stability (%)",
                "value": 98.9,
                "unit": "percent (mean +/- 95 % CI over 3 evaluation runs)",
                "locator": (
                    "arXiv:2305.01140v1, icml2023.tex:491 (label "
                    "tab:qm9_results); source file sha256 "
                    "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, "
                    "e-print sha256 "
                    "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23"
                ),
                "quote": (
                    "\\rowcolor{lightgray} \\textbf{\\method} & \\textbf{98.9} "
                    "$\\pm$ 0.1 & \\textbf{89.4} $\\pm$ 0.5 & \\textbf{93.8} "
                    "$\\pm$ 0.4 & \\textbf{92.7} $\\pm$ 0.5 & \\textbf{84.4} & "
                    "\\textbf{99.3} \\\\"
                ),
            },
            {
                "label": "QM9 sample size behind every cell of GeoLDM's row",
                "row": "GeoLDM",
                "column": "(the table caption, not a column)",
                "value": 10000,
                "unit": "molecules",
                "locator": (
                    "arXiv:2305.01140v1, icml2023.tex:468 (tab:qm9_results "
                    "caption) and icml2023.tex:549; source file sha256 "
                    "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, "
                    "e-print sha256 "
                    "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23"
                ),
                "quote": (
                    "Metrics are calculated with 10000 samples generated from "
                    "each model. On QM9, we run the evaluation for 3 times and "
                    "report the derivation."
                ),
            },
        ],
        "cells": {
            "DENOMINATOR": {
                "verdict": "BREAKS",
                "quote": (
                    "We generate $10,000$ samples from each method to calculate "
                    "the above metrics, and the results are reported in "
                    "\\cref{tab:qm9_results}."
                ),
                "locator": (
                    "arXiv:2305.01140v1, icml2023.tex:549 and the "
                    "tab:qm9_results caption at icml2023.tex:468; source file "
                    "sha256 "
                    "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, "
                    "e-print sha256 "
                    "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23"
                ),
                "note": (
                    "10,000 unconditional QM9 samples, the evaluation run 3 "
                    "times with the DERIVATION reported. Ours is one decode of "
                    "one pool of 2,112 or 12,669 rows. The protocol is the same "
                    "shape as the other frontier papers' (sampling from the "
                    "model trained on the dataset), so the break is a magnitude "
                    "break, not a kind break -- the SIZES differ by 5x to 6x."
                ),
            },
            "METRIC": {
                "verdict": "BREAKS",
                "quote": (
                    "\\rowcolor{lightgray} \\textbf{\\method} & \\textbf{98.9} "
                    "$\\pm$ 0.1 & \\textbf{89.4} $\\pm$ 0.5 & \\textbf{93.8} "
                    "$\\pm$ 0.4 & \\textbf{92.7} $\\pm$ 0.5 & \\textbf{84.4} & "
                    "\\textbf{99.3} \\\\"
                ),
                "locator": (
                    "arXiv:2305.01140v1, icml2023.tex:491 (label "
                    "tab:qm9_results, row GeoLDM) and the metric definitions at "
                    "icml2023.tex:532; source file sha256 "
                    "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, "
                    "e-print sha256 "
                    "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23"
                ),
                "note": (
                    "GeoLDM's QM9 row reports FOUR observables -- atom "
                    "stability, molecule stability, validity and validity x "
                    "uniqueness -- and the cell this record's Reading A names "
                    "as the band's lower endpoint is the 'validity x "
                    "uniqueness' one, i.e. a per-molecule RDKit-sanitisable AND "
                    "non-duplicate rate ('we report validity and uniqueness "
                    "metrics, which are the percentages of valid (measured by "
                    "RDKit) and unique molecules among all the generated "
                    "compounds'). The paper states NO connectivity or "
                    "fragment criterion for the QM9 metrics, so the cell is not "
                    "a connected-component count. THIS IS A STATEMENT ABOUT "
                    "GeoLDM's OWN CELL; it does not settle which cell the "
                    "repo's band names -- that contest is still open and is "
                    "recorded under `band_provenance`."
                ),
            },
            "POST-PROCESSING": {
                "verdict": "DOES-NOT-BREAK",
                "quote": (
                    "the model occasionally generates disconnected components, "
                    "as shown in the rightest column of "
                    "\\cref{app:fig:vis-drug}, which happens more often when "
                    "trained on the large molecule DRUG dataset. However, this "
                    "phenomenon actually is not a problem and is common in all "
                    "non-autoregressive molecule generative "
                    "models~\\cite{zang2020moflow,jo2022score}, and can be "
                    "easily fixed by just filtering the smaller components."
                ),
                "locator": (
                    "arXiv:2305.01140v1, icml2023.tex:917; source file sha256 "
                    "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, "
                    "e-print sha256 "
                    "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23"
                ),
                "recipe_verdict": "NOT-APPLICABLE",
                "recipe": (
                    "NOT APPLICABLE -- GeoLDM's QM9 pipeline has NO "
                    "post-processing step. The paper's only mention of "
                    "filtering is the DRUG-side remark 'can be easily fixed by "
                    "just filtering the smaller components', which is a REMEDY "
                    "PROPOSED for the reader, not a procedure applied to the "
                    "reported numbers. The paper's source contains the word "
                    "'filter' exactly once, in that sentence."
                ),
                "note": (
                    "ON THE POST-PROCESSING AXIS, GeoLDM's OWN reported QM9 "
                    "numbers are on the same footing as ours: no "
                    "largest-fragment retention, no explicit hydrogens, no "
                    "UFF step. THE READING THAT THEY ARE RAW IS AN ARGUMENT "
                    "FROM SILENCE plus ONE sentence, and it is labelled as "
                    "such here: the paper never says 'no post-processing', it "
                    "simply never describes a post-processing step, and the "
                    "single 'filtering' sentence is a DRUG-side observation "
                    "('happens more often when trained on the large molecule "
                    "DRUG dataset') whose remedy is proposed, not applied. THIS "
                    "VERDICT IS ABOUT GeoLDM's OWN PROTOCOL. It says nothing "
                    "about the band's provenance, and it says nothing about the "
                    "band's other endpoint."
                ),
            },
            "HYDROGENS": {
                "verdict": "NOT-STATED",
                "marker": (
                    "NOT-STATED: the paper's ONLY mention of hydrogens is the "
                    "QM9 DATASET description ('limited to 9 heavy atoms (29 "
                    "atoms including hydrogens)', icml2023.tex:517); it states "
                    "NO hydrogen convention for the molecules it GENERATES, "
                    "and it says nothing about an explicit-H step."
                ),
                "quote": (
                    "QM9 contains 3D structures together with several quantum "
                    "properties for 130k small molecules, limited to 9 heavy "
                    "atoms (29 atoms including hydrogens)."
                ),
                "locator": (
                    "arXiv:2305.01140v1, icml2023.tex:517 (the QM9 dataset "
                    "paragraph); source file sha256 "
                    "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, "
                    "e-print sha256 "
                    "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23"
                ),
                "note": (
                    "The paper does not state a hydrogen convention for its "
                    "generated molecules, so the cell cannot be compared "
                    "against ours on that axis. Recorded as NOT-STATED rather "
                    "than guessed."
                ),
            },
        },
        "caveats_stated_by_the_source": [
            (
                "tab:qm9_results caption: 'Note that, for DRUG dataset, molecule "
                "stability and uniqueness metric are omitted since they are "
                "nearly $0\\%$ and $100\\%$ respectively for all the methods.' "
                "(icml2023.tex:468)"
            ),
            (
                "tab:qm9_results caption: 'Compared with previous methods, the "
                "latent space with both invariant and equivariant variables "
                "enables \\method to achieve up to 7\\% improvement for the "
                "validity of large molecule generation.' (icml2023.tex:468)"
            ),
        ],
        "extra_observations": [
            (
                "GEO-LDM'S OWN PAPER NAMES THE ARTEFACT. 'the model occasionally "
                "generates disconnected components ... this phenomenon actually "
                "is not a problem and is common in all non-autoregressive "
                "molecule generative models, and can be easily fixed by just "
                "filtering the smaller components.' (icml2023.tex:917) The "
                "remedy is PROPOSED, not applied: the paper's source contains "
                "the word 'filter' exactly once, in that sentence, and the "
                "numbers in tab:qm9_results are the unfiltered ones. The same "
                "sentence says the disconnection 'happens more often when "
                "trained on the large molecule DRUG dataset', i.e. it is a "
                "DRUG-side observation, not a QM9-side correction."
            ),
            (
                "THE TWO SOURCE TABLES DISAGREE ON THE SAME OBSERVABLE, AND "
                "THE DISAGREEMENT IS ANNOTATED HERE. GeoLDM's OWN paper prints "
                "Valid = 93.8 +/- 0.4 for its own row (icml2023.tex:491, "
                "sha256 b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b); "
                "HLTF prints the GeoLDM column's Val as 93.6 +/- 0.2 in its "
                "Table 1(a) (tables/qm9_benchmarks_og.tex:38, sha256 "
                "1c81ad499093f9d20d512e29787e934ab68c6d32dc5bb3b791970830c05c8f39) "
                "and again in its Table 1(b) (:63), and its Table 1 caption "
                "states 'Baselines reproduced from GCDM Table 1', i.e. the "
                "column is a TRANSCRIPT. NEITHER PAPER FLAGS THE DIFFERENCE. "
                "BOTH NUMERALS ARE ON DISK IN THIS REPO: the Wf-A175 frontier "
                "comparability record carries GeoLDM's own 93.8 +- 0.4 at the "
                "JSON PATH /sources/4/metrics/0/value (and in "
                "/sources/4/metrics/0/evidence_quote) and HLTF's GeoLDM-column "
                "Val 93.6 +- 0.2 at the JSON PATH "
                "/sources/11/metrics/2/evidence_quote, both in "
                "molmetal/reports/wfa175_frontier_comparability/sources.json "
                "(sha256 "
                "2e665cb217759a4099c83716a8f27829797f2401c6c630adaeb1b4cfbd6e3c78, "
                "as measured on 2026-11-02). What the Wf-A175 record does NOT "
                "do is FLAG the two as a disagreement -- its report.md "
                "(sha256 "
                "170f642846192837c334d45bb6a006414bf1b53a7089c8907f4061ff0d0f10e0, "
                "as measured on 2026-11-02) calls them 'the same GeoLDM row' "
                "without noticing the differing Val, in the sentence '(Table "
                "1(a), line 38, carries the same GeoLDM row: `GeoLDM & "
                "98.9+-0.1* & 89.4+-0.5* & 93.6+-0.2 & 92.7+-0.5`.)'. THE LINE "
                "NUMBERS THIS REPLACES ARE NOT CITED ANY MORE: a line number "
                "is a PERISHABLE locator in an artefact that is regenerated "
                "and grows, so the JSON PATH plus the sha256 above is the "
                "durable locator. NEITHER NUMERAL IS A BAND ENDPOINT: under "
                "Reading A the endpoints are 92.7 / 94.0 and under Reading B "
                "they are 92.98 / 94.33, so 93.8 and 93.6 must never be "
                "published as band endpoints."
            ),
            (
                "THE RECORD'S OWN READING OF THE LOWER ENDPOINT IS UNCHANGED "
                "BY THIS ENTRY. Adding GeoLDM as a FIRST-CLASS source means "
                "the 92.7 % cell is now substantiated by the originating "
                "paper's OWN e-print (arXiv:2305.01140v1), not only by a "
                "transcript of HLTF's table. It does NOT settle which cell "
                "the repo's band names: the two competing readings (and the "
                "catalogue's third) remain exactly as they were."
            ),
        ],
        # 2026-11-02 (Wf-A197) -- the superseded rendering is PRESERVED
        # VERBATIM (AGENTS.md §5.5 rule #6).  It cited the Wf-A175 frontier
        # comparability record by LINE NUMBER, and a line number is a
        # PERISHABLE locator in an artefact that is regenerated and grows.
        "extra_observations_superseded_2026_11_02": [
            WFA197_R1_SUPERSEDED_OBSERVATION_2026_11_02
        ],
        "extra_observations_superseded_note_2026_11_02": (
            "2026-11-02 residual repair (Wf-A197): the rendering above is "
            "SUPERSEDED.  It cited the Wf-A175 frontier comparability record "
            "by LINE NUMBER (report.md:40, report.md:133, report.md:138, "
            "sources.json:170, sources.json:417) -- all five of which were "
            "ACCURATE when measured, but are PERISHABLE: that artefact is "
            "regenerated and grows.  The live rendering replaces them with "
            "the artefact's JSON PATHS plus its sha256, as measured on "
            "2026-11-02.  The superseded rendering is PRESERVED VERBATIM in "
            "`extra_observations_superseded_2026_11_02`."
        ),
        "not_retrieved": [
            {
                "verdict": "NOT-RETRIEVED",
                "what": "the arXiv abs-page HTML rendering of the paper",
                "url": "https://arxiv.org/abs/2305.01140v1",
                "http_status": None,
                "reason": (
                    "the METADATA was taken from the arXiv Atom API record "
                    "(title, authors, published date) and the CONTENT from the "
                    "e-print source tarball; the abs-page HTML was not fetched "
                    "in this session"
                ),
                "consequence": (
                    "every GeoLDM cell above is stated for the e-print SOURCE "
                    "of v1 (the .tex read verbatim out of the tarball), which "
                    "is the more precise artefact than a rendered page"
                ),
            },
            {
                "verdict": "NOT-RETRIEVED",
                "what": "the ICML 2023 proceedings version (PMLR)",
                "url": "not fetched in this session",
                "http_status": None,
                "reason": (
                    "the arXiv record carries NO journal-ref, NO comment and NO "
                    "DOI, so no proceedings URL was pinned; the venue is taken "
                    "from the paper's own ICML 2023 class file"
                ),
                "consequence": (
                    "the numbers above are stated for the arXiv e-print of v1; "
                    "the proceedings version was not compared"
                ),
            },
        ],
    },
    "HLTF": {
        "label": "HLTF -- Hierarchy-Guided Latent Topology Flow (explicit bond topology + 3D coordinates)",
        "citation": {
            "arxiv_id": "2603.27113",
            "version_used": "v1 (only version on arXiv; published 2026-03-28)",
            "title": "Hierarchy-Guided Topology Latent Flow for Molecular Graph Generation",
            "authors": "Urvi Awasthi, Alexander Arjun Lobo, Leonid Zhukov",
            "venue": "ICLR 2026 AI4Mat Workshop (accepted) -- verified from the arXiv abs-page comments field: '22 pages, 2 figures, 6 tables. Accepted to ICLR 2026 AI4Mat Workshop'",
            "doi": None,
            "retrieved_at": RETRIEVAL_DATE,
            "retrieval_url": "https://arxiv.org/html/2603.27113v1",
        },
        "representation": "3D explicit bond topology (bond types GENERATED, not inferred); unconditional; QM9 and GEOM-DRUGS",
        "reported_band": {
            "headline": "QM9 Val&Uniq 92.9 +/- 0.2 ; PB-Valid 94 +/- 0.5 ; GEOM-DRUGS Valid 85.5 % RAW -> 92.2 % after post-processing",
            "table": "Table 1(a), Table 1(b) (QM9) and Table 2 (GEOM-DRUGS)",
            "sample_nature": "training-distribution (unconditional), NOT held-out",
        },
        "reported_numbers": [
            {
                "label": "Val&Uniq on QM9 (GeoLDM baseline -- the band's 92.7 % endpoint)",
                "value": 92.7,
                "unit": "percent (mean +/- 95 % CI)",
                "locator": "arXiv:2603.27113v1, Table 1(b) row 'Val&Uniq (%)', column 'GeoLDM'; identical value in Table 1(a) row 'Val&Uniq (%)'",
                "quote": "Val&Uniq (%) | 92.7 ± 0.5 | 93.3 ± 0.0 * | 92.9 ± 0.2",
            },
            {
                "label": "PB-Valid on QM9 (HLTF -- the band's 94 % endpoint)",
                "value": 94.0,
                "unit": "percent (mean +/- 95 % CI)",
                "locator": "arXiv:2603.27113v1, Table 1(b) row 'PB-Valid (%)', column 'HLTF (ours)'",
                "quote": "PB-Valid (%) | 93.1 ± 0.4 | 91.9 ± 0.5 | 94 ± 0.5 *",
            },
            {
                "label": "Valid on GEOM-DRUGS, RAW (no post-processing) -- HLTF",
                "value": 85.5,
                "unit": "percent",
                "locator": "arXiv:2603.27113v1, Table 2, row 'HLTF (ours)'",
                "quote": "HLTF (ours) | 85.5 | 85.4 | 85.0",
            },
            {
                "label": "Valid on GEOM-DRUGS, AFTER post-processing -- HLTF",
                "value": 92.2,
                "unit": "percent",
                "locator": "arXiv:2603.27113v1, Table 2, row '+PP' immediately under 'HLTF (ours)'",
                "quote": "+PP | 92.2 | 92.0 | 91.2",
            },
            {
                "label": "fraction of pre-PP invalid samples converted by post-processing (HLTF's own formula)",
                "value": 46.0,
                "unit": "percent of the pre-PP invalid samples",
                "locator": "arXiv:2603.27113v1, Section 4.4 'What changes with post-processing', footnote 2: 'Computed as (92.2 - 85.5) / (100 - 85.5)'",
                "quote": "For HLTF it raises Valid from 85.5 to 92.2, converting ≈ 46 % of the pre-PP invalid samples under the same pipeline.",
            },
            {
                "label": "GEOM-DRUGS sample size",
                "value": 100000,
                "unit": "molecules",
                "locator": "arXiv:2603.27113v1, Section 4.1",
                "quote": "On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation Rappe et al. (1992) ).",
            },
            {
                "label": "QM9 sample size",
                "value": 10000,
                "unit": "molecules",
                "locator": "arXiv:2603.27113v1, Section 4.1",
                "quote": "On QM9 we report atom stability (AS), molecule stability (MS), RDKit validity (Val), and Val&Uniq on 10,000 unconditional samples.",
            },
            {
                "label": "Valid on GEOM-DRUGS, RAW -- FlowMol baseline",
                "value": 59.8,
                "unit": "percent",
                "locator": "arXiv:2603.27113v1, Table 2, row 'FlowMol'",
                "quote": "FlowMol | 59.8 | 59.8 | 59.7",
            },
            {
                "label": "Valid on GEOM-DRUGS, AFTER post-processing -- FlowMol baseline",
                "value": 84.2,
                "unit": "percent",
                "locator": "arXiv:2603.27113v1, Table 2, row '+PP' immediately under 'FlowMol'",
                "quote": "+PP | 84.2 | 84.2 | 84.1",
            },
            {
                "label": "Valid on GEOM-DRUGS, RAW -- GCDM baseline",
                "value": 0.2,
                "unit": "percent",
                "locator": "arXiv:2603.27113v1, Table 2, row 'GCDM'",
                "quote": "GCDM | 0.2 | 0.2 | 0.2",
            },
            {
                "label": "Valid on GEOM-DRUGS, AFTER post-processing -- GCDM baseline",
                "value": 95.2,
                "unit": "percent",
                "locator": "arXiv:2603.27113v1, Table 2, row '+PP' immediately under 'GCDM'",
                "quote": "+PP | 95.2 | 95.2 | 95.2",
            },
        ],
        "cells": {
            "DENOMINATOR": {
                "verdict": "BREAKS",
                "quote": "On QM9 we report atom stability (AS), molecule stability (MS), RDKit validity (Val), and Val&Uniq on 10,000 unconditional samples.",
                "locator": "arXiv:2603.27113v1, Section 4.1 'Benchmarks, protocols, and metrics'; Table 2's denominator is 100,000 samples on GEOM-DRUGS (same paragraph)",
                "note": (
                    "10,000 unconditional QM9 samples (Table 1) and 100,000 "
                    "unconditional GEOM-DRUGS samples (Table 2), from a model "
                    "trained on the TRAINING split. Ours is one decode of one "
                    "pool of 2,112 or 12,669 rows. The protocol is the same "
                    "shape as the other frontier papers' (training-distribution "
                    "sampling), which is why the break is a magnitude break, not "
                    "a kind break -- the SIZES differ by 5x to 6x."
                ),
            },
            "METRIC": {
                "verdict": "BREAKS",
                "quote": "PB-Valid (%) | 93.1 ± 0.4 | 91.9 ± 0.5 | 94 ± 0.5 *",
                "locator": "arXiv:2603.27113v1, Table 1(b) metric list (AS / MS / Val / Val&Uniq / Novel / PB-Valid) and the Section 4.3 'Why PoseBusters matters here' paragraph",
                # Wf-A177 (2026-11-01) CONTESTED-CLAIM ANNOTATION -- SOURCE COMMENT
                # ONLY.  The emitted note below is DELIBERATELY UNCHANGED: the
                # artefact on disk must keep its original rendering (AGENTS.md
                # §5.5 rule #6).  Only the producer is annotated, so that a
                # future reader of THIS SOURCE knows the claim it emits is
                # contested.
                #
                # The sentence "NEITHER is a connected-component count" is TRUE
                # under Reading A (`frontier_matrix.json` -> `band_provenance`:
                # GeoLDM's Validity & Uniq 92.7 % and HLTF's PB-Valid
                # 94.0 %, both `post_processed: false`, plus YuelBond's
                # per-bond-type classification F1 92.7 % as a third candidate
                # for the same numeral) and FALSE under Reading B (the same
                # file -> `sources`.`EF-TALFM`, EF-TALFM
                # arXiv:2609.08333v4: the band's lower endpoint is EF-TALFM's
                # OWN Table 3(a) RAW connectivity cell C. = 92.98 %, counted
                # BEFORE the largest-component retention, mixed with the
                # post-processed N. & U. & S. = 94.33 % companion cell).  The
                # two readings do not even agree on the numerals -- Reading A
                # reproduces the quoted 92.7 / 94 exactly, Reading B's cells
                # are 92.98 / 94.33 ("93.0-94.3") -- so the source cells of the
                # band are NOT settled on disk and the published claim must
                # therefore be reading-qualified.  Only (i) the band is
                # EXTERNAL, (ii) it is NOT one homogeneous quantity, (iii) it
                # is NOT on our scale -- OUR 8.51 % IS a connected-component
                # count on the RAW emitted SMILES, and nothing else -- and
                # (iv) the source cells are NOT settled, may be published.
                # Wf-A180 (2026-11-01) SUPERSEDED RENDERING -- PRESERVED
                # VERBATIM.  The Wf-A177 comment above (which deliberately
                # left the emitted string UNCHANGED) is itself superseded:
                # AGENTS.md §5.5 rule #6 requires the superseded rendering
                # to be PRESERVED, not to stay LIVE.  The emitted string
                # below is now the READING-SAFE rendering.
                #
                # SUPERSEDED (the Wf-A177-live rendering, verbatim):
                #   THE BAND'S UPPER ENDPOINT IS NOT A CONNECTIVITY NUMBER AT ALL. Table 1(b) reports SIX observables -- atom stability (AS), molecule stability (MS), RDKit validity (Val), Validity&Uniq, Novelty and PoseBusters validity (PB-Valid) -- and the 94 +/- 0.5 % figure is PB-Valid, a SIX-CHECK POSE VERDICT (bond lengths/angles, steric clash, aromatic and double-bond flatness, internal energy). The LOWER endpoint (92.7 %) is GeoLDM's Validity&Uniq -- a sanitisation-and-duplication rate. NEITHER is a connected-component count. Ours is a connected-component count and nothing else. This is a total METRIC break, not a partial one.
                #
                # WHY IT IS SUPERSEDED: "NEITHER is a connected-component
                # count" is TRUE of the UPPER endpoint under both readings
                # but FALSE of the LOWER endpoint under Reading B (EF-TALFM's
                # RAW connectivity cell C. = 92.98 %).  "This is a total
                # METRIC break" is likewise a settled claim that only
                # Reading A supports.
                "note": (
                    "UNDER BOTH PROVENANCE READINGS the band's UPPER "
                    "endpoint is NOT a connectivity number: under Reading A "
                    "it is HLTF's own PB-Valid figure of 94 +/- 0.5 %, a "
                    "POSE VERDICT -- arXiv:2603.27113v1 Table 1(b) reports "
                    "SIX observables (atom stability (AS), molecule stability "
                    "(MS), RDKit validity (Val), Validity&Uniq, Novelty and "
                    "PoseBusters validity (PB-Valid)), which is a TABLE-ROW "
                    "COUNT and NOT a count of PoseBusters checks; HLTF states "
                    "NO CHECK COUNT for its PB-Valid cell "
                    "(NOT-STATED-IN-SOURCE): it cites PoseBusters "
                    "(Buttenschoen et al. 2024) and says only that "
                    "PoseBusters 'applies stricter structural checks than "
                    "RDKit'. The SIX-check PoseBusters set is EF-TALFM's, "
                    "not HLTF's: arXiv:2609.08333v4 Section A.1.4 counts 'all "
                    "six PoseBusters sanity checks used here' and lists them "
                    "(Reasonable bond lengths, Reasonable bond angles, "
                    "Aromatic ring flatness, Double bond flatness, "
                    "Reasonable internal energy, No internal steric clash), "
                    "and its Table 3(b) reports the joint row 'All "
                    "PoseBusters checks'; under Reading B the band's upper "
                    "endpoint is EF-TALFM's post-processed `N. & U. & S. = "
                    "94.33 %` companion cell. The LOWER endpoint is the "
                    "CONTESTED one: Reading A names GeoLDM's Validity&Uniq "
                    "92.7 % (a sanitisation-and-duplication rate, "
                    "`post_processed: false`), Reading B names EF-TALFM's RAW "
                    "connectivity cell `C. = 92.98 %` (counted BEFORE the "
                    "largest-component retention). WHICH reading the repo's "
                    "band follows is NOT settled on disk -- the two "
                    "readings do not even agree on the numerals (92.7/94.0 "
                    "vs 92.98/94.33) -- so it may NOT be published that "
                    "'neither endpoint is a connected-component count': "
                    "that is true of the UPPER endpoint under both readings, "
                    "but the LOWER endpoint IS a connected-component count "
                    "under Reading B. Under BOTH readings the band is "
                    "EXTERNAL, is NOT one homogeneous quantity, and is NOT "
                    "on our scale -- OUR 8.51 % IS a connected-component "
                    "count on the RAW emitted SMILES, and nothing else. The "
                    "two readings are recorded under "
                    "`band_provenance`.`competing_reading_and_what_this_"
                    "does_not_settle`."
                ),
                # 2026-11-01 RESIDUAL REPAIR (RESIDUAL 3 / D8) -- the
                # six-check attribution.  The previous live rendering is
                # PRESERVED VERBATIM below.
                "note_superseded_2026_11_01_residual": (
                    "UNDER BOTH PROVENANCE READINGS the band's UPPER "
                    "endpoint is NOT a connectivity number: under Reading A "
                    "it is HLTF's own PB-Valid figure of 94 +/- 0.5 %, a "
                    "SIX-CHECK POSE VERDICT (bond lengths/angles, steric "
                    "clash, aromatic and double-bond flatness, internal "
                    "energy) -- arXiv:2603.27113v1 Table 1(b) reports SIX "
                    "observables (atom stability (AS), molecule stability "
                    "(MS), RDKit validity (Val), Validity&Uniq, Novelty and "
                    "PoseBusters validity (PB-Valid)); under Reading B it is "
                    "EF-TALFM's post-processed `N. & U. & S. = 94.33 %` "
                    "companion cell. The LOWER endpoint is the CONTESTED "
                    "one: Reading A names GeoLDM's Validity&Uniq 92.7 % (a "
                    "sanitisation-and-duplication rate, `post_processed: "
                    "false`), Reading B names EF-TALFM's RAW connectivity "
                    "cell `C. = 92.98 %` (counted BEFORE the "
                    "largest-component retention). WHICH reading the repo's "
                    "band follows is NOT settled on disk -- the two "
                    "readings do not even agree on the numerals (92.7/94.0 "
                    "vs 92.98/94.33) -- so it may NOT be published that "
                    "'neither endpoint is a connected-component count': "
                    "that is true of the UPPER endpoint under both readings, "
                    "but the LOWER endpoint IS a connected-component count "
                    "under Reading B. Under BOTH readings the band is "
                    "EXTERNAL, is NOT one homogeneous quantity, and is NOT "
                    "on our scale -- OUR 8.51 % IS a connected-component "
                    "count on the RAW emitted SMILES, and nothing else. The "
                    "two readings are recorded under "
                    "`band_provenance`.`competing_reading_and_what_this_"
                    "does_not_settle`."
                ),
                "note_superseded_2026_11_01_residual_note": (
                    "2026-11-01 residual repair: the rendering above is "
                    "SUPERSEDED.  It attributed a SIX-check PoseBusters set to "
                    "HLTF.  HLTF states NO check count for its PB-Valid cell "
                    "(NOT-STATED-IN-SOURCE); the six-check set is EF-TALFM's "
                    "(arXiv:2609.08333v4, Section A.1.4 + Table 3(b)).  The "
                    "CLAIM is unchanged -- the UPPER endpoint is not a "
                    "connectivity number under both readings; only the "
                    "misattributed check count is corrected.  PRESERVED "
                    "VERBATIM per AGENTS.md §5.5 rule #6."
                ),
                # Wf-A180 (2026-11-01) SUPERSEDED RENDERING -- PRESERVED
                # VERBATIM.
                "note_superseded_2026_11_01": (
                    "THE BAND'S UPPER ENDPOINT IS NOT A CONNECTIVITY NUMBER AT ALL. "
                    "Table 1(b) reports SIX observables -- atom stability (AS), "
                    "molecule stability (MS), RDKit validity (Val), Validity&Uniq, "
                    "Novelty and PoseBusters validity (PB-Valid) -- and the 94 +/- 0.5 "
                    "% figure is PB-Valid, a SIX-CHECK POSE VERDICT (bond "
                    "lengths/angles, steric clash, aromatic and double-bond flatness, "
                    "internal energy). The LOWER endpoint (92.7 %) is GeoLDM's "
                    "Validity&Uniq -- a sanitisation-and-duplication rate. NEITHER is a "
                    "connected-component count. Ours is a connected-component count and "
                    "nothing else. This is a total METRIC break, not a partial one."
                ),
                "note_superseded_note_2026_11_01": (
                    "Wf-A180 (2026-11-01): the rendering above is SUPERSEDED.  It "
                    "asserted 'NEITHER is a connected-component count' and 'This is a "
                    "total METRIC break, not a partial one', both of which hold only "
                    "under Reading A.  PRESERVED VERBATIM per AGENTS.md §5.5 rule #6."
                ),
            },
            "POST-PROCESSING": {
                "verdict": "BREAKS",
                "quote": "On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation Rappe et al. (1992) ).",
                "locator": "arXiv:2603.27113v1, Section 4.1; Table 2 (the '+PP' rows); Section 4.4 'What changes with post-processing'",
                "recipe_verdict": "STATED",
                "recipe": (
                    "standardized post-processing = largest fragment + explicit "
                    "hydrogens + UFF relaxation, cited as Buttenschoen et al. "
                    "(2025) -- the SAME protocol family as our own post-processing "
                    "arm (which cites arXiv:2505.00518 by the same authors)."
                ),
                "note": (
                    "THE FRONTIER MEASURES THE POST-PROCESSING AXIS ITSELF, "
                    "IN ONE TABLE, FOR EVERY MODEL. HLTF Table 2 reports the "
                    "raw and the post-processed number side by side for six "
                    "models: EQGAT-diff 59.7 -> 84.2, FlowMol 59.8 -> 84.2, "
                    "GCDM 0.2 -> 95.2, GeoLDM 2.9 -> 69.6, SemlaFlow 87.5 -> "
                    "93.1, HLTF 85.5 -> 92.2 (Valid, GEOM-DRUGS, 100,000 "
                    "samples). The post-processing is worth +6.7 pp on the best "
                    "model and +95.0 pp on the worst. HLTF also states the "
                    "recovery fraction it defines -- 'converting ~ 46 % of the "
                    "pre-PP invalid samples under the same pipeline', computed "
                    "as (92.2 - 85.5) / (100 - 85.5) -- which is exactly the "
                    "quantity our own post-processing arm can be reported on. "
                    "ON OUR SIDE the same recipe family moves 8.51 % -> 100 % "
                    "for connectivity alone, with the UFF third step INAPPLICABLE "
                    "to our metal corpus (uff_ok_rate_over_postprocessed = 0.0). "
                    "So the frontier's own table shows that a published "
                    "'92-95 %' validity number can sit 6.7 to 95 points above "
                    "the same model's raw number: the post-processing axis is "
                    "not a rounding detail, it IS most of the number."
                ),
            },
            "HYDROGENS": {
                "verdict": "BREAKS",
                "quote": "On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation Rappe et al. (1992) ).",
                "locator": "arXiv:2603.27113v1, Section 4.1 (the post-processing adds explicit hydrogens) and Section 4.2",
                "note": (
                    "The post-processing step adds EXPLICIT hydrogens "
                    "('explicit hydrogens' in the recipe sentence), so the "
                    "post-processed numbers are computed on explicit-H "
                    "molecules. Ours is implicit-H. The paper's raw numbers are "
                    "on whatever its generator emits; the paper does not state a "
                    "hydrogen convention for the raw arm, so the raw arm is "
                    "implicit-H by construction (it generates a graph, and "
                    "RDKit sanitisation implies implicit H). VERDICT IS "
                    "'BREAKS' because the published headline pair (85.5 % raw / "
                    "92.2 % PP) straddles the two conventions -- the two numbers "
                    "we would have to compare against are not on the same "
                    "convention as each other, let alone ours."
                ),
            },
        },
        "caveats_stated_by_the_source": [
            "Section 4.4 'What changes with post-processing': 'Standardized post-processing mainly relieves geometric strain (e.g., unrealistic bond lengths/angles and local steric clashes) and cannot repair infeasible topology (e.g., invalid valence or disconnected graphs).'",
            "Section 4.4: 'we do not claim state-of-the-art headline success on this benchmark'.",
            "Section 4.2 'Explicit-topology strictness': HLTF is evaluated on the bonds it generates, a stricter requirement than inferred-bond evaluation; the matched-protocol study is left to future work.",
            "Section 4.4: 'Within each target...' is not applicable -- the paper notes the comparison to SemlaFlow is 0.9 points 'under the same PP pipeline'.",
        ],
        "extra_observations": [
            "HLTF IS THE BEST SINGLE SOURCE FOR THIS BRANCH: it is the only "
            "retrieved frontier paper that publishes the RAW and the "
            "POST-PROCESSED number for the SAME model under the SAME pipeline, "
            "in one table, and that defines a recovery fraction "
            "((PP - raw) / (100 - raw)) that our own post-processing arm can "
            "be reported on with the identical formula.",
            "Its post-processing is the SAME protocol family as ours: "
            "largest-fragment + explicit hydrogens + UFF, cited to Buttenschoen "
            "et al. (2025) -- the same reference our own post-processing arm "
            "cites. So the two recipes are comparable in the one case where "
            "the frontier published both sides of it.",
            "HLTF's Table 2 also shows that the largest post-processing effect "
            "in the frontier is on the WORST raw model (GCDM 0.2 -> 95.2, i.e. "
            "95.2 % of its invalid samples recovered). A published 'validity' "
            "number therefore says very little about the raw generator: the "
            "same table contains a 0.2 % and a 95.2 % validity rate for "
            "models whose post-processed numbers differ by 1.0 point.",
        ],
        "not_retrieved": [
            {
                "verdict": "NOT-RETRIEVED",
                "what": "the workshop proceedings PDF / the supplementary tables (Table S1-S4) that the paper's numbers refer to",
                "url": "not fetched in this session",
                "http_status": None,
                "reason": "the arXiv HTML carries the main tables only; the supplementary material was not fetched",
                "consequence": (
                    "every HLTF cell above is stated for the arXiv v1 HTML. The "
                    "per-dataset per-metric detail behind the tables is not "
                    "quoted here."
                ),
            }
        ],
    },
    "YuelBond": {
        "label": "YuelBond -- a learned bond-type decoder for 3D molecules (NOT a molecule generator)",
        "citation": {
            "arxiv_id": None,
            "version_used": "bioRxiv v1 preprint of J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052",
            "title": "YuelBond: a bond-type decoder for reconstructing molecular bonds from atomic coordinates",
            "authors": "not extracted in this session",
            "venue": (
                "bioRxiv preprint (DOI 10.1101/2025.05.06.652517); published in "
                "J. Chem. Inf. Model. 2026, 66(2), 1003-1012 "
                "(doi:10.1021/acs.jcim.5c03052) per the repo's "
                "molmetal/reports/wf_cfm_frontier_research/research_phase1b.md -- "
                "the published version was NOT fetched in this session"
            ),
            "doi": "10.1101/2025.05.06.652517 (preprint); 10.1021/acs.jcim.5c03052 (published, per the repo)",
            "retrieved_at": RETRIEVAL_DATE,
            "retrieval_url": "https://www.biorxiv.org/content/10.1101/2025.05.06.652517v1.full",
        },
        "representation": "a BOND-TYPE CLASSIFIER over given 3D coordinates -- it does not generate molecules and does not generate coordinates",
        "reported_band": {
            "headline": "F1 98.4 % on standard 3-D structures ; F1 92.7 % on distorted CDGs",
            "table": "the abstract and the CDG bonds-reconstruction results paragraph",
            "sample_nature": "a held-out TEST SET of 1,000 molecules with synthetic coordinate noise (sigma = 0.2 A) -- the perturbation is applied BY THE PAPER to a real test set, not produced by a generative model",
        },
        "reported_numbers": [
            {
                "label": "F1-score on distorted CDGs (the band's 92.7 % endpoint, candidate 2)",
                "value": 92.7,
                "unit": "percent (multi-class bond-type classification F1)",
                "locator": "bioRxiv 10.1101/2025.05.06.652517v1, abstract and the 'Performance Evaluation of YuelBond on the CDG Bonds Reconstruction Task' results paragraph",
                "quote": "achieving average accuracy, precision, recall, and F1-score of 92.7%, 93.6%, 92.7%, and 92.7%, respectively",
            },
            {
                "label": "F1-score on standard 3-D structures",
                "value": 98.4,
                "unit": "percent (multi-class bond-type classification F1)",
                "locator": "bioRxiv 10.1101/2025.05.06.652517v1, abstract",
                "quote": "achieving 98.4% F1-score on standard 3D structures and maintaining strong performance (92.7% F1-score) on distorted CDGs, even when RDKit fails on most cases.",
            },
            {
                "label": "RDKit processing rate on the noisy CDG cohort (the frontier's own baseline)",
                "value": 217,
                "unit": "molecules processed of 1,000",
                "locator": "bioRxiv 10.1101/2025.05.06.652517v1, the RDKit comparison paragraph",
                "quote": "Out of 1,000 test molecules with noisy coordinates, RDKit was able to process only 217, while it failed on the remaining 783 molecules.",
            },
        ],
        "cells": {
            "DENOMINATOR": {
                "verdict": "BREAKS",
                "quote": "Out of 1,000 test molecules with noisy coordinates, RDKit was able to process only 217, while it failed on the remaining 783 molecules.",
                "locator": "bioRxiv 10.1101/2025.05.06.652517v1, the RDKit comparison paragraph and the 'Performance Evaluation ... on the CDG Bonds Reconstruction Task' section",
                "note": (
                    "1,000 held-out test-set molecules with SYNTHETIC coordinate "
                    "noise (sigma = 0.2 A) applied by the paper itself. This is "
                    "not a generative-model sampling population at all: the "
                    "paper perturbs REAL test molecules. Ours is one decode of "
                    "one pool. Note the repo's own correction (Wf-A146): the "
                    "earlier wording '78.3 % failure rate' was wrong -- the "
                    "paper prints 783 failures out of 1,000."
                ),
            },
            "METRIC": {
                "verdict": "BREAKS",
                "quote": "The model’s performance is evaluated using standard multi-class classification metrics: accuracy, precision, recall, and F1-score. They are computed on a per-molecule basis and aggregated across the test set.",
                "locator": "bioRxiv 10.1101/2025.05.06.652517v1, Methods -- 'The model's performance is evaluated using standard multi-class classification metrics...'",
                "note": (
                    "THE BAND'S 92.7 % IS A BOND-TYPE CLASSIFICATION F1, NOT A "
                    "PER-MOLECULE RATE. The model's task is to label each "
                    "EXISTING bond as single / double / triple / aromatic, "
                    "given coordinates. The per-bond-type breakdown in the "
                    "paper (single-bond F1 83.3 %, double 56.3 %, triple "
                    "59.9 % on the noisy cohort) makes this explicit. There is "
                    "no per-molecule connectivity count anywhere in the paper. "
                    "A per-bond F1 and a per-molecule connected-component rate "
                    "are not two estimates of one quantity -- they are two "
                    "different observables, and placing our 8.51 % beside a "
                    "bond-type F1 asserts a comparability that does not exist."
                ),
            },
            "POST-PROCESSING": {
                "verdict": "NOT-STATED",
                "quote": "",
                "marker": "NOT-STATED: YuelBond is a DECODER, not a counter. It has no 'before the count' pipeline of its own, so there is no post-processing recipe to state. (Its 'post-processing' in the repo's usage is the DECODER ITSELF replacing RDKit's rule-based xyz2mol.)",
                "locator": "bioRxiv 10.1101/2025.05.06.652517v1 -- the paper describes a bond-reconstruction model with no molecular-counting pipeline",
                "recipe_verdict": "NOT-APPLICABLE",
                "recipe": (
                    "NOT-APPLICABLE: there is no pre-count recipe because there "
                    "is no per-molecule count to pre-process. The paper's own "
                    "post-processing IS the learned decoder."
                ),
                "note": (
                    "Recorded as NOT-STATED rather than BREAKS because the axis "
                    "does not exist for this source, not because the paper "
                    "declined to describe a recipe it has."
                ),
            },
            "HYDROGENS": {
                "verdict": "NOT-STATED",
                "quote": "",
                "marker": "NOT-STATED: the paper reports no per-molecule metric, so no hydrogen convention is stated or implied for any counted molecule.",
                "locator": "bioRxiv 10.1101/2025.05.06.652517v1 -- no per-molecule metric is reported",
                "note": (
                    "The paper's inputs are real 3-D structures (which do carry "
                    "hydrogens) and its outputs are bond labels, not molecules, "
                    "so there is no counted molecule whose hydrogen convention "
                    "could differ from ours."
                ),
            },
        },
        "caveats_stated_by_the_source": [
            "The paper's own per-bond-type breakdown shows the aggregate F1 is a mixture of very different per-type scores (single 83.3 %, double 56.3 %, triple 59.9 % on the noisy cohort).",
            "The paper states the RDKit baseline is handicapped by the distortion it introduces, not by a chemistry failure: 'This limitation arises because RDKit relies on chemically valid input geometries and rule-based heuristics that become unreliable when the structure is distorted.'",
        ],
        "extra_observations": [
            "TWO DIFFERENT PAPERS BOTH PRODUCE A '92.7 %', AND NEITHER IS A "
            "CONNECTIVITY RATE. (i) GeoLDM's Validity&Uniq on QM9 = 92.7 % "
            "+/- 0.5 (HLTF Table 1) is a sanitisation-and-duplication rate; "
            "(ii) YuelBond's bond-type F1 on noisy CDGs = 92.7 % is a "
            "classification score. The repo's band was assembled from a "
            "literature scan; this branch did NOT establish which of the two "
            "was the intended endpoint, and does not claim to. What IS "
            "established: both candidates are NOT connectivity rates."
        ],
        "not_retrieved": [
            {
                "verdict": "NOT-RETRIEVED",
                "what": "the published J. Chem. Inf. Model. version (doi:10.1021/acs.jcim.5c03052) and its supplementary Tables S1-S5",
                "url": "not fetched in this session",
                "http_status": None,
                "reason": "the branch brief supplied the DOI; the bioRxiv preprint was fetched instead",
                "consequence": (
                    "every YuelBond cell above is stated for bioRxiv v1. If the "
                    "published version changed the protocol, these cells are "
                    "stale with respect to it."
                ),
            }
        ],
    },
    "PAFlow": {
        "label": "PAFlow -- Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number",
        "citation": {
            "arxiv_id": "2509.01486",
            "version_used": "v2 (published 2025-09-01, updated 2025-11-04; comment: 'Accepted by NeurIPS 2025')",
            "title": "Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number",
            "authors": "not extracted in this session",
            "venue": "NeurIPS 2025 (per the arXiv comment field)",
            "doi": None,
            "retrieved_at": RETRIEVAL_DATE,
            "retrieval_url": "https://arxiv.org/html/2509.01486v2",
        },
        "representation": "3D all-atom, TARGET-CONDITIONED (SBDD) generation on CrossDocked2020",
        "reported_band": {
            "headline": "NO connectivity or validity band is reported",
            "table": "n/a -- Tables 1/3/4 report Vina Score, Vina Min, Vina Dock, High Affinity, QED, SA, Diversity",
            "sample_nature": "held-out TEST PROTEINS (CrossDocked2020), 100 molecules per test protein",
        },
        "reported_numbers": [
            {
                "label": "molecules sampled per test protein",
                "value": 100,
                "unit": "molecules",
                "locator": "arXiv:2509.01486v2, Section 4.1",
                "quote": "100 molecules are sampled for each test protein.",
            },
            {
                "label": "molecules generated across all test proteins (ablation)",
                "value": 1000,
                "unit": "molecules",
                "locator": "arXiv:2509.01486v2, Section 4.1 (ablation paragraph)",
                "quote": "we generate 1,000 molecules in all test proteins with PAFlow w/o P",
            },
        ],
        "cells": {
            "DENOMINATOR": {
                "verdict": "NOT-STATED",
                "quote": "",
                "marker": "NOT-STATED: this paper reports no connectivity/validity rate, so it has no denominator for one. The denominator below is for its OWN (affinity) metrics, recorded for context only.",
                "locator": "arXiv:2509.01486v2 -- no connectivity/validity denominator exists; Section 4.1 states '100 molecules are sampled for each test protein.' for its AFFINITY metrics",
                "note": (
                    "There is no connectivity/validity band in this paper, so "
                    "there is no denominator for one. For the record, the "
                    "paper's own metrics are sampled 100 molecules per test "
                    "protein (and 1,000 across all test proteins in the "
                    "ablation), on HELD-OUT test proteins -- a genuinely "
                    "different protocol from the unconditional "
                    "training-distribution bands of the other three sources."
                ),
            },
            "METRIC": {
                "verdict": "NOT-STATED",
                "quote": "The generated molecules are evaluated from two aspects: binding affinity with the target protein and molecular properties.",
                "marker": "NOT-STATED: the paper reports NO connectivity metric, NO validity rate and NO sanitisation rate. The quote above is the evidence -- the evaluation axes are affinity and properties, and the quoted metric list names QED / SA / diversity, none of which is a connectivity count.",
                "locator": "arXiv:2509.01486v2, Section 4.1 'Evaluation metrics'",
                "note": (
                    "The paper's evaluation metrics are AutoDock Vina "
                    "affinity measures, High Affinity, QED, SA and diversity -- "
                    "no connectivity, no sanitisation rate, no validity rate. "
                    "MEASURED absence (exhaustive grep of the extracted full "
                    "text): 'sanitiz' 0 hits; 'valid*' 6 hits, ALL of them "
                    "'validation loss' / 'validate' in the TRAINING sense; "
                    "'connect' 1 hit, in the knn-graph construction sentence, "
                    "not a molecular count; 'fragment' 3 hits, all describing "
                    "fragment-BASED generation as a PARADIGM ('fragment by "
                    "fragment', 'molecular fragments', 'unrealistic fragments'), "
                    "not a fragmentation metric; 'valency'/'valence' 0 hits; "
                    "'Vina' 59 hits. This is the answer to the branch's "
                    "fourth-source question: a 4th graph flow-matching paper "
                    "exists and does NOT report a connectivity/validity band "
                    "at all."
                ),
            },
            "POST-PROCESSING": {
                "verdict": "NOT-STATED",
                "quote": "",
                "marker": "NOT-STATED: no post-processing recipe is described for any reported metric, because no molecular count is reported at all.",
                "locator": "arXiv:2509.01486v2 -- no post-processing is described for any reported metric",
                "recipe_verdict": "NOT-APPLICABLE",
                "recipe": (
                    "NOT-APPLICABLE: there is no pre-count recipe to state, "
                    "because the paper counts no molecules -- it reports "
                    "docking scores, QED, SA and diversity."
                ),
                "note": (
                    "Nothing in the paper describes a post-processing recipe "
                    "applied before a molecular count, because no molecular "
                    "count is reported. There is no 'largest fragment' step, "
                    "no sanitisation step and no hydrogen-manipulation step "
                    "to compare against ours."
                ),
            },
            "HYDROGENS": {
                "verdict": "NOT-STATED",
                "quote": "",
                "marker": "NOT-STATED: the paper calls itself 'a 3D all-atom Flow matching model' but states no hydrogen convention for any reported metric, and reports no connectivity/validity metric that a convention could apply to.",
                "locator": "arXiv:2509.01486v2 -- the paper describes itself as 'a 3D all-atom Flow matching model' but states no hydrogen convention for any reported metric",
                "note": (
                    "The paper calls itself 'a 3D all-atom Flow matching "
                    "model' in the abstract, which implies explicit hydrogens "
                    "as nodes, but it reports no connectivity/validity metric, "
                    "so the hydrogen convention of the counted molecule is "
                    "not specified and cannot be compared."
                ),
            },
        },
        "caveats_stated_by_the_source": [
            "The paper's evaluation is on HELD-OUT test proteins, unlike the unconditional training-distribution bands of the other three sources -- so it is not a like-for-like frontier comparator either.",
        ],
    },
}

#: every quote embedded above, with the local file it was verified against.
#: ``--verify-quotes-from DIR`` re-checks these against locally saved
#: extractions of the retrieved pages.
QUOTE_VERIFICATION_INDEX: List[Dict[str, Any]] = [
    {"id": "EFT_DEN_1", "file": "2609.08333v4.txt", "quote": "Panel (a) rates are computed over 10,000 generated molecules."},
    {"id": "EFT_DEN_2", "file": "2609.08333v4.txt", "quote": "All rates are computed over 10,000 generated molecules."},
    {"id": "EFT_DEN_3", "file": "2609.08333v4.txt", "quote": "Connectivity, sanitization, unique sanitized yield and novel unique sanitized yield use all generated samples as their denominator"},
    {"id": "EFT_DEN_4", "file": "2609.08333v4.txt", "quote": "using a 90%/10% train/validation split after preprocessing and the same H100 hardware setting"},
    {"id": "EFT_DEN_5", "file": "2609.08333v4.txt", "quote": "its official training split provides DFT-optimized 3D structures for approximately 3.4M molecules"},
    {"id": "EFT_MET_1", "file": "2609.08333v4.txt", "quote": "Connectivity: % of molecules with exactly one connected component, i.e., one fragment."},
    {"id": "EFT_BAND_1", "file": "2609.08333v4.txt", "quote": "C. | 99.60 | 100.00 | 92.98"},
    {"id": "EFT_BAND_2", "file": "2609.08333v4.txt", "quote": "S. | 83.84 | 98.72 | 98.25"},
    {"id": "EFT_BAND_3", "file": "2609.08333v4.txt", "quote": "U. & S. | 83.81 | 98.27 | 98.21"},
    {"id": "EFT_BAND_4", "file": "2609.08333v4.txt", "quote": "N. & U. & S. | 76.03 | 69.83 | 94.33"},
    {"id": "EFT_BAND_5", "file": "2609.08333v4.txt", "quote": "Panel (b) PoseBusters rates are computed among the N. & U. & S. molecules: 7,603 for UAE-3D, 6,983 for FlowMol, and 9,433 for EF-TALFM."},
    {"id": "EFT_BAND_6", "file": "2609.08333v4.txt", "quote": "All three methods are trained for heavy-atom generation on PCQM4Mv2"},
    {"id": "EFT_MET_2", "file": "2609.08333v4.txt", "quote": "the fraction of all generated samples that are unique, absent from the training set, sanitized, and pass the PoseBusters checks used here"},
    {"id": "EFT_MET_3", "file": "2609.08333v4.txt", "quote": "To derive a generated heavy-atom molecule's bonds, we first apply a deterministic chemistry-aware reconstruction procedure that is separate from the learned generator."},
    {"id": "EFT_PP_1", "file": "2609.08333v4.txt", "quote": "We report connectivity of the reconstructed molecule and sanitization, uniqueness, and novelty rates computed after retaining the atom-count largest connected component of each sampled molecule and applying strict RDKit sanitization."},
    {"id": "EFT_PP_2", "file": "2609.08333v4.txt", "quote": "For fragmented outputs, SMILES-based and PoseBusters metrics are computed on the atom-count largest connected component after RDKit sanitization."},
    {"id": "EFT_PP_3", "file": "2609.08333v4.txt", "quote": "For fragmented outputs, SMILES-based and PoseBusters metrics use the atom-count largest connected component after sanitization."},
    {"id": "EFT_PP_4", "file": "2609.08333v4.txt", "quote": "we retain the largest connected component and discard the rest, since the reconstructed graph may contain multiple components."},
    {"id": "EFT_PP_5", "file": "2609.08333v4.txt", "quote": "For heavy-atom generation, RDKit first adds temporary explicit hydrogens; all heavy atoms are then restrained near their decoded positions during minimization (Step 3), and the refined heavy-atom coordinates are transferred back while the temporary hydrogens are discarded (Step 4)."},
    {"id": "EFT_PP_6", "file": "2609.08333v4.txt", "quote": "Among the 92,221 connected molecules, 90,807 reconstructed graphs pass strict sanitization before geometry refinement."},
    {"id": "EFT_H_1", "file": "2609.08333v4.txt", "quote": "Following common practice in 3D small-molecule generation, we model only the heavy-atom scaffold."},
    {"id": "EFT_H_2", "file": "2609.08333v4.txt", "quote": "the heavy-atom branch, which returns an implicit-hydrogen molecule"},
    {"id": "EFT_H_3", "file": "2609.08333v4.txt", "quote": "Identity is compared using RDKit canonical non-isomeric SMILES after removing explicit hydrogens, retaining bond orders and formal charges without tautomer normalization."},
    {"id": "FM3_DEN_1", "file": "2508.12629v1.txt", "quote": "For all baseline models except ADiT we sampled 5000 molecules from the trained models using default settings provided by the authors."},
    {"id": "FM3_DEN_2", "file": "2508.12629v1.txt", "quote": "Sampled molecules are split randomly into 5 subsets before computing metrics; we obtain 95% confidence intervals on the mean metric values by assuming the sample mean is normally distributed with the standard deviation in means obtained from the five subsets."},
    {"id": "FM3_DEN_3", "file": "fm3_evals_readme.md", "quote": "--n_timesteps=250 --metrics --n_mols=5000 --reos_raw --n_subsets=5"},
    {"id": "FM3_DEN_4", "file": "fm3_evals_baselines_compute_baseline_comparison.py", "quote": "mols_per_subset = len(rdkit_mols) // args.n_subsets"},
    {"id": "FM3_DEN_5", "file": "fm3_evals_baselines_compute_baseline_comparison.py", "quote": "n_raw_samples = len(rdkit_mols)"},
    {"id": "FM3_DEN_6", "file": "2508.12629v1.txt", "quote": "Metrics on the sampled molecules were then computed using the same script in the FlowMol repository."},
    {"id": "FM3_MET_1", "file": "2508.12629v1.txt", "quote": "We report “% Valid” which is the percent of molecules that can be sanitized by rdkit."},
    {"id": "FM3_MET_2", "file": "2508.12629v1.txt", "quote": "We additionally report “% PB-valid,” which is the percent of molecules that are valid and pass several other physical plausibility checks implemented in the PoseBusters package"},
    {"id": "FM3_MET_3", "file": "2508.12629v1.txt", "quote": "These additional checks include whether the molecule forms one component (not fragmented into separate molecules), has reasonable bond lengths and angles, has planar aromatic rings and planar double bonds, has no steric clashes, and whether the internal energy of the conformer is reasonably close to that of an ensemble of conformers produced by ETKDGv3"},
    {"id": "FM3_MET_4", "file": "2508.12629v1.txt", "quote": "We argue that producing “valid” molecules is a necessary but insufficient condition for a practically useful molecular generative model."},
    {"id": "FM3_MET_5", "file": "flowmol_analysis_metrics.py", "quote": "n_pb_valid = df_pb[df_pb['sanitization'] == True].values.astype(bool).all(axis=1).sum()"},
    {"id": "FM3_MET_6", "file": "flowmol_analysis_metrics.py", "quote": "rdmols = [sample.rdkit_mol for sample in sampled_molecules]"},
    {"id": "FM3_PP_8", "file": "flowmol_analysis_metrics.py", "quote": "if len(mol_frags) > 1: error_message['disconnected'] += 1 else: n_connected += 1"},
    {"id": "FM3_PP_9", "file": "flowmol_analysis_metrics.py", "quote": "'frac_valid_mols': n_valid / len(sampled_molecules)"},
    {"id": "FM3_PP_10", "file": "flowmol_analysis_metrics.py", "quote": "'frac_connected': n_connected / len(sampled_molecules)"},
    {"id": "FM3_PP_11", "file": "flowmol_analysis_metrics.py", "quote": "'avg_frag_frac': sum(frag_fracs) / len(frag_fracs),"},
    {"id": "FM3_PP_12", "file": "flowmol_analysis_metrics.py", "quote": "largest_frag_frac = largest_mol_n_atoms / mol.num_atoms"},
    {"id": "FM3_PP_1", "file": "flowmol_analysis_metrics.py", "quote": "mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)"},
    {"id": "FM3_PP_2", "file": "flowmol_analysis_metrics.py", "quote": "largest_mol = max(mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms())"},
    {"id": "FM3_PP_3", "file": "flowmol_analysis_metrics.py", "quote": "Chem.SanitizeMol(largest_mol)"},
    {"id": "FM3_PP_4", "file": "flowmol_analysis_metrics.py", "quote": "smiles = Chem.MolToSmiles(largest_mol)"},
    {"id": "FM3_PP_5", "file": "fm3_evals_baselines_compute_baseline_comparison.py", "quote": "supplier = Chem.SDMolSupplier(str(args.sample_file), sanitize=False, removeHs=False)"},
    {"id": "FM3_PP_6", "file": "fm3_evals_baselines_compute_baseline_comparison.py", "quote": "metrics['frac_valid_mols']= metrics['frac_valid_mols'] * len(sampled_mols) / n_raw_samples"},
    {"id": "FM3_PP_7", "file": "2508.12629v1.txt", "quote": "we kekulize all molecules in the dataset and do not explicitly model aromatic bonds."},
    {"id": "FM3_H_1", "file": "2508.12629v1.txt", "quote": "We train models on the GEOM-Drugs dataset 57 using explicit hydrogens."},
    {"id": "FM3_H_2", "file": "2508.12629v1.txt", "quote": "All models were trained on GEOM-Drugs with explicit hydrogens."},
    {"id": "FM3_H_3", "file": "mol_builder.py", "quote": "mol = Chem.RWMol()"},
    {"id": "FM3_H_4", "file": "mol_builder.py", "quote": "a = Chem.Atom(atom_type)"},
    {"id": "FM3_H_5", "file": "pb_config.yaml", "quote": "ignore_hydrogens: True"},
    {"id": "FM3_H_6", "file": "pb_config.yaml", "quote": "add_hs: False"},
    {"id": "GG_DEN_1", "file": "2411.05676v1.txt", "quote": "To calculate these metrics, we sampled 10,000 molecules."},
    {"id": "GG_DEN_2", "file": "2411.05676v1.txt", "quote": "Results are the means of 3 different runs."},
    {"id": "GG_DEN_3", "file": "2411.05676v1.txt", "quote": "We follow the train/test split from GDSS, using 12,000 molecules for training and the remaining 1,000 for testing."},
    {"id": "GG_MET_1", "file": "2411.05676v1.txt", "quote": "Validity We permit atoms to exhibit formal charges during valency checks because of the presence of formal charges in the training molecules. It is the fraction of valid molecules after valency correction or edge resampling."},
    {"id": "GG_MET_2", "file": "2411.05676v1.txt", "quote": "Validity w/o correction This metric explicitly evaluates the quality of molecule generation before any correction phase, providing a baseline for raw generation performance."},
    {"id": "GG_PP_1", "file": "2411.05676v1.txt", "quote": "It is the fraction of valid molecules after valency correction or edge resampling."},
    {"id": "GG_PP_2", "file": "2411.05676v1.txt", "quote": "Validity w/o correction This metric explicitly evaluates the quality of molecule generation before any correction phase, providing a baseline for raw generation performance."},
    {"id": "GG_PP_3", "file": "2411.05676v1.txt", "quote": "The high Validity without correction suggests that GGFlow successfully learns chemical valency rules."},
    {"id": "GG_PP_4", "file": "2411.05676v1.txt", "quote": "each containing up to 9 heavy atoms: carbon, oxygen, nitrogen, and fluorine"},
    {"id": "GG_H_1", "file": "2411.05676v1.txt", "quote": "QM9 | Real | 133,885 | [1, 9] | 4 | 3"},
    {"id": "PAF_DEN_1", "file": "2509.01486v2.txt", "quote": "100 molecules are sampled for each test protein."},
    {"id": "PAF_DEN_2", "file": "2509.01486v2.txt", "quote": "we generate 1,000 molecules in all test proteins with PAFlow w/o P"},
    {"id": "PAF_MET_1", "file": "2509.01486v2.txt", "quote": "The generated molecules are evaluated from two aspects: binding affinity with the target protein and molecular properties"},
    {"id": "PAF_MET_2", "file": "2509.01486v2.txt", "quote": "Following [ 3 , 22 ] , QED [ 42 ] (drug-likeness), SA [ 43 ] (synthesize accessibility), and diversity are adopted to evaluate critical molecular properties."},
    {"id": "PAF_DS_1", "file": "2509.01486v2.txt", "quote": "We train and evaluate PAFlow on the CrossDocked2020 dataset"},
    {"id": "PAF_TITLE", "file": "2509.01486v2.txt", "quote": "Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number"},
    {"id": "HLT_DEN_1", "file": "2603.27113v1.txt", "quote": "On QM9 we report atom stability (AS), molecule stability (MS), RDKit validity (Val), and Val&Uniq on 10,000 unconditional samples."},
    {"id": "HLT_DEN_2", "file": "2603.27113v1.txt", "quote": "On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation Rappe et al. (1992) )."},
    {"id": "HLT_MET_1", "file": "2603.27113v1.txt", "quote": "Val&Uniq (%) | 92.7 ± 0.5 | 93.3 ± 0.0 * | 92.9 ± 0.2"},
    {"id": "HLT_MET_2", "file": "2603.27113v1.txt", "quote": "PB-Valid (%) | 93.1 ± 0.4 | 91.9 ± 0.5 | 94 ± 0.5 *"},
    {"id": "HLT_PP_1", "file": "2603.27113v1.txt", "quote": "HLTF (ours) | 85.5 | 85.4 | 85.0"},
    {"id": "HLT_PP_2", "file": "2603.27113v1.txt", "quote": "+PP | 92.2 | 92.0 | 91.2"},
    {"id": "HLT_PP_3", "file": "2603.27113v1.txt", "quote": "For HLTF it raises Valid from 85.5 to 92.2, converting ≈ 46 % of the pre-PP invalid samples under the same pipeline."},
    {"id": "HLT_PP_4", "file": "2603.27113v1.txt", "quote": "FlowMol | 59.8 | 59.8 | 59.7"},
    {"id": "HLT_PP_5", "file": "2603.27113v1.txt", "quote": "GCDM | 0.2 | 0.2 | 0.2"},
    {"id": "HLT_PP_6", "file": "2603.27113v1.txt", "quote": "+PP | 95.2 | 95.2 | 95.2"},
    {"id": "HLT_PP_7", "file": "2603.27113v1.txt", "quote": "Standardized post-processing mainly relieves geometric strain (e.g., unrealistic bond lengths/angles and local steric clashes) and cannot repair infeasible topology (e.g., invalid valence or disconnected graphs)."},
    {"id": "HLT_PP_8", "file": "2603.27113v1.txt", "quote": "Without post-processing, HLTF achieves Valid/V&U/V&U&N = 85.5/85.4/85.0, within 2.0 points of the best raw Valid in the table. With post-processing, HLTF reaches 92.2/92.0/91.2"},
    {"id": "YB_MET_1", "file": "yuelbond.txt", "quote": "achieving average accuracy, precision, recall, and F1-score of 92.7%, 93.6%, 92.7%, and 92.7%, respectively"},
    {"id": "YB_MET_2", "file": "yuelbond.txt", "quote": "achieving 98.4% F1-score on standard 3D structures and maintaining strong performance (92.7% F1-score) on distorted CDGs, even when RDKit fails on most cases."},
    {"id": "YB_DEN_1", "file": "yuelbond.txt", "quote": "Out of 1,000 test molecules with noisy coordinates, RDKit was able to process only 217, while it failed on the remaining 783 molecules."},
    {"id": "YB_MET_3", "file": "yuelbond.txt", "quote": "The model’s performance is evaluated using standard multi-class classification metrics: accuracy, precision, recall, and F1-score. They are computed on a per-molecule basis and aggregated across the test set."},
    # ---------------------------------------------------------------------
    # 2026-11-01 (the RESIDUAL-2 / RESIDUAL-3 repair) -- the GeoLDM
    # primary-source pins, and the HLTF / EF-TALFM LaTeX-source pins that
    # carry the check-count attribution.  These are read out of the arXiv
    # E-PRINT SOURCE TARBALLS (not a rendered page), which is why the
    # extraction files are .tex and the quotes are LaTeX.  The e-print and
    # per-file sha256 pins match the Wf-A186 primary-source artefact.
    # ---------------------------------------------------------------------
    {"id": "GLD_METHOD", "file": "2305.01140v1_icml2023.tex", "quote": "\\newcommand{\\method}{\\textsc{GeoLDM}\\xspace}"},
    {"id": "GLD_ROW", "file": "2305.01140v1_icml2023.tex", "quote": "\\rowcolor{lightgray} \\textbf{\\method} & \\textbf{98.9} $\\pm$ 0.1 & \\textbf{89.4} $\\pm$ 0.5 & \\textbf{93.8} $\\pm$ 0.4 & \\textbf{92.7} $\\pm$ 0.5 & \\textbf{84.4} & \\textbf{99.3} \\\\"},
    {"id": "GLD_CAP", "file": "2305.01140v1_icml2023.tex", "quote": "Metrics are calculated with 10000 samples generated from each model. On QM9, we run the evaluation for 3 times and report the derivation."},
    {"id": "GLD_SAMPLES", "file": "2305.01140v1_icml2023.tex", "quote": "We generate $10,000$ samples from each method to calculate the above metrics, and the results are reported in \\cref{tab:qm9_results}."},
    {"id": "GLD_METRICS", "file": "2305.01140v1_icml2023.tex", "quote": "Given molecular geometries, we first predict bond types (single, double, triple, or none) by pair-wise atomic distances and atom types."},
    {"id": "GLD_QM9", "file": "2305.01140v1_icml2023.tex", "quote": "QM9 contains 3D structures together with several quantum properties for 130k small molecules, limited to 9 heavy atoms (29 atoms including hydrogens)."},
    {"id": "GLD_FRAG", "file": "2305.01140v1_icml2023.tex", "quote": "the model occasionally generates disconnected components, as shown in the rightest column of \\cref{app:fig:vis-drug}, which happens more often when trained on the large molecule DRUG dataset. However, this phenomenon actually is not a problem and is common in all non-autoregressive molecule generative models~\\cite{zang2020moflow,jo2022score}, and can be easily fixed by just filtering the smaller components."},
    {"id": "HLT_ROW38", "file": "2603.27113v1_qm9_benchmarks_og.tex", "quote": "GeoLDM & $98.9\\!\\pm\\!0.1$* & $89.4\\!\\pm\\!0.5$* & $93.6\\!\\pm\\!0.2$ & $92.7\\!\\pm\\!0.5$ \\\\"},
    {"id": "HLT_ROW63", "file": "2603.27113v1_qm9_benchmarks_og.tex", "quote": "Val (\\%) $\\uparrow$           & $93.6\\pm0.2$       & $94.8\\pm0.2$*   & \\underline{$93.8\\pm0.6$} \\\\"},
    {"id": "HLT_ROW66", "file": "2603.27113v1_qm9_benchmarks_og.tex", "quote": "PB-Valid (\\%) $\\uparrow$      & $93.1\\pm0.4$       & $91.9\\pm0.5$    & \\textbf{$94\\pm0.5$*} \\\\"},
    {"id": "HLT_CAP", "file": "2603.27113v1_qm9_benchmarks_og.tex", "quote": "Baselines reproduced from GCDM Table~1 \\cite{morehead2023gcdm}."},
    {"id": "EFT_SIX", "file": "2609.08333v4_appendix.tex", "quote": "pass strict sanitization and all six PoseBusters sanity checks used here~\\citep{Buttenschoen2024posebuster}, divided by all generated samples."},
    {"id": "EFT_LIST", "file": "2609.08333v4_appendix.tex", "quote": "We use the six checks listed below from the PoseBusters \\texttt{mol} configuration."},
    {"id": "EFT_CONN", "file": "2609.08333v4_appendix.tex", "quote": "\\item Connectivity: \\% of molecules with exactly one connected component, i.e., one fragment."},
    {"id": "EFT_ALL", "file": "2609.08333v4_appendix.tex", "quote": "\\textbf{All PoseBusters checks} & 98.91 & \\bfseries 99.73 & 93.13 \\\\"},
]


# ---------------------------------------------------------------------------
# Normalisation used by --verify-quotes-from
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Fold the cosmetic differences between the rendered page and the quote.

    arXiv's LaTeX HTML renders every inline number twice (rendered value +
    LaTeX source), and writes thousands separators as ``1{,}000``.  Both are
    artefacts of the HTML renderer, not of the paper.
    """
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    text = re.sub(r"(\d{1,3})\{,\}(\d{3})", r"\1,\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dedup_math(text: str) -> str:
    """Collapse the doubled inline math arXiv's HTML emits.

    arXiv's LaTeX HTML renders every inline formula twice: once as the
    rendered form (``92,221``) and once as the LaTeX source (``92{,}221``).
    The renderer artefacts are not part of the paper.
    """
    prev = None
    while prev != text:
        prev = text
        # plain doubling: "92,221 92,221" -> "92,221"
        text = re.sub(r"(\d[\d,\.]*)\s+\1\b", r"\1", text)
        # "a ± b" then its LaTeX source "a\pm b"
        text = re.sub(
            r"(\d[\d,\.]*)\s*±\s*(\d[\d,\.]*)\s+\1\\pm\s*\2", r"\1 ± \2", text
        )
        # "≈ n %" then its LaTeX source "\approx n\%"
        text = re.sub(
            r"(≈\s*)(\d[\d,\.]*)\s*%\s*\\approx\s*\2\\%", r"\1\2 %", text
        )
        # the higher-is-better table legend renders as "↑ \uparrow" (glyph +
        # LaTeX source).  It is a legend, not part of the numeric claim.
        text = re.sub(r"↑\s*\\uparrow", "", text)
        text = re.sub(r"\s{2,}", " ", text)
    return text


def verify_quotes(verify_dir: Path) -> List[str]:
    """Re-check every embedded quote against locally saved extractions.

    Returns a list of human-readable problems (empty == all quotes verified).
    """
    problems: List[str] = []
    cache: Dict[str, str] = {}
    for item in QUOTE_VERIFICATION_INDEX:
        path = verify_dir / item["file"]
        if not path.exists():
            problems.append(f"{item['id']}: missing extraction file {path}")
            continue
        if item["file"] not in cache:
            cache[item["file"]] = dedup_math(normalize(path.read_text(encoding="utf-8", errors="replace")))
        quote = dedup_math(normalize(item["quote"]))
        if quote not in cache[item["file"]]:
            problems.append(f"{item['id']}: quote NOT found in {item['file']}: {quote[:120]!r}")
    return problems


def quote_count() -> int:
    """How many quotes are pinned for verification."""
    return len(QUOTE_VERIFICATION_INDEX)


# ---------------------------------------------------------------------------
# Build + validate
# ---------------------------------------------------------------------------


def build_matrix() -> Dict[str, Any]:
    """Return the matrix artifact as a plain dict (deterministic).

    Wf-A180 (2026-11-01) -- the producer now REPRODUCES the artefact on disk
    instead of diverging from it.  Before this wave a re-run of this function
    emitted 710 leaves where the record carries 739, so a re-run would have
    DELETED the Wf-A177 additive history; and the markdown renderer emitted
    NEITHER dated correction section, so a re-run would have deleted the whole
    rule-#6 record of the contest.  Both are now emitted from here.
    """
    matrix = {
        "artifact": "wfa158_frontier_postproc_matrix",
        "artifact_version": "1.0",
        "generated_by": "molmetal/scripts/wfa158_frontier_postproc_matrix.py",
        "gpu": "0 (CPU-only, no torch, no checkpoint, no network at build time)",
        "retrieval_date": RETRIEVAL_DATE,
        "scope": (
            "Comparability matrix for the published frontier connectivity / "
            "validity bands. One row per source, one column per break. Each "
            "cell records what the SOURCE says, with a verbatim quote and a "
            "locator. THIS IS NOT A SCOREBOARD: a BREAKS verdict means the "
            "two protocols are not the same measurement, not that either "
            "side is wrong."
        ),
        "verdict_vocabulary": {
            "BREAKS": "the source's protocol differs from ours on this axis",
            "DOES-NOT-BREAK": "the two protocols agree on this axis",
            "NOT-STATED": "the source does not state this at all",
            "NOT-RETRIEVED": "we could not retrieve this (e.g. a bot-challenge page)",
        },
        "breaks": list(BREAKS),
        "verdicts": list(VERDICTS),
        "anchor_our_side": ANCHOR,
        "band_provenance": copy.deepcopy(BAND_PROVENANCE),
        "sources": {name: SOURCES[name] for name in SOURCES},
        "retrieval_log": RETRIEVAL_LOG,
        "honest_framing": [
            "This matrix compares PROTOCOLS, not models. It is NOT a "
            "scoreboard. It does not rank our decode against the frontier and "
            "it does not conclude that our model is bad or that the frontier "
            "is cheating.",
            "A NOT-STATED cell is a first-class result, not a gap: it is "
            "recorded here rather than filled in from memory.",
            "No frontier number is quoted unless it was retrieved live in the "
            "session dated in retrieval_log.",
            "Verdicts are relative to the anchor (our RAW, no-post-processing "
            "measurement). If the anchor changes, the verdicts must be re-read.",
        ],
        "what_this_branch_did_not_establish": [
            "The retitled TMLR version of GGFlow (OpenReview K8RlXtMgzo) was "
            "NOT retrieved (HTTP 403 browser challenge); only arXiv v1 was read.",
            "The Digital Discovery 2026 5(5):2052-2066 publication of FlowMol3 "
            "was NOT independently verified; only arXiv v1 was read.",
            "Author lists for GGFlow and PAFlow were not extracted.",
            "The FlowMol3 paper does not state its own row's denominator; the "
            "5,000 figure comes from the authors' public evaluation harness, "
            "not from the paper.",
            "GGFlow's 'valency correction or edge resampling' phase is named "
            "but never described in the paper; the recipe is NOT-RETRIEVED "
            "beyond the metric definition.",
            "The raw HTML/PDF of these papers was read on the retrieval date; "
            "a later revision of any source could change a cell.",
        ],
    }
    # The ONE dated key that records this reconciliation (AGENTS.md §5.5
    # rule #6).  It is COPIED so that the module constant is never mutated,
    # and its byte evidence is filled in from the artefact this producer
    # actually emits (see :func:`_fill_byte_evidence`).
    matrix["band_provenance"][RECONCILIATION_KEY] = copy.deepcopy(RECONCILIATION)
    # 2026-11-02 (Wf-A194) -- the locator-class + GeoLDM dated key.
    matrix["band_provenance"][LOCATOR_CLASS_KEY] = copy.deepcopy(LOCATOR_CLASS)
    # 2026-11-02 (Wf-A197) -- the residual-closure key and the R2 note.
    matrix["band_provenance"][WFA197_RESIDUALS_KEY] = copy.deepcopy(WFA197_RESIDUALS)
    matrix["band_provenance"][WFA197_R2_NOTE_KEY] = copy.deepcopy(WFA197_R2_NOTE)
    _fill_byte_evidence(matrix)
    return matrix


def validate_matrix(matrix: Dict[str, Any]) -> List[str]:
    """Return a list of invariant violations (empty == the matrix is valid)."""
    problems: List[str] = []

    if matrix.get("breaks") != list(BREAKS):
        problems.append(f"breaks must be exactly {list(BREAKS)}, got {matrix.get('breaks')}")

    sources = matrix.get("sources", {})
    if not sources:
        problems.append("no sources in the matrix")

    for name, src in sources.items():
        cit = src.get("citation", {})
        for key in ("label", "representation", "reported_band"):
            if not str(src.get(key, "")).strip():
                problems.append(f"{name}: source is missing {key!r}")
        if not cit.get("retrieval_url"):
            problems.append(f"{name}: citation.retrieval_url is empty")
        if not cit.get("retrieved_at"):
            problems.append(f"{name}: citation.retrieved_at is empty")
        if not cit.get("arxiv_id") and not cit.get("doi"):
            problems.append(
                f"{name}: citation carries neither an arxiv_id nor a doi"
            )

        cells = src.get("cells", {})
        if set(cells) != set(BREAKS):
            problems.append(f"{name}: cells must cover exactly {list(BREAKS)}, got {sorted(cells)}")
        for brk in BREAKS:
            cell = cells.get(brk)
            if not isinstance(cell, dict):
                problems.append(f"{name}/{brk}: cell is not a dict")
                continue
            verdict = cell.get("verdict")
            if verdict not in VERDICTS:
                problems.append(f"{name}/{brk}: bad verdict {verdict!r}")
            quote = cell.get("quote")
            quote = quote.strip() if isinstance(quote, str) else ""
            locator = cell.get("locator")
            locator = locator.strip() if isinstance(locator, str) else ""
            marker = cell.get("marker")
            marker = marker.strip() if isinstance(marker, str) else ""
            if not locator:
                problems.append(f"{name}/{brk}: empty locator")
            if verdict in MARKER_VERDICTS:
                # A marker cell must SAY what was not found.  It MAY also
                # carry a supporting quote (e.g. PAFlow's METRIC cell quotes
                # the metrics the paper *does* report, which is the evidence
                # that the connectivity metric is absent).
                if not marker:
                    problems.append(
                        f"{name}/{brk}: verdict {verdict!r} carries no explicit marker"
                    )
            elif not quote:
                problems.append(
                    f"{name}/{brk}: verdict {verdict!r} carries no quote "
                    f"(only {list(MARKER_VERDICTS)} may go quoteless)"
                )
            if "recipe_verdict" in cell:
                rv = cell["recipe_verdict"]
                if rv not in RECIPE_VERDICTS:
                    problems.append(f"{name}/{brk}: bad recipe_verdict {rv!r}")
                if not str(cell.get("recipe", "")).strip():
                    problems.append(f"{name}/{brk}: recipe_verdict set but recipe is empty")
            if brk == "POST-PROCESSING" and "recipe_verdict" not in cell:
                problems.append(f"{name}/{brk}: POST-PROCESSING must state recipe_verdict")

        for nr in src.get("not_retrieved", []):
            if nr.get("verdict") != "NOT-RETRIEVED":
                problems.append(f"{name}: not_retrieved entry has verdict {nr.get('verdict')!r}")
            for key in ("what", "reason", "consequence"):
                if not str(nr.get(key, "")).strip():
                    problems.append(f"{name}: not_retrieved entry missing {key!r}")

        for num in src.get("reported_numbers", []):
            if not isinstance(num, dict):
                problems.append(f"{name}: reported_numbers entry is not a dict")
                continue
            for key in ("label", "value", "locator", "quote"):
                if key not in num or num[key] in (None, ""):
                    problems.append(f"{name}: reported_numbers entry missing {key!r}: {num.get('label')!r}")
            if "value" in num and not isinstance(num["value"], (int, float)):
                problems.append(f"{name}: reported_numbers value is not numeric: {num.get('label')!r}")

    log = matrix.get("retrieval_log", [])
    if not log:
        problems.append("retrieval_log is empty")
    for entry in log:
        for key in ("url", "retrieved_at", "extracted"):
            if not entry.get(key):
                problems.append(f"retrieval_log entry missing {key!r}: {entry.get('url')!r}")

    # every source must be backed by at least one retrieval-log entry
    urls = {e.get("url") for e in log}
    for name, src in sources.items():
        url = src.get("citation", {}).get("retrieval_url")
        if url not in urls:
            problems.append(f"{name}: retrieval_url {url!r} has no retrieval_log entry")

    problems.extend(_validate_band_provenance(matrix, sources))

    return problems


def _validate_band_provenance(
    matrix: Dict[str, Any], sources: Dict[str, Any]
) -> List[str]:
    """The band-provenance block must be traceable, not asserted.

    Every endpoint candidate must name a source we read, carry a locator and
    a quote, and every raw->post-processed pair must be the arithmetic it
    claims.
    """
    problems: List[str] = []
    bp = matrix.get("band_provenance")
    if not isinstance(bp, dict):
        return ["band_provenance is missing or is not a dict"]

    for key in ("band_as_quoted_in_the_repo", "headline_finding",
                "where_the_repo_states_it", "what_this_does_not_settle"):
        if not bp.get(key):
            problems.append(f"band_provenance: {key} is empty")

    pinned = {q["quote"] for q in QUOTE_VERIFICATION_INDEX}
    for entry in bp.get("endpoint_candidates", []):
        if not entry.get("endpoint"):
            problems.append("band_provenance: endpoint_candidate has no endpoint")
        for key, cand in entry.items():
            if not key.startswith("candidate_"):
                continue
            # 2026-11-02 (Wf-A194) -- a DATED SIBLING of a candidate
            # (`candidate_a_what_it_is_superseded_2026_11_01` and its dated
            # note) holds a STRING, not a candidate object.  Same defect as
            # in `render_markdown_body`: the loop assumed every
            # `candidate_*` key held a dict and raised AttributeError on the
            # string, so the validator could not run at all.  DATED siblings
            # are not candidates and are not validated as such.
            if not isinstance(cand, dict):
                continue
            for field in ("paper", "what_it_is", "denominator", "locator",
                          "quote", "status"):
                if not str(cand.get(field, "")).strip():
                    problems.append(
                        f"band_provenance: {entry.get('endpoint')} {key} missing {field!r}"
                    )
            if cand.get("quote") and cand["quote"] not in pinned:
                problems.append(
                    f"band_provenance: {entry.get('endpoint')} {key} quote is not pinned "
                    "for re-verification"
                )

    ev = bp.get("post_processing_evidence_from_the_frontier_itself", {})
    for row in ev.get("raw_to_post_processed_pairs", []):
        try:
            delta = float(row["pp"]) - float(row["raw"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"band_provenance: bad raw->pp pair {row!r}")
            continue
        if abs(delta - float(row.get("delta_pp", -1))) > 1e-9:
            problems.append(
                f"band_provenance: {row.get('model')!r} delta_pp {row.get('delta_pp')!r} "
                f"!= pp - raw = {delta}"
            )
    rec = ev.get("recovery_fraction_our_side", {})
    if rec:
        for field in ("formula", "units", "raw", "post_processed", "value"):
            if rec.get(field) in (None, ""):
                problems.append(f"band_provenance: recovery_fraction_our_side missing {field!r}")
        raw = rec.get("raw")
        pp = rec.get("post_processed")
        if isinstance(raw, (int, float)) and isinstance(pp, (int, float)) and raw != pp:
            expected = (float(pp) - float(raw)) / (1.0 - float(raw))
            if abs(expected - float(rec.get("value", -1))) > 1e-9:
                problems.append(
                    f"band_provenance: recovery_fraction value {rec.get('value')!r} "
                    f"!= (pp - raw) / (1 - raw) = {expected}"
                )

    return problems


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown_body(matrix: Dict[str, Any]) -> str:
    """Render the BODY of the markdown -- everything before the dated sections.

    The body is the LIVE rendering of the record.  Two rules it obeys:

    * it lists the sources in the SAME ORDER as the JSON this producer
      ships (the record is written with ``sort_keys=True``, so the JSON
      lists them alphabetically -- a body in a different order would make
      the two artefacts describe the same matrix in two different orders);
    * it carries NO superseded claim: every superseded rendering lives in a
      DATED section (AGENTS.md §5.5 rule #6).
    """
    lines: List[str] = []
    a = lines.append
    a("# Wf-A158 BRANCH FRONT -- frontier post-processing comparability matrix")
    a("")
    a(f"Retrieved: **{matrix['retrieval_date']}**. Built by "
      f"`molmetal/scripts/wfa158_frontier_postproc_matrix.py` (CPU-only, 0 GPU).")
    a("")
    a("> " + matrix["scope"])
    a("")
    a("## The matrix")
    a("")
    a("| source | " + " | ".join(BREAKS) + " |")
    a("|---|" + "---|" * len(BREAKS))
    for name in sorted(matrix["sources"]):
        src = matrix["sources"][name]
        cells = src["cells"]
        row = [f"**{name}**"]
        for brk in BREAKS:
            row.append(cells[brk]["verdict"])
        a("| " + " | ".join(row) + " |")
    a("")
    for name in sorted(matrix["sources"]):
        src = matrix["sources"][name]
        cit = src["citation"]
        a(f"## {name}")
        a("")
        a(src["label"])
        a("")
        ident = (
            f"arXiv **{cit['arxiv_id']}**"
            if cit.get("arxiv_id")
            else f"DOI **{cit['doi'].split(';')[0].strip()}** (no arXiv id)"
        )
        a(f"* {ident} ({cit['version_used']}) -- {cit['title']}")
        a(f"* retrieved {cit['retrieved_at']} from `{cit['retrieval_url']}`")
        a(f"* venue: {cit['venue']}")
        a(f"* representation: {src['representation']}")
        a(f"* reported band: **{src['reported_band']['headline']}** ({src['reported_band']['table']}); "
          f"sample nature: {src['reported_band']['sample_nature']}")
        a("")
        for brk in BREAKS:
            cell = src["cells"][brk]
            a(f"### {brk} -- {cell['verdict']}")
            a("")
            if cell.get("marker"):
                a(f"**{cell['marker']}**")
                a("")
            if cell.get("quote"):
                a(f"> {cell['quote']}")
                a("")
            a(f"*locator*: {cell['locator']}")
            a("")
            if "recipe_verdict" in cell:
                a(f"*recipe verdict*: **{cell['recipe_verdict']}** -- {cell['recipe']}")
                a("")
            if cell.get("note"):
                a(cell["note"])
                a("")
        if src.get("caveats_stated_by_the_source"):
            a("Caveats the source states about itself:")
            for c in src["caveats_stated_by_the_source"]:
                a(f"* {c}")
            a("")
        if src.get("extra_observations"):
            a("Other observations:")
            for c in src["extra_observations"]:
                a(f"* {c}")
            a("")
        if src.get("not_retrieved"):
            a("NOT-RETRIEVED (version-level):")
            for nr in src["not_retrieved"]:
                a(f"* **{nr['what']}** -- {nr['url']} (HTTP {nr.get('http_status')}). "
                  f"Reason: {nr['reason']} Consequence: {nr['consequence']}")
            a("")
    a("## Our side (the anchor)")
    a("")
    a(f"* value: **{a_anchor_value(matrix)}** at `{matrix['anchor_our_side']['path']}` "
      f"jsonpath `{matrix['anchor_our_side']['jsonpath']}`")
    a(f"* definition: {matrix['anchor_our_side']['definition']}")
    a(f"* post-processing: {matrix['anchor_our_side']['post_processing']}")
    a(f"* hydrogen convention: {matrix['anchor_our_side']['hydrogen_convention']}")
    a(f"* provenance: {matrix['anchor_our_side']['measured_by']}. "
      f"{matrix['anchor_our_side']['cross_branch_provenance']}")
    a("")
    a("## Where the repo's band comes from")
    a("")
    bp = matrix["band_provenance"]
    a(f"* band as quoted in the repo: **{bp['band_as_quoted_in_the_repo']}**")
    a(f"* finding: {bp['headline_finding']}")
    a("")
    a("Where the repo states it (read-only for this branch):")
    for w in bp["where_the_repo_states_it"]:
        a(f"* {w}")
    a("")
    cr = bp.get("competing_reading_and_what_this_does_not_settle")
    if cr:
        a("### The band's source cells are CONTESTED on disk (reading-qualified)")
        a("")
        a(f"* status: {cr['status']}")
        a(f"* canonical statement: {cr['canonical_reading_safe_band_statement']}")
        a("")
        for r in cr.get("readings", []):
            a(f"* reading {r['reading']} -- locator: {r['locator']}")
        a("")
    for entry in bp["endpoint_candidates"]:
        a(f"### endpoint {entry['endpoint']}")
        a("")
        for key, cand in entry.items():
            if not key.startswith("candidate_"):
                continue
            # 2026-11-02 (Wf-A194) -- a DATED SIBLING of a candidate
            # (`candidate_a_what_it_is_superseded_2026_11_01` and its dated
            # note) holds a STRING, not a candidate object.  The loop
            # assumed every `candidate_*` key held a dict and raised
            # `TypeError: string indices must be integers, not 'str'`, so
            # the producer could not regenerate the record AT ALL -- the
            # published artefact was then OLDER than the producer that
            # claims to emit it.  DATED siblings are rendered below, after
            # the candidate they belong to.
            if not isinstance(cand, dict):
                continue
            a(f"* **{key}** -- {cand['paper']}")
            a(f"  * what it is: {cand['what_it_is']}")
            a(f"  * denominator: {cand['denominator']}")
            a(f"  * post-processed: {cand.get('post_processed')}")
            a(f"  * locator: {cand['locator']}")
            a(f"  * quote: > {cand['quote']}")
            a(f"  * status: {cand['status']}")
        # 2026-11-02 (Wf-A194) -- the DATED siblings of a candidate (a
        # superseded rendering and its dated note) are rendered too, so the
        # markdown carries the same dated history the JSON carries.  They
        # are DATED records, not live claims.
        for key, val in entry.items():
            if not key.startswith("candidate_") or isinstance(val, dict):
                continue
            a(f"  * **{key}** -- DATED, SUPERSEDED RENDERING (PRESERVED "
              f"VERBATIM, not a live claim): {val}")
        if entry.get("note"):
            a(f"* note: {entry['note']}")
        a("")
    ev = bp["post_processing_evidence_from_the_frontier_itself"]
    a("### The frontier publishes the post-processing axis itself")
    a("")
    a(f"* source: {ev['source']}")
    a(f"* recipe: {ev['recipe']}")
    a("")
    a("| model | raw (%) | post-processed (%) | delta (pp) |")
    a("|---|---|---|---|")
    for row in ev["raw_to_post_processed_pairs"]:
        a(f"| {row['model']} | {row['raw']} | {row['pp']} | {row['delta_pp']} |")
    a("")
    a(ev["note"])
    a("")
    rec = ev["recovery_fraction_our_side"]
    a(f"* the same recovery formula on our side: `{rec['formula']}`")
    a(f"  * raw (fraction) = {rec['raw']}; post-processed (fraction) = {rec['post_processed']}; "
      f"recovery fraction = **{rec['value']}** ({rec['units']})")
    a(f"* caveat: {rec['caveat']}")
    a("")
    a(f"* what this does not settle: {bp['what_this_does_not_settle']}")
    a("")
    a("## What this branch did NOT establish")
    a("")
    for c in matrix["what_this_branch_did_not_establish"]:
        a(f"* {c}")
    a("")
    a("## Retrieval log")
    a("")
    a("| url | retrieved | status | what was extracted |")
    a("|---|---|---|---|")
    for e in matrix["retrieval_log"]:
        a(f"| `{e['url']}` | {e['retrieved_at']} | {e.get('http_status')} | {e['extracted']} |")
    a("")
    return "\n".join(lines) + "\n"


def _json_bytes(matrix: Dict[str, Any]) -> str:
    """The exact bytes of the record artefact.

    ONE function, used by the writer, by the parity check and by the byte
    evidence, so the md5 the evidence quotes is the md5 of the file that
    is actually written.  Two copies of this expression would drift.
    """
    return json.dumps(matrix, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def render_markdown(matrix: Dict[str, Any]) -> str:
    """Render the markdown artefact: BODY + the three dated sections.

    The two dated correction sections are constants here -- they were
    EXTRACTED verbatim from the artefact on disk and never retyped -- and
    the third, the one that records this reconciliation, is rendered FROM
    the record's own reconciliation key, so the JSON and the markdown
    cannot drift apart.
    """
    return (
        render_markdown_body(matrix)
        + WF_A177_MD_SECTION
        + WF_A180_MD_SECTION
        + render_reconciliation_section(matrix)
        + render_locator_class_section(matrix)
        + render_wfa197_record_residuals_section(matrix)
    )


def render_reconciliation_section(matrix: Dict[str, Any]) -> str:
    """Render the ONE dated section that records this reconciliation.

    It is rendered from ``band_provenance``.
    ``producer_reproduces_the_record_2026_11_01`` -- the single source of
    truth for what changed, why, and the byte evidence -- so the markdown and
    the record always say the same thing.
    """
    recon = matrix["band_provenance"][RECONCILIATION_KEY]
    ev = recon["byte_evidence"]
    lines: List[str] = []
    a = lines.append
    a("")
    a(f"## {recon['heading']}")
    a("")
    a("This section is APPENDED.  Per AGENTS.md §5.5 rule #6 nothing above "
      "this line was deleted: the two dated sections above are PRESERVED "
      "VERBATIM, and every superseded rendering they quote is still the "
      "record of what the body said before this repair.")
    a("")
    a("**The body above was re-rendered by this repair.**  The statements in "
      "the dated sections above that nothing above them was edited, removed, "
      "re-rendered or re-ordered were TRUE at their date and are SUPERSEDED "
      "by this section: the body was re-rendered from the record, which "
      "(a) replaced the three contested renderings with the reading-safe "
      "ones, (b) aligned the source order with the JSON's source order, and "
      "(c) added the band's contested-source section.  The superseded "
      "renderings themselves are preserved verbatim -- the three claim "
      "renderings quoted in the Wf-A177 and Wf-A180 sections above, and the "
      "eight live strings this repair changed, in the record's dated key "
      f"`band_provenance`.`{RECONCILIATION_KEY}`.")
    a("")
    a(f"**What changed.**  {recon['what_changed']}")
    a("")
    a(f"**Why.**  {recon['why']}")
    a("")
    a("**What changed in the record.**")
    a("")
    for item in recon["changed_in_this_record"]:
        a(f"* {item}")
    a("")
    a("**The superseded renderings (PRESERVED VERBATIM).**")
    a("")
    for item in recon["superseded_renderings"]:
        a(f"* `{item['path']}` -- superseded:")
        a("")
        a(f"> {item['rendering']}")
        a("")
        a(f"  {item['note']}")
        a("")
    a("**The byte evidence.**")
    a("")
    a(f"* markdown BODY (everything before the three dated sections): "
      f"{ev['md_body_chars']} characters")
    a(f"* the Wf-A177 dated section: {ev['wf_a177_md_section_chars']} characters")
    a(f"* the Wf-A180 dated section: {ev['wf_a180_md_section_chars']} characters")
    a(f"* the record carries {ev['record_leaf_count']} leaves, of which "
      f"{ev['record_numeric_leaf_count']} are numeric")
    a(f"* numeric leaves moved by this repair: "
      f"{ev['numeric_leaves_moved_by_this_repair']} -- the repair is "
      "TEXT-ONLY plus NEW dated keys, which is the publish gate for this "
      "repair.  The parity check measures it against the artefact on disk "
      "and prints it as 'numeric leaves moved'.")
    json_bytes = _json_bytes(matrix)
    a(f"* the record JSON on disk: {len(json_bytes.encode('utf-8'))} bytes, "
      f"md5 `{hashlib.md5(json_bytes.encode('utf-8')).hexdigest()}`")
    a("* the markdown's OWN size and md5 are NOT quoted here: a file cannot "
      "contain its own hash.  They are recorded in the dated sibling file "
      f"`{PARITY_EVIDENCE_FILENAME}`.")
    a("")
    a(f"**What this section does NOT change.**  {recon['what_it_does_not_change']}")
    a("")
    a(f"**The producer now reproduces this artefact.**  "
      f"{recon['producer_now_reproduces_the_artefact']}")
    return "\n".join(lines) + "\n"


def render_locator_class_section(matrix: Dict[str, Any]) -> str:
    """Render the dated section that records the locator-class closure and
    the GeoLDM source entry.

    It is rendered FROM ``band_provenance``.
    ``locator_class_and_geo_ldm_source_2026_11_02`` -- the single source of
    truth -- so the markdown and the record always say the same thing.
    """
    lc = matrix["band_provenance"][LOCATOR_CLASS_KEY]
    gl = lc["the_geo_ldm_entry"]
    drift = lc["the_drift_measured_on_2026_11_02"]
    lines: List[str] = []
    a = lines.append
    a("")
    a(f"## {lc['heading']}")
    a("")
    a("This section is APPENDED.  Per AGENTS.md §5.5 rule #6 nothing above "
      "this line was deleted, edited, removed, re-rendered or re-ordered: "
      "the dated sections above are PRESERVED VERBATIM, and every "
      "superseded rendering they quote is still the record of what the body "
      "said before this repair.")
    a("")
    a(f"**{lc['what_this_is']}**")
    a("")
    a(f"**The locator-class rule.**  {lc['the_locator_class_rule']}")
    a("")
    a("**The demoted line numbers are PRESERVED VERBATIM in:**")
    a("")
    for item in lc["the_demoted_line_numbers_are_preserved_verbatim_in"]:
        a(f"* {item}")
    a("")
    a("**The drift, measured on 2026-11-02.**")
    a("")
    a(drift["what_this_is"])
    a("")
    a("| cited position | old claim | actual position (measured 2026-11-02) | verdict |")
    a("|---|---|---|---|")
    for row in drift["rows"]:
        a(f"| {row['cited_position']} | {row['old_claim']} | "
          f"{row['actual_position_measured_2026_11_02']} | {row['verdict']} |")
    a("")
    a(f"*{drift['summary']}*")
    a("")
    a("**The GeoLDM source entry.**")
    a("")
    a(f"* what it is: {gl['what_it_is']}")
    a(f"* locator class: {gl['locator_class']}")
    a(f"* F1: {gl['the_f1_fact']}")
    a(f"* F2: {gl['the_f2_fact']}")
    a(f"* F3: {gl['the_f3_fact_and_why_it_matters_here']}")
    a(f"* F4: {gl['the_f4_disagreement']}")
    a(f"* F5: {gl['the_f5_fact']}")
    a(f"* why the band is still not settled: {gl['why_the_band_is_not_settled_by_it']}")
    a("")
    a("**The retrieval receipt for arXiv:2305.01140v1.**")
    a("")
    r = gl["the_retrieval_receipt"]
    a(f"* e-print url: {r['eprint_url']}")
    a(f"* e-print sha256: `{r['eprint_sha256']}`")
    a(f"* e-print bytes: {r['eprint_bytes']}")
    a(f"* retrieved (UTC): {r['retrieved_utc']}")
    a(f"* title locator: {r['title_locator']}")
    a(f"* source file: `{r['source_file']}` sha256 `{r['source_file_sha256']}`")
    a(f"* source of the receipt: `{r['source_of_the_receipt']}`")
    a("")
    a(f"**{lc['the_md_section_is_appended']}**")
    a("")
    a(f"**What this repair does NOT change.**  {lc['what_this_repair_does_not_change']}")
    return "\n".join(lines) + "\n"


def render_wfa197_record_residuals_section(matrix: Dict[str, Any]) -> str:
    """Render the dated section that closes the SIX record residuals.

    Rendered FROM ``band_provenance``.``record_residuals_closed_2026_11_02``
    -- the single source of truth -- so the markdown and the record always
    say the same thing.  It is APPENDED LAST: it supersedes the false
    "nothing above this line was re-rendered" assertion of the dated
    2026-11-02 section, which is left VERBATIM above it.
    """
    rr = matrix["band_provenance"][WFA197_RESIDUALS_KEY]
    lines: List[str] = []
    a = lines.append
    a("")
    a(f"## {rr['heading']}")
    a("")
    a(rr["the_append_assertion"])
    a("")
    a(f"**{rr['what_this_is']}**")
    a("")
    for res in rr["the_six_residuals"]:
        a(f"### {res['residual']} -- {res['short_title']}")
        a("")
        a(f"* **the defect:** {res['defect']}")
        a(f"* **the repair:** {res['repair']}")
        a(f"* **the superseded rendering is preserved VERBATIM at:** "
          f"{res['superseded_rendering_preserved_verbatim_at']}")
        a("")
    a("**The passages that were re-rendered** (measured by diffing the "
      "pre-Wf-A194 published bytes of this record against the published bytes).")
    a("")
    for item in rr["the_re_rendered_passages"]:
        a(f"* {item}")
    a("")
    a("**What this repair did NOT do.**")
    a("")
    for item in rr["what_this_repair_did_not_do"]:
        a(f"* {item}")
    return "\n".join(lines) + "\n"


def a_anchor_value(matrix: Dict[str, Any]) -> str:
    anc = matrix["anchor_our_side"]
    return f"{anc['value_display']} ({anc['value']})"


def render_parity_evidence(matrix: Dict[str, Any]) -> str:
    """Render the DATED SIBLING FILE: the byte evidence for BOTH artefacts.

    A file cannot contain its own hash, so the markdown's own size and md5
    cannot live inside the markdown.  They live here, next to the record's
    own size and md5.

    It is rendered from the SAME matrix as the two artefacts, so it cannot
    drift away from them: if the record JSON and the markdown are
    byte-identical to what the producer emits, this file necessarily is
    too.
    """
    recon = matrix["band_provenance"][RECONCILIATION_KEY]
    json_bytes = _json_bytes(matrix)
    md_bytes = render_markdown(matrix)
    leaves, numeric = _leaf_stats(matrix)
    payload = {
        "dated": recon["dated"],
        "heading": (
            "Producer/artefact parity -- byte evidence for both artefacts "
            "(2026-11-01)"
        ),
        "reconciliation_key": RECONCILIATION_KEY,
        "what_this_is": (
            "The dated byte evidence for the producer/artefact parity "
            "reconciliation.  It is written BY the producer, from the same "
            "matrix as the two artefacts, so it cannot drift away from "
            "them.  The markdown's own size and md5 live HERE rather than "
            "inside the markdown because a file cannot contain its own hash."
        ),
        "artifacts": {
            "frontier_matrix.json": {
                "path": (
                    "molmetal/reports/wfa158_frontier_postproc_matrix/"
                    "frontier_matrix.json"
                ),
                "bytes": len(json_bytes.encode("utf-8")),
                "md5": hashlib.md5(json_bytes.encode("utf-8")).hexdigest(),
            },
            "frontier_matrix.md": {
                "path": (
                    "molmetal/reports/wfa158_frontier_postproc_matrix/"
                    "frontier_matrix.md"
                ),
                "bytes": len(md_bytes.encode("utf-8")),
                "md5": hashlib.md5(md_bytes.encode("utf-8")).hexdigest(),
            },
        },
        "leaf_counts": {
            "total": leaves,
            "numeric": numeric,
        },
        "numeric_leaves_moved_by_this_repair": 0,
        "how_to_re_verify": (
            "Run the producer with `--check-only --require-parity` and read "
            "the PARITY report it prints: the two artefacts must be "
            "byte-identical to the producer's output, with 0 numeric leaves "
            "moved and 0 leaves on only one side."
        ),
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Leaf census, byte evidence, and the PARITY check
# ---------------------------------------------------------------------------


def _leaf_paths(obj: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """Yield ``(jsonpath, leaf)`` for every leaf of ``obj``.

    The jsonpath is the same slash-separated form the record uses, so a leaf
    can be named unambiguously in a report.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _leaf_paths(value, f"{prefix}/{key}")
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from _leaf_paths(value, f"{prefix}/{idx}")
    else:
        yield prefix, obj


def _leaf_stats(obj: Any) -> Tuple[int, int]:
    """Return ``(leaves, numeric_leaves)`` of ``obj``."""
    total = 0
    numeric = 0
    for _path, value in _leaf_paths(obj):
        total += 1
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric += 1
    return total, numeric


def _fill_byte_evidence(matrix: Dict[str, Any]) -> None:
    """Fill the byte evidence of the reconciliation key.

    Every number here is MEASURED from the artefact this producer emits.

    It deliberately does NOT quote the markdown's TOTAL length: this key is
    part of that markdown, so a total length quoted inside it could never be
    true of the file that carries it -- a self-referential number would be a
    fabrication, not a measurement.

    ``numeric_leaves_moved_by_this_repair`` is 0 because every leaf this
    repair adds or changes is a STRING.  It is MEASURED at run time by the
    parity check (``numeric leaves moved``) and by the coupled test.
    """
    evidence = matrix["band_provenance"][RECONCILIATION_KEY]["byte_evidence"]
    leaves, numeric = _leaf_stats(matrix)
    evidence["record_leaf_count"] = leaves
    evidence["record_numeric_leaf_count"] = numeric
    evidence["md_body_chars"] = len(render_markdown_body(matrix))
    evidence["wf_a177_md_section_chars"] = len(WF_A177_MD_SECTION)
    evidence["wf_a180_md_section_chars"] = len(WF_A180_MD_SECTION)


def parity_report(matrix: Dict[str, Any], art_dir: Path) -> Dict[str, Any]:
    """Compare the producer's own output against the artefacts on disk.

    This is the early warning the next agent needs.  Before this check
    existed, the producer and its artefact had DIVERGED and a re-run would
    have silently destroyed the append-only history -- and nothing compared
    the two, so the divergence survived two waves.

    The report carries the numbers the publish gate is stated in: how many
    NUMERIC leaves moved (the gate is 0), how many string leaves moved, and
    how many leaves exist on only one side (0 on both sides -- a leaf that
    exists on only one side would be DELETED or ADDED by a re-run).
    """
    produced_json = _json_bytes(matrix)
    produced_md = render_markdown(matrix)
    json_path = art_dir / "frontier_matrix.json"
    md_path = art_dir / "frontier_matrix.md"
    report: Dict[str, Any] = {
        "art_dir": str(art_dir),
        "json_present": json_path.is_file(),
        "md_present": md_path.is_file(),
        "json_byte_identical": False,
        "md_byte_identical": False,
        "numeric_leaves_moved": None,
        "string_leaves_moved": None,
        "leaves_only_on_disk": None,
        "leaves_only_in_producer": None,
        "ok": False,
    }
    if report["json_present"]:
        disk_text = json_path.read_text(encoding="utf-8")
        report["json_byte_identical"] = disk_text == produced_json
        try:
            disk_obj = json.loads(disk_text)
        except json.JSONDecodeError:
            disk_obj = None
        if disk_obj is not None:
            disk_leaves = dict(_leaf_paths(disk_obj))
            prod_leaves = dict(_leaf_paths(matrix))
            moved_num = 0
            moved_str = 0
            only_disk = 0
            for path, value in disk_leaves.items():
                if path not in prod_leaves:
                    only_disk += 1
                    continue
                if prod_leaves[path] == value:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    moved_num += 1
                else:
                    moved_str += 1
            report["numeric_leaves_moved"] = moved_num
            report["string_leaves_moved"] = moved_str
            report["leaves_only_on_disk"] = only_disk
            report["leaves_only_in_producer"] = sum(
                1 for path in prod_leaves if path not in disk_leaves
            )
    if report["md_present"]:
        report["md_byte_identical"] = (
            md_path.read_text(encoding="utf-8") == produced_md
        )
    report["ok"] = bool(
        report["json_byte_identical"] and report["md_byte_identical"]
    )
    return report


def _print_parity(report: Dict[str, Any]) -> None:
    """Print the parity report in the CLI's style."""
    art = report["art_dir"]
    if not report["json_present"] and not report["md_present"]:
        print(f"  parity: no artefact under {art} -- nothing to compare")
        return
    for kind, label in (("json", "record json"), ("md", "markdown")):
        if not report[f"{kind}_present"]:
            print(f"  parity ({label}): SKIP -- no artefact under {art}")
            continue
        state = "OK" if report[f"{kind}_byte_identical"] else "MISMATCH"
        print(f"  parity ({label}): {state} -- {art}")
    if report["numeric_leaves_moved"] is not None:
        print(f"    numeric leaves moved: {report['numeric_leaves_moved']} "
              "(the publish gate is 0)")
        print(f"    string leaves moved: {report['string_leaves_moved']}")
        print(f"    leaves only on disk: {report['leaves_only_on_disk']} "
              "(0 -- nothing may be dropped)")
        print(f"    leaves only in the producer: "
              f"{report['leaves_only_in_producer']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="where the matrix artifact is written")
    parser.add_argument("--check-only", action="store_true",
                        help="validate the in-memory matrix and exit without writing")
    parser.add_argument("--parity-dir", default=str(DEFAULT_OUT_DIR), metavar="DIR",
                        help="the directory whose artefacts the PARITY check compares "
                             "the producer's output against (default: the artefact "
                             "directory).  The report is printed by --check-only.")
    parser.add_argument("--require-parity", action="store_true",
                        help="fail (exit 1) when the producer's output does not "
                             "reproduce the artefacts in --parity-dir byte-for-byte. "
                             "Off by default so that --check-only keeps reporting "
                             "the parity of an artefact that has not been "
                             "regenerated yet.")
    parser.add_argument("--allow-leaf-drop", action="store_true",
                        help="permit a write that would drop leaves the artefact "
                             "carries and this producer does not emit (refused by "
                             "default: AGENTS.md §5.5 rule #6)")
    parser.add_argument("--verify-quotes-from", default=None, metavar="DIR",
                        help="re-check every embedded quote against locally saved "
                             "extractions of the retrieved pages (see RETRIEVAL_LOG "
                             "for the file names); exits 2 if the directory is absent")
    args = parser.parse_args(argv)

    matrix = build_matrix()
    problems = validate_matrix(matrix)

    skip_reason: Optional[str] = None
    if args.verify_quotes_from is not None:
        vdir = Path(args.verify_quotes_from)
        if not vdir.is_dir():
            skip_reason = f"verify directory not found: {vdir}"
            print(f"SKIP: {skip_reason}")
            print("SKIP")
            return 2
        qproblems = verify_quotes(vdir)
        if qproblems:
            problems.extend(qproblems)
        else:
            print(f"  quote re-verification: {quote_count()}/{quote_count()} quotes found in {vdir}")

    if problems:
        print("Wf-A158 frontier post-processing comparability matrix -- FAIL")
        for p in problems:
            print(f"  PROBLEM: {p}")
        print("FAIL")
        return 1

    n_cells = sum(len(s["cells"]) for s in matrix["sources"].values())
    n_quotes = sum(
        1
        for s in matrix["sources"].values()
        for c in s["cells"].values()
        if c.get("quote")
    )
    print("Wf-A158 frontier post-processing comparability matrix")
    print(f"  retrieval date: {matrix['retrieval_date']}")
    print(f"  sources: {len(matrix['sources'])}  cells: {n_cells}  cells carrying a quote: {n_quotes}")
    for name, src in matrix["sources"].items():
        verdicts = " ".join(f"{b}={src['cells'][b]['verdict']}" for b in BREAKS)
        print(f"  {name}: {verdicts}")
    print(f"  anchor (ours): {a_anchor_value(matrix)} -- RAW, no post-processing")

    # The PARITY report: does this producer still reproduce its artefact?
    # It is printed ALWAYS (not only in --check-only) so a regeneration
    # cannot quietly destroy the history it cannot reproduce.
    parity = parity_report(matrix, Path(args.parity_dir))
    _print_parity(parity)

    if args.check_only:
        if args.require_parity and not parity["ok"]:
            print("FAIL (check-only: the producer does not reproduce the artefact)")
            return 1
        print("PASS (check-only: nothing written)")
        return 0

    out_dir = Path(args.out_dir)
    json_path = out_dir / "frontier_matrix.json"
    md_path = out_dir / "frontier_matrix.md"
    # Refuse to overwrite an artefact that carries leaves this producer does
    # NOT emit: a re-run would DELETE them, and AGENTS.md §5.5 rule #6 says
    # nothing is deleted silently.  This is the guard that would have caught
    # the divergence this repair fixes.
    if json_path.is_file() and not args.allow_leaf_drop:
        existing = parity_report(matrix, out_dir)
        dropped = existing["leaves_only_on_disk"]
        if dropped:
            print(f"REFUSING to write: {dropped} leaves of {json_path} are not "
                  "emitted by this producer and would be DELETED by this run.  "
                  "(--allow-leaf-drop overrides.)")
            return 1
    atomic_write_file(json_path, _json_bytes(matrix))
    atomic_write_file(md_path, render_markdown(matrix))
    atomic_write_file(out_dir / PARITY_EVIDENCE_FILENAME, render_parity_evidence(matrix))
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")
    print(f"wrote: {out_dir / PARITY_EVIDENCE_FILENAME}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
