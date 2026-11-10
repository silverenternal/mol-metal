"""Wf-A175 BAND branch -- build sites.json (one record per band match).

Classification is HAND-CURATED (see CLASS below) from the sentence context
read on disk; the census (paths/lines/matched text) is machine-derived.
CPU-only.  Writes ONLY into this directory.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/hugo/codes/mol-metal")
OUT = ROOT / "molmetal" / "reports" / "wfa175_band_provenance"

# (assertion, raw_or_post, denominator_stated, severity, note)
Q = "quoted"
U = "unstated"
R = "raw"
P = "post"
N = "none"
L = "low"
H = "high"
T = True
F = False

CLASS: dict[tuple[str, int], tuple] = {
    # ---------------- governance / narrative pointers ----------------
    ("AGENTS.md", 63): (Q, U, F, N, "states the band is NOT our band"),
    ("AGENTS.md", 172): (Q, U, F, N, "states the band is NOT our band"),
    ("AGENTS.md", 372): (Q, U, F, N, "states the band is NOT our band"),
    ("TODO/INDEX.md", 3): (Q, U, F, N, "states the band is NOT our band"),
    ("TODO/README.md", 3): (Q, U, F, N, "states the band is NOT our band"),
    # ---------------- HONEST_NEGATIVE_CATALOG ----------------
    ("docs/HONEST_NEGATIVE_CATALOG.md", 20288): (Q, U, F, N, "S6.264 heading: band is NOT our band"),
    ("docs/HONEST_NEGATIVE_CATALOG.md", 20487): (Q, U, F, N, "S6.264: band is NOT our band"),
    ("docs/HONEST_NEGATIVE_CATALOG.md", 20517): (Q, U, F, N, "S6.264 conclusion: band is not our band"),
    ("docs/HONEST_NEGATIVE_CATALOG.md", 20808): (Q, U, F, N, "S6.264 S4: band is NOT our band"),
    ("docs/HONEST_NEGATIVE_CATALOG.md", 20817): (Q, P, T, H,
        "DENOMINATOR row of the S6.264 four-break table; the adjacent POST-PROCESSING row (L20819) asserts the band is POST-PROCESSED -- REFUTED: both band endpoints carry post_processed=false in frontier_matrix.json /band_provenance/endpoint_candidates and Wf-A168 measured them RAW"),
    ("docs/HONEST_NEGATIVE_CATALOG.md", 20823): (Q, U, F, N, "rule statement: band must not sit beside our rates unqualified"),
    ("docs/HONEST_NEGATIVE_CATALOG.md", 21273): (Q, U, T, N, "S6.266.8: band is a composite of 92.98/94.33/93.13 with three denominators"),
    ("docs/HONEST_NEGATIVE_CATALOG.md", 21281): (Q, U, F, N, "S6.266.8: band is NOT our band and appears nowhere in paper/"),
    # ---------------- published metric artefacts ----------------
    ("metrics/by_metric/wfa155_ligand_connected_rate.json", 224): (Q, U, F, L,
        "comparability.statement; RAW/POST + denominator live on DIFFERENT lines (237-238, 228) -- a line-level census cannot see them"),
    ("metrics/by_metric/wfa155_ligand_connected_rate.json", 228): (Q, P, T, H,
        "DENOMINATOR break 'frontier' cell; the block's POST-PROCESSING break (L237-238) asserts POST-PROCESSED -- REFUTED (see note on catalog L20817)"),
    ("metrics/by_metric/wfa155_ligand_connected_rate.json", 248): (Q, U, F, L,
        "bare machine field 'frontier_band'; no RAW/POST, no denominator on this line"),
    ("metrics/by_metric/wfa155_metal_coordination_sphere.json", 22): (Q, U, F, L,
        "what_this_is_not.not_a_frontier_comparison; breaks live on a different line (L26-29)"),
    ("metrics/by_metric/wfa155_metal_coordination_sphere.json", 26): (Q, P, T, H,
        "comparability_statement.summary; break_3_post_processing (L29) asserts POST-PROCESSED -- REFUTED; artefact is STALE vs its script (Wf-A171 S6)"),
    ("metrics/by_metric/wfa155_posebusters_validity.json", 608): (Q, P, F, H,
        "comparability_vs_frontier_band.statement; break 3 (L595-596) asserts POST-PROCESSED -- REFUTED"),
    # ---------------- production code ----------------
    ("molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py", 271): (Q, U, T, L,
        "module comment: band is NOT comparable to our 8.51 %; names EF-TALFM's 92.98/98.25/98.21/94.33 over 10,000 molecules but does not say whether the BAND endpoints are raw or post-processed"),
    ("molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py", 947): (Q, U, T, L,
        "opt-in docstring: band + our 8.51 % + four breaks; band RAW/POST unstated"),
    ("molmetal/scripts/wf_a155_ligand_connected_rate.py", 7): (Q, U, F, L,
        "module docstring: frontier connectivity band; the POST-PROCESSING break at L220-224 asserts POST-PROCESSED (REFUTED)"),
    ("molmetal/scripts/wf_a155_ligand_connected_rate.py", 91): (Q, U, F, L,
        "docstring: not comparable to the frontier band; four breaks named but not enumerated here"),
    ("molmetal/scripts/wf_a155_ligand_connected_rate.py", 156): (Q, U, T, N,
        "machine constant FRONTIER_BAND; adjacent comment says 'the FRONTIER's numbers, not ours' and L157 carries the denominator"),
    ("molmetal/scripts/wf_a155_metal_coordination_sphere.py", 104): (Q, U, F, N,
        "changelog: band is labelled a FRONTIER CONSTANT and cited to its primary record"),
    ("molmetal/scripts/wf_a155_metal_coordination_sphere.py", 305): (Q, U, T, N,
        "machine constant FRONTIER_BAND under the header 'never one of our numbers'; L309 carries the denominator"),
    ("molmetal/scripts/wf_a155_metal_coordination_sphere.py", 380): (Q, U, T, N,
        "counter-quote: EF-TALFM's 87.9 %, not 92.7-94 %"),
    ("molmetal/scripts/wfa156_bond_threshold_frontier.py", 325): (Q, U, F, N,
        "machine constant FRONTIER_BAND; the comment above labels it the branch's RESTATEMENT, not the paper's wording"),
    ("molmetal/scripts/wfa156_bond_threshold_frontier.py", 402): (Q, U, F, L,
        "calls it 'the 92.7-94 % frontier band' with no restatement label (already logged as a Low defect by Wf-A164 refutation item 5)"),
    ("molmetal/scripts/wfa158_frontier_postproc_matrix.py", 8): (Q, U, F, L,
        "module docstring: 'a published frontier band'"),
    ("molmetal/scripts/wfa158_frontier_postproc_matrix.py", 448): (Q, U, F, N,
        "comment header: 'Where the repo's \"92.7-94 %\" band actually comes from' -- provenance question, not an ownership claim"),
    ("molmetal/scripts/wfa158_frontier_postproc_matrix.py", 456): (Q, U, F, L,
        "machine field band_as_quoted_in_the_repo. OWNERSHIP AMBIGUITY: repo-as-restater, not repo-as-measurer"),
    ("molmetal/scripts/wfa158_frontier_postproc_matrix.py", 717): (Q, R, F, N,
        "names 92.98 as RAW connectivity vs 94.33 as post-processed -- the correct per-cell framing"),
    ("molmetal/scripts/wfa158_frontier_postproc_matrix.py", 749): (Q, R, F, N,
        "per-cell framing: mixes the LOWEST raw connectivity with a post-processed companion"),
    ("molmetal/scripts/wfa158_frontier_postproc_matrix.py", 754): (Q, R, F, N,
        "per-cell framing"),
    ("molmetal/scripts/wfa158_population_reconciliation.py", 215): (Q, U, T, N,
        "F12_external_band: 'PUBLISHED FRONTIER BAND (external literature -- not our metric)'; denominator given"),
    ("molmetal/scripts/wfa158_population_reconciliation.py", 1060): (Q, U, F, N,
        "F12_external_band value: '92.7 - 94 % (a band, not a point value)', jsonpath = external literature"),
    ("molmetal/scripts/wfa158_population_reconciliation.py", 1798): (Q, U, F, N,
        "'the 92.7-94 % band is an EXTERNAL number; this repo publishes no measurement of it'"),
    ("molmetal/scripts/wfa161_frontier_labelling.py", 7): (Q, U, F, N,
        "module docstring: a published band that must be labelled identically for seven populations"),
    ("molmetal/scripts/wfa161_frontier_labelling.py", 934): ("ours", P, F, H,
        "LOAD-BEARING DEFECT: 'Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing' -- (a) asserts the band as OURS (every other site calls it EXTERNAL); (b) the Open Babel provenance is REFUTED by Wf-A168 S1.3 (92.7 % = HLTF arXiv:2603.27113v1 Table 1(b) 'Val&Uniq (%)' GeoLDM cell; 94 % = the same table's 'PB-Valid (%)' HLTF cell); (c) 'DIFFERENT post-processing' is REFUTED -- both endpoints carry post_processed=false in frontier_matrix.json"),
    ("molmetal/scripts/wfa161_paper_unqualified_numbers.py", 987): (Q, U, F, N,
        "scanner seed table: describes the unqualified-band pattern it hunts"),
    ("molmetal/scripts/wfa161_paper_unqualified_numbers.py", 990): (Q, U, F, N,
        "scanner seed table: the band occurs 0 times in paper/**"),
    ("molmetal/scripts/wfa163_narrative_truth_scan.py", 534): ("unattributed", U, F, N,
        "docstring example of a range token; no claim asserted"),
    # ---------------- tests ----------------
    ("molmetal/tests/test_wfa155_decoder_bond_candidacy.py", 37): (Q, U, F, N,
        "test fixture sentence mirroring the decoder docstring"),
    ("molmetal/tests/test_wfa155_ligand_connected_rate.py", 374): (Q, U, F, N,
        "asserts the artefact's comparability.frontier_band literal"),
    ("molmetal/tests/test_wfa155_ligand_connected_rate.py", 375): (Q, U, F, N,
        "asserts the artefact's comparability.statement literal"),
    ("molmetal/tests/test_wfa158_frontier_postproc_matrix.py", 466): (Q, U, F, N,
        "test comment: band provenance must be traceable"),
    ("molmetal/tests/test_wfa164_coordination_sphere_narrative.py", 116): (Q, U, F, N,
        "expected narrative literal mirroring the artefact"),
    ("molmetal/tests/test_wfa164_coordination_sphere_narrative.py", 357): (Q, U, F, N,
        "asserts band_as_quoted_in_the_repo literal"),
    # ---------------- prior-wave reports ----------------
    ("molmetal/reports/wfa156_bond_threshold_frontier/report.md", 44): (Q, U, F, L,
        "calls it 'the 92.7-94 % frontier band' with no restatement label (Wf-A164 refutation item 5)"),
    ("molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md", 52): (Q, R, T, N,
        "names 92.98 (raw connectivity) and 94.33 (post-processed N.&U.&S.) -- correct per-cell framing"),
    ("molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md", 68): (Q, P, F, N,
        "names 94.33 as post-processed; correct per-cell"),
    ("molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md", 69): (Q, R, F, N,
        "correct per-cell framing"),
    ("molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md", 353): (Q, U, F, L,
        "table row 'band as quoted in the repo'. OWNERSHIP AMBIGUITY: repo-as-restater, not repo-as-measurer"),
    ("molmetal/reports/wfa158_population_reconciliation/census.md", 34): (Q, U, T, N,
        "F12 row: 'PUBLISHED FRONTIER BAND (external literature -- not our metric)'"),
    ("molmetal/reports/wfa161_frontier_labelling/report.md", 139): ("ours", P, F, H,
        "GENERATED TWICE from wfa161_frontier_labelling.py:934 -- same defect, second surface"),
    ("molmetal/reports/wfa161_paper_unqualified_numbers/report.md", 35): (Q, U, F, N,
        "scanner seed table (2 occurrences on one line); states the band occurs 0 times in paper/**"),
    ("molmetal/reports/wfa164_bond_threshold_narrative/c7_pool_check/report.md", 44): (Q, U, F, L,
        "generated from wfa156_bond_threshold_frontier.py:402 -- same unlabelled-restatement wording"),
    ("molmetal/reports/wfa164_bond_threshold_narrative/report.md", 44): (Q, U, F, L,
        "generated from wfa156_bond_threshold_frontier.py:402 -- same unlabelled-restatement wording"),
    ("molmetal/reports/wfa164_c_refutation/refutation.md", 69): (Q, U, F, N,
        "refutation item 5: names the unlabelled-restatement defect"),
    ("molmetal/reports/wfa164_c_refutation/refutation.md", 70): (Q, U, F, N,
        "refutation item 5 continued"),
    ("molmetal/reports/wfa164_coordination_sphere_refutation/refutation.md", 20): (Q, U, T, N,
        "refutation: 'the labelled frontier constant'"),
    ("molmetal/reports/wfa164_coordination_sphere_refutation/refutation.md", 61): (Q, R, T, N,
        "refutation D3: QM9 Table 1(b) is raw-only -- the band has ONE arm"),
    ("molmetal/reports/wfa164_posebusters_narrative/pre_repair_script_snapshot.py.txt", 1132): (Q, U, F, L,
        "pre-repair snapshot of a script; superseded text"),
    ("molmetal/reports/wfa164_synthesis/synthesis.md", 120): (Q, U, F, N,
        "defect D3 record: the band has one arm"),
    ("molmetal/reports/wfa164_synthesis/synthesis.md", 127): (Q, U, F, N,
        "defect C record: FRONTIER_BAND left in place"),
    ("molmetal/reports/wfa164_synthesis/synthesis.md", 151): (Q, U, F, N,
        "'FRONTIER constants' list -- explicitly external"),
    ("molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md", 20): (Q, R, F, L,
        "correct per-cell framing: the band is cell-heterogeneous in post-processing. OWNERSHIP AMBIGUITY: calls it 'the repo's published band', in tension with the 'NOT our band' framing at catalog S6.264"),
    ("molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md", 115): (Q, R, T, L,
        "pins both endpoints to HLTF Table 1(b), RAW. OWNERSHIP AMBIGUITY: 'the repo's published band'"),
    ("molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md", 284): (Q, U, F, N,
        "section heading"),
    ("molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md", 288): (Q, R, T, N,
        "'MAY say' item 1: names the endpoint, metric and denominator"),
    ("molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md", 307): (Q, U, F, N,
        "'MAY NOT say' item 1: never write the band beside our rates without the four breaks"),
    ("molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md", 308): (Q, U, F, N,
        "quotes the rule verbatim"),
    ("molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md", 407): (Q, U, F, N,
        "paper-side work needs user OK (TODO-84)"),
    ("molmetal/reports/wfa171_band_publication_audit/report.md", 1): (Q, U, F, N,
        "audit title"),
    ("molmetal/reports/wfa171_band_publication_audit/report.md", 5): (Q, U, F, N,
        "audit headline: no band occurrence reaches a published metric VALUE"),
    ("molmetal/reports/wfa171_band_publication_audit/report.md", 162): (Q, U, F, N,
        "negative-control description"),
    ("molmetal/reports/wfa171_band_publication_audit/report.md", 171): (Q, U, F, N,
        "quotes the OPEN-1 defect sentence from wfa161_frontier_labelling.py:934"),
    ("molmetal/reports/wfa171_frontier_postproc/report.md", 164): (Q, U, F, N,
        "section heading asking whether the band has a named method and denominator"),
    ("molmetal/reports/wfa171_frontier_postproc/report.md", 179): (Q, U, T, N,
        "the six source numbers with three denominators"),
    ("molmetal/reports/wfa171_frontier_postproc/report.md", 314): (Q, U, F, L,
        "'the band's provenance inside our own repo was not re-censused'. OWNERSHIP AMBIGUITY: repo-as-publisher"),
}


def main() -> int:
    occ = json.loads((OUT / "_occurrences_v1_dash_any.json").read_text(encoding="utf-8"))
    seen = set()
    records = []
    missing = []
    for o in occ:
        key = (o["path"], o["line"])
        seen.add(key)
        c = CLASS.get(key)
        if c is None:
            missing.append(key)
            continue
        assertion, raw_post, denom, sev, note = c
        records.append(
            {
                "path": o["path"],
                "line": o["line"],
                "matched_text": o["matched_text"],
                "full_sentence": o["full_sentence"],
                "assertion": assertion,
                "raw_or_post": raw_post,
                "denominator_stated": denom,
                "severity": sev,
                "ownership_ambiguity": "OWNERSHIP AMBIGUITY" in note,
                "note": note,
                "line_text": o["line_text"],
            }
        )
    if missing:
        raise SystemExit(f"UNCLASSIFIED sites ({len(missing)}): {missing}")
    unused = sorted(set(CLASS) - seen)
    if unused:
        raise SystemExit(f"CLASS entries with no census match: {unused}")

    (OUT / "sites.json").write_text(
        json.dumps(
            {
                "branch": "BAND",
                "wave": "Wf-A175",
                "generated": "2026-11-01",
                "census_regex": r"92\.7\s*[-‐‑‒–—―~]\s*94",
                "population": "git ls-files --cached + git ls-files --others --exclude-standard",
                "n_sites": len(records),
                "sites": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from collections import Counter

    print("total sites:", len(records))
    print("assertion:", Counter(r["assertion"] for r in records))
    print("raw_or_post:", Counter(r["raw_or_post"] for r in records))
    print("denominator_stated:", Counter(r["denominator_stated"] for r in records))
    print("severity:", Counter(r["severity"] for r in records))
    print("files:", len({r["path"] for r in records}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
