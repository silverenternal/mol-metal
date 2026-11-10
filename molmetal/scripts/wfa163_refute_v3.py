"""Write the Wf-A163 adversarial-refutation verdict artefacts.

Atomic writes via molmetal.scripts.wf386a3_atomic_write.atomic_write_file.
CPU-only; no torch, no CUDA, no network.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/hugo/codes/mol-metal")
sys.path.insert(0, str(ROOT))
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

OUT = ROOT / "molmetal/reports/wfa163_refute_v3"
OUT.mkdir(parents=True, exist_ok=True)

ENTRIES = [
    {
        "id": "A",
        "claim": "The re-derivation shares NO code with the tool under test "
                 "(molmetal/scripts/wf_a155_ligand_connected_rate.py): own CSV "
                 "reader, own connected-components, own metal predicate, own "
                 "McNemar.",
        "verdict": "NOT REFUTED (CONFIRMED)",
        "evidence": "Own AST parse of both files (my script /tmp/wfa163_refute/checks8.py): "
                    "branch imports = {argparse, ast, collections, csv, fractions, json, math, "
                    "os, pathlib, random, rdkit, scipy.stats, typing, __future__, "
                    "molmetal.scripts.wf386a3_atomic_write}; 'wf_a155' does NOT appear in "
                    "the branch's import set. RDKit graph calls: branch uses "
                    "{MolFromSmiles, GetAtoms, GetAtomicNum, GetBeginAtomIdx, GetEndAtomIdx, "
                    "SanitizeMol} and NEVER GetMolFrags / RemoveAtom / RWMol; the tool uses "
                    "GetMolFrags / RWMol / RemoveAtom and never GetBeginAtomIdx. Function-name "
                    "overlap between the two files = {main} only. The metal list is not "
                    "written into the branch script at all: it is extracted at run time with "
                    "ast from molmetal/molmetal_lam/lam_chem/{platinai_dataset.py, tmqm_dataset.py, "
                    "pt_click_compat.py}. The branch's own AST guard "
                    "(molmetal/tests/test_wfa163_c7_rederive.py) independently enforces this and "
                    "PASSES: I ran it CPU-only, 6 passed in 6.09 s.",
        "independently_measured": "0 shared non-stdlib imports touching the tool; "
                                  "GetMolFrags present in branch calls: False; RemoveAtom "
                                  "present in branch calls: False; 'wf_a155' in branch imports: False",
    },
    {
        "id": "B",
        "claim": "The McNemar p-value is computed by an INDEPENDENT implementation "
                 "(not by calling the tool's own routine).",
        "verdict": "NOT REFUTED as to the VALUE; PARTIALLY REFUTED as to ALGORITHMIC independence",
        "evidence": "The branch's mcnemar_exact (molmetal/scripts/wfa163_c7_rederive.py:258) "
                    "and the tool's exact_mcnemar_p "
                    "(molmetal/scripts/wf_a155_ligand_connected_rate.py:378) are separate "
                    "functions in separate files with no import between them. My AST "
                    "statement-level diff: only 2 of 9 body statements are textually identical "
                    "(the trivial 'n = int(b)+int(c)' and 'k_obs = min(int(b),int(c))'); the "
                    "accumulation and the float conversion differ (branch: Fraction(numer,denom) "
                    "-> float(frac); tool: numer/denom as a float division). They therefore use "
                    "the SAME standard algorithm (sum the pmf of outcomes at most as likely as "
                    "the observed one) written twice, so bit-identity of the two floats "
                    "demonstrates the conversion path, NOT algorithmic divergence. The value is "
                    "nevertheless independently confirmed by me via a THIRD, arithmetically "
                    "different route (the symmetry identity p = 2*sum_{x<=k} C(n,x)/2^n, which "
                    "never evaluates the threshold rule) and by scipy.stats.binomtest.",
        "independently_measured": "p(symmetry identity, my own) = 8.635435522167073e-197  ==  "
                                  "A162 p_value_two_sided 8.635435522167073e-197 (bit-for-bit); "
                                  "p(scipy.stats.binomtest(44, 902, 0.5, two-sided)) = "
                                  "8.635435522166859e-197 (rel. delta 2.4767e-14). My exact "
                                  "rational 91241067206710441686101386721479676138339271330704370608107248628864238723 / "
                                  "2^902 equals A162's published numerator/denominator exactly "
                                  "(A162's stored fraction is UNREDUCED, gcd != 1; mine is reduced "
                                  "but Fraction(pub_num,pub_den) == my_fraction -> True). "
                                  "A162 p_value_exact_numerator == 2*sum_{x<=44} C(902,x) -> True, "
                                  "so the publisher used the symmetry/threshold integer form.",
    },
    {
        "id": "C",
        "claim": "The probes actually BOUND the claim: a probe that changes the number is "
                 "reported as changing it.",
        "verdict": "NOT REFUTED (CONFIRMED, with two scope caveats recorded below)",
        "evidence": "The metal-predicate probe table (report.md:87-93) reports the "
                    "STRAWMAN_metalloid_widened predicate as MOVING the ligand-only rate "
                    "(0.02083991158825387 -> 0.018866435112093464, n 264 -> 239) and labels it "
                    "'**(strawman control)**'; the report devotes a whole section "
                    "('The one place the invariance WOULD break', report.md:154-156) to it. So "
                    "the probe that changes the number IS reported as changing it. The other "
                    "three probes move nothing and are reported as moving nothing: row order "
                    "identical (report.md:95-100), epoch column 'pooled rate moved: False' "
                    "(report.md:102-108), connected-definition 'hidden_work_detected: False' "
                    "(report.md:110-118). "
                    "CAVEAT 1 (tautology): the metal-INCLUSIVE column of the predicate probe is "
                    "predicate-invariant BY CONSTRUCTION - components(adj, all_idx) at "
                    "wfa163_c7_rederive.py:212 never consults metal_symbols, so the full arm "
                    "cannot move under any predicate. Its 'identical under all four vocabularies' "
                    "row is therefore not evidence. "
                    "CAVEAT 2 (weak discrimination): my element census over the pool shows only "
                    "6 metal symbols actually occur (Ir, Os, Pt, Re, Rh, Ru); all four 'genuine' "
                    "vocabularies (dblock_platinai, dblock_tmqm, mle82, pool_derived) induce the "
                    "IDENTICAL 6-element set on this pool, so their agreement is partly guaranteed, "
                    "not discovered. Only the deliberately-widened strawman (adding B, Si, As, Se, "
                    "Ge, Sb, Te) actually discriminates.",
        "independently_measured": "STRAWMAN ligand-only rate 0.018866435112093464 (n=239) vs "
                                  "0.02083991158825387 (n=264) - CONFIRMED by my own run. "
                                  "Vocabulary intersection with the pool's 19 elements: "
                                  "dblock_platinai 6, dblock_tmqm 6, mle82 6, pool_derived 6 - all "
                                  "equal to {Ir,Os,Pt,Re,Rh,Ru}; mle82 symbols NOT in dblock that "
                                  "occur in pool: [] (none); strawman induces 10 "
                                  "(adds As,B,Se,Si). Element counts in pool: As 270, B 864, Se 450, Si 792.",
    },
    {
        "id": "D",
        "claim": "The McNemar pairing direction is consistent: cell b is 'ligand-only connected, "
                 "metal-inclusive NOT' and cell c is 'metal-inclusive connected, ligand-only NOT'.",
        "verdict": "NOT REFUTED (CONFIRMED)",
        "evidence": "The tool's contingency() at wf_a155_ligand_connected_rate.py:359-374 counts "
                    "b at line 370 and c at line 372, and publishes them under the names "
                    "'b_ligand_only_connected_metal_inclusive_not' (line 597) and "
                    "'c_metal_inclusive_connected_ligand_only_not' (line 598). The branch's "
                    "compare() maps its own cells through the same names "
                    "(A162_CELL_KEYS in wfa163_c7_rederive.py) and its report.md:31-32 tabulates "
                    "b=44 and c=858 under exactly those A162 key names. The A162 artefact itself "
                    "carries the same semantics and even an explanatory note in the contingency "
                    "block. My own computation, using the definitions read off the tool's source, "
                    "reproduces b=44 and c=858 in the same cells - so the direction is neither "
                    "flipped nor mislabelled.",
        "independently_measured": "contingency a=220 b=44 c=858 d=11546 (my own, third "
                                  "implementation: RDKit GetMolFrags + RWMol.RemoveAtom) == "
                                  "A162 paired.contingency a_both_connected 220 / "
                                  "b_ligand_only_connected_metal_inclusive_not 44 / "
                                  "c_metal_inclusive_connected_ligand_only_not 858 / "
                                  "d_neither_connected 11546. Discordant total 902; "
                                  "McNemar uses min(b,c)=44.",
    },
    {
        "id": "E",
        "claim": "The 13/13 headline reproduction and its supporting numbers "
                 "(denominator, per-epoch, free-halide, connected-definition) are correct.",
        "verdict": "NOT REFUTED (CONFIRMED - all 13 reproduced independently)",
        "evidence": "I re-derived every headline quantity with my OWN implementation "
                    "(own csv.reader, RDKit GetMolFrags for the full arm, RWMol.RemoveAtom for "
                    "the ligand arm, own 30-symbol d-block list written from scratch, symmetry-"
                    "identity McNemar) and compared against the artefact under test and the "
                    "published metric on disk.",
        "independently_measured": "pool rows 12669; empty 0; unparseable 1 (line 9848, epoch 13, "
                                  "len 244, 'Maximum BFS search size exceeded'); analysed 12668; "
                                  "metal-inclusive rate 0.08509630565203663 (n=1078); ligand-only "
                                  "rate 0.02083991158825387 (n=264); mean_n_frags metal-inclusive "
                                  "10.244316387748658 and ligand-only 11.602778654878433 (both "
                                  "delta 0.0); contingency a=220 b=44 c=858 d=11546; per-epoch "
                                  "raw_single_rate 18/18 MATCH against "
                                  "metrics/by_metric/cfm_fragmentation_postproc_arm.json "
                                  "(18/18 n_input MATCH too - note epoch 13 has n_input 703 but "
                                  "n_parsed 702 because the unparseable row is counted in "
                                  "n_input); n_frags_raw vs the pool's own column: 0 mismatches "
                                  "over 12668 rows; max |largest_frag_frac| delta "
                                  "5.000000000000143e-05 (4-dp rounding); rows with <2 heavy atoms: "
                                  "0; alternative-denominator rate 1078/12669 = "
                                  "0.08508958875996527; free-halide rows 1903 -> after exclusion "
                                  "analysed 10765, MI rate 0.10013934045517882 (n=1078), LO rate "
                                  "0.024523920111472364 (n=264), contingency {a:220,b:44,c:858,"
                                  "d:9643}, LO relative lift +17.68% (branch says 17.7%). "
                                  "Re-running the branch script into a clean out-dir reproduced "
                                  "the shipped rederive.json with 0 differing leaf keys out of 308, "
                                  "and report.md byte-identical.",
    },
    {
        "id": "F",
        "claim": "The branch wrote no forbidden path (paper/**, docs/**, AGENTS.md, README.md, "
                 "TODO/**, configs/**, metrics/**, registry/**, molmetal_lam/**, "
                 "pre_flight_check.py, wf_a155_ligand_connected_rate.py, or a foreign test_wfa*).",
        "verdict": "NOT REFUTED (CONFIRMED)",
        "evidence": "git status --porcelain over the branch's own paths lists exactly five "
                    "untracked additions: molmetal/reports/wfa163_c7_rederive/, "
                    "molmetal/scripts/wfa163_c7_rederive.py, "
                    "molmetal/tests/test_wfa163_c7_rederive.py (branch V's own), plus the "
                    "pre-existing untracked molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ "
                    "and molmetal/scripts/wf_a155_ligand_connected_rate.py (the A162 branch's, "
                    "untracked before branch V ran). The tool under test was not modified: its "
                    "mtime is 10:27:42, earlier than branch V's artefacts (10:37-10:43). "
                    "molmetal/tests/test_wfa163_c7_rederive.py is branch V's own file (created by "
                    "the branch, and I only READ it).",
        "independently_measured": "0 modifications by branch V to any forbidden path; "
                                  "the branch's script writes only into --out-dir "
                                  "(atomic_write_file on report.md and rederive.json).",
    },
    {
        "id": "G",
        "claim": "FINDING (not a branch claim, but a durability defect in what the branch "
                 "shipped): the artefact under test and the branch's own machine-readable "
                 "artefact are both gitignored and therefore not durable in git.",
        "verdict": "CONFIRMED (defect in durability, not in the numbers)",
        "evidence": ".gitignore:158 is 'molmetal/reports/**/*.json'. "
                    "git check-ignore -v confirms it matches "
                    "molmetal/reports/wfa163_c7_rederive/rederive.json AND "
                    "molmetal/reports/wfa162_c7_pool_ligand_connected_rate/"
                    "ligand_connected_rate_c7.json. Both are untracked "
                    "(git ls-files returns nothing for the A162 dir). The repo already has the "
                    "whitelist mechanism for exactly this (.gitignore:159-165 '!molmetal/reports/"
                    "**/final.json' etc., and :177-178 a directory-level exception for "
                    "wfa114_cohort_matched_compare), but no exception was added for either "
                    "wfa163_c7_rederive or wfa162_c7_pool_ligand_connected_rate. Consequence: a "
                    "clean checkout of HEAD contains the scripts but neither JSON, so the "
                    "13/13 reproduction table cannot be re-verified from a fresh clone. "
                    "(branch V's report.md and my verdict.md ARE trackable - report.md is not "
                    "matched by the .json rule, and .gitignore:162 whitelists "
                    "'!molmetal/reports/**/verdict.md'.)",
        "independently_measured": "git check-ignore -v -> '.gitignore:158:molmetal/reports/**/*.json' "
                                  "for BOTH rederive.json and ligand_connected_rate_c7.json; "
                                  "git ls-files on the A162 report dir returns 0 files.",
    },
]

# --- write verdict.json ----------------------------------------------------
payload = {
    "wave": "Wf-A163",
    "role": "adversarial reviewer (branch V under test: from-scratch re-derivation of the c7 measurement)",
    "tag": "MEASURED",
    "date": "2026-11-01",
    "gpu": "0 (CPU-only; every python/pytest invocation prefixed with CUDA_VISIBLE_DEVICES=\"\")",
    "reviewer_scripts": [
        "/tmp/wfa163_refute/final_check.py",
        "/tmp/wfa163_refute/checks3.py",
        "/tmp/wfa163_refute/checks4.py",
        "/tmp/wfa163_refute/checks5.py",
        "/tmp/wfa163_refute/checks6.py",
        "/tmp/wfa163_refute/checks7.py",
        "/tmp/wfa163_refute/checks8.py",
        "/tmp/wfa163_refute/checks9.py",
        "/tmp/wfa163_refute/per_epoch_diff.py",
    ],
    "artefacts_under_test": [
        "molmetal/reports/wfa163_c7_rederive/report.md",
        "molmetal/reports/wfa163_c7_rederive/rederive.json",
        "molmetal/scripts/wfa163_c7_rederive.py",
        "molmetal/tests/test_wfa163_c7_rederive.py",
    ],
    "summary": "NO refutation succeeded. All 13 headline quantities reproduced under an "
               "independent implementation; suspects A, C, D confirmed as stated; suspect B "
               "confirmed as to the VALUE but only partially as to algorithmic independence "
               "(the two McNemar routines are two writings of the same standard algorithm - "
               "the value is nonetheless confirmed by my symmetry-identity route and by scipy). "
               "One durability defect found outside the branch's numeric claims: entry G.",
    "n_refuted": 0,
    "n_confirmed": 7,
    "entries": ENTRIES,
}
atomic_write_file(OUT / "verdict.json", json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

# --- write verdict.md ------------------------------------------------------
L = []
L.append("# Wf-A163 — ADVERSARIAL REVIEW of branch V (from-scratch re-derivation of the c7 measurement)")
L.append("")
L.append("- tag: **MEASURED**  date: 2026-11-01  wave: Wf-A163  role: adversarial reviewer")
L.append("- branch under test: `molmetal/reports/wfa163_c7_rederive/{report.md,rederive.json}`, "
         "`molmetal/scripts/wfa163_c7_rederive.py`, `molmetal/tests/test_wfa163_c7_rederive.py`")
L.append("- artefact under test: `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json`")
L.append("- tool under test: `molmetal/scripts/wf_a155_ligand_connected_rate.py`")
L.append("- GPU: 0 (CPU-only; every python/pytest call prefixed with `CUDA_VISIBLE_DEVICES=\"\"`)")
L.append("")
L.append("## VERDICT: **NO REFUTATION SUCCEEDED** (7 entries: 6 NOT REFUTED / CONFIRMED, "
         "1 CONFIRMED defect outside the numeric claims)")
L.append("")
L.append("Every headline number in the branch was re-derived by MY OWN implementation "
         "(own csv reader, RDKit `GetMolFrags` for the full arm, `RWMol.RemoveAtom` for the "
         "ligand arm, a from-scratch 30-symbol d-block list, symmetry-identity McNemar) — "
         "not by re-reading the branch's own artefact. All 13 reproduce exactly.")
L.append("")
L.append("## Independent re-derivation (my numbers, my code)")
L.append("")
L.append("| quantity | my value | on disk at |")
L.append("|---|---|---|")
L.append("| pool data rows / empty / unparseable | 12669 / 0 / 1 | "
         "`molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv` |")
L.append("| unparseable site | line 9848, epoch 13, len 244 | same CSV |")
L.append("| analysed denominator | 12668 | `ligand_connected_rate_c7.json:/denominators/n_analyzed` |")
L.append("| metal-inclusive single-fragment rate | **0.08509630565203663** (n=1078) | "
         "`/metal_inclusive/single_fragment_rate` |")
L.append("| ligand-only single-fragment rate | **0.02083991158825387** (n=264) | "
         "`/ligand_only/single_fragment_rate` |")
L.append("| mean_n_frags (MI / LO) | 10.244316387748658 / 11.602778654878433 (delta 0.0) | "
         "`/metal_inclusive/mean_n_frags`, `/ligand_only/mean_n_frags` |")
L.append("| contingency a/b/c/d | 220 / 44 / 858 / 11546 | `/paired/contingency` |")
L.append("| McNemar p (symmetry identity, my route) | **8.635435522167073e-197** | "
         "`/paired/mcnemar_exact/p_value_two_sided` |")
L.append("| McNemar p (scipy binomtest) | 8.635435522166859e-197 (rel. delta 2.4767e-14) | "
         "cross-check only |")
L.append("| per-epoch raw_single_rate | **18/18 MATCH** | "
         "`metrics/by_metric/cfm_fragmentation_postproc_arm.json:/per_epoch` |")
L.append("| per-epoch n_input | **18/18 MATCH** (epoch 13: n_input 703, n_parsed 702) | same |")
L.append("| n_frags_raw vs pool's own column | 0 mismatches / 12668 rows | pool CSV col 3 |")
L.append("| max \\|largest_frag_frac\\| delta | 5.000000000000143e-05 (4-dp rounding) | pool CSV col 4 |")
L.append("| rows with <2 heavy atoms | 0 | — |")
L.append("| rate if denominator were 12669 | 0.08508958875996527 | branch report.md:49 |")
L.append("| free-halide rows -> after exclusion | 1903 -> analysed 10765 | branch report.md:123-127 |")
L.append("| free-halide-excluded MI / LO rate | 0.10013934045517882 (n=1078) / "
         "0.024523920111472364 (n=264) | branch report.md:125-126 |")
L.append("| free-halide-excluded contingency | {a:220, b:44, c:858, d:9643} | branch report.md:127 |")
L.append("")
L.append("**Artefact/script consistency:** re-running `molmetal/scripts/wfa163_c7_rederive.py "
         "--out-dir <fresh>` reproduced the shipped `rederive.json` with **0 differing leaf keys "
         "out of 308**, and `report.md` byte-identical (`diff -q` -> IDENTICAL).")
L.append("")
L.append("**Branch's own test:** `molmetal/tests/test_wfa163_c7_rederive.py` — CPU-only run, "
         "**6 passed in 6.09 s**.")
L.append("")
L.append("## The four suspects")
L.append("")
L.append("### (a) No shared code with the tool under test — **NOT REFUTED (CONFIRMED)**")
L.append("")
L.append("Own AST parse of both files: the branch's import set does **not** contain `wf_a155`; "
         "the branch never calls `GetMolFrags`, `RemoveAtom` or `RWMol` (it builds its own bond "
         "adjacency from `GetBeginAtomIdx`/`GetEndAtomIdx` and runs its own BFS); the tool does "
         "the opposite. Function-name overlap between the two files = `{main}` only. The metal "
         "vocabulary is not written into the branch script at all — it is `ast`-extracted at run "
         "time from `molmetal/molmetal_lam/lam_chem/{platinai_dataset,tmqm_dataset,pt_click_compat}.py`. "
         "The branch's own AST guard enforces this and passes.")
L.append("")
L.append("### (b) McNemar p-value independently computed — **VALUE CONFIRMED / "
         "ALGORITHM-ONLY-PARTIALLY INDEPENDENT**")
L.append("")
L.append("The two routines are separate functions in separate files with no import between "
         "them (`wfa163_c7_rederive.py:258` `mcnemar_exact`; `wf_a155_ligand_connected_rate.py:378` "
         "`exact_mcnemar_p`). My statement-level AST diff finds only **2 of 9** body statements "
         "textually identical (`n = int(b)+int(c)`, `k_obs = min(int(b),int(c))`); the "
         "accumulation and float conversion differ (branch `Fraction(numer,denom)` -> "
         "`float(frac)`; tool `numer/denom`). So they are two writings of the **same standard "
         "algorithm**, and the bit-for-bit float identity demonstrates the conversion path, not "
         "algorithmic divergence.")
L.append("")
L.append("The value is nonetheless confirmed by me via a **third, arithmetically different "
         "route** — the symmetry identity `p = 2*sum_{x<=k} C(n,x)/2^n`, which never evaluates "
         "the threshold rule — giving **8.635435522167073e-197**, bit-identical to the published "
         "value. `scipy.stats.binomtest(44, 902, 0.5, 'two-sided')` gives "
         "8.635435522166859e-197 (rel. delta 2.4767e-14), confirming the branch's inference that "
         "the publisher used an exact rational computation, not scipy's float. My exact rational "
         "equals A162's stored `p_value_exact_numerator` / `p_value_exact_denominator` exactly "
         "(A162's stored fraction is UNREDUCED, gcd != 1; `Fraction(pub_num,pub_den) == mine` -> True; "
         "and `pub_num == 2*sum_{x<=44} C(902,x)` -> True).")
L.append("")
L.append("### (c) Do the probes bound the claim? — **NOT REFUTED (CONFIRMED), two scope caveats**")
L.append("")
L.append("The probe that changes the number IS reported as changing it: the "
         "`STRAWMAN_metalloid_widened` predicate moves the ligand-only rate "
         "**0.02083991158825387 -> 0.018866435112093464 (n 264 -> 239)**, is labelled "
         "'**(strawman control)**' in the table (report.md:93), and gets its own report section "
         "(report.md:154-156). The three probes that move nothing are reported as moving nothing.")
L.append("")
L.append("**CAVEAT 1 — the metal-inclusive column of the predicate probe is a tautology.** "
         "`components(adj, all_idx)` (wfa163_c7_rederive.py:212) never consults `metal_symbols`, "
         "so the full arm cannot move under any predicate; its 'identical under all four "
         "vocabularies' row is not evidence.")
L.append("")
L.append("**CAVEAT 2 — the four 'genuine' vocabularies are partly guaranteed to agree.** My "
         "element census over the pool shows only **6** metal symbols occur "
         "(Ir, Os, Pt, Re, Rh, Ru). Intersecting each vocabulary with the pool's 19 elements: "
         "dblock_platinai 6, dblock_tmqm 6, mle82 6, pool_derived 6 — **all the identical set**; "
         "no mle82 symbol outside the d-block occurs in the pool at all. Their agreement is "
         "structural, not discovered. Only the widened strawman (adds B, Si, As, Se, Ge, Sb, Te; "
         "B 864, Si 792, Se 450, As 270 occurrences) actually discriminates. The branch does not "
         "overclaim here — it labels the strawman a control — but a reader could take the "
         "4-vocabulary agreement for four independent confirmations when it is one.")
L.append("")
L.append("### (d) McNemar pairing direction — **NOT REFUTED (CONFIRMED)**")
L.append("")
L.append("`wf_a155_ligand_connected_rate.py:359-374` counts b at :370 and c at :372, and "
         "publishes them as `b_ligand_only_connected_metal_inclusive_not` (:597) and "
         "`c_metal_inclusive_connected_ligand_only_not` (:598). The branch's `compare()` binds "
         "its cells through the same A162 key names and tabulates b=44, c=858 under them "
         "(report.md:31-32). My own computation using those definitions reproduces **b=44, c=858** "
         "in the same cells. Direction is neither flipped nor mislabelled. Discordant total 902; "
         "the test uses min(b,c)=44.")
L.append("")
L.append("## Denominators and precision")
L.append("")
L.append("- The **population** is what the branch says: the 12669-row CFM generation pool "
         "(`molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`), 1 RDKit-unparseable "
         "row excluded for a stated and reproducible reason ('Maximum BFS search size exceeded', "
         "epoch 13, line 9848) -> denominator 12668. The published metric's own "
         "`/denominators/n_parsed` is 12668. The alternative-denominator rate (0.08508958875996527) "
         "is stated, not hidden.")
L.append("- **Precision**: every 'N of M' in the branch survives adjudication at full precision — "
         "1078/12668 and 264/12668 reproduce to the last bit, as do the mean_n_frags pair and the "
         "exact-rational p-value. The branch quotes the p-value at full double precision and "
         "additionally publishes the exact numerator/denominator, which I matched exactly.")
L.append("- **Free-halide probe**: I independently confirmed 1903 free-halide rows and, "
         "critically, that **0** of them are connected in either arm and **0** have the halide as "
         "their only non-metal heavy content — so the branch's 'numerator unchanged, denominator "
         "+1903' reasoning holds and the LO rate does dilute (not inflate) the ligand-only arm. "
         "My recomputed relative lift is **+17.68 %** vs the branch's stated 17.7 %.")
L.append("")
L.append("## Forbidden-path writes: **none by branch V**")
L.append("")
L.append("`git status --porcelain` over the branch's paths shows exactly the branch's own "
         "additions (`molmetal/reports/wfa163_c7_rederive/`, `molmetal/scripts/wfa163_c7_rederive.py`, "
         "`molmetal/tests/test_wfa163_c7_rederive.py`) plus the A162 branch's pre-existing "
         "untracked files. The tool under test was **not** modified (mtime 10:27:42, earlier than "
         "branch V's artefacts at 10:37-10:43). The branch script writes only into `--out-dir`, "
         "via `atomic_write_file`.")
L.append("")
L.append("## FINDING G — durability defect (CONFIRMED; outside the branch's numeric claims)")
L.append("")
L.append("`.gitignore:158` is `molmetal/reports/**/*.json`, and it matches **both**")
L.append("`molmetal/reports/wfa163_c7_rederive/rederive.json` **and**")
L.append("`molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json`.")
L.append("Both are untracked (`git ls-files` on the A162 dir returns 0 files). The repo already has")
L.append("the whitelist mechanism for exactly this — `.gitignore:159-165`")
L.append("(`!molmetal/reports/**/final.json`, `!molmetal/reports/**/verdict.md`, ...) and")
L.append("`:177-178` (a directory-level exception for `wfa114_cohort_matched_compare`) — but no")
L.append("exception was added for `wfa163_c7_rederive` or `wfa162_c7_pool_ligand_connected_rate`.")
L.append("Consequence: a clean checkout of HEAD carries the scripts but neither JSON, so the")
L.append("13/13 reproduction table cannot be re-verified from a fresh clone. This is a durability")
L.append("gap, not a numerical error.")
L.append("")
L.append("(Branch V's `report.md` is trackable — the `.json` rule does not match it — and this")
L.append("file is whitelisted by `.gitignore:162` `!molmetal/reports/**/verdict.md`. The")
L.append("machine-readable `verdict.json` in this directory IS gitignored by the same rule.)")
L.append("")
L.append("## What would still be worth doing")
L.append("")
L.append("1. Add `.gitignore` exceptions for `molmetal/reports/wfa163_c7_rederive/**` and")
L.append("   `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/**` so the reproduction is")
L.append("   re-verifiable from a clean clone (Finding G).")
L.append("2. Re-label the metal-inclusive column of the predicate probe as **invariant by")
L.append("   construction** rather than as agreement between four vocabularies, and note that")
L.append("   the four genuine vocabularies induce one and the same 6-symbol set on this pool")
L.append("(Caveats 1-2).")
L.append("3. Optional: the McNemar agreement is a same-algorithm re-implementation; a scipy")
L.append("   cross-check already present in the branch would be the cleaner independence claim")
L.append("   if the wording is tightened.")
L.append("")
L.append("## Citation")
L.append("")
L.append("- branch: `molmetal/reports/wfa163_c7_rederive/report.md`, `rederive.json`")
L.append("- artefact under test: `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json`")
L.append("- tool under test: `molmetal/scripts/wf_a155_ligand_connected_rate.py`")
L.append("- published metric: `metrics/by_metric/cfm_fragmentation_postproc_arm.json`")
L.append("- pool: `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`")
L.append("- my reviewer scripts: `/tmp/wfa163_refute/{final_check,checks3..checks9,per_epoch_diff}.py`")
L.append("- this verdict: `molmetal/reports/wfa163_refute_v3/verdict.md` + `verdict.json`")
L.append("")

atomic_write_file(OUT / "verdict.md", "\n".join(L))
print("WROTE", OUT / "verdict.md", "and", OUT / "verdict.json")
