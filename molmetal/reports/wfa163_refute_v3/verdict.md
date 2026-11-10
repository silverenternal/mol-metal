# Wf-A163 — ADVERSARIAL REVIEW of branch V (from-scratch re-derivation of the c7 measurement)

- tag: **MEASURED**  date: 2026-11-01  wave: Wf-A163  role: adversarial reviewer
- branch under test: `molmetal/reports/wfa163_c7_rederive/{report.md,rederive.json}`, `molmetal/scripts/wfa163_c7_rederive.py`, `molmetal/tests/test_wfa163_c7_rederive.py`
- artefact under test: `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json`
- tool under test: `molmetal/scripts/wf_a155_ligand_connected_rate.py`
- GPU: 0 (CPU-only; every python/pytest call prefixed with `CUDA_VISIBLE_DEVICES=""`)

## VERDICT: **NO REFUTATION SUCCEEDED** (7 entries: 6 NOT REFUTED / CONFIRMED, 1 CONFIRMED defect outside the numeric claims)

Every headline number in the branch was re-derived by MY OWN implementation (own csv reader, RDKit `GetMolFrags` for the full arm, `RWMol.RemoveAtom` for the ligand arm, a from-scratch 30-symbol d-block list, symmetry-identity McNemar) — not by re-reading the branch's own artefact. All 13 reproduce exactly.

## Independent re-derivation (my numbers, my code)

| quantity | my value | on disk at |
|---|---|---|
| pool data rows / empty / unparseable | 12669 / 0 / 1 | `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv` |
| unparseable site | line 9848, epoch 13, len 244 | same CSV |
| analysed denominator | 12668 | `ligand_connected_rate_c7.json:/denominators/n_analyzed` |
| metal-inclusive single-fragment rate | **0.08509630565203663** (n=1078) | `/metal_inclusive/single_fragment_rate` |
| ligand-only single-fragment rate | **0.02083991158825387** (n=264) | `/ligand_only/single_fragment_rate` |
| mean_n_frags (MI / LO) | 10.244316387748658 / 11.602778654878433 (delta 0.0) | `/metal_inclusive/mean_n_frags`, `/ligand_only/mean_n_frags` |
| contingency a/b/c/d | 220 / 44 / 858 / 11546 | `/paired/contingency` |
| McNemar p (symmetry identity, my route) | **8.635435522167073e-197** | `/paired/mcnemar_exact/p_value_two_sided` |
| McNemar p (scipy binomtest) | 8.635435522166859e-197 (rel. delta 2.4767e-14) | cross-check only |
| per-epoch raw_single_rate | **18/18 MATCH** | `metrics/by_metric/cfm_fragmentation_postproc_arm.json:/per_epoch` |
| per-epoch n_input | **18/18 MATCH** (epoch 13: n_input 703, n_parsed 702) | same |
| n_frags_raw vs pool's own column | 0 mismatches / 12668 rows | pool CSV col 3 |
| max \|largest_frag_frac\| delta | 5.000000000000143e-05 (4-dp rounding) | pool CSV col 4 |
| rows with <2 heavy atoms | 0 | — |
| rate if denominator were 12669 | 0.08508958875996527 | branch report.md:49 |
| free-halide rows -> after exclusion | 1903 -> analysed 10765 | branch report.md:123-127 |
| free-halide-excluded MI / LO rate | 0.10013934045517882 (n=1078) / 0.024523920111472364 (n=264) | branch report.md:125-126 |
| free-halide-excluded contingency | {a:220, b:44, c:858, d:9643} | branch report.md:127 |

**Artefact/script consistency:** re-running `molmetal/scripts/wfa163_c7_rederive.py --out-dir <fresh>` reproduced the shipped `rederive.json` with **0 differing leaf keys out of 308**, and `report.md` byte-identical (`diff -q` -> IDENTICAL).

**Branch's own test:** `molmetal/tests/test_wfa163_c7_rederive.py` — CPU-only run, **6 passed in 6.09 s**.

## The four suspects

### (a) No shared code with the tool under test — **NOT REFUTED (CONFIRMED)**

Own AST parse of both files: the branch's import set does **not** contain `wf_a155`; the branch never calls `GetMolFrags`, `RemoveAtom` or `RWMol` (it builds its own bond adjacency from `GetBeginAtomIdx`/`GetEndAtomIdx` and runs its own BFS); the tool does the opposite. Function-name overlap between the two files = `{main}` only. The metal vocabulary is not written into the branch script at all — it is `ast`-extracted at run time from `molmetal/molmetal_lam/lam_chem/{platinai_dataset,tmqm_dataset,pt_click_compat}.py`. The branch's own AST guard enforces this and passes.

### (b) McNemar p-value independently computed — **VALUE CONFIRMED / ALGORITHM-ONLY-PARTIALLY INDEPENDENT**

The two routines are separate functions in separate files with no import between them (`wfa163_c7_rederive.py:258` `mcnemar_exact`; `wf_a155_ligand_connected_rate.py:378` `exact_mcnemar_p`). My statement-level AST diff finds only **2 of 9** body statements textually identical (`n = int(b)+int(c)`, `k_obs = min(int(b),int(c))`); the accumulation and float conversion differ (branch `Fraction(numer,denom)` -> `float(frac)`; tool `numer/denom`). So they are two writings of the **same standard algorithm**, and the bit-for-bit float identity demonstrates the conversion path, not algorithmic divergence.

The value is nonetheless confirmed by me via a **third, arithmetically different route** — the symmetry identity `p = 2*sum_{x<=k} C(n,x)/2^n`, which never evaluates the threshold rule — giving **8.635435522167073e-197**, bit-identical to the published value. `scipy.stats.binomtest(44, 902, 0.5, 'two-sided')` gives 8.635435522166859e-197 (rel. delta 2.4767e-14), confirming the branch's inference that the publisher used an exact rational computation, not scipy's float. My exact rational equals A162's stored `p_value_exact_numerator` / `p_value_exact_denominator` exactly (A162's stored fraction is UNREDUCED, gcd != 1; `Fraction(pub_num,pub_den) == mine` -> True; and `pub_num == 2*sum_{x<=44} C(902,x)` -> True).

### (c) Do the probes bound the claim? — **NOT REFUTED (CONFIRMED), two scope caveats**

The probe that changes the number IS reported as changing it: the `STRAWMAN_metalloid_widened` predicate moves the ligand-only rate **0.02083991158825387 -> 0.018866435112093464 (n 264 -> 239)**, is labelled '**(strawman control)**' in the table (report.md:93), and gets its own report section (report.md:154-156). The three probes that move nothing are reported as moving nothing.

**CAVEAT 1 — the metal-inclusive column of the predicate probe is a tautology.** `components(adj, all_idx)` (wfa163_c7_rederive.py:212) never consults `metal_symbols`, so the full arm cannot move under any predicate; its 'identical under all four vocabularies' row is not evidence.

**CAVEAT 2 — the four 'genuine' vocabularies are partly guaranteed to agree.** My element census over the pool shows only **6** metal symbols occur (Ir, Os, Pt, Re, Rh, Ru). Intersecting each vocabulary with the pool's 19 elements: dblock_platinai 6, dblock_tmqm 6, mle82 6, pool_derived 6 — **all the identical set**; no mle82 symbol outside the d-block occurs in the pool at all. Their agreement is structural, not discovered. Only the widened strawman (adds B, Si, As, Se, Ge, Sb, Te; B 864, Si 792, Se 450, As 270 occurrences) actually discriminates. The branch does not overclaim here — it labels the strawman a control — but a reader could take the 4-vocabulary agreement for four independent confirmations when it is one.

### (d) McNemar pairing direction — **NOT REFUTED (CONFIRMED)**

`wf_a155_ligand_connected_rate.py:359-374` counts b at :370 and c at :372, and publishes them as `b_ligand_only_connected_metal_inclusive_not` (:597) and `c_metal_inclusive_connected_ligand_only_not` (:598). The branch's `compare()` binds its cells through the same A162 key names and tabulates b=44, c=858 under them (report.md:31-32). My own computation using those definitions reproduces **b=44, c=858** in the same cells. Direction is neither flipped nor mislabelled. Discordant total 902; the test uses min(b,c)=44.

## Denominators and precision

- The **population** is what the branch says: the 12669-row CFM generation pool (`molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`), 1 RDKit-unparseable row excluded for a stated and reproducible reason ('Maximum BFS search size exceeded', epoch 13, line 9848) -> denominator 12668. The published metric's own `/denominators/n_parsed` is 12668. The alternative-denominator rate (0.08508958875996527) is stated, not hidden.
- **Precision**: every 'N of M' in the branch survives adjudication at full precision — 1078/12668 and 264/12668 reproduce to the last bit, as do the mean_n_frags pair and the exact-rational p-value. The branch quotes the p-value at full double precision and additionally publishes the exact numerator/denominator, which I matched exactly.
- **Free-halide probe**: I independently confirmed 1903 free-halide rows and, critically, that **0** of them are connected in either arm and **0** have the halide as their only non-metal heavy content — so the branch's 'numerator unchanged, denominator +1903' reasoning holds and the LO rate does dilute (not inflate) the ligand-only arm. My recomputed relative lift is **+17.68 %** vs the branch's stated 17.7 %.

## Forbidden-path writes: **none by branch V**

`git status --porcelain` over the branch's paths shows exactly the branch's own additions (`molmetal/reports/wfa163_c7_rederive/`, `molmetal/scripts/wfa163_c7_rederive.py`, `molmetal/tests/test_wfa163_c7_rederive.py`) plus the A162 branch's pre-existing untracked files. The tool under test was **not** modified (mtime 10:27:42, earlier than branch V's artefacts at 10:37-10:43). The branch script writes only into `--out-dir`, via `atomic_write_file`.

## FINDING G — durability defect (CONFIRMED; outside the branch's numeric claims)

`.gitignore:158` is `molmetal/reports/**/*.json`, and it matches **both**
`molmetal/reports/wfa163_c7_rederive/rederive.json` **and**
`molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json`.
Both are untracked (`git ls-files` on the A162 dir returns 0 files). The repo already has
the whitelist mechanism for exactly this — `.gitignore:159-165`
(`!molmetal/reports/**/final.json`, `!molmetal/reports/**/verdict.md`, ...) and
`:177-178` (a directory-level exception for `wfa114_cohort_matched_compare`) — but no
exception was added for `wfa163_c7_rederive` or `wfa162_c7_pool_ligand_connected_rate`.
Consequence: a clean checkout of HEAD carries the scripts but neither JSON, so the
13/13 reproduction table cannot be re-verified from a fresh clone. This is a durability
gap, not a numerical error.

(Branch V's `report.md` is trackable — the `.json` rule does not match it — and this
file is whitelisted by `.gitignore:162` `!molmetal/reports/**/verdict.md`. The
machine-readable `verdict.json` in this directory IS gitignored by the same rule.)

## What would still be worth doing

1. Add `.gitignore` exceptions for `molmetal/reports/wfa163_c7_rederive/**` and
   `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/**` so the reproduction is
   re-verifiable from a clean clone (Finding G).
2. Re-label the metal-inclusive column of the predicate probe as **invariant by
   construction** rather than as agreement between four vocabularies, and note that
   the four genuine vocabularies induce one and the same 6-symbol set on this pool
(Caveats 1-2).
3. Optional: the McNemar agreement is a same-algorithm re-implementation; a scipy
   cross-check already present in the branch would be the cleaner independence claim
   if the wording is tightened.

## Citation

- branch: `molmetal/reports/wfa163_c7_rederive/report.md`, `rederive.json`
- artefact under test: `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json`
- tool under test: `molmetal/scripts/wf_a155_ligand_connected_rate.py`
- published metric: `metrics/by_metric/cfm_fragmentation_postproc_arm.json`
- pool: `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`
- my reviewer scripts: `/tmp/wfa163_refute/{final_check,checks3..checks9,per_epoch_diff}.py`
- this verdict: `molmetal/reports/wfa163_refute_v3/verdict.md` + `verdict.json`
