# Wf-A164 — Adversarial refutation of Branch A (`molmetal/scripts/wf_a155_posebusters_validity.py`)

**VERDICT: NOT REFUTED** (one branch sub-claim REFUTED; one confirmed defect of the same
class survives in the opt-in `--sample-size` path).

Reviewer: read-only adversarial branch. 0 GPU (`CUDA_VISIBLE_DEVICES=""` on every call),
single venv `/home/hugo/codes/mol-metal/.venv/bin/python`, no `os.walk`, every number below
traces to a file:line or to an artifact this reviewer produced.

## Checks

| # | Check | Result |
|---|---|---|
| C1.1 | Claimed literals really gone from the artifact path; replacements COMPUTED | **PASS** |
| C1.2 | Branch claim "`529/2112 = 25.05 %` -> `n_single_fragment_rows/n_data_rows`" | **REFUTED** |
| C1.3 | Second scope-vs-pool conflation in the same repair | **FAIL** |
| C2 | Regression test bites; negative control really fails | **PASS** |
| C3 | Guard is not vacuous | **PASS** (with a recorded blind spot) |
| C4 | Nothing broke on the default pool | **PASS** |
| C5 | Scope discipline | **PASS** |
| C6 | Frontier numbers re-fetched from PRIMARY arXiv records | **PASS** |
| C7 | Branch's own report numbers spot-checked | **PASS** |

### C1.1 — the defect was real and the repair removed it
Running the branch's own pre-repair snapshot
(`molmetal/reports/wfa164_posebusters_narrative/pre_repair_script_snapshot.py.txt`) on a 5-row
copy pool emits, in a `tag: MEASURED` artifact: `2112 rows ... 704 rows each, 6 seeds per
epoch ... cfm_c8_fullgen_2026_10_24/generated_smiles.csv`, `cfm_yuelbond_wasserstein_seed0`,
`post_hoc_decode_pool_3_epochs_1_checkpoint`, `529/2112 = 25.05 %`. The repaired script on the
SAME pool emits none of {2112, "704 rows", 8.51, 25.05, cfm_c8_fullgen_2026_10_24, 529, seed0,
post_hoc_decode_pool_3_epochs_1_checkpoint} and instead renders the copy's own numbers
(`ONE pool of 5 rows spanning 2 epoch labels with 3, 2 rows respectively ... (…/wfa164_pool_copy/generated_smiles.csv)`).
On the default pool the same code path re-derives 2112 rows / 704 per epoch / 6 seeds / 1
checkpoint / 529 single-fragment from the CSV, the sibling `events.jsonl` and the sibling
`checkpoints/` dir — never typed (script `:330-399`, `:256-327`, `:402-443`).

### C1.2 / C1.3 — the one surviving defect (CONFIRMED, same class as the wave)
`molmetal/scripts/wf_a155_posebusters_validity.py:1859` sets `n_rows = counts["n_rows_in_scope"]`
— the **scope** count — and that value is used as the denominator at `:1934`
(`f"{profile['n_single_fragment_rows']}/{n_rows} = {single_pct:.2f} %"`) and in the population
claim at `:1962` (`"all {n_rows} rows of the pool named in source_pool"`), while `:1877`
`single_pct = profile["single_fragment_fraction"] * 100.0` is **pool**-level.

MEASURED, default pool, 20-row sample: the artifact renders `529/20 = 25.05 %` —
100 x 529 / 20 = **2645.0**, not 25.05 (25.05 % is 529/2112).
MEASURED, 10-row pool sampled to 3: `5/3 = 50.00 %` — 100 x 5 / 3 = **166.667**, not 50.0;
and `... IS the first POOL-SCALE PoseBusters measurement in the repo: all 3 rows of the pool named in source_pool`
while the pool has 10 rows and `scope.label` says `SUBSET estimate -- NOT the pool rate`.

This lands ON DISK in the branch's own evidence artifact
`molmetal/reports/wfa164_posebusters_narrative/default_pool_bounded_repaired.json` (`tag: MEASURED`).
Pre-repair, the same case rendered `529/2112 = 25.05 %` and `all 2112 rows of the published CFM
decode pool` — also false, but stale rather than inconsistent. So the bounded path was false
before the repair and is false after it in a new shape.

NOT triggered by the default full-pool run (`--sample-size` defaults to `None`, `:2662-2666`);
`metrics/by_metric/wfa155_posebusters_validity.json` (full_pool, 2112/2112) is internally
consistent (`529/2112 = 25.05 %`, 100 x 529/2112 = 25.0473) and was not republished.

### C2 — the test bites
`CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa164_posebusters_narrative.py -v`
-> **9 passed in 5.12s**. The negative control is real, not a tautology: it monkeypatches
`tool.build_comparability_breaks` back to the OLD literal break (verbatim from pre-repair
`:148-158`) and asserts the guard rejects (`unbacked_numbers` contains `2112` AND `704`,
`stale_paths` contains the default pool CSV); a second test proves `tool.main(...) == 1` and a
`status: FAIL` artifact on disk.

### C3 — the guard is not vacuous (independent re-run, not the branch's test)
Prose `the pool has 314159 rows in total` with no machine counterpart -> `pass=False`,
`unbacked_numbers={'314159': ['narrative.claim']}`. Prose repo path with no machine string ->
`pass=False`, `stale_paths={'molmetal/reports/other_pool/generated_smiles.csv': [...]}`. It fires.
**Blind spot**: the guard proves each prose number is backed by *some* machine field; it does not
prove a rendered ratio is internally consistent, nor that a rendered population is the pool
rather than the scope. That is why C1.2/C1.3 pass the guard.

### C4 — nothing broke
Independent default-pool bounded run (20 / seed 0 / probe off / row-timeout 20): exit 0,
**1/20 strict, 1/13 scored, 7 embed-failed, 9 force-field failures** — identical to the branch's
reported counts. Guard: `pass=True`, `n_prose_leaves=68`, `n_prose_numbers=66`,
`n_machine_leaves=462`, `unbacked_numbers={}`, `stale_paths={}`, `unclassified_leaves={}`,
`unresolved_narrative_paths=[]` — matches the branch's "68 prose leaves / 66 prose numbers".

### C5 — scope discipline
Touched: `molmetal/scripts/wf_a155_posebusters_validity.py` (owned),
`molmetal/tests/test_wfa164_posebusters_narrative.py` (own test),
`molmetal/reports/wfa164_posebusters_narrative/*` (own dir). NOT written:
`metrics/by_metric/wfa155_posebusters_validity.json` (mtime 2026-11-01 02:20:37, 9 h before the
window), `registry/metrics.yaml` (10:40), `molmetal/reports/run_ledger.jsonl` (2026-10-31);
nothing in `paper/ docs/ TODO/ configs/` created or modified in 11:30-12:00.
`molmetal/scripts/promote_metrics.py` has mtime 11:33 but its diff is Wf-A123/Wf-A163 work
(`--target-root`/`--log-path`, `wfa163_narrative_truth_guard` import) — a different wave.

### C6 — frontier numbers from PRIMARY records
`curl -sSL "https://export.arxiv.org/api/query?id_list=2609.08333&max_results=1"` -> "Fixed-Dimensional
Latent Flow for Generating Variable-Size 3D Molecules", Yao / Gruich / Goldsmith / Wang, published
2026-09-08T07:03:51Z, abstract verbatim: "On PCQM4Mv2, unconditional generation yields 87.9%
unique and novel molecules passing sanitization and PoseBusters checks" — matches the branch's
recorded `record_text_used` exactly (87.9 %, not 89.4 %). `2508.12629` -> "FlowMol3: Flow Matching
for 3D De Novo Small-Molecule Generation", Dunn / Koes, published 2025-08-18T05:13:27Z, "nearly
100% molecular validity for drug-like molecules with explicit hydrogens" — matches. The EF-TALFM
table sub-numbers 92.98 / 98.25 / 98.21 / 94.33 are in neither abstract and are traced to
`molmetal/reports/wfa161_frontier_labelling/report.md:37-38`, which carries the table row and the
caption "EF-TALFM achieves 94.33%" — the branch's "not re-derived here" disclosure is accurate.

## Surviving concerns
1. **CONFIRMED DEFECT** — the scope-vs-pool conflation (C1.2 / C1.3). One-line fix: use
   `profile["n_data_rows"]` for the pool-level ratio denominator and the "all N rows of the pool"
   claim, and say "the N rows measured" when `scope.is_subset`.
2. **Guard blind spot** (C3) — add a rendered-ratio consistency check.
3. `pre_repair_script_snapshot.py.txt` is an executable second copy of a measurement tool that can
   write artifacts; consider a `.md` suffix or a header note.
4. Docstring-only pool literals remain in the repaired script (`:343`, `:453`, `:538`, `:638`,
   `:966`, `:1452`) — outside the artifact and disclosed, but a grep for `2112`/`704`/`8.51` still hits.
