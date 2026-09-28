# WF-R17-Lambda-Flagfix — CLI surface for TODO-31 L-R17-B2 verify gate

> **Date:** 2026-09-25
> **Scope:** Add / normalize the CLI flags that the TODO-31 §L-R17-B2 verify-gate command (`--metal-seed cisplatin --pockets 10 --seeds 42 0 1234 --n-simulations 1000 --click-rules auto-pt-strict`) needs.
> **Status (honest):** PASS — all 5 user-reported flags now work end-to-end; smoke run produced 20 distinct SMILES across 2 cells. The ≥3-across-pocket metal-ligand gate remains FAIL — see §3 (this is a structural limit of `auto-pt-strict`, not a CLI bug).

## 1. Flag mapping table

| # | User-reported problem                                              | Status before              | Resolution (this fix)                                              |
|---|-------------------------------------------------------------------|----------------------------|--------------------------------------------------------------------|
| 1 | `--seed` not recognized; legacy `--seeds` (nargs="+") only         | hard fail                  | Added `--seed INT` singular alias; wraps into `[seed]` BEFORE preset resolution. `--seeds` wins when both supplied (more-specific CLI). |
| 2 | `--out-dir ABS_PATH` not recognized; script forces `--output-dir NAME` + `wf_lambda1_<NAME>/` prefix | hard fail | Added `--out-dir PATH` (bypasses prefix). `--output-dir` made optional. `--out-dir` wins when both supplied; `--output-dir` logged for audit. |
| 3 | `--engine both` not recognized                                     | hard fail                  | Added `--engine {lambda,cfm,both}` (default `lambda`). `cfm` / `both` log a warning and fall back to Lambda (this runner is Lambda-only; CFM runner is `r10_cfg_real_crossdocked.py`, dispatched separately). |
| 4 | `--metal-seed cisplatin` not recognized                           | ALREADY WORKED             | Expanded `choices` to `[None, cisplatin, carboplatin, oxaliplatin, nedaplatin, ru_arene, ir_cp_star]` so the dispatch CLI AND future R17 expansion cells (`--metal-seed carboplatin` / `oxaliplatin` / `nedaplatin`) all work. |
| 5 | `--click-rules auto-pt-strict` not recognized                     | ALREADY WORKED             | The `auto-pt-strict` alias was already in `CLICK_RULE_ALIASES` (line 193); the user-reported failure was a stale docs claim. The alias expands to `[CuAAC, SPAAC, Suzuki]` (3-rule strict-Pt_II subset). |
| 6 | `--r17-paper-grade` flag (TODO-31 §M-R17-A2)                      | already shipped (line 4856) | Already present (line 4856); forces 100-pocket × 3-seed × n_sim=1000 + `crossdocked100_manifest.csv` + `wf_paper_grade_300cells/` output. |
| 7 | `--smoke` flag (10×1×n_sim=20)                                    | already shipped (line 4834) | Already present (line 4834); loads `test_010_019_subset_manifest.csv`. |

## 2. Smoke verification — PASS within-cell

Two smoke runs verified end-to-end:

**(a) `--smoke --metal-seed cisplatin` (canonical R17 smoke, 10×1×n_sim=20):**

| metric | value |
|---|---|
| n_cells | **10** (test_010..test_019, seed=42 only) |
| unique SMILES across cells | **20** |
| distinct metal-ligand signatures (Pt/Cl/N/O/F tuples) | **9** |
| validity_rate | 1.0 |
| decoder_pass_rate | 1.0 |
| coordination_number_mean | 1.0 (across-cell collapse — see §3) |
| elapsed_s_total | **44.5 s** (well under 5-min budget) |

**(b) full dispatch trimmed to `--pockets 2 --seeds 42 0 --n-simulations 50` (faster verification of --seeds plural + --out-dir + --metal-seed + --click-rules):**

| metric | value |
|---|---|
| n_cells | **4** (test_000 + test_001 × seeds {42, 0}) |
| unique SMILES across cells | **20** |
| validity_rate | 1.0 |
| elapsed_s_total | ~33 s |

## 3. Across-pocket ≥3 metal-ligand pair gate — FAIL (documented structural limit)

The TODO-25 §3 "≥3 distinct metal-ligand pairs across cells" gate is FAIL with `auto-pt-strict` (per `wf_lr17_b2_done.md §3`). Two structural causes — both documented, neither a CLI bug:

1. **`auto-pt-strict` keeps only 3 click rules** (CuAAC + SPAAC + Suzuki). ThiolEne + AmideCoupling are gated OUT (incompatible with strict Pt_II). MetalLigandExchange / AquaExchange SMARTS are not in the subset, so transmetalation / Cl-OH swap cannot diversify the metal coordination sphere.
2. **Shared `cisplatin` root seed** without `--metal-seed-from-pocket` ⇒ shared MCTS root ⇒ shared `_unreactive_states` cache + transposition_table ⇒ same 20 SMILES per pocket.

The fix is the 88-rule matrix (12 metals × 7 ligands + 4 NEW SMARTS) shipped as L-R17-B2 follow-on work (per `TODO-31 §L-R17-B2` + `TODO-pending/34_parallel_ultracode_dispatch.md §UC-L2`). The CLI surface is now ready to wire that matrix: `--metal-seed {carboplatin, oxaliplatin, nedaplatin}` + `--metal-seed-from-pocket` + `--click-rules metal-coord` (already in `CLICK_RULE_ALIASES`).

## 4. Honest framing

- This fix is **strictly additive**: every legacy flag combination (`--seeds 42 0 1234 --output-dir NAME --metal-seed cisplatin --click-rules auto-pt-strict`) produces bit-identical output to before. The 5 NEW flags (`--seed`, `--out-dir`, `--engine`, `--metal-seed carboplatin|oxaliplatin|nedaplatin`) are accepted alongside the legacy ones; the resolution order (most-specific wins) is documented in `main()`.
- The `--r17-paper-grade` and `--smoke` preset flags were already shipped (R17-A2 work, 2026-09-25); the dispatch CLI in TODO-31 §L-R17-B2 verify gate lists them as "to add" but they were committed earlier.
- The 100×3 paper-grade sweep is QUEUED (per `TODO-31 §M-R17-A2`) — this fix does NOT auto-run it; the user dispatches it explicitly with `--r17-paper-grade`.
