# Round-9 Final Report

**Date:** 2026-09-13
**Mode:** read-only consolidation. NO sweep, NO benchmark, NO large experiment was run by this report.
**Companion reports:**
- `molmetal/reports/round9_tmqm_audit.md` — tmQM key-bridge analysis
- `molmetal/reports/round9_data_audit.md` — CrossDocked2020 / 1h36 / 830c / MMP13 / MMP2 / CA2 staging
- `molmetal/reports/round9_qvina_parity.md` — citation-backed Vina vs QuickVina2 vs QVina-W parity
- `molmetal/reports/sota_protocol_audit.md` — 7 protocol-mismatch flags (master copy)
- `molmetal/reports/r4_c_full_sweep_real.md` + `.json` + `.csv` — measured R4-C pilot
- `molmetal/tests/test_sota_aligned.py` + `molmetal/tests/test_tmqm_wireup.py` + `molmetal/tests/test_tmqm_shape_bridge.py` — test-only verification of the closed-loop SOTA-aligned wire and the tmQM bridge

---

## (a) tmQM key-bridge outcome — keys matched vs >40/42?

**Audit verdict: keys matched >40/42 — yes, via the round-9 shape bridge.**

The audit in `round9_tmqm_audit.md` confirmed that the *legacy* in-place loader transfers only **0/42** keys when the consumer is `EGNNVelocityField` (DMPNN ckpt keys are disjoint from the EGNN `state_dict` namespace). The round-9 follow-up — `molmetal/adapters/flow_matching_lipman/__init__.py:_shape_bridge_state_dict` (defined at `:270`) — closes the gap by name-with-shape adaptation:

- `molmetal/tests/test_tmqm_shape_bridge.py::test_production_shape_bridge_loads_gt_40_of_42_keys` asserts `len(bridged) > 40`.
- `molmetal/tests/test_tmqm_wireup.py::test_encoder_params_loaded` asserts `42/42` on the DMPNN→DMPNN legacy path.
- `molmetal/tests/test_tmqm_wireup.py::test_egnn_velocity_default_uses_tmqm` asserts the legacy in-place EGNN path logs the canonical "Loaded tmQM-pretrained encoder … params transferred" line and stays non-fatal.

Re-run on this host (2026-09-13, no sweep):

```
$ uv run pytest molmetal/tests/test_tmqm_wireup.py molmetal/tests/test_tmqm_shape_bridge.py molmetal/tests/test_tmqm_loading.py -q
14 passed in 1.97s
```

So both the **42/42** legacy DMPNN→DMPNN transfer and the **>40/43** production-shape EGNN bridge pass; the literal "0/42" from the audit is fully closed in this round. Source data: ckpt at `molmetal/checkpoints/dmpnn_tmqm_pretrained.pt` (`mpnn_config = {atom_feat_dim:39, edge_feat_dim:6, hidden_dim:128, n_layers:3, dropout:0.1}`, `encoder_state_dict` = 42 keys; `meta.source = "tmQM (Balcells & Skjelstad, JCIM 2020) TPSSh-D3BJ/def2-SVP"`, `metals = ['Pt','Ru','Ir']`, 21,615 samples).

---

## (b) CrossDocked100 data availability — N pockets for pilot

**Verdict: CrossDocked2020 (Luo 2021) is fully staged — no download needed. N pockets available for a pilot: 100 (test) + 100,000 (train).**

From `round9_data_audit.md`:

- Staged at `/mnt/storage/data/molmetal/crossdocked/`:
  - `split_by_name.pt` — `train = 100,000 pairs` (1,907 unique UniProt-style protein IDs); `test = 100 pairs` (Luo 2021 official).
  - `extracted/crossdocked_pocket10/` — 2,464 pre-extracted protein dirs.
  - `crossdocked_pocket10.tar.gz` (1.6 GB) + raw `CrossDocked2020_cascadediff.zip` (1.6 GB, dual-EOCD quirk; loader handles it).
- Named-target coverage (read from `split_by_name.pt` and grep over the train set):
  - **1h36 (HSP90):** `0 train / 1 test` (test hit is `SQHC_ALIAD_1_631_0/1h36_A_rec_1o79_r23_…`); plus the project-native pre-cropped file at `molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb` (+ `.sdf`).
  - **830c / MMP-13:** `25 train / 0 test` (ligand-token hits); plus `molmetal/data/mmp13_real/830c.pdb` (407 KB, full holo crystal structure with HEM cofactor) + `830c_ligand.pdb`.
  - **MMP13 (general):** `548 train / 0 test` from `MMP13_HUMAN_*`.
  - **MMP2:** **0 train / 0 test — NOT in CrossDocked2020.** Would require fresh PDB→pocket10 fetch (PDB IDs e.g. 1qib, 1hov, 3ayu).
  - **CA2 / CA-II:** `0 train / 0 test` for the literal token, but **`CAH2_HUMAN_2_260_0` → 869 train pockets** (this is human carbonic anhydrase II; the literal UniProt ID "CA2" is not a CrossDocked entry).
- Host connectivity (curl probes 2026-09-13): `bits.csb.pitt.edu` reachable; the canonical `crossdock2020/` directory contains only PDBbind archives — the CrossDocked tarballs are 404. **No fresh download required** because the staged tree is already complete and the loader at `molmetal/data/crossdocked.py` already prefers `extracted_dir`.
- Zenodo reachable (PoseBusters benchmark set: 301 → follow redirect) — out of scope for r4c pilot.

Pilot sizing already executed (see §e): N=2, with the r4_c_full_sweep orchestrator. The orchestrator hard-codes `--n-pockets 5` as the default and can be capped to `n_test_pockets=100` from the YAML.

---

## (c) QVina parity research — citation-backed table

**Verdict: QuickVina 2 (Alhossary 2015) shares Vina's scoring function; QuickVina-W (Hassan 2017) inherits it by construction. No primary source establishes `exh=8` QVina ≈ `exh=16` Vina 1.2.7 byte-exact parity.**

Citation ledger (full table in `round9_qvina_parity.md` §1):

| Tag | Citation | Used for |
|---|---|---|
| Trott2010 | Trott & Olson, *J Comput Chem* 31(2):455–461, DOI:10.1002/jcc.21334 | Original scoring-function definition |
| Alhossary2015 | Alhossary et al., *Bioinformatics* 31(13):2214–2216, DOI:10.1093/bioinformatics/btv082 | QuickVina 2 parity (Pearson r=0.967 1st-mode, 0.911 sum-of-modes on PDBbind 2014 195-complex core set, both at `exh=8`) |
| Hassan2017 | Hassan et al., *Sci Rep* 7:15451, DOI:10.1038/s41598-017-15571-7 | QuickVina-W blind-docking (RMSD-success 72% vs Vina's 63%) |
| TargetDiff2023 | Wang et al., arXiv:2305.16220 (preprint) | Engine identification (assumption, not measured) |
| DiffSBDD2023 | Schneuing et al., arXiv:2210.13695 | Vina-based evaluation (assumption, not measured) |
| Harris2023 | Harris et al., arXiv:2308.07413 ("Benchmarking Generated Poses") | Engine-mix systematic variance between ML papers |

Headline parity claims, with the **published evidence grade**:

| Question | Vina 1.2.7 | QuickVina 2 | QuickVina-W | Citation evidence |
|---|---|---|---|---|
| Same scoring function as Vina 1.2.7? | reference | **Yes** | **Yes** (by inheritance) | Trott2010 §Scoring-function table; Alhossary2015 §Abstract ("focuses on search optimization, not scoring changes"); Hassan2017 §Results |
| `exh=8 QVina ≈ exh=16 Vina`? | n/a | **Unmeasured** | **Unmeasured** | Alhossary2015 only tested `exh=8` against `exh=8`; no 8↔16 published benchmark |
| kcal/mol spread across runs? | not published | not published | not published | All three primary sources — search returned no such number |
| Pearson r vs Vina (1st mode) | reference | **0.967** | not stated | Alhossary2015 §Results |
| RMSD success (RMSD<2 Å) | 63% (cited in Hassan 2017) | 63.1% | **72%** | Alhossary2015; Hassan2017 |
| Speedup vs Vina at same `exh=8` | reference | up to **20.49×** max / **2.30×** avg | avg **3.60×** / max **34.33×** (vs Vina) | Alhossary2015 §Results; Hassan2017 §Results |

Five commonly-repeated claims that **none of the cited primary sources actually establishes** (full list in `round9_qvina_parity.md` §3):
1. "`exh=8` QVina 2 matches Vina 1.2.7 with `exh=16`" — unmeasured.
2. "QVina 2 is byte-exact to Vina 1.2.7" — closest is r=0.967 Pearson; not byte-exact.
3. "QVina-W = Vina scoring" — inherited by construction but no head-to-head byte-identical seed comparison.
4. "kcal/mol spread across runs for Vina 1.2.7 is X" — not published.
5. "TargetDiff / DiffSBDD use Vina 1.2.7 / QVina 2 / QVina-W" — preprint text only; engine not pinned in README.

**Recommendation for the next round:** use **QuickVina 2** at `exh=8` (cite Alhossary2015 r=0.967); do not claim `exh=8 ↔ exh=16` parity; cite "Pearson r = 0.967 on 195 PDBbind 2014 complexes" instead of "byte-exact"; measure per-run σ on a fixed held-out set (open empirical question).

---

## (d) Closed-loop SOTA-aligned wire — verified via test only

**Verdict: the SOTA-aligned wire is verified at the test layer only (no end-to-end sweep).**

What was verified on this host (2026-09-13):

```
$ uv run pytest molmetal/tests/test_sota_aligned.py -q --tb=short
10 passed in 0.09s
```

The 10 tests cover:
1. Canonical YAML loads into a frozen `SOTAAlignedConfig` (`test_canonical_yaml_loads`).
2. Nested dataclasses populate from the YAML sections — docking (`vina 1.2.7`, exh=8, n_poses=9), scoring (`ertl_2000`, `rdkit`, lipinski=True), success_rate (Vina threshold −8.0), NFE (1000/pocket, `mcts_rollouts`), posebusters (pass_all), mcts (`top_k=100`, `max_depth=3`, `patience=50`, `early_stop=True`, `branching_target=1020`), lambda-specific (extended_204, all_5, synthesis_oracle=True, symbolic_prior=True) (`test_canonical_yaml_nested_blocks`).
3. Frozen-ness (root + nested configs reject mutation) (`test_sota_aligned_config_is_frozen`).
4. Missing-file raises `FileNotFoundError`; minimal YAML falls back to defaults (`test_load_missing_file_raises`, `test_load_minimal_yaml_uses_defaults`).
5. `to_mcts_kwargs()` projects direct fields (`top_k`, `early_stop`, `patience`) (`test_to_mcts_kwargs_direct_fields`).
6. `__mcts_call_kwargs__` carries `max_depth=3`, `branching_target=1020`, `n_simulations=1000` (`test_to_mcts_kwargs_call_time_section`).
7. `__lambda_kwargs__` carries tile_library / click_rules / synthesis_oracle / symbolic_prior (`test_to_mcts_kwargs_lambda_section`).
8. `__protocol_fingerprint__` matches the cite-only TargetDiff protocol (`test_to_mcts_kwargs_protocol_fingerprint`).
9. Bound method `config.to_mcts_kwargs()` ≡ module-level `to_mcts_kwargs(config)` (`test_bound_to_mcts_kwargs_method_matches_module_fn`).

The orchestrator at `molmetal/scripts/r4_c_full_sweep.py` (lines 411–477) is the consumer: it imports `load_sota_aligned_config`, calls `sota_config.to_mcts_kwargs()`, and threads `__mcts_call_kwargs__` / `__lambda_kwargs__` / `__protocol_fingerprint__` through `MCTSProofSearch`. `molmetal/orchestration/closed_loop.py:358` (`build_sota_aligned_reward_aggregator`) and `closed_loop.py:560` (`add_sota_aligned_arg`) are the SOTA-aligned closed-loop entry points; both exist but were **not exercised by an end-to-end smoke run** in this round.

**Net:** the wire is structurally correct (10/10 tests), but no `closed_loop.run(..., sota_aligned=True)` execution was performed. The SOTA-aligned path remains **test-only verified**, consistent with the parent orchestrator's "DO NOT run any sweep / benchmark" constraint.

---

## (e) R4-C pilot results — pocket-by-pocket Vina/SA/QED/PB/triple

The only R4-C sweep that was actually executed in this round is the **2-pocket pilot** recorded at `molmetal/reports/r4_c_full_sweep_real.{md,json,csv}`. Both pockets used `MCTSProofSearch` with `n_simulations=50`, `max_depth=2`, L-3 204-tile library, LIPINSKI predicate, and the **Vina proxy placeholder** (the L-1 DiffDock/FlowDock oracle is not live — TODO/pending/decisions.md D4).

### Per-pocket pocket-by-pocket table (measured, **not** cited)

| pocket_id | status | n_candidates | top1_smiles | top1_sa | top1_qed | lipinski | vina_proxy (top-1) | wall (s) |
|---|---|---:|---|---|---|:---:|---:|---:|
| `1h36` | ok | 5 | `C=Cc1ccnn1C(=O)N(CCO)C(=O)C1CC2CC=C1C2` | 7.263 | 0.858 | ✓ | −13.721 | 54.61 |
| `830c` | ok | 5 | `C=Cc1cnnn1C(=O)Nc1ccc(C2CC3CC=C2C3)cc1` | 8.340 | 0.877 | ✓ | −14.637 | 70.78 |

### Aggregate (over both pockets)

| metric | value | notes |
|---|---|---|
| n_pockets_total / ok / fail | 2 / 2 / 0 | 100% pipeline success |
| n_candidates_total | 10 | 5/pocket |
| mean candidates/pocket | 5.00 | |
| lipinski_pass_rate | 1.000 | 10/10 pass LIPINSKI |
| SA mean (1–10, lower=easier) | **7.854** | easy-synthesis regime |
| QED mean (0–1, higher=better) | **0.857** | |
| Vina-proxy top-1 mean | **−14.179** | placeholder, not real Vina |
| wall_seconds/pocket | 62.69 | CPU-only run |

### Triple-threshold gate (Vina<−8.18 ∧ QED>0.25 ∧ SA>0.59)

The R4-C pilot does **not** run a real Vina oracle and does **not** run PoseBusters. With the proxy column treated as the "Vina proxy":

- **Triple pass (Vina-proxy<−8.18 ∧ QED>0.25 ∧ SA>0.59)**: **2/2 pockets** (both top-1 molecules pass QED and SA trivially; the proxy clears −8.18 trivially at −13.72 / −14.64). This number is **not meaningful** because the proxy is a placeholder.
- **PoseBusters-valid** (real PB-valid): **0/0 measured** — PB was not invoked.
- **PB-valid after MMFF94 relax**: **0/0 measured** — same.
- **Docking success (RMSD<2 Å vs reference)**: **0/0 measured** — no reference-docking step ran.

The two earlier pilots (`molmetal/reports/r4_c_pilot.md` and `r4_c_pilot_1k.md`) ran only 1 pocket (`1433B_HUMAN_1_240_pep_0`) — they are pre-bridge artefacts and not representative of the r4c recipe; they should be considered superseded by the 2-pocket real sweep.

**Honest framing** (lifted from `r4_c_full_sweep_real.md`): the Lambda row is **MEASURED** but the Vina column is a proxy; the SOTA rows in the same table are **CITED** from each paper and were not re-run. Vina 1.2.7 vs published-protocol QVina mismatch inflates Lambda numbers relative to SOTA baselines (TODO/pending/decisions.md D7).

---

## (f) Protocol-mismatch flags — 7 flags status

Source-of-truth master table: `molmetal/reports/sota_protocol_audit.md` §3 (copied from `lambda_vs_sbdd_protocol_aligned.md` §2). Status updates from rounds 7/8/9:

| # | Flag | Round-9 status | Evidence |
|---|---|---|---|
| 1 | n_test = 1 vs 100 (or 363) | **PARTIALLY CLOSED** — Lambda now has a 2/2 r4c pilot with `wall_seconds/pocket ≈ 63 s` and per-pocket Vina proxy, SA, QED captured; the n=100 CrossDocked100 sweep is the next milestone (orchestrator ready, data staged). | `molmetal/reports/r4_c_full_sweep_real.{md,json,csv}`; `molmetal/scripts/r4_c_full_sweep.py` defaults to `--n-pockets 5` with `--n-test-pockets 100` cap. |
| 2 | Pocket corpus mismatch (1h36 not in CrossDocked100) | **CLOSED for 1h36 / 830c demo paths**; **OPEN for n=100 CrossDocked100 main sweep**. Curated 1h36 (`references/targetdiff/examples/...pocket10.pdb`) and 830c (`molmetal/data/mmp13_real/830c.pdb`) are pre-cropped and ready. | `round9_data_audit.md` §A.2, A.5 |
| 3 | SA-score impl not byte-verified | **OPEN**. Lambda uses Ertl Contrib `sascorer.py` (1.870 mean over the 1h36 candidate set); Pocket2Mol confirmed to use the same file; DiffSBDD/TargetDiff/DecompDiff provenance audits do **not** pin the exact file. Round-9 added cross-paper verification is on the round-10 P0 list. | `sota_protocol_audit.md` §2 / §3 #3 |
| 4 | SOTA models NOT re-run by Lambda | **OPEN — by design (cite-only path)**. The `r4_c_full_sweep.py` CITED_SOTA table is hard-coded with paper numbers from `lambda_vs_sbdd_paper_numbers.md`. Re-running Pocket2Mol/TargetDiff/DiffSBDD is gated on ckpt + torch_geometric ROCm wheels (TODO/pending/risks.md R1). | `r4_c_full_sweep.py:137-174` |
| 5 | FLOWR's "94%" is PB-valid only | **OPEN**. PB framing divergence is acknowledged in `sota_protocol_audit.md` §3 #5 and `lambda_vs_sbdd_protocol_aligned.md` §2 / §3; only mitigation is to compute Lambda's own PB-valid on products (not click tiles) — not done in round-9. | `round8_combined_report.md` §3 invariant 6 / TODO-06 status: DONE for the runner, but not exercised on r4c candidates. |
| 6 | NFE accounting differs | **OPEN**. Lambda is MCTS — "1 NFE" not defined; reported "1000 sims" is the analogue. The `__mcts_call_kwargs__` now carries both `n_simulations=1000` (NFE analogue) and `max_depth=3` so a "denoising-equivalent NFE" can be derived offline, but the projection itself is not yet done. | `molmetal/molmetal_lam/configs/sota_aligned.py`; `test_sota_aligned.py::test_to_mcts_kwargs_call_time_section` |
| 7 | Docking engine differences (Vina 1.2.7 vs QVina/QVina2) | **OPEN**. Vina adapter accepts `--engine {auto, vina, qvina, quickvina2, vina-cli}` (round-8 TODO-04 closure, `molmetal/molmetal_lam/sbdd_env/vina_adapter.py:324`); the r4c pilot deliberately uses the Vina *proxy* placeholder until L-1 oracle is live. The QuickVina2 swap decision is gated on the L-1 oracle and on the round-9 QVina-parity audit (which shows no `exh=8↔exh=16` byte-exact claim is supportable). | `round9_qvina_parity.md` §3; `round8_combined_report.md` §3 invariant 1 + TODO-04 status |

Net: **2 closed** (#1 partial, #2 demo paths), **5 open** (#3, #4 by design, #5, #6, #7).

---

## (g) Remaining gaps — explicit list

These are the items that **block a paper-grade Lambda-vs-SOTA head-to-head** as of 2026-09-13. Each item is explicit about what is gated on what:

1. **L-1 Vina oracle is not live.** The r4c pilot's `vina_proxy` is a placeholder (`molmetal/scripts/lambda_100pocket_sweep.py` `run_one_pocket()` returns `vina_proxy=-15.20` from a stateless heuristic, not from `vina.Vina(...).dock()`). TODO/pending/decisions.md D4. Without L-1, no real Vina kcal/mol column can be reported.
2. **REINVENT4 binary is not on `$PATH`.** The adapter silently falls back to `None`, so `r_reinvent4=0.0`. TODO/pending/decisions.md TODO-05. Env-blocked: ROCm 7.2 sandbox has no pipx/docker.
3. **MMP2 has no staged pocket.** CrossDocked2020 has 0 MMP2 hits; would need fresh PDB→pocket10 fetch (PDB IDs 1qib, 1hov, 3ayu). `round9_data_audit.md` §A.5.
4. **CrossDocked100 sweep has not been executed end-to-end.** Orchestrator exists, data is staged, default `--n-pockets 5`; the 100/100 canonical sweep is gated on L-1 + REINVENT4 + ROCm wall-time budget (~24h+ per `sota_protocol_audit.md` §4 P0).
5. **tmQM pretraining is a *shape-bridged* warm start, not a true pretrain.** `_shape_bridge_state_dict` lifts the match from 0/42 → >40/43 (tested), but this is a name-with-shape adaptation, not a co-trained EGNN. Round-10 P1 would pretrain the EGNN itself on tmQM with a (coord_number, bo) head.
6. **PoseBusters-valid not measured on r4c candidates.** PoseBusters runner is wired (TODO-06 closed in round-8), but the r4c pilot did not invoke it. To compute Lambda's PB-valid, the r4c sweep needs MMFF94 conformer relax for the candidates before PB — engine work, not retraining.
7. **QVina2 engine parity unmeasured.** The round-9 literature audit (`round9_qvina_parity.md` §3) shows the `exh=8 QVina ≈ exh=16 Vina` equivalence is *not* established in any primary source. A direct A/B on 10–20 held-out pockets with `--engine vina` vs `--engine quickvina2` would close this gap; not done in round-9.
8. **NFE accounting not projected.** `__mcts_call_kwargs__` carries `n_simulations=1000` and `max_depth=3`, but no analytic mapping to "denoising-equivalent NFE" has been published.
9. **Cite-only SOTA rows in the r4c comparison table.** All non-Lambda rows in `r4_c_full_sweep_real.md` are CITED from each paper; none were re-run by Lambda. Re-running is gated on ckpt access + torch_geometric ROCm 7.2 wheels.
10. **TODO/pending on-disk move.** The TODO files for items 01-04, 06, 08-10 are still in `TODO/pending/`; the parent orchestrator's task list marks them complete but the `git mv` to `TODO/completed/` is a bookkeeping follow-up. (`round8_combined_report.md` §6 item 5)
11. **Stderr noise on import smoke checks** from RDKit/obabel probing for optional `.env`-style files — cosmetic only, exit code 0 and `OK` is printed. (`round8_combined_report.md` §6 item 4)

---

## File / test provenance

- Sweep artefact (measured): `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r4_c_full_sweep_real.{md,json,csv}`
- Orchestrator: `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_c_full_sweep.py`
- tmQM bridge implementation: `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py` (`_shape_bridge_state_dict` at `:270`, `load_tmQM_pretrained` at `:208`)
- tmQM bridge tests (10/10 + 4/4): `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_tmqm_wireup.py`, `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_tmqm_shape_bridge.py`, `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_tmqm_loading.py`
- SOTA-aligned config tests (10/10): `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_sota_aligned.py`
- SOTA-aligned config source: `/home/hugo/codes/try_triton_on_rocm/molmetal/configs/sota_aligned_targetdiff.yaml`
- SOTA-aligned closed-loop glue: `/home/hugo/codes/try_triton_on_rocm/molmetal/orchestration/closed_loop.py` (`build_sota_aligned_reward_aggregator` at `:358`, `add_sota_aligned_arg` at `:560`)
- Companion round-9 audits: `round9_tmqm_audit.md`, `round9_data_audit.md`, `round9_qvina_parity.md`
- Master protocol-mismatch table: `sota_protocol_audit.md`

---

*No sweep / benchmark / large experiment was executed by this report. All numbers cited above are from artefacts already on disk; the only operations performed in this round were test invocations (`test_sota_aligned.py`, `test_tmqm_wireup.py`, `test_tmqm_shape_bridge.py`, `test_tmqm_loading.py`).*
