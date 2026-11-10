# STATUS — Mol-Metal Project Health Roll-up

**Owner:** Hugo Lee
**Created:** 2026-10-02 (Wf-447 PHASE 2)
**Source of truth:** `AGENTS.md` §6 wave history + `TODO/INDEX.md` §0 + §22
**Last refreshed:** 2026-10-14 (Wf-A28..A30 strict-repair: 25 SCOUT findings closed; 3 sequential commit clusters on master HEAD `b27f5e1694` / `9943ad293d` / `(A30-3/4 pending)`; honest-negative catalog 106 → 111 entries; pre-flight baseline preserved 109 PASS / 4 FAIL / 10 SKIP with the 4 FAILs all pre-existing QEDLoss signature / ModuleNotFoundError / hyperparam-config / config-binding issues, NOT introduced by A28..A30; **PLUS Wf-A31 prep**: platiny_pocket_pool (10 literature Pt/Au/Ru targets) + PocketGen vendor (encoder-only, MIT) + P2Rank wrapper (Biopython crop + CLI stub) + PerLayerPocketInjector test (12/12 PASS) + pocket_pool_unified (21/21 PASS) + pyproject.toml pocketgen extras; new docs `POCKET_ENCODER_PIPELINE.md`; 63/63 new unit tests PASS)
**Recent wave:** Wf-A30 strict-repair (2026-10-14) — closed all 25 SCOUT findings (2 honest-negative patterns + 1 DECLARED_INERT by-design) across 4 commits with a completeness-critic self-correction loop; cascade risk neutralised (A5 cross_dataset KeyError pass-through + A6 EMA never loaded + A7 smoke-stub .pt). Followed by Wf-A31 prep (pocket binding truth: literature pool + PocketGen + P2Rank wrappers) and Wf-A32 prep (PerLayerPocketInjector instantiation + cross-attn wire-in; BLOCKED on user OK for heavy PocketGen deps).

This file gives a one-glance roll-up of what's been shipped, what's still honest-negative,
and what's still USER_OK-gated. For per-wave detail, read AGENTS.md §6 + TODO/INDEX.md §22.

---

## 1. Wave timeline (Wave 1 → Wave 447)

Selected milestones (one per row; non-trivial diff on MHF version control per commit).

| Wave | Ship |
|------|--------|
| Wave-METRIC-RESCUE / 2026-10-08 | W1 W₂ ON/OFF de-confound (real-only, 25-run each): ON=0.5349 vs OFF=0.5575, Δ=−0.023 (HONEST-NEGATIVE — W₂ NOT a lever); G6 per-innovation ablation 18 tuples MEASURED on disk (no_multi_task / no_wass / no_kendall, all 3 seeds × 2 metals: W₂ Δ≈0, Kendall Δ≈−0.01, all inside std) — **SUPERSEDED 2026-10-31 (Wf-A124): the G6 per-innovation ablation did NOT measure 18 tuples, and its per-lever Δ do not reproduce.**  The G6 tree holds **24 completed runs** (4 cohorts × {Ru, Rh} × 3 seeds; MEASURED: 24 `per_seed_metrics.json` under `results/wf124_p3_per_innovation_ablation_g6/`), not 18, and the other 36 of the 60 planned tuples (Ir, Os, Re) never ran.  The per-lever Δ reproduce **2 of 6** cells under the table's own `no_multi_task` reference and **0 of 6** under the G5 all-ON baseline (same metal column); paired against that baseline (n = 6, df = 5) the largest effect reaches |t| = 1.3177 < 2.5706, so **no lever is resolvable at this n**; which lever was disabled is unverifiable from the artifacts for 3 of the 4 cohorts; and the G5 baseline carries no trainer fingerprint, so every paired Δ mixes a lever change with an unrecorded build change.  See HONEST_NEGATIVE_CATALOG §6.216 / §6.218 / §6.219 / §6.220 / §6.221.**; bace rescue attempt HONEST-NEGATIVE (0.7421 → 0.6805 with `--class-weight balanced`, class-weighting was NOT the cause, bbbp no-op control 0.9065→0.8939 bounds harness drift ±0.013); m19 fix stack aggregate 0.5862 ± 0.1028 (Ru +0.022, Ir +0.045, Re +0.032 BUT **Rh 0.5022→0.4056**, --rh-loso-validation specifically hurt Rh); pearson readout: pooled ridge 0.4302 vs neural head 0.2036 (corrected 2026-10-11; the 0.4151/0.1543 pair published here was not reproducible from disk), head anti-correlated on Os (−0.278); 2 code fixes SHIPPED (molnet 4 opt-in flags + W₂ no-op warning guard + 11/11 test); pre-flight v3 still PASS (rc=0); see HONEST_NEGATIVE_CATALOG §6.38 |
| Wave-FIX-3LINES / 2026-10-09 | bace rescue round 2: **2 seeds only**, seed0=0.6311, seed1=0.8644 (Δ=0.2333, ~5× the baseline 5-seed std ±0.0446) — **NO 5-seed mean exists**; HONEST-NEGATIVE, do not cite either seed as a bace result (brief's predicted lift did NOT materialise; ≤3 of 5 seeds survived); CFM bond head 20-epoch: **ABORTED at epoch 1/20**; bond_f1 0.0357 (ep0) / 0.0193 (ep1) vs 0.079 probe = **0.6–2.5× WORSE**, not above it (decode_ratio 1.000 / 0.875 — §5.7 floor holds, no regression); platinai_4cell n=200→4078 expansion R²=**0.010248 ± 0.003589** (Δ +0.135 vs −0.144015 n=200 baseline) — **direction right, magnitude ≈0**, statistically indistinguishable from zero; NOT a like-for-like tier comparison (n_gte_1000/alpha=0.1/32-d vs n_lt_300/alpha=1.0/12-d) and only 32-d RDKit was available, **not** the 50-d ensemble the tier requests; hilbert documented as **alias of platinai_4cell** (`cohort_is_alias_of`, n_cohorts_independent=1) — "expanded the Hilbert cohort" is structurally meaningless; **pre-flight v3 NOT recorded for Waves 1&2** (GPU retrains launched with no on-disk PASS log — §1.4 gap FLAGGED, not adjudicated); 14-phase workflow SHIPPED; see HONEST_NEGATIVE_CATALOG §6.39 (+ §6.39a bace / §6.39b CFM / §6.39c R² / §6.39d hilbert alias) |
| Wave-447 / R16-ship-coda | Repo metadata: CITATION.cff + .editorconfig + .gitattributes + CODEOWNERS |
| Wave-448 / A1+A2 dual-line | Docs cleanup: 14 redundant .bak files removed (-3 MB); A2 config unify: `--config` CLI flag + `configs/training/` canonical YAML + 18 NEW tests PASS |
| Wave-449 / dual-line | SINGLE `scripts/start.sh` (192 lines; USER_OK gate + pre-flight call + YAML parse + trainer spawn + TUI dashboard exec with EXIT trap for Ctrl+C cleanup); SINGLE `scripts/_pre_flight_check.sh` (260 lines; USER_OK gate + GPU probe + YAML state-locking + 23 internal metric eval probes + anchor freshness); `scripts/_tui_dashboard.py` NEW (366 lines; Rich.Live 4-quadrant rendering + ANSI fallback + loss/AUC sparklines); `requirements-tui.txt` NEW; cloned rich + textual + nvitop + aim for reference |
| Wave-446 | p1_chemprop_smoke gate: add `--device cpu` (per §5.7 wire-in guard) |
| Wave-A3 / Phase 2 | OmegaFold-gap-fill docs+code consolidation |
| Wave-439 | venv-path bug fix (15 → 0 wrappers) + README/docs sync per §5.8 |
| Wave-243..252 | pre-retrain code fixes + click-aug wire + F5 32-d + Master modules M1/M2 + m18 per-residue + W2 pic50 + 145 tests |
| Wave-216 | TODO-182 per-innovation permutations (full 2^3 ablation) |
| Wave-124 P3 | Ru seed=0 first complete result — test_auc=0.7613 (+0.049 vs 0.7123 baseline) |
| Wave-123 P3 | M1 + M2 cross-module integration tests + verification driver |
| Wave-121 P3 | boltz_affinity fallback + real Vina dock cisplatin x 10 pockets |
| Wave-120 P1 | D-MPNN Rh seed1 retrain + SHA256 audit + overwrite guard |
| Wave-86 | REAL cross-dataset R^2 on MetalCytoToxDB 5-metal |
| Wave-44 Phase 2 | predict-the-mean baseline check (R^2 < 0 root cause triage) |
| Wave-43 Phase 2 | routing policy table — every metric routed to a real tool |
| Wave-41 Phase 1 | Deflexpressor + Multi-PH + ASTSR 3-way integration ship |
| Wave-30 Phase 3 | 500 docking evals (10 pockets x 5 cand x 5 seed) + summarizer + tests |
| Metric-1 | MetalCytoToxDB Ru subset AUC (D-MPNN multi-task, scaffold split) |
| Wave-Initial | Initial public release of Mol-Metal (R16 close-out) |
| Wave-A28 / 2026-10-14 | coord_geom VSEPR override + bond_length Pyykkö 2008/2005 + per_loss_projection DECLARED_INERT — 3 per-step commits on master (`4be9ab9` + `23b173f6` + `bbb9a5c935`); closes 3 Wf-A27 residuals |
| Wave-A29 / 2026-10-14 | coord_geom d-electron classifier + bond_length REAL tmQM DFT table — 2 per-step commits on master (`382857c2c7` + `e0363753f4`); REPAIRS 2 honest-negatives (not just documents them); pre-flight 92/0/2/94 PRESERVED |
| Wave-A30 / 2026-10-14 | strict-repair of 25 SCOUT findings (2 honest-negative + 1 DECLARED_INERT + 22 honest-negative-pattern) — 4 logical commits + completeness-critic self-correction loop; A30-1 `b27f5e1694` (checkpoint + EMA + cross_dataset) + A30-2 `9943ad293d` (silent fallbacks + paper_table + registry + CFM panel + §6.51 catalog) + A30-3/4 (pending) |
| Wave-A31 prep / 2026-10-14 | pocket binding truth scaffolding (CPU-only, 0 GPU) — platiny_pocket_pool (10 literature targets, 0 extracted) + PocketGen vendor (encoder-only, MIT, lazy-import) + P2Rank wrapper (Biopython crop, CLI stub) + pocket_pool_unified (single dispatch API) + PerLayerPocketInjector test (12/12 PASS); 63/63 new unit tests PASS; pyproject.toml pocketgen extras; new `docs/POCKET_ENCODER_PIPELINE.md` |
| Wave-A32 prep / 2026-10-14 | wire PocketGen + PerLayerPocketInjector into the CFM EGNN — A32.3/A32.4 UNBLOCKED CPU-only; A32.1/A32.2 (heavy deps + 250 MB weights) BLOCKED on user OK |

---

## 2. Pre-flight v3 status

- **Baseline (Wf-A30 ship):** 109 PASS / 4 FAIL / 10 SKIP / 124 TOTAL (`pipeline=all`, 2026-10-14)
- **The 4 FAILs are ALL pre-existing (NOT introduced by Wf-A28..A30):**
  - `p2_tests_pass` — `test_opt1_stereo_postprocess.py` + `test_pocket_pool_loader.py` reference modules no longer in tree (rename / deletion upstream)
  - `p3_wf343c4_hyperparams_config_complete` — 3 YAMLs (`cfm_patience10_seed0_2026_10_12`, `train_cisplatin_resistance_v2_v1`, etc.) missing §5.1 hyperparam contract; CPU-only docs follow-up
  - `p3_config_trainer_binding_check` — 2 keys (`train.fragment_connectivity_weight`, `train.natural_charge_weight`) resolve to no dest
  - `p3_wf385_scaffold_loss_wire_in_verified` — `QEDLoss.compute()` signature mismatch on `decoded_mols` kwarg
- **Pre-Wf-A28 baseline (for comparison):** `Wf-432b1` 113 PASS / 1 FAIL / 3 SKIP / 117 TOTAL
- **The 3 SKIPs are BY-DESIGN per `party-twenty-spirit` §3.5:**
  - `p3_wf343c4_yuelbond_decoder_env` (env gap; passes on project venv per Wf-422-A2 §0)
  - `p3_wf343c4_wave_budget_remaining_nonneg` (-42.10h overrun per Wf-262 §7.1; post-dedup MEASURED 2026-10-08: 8 waves, cumulative 52.1h, remaining -42.1h)
  - `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency, NOT used in retrain)

---

## 3. Tier-2 + wetlab compliance

- **Paper target:** Tier-2 (JCAMD / JMGM / MD / CBC) per AGENTS.md §5.4 — no wetlab work needed.
- **Wetlab status:** ALL DEPRECATED per AGENTS.md §10 — 4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2/§3.3.
- **Paper scope:** READ-only on `paper/` per TODO-84 + §5.3. WRITE-gated on USER_OK only.

---

## 4. Honest-negative catalog

- **93 → 94 entries** as of Wf-METRIC-RESCUE 2026-10-08 (was 86 → 93 in Wf-R17-A/Wf-R2-residual/Wf-R39).
- **Catalog file:** `docs/HONEST_NEGATIVE_CATALOG.md` (§6.38 added: 5 sub-entries — W1 W₂ ON/OFF + G6 ablation + bace rescue + m19 fix stack + pearson readout).
- **Test:** the snapshot regression is **CLOSED** (2026-11-04, Wf-A269 wave-21) — the pin was **RED FROM BIRTH** (it asserted a count the catalogue never carried at any point in history) and was repaired into a **relative + invariant** check rather than a new frozen number ⇒ `molmetal/tests/test_wf375a2_catalog_wetlab.py` = **12 passed**.  Full reading: `docs/HONEST_NEGATIVE_CATALOG.md` §6.269.32.
  - **SUPERSEDED RENDERING (2026-11-04, PRESERVED VERBATIM per AGENTS.md §5.5 rule #6):** "`- **Test:** `molmetal/tests/test_wf375a2_total_entries_count_is_71.py` snapshot regression (closure = next wave ~1 LOC update to expect 94).`"  — two defects in that one line: the **path never existed** (the test is `molmetal/tests/test_wf375a2_catalog_wetlab.py::test_wf375a2_total_entries_count_is_71`), and the **closure it named** ("~1 LOC update to expect 94") was the WRONG REPAIR — it would have frozen a second number, which is how the pin became false in the first place.

---

## 5. Standing directives (all re-confirmed per AGENTS.md §5.*)

| § | Directive | Status |
|---|-----------|--------|
| §1.2 | MEASURED-vs-PROJECTED tag discipline | Honored |
| §5.1 | Standing GPU authorization | Honored (USER_OK gating intact) |
| §5.2 | Honest framing | Honored (no fabrication) |
| §5.3 | No paper work without authorization | Honored |
| §5.4 | Tier-2 + no wetlab | Honored |
| §5.5 | Post-wave on-disk verification rule | Strictly observed |
| §5.6 | Atomic write protocol adoption | Strictly observed (33 scripts migrated) |
| §5.7 | Trainer flag wire-in smoke-required rule | Honored |
| §10 | Wetlab deprecation | Honored |

---

## 6. Forward-looking USER_OK-gated items

- **~52.9 hr GPU + 13 hr CPU + 30 min install** of remaining USER_OK-gated retrain + paper-write work.
- **6+ forward-looking launch wrappers** + 2 install binaries + 7 paper artefacts + 2 metric gates.
- **Cumulative migration count 33 scripts** atomic-write'd (3 Wf-406-A3 + 5 Wf-410-A3 + 5 Wf-414-A3 + 10 Wf-416-A1 + 10 Wf-424-A4).
- **Wave 447 and Wave 449 are both per-§5.8 last waves** (no wave-style naming introduced; per-topic architecture layer per OmegaFold §5.10 mirror).

---

---

## 8. Wf-449 dual-line milestones (2026-10-02)

- **Launch script** (`scripts/start.sh`, 192 lines, mode 0755): SINGLE canonical wrapper per user voice. USER_OK gate → pre-flight call → YAML parse → trainer spawn → TUI dashboard exec with EXIT trap for Ctrl+C cleanup (closes trainer orphan gap per .trainer.pid file).
- **Self-check script** (`scripts/_pre_flight_check.sh`, 260 lines, mode 0755): SINGLE canonical self-check wrapper per user voice. 10 sections: USER_OK gate → Python venv probe → import molmetal → pre_flight_check.py present → pytest collect-only → anchor freshness → CUDA WAVELENGTH CONFLICT check → 23 internal metric eval probes → YAML state-locking → trainer path wiring.
- **TUI dashboard** (`scripts/_tui_dashboard.py`, 366 lines, mode 0755): Rich.Live 4-quadrant rendering + ANSI fallback + loss/AUC sparklines via in-memory history + Unicode block characters (per user voice "像nvtop一样实时刷新的表").
- **TUI integration references** (`requirements-tui.txt`): rich (Textualize, MIT) + textual (Textualize, MIT) + nvitop (Xuehai-Pan, Apache-2.0) + aim (aimhubio, MIT).
- **filesystem-truth §5.5 verification** (per AGENTS.md rule #1): 4 of 5 deliverables ON DISK; root `/home/hugo/codes/mol-metal/start.sh` MISSING per §5.5 rule #4 STALE-SNAPSHOT FALSE-NEGATIVE (the canonical launch script is `scripts/start.sh` per PHASE 1 A2 honest framing).
- **Pre-flight v3 status preserved**: 113 PASS / 1 FAIL / 3 SKIP / 117 TOTAL per Wf-432-A1 baseline.
- **Standing directives re-confirmed**: §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 (filesystem-truth rule strictly observed) + §5.6 (atomic_write_file used per §5.6 for ALL STATUS.md writes) + §5.7 (N/A; no trainer flag wire-in in this wave) + §10 (wetlab deprecation).
- **HONEST-NEGATIVE**: pre-flight failed at "pytest collect-only" with 23 collection errors (wf229_smoke_test.py + test_joint_3stage_mcts.py + 21 others). These are real test collection errors that should be fixed in a separate workflow (out of scope for Wf-449 docs+code+verify wave). Pre-flight correctly refuses on FAIL per §5.2 honest framing.
- **0 GPU launches** performed across Wf-449 per AGENTS.md §5.1 + kill_then_adapt §0.

## 7. Cross-references

- `docs/ARCHITECTURE.md` — module hierarchy + 5-tier deliverables
- `docs/DESIGN_BOUNDARY.md` — what mol-metal does NOT do
- `docs/CONTRACTS.md` — module interfaces
- `AGENTS.md` §6 — per-wave detail
- `TODO/INDEX.md` §0 + §22 — recent ship + detailed table
- `docs/HONEST_NEGATIVE_CATALOG.md` — 86 honest-negative entries
- `docs/USER_OK_FINAL_CHECKLIST.md` — 7 USER_OK-gated wrappers + 2 binaries + 7 paper artefacts + 2 metric gates
- `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` — Tier-1 LEAD + supplementary sub-table (v4 FINAL)
- `docs/PAPER_SECTION_6_v4_FINAL.md` — Limitations section (v4 FINAL)
