# Wf-321D — Pre-flight gate consolidation audit (DEDUPE + MEASURED-only)

**Wf:** Wf-321D
**TODO:** TODO-321D
**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** SHIPPED (audit complete; new `p3_gate_count_consistency_audit` gate added to `all` pipeline)
**Standing-rule compliance:** AGENTS.md §1 (MEASURED-vs-PROJECTED), §5.2 (honest framing)
---

## §1 Scope

Pre-flight v2 (`molmetal/scripts/pre_flight_check.py`) gate count has grown
across 12+ workflows since Wf-259:

* Wf-306C (P3 frontier-math wire-in + 2026-09-29 follow-ups)
* Wf-308A (smoke-test FAIL-on-missing global audit)
* Wf-308B / Wf-312C / Wf-315A (ADMET pass-through druglikeness gates)
* Wf-312B (NCI-60 9-cell per-cell aggregate)
* Wf-312E (4-LOI send-ready)
* Wf-313C (Day 3-5 integration tests + paper refs)
* Wf-313H / Wf-320F (R20 Phase 0 loaders closeout)
* Wf-314E (OOD 8/30 metals 0.0 reachability per-metal diagnosis)
* Wf-315D / Wf-319C (PoseBusters packages status)
* Wf-319A (ablation_runner tests + aggregate.csv)
* Wf-319B (UFF-min Pt-pool presence)
* Wf-321A (Wf-135 Phase 2 CPU scaffold readiness)
* Wf-321E (ABLATION_TABLE registry mirror check)
* Wf-321H (TODO-119 P1.4 CPU scaffold)
* Wf-322F (Wf-176 m19 LOMO Rh oversample final closeout documented)
* Wf-323F (m4 SA/QED sub-criterion documented)
* Wf-325D (m6_kappa BELOW threshold + NEGATIVE_WEAK MetalCytoToxDB
  validation FINAL closeout documented)
* Wf-325E (ADMET 4-below documented gate)
* Wf-325F (Phase3G F2A runtime unblock verified gate)
* Wf-327 (`p3_wf327_cpu_num_workers_sufficient` launch-script
  cpu-num-workers ≥ 4 regression gate — Wf-327b vs Wf-328-W2/W6)

This audit (a) tabulates the per-pipeline gate counts, (c) identifies
overlapping coverage, (d) verifies every gate is MEASURED-only, and
(e) adds a guard gate that detects future count drift.

---

## §2 Gate inventory (per-pipeline; counts synced 2026-10-03, HOLE-1 + HOLE-5 + L4 + L5)

### §2.1 `all` pipeline — 133 gates

> **Count synced 2026-11-04 (Wf-A269 wave-19): 132 → 133.** MEASURED
> 2026-11-04 by live import: `len(build_gates("all")) == 133`,
> `len(build_gates("launch")) == 135`,
> `EXPECTED_GATE_COUNTS["all"] == 133` and
> `EXPECTED_GATE_COUNTS["launch"] == 135` — the code constant and the
> live count agree with each other. The bump was made by registering
> `p3_script_contract_check`, the FOURTH §5.11 gate: the three gates
> before it anchored the "does the flag reach a parser that DECLARES
> it?" question to the `^train_` name prefix
> (`scripts/_launch_contract_gate.py:189`), so a repo-local script whose
> name is NOT `train_*` — a runner, a driver, a smoke helper, a
> scaffold — was NEVER checked. MEASURED 2026-11-04 by the gate itself:
> 101 wrappers / 218 invocations / 560 flags / 0 undecidable / 1
> disclosed quoted-blob site; it found 2 live violations
> (`molmetal/scripts/smoke/smoke_lambda.sh` passed `--out-dir` to a
> parser declaring `--output-dir`; a GPU launcher passed `--out` to a
> parser declaring `--json-dir`), both REPAIRED. `cfm` (8), `frontier`
> (25) and `chemprop_v2_v3` (18) are unchanged.
>
> **The superseded rendering is preserved VERBATIM below**, per
> AGENTS.md §5.5 rule #6 (the 2026-11-04 Wf-A269 wave-5/6 note, which
> is the reading this note supersedes).

> **Count synced 2026-11-04 (Wf-A269 wave-5/6): 129 → 132.** MEASURED
> 2026-11-04 by live import: `len(build_gates("all")) == 132`,
> `len(build_gates("launch")) == 134`,
> `EXPECTED_GATE_COUNTS["all"] == 132` and
> `EXPECTED_GATE_COUNTS["launch"] == 134` — the code constant and the
> live count agree with each other. The bump was made by registering
> `p3_runtime_transport_check`, the third §5.11 gate: the two gates
> before it read the wrapper's **literal invocation lines**, and a flag
> that is assembled at RUN TIME (`scripts/start.sh`'s `EVENTS_FLAG`,
> built from a YAML key and interpolated into `$TRAINER_CMD`) is
> invisible to both. MEASURED 2026-11-04 by the gate itself: 2 runtime
> strings, 14 transport sites, **2 live violations** — a
> `--no-events-jsonl` transported to the CFM trainer (which did not
> declare it) and a `--config` transported to
> `molmetal/scripts/bench_cfm_decode_real.py` (which declares no such
> flag). Both would have aborted the launch with argparse's
> "unrecognized arguments" before any work was done; both are
> REPAIRED. `cfm` (8), `frontier` (25) and `chemprop_v2_v3` (18) are
> unchanged.
>
> **The superseded rendering is preserved VERBATIM below**, per
> AGENTS.md §5.5 rule #6 (the 2026-11-01 Wf-A154 note, which is the
> reading this note supersedes).

> **Count synced 2026-11-01 (Wf-A154): 127 → 129.** MEASURED 2026-11-01 by
> live import: `len(build_gates("all")) == 129`,
> `len(build_gates("launch")) == 131`,
> `EXPECTED_GATE_COUNTS["all"] == 129` and
> `EXPECTED_GATE_COUNTS["launch"] == 131` — the code constant and the live
> count agree with each other. The bump was made by Wf-A154 adding the two
> SSOT-document consistency gates `p3_ssot_path_reference_check` and
> `p3_ssot_census_claim_check` to the `all` pipeline. Those are the FIRST
> gates that make a **document** defect able to FAIL a pre-flight: before
> this, a commit that introduced a broken repo-path citation or a stale
> census statistic still passed, because both audits existed only as pytest
> modules (`molmetal/tests/test_wfa131_repo_path_reference_guard.py` and
> `molmetal/tests/test_wfa152_census_guard_coverage.py`). Each gate runs a
> thin runner script (`molmetal/scripts/wf_a131_path_reference_guard_runner.py`
> and `molmetal/scripts/wf_a152_census_claim_guard_runner.py`) that imports
> its guard module and calls that module's OWN assertions, so the rule has
> ONE implementation and cannot fork. `cfm` (8), `frontier` (25) and
> `chemprop_v2_v3` (18) are unchanged.
>
> **Count synced 2026-10-31 (Wf-A127): 126 → 127.** MEASURED 2026-10-31 by
> live import: `len(build_gates("all")) == 127`,
> `len(build_gates("launch")) == 129`,
> `EXPECTED_GATE_COUNTS["all"] == 127` and
> `EXPECTED_GATE_COUNTS["launch"] == 129` — the code constant and the live
> count agree with each other. The bump was made by Wf-A125 G5 (adding
> `p3_run_ledger_provenance`); Wf-A125 bumped the constant but not this
> doc, so `test_wf327f11_gate_count_consistency.py::TestAuditDocSync::test_audit_doc_table_count_matches_constant`
> was RED with `doc=126 vs code=127`. This is the **same class of defect** as
> the 2026-10-20 sync below, one wave later: the count lives in **four**
> places (the code constant, this header, the §6 table, and
> `molmetal/tests/test_wf415a3_preflight_fail_audit.py`), and a wave that
> bumps one must bump all four. Catalog §6.236.

> **Count synced 2026-10-20: 121 → 126.** The constant and
> `len(build_gates("all"))` had both been at 126 since 2026-10-18 (§9), but
> this header and the §6 row still said 121 — a partial sync that left
> `test_audit_doc_table_count_matches_constant` red. The code was right; the
> doc was stale. `launch` is 128 (126 + the 2 pre-retrain gates).

> **Gate inventory (2026-10-08): 120 → 121.** 1 NEW report-only gate
> `p0_gate_inventory_audit` that introspects `build_gates()` (no gate
> executed) and emits the registered-gate count + the "zombie" count
> (opt-in gates whose env var is unset in a normal run → registered but
> never enforced). Always PASSes; never changes an existing verdict.
> Implementation: `molmetal/scripts/gate_inventory.py`.

> **Registry consistency (2026-10-08): 118 → 119.** 1 NEW opt-in gate
> `p3_registry_consistency` verifying `registry/{datasets,protocols,metrics}.yaml`
> facts against disk by content hash, so a status refresh QUERIES the registry
> instead of re-deriving "dataset not ready / protocol lost / metric stale" from
> prose. Opt-in via `WF_REGISTRY_ENFORCE=1` (report-only default).
> Implementation: `molmetal/scripts/registry.py`.
>
> **Wf-R43 m19 (2026-10-08): 117 → 118.** 1 NEW opt-in fail-loud guard
> `p3_wf_r43_m19_lomo_headline` guarding the m19 LOMO (Rh+Re) **headline
> real-data usability**. `metal_lomo_auc_wasserstein.json` records a mean
> test ROC-AUC 0.7577 over 7 runs (Rh 0.8798 / Re 0.7088) against an
> INTERNAL gate of 0.7, but the wave appended **500 SYNTHETIC Rh rows** to
> the shared 5-metal pool *before* the LOMO split — contaminating the Rh
> train fold and entering the Re train fold too — those rows carry
> hard-coded potency labels (active IC50=0.1 / pIC50=7.0), and the sweep
> covers only **2 of the 5** metals in the pool. The headline is held-out but
> **not real-data-only**. MEASURED corroboration: the pool holds 26,632 rows
> and the wave's `run.log` records `n=27132` = 26,632 + 500, so the
> contamination is arithmetic rather than an inherited prose claim. A
> real-labelled Rh replacement cohort IS on disk (wf387a2, 500 compounds,
> `is_synthetic=false`, zero scaffold leak), so only the 5-metal GPU sweep
> remains, USER_OK-gated per AGENTS.md §5.1. Opt-in via
> `WF43_M19_LOMO_HEADLINE=1` (Wf-R35 P-A precedent), so the count moves
> without changing any existing pre-flight verdict. Implementation:
> `molmetal/scripts/wf_r43a1_m19_lomo_headline_guard.py`; tests:
> `molmetal/tests/test_wf_r43a1_m19_lomo_headline_guard.py` (15/15 PASS).

> **Wf-R43 m1 / m2 / m7 (2026-10-08): 114 → 117.** 3 NEW opt-in fail-loud
> guards registered by the sibling Wf-R43 items and counted here for the
> first time (the §2.1 count was stale by 3 until this sync):
> `p3_wf_r43_m1_paper_cohort` (Ru-AUC paper-cohort comparability),
> `p3_wf_r43_m2_cohort_independence` (R² per-cohort independence — 2 of 3
> "cohorts" were one measurement), and
> `p3_wf_r43_m7_subpocket_uniform` (diversity_subpocket uniform placeholder
> + n_unique=1 over 30 cells). All three are opt-in via `WF43_M1_PAPER_COHORT=1`
> / `WF43_M2_COHORT_INDEPENDENCE=1` / `WF43_M7_SUBPOCKET_UNIFORM=1`.

> **Wf-R43 m16 (2026-10-08): 113 → 114.** 1 NEW opt-in fail-loud guard
> `p3_wf_r43_m16_pt_ckpt_gap` guarding the m16 D-MPNN pIC50 **Pt
> checkpoint gap** (the last red `MOCK_NEEDS_REMEASURE` item in
> `TODO/INDEX.md` §17.1). `dmpnn_ckpt_status.json` inventories 25 ckpts
> across Ru/Ir/Rh/Os/Re × seeds 0-4; Pt is absent. The gap is **not**
> trainable from this repo: the MetalCytoToxDB-derived CSVs carry zero Pt
> rows (MEASURED on disk) and the only Pt cohort reachable via
> `--dataset platinai_4cell` carries SYNTHETIC labels per
> `molmetal/data/platinai_4cell.meta.json`. So the gate audits and FAILs
> loud while the gap is open rather than training on labels that would
> restate their own generator (AGENTS.md §5.2). Opt-in via
> `WF43_M16_PT_CKPT=1` (Wf-R35 P-A precedent), so the count moves without
> changing any existing pre-flight verdict. Implementation:
> `molmetal/scripts/wf_r43a1_m16_pt_ckpt_gap.py`; tests:
> `molmetal/tests/test_wf_r43a1_m16_pt_ckpt_gap.py` (8/8 PASS).

> **Wf-R42 MOCK→REAL (2026-10-08): 108 → 113.** 5 L3 MOCK→REAL gates
> re-registered: `p3_mock_to_real_{m3_vina_kcal_per_mol,m15_rmsd,m14_decode_ratio,m17_drugood_assay,m21_nci60_precious_9cell}_*_REAL`.
> Helper functions at `pre_flight_check.py:9128..9420` stayed intact from
> Wf-432-A1 §6 ship surface; dropped in Wf-437 A2 audit as SCAFFOLD-only
> doc-trackers. Wf-R42 re-registered after the 4 of 5 `_REAL` fields were
> added to the source JSONs with provenance + honest_framing (m14's
> `all_measured_cfm_decode_ratio_REAL` was already there from Wf-398-A3).
> Each gate emits **PASS** when the field exists + value is in range + the
> honest_framing string is present. MEASURED on 2026-10-08: 5/5 PASS,
> pre-flight summary 106/0/7/113.

> **Wf-R35 P-A (2026-10-07): 103 → 108.** 5 NEW opt-in protocol-lock gates

> **Wf-R35 P-A (2026-10-07): 103 → 108.** 5 NEW opt-in protocol-lock gates
> registered: `p3_wf35_protocol_lock_{m1_ru_auc,m2_r2_per_cell,m17_drugood_ood,m19_lomo,m21_nci60_9cell}`.
> Each enforces `configs/protocol_lock.yaml` — per-metric dataset path + md5 +
> split + n_folds — so a retrain cannot silently run on a different cohort
> than the paper comparison was written against. Each emits **SKIP** unless
> `WF35_PROTOCOL_LOCK=1`, so the count moves but no existing pre-flight
> verdict changes. The metric → trainer-config pairing lives in the lock's own
> `trainer_config:` key, not in the gate wiring.
>
> MEASURED under `WF35_PROTOCOL_LOCK=1`: m1 PASS, m19 PASS, m21 PASS,
> **m2 FAIL** (`molmetal/configs/m2_r2_per_cohort.yaml` names no dataset
> artifact — it is a per-cohort feature config, not a trainer launch config),
> **m17 FAIL** (`configs/dmpnn.yaml` routes DrugOOD through a CLI flag, so the
> dataset it actually names is the `metacytotox_fallback` CSV, not the official
> DrugOOD IC50 splits — the wf-R34-B2 gap the lock was built to surface).

> **Wf-A16 (2026-10-03): 105 → 103.** The two PRE-RETRAIN gates
> (`p3_wf343c4_anchor_freshness_24h`, `p3_wf343c2_per_wave_runtime_check`) moved
> out of `all` into a NEW `launch` pipeline (105 gates). `all` is what the
> commit hook runs AND what every launcher ran, so a commit that only touched
> gate tooling failed because the paper-grade anchors were 24 h old — a purely
> temporal trigger, with no cheap remedy (refreshing an anchor means re-running
> the producing workload; `touch` would falsely claim a re-measurement).
> `launch` = `all` + those two. Launchers pass `--pipeline launch`.

| Priority | n | Sections |
|---|---:|---|
| P0 | 5 | config schema / CSV hash / scaffold zero-leak (v2+v3+v4) / nci60 split no-leak (HOLE-5) / metalcytotox v1 known-leaky visibility (L5) |
| P1 chemprop_v2_v3 | 14 | 12 Wf-259 baseline + Wf-321H (`p3_todo_119_p14_cpu_smoke_ready`) + Wf-350-A2 (`p1_compute_manifest`) |
| P1 cfm | 7 | Wf-259 baseline 6 + `p1_cfm_live_decode_canary` (HOLE-1, cfm-pipeline-only live decode smoke) |
| P2 | 4 | data metalcytotox / data metallo / tests / vina |
| P3 | 38 | Wf-306C (9) + Wf-308A (1) + Wf-308B / Wf-312C / Wf-315A (8 ADMET) + Wf-312B (1) + Wf-312E (1) + Wf-313C (1) + Wf-313H / Wf-320F / Wf-322E (3 R20 Phase 0) + Wf-314E (1) + Wf-315D / Wf-319C (2 PB) + Wf-319B (1 UFF-min) + Wf-319A (2 ablation_runner) + Wf-321A (1 Wf-135) + Wf-321E (1 ABLATION_TABLE) + this Wf-321D self-audit (1) + Wf-256 Catherine 30-metal closeout (1) + Wf-323F (1 m4 SA/QED subcriterion) + Wf-325E (1 ADMET 4-below documented) + Wf-325F (1 Phase3G F2A runtime unblock verified) + Wf-327 (1 cpu-num-workers ≥ 4 launch-script gate) + Wf-343C4 (4 pre-retrain internal-check gates: hyperparams_config_complete + yuelbond_decoder_env + anchor_freshness_24h + wave_budget_remaining_nonneg) + Wf-347A1 (2 pre-retrain internal-check companion gates: p3_wf343c2_per_wave_runtime_check + p3_wf343c3_data_quality_check) + Wf-350-A3 (1 pre-retrain hp-provenance gate) + Wf-350-A2 (1 compute-manifest P1 gate, also surfaces in `all`) |
| **Total** | **103** | |

> **Sync note (2026-10-03, HOLE-1 + HOLE-5 + L5):** the §2.x header counts and the §6
> `n_gates_total` row are the ENFORCED invariants — they are kept in
> sync with `EXPECTED_GATE_COUNTS` by
> `molmetal/tests/test_wf327f11_gate_count_consistency.py`.  The
> per-priority rows in this table (P0 / P1-chemprop / P2 / P3) and the
> §4 gate-type table remain the Wf-321D-era (2026-09-29) tally of the
> then-71-gate roster; the roster has since grown (Wf-32x..Wf-A* waves)
> and those rows are NOT re-tallied here.  Cf. §6.2 + §6.3 for the change log.

### §2.2 `frontier` pipeline — 25 gates

Subset of `all` covering only the P3 paper-grade gates (per Wf-306C
standalone design).  The Wf-321D self-audit gate is intentionally
**excluded** from `frontier` because (a) gate-count consistency is most
useful for the end-to-end `all` pipeline, (b) `frontier` is run
frequently during Wf-313C paper-rule parity checks and the 25 gates
should remain stable. Wf-323F m4 SA/QED subcriterion gate was
intentionally surfaced to `frontier` for paper-rule parity per Wf-323F
brief (the Vina sub-criterion bind cascades to m3 USER_OK-gated GPU
work).

Wf-322F (Wf-176 m19 LOMO closeout documented), Wf-325D (m6_kappa
final closeout documented), and Wf-325E (ADMET 4-below documented)
are **documentation-only closeout gates** that live ONLY in `frontier`
— they verify TODO/completed/{322f, 325d, 325e}_*.md ship surfaces
and are intentionally NOT in `all` (they mirror the Wf-322F precedent;
the Wf-321A Wf-135 phase-2 CPU smoke gate at
`p3_wf135_phase2_cpu_smoke_ready` is the lone `frontier`-only
non-closeout gate, kept there for paper-rule parity per Wf-321A brief).
Wf-325F Phase3G F2A runtime unblock verifier is `all`-only (not
paper-relevant; runtime unblock is a CPU-only sanity check).

### §2.3 `chemprop_v2_v3` pipeline — 18 gates

Wf-259 baseline 12 + Wf-321H (`p3_todo_119_p14_cpu_smoke_ready`) +
Wf-350-A2 (`p1_compute_manifest`) + Wf-A8 (4 `p3_protocol_alignment_yaml_*`
gates; registered for `all` + `chemprop_v2_v3`).

### §2.4 `cfm` pipeline — 8 gates

Wf-259 baseline 6 + `p1_cfm_live_decode_canary` (HOLE-1 fix, 2026-10-03:
live 3-epoch CPU smoke of today's `train_cfm_yuelbond_wasserstein.py` +
YuelBond decoder — loss decrease 0.3% floor + `decode_ratio_rdkit_strict
> 0`, fail-closed). The pre-existing `p1_cfm_smoke_decoded` frozen Wf-188
anchor is kept but relabelled HISTORICAL (it does not exercise today's
code).

**Wf-A132 (2026-10-31): 7 → 8.** `p3_cfm_fragmentation_preflight` — the
pre-flight REGISTRATION of the Wf-A128 standalone fragmentation gate
(`molmetal/scripts/pre_flight_fragmentation_check.py`). The census
artifact's per-run `fragmentation_gate` verdict was recorded but never
consulted by any gate, so a run could train for hours and publish a
fragmented pool with nothing in the way. The gate is registered on the
`cfm` pipeline ONLY, on purpose: it reads a HISTORICAL record, and a FAIL
there would block every commit (`--pipeline all` is the git hook chain)
and every unrelated launcher over a defect that belongs to the CFM line
alone. MEASURED 2026-10-31: `all=127 launch=129 frontier=25
chemprop_v2_v3=18 cfm=8`; re-measured 2026-11-01 (Wf-A154) after the two
SSOT-document consistency gates landed on the `all` pipeline: `all=129 launch=131
frontier=25 chemprop_v2_v3=18 cfm=8`.

---

## §3 Overlap analysis

### §3.1 `p3_pb_packages_available_or_documented` (Wf-315D) vs `p3_pb_packages_status_present` (Wf-319C)

**Overlap description.** Both gates probe the same three artefacts:

1. The venv-python probe for `rdkit` + `posebusters` packages.
2. The `metrics/by_metric/steric_clash_pb_dock_1h36.json` metric JSON.
3. The `molmetal/reports/wf315d_pb_smoke/m8_prod_1h36.json` smoke output.

**Why both are needed (NOT a dedupe candidate).** The gates enforce
orthogonal contracts:

* **Wf-315D (`available_or_documented`)** — *block-aware*: PASS if
  either (a) packages are installed + metric is `MEASURED_SMOKE` +
  smoke output is valid, OR (b) packages are missing + metric carries
  `HONEST_NEGATIVE_BLOCKED` tag + non-empty `blocker_description`.
  This is the BLOCKED-path contract — the gate PASSES when work is
  blocked but documented honestly, and FAILS only when the situation
  is contradictory (e.g. packages installed but no MEASURED_SMOKE).
* **Wf-319C (`status_present`)** — *artefact-presence*: strictly
  requires all three artefacts to be present and the metric to be
  `MEASURED_SMOKE`.  No fallback to HONEST_NEGATIVE_BLOCKED.  This is
  the "ready to ship" verifier.

**Decision: DOCUMENT (keep both).** Removing Wf-315D would lose the
BLOCKED-path honest-negative contract (critical for §9 honest-framing
catalog).  Removing Wf-319C would lose the strict artefact-presence
verifier.  Both contracts are independently valuable.

### §3.2 P0 CSV line-count vs P2 CSV line-count

**Overlap description.** Both gate families check row counts on the
same CSV files:

| File | `p0_csv_hash_verify` min | `p2_data_*_exists` min |
|---|---:|---:|
| `cytox_csv` (v1) | 26633 | 26800 |
| `metallo_csv` | 450 | 449 |
| `cytox_csv_v2/v3/v4` | 26633 each | (not in P2) |

**Why both are needed.** The P0 gates are **launch-blockers** (v2/v3/v4
CSV schema coverage; pre-Wf-261 zero-tolerance contract).  The P2
gates are **post-launch PASS contracts** (data-stability floor for
the canonical v1 CSV).  Different thresholds reflect different
contract intent: P0 strictness catches sub-P50 row loss; P2 strictness
catches catastrophic row loss (e.g. CSV truncation).

**Decision: DOCUMENT (keep both).**

### §3.3 Wf-319A: tests vs aggregate.csv vs registry_complete (Wf-321E)

Three different artefacts, three different layers:

* `p3_ablation_runner_tests_present` — pytest on `test_wf319a_ablation_runner.py` (12 NEW tests).
* `p3_ablation_runner_aggregate_csv_present` — `aggregate.csv` ≥ 11 data rows + DESIGN_ONLY rows carry empty metric columns.
* `p3_ablation_runner_registry_complete` — `ABLATION_TABLE` exported from `ablation_runner.py` with §7 contract fields (Wf-321E).

**Decision: NOT overlapping** (test-runner side vs artefact side vs
source-side).  All three are kept.

### §3.4 P3 frontier-math: wire_in vs Day-3-5 integration tests vs paper refs

Three different layers:

* `p3_frontier_math_wire_in` — 4 adapter `.py` files are parseable (file existence + non-empty).
* `p3_frontier_math_day_3_5_integration_tests` — pytest on `test_wf313c_frontier_math_integration_e2e.py` (8 NEW tests).
* `p3_frontier_math_paper_refs_present` — 4 per-adapter markdown refs + 1 index README on disk.

**Decision: NOT overlapping.** Wire-in is structural; integration
tests are behavioural; paper refs are documentation.  All three are
kept.

### §3.5 AnchorGuard: version_pin vs consumers_adopt

* `p3_anchorguard_version_pin` — walks paper-grade `*_anchor.json` files; asserts top-level `"version"` field pins `1.0.0`.
* `p3_anchorguard_consumers_adopt` — checks the 3 consumer scripts (`pre_flight_check.py`, `paper_auto_update.py`, `paper_dashboard.py`) exist on disk.

**Decision: NOT overlapping.** Different concerns: pin existing
anchors vs adopt the AnchorGuard library in consumers.  Both kept.

### §3.6 ADMET druglikeness: value gates vs honest-negative tag gates

* `p3_admet_ames_druglikeness` — fail-loud: PASS iff `value ≤ 0.10`.
* `p3_admet_cell_perm_druglikeness` — fail-loud: PASS iff `value ≥ -5.0`.
* `p3_admet_aq_sol_honest_negative_tag` (and 5 others) — fail-soft: PASS iff JSON carries `HONEST_NEGATIVE: true` + reason.

**Decision: NOT overlapping.** Value gates enforce druglikeness band
contracts; honest-negative tag gates enforce honest-framing contracts
(cohort mean below band → must be tagged honestly).  Both kept.

---

## §4 MEASURED-only verification

Per AGENTS.md §1 MEASURED-vs-PROJECTED discipline, every numeric value in
any agent output MUST trace to a JSON / report path on disk.  Audit
results for the 71 gates:

| Gate type | n | Reads from | MEASURED? |
|---|---:|---|:---:|
| `gate_grep` (P1 chemprop + Wf-321H) | 13 | chemprop trainer script source | YES (source-as-frozen) |
| `gate_help` (P1 chemprop + Wf-321H) | (subtotal of above) | chemprop trainer --help output | YES (CLI surface) |
| `gate_file_exists` (P1 cfm + P2 vina + harness) | 7 | on-disk files | YES |
| `gate_min_lines` (P2 data) | 2 | CSV row counts | YES |
| `gate_pytest` (P2 tests + Wf-313C + Wf-319A + Wf-308A) | 5 | pytest exit codes | YES |
| `gate_which_or_apt` (P2 vina) | 1 | `which` + `apt list` output | YES |
| `gate_yaml_schema` (P0) | 1 | YAML parseability | YES |
| `gate_csv_hash` (P0) | 1 | CSV row counts (cheap fingerprint) | YES |
| `gate_scaffold_zero_leak` (P0) | 1 | scaffold_id disjoint sets via pandas | YES |
| `gate_python_importable` (P3 frontier) | 1 | adapter file existence + parseability | YES |
| `gate_all_files_exist` (P3 paper_refs, drivers, harness, OOD, R20 Phase 0, selenium, nci60_9cell, ablation_runner) | 9 | on-disk files | YES |
| `gate_json_threshold` (P1 cfm_smoke_decoded + P1 catherine_30metals + P3 ADMET value gates) | 4 | on-disk JSON values | YES |
| `gate_admet_honest_negative_tag` (P3 ADMET honest-negative gates) | 6 | on-disk JSON `HONEST_NEGATIVE` field | YES |
| `gate_nci60_9cell_per_cell_aggregate` (P3 nci60) | 1 | on-disk aggregate JSON shape | YES |
| `gate_loi_send_ready` (P3 LOI) | 1 | 4 LOI files + review helper | YES |
| `gate_r20_phase0_loaders_complete` (P3 R20 Phase 0) | 1 | 9 loader artefacts + pytest | YES |
| `gate_wf315d_pb_smoke` (P3 PB) | 1 | venv probe + metric JSON + smoke output | YES |
| `gate_wf319c_pb_packages_status_present` (P3 PB status) | 1 | venv probe + metric JSON + smoke output | YES |
| `gate_wf319b_uff_min_pt_pool_present` (P3 UFF-min) | 1 | pool.json + metrics.json tag | YES |
| `gate_wf319a_ablation_aggregate_csv_present` (P3 aggregate.csv) | 1 | on-disk CSV shape | YES |
| `gate_wf321a_wf135_phase2_cpu_smoke` (P3 Wf-321A) | 1 | 3 artefact existence | YES |
| `gate_wf321e_ablation_registry_complete` (P3 Wf-321E) | 1 | `ABLATION_TABLE` import + §7 contract | YES |
| `gate_section9_entry_count` (P3 §9 catalog) | 1 | `207_unified_pre_flight_checklist_v2.md` §9 entry count | YES |
| `gate_anchor_version_pin` (P3 anchors) | 1 | `*_anchor.json` `version` field | YES |
| `gate_gate_count_consistency_audit` (P3 Wf-321D) | 1 | `build_gates()` introspection | YES |
| **Total** | **71** | | **71 / 71 MEASURED-only** |

**Result: 71/71 gates are MEASURED-only.** Zero PROJECTED / ASPIRATIONAL
gates.  Threshold values (e.g. `0.10` for ames_druglikeness) are
druglikeness band contracts (not aspirations — they are derived from
Krasnov 2026 cohort band means and Wf-300A forward-ship targets).

---

## §5 New gate: `p3_gate_count_consistency_audit`

### §5.1 Why

Silent drift in the gate list is the highest-risk regression for
pre-flight v2: a gate silently removed → downstream consumer breaks.
Adding a gate without bumping the audit → next workflow wave FAILs
without explanation.

### §5.2 Implementation

Direct introspection of `build_gates(pipeline)`.  Cheap (< 50 ms).
The gate is appended LAST to the `all` pipeline runners list so its
own presence is included in the count it audits.

```python
EXPECTED_GATE_COUNTS = {
    "all": 105,
    "frontier": 25,
    "chemprop_v2_v3": 18,
    "cfm": 7,
}
```

### §5.3 Bump protocol

When a new gate is added to `build_gates(pipeline)`:

1. Increment `EXPECTED_GATE_COUNTS[pipeline]` by 1.
2. Update `TODO/completed/321d_preflight_gate_audit.md` §2 with the
   new gate name + section.
3. Update §4 row count for the gate type.
4. Re-run `pre_flight_check.py --pipeline all --json` and confirm
   `gate p3_gate_count_consistency_audit` PASSes.

When a gate is removed (rare; requires Wf-level justification):

1. Decrement `EXPECTED_GATE_COUNTS[pipeline]` by 1.
2. Update §2 + §4.
3. Re-run pre-flight.

---

## §6 Audit summary

| Metric | Value |
|---|---:|
| `n_gates_total` (per-pipeline COUNT) | 133 (`all`) / 25 (`frontier`) / 18 (`chemprop_v2_v3`) / 8 (`cfm`) — plus the `launch` pipeline at 135 (`launch` = `all` + the 2 pre-retrain gates; MEASURED by `test_wf344b3_gate_count.py`; **re-synced 2026-10-31 (Wf-A127)** — the 2026-10-20 values were 126 (`all`) / 128 (`launch`) and were stale after Wf-A125 G5 added `p3_run_ledger_provenance`; MEASURED 2026-10-31 by live import: `all=127 launch=129 frontier=25 chemprop_v2_v3=18 cfm=7`; **re-synced again 2026-10-31 (Wf-A132)** — `cfm` 7 → 8 after `p3_cfm_fragmentation_preflight` was registered on the `cfm` pipeline only (`all` and `launch` unchanged); MEASURED 2026-10-31 by live import: `all=127 launch=129 frontier=25 chemprop_v2_v3=18 cfm=8`; **re-synced 2026-11-01 (Wf-A154)** — `all` 127 → 129 and `launch` 129 → 131 after the two SSOT-document consistency gates (`p3_ssot_path_reference_check`, `p3_ssot_census_claim_check`) were registered on the `all` pipeline; MEASURED 2026-11-01 by live import: `all=129 launch=131 frontier=25 chemprop_v2_v3=18 cfm=8`; **re-synced 2026-11-04 (Wf-A269 wave-5/6)** — `all` 129 → 132 and `launch` 131 → 134 after `p3_runtime_transport_check` was registered on the `all` pipeline; MEASURED 2026-11-04 by live import: `all=132 launch=134 frontier=25 chemprop_v2_v3=18 cfm=8`; **re-synced 2026-11-04 (Wf-A269 wave-19)** — `all` 132 → 133 and `launch` 134 → 135 after `p3_script_contract_check` was registered on the `all` pipeline; MEASURED 2026-11-04 by live import: `all=133 launch=135 frontier=25 chemprop_v2_v3=18 cfm=8`) |
| `n_overlaps_found` | 2 (`p3_pb_*` family + P0/P2 CSV line-count) |
| `n_deduped` | 0 (all overlaps were DOCUMENTED instead) |
| `n_documented` | 2 |
| `n_measured_only` | 71 / 71 |
| `preflight_status` | **PASS** (gate-count consistency audit ships; 4/4 pipelines consistent) |

### §6.1 Wf-350-A2 update (2026-09-29)

Per Wf-350-A2 ship (NeurIPS §E.5 compute-manifest surface; gate #2 of 3
per Wf-343a-R2 recommendation):

* `all` bumped from 70 → 71 (added `p1_compute_manifest`).
* `chemprop_v2_v3` bumped from 13 → 14 (added `p1_compute_manifest`).
* `frontier` / `cfm` unchanged (compute-manifest is NOT a paper-rule
  parity gate; it lives at the chemprop_v2_v3 P1 level per the
  existing Wf-259 'p1_*' gate convention for host-runtime probes,
  and surfaces in `all` because it is a pre-retrain internal check per
  §3.4 design contract).
* See `TODO/completed/350a2_compute_manifest.md` for the gate contract
  + live-host MEASURED manifest.

### §6.2 HOLE-1 update (2026-10-03)

Per the HOLE-1 pre-flight fix (the cfm decode canary now verifies TODAY's
code):

* `cfm` bumped 6 → 7 (added `p1_cfm_live_decode_canary`, cfm-only: live
  3-epoch CPU smoke of today's `train_cfm_yuelbond_wasserstein.py` +
  YuelBond decoder; MEASURED 6.5 s wall on the dev host; requires loss
  decrease with a 0.3% floor AND `decode_ratio_rdkit_strict > 0`,
  fail-closed).
* `p1_cfm_smoke_decoded` (frozen UNTRACKED Wf-188 artifact whose source
  checkpoint changed after it was written) is kept but relabelled
  HISTORICAL — its evidence now states explicitly that it does not
  exercise today's code.
* `all` (101 at the Wf-A12-era sync; **104** as of the §6.4 L4
  update below) / `chemprop_v2_v3` 18 are the MEASURED constants synced
  into this doc (the previous doc values 71 / 14 were already stale
  before HOLE-1; see the constant comments in `pre_flight_check.py`
  for the Wf-A12 drift note).

### §6.3 HOLE-5 update (2026-10-03)

Per the HOLE-5 pre-flight fix (extend split-leak coverage beyond the
single MetalCytoTox gate):

* `all` bumped 101 → 103, MEASURED via `len(build_gates("all"))`:
  * 101 → 102 — the H4 fix (this run, Wf-446) registered the all-only
    gate `p3_dmpnn_loss_wire_in_smoke` (D-MPNN 50-step CPU loss
    wire-in smoke) and the `all` constant was left at 101 although the
    measured count had moved to 102 (the `wf327f11`
    gate-count-consistency test was RED at 101 vs 102; the HOLE-5 edit
    re-syncs the constant to the measured value).
  * 102 → 103 — HOLE-5 added the all-only gate
    `p0_nci60_split_no_leak` (nci60 9-cell split integrity: loader
    reproduction vs the frozen 3645/521/1041 unique-size reference,
    per-cell × seed helper re-run with pairwise index disjointness +
    coverage, and real Bemis-Murcko scaffold overlap reported) and
    widened `p0_scaffold_zero_leak` from the single `metalcytotox_v4`
    CSV to `metalcytotox_v2` + `v3` + `v4` (one subprocess, one
    OK/LEAK row per CSV; v1 stays excluded as the pre-Wf-261 leaky
    artifact — honest-negative noted in the gate evidence).
* `cfm` was NOT moved by HOLE-5 (the HOLE-1 canary is cfm-only; `all`
  runs the identical smoke under
  `p3_wf385_scaffold_loss_wire_in_verified`).
* HONEST-NEGATIVE (MEASURED, not fixed here): the shipped nci60
  "scaffold" split is molecule-level, not scaffold-level — real
  Bemis-Murcko overlap train/val=165, train/test=282, val/test=121 —
  root cause is the swallowed `TypeError` at
  `molmetal/data/splits.py::ScaffoldSplitter._scaffold_for` (L595-609;
  `MurckoScaffold.GetScaffoldForMol` called with an unsupported
  `includeChirality` kwarg).  Out of this fix's file ownership; see
  `TODO/completed/h5_nci60_split_gate.md` for the FAIL/PASS proofs +
  deliberate non-fixes.

### §6.4 L4 update (2026-10-03)

Per the L4 pre-flight fix (wire the previously-orphaned
`molmetal/scripts/check_data_leakage.py` diagnostic into pre-flight):

* `all` bumped 103 → 104, MEASURED via `len(build_gates("all")) == 104`
  after the L4 fix added the all-only gate `p2_data_leakage_diag`
  (runs the Wf-44 Phase 3 diagnostic with explicit inputs —
  `molmetal/data/platinai_5k.csv` + `.data/molmetal/`
  `metalcytotox_5metal_multitask_v4.csv` — and requires scaffold-split
  strictness for both datasets + shipped-split strictness for
  metalcytotox v4; the random-split overlap is recorded as the expected
  honest-negative, not gated).
* Both-ways MEASURED on the dev host (per AGENTS.md §1): v1 CSV
  (`.data/molmetal/metalcytotox_5metal_multitask.csv`) → gate FAIL
  (`shipped_split: strict=FAIL (48 shared scaffolds)`); v4 CSV →
  gate PASS (`shipped_split: strict=PASS (0 shared scaffolds)`).
* `frontier` / `chemprop_v2_v3` / `cfm` unchanged (all-only per §3.4).
* See `TODO/completed/h4_data_leakage_gate.md` for the ship surface +
  FAIL/PASS proofs + the test file
  `molmetal/tests/test_wf446_l4_data_leakage_gate.py` (8/8 PASS).

### §6.5 L5 update (2026-10-03)

Per the L5 pre-flight fix (cover the known-leaky v1 CSV explicitly —
the HOLE-5 comment claimed v1 "is not a trainer input", which is FALSE):

* `all` bumped 104 → 105, MEASURED via `len(build_gates("all")) == 105`
  after the L5 fix added the all-only visibility gate
  `p0_metalcytotox_v1_known_leaky`.
* Gate contract (report-only; NOT red-by-design): PASS-with-evidence iff
  the v1 CSV still leaks > 0 (evidence records rows + per-pair scaffold
  overlap counts + sha256 prefix + the grep-MEASURED default-consumer
  roster as the OPEN remedy — repointing the consumers at v4 is out of
  the gate's file ownership); FAIL iff 0 leaks (the artifact no longer
  matches its known-leaky historical record — mutated / re-split /
  replaced; the FAIL message names the `configs/data_checksums.json` +
  `p0_csv_hash_verify` restore path); SKIP on an absent gitignored
  `.data/**` CSV (presence is owned by the blocking hash gate) and on a
  pandas-less host runtime (Wf-345B1 env-aware contract).
* Both-ways MEASURED on the dev host (repo venv, ~0.4 s per run):
  * live v1 (`.data/molmetal/metalcytotox_5metal_multitask.csv`,
    sha256 `f167c119…`) → PASS: `rows=26801, scaffold overlaps
    train/val=29,train/test=48,val/test=32 (total=109)`.
  * constructed 0-leak input (v4 bytes `5a2550c8…` placed at the v1
    path — the "mutated / re-split / replaced" shape) → FAIL:
    `v1 no longer leaks: 0 scaffold overlaps`.
  * missing path → SKIP; simulated pandas env-gap → SKIP; unparsable
    probe output → FAIL.
* The FALSE "not a trainer input" claims (the `gate_scaffold_zero_leak`
  docstring + its registration comment) are corrected in the same edit;
  the grep-MEASURED roster names 10+ default consumers
  (`train_ru_multitask.py` L156, `train_multimetal_lomo.py` L790,
  `train_ru_5metal_se3sb.py` L201, `train_multiph.py` L153,
  `train_transport.py` L141, `train_se3sb.py` L142,
  `benchmark_r18_metalcytotox.py` L99, `benchmark_wf25_5seed.py` L121,
  the 3 `wf25_*_real_eval.py` argparse defaults,
  `bench_lambda_cfm_e2e.py` L147).
* `frontier` / `chemprop_v2_v3` / `cfm` unchanged (all-only per §3.4;
  the v2+v3+v4 zero-leak contract remains on `p0_scaffold_zero_leak`).
* Ship surface: `molmetal/tests/test_wf446_l5_v1_known_leaky_gate.py`
  (7/7 PASS in 1.3 s) + `gate_metalcytotox_v1_known_leaky` +
  `METALCYTOTOX_V1_DEFAULT_CONSUMERS` in
  `molmetal/scripts/pre_flight_check.py`.

---

## §7 References

* `molmetal/scripts/pre_flight_check.py` — auto-verify script (this audit's home)
* `TODO/completed/207_unified_pre_flight_checklist_v2.md` — source-of-truth checklist referenced by every gate
* `TODO/completed/315d_pb_smoke_retry.md` — Wf-315D ship (PB packages block-aware gate)
* `TODO/completed/319c_pb_packages_status_present.md` — Wf-319C ship (PB packages strict verifier)
* `TODO/completed/319a_ablation_runner_smoke.md` — Wf-319A ship (ablation_runner tests + aggregate.csv)
* `TODO/completed/321a_wf135_phase2_cpu_scaffold.md` — Wf-321A ship (Wf-135 Phase 2 CPU scaffold)
* `TODO/completed/321e_ablation_runner_integration_test.md` — Wf-321E ship (ABLATION_TABLE registry mirror)
* `AGENTS.md` §1 (MEASURED-vs-PROJECTED) + §5.2 (honest framing)

---

## §8 2026-10-10 — narrative claim provenance gate (AGENTS.md §5.2)

**Gate added:** `p3_narrative_claims_check` (pipeline `ag52_narrative_claims`).
Gate counts moved `all` **123 → 124** and `launch` **125 → 126**,
MEASURED via `len(build_gates("all")) == 124` / `len(build_gates("launch")) == 126`.

**What it checks.** Every headline number in `docs/00_MODEL_NARRATIVE.md` is
declared once in `registry/narrative_claims.yaml` together with the JSON file
and dotted field path it came from. The gate re-resolves all 27 claims and
fails if any stated value no longer equals what is on disk.

**Why it exists.** §5.2 traceability was being enforced by hand, and hand
enforcement missed two real defects during the 2026-10-10 audit:

1. `docs/ALIGNED_COMPARISON.md` §1 listed "MCF7 0.846 / A549 0.870 / A2780cis
   0.721". On disk: MCF7 **0.7441**, A549 **0.7758**, A2780cis **0.7278** — and
   the 0.870 belongs to **PC3**. Those values had been copied verbatim into the
   pinned narrative.
2. `n_distinct`'s headline "20" is `aggregate.n_top_k_cap` — the file's own
   interpretation says "saturate the n_top_k cap on every cell". Rendering it
   as "20 vs gate 3 = 6.7×" presents a saturated cap as a measurement.

Neither was visible to a reader who trusted the prose, which is exactly the
failure mode this gate exists to convert from silent into red.

**Scope limit.** The gate checks existence and value only. Whether a claim
*belongs* in the paper is a judgement, not provenance, and is not gated here.

---

## §9 2026-10-18 — MSRS representation gates (FIX-E1 + FIX-P2)

**Gates added:** `p3_msrs_dual_split_check` (pipeline `fix_e1`) and
`p3_msrs_granularity_check` (pipeline `fix_p2`).
Gate counts moved `all` **124 → 126** and `launch` **126 → 128**,
MEASURED via `len(build_gates("all")) == 126` / `len(build_gates("launch")) == 128`.

**Why they exist.** Two silent failures in the MSRS feature audit:

1. **Single-split scoreboards (FIX-E1).** The same arm scores 0.3366 under
   fragment-grouped CV and 0.3972 under canonical-grouped CV — 0.0606 of pure
   split artifact. The historical scoreboard only ever published the first.
   The gate requires both `fragment_grouped` and `canonical_grouped` in every
   artifact that reports per-arm R2, and `write_report()` refuses to write one
   that does not.
2. **Granularity mismatch (FIX-P2 / DEF-6).** The cohort is built per complex
   (`ligand_canonical`) while every historical builder featurized the main
   fragment (`ligand_main_fragment`): 5,861 complexes collapsed onto 2,846
   fragments. Both columns exist and nothing type-checked the pairing. The gate
   requires the headline arm of any complex-grouped split to declare a
   complex-level ligand source.

**What they do not check.** Whether the reported delta is large enough to
matter, or whether the arm belongs in the paper. Those are judgements; these
gates only make the two structural defects above impossible to ship silently.

---

## §10 2026-11-01 — the pre-flight's own behaviour, MEASURED (Wf-A154 → Wf-A162)

**Append-only.** Sections §1-§9 above are PRESERVED VERBATIM (AGENTS.md §5.5 rule #6). This
section records what the Wf-A154 → Wf-A162 chain MEASURED about the pre-flight itself. Every
number below traces to an artefact on disk or to a command in this repo; none is estimated.

### §10.1 MEASURED lineage of the LIVE pre-flight (`--pipeline all --strict`, 129 gates)

| when | PASS | FAIL | SKIP | WARN | TIMEOUT | TOTAL | wall |
|---|---|---|---|---|---|---|---|
| 2026-09-30 (Wf-415-A3 v3) | 107 | 7 | 3 | — | — | 117 | 191.19 s |
| 2026-10-31 (Wf-A127) | 113 | 4 | 10 | 0 | — | 127 | — |
| 2026-11-01 01:16 (Wf-A154) | 101 | 3 | 11 | 0 | **14** | 129 | **686.63 s** |
| 2026-11-01 01:28 (Wf-A157) | **118** | **0** | 11 | 0 | **0** | 129 | 280.78 s |
| 2026-11-01 02:35 (Wf-A162, **CPU lock**, **CONTENDED**) | **118** | **0** | 11 | 0 | **0** | 129 | **81.603 s** |

The 2026-11-01 02:35 row is the newest live measurement; it is published at
`molmetal/reports/wfa162_preflight_freshrun_2026_11_01.txt` (2,415 B). The Wf-415-A3 report at
`molmetal/reports/wf415a3_preflight_freshrun.txt` (161 B, mtime 2026-09-30 18:38,
112/2/3/117) is **left byte-identical** as the historical rendering — it is what the four
`TODO/completed/415*.md` documents cite, and those citations remain true.

**Why the wall time fell from 280.78 s to 81.603 s.** The run was executed under the standing
CPU lock (`CUDA_VISIBLE_DEVICES=""`), which short-circuits the GPU-visibility probe (§10.4) and
skips the GPU smoke spawns. The PASS/FAIL/SKIP/TIMEOUT counts are **IDENTICAL** to the
Wf-A157 quiet measurement — the wall-time difference is a *lock* effect, not a *gate* effect.

### §10.2 Wf-A157 — the `--help` probe concurrency race

The 14 TIMEOUTs of the 2026-11-01 01:16 run included **all 12** `--help` probe gates. Those
probes shared one cache slot, so 12 concurrent spawns of the same key serialised on each other
and starved each other's wall budget. The repair collapsed the 12 spawns to **1** via a
per-key probe lock and raised the budget from 30 s to 180 s. **MEASURED:** wall 686.63 s →
280.78 s (−59 %) and 14 TIMEOUTs → 0.

### §10.3 Wf-A158 — a TIMEOUT is not a FAIL

**MEASURED (Wf-A159, `molmetal/reports/wfa159_timeout_verdict_census/timeout_verdict_census.json`):**
**35** sites carry a timeout branch; **35** of them map to a TIMEOUT verdict; **0** unhandled.
By role: producer **1**, gate **31**, helper **3**. By effect: `PRODUCES_RC_124` **1**,
`TIMEOUT_VERDICT` **34**, `NOT_MEASURED_FALLBACK` **0**, `BRANCH_MAPS_TO_DEFECT` **0**.

**The rule (Wf-A135, load-bearing):** a wall-clock timeout means **the measurement was NOT
obtained**. That is never a FAIL — it is a TIMEOUT, and a TIMEOUT is not evidence of a defect.
Three earlier waves had been reporting timeouts as failures. The invariant is now regression-
guarded by `molmetal/tests/test_wf415a3_preflight_fail_audit.py::test_preflight_no_timeout_derived_fail`:
**no FAIL gate's evidence may carry a timeout marker** (`TIMEOUT:`, *"the measurement was NOT
obtained"*, `TIMED OUT`, *"wall budget"*).

### §10.4 Wf-A162 — the GPU-visibility probe allocated GPU memory to ask whether there was a GPU

**MEASURED.** The mol-metal GPU watchdog recorded **exactly one** mol-metal allocation in its
life: **550 MiB**, pid 700241, 24 consecutive samples ⇒ ≈120 s held. Its last write is
**2026-10-31 17:52**, so the event is **2026-10-31 ≈17:50-17:52**. **No `.pt` file was written
anywhere in the repo that day** ⇒ no training, sampling or decode ran; the allocation was a
**CUDA context** created by a *device-visibility query*.

**Mechanism, read from the installed torch's SOURCE (never executed):**
`.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:205-229` — `torch.cuda.is_available()`
takes the `else` branch unless `PYTORCH_NVML_BASED_CUDA_CHECK=1`, and that branch's own comment
says it *"uses the CUDA Runtime API `cudaGetDeviceCount` which in turn initializes the CUDA
Driver API via `cuInit`"*.

**Two aggravating facts.** (1) The pre-flight **already computed** the repo's CPU lock for the
GPU-REQUIRED gate (`_session_cuda_locked` at `pre_flight_check.py:6902`) and SKIPped honestly;
`_smoke_device()` never consulted it. (2) When the lock is absent, `_smoke_device()` returns
`"cuda:0"`, and the two gates that consume it (Wf-446 H4 D-MPNN smoke, H2 chemprop smoke) then
run an **actual 2-epoch training on the GPU**. ⇒ **the repo's own gate tool could silently
start GPU training.**

**REPAIRED (Wf-A162):** `_smoke_device()` short-circuits to `"cpu"` under the lock **without
spawning torch at all** (behaviour-identical, allocation-free); when it runs it carries
`PYTORCH_NVML_BASED_CUDA_CHECK=1`; a new `_device_authorisation_error(device)` refuses any
device the session did not authorise (including a `-1` mask, which hides every device without
blanking the variable); **both** consuming gates now **FAIL loudly, smoke NOT run**.
Regression surface: `molmetal/tests/test_wfa162_smoke_device_cpu_lock.py` — **10 tests**, two
of them AST-based structural guards (every `_smoke_device()` call site must live in a function
that also calls `_device_authorisation_error(`; no probe may run without an explicit `env=`).

**MEASURED CONFIRMATION (2026-11-01 02:35):** the 129-gate contended run above executed with
**ZERO GPU allocation** — the watchdog's output file mtime was still **2026-10-31 17:52**
(not written during the run) and `nvidia-smi --query-compute-apps` lists **no** mol-metal
process. The 2026-10-31 run of the same tool allocated 550 MiB. **The repair is measured, not
asserted.**

### §10.5 The pre-flight's wall time is contention-sensitive

The same 129-gate run does not take the same time twice on the same machine: 686.63 s under
Wf-A154's conditions, 280.78 s under Wf-A157's, 81.603 s under the CPU lock. ⇒ **a timeout
measured while other work is running is not a defect measured while other work is running.**
The repo has no contention-aware timeout policy yet — recorded as an open item in
`docs/HONEST_NEGATIVE_CATALOG.md` §6.265 §12.


### §10.6 The frozen COUNT was extended by a frozen NAME SET (2026-11-04, Wf-A269 frontier sweep)

**The defect.** `EXPECTED_GATE_COUNTS` froze ONE NUMBER per pipeline. A number cannot
distinguish (a) a gate **RENAME** (`p3_foo` → `p3_bar`), (b) a gate **MOVED** from the
`all` pipeline to the `frontier` pipeline, or (c) a gate **REPLACED** by a different gate
`p3_baz` — all three leave the count unchanged, so `p3_gate_count_consistency_audit`
reported "consistent" while the surface it certifies had changed. That is the §6.269.17
DEAD-GATE class in its "counting instead of reading" costume: a check that cannot fail.

**The repair.** `molmetal/scripts/pre_flight_check.py` now carries
`EXPECTED_GATE_NAMES: dict[str, tuple[str, ...]]` NEXT TO `EXPECTED_GATE_COUNTS`. The
tuples were GENERATED from a LIVE import of `build_gates` — never typed by hand.
MEASURED 2026-11-04 by live import: `all` 133 names, `launch` 135, `frontier` 25,
`chemprop_v2_v3` 18, `cfm` 8 — identical to the counts in §6.

**The count is now a DERIVED ASSERTION.** `gate_gate_count_consistency_audit` (in
EXPLICIT mode, i.e. the commit path) additionally checks that
`expected_total == len(EXPECTED_GATE_NAMES[pipeline])` and that the live registration's
names equal the frozen tuple. A rename / a cross-pipeline move / a duplicate / an
add-remove now FAILs the pre-flight with a `GATE NAME DRIFT` evidence string.

**Bump protocol (§5.3) EXTENDED — three artefacts move together.** Adding, removing or
renaming a gate now requires: (1) `EXPECTED_GATE_COUNTS[pipeline]`,
(2) `EXPECTED_GATE_NAMES[pipeline]`, (3) this doc's §2 + §6.

**HONEST LIMITS.**
* The comparison is **ORDER-INSENSITIVE** (the frozen tuples are sorted): a change of
  REGISTRATION ORDER is not detected by the name-set check (the count never moved in
  that case either).
* The name check runs in EXPLICIT mode only. The Wf-345-B3 auto-compute escape hatch
  (`expected_total` omitted) is byte-identical to its previous rendering, because its
  evidence string claims the gate "compares the count to itself" and that claim must
  stay literally true.
* The frozen set is a snapshot of ONE tree state: it says what the registered surface
  must BE; `build_gates` says what it IS.

**Regression surface.** `molmetal/tests/test_wf327f11_gate_count_consistency.py` —
`TestExpectedGateNames` (4 tests) + `TestNameDriftIsDetected` (5 tests, 4 of them
mutation proofs against the production gate). MEASURED 2026-11-04: **12 passed**
(CPU-only, < 1 s).
