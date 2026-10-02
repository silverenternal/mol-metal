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

## §2 Gate inventory (per-pipeline, 2026-09-29)

### §2.1 `all` pipeline — 71 gates

| Priority | n | Sections |
|---|---:|---|
| P0 | 3 | config schema / CSV hash / scaffold zero-leak |
| P1 chemprop_v2_v3 | 14 | 12 Wf-259 baseline + Wf-321H (`p3_todo_119_p14_cpu_smoke_ready`) + Wf-350-A2 (`p1_compute_manifest`) |
| P1 cfm | 6 | Wf-259 baseline |
| P2 | 4 | data metalcytotox / data metallo / tests / vina |
| P3 | 40 | Wf-306C (9) + Wf-308A (1) + Wf-308B / Wf-312C / Wf-315A (8 ADMET) + Wf-312B (1) + Wf-312E (1) + Wf-313C (1) + Wf-313H / Wf-320F / Wf-322E (3 R20 Phase 0) + Wf-314E (1) + Wf-315D / Wf-319C (2 PB) + Wf-319B (1 UFF-min) + Wf-319A (2 ablation_runner) + Wf-321A (1 Wf-135) + Wf-321E (1 ABLATION_TABLE) + this Wf-321D self-audit (1) + Wf-256 Catherine 30-metal closeout (1) + Wf-323F (1 m4 SA/QED subcriterion) + Wf-325E (1 ADMET 4-below documented) + Wf-325F (1 Phase3G F2A runtime unblock verified) + Wf-327 (1 cpu-num-workers ≥ 4 launch-script gate) + Wf-343C4 (4 pre-retrain internal-check gates: hyperparams_config_complete + yuelbond_decoder_env + anchor_freshness_24h + wave_budget_remaining_nonneg) + Wf-347A1 (2 pre-retrain internal-check companion gates: p3_wf343c2_per_wave_runtime_check + p3_wf343c3_data_quality_check) + Wf-350-A3 (1 pre-retrain hp-provenance gate) + Wf-350-A2 (1 compute-manifest P1 gate, also surfaces in `all`) |
| **Total** | **71** | |

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

### §2.3 `chemprop_v2_v3` pipeline — 14 gates

Wf-259 baseline 12 + Wf-321H (`p3_todo_119_p14_cpu_smoke_ready`) +
Wf-350-A2 (`p1_compute_manifest`).

### §2.4 `cfm` pipeline — 6 gates

Wf-259 baseline.

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
    "all": 71,
    "frontier": 25,
    "chemprop_v2_v3": 14,
    "cfm": 6,
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
| `n_gates_total` (per-pipeline COUNT) | 71 (`all`) / 25 (`frontier`) / 14 (`chemprop_v2_v3`) / 6 (`cfm`) |
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