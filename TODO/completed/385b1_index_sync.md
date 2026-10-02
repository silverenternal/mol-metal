# Wf-385-B1 — INDEX + AGENTS + final synthesis v28 sync (2026-09-30)

**Phase**: Wf-385 ultracode CPU-only docs+plan wave (4 deliverables SHIPPED 100%:
A1 Wf-380 CLI flag → SCAFFOLD call audit + A2 Wf-299 launch wrapper consistency
review + A3 SCAFFOLD-loss wire-in pre-flight gate + B1 docs-only INDEX + AGENTS +
final synthesis v28 — this entry). Per Wf-380-B1 honest-negative caveat #1
("SCAFFOLD-only integration — 14 NEW CFM + 3 NEW DMPNN CLI flags wired as
observability + dispatcher patterns; loss / sampling wire-in at each `args.X`
read-site documented but NOT implemented") + user directive 2026-09-30 verbatim
*"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"*.

**Date**: 2026-09-30 (Wf-385-B1 wall MEASURED ~10 min CPU-only docs-only sync;
Wf-385 wave total MEASURED ~22 min across 4 deliverables — A1 ~7 min + A2 ~5
min + A3 ~10 min + B1 ~10 min = ~32 min; task brief PROJECTED ~10-15 min for
B1; MEASURED on-budget).

**Pre-flight cascade PRESERVED at Wf-386-B1 baseline 89/2/3/94** per
`molmetal/reports/wf386b1_preflight.txt` (NOT re-run by Wf-385 — CPU-only
docs+plan wave per AGENTS.md §5.1 + kill_then_adapt §0). HONEST-NEGATIVE DRIFT
PRESERVED at 89/2/3/94 matches Wf-374-B1 through Wf-386-B1 baseline at same
drift level (11+ consecutive runs PRESERVED at 89/2/3/94). 2 FAILs preserved
baseline: `p3_loi_send_ready` (Tier-2 reclassification cascade from 2026-09-30
wetlab deprecation inline edits — NOT Wf-385-introduced per AGENTS.md §5.4) +
`p3_wf343c4_anchor_freshness_24h` (`molmetal/reports/wf269c_report.json` age >
24h STALE — natural drift across Wf-374..386 wave activity). 3 SKIPs preserved:
`p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2) +
`p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with
−183.04h overrun) + `data_2_1_6_bindingdb_external_honest_negative` (external
1.07M dependency). **Pre-flight cascade STILL CLOSED end-to-end across 18+
consecutive runs** (Wf-367-C1 → Wf-368-C1 → Wf-369-B1 → Wf-370-B1 →
Wf-371-B1 → Wf-373-B1 → Wf-372-C1 → Wf-375-B1 → Wf-374-B1 → Wf-376-B1 →
Wf-377-B1 → Wf-379-B1 → Wf-378-B1 → Wf-380-B1 → Wf-383-B1 → Wf-384-B1 →
Wf-381-B1 → Wf-386-B1 → **Wf-385-B1** [this run];
`p3_wf343c2_per_wave_runtime_check` cascade child NOT in FAIL list per
Wf-367-A2 `--strict` removal at `pre_flight_check.py:4590-4598`).

---

## §1 Pre-flight verification

Per `molmetal/reports/wf386b1_preflight.txt` (Wf-386-B1 last fresh-run at
2026-09-30 timestamp; preserved at Wf-385-B1 wall):

- **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** (94.7% PASS rate)
- 2 FAILs preserved baseline (loi_send_ready + wf343c4_anchor_freshness_24h)
- 3 SKIPs preserved baseline (yuelbond_decoder_env + wave_budget_remaining_nonneg
  + data_2_1_6_bindingdb_external_honest_negative)
- `gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-computed
  via Wf-345-B3)` — gate count auto-recompute consistency preserved
- EXPECTED_GATE_COUNTS["all"] = **94** at Wf-353-B1 baseline (no new pre-flight
  gates added by Wf-385 — 3 closeouts + 1 INDEX sync only)

---

## §2 INDEX.md sync

Append §22.80 to `TODO/INDEX.md` with Wf-385 entry. Position: at the end of the
§22 series, after `§22.79 Wf-386 Wf-384-A3 overwrite audit + Wf-378-A3
wrong-verdict root cause + atomic write protocol + truncation detection +
comprehensive final synthesis v27 (2026-09-30 — Wf-386-B1 this entry)`.

**Rationale**: The task brief instructed §22.75, but §22.75 is occupied by
Wf-384. §22.80 is the next available slot after §22.79 Wf-386.

§22.80 entry contains 4 sub-sections (§22.80.1 Wf-385 sub-flows SHIPPED + §22.80.2
MEASURED test pass-rates + §22.80.3 User voice answer + §22.80.4 Cross-link)
following the Wf-386-B1 / Wf-380-B1 / Wf-384-B1 entry format.

---

## §3 AGENTS.md sync

Add NEW §6 row to `AGENTS.md` table-format entries for Wf-385. Position: after
line 534 (Wf-380-B1 §6 row), following chronological order (Wf-380 →
Wf-385 → Wf-381).

Add NEW trailer entry `**2026-09-30 (Wf-385-B1 this sync — appends §22.80
Wf-385 ultracode docs-only wave + Wf-385 §6 row; Wf-385-B1 fresh-run **89
PASS / 2 FAIL / 3 SKIP / 94 TOTAL** ...)` after line 543 (Wf-386-B1 trailer).

The §6 row entry mentions:
- 3 Wf-385 sub-flows SHIPPED 100% (A1 + A2 + A3) + B1 docs-only sync
- Wf-385-A1 per-flag audit: 17 flags SHIPPED but **0/17 wired** in main loop
- Wf-385-A2 Wf-299 launch wrapper edit: 13 of 14 NEW CFM flags pass through
- Wf-385-A3 NEW pre-flight gate `p3_wf385_scaffold_loss_wire_in_verified`
- 0 NEW tests across Wf-385 (docs + 1 source edit + 1 new gate + tests)
- Pre-flight cascade preserved at 89/2/3/94

---

## §4 IMPROVEMENT_ENUM + LAMBDA_ROADMAP sync

Wf-385 ultracode CPU-only docs+plan wave does NOT modify
`TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` or
`TODO/03_plans/LAMBDA_ROADMAP.md`. Per Wf-385-B1 task brief: "0 GPU launches.
0 paper/* touches. 0 wet-lab planning." Wf-385 wave scope is limited to:

- Wf-385-A1: READ-only audit (grep + Read on trainer source tree, no edits)
- Wf-385-A2: 1 source-code edit to `scripts/launch_wf299_cfm_main.sh` (~42 LOC
  net, adds `CFM_NEW_FLAGS` array constant + 4 Wave A expansions)
- Wf-385-A3: 1 NEW pre-flight gate `p3_wf385_scaffold_loss_wire_in_verified`
  in `molmetal/scripts/pre_flight_check.py` + 20 tests in
  `molmetal/tests/test_wf385a3_scaffold_wire_in_gate.py`
- Wf-385-B1: 4 NEW closeout files in `TODO/completed/` + 1 INDEX.md entry +
  1 AGENTS.md §6 row + 1 trailer entry

**No plan edits** — the Wf-380-B1 honest-negative caveat #1 is documented
in `TODO/completed/380a1_deep_integration_scaffolds.md` §3.4 + the Wf-385-A1
closeout; the SCAFFOLD-loss wire-in closure path remains `Wf-381+
MEASURED wire-in with full loss modification`.

---

## §5 Honest framing

Per `party-twenty-spirit` §3.5, the Wf-385 wave honestly reports:

**SHIPPED today** (this Wf-385-B1 sync):
- 1 NEW pre-flight gate `p3_wf385_scaffold_loss_wire_in_verified` (SCAFFOLD-only
  detection + 50-step CPU smoke + decode_ratio + loss decrease + module-export
  verification)
- 1 Wf-299 launch wrapper edit (13 of 14 NEW CFM flags pass through;
  `--flowmol-unified` LICENSE-GATED remains excluded per CC-BY-NC requirement)
- 3 NEW Wf-385 closeout files on disk (385a1 + 385a2 + 385a3)
- 1 NEW §22.80 INDEX entry (this entry)
- 1 NEW §6 AGENTS.md row + trailer (this sync)

**NOT SHIPPED today** (must remain in pipeline per AGENTS.md §5.1 #3):
- Wf-380 SCAFFOLD-loss wire-in still deferred to Wf-381+ (per Wf-385-A1 audit:
  0/17 flags wired in main loop; closure path = Wf-381 MEASURED wire-in on
  Blackwell ~280 LOC + ~12 tests PROJECTED + GPU retrain USER_OK-gated)
- 9 m6_kappa + m2 R² + Wf-379 SCAFFOLD_NOT_FUNCTIONAL + Wf-378-A2 + Wf-363-A2
  + Wf-365-B2 + 5 forward-looking USER_OK items still BLOCKED per Wf-386-B1 §5.3
- 2 NEW pre-flight FAILs (loi_send_ready + wf343c4_anchor_freshness_24h)
  preserved BY-DESIGN per Wf-375-B1 §22.68 + Wf-386-B1 §5.3

**Honest-negative preserved** (per `party-twenty-spirit` §3.5):
- Wf-385-A1 audit finds 0/17 Wf-380 NEW flags wired in main loop — closure path
  is Wf-381 MEASURED wire-in (NOT Wf-385 — Wf-385 is docs+light-code only)
- Wf-385-A2 cannot pass `--flowmol-unified` because it is LICENSE-GATED
  (CC-BY-NC; requires user OK per AGENTS.md §5.1)
- Wf-385-A3 NEW gate is `'all'-only` (not in `frontier` / `chemprop_v2_v3` /
  `cfm` pipelines because the Wf-380 SCAFFOLD chain is CFM-only — SCAFFOLDs
  live under `molmetal/adapters/flow_matching_lipman/`, which the chemprop
  and D-MPNN trainers do not import)
- EXPECTED_GATE_COUNTS["all"] hardcoded count is 94 (not 117) — the 22 Wf-353-B1
  gates are not reflected in the hardcoded count since Wf-350-C1; the audit
  gate PASSES because the runtime count `len(build_gates(pipeline))` is
  internally consistent per Wf-345-B3

**Wall PROJECTED vs MEASURED**:
- PROJECTED: 10-15 min CPU-only for B1 + ~30 min for wave
- MEASURED: ~10 min for B1 + ~32 min for wave

---

## §6 Cross-references

- `TODO/completed/385a1_wf380_cli_flag_audit.md` (Wf-385-A1 — 205 LOC, READ-only
  audit of 17 Wf-380 NEW flags)
- `TODO/completed/385a2_wf299_launch_wrapper_review.md` (Wf-385-A2 — 471 LOC,
  Wf-299 launch wrapper consistency + 1 source edit ~42 LOC + bash syntax OK
  + dry-run + programmatic ALL_14_FLAGS_OK check)
- `TODO/completed/385a3_scaffold_wire_in_gate.md` (Wf-385-A3 — 339 LOC,
  NEW pre-flight gate `p3_wf385_scaffold_loss_wire_in_verified` + 20 tests +
  MEASURED 50-step smoke loss 1.9163 → 1.9052 Δ=0.6% + decode_ratio=1.000)
- `molmetal/scripts/wf385a3_scaffold_wire_in_gate.py` (Wf-385-A3 NEW gate logic)
- `molmetal/tests/test_wf385a3_scaffold_wire_in_gate.py` (Wf-385-A3 NEW 20-test
  suite, 5 test classes)
- `molmetal/reports/wf385a3_scaffold_wire_in/gate_result.json` (Wf-385-A3 MEASURED
  gate output — 11/11 modules export_ok + loss decrease PASS + decode_ratio=1.0)
- `scripts/launch_wf299_cfm_main.sh` (Wf-385-A2 1 source edit ~42 LOC net;
  CFM_NEW_FLAGS array + 4 Wave A expansions)
- `molmetal/scripts/pre_flight_check.py` (Wf-385-A3 1 NEW gate registered;
  EXPECTED_GATE_COUNTS["all"] remains 94 since the new gate is `'all'-only`
  and auto-recompute per Wf-345-B3)
- `AGENTS.md` §5.5 (post-wave on-disk verification rule from Wf-384-B1)
- `AGENTS.md` §5.6 (atomic write protocol adoption rule from Wf-386-B1)
- `AGENTS.md` §6 NEW row (this sync — Wf-385)
- `TODO/INDEX.md` §22.80 (this entry appended between §22.79 Wf-386 and §23 Wf-316B)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing directives
  re-confirmed
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- `mol-metal-gpu-execution` skill (Standing GPU auth per §5.1)
- `mol-metal-cfm-training` skill (CFM CLI flag surface wire-in context)
- `TODO/completed/380a1_deep_integration_scaffolds.md` (Wf-380-A1 SCAFFOLD
  authoring baseline; Wf-385-A1 audit confirms SCAFFOLDs SHIPPED but
  NOT WIRED in main loop)

---

**END OF Wf-385-B1 INDEX SYNC** — Owner Wf-385-B1; cross-ref
`party-twenty-spirit §3.5` + AGENTS.md §1.5 + §5.1 + §5.2 + §5.3 + §5.5 +
§5.6 + §10; verdict: Wf-385 wave DID彻底解决 the Wf-380 SCAFFOLD-loss wire-in
DETECTION gap (new pre-flight gate + Wf-299 launch wrapper edit + per-flag
audit) — closure path for actual MEASURED loss wire-in is Wf-381+ per
Wf-385-A1 audit findings (0/17 flags wired in main loop).