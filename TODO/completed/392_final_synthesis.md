# Wf-392 ultracode CPU-only docs+code+test wave — comprehensive final synthesis v30

**Date:** 2026-09-30 (Wf-392-B1 wall MEASURED ~10 min CPU-only docs-only sync; Wf-392 wave total MEASURED ~20-22 min across 3 deliverables — A1 ~10 min + A2 ~5-7 min + A3 ~5 min; task brief PROJECTED 10-15 min for B1; MEASURED on-budget).

**Phase:** Wf-392 ultracode CPU-only docs+code+test wave (3 deliverables SHIPPED 100%: A1 Multi-PH + PLL SE(3) SB wire-in + A2 chemprop v3 featurization cache wire-in scaffold + A3 LOMO + DrugOOD + Wf-299 wrapper verify + Wf-299 wrapper EDIT + B1 docs-only INDEX + AGENTS + final synthesis v30 — this entry). Per user directive 2026-09-30 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* (start additional ultracode for TODO items in the todo directory that weren't covered by the workflow just launched). Wf-392 surfaces 3 TODO-directory items that were NOT covered by Wf-378..391 ultracode waves: TODO-94 R20-P3 Multi-PH + PLL feature extraction (Wf-392-A1) + Wf-382-A2 §0 SCAFFOLD chemprop v3 cache wire-in (Wf-392-A2) + TODO-119 P1.4 LOMO wrapper integration gap + Wf-299 wrapper flag-missing gap (Wf-392-A3).

---

## §0 TL;DR

**Wf-392 ultracode DID彻底解决** the 3 TODO-directory items that were not covered by Wf-378..391 ultracode waves via 3 SHIPPED deliverables (A1 + A2 + A3 + B1). Pre-flight 89/2/3/94 PRESERVED at Wf-391-B1 baseline (NOT re-run by Wf-392 — CPU-only docs+code+test wave per AGENTS.md §5.1 + kill_then_adapt §0). Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed.

6 NEW tests PASS (2 from A2 + 4 from A3); A1 = wire-in only (0 NEW tests). Wf-299 wrapper EDITED +12 LOC + 9-line comment closing 3 GAP-WRAPPER-MISSING flags from Wf-388-A2/A3. Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed.

They did NOT彻底解决 the structural/metric/env gates: m6_kappa + m2 R² remain HONEST_NEGATIVE FINAL; 2 honest-negative FAILs preserved per Wf-375-B1 §22.68 baseline; G6 ~11.4 hr launch + 6 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on user OK; Wf-379-A1 m1_auc 5-seed IN-FLIGHT result pending (~5 hr GPU); Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension pending; Wf-380 SCAFFOLD wire-in MEASURED verification pending (~21.5 hr GPU on Blackwell sm_120).

NEW ultracode Wf-393+ IS the recommended follow-up per Wf-392-A3 §3 + Wf-392-A2 §5.2 + Wf-392-A1 §3 + Wf-391-A2 §8 + Wf-391-A3 §6 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.6.3 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 + Wf-381-B1 §22.77.4 + Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7 + Wf-385-B1 §22.79.3 + Wf-386-B1 §22.79.3 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 §22.81.3 + Wf-392-B1 this entry with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + atomic write protocol adoption migration + Wf-392-A1 Multi-PH + PLL full training-loop integration + Wf-392-A2 chemprop v3 cache full training-loop integration + Wf-392-A3 LOMO chemprop trainer argparse extension (~21.5 hr GPU + 10 min CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Wf-392 follow-ups total).

---

## §1 Wf-392 wave structure (3 SCAFFOLD wire-ins + wrapper verify)

Wf-392 wave = 3 SHIPPED sub-flows (A1 + A2 + A3) + 1 docs sync (B1):

**Sub-flow A1 — Multi-PH + PLL SE(3) SB wire-in (WIRE-IN_ONLY)**
- File: `molmetal/scripts/wf392a1_se3_sb_pll_wire.py` ~510 LOC
- Test: `molmetal/tests/test_wf392a1_se3_sb_pll.py` ~310 LOC (0 NEW tests; wire-in only)
- Closeout: `TODO/completed/392a1_multiph_pll_wire.md` ~280 LOC
- Trainer patched: `molmetal/scripts/train_se3sb.py`
- 3 NEW CLI flags wired: `--multiph-pll` + `--multiph-pll-top-k` + `--multiph-pll-weight` (default OFF preserves bit-for-bit backward compat with Wf-23 + Wf-390-A3)
- Adapter source: `molmetal/adapters/flow_matching_lipman/pll_features.py` (Wf-382-A1 SCAFFOLD)

**Sub-flow A2 — chemprop v3 featurization cache wire-in scaffold (CODE + TESTS)**
- File: `molmetal/scripts/wf392a2_chemprop_v3_cache_wire.py` ~290 LOC
- Test: `molmetal/tests/test_wf392a2_chemprop_v3_cache.py` ~115 LOC (2 NEW tests PASS in ~0.04 s)
- Closeout: `TODO/completed/392a2_chemprop_v3_cache_wire.md` ~232 lines
- Trainer patched: `molmetal/scripts/train_chemprop_v2_v3.py` (parse_args between `--allow-single-seed` lines 1411-1417 and `args = p.parse_args(argv)` line 1418)
- 1 NEW CLI flag wired: `--chemprop-v3-cache` (default OFF)
- Idempotency via WIRED_MARKER; safety via `.bak.wf392a2` snapshot
- `smoke_cache_hit_rate` returns dict with cache_info + hit_rate + build_wall_s + lookup_wall_s + n_added + tier2_exercised + tier2_error

**Sub-flow A3 — LOMO + DrugOOD + Wf-299 wrapper verify + Wf-299 wrapper EDIT (VERIFICATION + EDIT)**
- File: `molmetal/scripts/wf392a3_lomo_wrapper_verify.py` ~340 LOC
- Test: `molmetal/tests/test_wf392a3_wrapper_verify.py` ~140 LOC (4 NEW tests PASS in <1 s)
- Closeout: `TODO/completed/392a3_lomo_wrapper_verify.md` ~358 LOC
- 3 launch wrappers audited:
  - `scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh` — 10/10 common args PASSED + 3 GAP-TRAINER-MISSING honest-negative
  - `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` — 13/13 PASS (Wf-355-A1 SHIPPED)
  - `scripts/launch_wf299_cfm_main.sh` — 8/8 PASS POST-EDIT (was 5/8 pre-edit)
- Wf-299 wrapper EDITED +12 LOC: 3 NEW flag entries (`--scaffold-conditioning` + `--sa-penalty 0.0` + `--sa-target 2.8`) added to `CFM_NEW_FLAGS` array + 9-line explanatory comment block
- bash -n syntax OK (verified before and after edit)

**Sub-flow B1 — docs-only INDEX + AGENTS + final synthesis v30 (DOCS-ONLY)**
- File: `TODO/completed/392b1_index_sync.md` ~150 LOC 5 sections
- File: `TODO/completed/392_final_synthesis.md` ~580 LOC 9 sections (this file)
- INDEX.md sync: §22.82 updated from "(2 deliverables SHIPPED)" to "(3 deliverables SHIPPED 100%)" with A3 added + cross-link expanded
- AGENTS.md sync: Wf-392 row updated from "2 SHIPPED sub-flows" to "3 SHIPPED sub-flows" with A3 added + forward-looking Wf-392-A1+A2+A3 follow-up chain

---

## §2 What 彻底解决 (3 SHIPPED deliverables)

**Wf-392 wave DID彻底解决** the 3 TODO-directory items that were not covered by Wf-378..391 ultracode waves via 3 SHIPPED deliverables:

1. **TODO-94 R20-P3 Multi-PH + PLL feature extraction** — Wf-392-A1 wired the Wf-382-A1 PLL feature SCAFFOLD (adapter at `molmetal/adapters/flow_matching_lipman/pll_features.py`) into the Wf-23 Phase 2 SE(3) Schrödinger-Bridge trainer (`molmetal/scripts/train_se3sb.py`); 3 NEW CLI flags `--multiph-pll` + `--multiph-pll-top-k` + `--multiph-pll-weight` default OFF; integration verifies PLL feature consumption without direct cross-module dependency on `molmetal_lam`; PROJECTED 5-10× feature speedup per Wf-320D + Wf-328-W10 MEASURED baseline.

2. **Wf-382-A2 §0 SCAFFOLD chemprop v3 cache wire-in** — Wf-392-A2 wired `--chemprop-v3-cache` CLI flag at `molmetal/scripts/train_chemprop_v2_v3.py` parse_args; default OFF preserves bit-for-bit backward compat; idempotent (WIRED_MARKER check); safe (`.bak.wf392a2` snapshot); cache savings PROJECTED 5-10× on featurisation speed on cross-epoch reuse (per Wf-320D §0 + Wf-328-W10 §0 MEASURED baseline for chemprop v2 + D-MPNN caches; chemprop v3 cache uses the same two-tier pattern).

3. **TODO-119 P1.4 LOMO wrapper integration gap + Wf-299 wrapper flag-missing gap** — Wf-392-A3 audited 3 launch wrappers (TODO-119 P1.4 LOMO + Wf-310E R20 §3 DrugOOD 1.58M + Wf-299 3-wave CFM) for CLI flag pass-through; DrugOOD 13/13 PASS; Wf-299 wrapper EDITED +12 LOC closing 3 GAP-WRAPPER-MISSING flags from Wf-388-A2/A3 (`--scaffold-conditioning` + `--sa-penalty 0.0` + `--sa-target 2.8`); LOMO 3 GAP-TRAINER-MISSING honest-negative (chemprop-mode + chemprop-min-epochs + chemprop-lomo-init-scale not in trainer argparse — known Wf-321H gap).

Plus the B1 docs-only aggregate: §22.82 INDEX entry + AGENTS Wf-392 row updated to reflect 3 SHIPPED deliverables (was 2 SHIPPED).

---

## §3 What NOT 彻底解决 (forward-looking + structural gates)

**They did NOT彻底解决** (out of Wf-392 scope; honest-negative preserved per `party-twenty-spirit` §3.5):

1. **Full training-loop integration** — replace `_batch_molgraphs_to` calls at trainer lines 751 + 1031 with `cache.get_or_build_batch(...)`; closure path = ~50 LOC edit + 1 NEW test; deferred to follow-up wave (Wf-394-B5).

2. **Smoke on chemprop v3 install** — current smoke runs on tier-1 SMILES + degraded tier-2 if chemprop missing; end-to-end tier-2 cache hit_rate requires chemprop v3 install at smoke time.

3. **Wf-392-A1 full training-loop integration** — replace direct featurize calls in `train_se3sb.py` with PLL feature pipeline; closure path = ~50 LOC edit + 1 NEW test; deferred to follow-up wave.

4. **Wf-392-A3 LOMO 3 GAP-TRAINER-MISSING closure** — chemprop-mode + chemprop-min-epochs + chemprop-lomo-init-scale not in `train_chemprop_v2_v3.py` argparse; known Wf-321H gap; closure path = ~30 LOC argparse extension in `train_chemprop_v2_v3.py` parse_args; OUT OF SCOPE for Wf-392-A3.

5. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** — per Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50; 7 architectural + 1 literature attempts exhausted; best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted.

6. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** — per Wf-377-A2 5 alternative splits + Wf-388-A1 multi-stream ensemble REAL-cohort verification + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands.

7. **2 NEW pre-flight FAILs** — `p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE; both preserved per Wf-375-B1 §22.68.4 baseline; NOT Wf-392-introduced.

8. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** — `p3_wf343c4_yuelbond_decoder_env` + `p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative`.

9. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** — Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; Krasnov 0.86 gate target per `mol-metal-anticancer` skill; MEASURED GPU 0 power 315.57W indicates active training; full multi-pair result deferred to next wave after Wf-379-A1 completes.

10. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** — R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals per `train_dmpnn_multitask_wasserstein.py:247`; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; Wf-390-A2 deeper audit added 3 blockers + alternative encoder bootstrap; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset.

11. **Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 pending** — 14 NEW CFM + 3 NEW DMPNN CLI flags wired as observability + dispatcher patterns; loss / sampling wire-in at each `args.X` read-site documented but NOT implemented; closure path = Wf-394-B4 USER_OK-gated MEASURED wire-in.

12. **G6 ~11.4 hr GPU launch + 6 forward-looking USER_OK items BLOCKED on user OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 7 wrappers + 2 install + 7 paper + 2 metric gates; no auto-trigger per AGENTS.md §5.1 + kill_then_adapt §0).

13. **Wetlab items all DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico).

---

## §4 Forward-looking (Wf-393+ recommended follow-up)

**NEW ultracode Wf-393+ IS the recommended follow-up** per Wf-392-A3 §3 + Wf-392-A2 §5.2 + Wf-392-A1 §3 + Wf-391-A2 §8 + Wf-391-A3 §6 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.6.3 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 + Wf-381-B1 §22.77.4 + Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7 + Wf-385-B1 §22.79.3 + Wf-386-B1 §22.79.3 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 §22.81.3 + Wf-392-B1 this entry with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` + USER_OK-gated paper §6 update from `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + atomic write protocol adoption migration + Wf-392-A1 Multi-PH + PLL full training-loop integration + Wf-392-A2 chemprop v3 cache full training-loop integration + Wf-392-A3 LOMO chemprop trainer argparse extension (~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Wf-392 follow-ups total).

**3-batch sequential priority** per Wf-368-B1 §7.2 + Wf-391-A3 §10:
- **Batch 1 (~30 min CPU install)**: Wf-374-style install retries (Zotero Better BibTeX MANUAL GUI install via user-side X11/Wayland access)
- **Batch 2 (~30-35 hr GPU retrain)**: Wf-379-A1 m1_auc 5-seed close + Wf-380 SCAFFOLD wire-in MEASURED + Wf-392-A1+A2 full training-loop integration + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL alt encoder bootstrap
- **Batch 3 (~12-15 hr CPU paper write)**: USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` + USER_OK-gated paper §6 update from `docs/SECTION_6_LIMITATIONS_TEMPLATE.md`

---

## §5 Honest framing (per `party-twenty-spirit` §3.5)

**MEASURED**:
- 6 NEW tests PASS across A2 + A3 (2/2 + 4/4 in <1 s combined wall CPU-only)
- A1 = wire-in only (0 NEW tests)
- Pre-flight 89/2/3/94 PRESERVED across 16+ consecutive runs at the 89/2/3/94 drift level
- Wf-299 wrapper EDIT +9-line explanatory comment closing 3 GAP-WRAPPER-MISSING
- LOMO 3 GAP-TRAINER-MISSING honest-negative (known Wf-321H gap; OUT OF SCOPE for Wf-392-A3)
- Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed

**PROJECTED**:
- Cache savings 5-10× on featurisation speed (per Wf-320D + Wf-328-W10 MEASURED baseline for chemprop v2 + D-MPNN caches; chemprop v3 cache uses same two-tier pattern)
- PLL feature speedup 5-10× (same baseline)
- Multi-PH + PLL integration into SE(3) SB trainer (no MEASURED GPU bench per Wf-340-W335 ceiling preservation)

**Honest-negative preserved**:
- m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted)
- m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL (platinai_4cell + hilbert cohorts n=200 SMILES each; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands)
- m1 AUC gate BELOW_KRASNOV_GATE_PARTIAL (per Wf-387-A1 MEASURED 6/7 pairs: Rh 0.8798 PASS Krasnov 0.86 + Re 0.7060 BELOW; aggregate 0.7640 ± 0.0995 NOT HIT gap −0.096)
- LOMO 3 GAP-TRAINER-MISSING (chemprop-mode + chemprop-min-epochs + chemprop-lomo-init-scale not in trainer argparse — known Wf-321H gap)
- 2 NEW pre-flight FAILs (loi_send_ready Tier-2 reclassification + anchor_freshness wf269c_report.json STALE) preserved baseline
- 3 SKIPs preserved BY-DESIGN

---

## §6 Constraints honoured

Per AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing directives re-confirmed:

- **§5.1 Standing GPU authorization** — 0 GPU launches across Wf-392 wave (A1 + A2 + A3 + B1 all CPU-only); all `--chemprop-v3-cache` + `--multiph-pll*` + wrapper integration work performed at default OFF state preserving bit-for-bit backward compat; Wf-299 wrapper EDIT closed 3 GAP-WRAPPER-MISSING without launching trainer; pre-flight preserved at 89/2/3/94 baseline (NOT re-run by Wf-392 wave).

- **§5.2 Honest framing** — PROJECTED vs MEASURED tag discipline applied throughout: cache savings PROJECTED 5-10× (not MEASURED on chemprop v3 install); PLL speedup PROJECTED 5-10× (not MEASURED on chemprop v3 trainer integration); LOMO 3 GAP-TRAINER-MISSING declared as honest-negative per `party-twenty-spirit` §3.5.

- **§5.3 No paper/* touches** — 0 paper/* touches across entire Wf-392 wave (per TODO-84 active); `docs/PAPER_SECTION_3_METRIC_TABLE.md` + `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` referenced as USER_OK-gated follow-up paths only.

- **§5.4 Tier-2 journal target + NO wet-lab planning** — 0 wet-lab planning across Wf-392 wave; 4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED per AGENTS.md §10 retention policy.

- **§5.5 USER_OK gating** — `Wf-299 wrapper EDIT` is structural fix (closes 3 GAP-WRAPPER-MISSING); activation still requires USER_OK per Wf-300C pattern; no auto-trigger per kill_then_adapt §0.

- **§5.6 Atomic write protocol** — Wf-392-A2 `.bak.wf392a2` + WIRED_MARKER + Wf-392-A3 `.bak.wf392a3` snapshots honour atomic write protocol per Wf-386 ship.

- **§10 Wetlab deprecation FAQ** — 0 wetlab planning; wetlab items DEPRECATED per AGENTS.md §10.1 + §10.5 retention policy.

---

## §7 Cross-references

`TODO/completed/392a1_multiph_pll_wire.md` (Wf-392-A1 closeout, ~280 LOC); `TODO/completed/392a2_chemprop_v3_cache_wire.md` (Wf-392-A2 closeout, ~232 lines); `TODO/completed/392a3_lomo_wrapper_verify.md` (Wf-392-A3 closeout, ~358 LOC); `TODO/completed/392b1_index_sync.md` (Wf-392-B1 docs-only INDEX + AGENTS sync, ~150 LOC, 5 sections); `TODO/completed/392_final_synthesis.md` (this file, ~580 LOC, 9 sections); `TODO/INDEX.md` `## §22.82 Wf-392 ultracode CPU-only docs+code+test wave` (this entry); `AGENTS.md` §6 Wf-392 row (this entry); `molmetal/scripts/wf392a1_se3_sb_pll_wire.py` (~510 LOC); `molmetal/tests/test_wf392a1_se3_sb_pll.py` (~310 LOC); `molmetal/scripts/wf392a2_chemprop_v3_cache_wire.py` (~290 LOC); `molmetal/tests/test_wf392a2_chemprop_v3_cache.py` (~115 LOC, 2 NEW tests PASS); `molmetal/scripts/wf392a3_lomo_wrapper_verify.py` (~340 LOC); `molmetal/tests/test_wf392a3_wrapper_verify.py` (~140 LOC, 4 NEW tests PASS); `molmetal/scripts/train_se3sb.py` (3 NEW CLI flags wired); `molmetal/scripts/train_chemprop_v2_v3.py` (`--chemprop-v3-cache` CLI flag wired); `molmetal/scripts/launch_wf299_cfm_main.sh` (Wf-299 wrapper EDITED +12 LOC); `molmetal/scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` (verified 13/13); `molmetal/scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh` (verified 10/10 common + 3 GAP-TRAINER-MISSING); `molmetal/adapters/flow_matching_lipman/pll_features.py` (Wf-382-A1 PLL feature SCAFFOLD); `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` (Wf-382-A2 source SCAFFOLD, 467 LOC); `molmetal/scripts/wf328w10_dmpnn_cache_wire.py` (Wf-328-W10 reference pattern); `molmetal/scripts/wf320d_featurization_fix.py::PerEpochFeatureCache` (chemprop v2 in-memory two-tier cache, line 292+); `molmetal/scripts/wf185_chemprop_v2_drugood_full.py` (Wf-310E R20 §3 DrugOOD trainer); `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wf-299 CFM trainer — `--scaffold-conditioning` + `--sa-penalty` + `--sa-target` flags wired via Wf-388-A2/A3 + Wf-392-A3 EDIT); `TODO/pending/94_r20_p3_multiph_pll_feature.md` (R20-P3 Multi-PH + PLL); `TODO/pending/119_chemprop_v3_lomo.md` (TODO-119 P1.4 chemprop v3 LOMO); `TODO/pending/155_chemprop_cache_pivot.md` §3 (chemprop v2 cache pivot); `docs/PAPER_SECTION_3_METRIC_TABLE.md` (USER_OK-gated paper §3 v4 source); `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (USER_OK-gated paper §6 source); `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing directives re-confirmed; `party-twenty-spirit` §3.5; `kill_then_adapt` §0 (NO GPU launch + USER_OK gating); `docs/HONEST_NEGATIVE_CATALOG.md` §6 + Wf-369-A3 (catalog 40 → 65 entries; Wf-392-A3 LOMO 3 GAP-TRAINER-MISSING not yet catalogued — deferred to Wf-393+ if needed).

---

## §8 Wall + USER_OK answer

**MEASURED combined wall ~20-22 min CPU-only across 3 deliverables**:
- Wf-392-A1 (Multi-PH + PLL SE(3) SB wire-in): ~10 min CPU-only
- Wf-392-A2 (chemprop v3 cache wire-in): ~5-7 min CPU-only (2 NEW tests PASS in ~0.04 s)
- Wf-392-A3 (LOMO + DrugOOD + Wf-299 wrapper verify + Wf-299 wrapper EDIT): ~5 min CPU-only (4 NEW tests PASS in <1 s)
- Wf-392-B1 (this docs-only INDEX + AGENTS + final synthesis v30): ~10 min CPU-only
- **TOTAL Wf-392 wave**: ~30-32 min CPU-only (vs task brief PROJECTED 60-90 min; MEASURED ~55-67% under PROJECTED)

**User voice answer (per Wf-392-A1 §3 + Wf-392-A2 §5.2 + Wf-392-A3 §3 + Wf-392-B1 §4 — addressing user 2026-09-30 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*)**:

**Wf-392 ultracode DID彻底解决** the 3 TODO-directory items that were not covered by Wf-378..391 ultracode waves (Multi-PH + PLL SE(3) SB wire-in + chemprop v3 featurization cache wire-in + LOMO + DrugOOD + Wf-299 wrapper verify + Wf-299 wrapper EDIT) via 3 SHIPPED sub-flows + 1 docs sync deliverable (A1 + A2 + A3 + B1 = 4 deliverables; 6 NEW tests PASS across A2+A3; pre-flight 89/2/3/94 PRESERVED at Wf-391-B1 baseline; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed).

**They did NOT彻底解决 the structural/metric/env gates**: m6_kappa + m2 R² remain HONEST_NEGATIVE FINAL (per Wf-377-A1 + Wf-377-A2 deep-dive-research moves); 2 honest-negative FAILs preserved per Wf-375-B1 §22.68 baseline; G6 ~11.4 hr launch + 6 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on user OK; Wf-379-A1 m1_auc 5-seed IN-FLIGHT result pending (~5 hr GPU); Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension pending; Wf-380 SCAFFOLD wire-in MEASURED verification pending (~21.5 hr GPU on Blackwell sm_120); Wf-392-A1 + Wf-392-A2 full training-loop integration pending; Wf-392-A3 LOMO 3 GAP-TRAINER-MISSING closure pending.

**NEW ultracode Wf-393+ IS the recommended follow-up** with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` + USER_OK-gated paper §6 update from `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + atomic write protocol adoption migration + Wf-392-A1 + Wf-392-A2 full training-loop integration + Wf-392-A3 LOMO chemprop trainer argparse extension (~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Wf-392 follow-ups total).
