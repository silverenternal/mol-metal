# Wf-396-B1 docs-only INDEX + AGENTS sync — decode_ratio root cause analysis aggregation

**Date**: 2026-09-30
**Wf**: Wf-396-B1 (docs-only INDEX + AGENTS sync delivery + comprehensive final synthesis v36)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED); no `paper/*` touches (TODO-84); no wet-lab planning (§10)
**Verbatim user directive (2026-09-30)**: *"这些问题都要解决，我给你授权，之前decode_ratio是1的现在我去哪退化的这个要重点解决。全部启动ultracode去做，我给你授权"* — focus on solving decode_ratio 1 → 0 regression; CPU-only docs+code+test; ultracode wave OK.

---

## §0 Phase

Wf-396 ultracode CPU-only docs-only wave (4 deliverables SHIPPED 100%):
- **Wf-396-A1** git-bisect (closed at `TODO/completed/396a1_decode_ratio_regression_git_bisect.md` ~9 KB)
- **Wf-396-A2** trainer diff analysis (closed at `TODO/completed/396a2_trainer_diff_analysis.md` ~24 KB)
- **Wf-396-A3** minimal reproduction (closed at `TODO/completed/396a3_minimal_reproduction.md` ~18 KB)
- **Wf-396-B1** this docs-only INDEX + AGENTS + final synthesis aggregation (this entry)

---

## §1 Pre-flight verification

Pre-flight NOT re-run by Wf-396-B1 (CPU-only docs-only wave per AGENTS.md §5.1 + `kill_then_adapt` §0). Pre-flight cascade PRESERVED at Wf-399-B1 / Wf-397-B1 / Wf-398-B1 baseline **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** per `molmetal/reports/wf399b1_preflight.txt` (NOT re-run by Wf-396-B1 — CPU-only docs-only wave).

**HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** matches Wf-374-B1 through Wf-399-B1 + Wf-397-B1 + Wf-398-B1 baseline at same drift level (25+ consecutive runs PRESERVED at 89/2/3/94).

**2 FAILs pre-existing baseline (NOT Wf-396-introduced)**:
- `p3_loi_send_ready` Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits
- `p3_wf343c4_anchor_freshness_24h` wf269c_report.json age > 24h STALE — natural drift from Wf-374..399 wave activity

**3 SKIPs preserved (BY-DESIGN)**:
- `p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2
- `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun
- `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency

---

## §2 INDEX.md sync

Appends `## §22.89 Wf-396 ultracode CPU-only docs+aggregation wave — decode_ratio regression root cause analysis aggregation (combined A1 + A2 + A3 closeouts) + comprehensive final synthesis v36` (NEW ~85 lines appended between `## §22.88 Wf-398 ultracode docs+data+verify wave` and `**END OF §22**`) to `TODO/INDEX.md`.

**Section number**: §22.89 (selected as the next available sequential number; §22.86 = Wf-399, §22.87 = Wf-397, §22.88 = Wf-398 are all taken). The task brief specified §22.86 but this conflicts with Wf-399 which already holds §22.86; Wf-396-B1 uses §22.89 to avoid collision while maintaining chronological ordering.

---

## §3 AGENTS.md sync

Appends NEW Wf-396-B1 §6 row to `AGENTS.md` (after the Wf-397-B1 §6 row at line 662). The new row captures the 4 SHIPPED deliverables (A1 + A2 + A3 + B1 all SHIPPED 100%) + 3 NEW tests + 1 combined root cause doc + 1 INDEX sync + 1 AGENTS row + 1 final synthesis.

**last-update trailer extended with Wf-396-B1 sync clause** (this row appended to last-update trailer per existing pattern from Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 + Wf-390-B1 + Wf-387-B1 + Wf-399-B1 + Wf-397-B1 + Wf-398-B1 prior syncs).

---

## §4 Honest framing

Per `party-twenty-spirit §3.5`: **6 honest-negative caveats preserved** per Wf-396-A1 + Wf-396-A2 + Wf-396-A3 closeouts + Wf-397-A1 + Wf-398-A3 + Wf-399-A1 + Wf-399-A2 chain:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 9 paths exhausted per Wf-389-A1 §3 + Wf-381-A4 + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB + Wf-365-A3 verdict doc + Wf-371-A1 + Wf-375-A4; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted)

2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands)

3. **m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL** (per Wf-387-A1; Rh fold PASS_KRASNOV 0.8798 + Re fold BELOW_KRASNOV 0.7061 + overall 0.7640 BELOW_KRASNOV 0.86)

4. **Wf-396-A3 CRITICAL FINDING: scaffold_conditioning has shape mismatch bug** (per `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py::scaffold_condition` expects `(B, N, V)` but trainer passes `(E, V)`; raises `ValueError` immediately; latent GPU-launch bug — not a decode_ratio regression but a separate code bug; closure path = reshape logits before `scaffold_condition` call per Wf-396 §4.2)

5. **Wf-396-A2 §2 CRITICAL FINDING REFUTED by Wf-396-A3** (Wf-380 SCAFFOLD banner combination does NOT regress decode_ratio at 50-step scale; MEASURED dr=1.0 in Path B `combination_wf380_scaffold_banner`; Wf-380 banner was the most likely suspect but empirical confirmation rules it out at 50-step scale)

6. **Wf-296..Wf-379 forward-looking USER_OK items still BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates; no auto-trigger per §5.1 + `kill_then_adapt` §0)

---

## §5 Cross-references

### §5.1 Source paths

| Path | Result | Source |
|---|---|---|
| `TODO/completed/396a1_decode_ratio_regression_git_bisect.md` | git-bisect findings | `cat` output |
| `TODO/completed/396a2_trainer_diff_analysis.md` | trainer diff triage | `cat` output |
| `TODO/completed/396a3_minimal_reproduction.md` | reproduction results | `cat` output |
| `molmetal/scripts/wf396a3_minimal_reproduction.py` | ~280 LOC | `wc -l` output |
| `molmetal/tests/test_wf396a3_reproduction.py` | 3 NEW tests | `stat` output |
| `molmetal/reports/wf396a3_minimal_reproduction/result.json` | 13 runs MEASURED | `cat` output |
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | working tree +330/-1 LOC | `git diff HEAD --stat` output |
| `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py` | shape mismatch source | `grep -n` output |
| `metrics/by_metric/decode_ratio.json` | historical record | `cat` output |
| `metrics/by_metric/cfm_decode_real.json` | secondary metric | `cat` output |

### §5.2 User voice answer (verbatim)

> *"这些问题都要解决，我给你授权，之前decode_ratio是1的现在我去哪退化的这个要重点解决。全部启动ultracode去做，我给你授权"*

**Translation**: All these problems need solving, I'm authorizing you. The decode_ratio was 1 before, now where did it degrade — focus on solving this. Launch ultracode for everything, I'm authorizing you.

**Wf-396 + parallel ultracode wave answer (Wf-396 + Wf-397 + Wf-398 + Wf-399)**:

**YES — DID彻底解决 the `decode_ratio=1 → 0` regression-of-focus** via the 4-wave parallel ultracode chain:

1. **Wf-396 (this wave)**: root cause identified as DATA-INTEGRITY (aggregate metric field drift 0.0) NOT a model regression. Unstaged Wf-388-A2 + Wf-388-A3 trainer diff is the regression suspect set; 8 NEW CLI flags default OFF (mathematically bit-exact identical to Wf-243..Wf-252 HEAD); empirical reproduction confirms decode_ratio=1.0 PRESERVED at default-CLI invocation; one CRITICAL FINDING: `scaffold_conditioning` has a shape mismatch bug (separate from the decode_ratio regression).

2. **Wf-397**: decode_ratio fix + regression-guard tests (Wf-397-A1 4/4 NEW AST-level tests PASS in ~0.5 s wall CPU-only + 12-line trainer pipeline edit) + Wave 1 GPU smoke decode_ratio=1.000 (32/32) on GPU 0 Blackwell 7.4 s end-to-end (Wf-397-A2) + Wave 2 + Wave 3 SCAFFOLD_NOT_FUNCTIONAL honest-negative close (Wf-397-A3).

3. **Wf-398**: aggregate metric field fix (Wf-398-A3: `decode_ratio.json::aggregate.all_measured_cfm_decode_ratio` 0.0 → **0.998** + `cfm_decode_real.json::honest_framing` PARTIAL → PASS).

4. **Wf-399**: AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (Wf-399-A1 + `molmetal/scripts/wf399a1_smoke_protocol.py` NEW ~340 LOC) + HONEST_NEGATIVE_CATALOG §6.26 NEW REGRESSION_DOCUMENTED entry (Wf-399-A2: catalog 81 → 82 entries).

---

**END OF Wf-396-B1 docs-only sync** — last update 2026-09-30 (Wf-396-B1 sync — 5 sections: Phase + Pre-flight verification + INDEX.md sync + AGENTS.md sync + Honest framing + Cross-references; Wf-396 SHIPPED 4/4 deliverables 100%; pre-flight 89/2/3/94 PRESERVED at Wf-397-B1 baseline; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10 re-confirmed; **last-update trailer extended with Wf-396-B1 sync clause**).
