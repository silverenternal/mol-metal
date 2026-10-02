# Wf-288D — Verify Cf-263C G5/G6 Configs (Post-Fix)

**Date:** 2026-09-29
**Workflow:** Wf-288D (verify Cf-263C G5/G6 3-deviation fix status)
**Owner:** Hugo Lee + ultracode agent
**Inputs:**
- `molmetal/reports/wf263c_audit/g5_g6_audit.md` — original 3 deviations list
- `configs/wf124_p3_50epoch_full_sweep_g5.yaml` — G5 (current)
- `configs/wf124_p3_per_innovation_ablation_g6.yaml` — G6 (current)
- `scripts/launch_wf124_p3_g5.sh`, `scripts/launch_wf124_p3_g6.sh` — launch wrappers
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — trainer CLI

---

## TL;DR (FLAT schema verdict)

| Gate | Value | Verdict |
|---|---|---|
| **G1_original_deviations** | 3 | AUDITED — see `wf263c_audit/g5_g6_audit.md` §"Gap Inventory" |
| **G2_n_fixed** | **3** | **ALL 3 original deviations FIXED** (CSV×2 + epochs×1) |
| **G3_n_remaining** | 0 | none — gap inventory closed |
| **G4_n_new_deviations** | 0 | none introduced |
| **G5_hygiene_nits** | 2 | doc-only (not config deviations) — see §6 |

**Headline honest-negative:** Wf-264 (config update) + Wf-265 (dashboard) + Wf-266 (LOMO) +
Wf-280A (no-enforce-budget) + Wf-283B (kendall/dann CLI flags) + Wf-285B (kendall/dann wiring)
have **fully closed the original 3-deviation gap inventory**. Both configs now use the
v4 CSV + 50-epoch protocol as specified. No new deviations introduced.

---

## §1 Per-deviation status (n_original = 3)

### Deviation #1 — G5 CSV path v2 → v4 (HIGH severity)

| Aspect | Original (Wf-263C audit) | Current (Wf-288D verify) |
|---|---|---|
| Expected | `metalcytotox_5metal_multitask_v4.csv` | (same) |
| Actual (audit) | `.data/molmetal/metalcytotox_5metal_multitask_v2.csv` (line 28) | **FIXED** — `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (line 28 of G5 yaml) |
| **Status** | DEVIATION | **FIXED** ✓ |
| Fix evidence | — | (a) `configs/wf124_p3_50epoch_full_sweep_g5.yaml` line 28 references v4<br>(b) `scripts/launch_wf124_p3_g5.sh` line 30 `DATA_CSV=…metalcytotox_5metal_multitask_v4.csv`<br>(c) dry-run log `wf263c_verify/g5_dry_run.log` line ~8: `DATA_CSV: /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv (26633 rows, 11 cols, OK)` |
| Fix workflow | — | Wf-264 (config update) — confirmed by `molmetal/reports/wf264_audit/` (paper_4_claim_audit.md) |

### Deviation #2 — G6 CSV path v2 → v4 (HIGH severity)

| Aspect | Original (Wf-263C audit) | Current (Wf-288D verify) |
|---|---|---|
| Expected | `metalcytotox_5metal_multitask_v4.csv` | (same) |
| Actual (audit) | `.data/molmetal/metalcytotox_5metal_multitask_v2.csv` (line 27) | **FIXED** — `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (line 27 of G6 yaml) |
| **Status** | DEVIATION | **FIXED** ✓ |
| Fix evidence | — | (a) `configs/wf124_p3_per_innovation_ablation_g6.yaml` line 27 references v4<br>(b) `scripts/launch_wf124_p3_g6.sh` line 34 `DATA_CSV=…metalcytotox_5metal_multitask_v4.csv`<br>(c) dry-run log `wf263c_verify/g6_dry_run.log`: `DATA_CSV: /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv (26633 rows, OK)` |
| Fix workflow | — | Wf-264 (config update) |

### Deviation #3 — G6 epochs 30 → 50 (MEDIUM severity)

| Aspect | Original (Wf-263C audit) | Current (Wf-288D verify) |
|---|---|---|
| Expected | `epochs: 50` | (same) |
| Actual (audit) | `epochs: 30` (line 52 of G6 yaml) | **FIXED** — `epochs: 50` (line 53 of G6 yaml) |
| **Status** | DEVIATION | **FIXED** ✓ |
| Fix evidence | — | (a) `configs/wf124_p3_per_innovation_ablation_g6.yaml` line 53: `epochs: 50                         # per TODO-182 §2 (50-epoch full sweep)`<br>(b) `scripts/launch_wf124_p3_g6.sh` line 39: `--epochs 50` in `COMMON_ARGS`<br>(c) `wf263c_verify/g6_dry_run.log`: `COMMON_ARGS: … --epochs 50 …` |
| Fix workflow | — | Wf-264 (config update) — `protocol.epochs: 50` |

---

## §2 Cascade fixes (post-audit, NOT counted as new deviations)

These were tracked as separate workflows; Wf-288D confirms their downstream effects
do not introduce config drift:

| Workflow | Effect on G5/G6 configs |
|---|---|
| **Wf-265 dashboard** | Dashboard `265.v1` schema reflects the 3 fixes; no config drift. |
| **Wf-266 LOMO fix** | Adds `--chemprop-min-epochs`, `--chemprop-lomo-init-scale`, `--chemprop-mode v3-fix` levers; **not in G5/G6 entry command** (orthogonal concern, deferred to Wf-267). |
| **Wf-266 G5 launch** | Launch wrapper attempted; 15/15 pairs failed with `CUDA error: invalid device ordinal` (cuda:1 not visible). **GPU-ordinal issue, not a config deviation**. Per `wf266_final/final_status.md` §4. |
| **Wf-280A no-enforce-budget** | YAML `enforce_budget: false` and `directives.no_enforce_budget: true` now present (G5 lines 123, 196; G6 line 225 + implicit via `directives`). CLI `--no-enforce-budget` flag still NOT in `entry.command` block — see §6 hygiene nit. |
| **Wf-283B kendall/dann CLI** | Added `--no-kendall` + `--no-dann` to trainer argparse (now in `--help`). |
| **Wf-285B kendall/dann wiring** | Wired `args.kendall` → `mtl_loss(use_kendall=…)` (line 1493) and `args.dann` → `MultiTaskWassersteinModel(..., use_dann=…)` (line 970) + `mtl_loss(use_dann=…)` (line 1494). Both flags now actually take effect on real launch. |
| **Wf-266 verify v3-fix** | LOMO-fix levers verified; no impact on G5/G6 yaml. |

---

## §3 New deviations introduced (n_new = 0)

No new config deviations detected. Specifically verified:

| Item | Check | Status |
|---|---|---|
| G5 yaml — `data.csv` path | `metalcytotox_5metal_multitask_v4.csv` (line 28) | OK |
| G5 yaml — `protocol.epochs` | `50` (line 57) | OK |
| G5 yaml — `cohort_filter.metals` | `[Ru, Ir, Os, Rh, Re]` (line 32) | OK |
| G5 yaml — `cohort_filter.scaffold_split` | `true` (line 34) | OK |
| G5 yaml — `dry_run` + `user_ok_required` | `true` + `true` | OK |
| G5 yaml — `waves.enforce_budget` | `false` (line 123) | OK |
| G6 yaml — `data.csv` path | `metalcytotox_5metal_multitask_v4.csv` (line 27) | OK |
| G6 yaml — `protocol.epochs` | `50` (line 53) | OK |
| G6 yaml — `cohort_filter.metals` | `[Ru, Ir, Os, Rh, Re]` (line 31) | OK |
| G6 yaml — `cohort_filter.scaffold_split` | `true` (line 33) | OK |
| G6 yaml — `dry_run` + `user_ok_required` | `true` + `true` | OK |
| G6 yaml — `ablation_configs.no_kendall.status` | still `"verify-flag-before-launch"` (line 103) | **STALE DOC** — see §6 nit #1 |
| G6 yaml — `ablation_configs.no_dann.status` | still `"verify-flag-before-launch"` (line 113) | **STALE DOC** — see §6 nit #1 |
| Trainer CLI — `--no-kendall` / `--no-dann` in `--help` | present (per `wf283b_g6_flags_fix/fix_report.md` §4.1) | OK |
| Trainer code — `args.kendall` → `mtl_loss(use_kendall=…)` | wired at line 1493 | OK |
| Trainer code — `args.dann` → `model(use_dann=…)` | wired at line 970 + 1494 | OK |
| Launch wrappers — `DATA_CSV` paths | both reference v4 (g5 line 30, g6 line 34) | OK |
| Launch wrappers — `--epochs 50` | both in `COMMON_ARGS` (g5 line 35, g6 line 39) | OK |
| `--no-enforce-budget` CLI flag in entry command | **NOT present** (g5 line 34-50, g6 line 38-52) | **SEE §6 NIT #2** |

---

## §4 Summary scoreboard

```
n_deviations_original = 3
n_fixed               = 3   (G5 CSV, G6 CSV, G6 epochs)
n_remaining           = 0
n_new                 = 0
n_hygiene_nits        = 2   (stale "verify-flag-before-launch" status fields + missing --no-enforce-budget CLI flag in entry command)
```

---

## §5 Per-deviation status (compact, for parent parser)

```
deviation_1_g5_csv_v2_to_v4: FIXED (Wf-264; G5 yaml line 28 + launch_wf124_p3_g5.sh line 30 + dry_run.log)
deviation_2_g6_csv_v2_to_v4: FIXED (Wf-264; G6 yaml line 27 + launch_wf124_p3_g6.sh line 34 + dry_run.log)
deviation_3_g6_epochs_30_to_50: FIXED (Wf-264; G6 yaml line 53 + launch_wf124_p3_g6.sh line 39 + dry_run.log)
```

---

## §6 Hygiene nits (NOT config deviations; doc-only)

These were called out in the Wf-263C audit's "Recommended Fixes" section but were
not counted in the 3-deviation gap inventory. They are **nice-to-haves** and do not
block any honest framing.

### Nit #1 — G6 yaml `ablation_configs.no_kendall.status` / `no_dann.status` are stale

- Current text: `status: "verify-flag-before-launch"` (G6 yaml lines 103, 113).
- Reality: Wf-283B added `--no-kendall` + `--no-dann` to trainer argparse;
  Wf-285B wired them into the loss/model. Both flags are now confirmed-functional.
- Recommendation: update both status fields to `status: "verified-by-wf283b+wf285b"`
  and remove the `verify-flag-before-launch` caveat from `entry.pre_launch_checklist`
  (G6 yaml lines 256-259). Doc-only; no behaviour change.

### Nit #2 — `--no-enforce-budget` CLI flag NOT in entry command

- Current state: YAML `enforce_budget: false` + `directives.no_enforce_budget: true`
  are present (G5 lines 123, 196), but the `entry.command` block does NOT include
  `--no-enforce-budget` (G5 lines 208-221, G6 lines 239-251).
- Wf-263C audit §"Honest Framing Note" explicitly stated this was NOT a deviation
  (achieved via `dry_run: true` instead).
- Recommendation: append `--no-enforce-budget` to both `entry.command` blocks for
  honest-framing explicitness at CLI level. Optional; current `dry_run: true` is
  sufficient.

---

## §7 Honest Framing (party-twenty-spirit §3.5)

- This verify task is **READ-ONLY** on configs; no source modifications were applied.
- The 3 original deviations are now confirmed FIXED via three independent artifacts:
  the YAML config, the launch wrapper shell script, and the dry-run log.
- The cascade fixes (Wf-265 / Wf-266 / Wf-280A / Wf-283B / Wf-285B) did NOT introduce
  any new config-level deviations; they only added orthogonal levers (LOMO fix) or
  flag surface (kendall/dann).
- G5 50-epoch real launch is **NOT validated** here — that requires GPU + user OK.
  The Wf-266 G5 launch attempt (`wf266_launch_g5/launch_report.md`) failed due to
  CUDA device-ordinal mismatch (`cuda:1` not visible). This is a hardware/environment
  issue, not a config deviation.
- Per `wf263c_verify/verify_report.md` §"Honest Framing Note", real launch of either
  G5 or G6 remains **BLOCKED on user OK + GPU availability** (kill_then_adapt
  posture).

---

## §8 Files Touched (this Wf-288D verify)

- READ: `molmetal/reports/wf263c_audit/g5_g6_audit.md`
- READ: `configs/wf124_p3_50epoch_full_sweep_g5.yaml`
- READ: `configs/wf124_p3_per_innovation_ablation_g6.yaml`
- READ: `scripts/launch_wf124_p3_g5.sh` (lines 30-50)
- READ: `scripts/launch_wf124_p3_g6.sh` (lines 30-72)
- READ: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (lines 965-972, 1485-1499, 1590-1591)
- READ: `molmetal/reports/wf263c_verify/verify_report.md`
- READ: `molmetal/reports/wf283b_g6_flags_fix/fix_report.md`
- READ: `molmetal/reports/wf266_final/final_status.md` (lines 1-120)
- WRITE: `molmetal/reports/wf288d_cf263c_verify/verify_report.md` (this file)
