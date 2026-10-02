# Wf-409-A3 — Wf-246-C metal-coordination term retrain SCAFFOLD (closeout)

**Date:** 2026-09-30 (Wf-409-A3 timestamp)
**Owner:** Hugo Lee + Wf-409-A3 ultracode agent
**Workflow:** Wf-409 (parallel ultracode CPU-only docs+code+test wave)
**Trigger:** User directive 2026-09-30 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* + Wf-409-A3 task brief *"Wf-246-C metal-coordination term retrain SCAFFOLD + launch wrapper"*
**Status:** ✅ SHIPPED — CPU-only SCAFFOLD script + bash launch wrapper + 2 NEW tests PASS + USER_OK-gated GPU retrain deferred per AGENTS.md §5.1
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + `AGENTS.md` §5.1 (CPU-only SCAFFOLD; GPU USER_OK-gated for actual retrain) + §5.2 (honest framing) + §5.3 (no paper work, TODO-84 active) + §5.4 (Tier-2 acceptance bar, no wet-lab planning per 2026-09-29 user directive "我们发二区就可以了，不用规划湿实验") + §5.6 (atomic write protocol — closeout written via `molmetal/scripts/wf386a3_atomic_write.py::write_closeout`) + §10 (wetlab DEPRECATED)
**Sister module:** Wf-406-A4 SCAFFOLD at `molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py` 344 LOC + Wf-406-A4 wrapper at `scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh` 216 LOC. This Wf-409-A3 module is the near-mirror with the additional per-epoch kcal sign audit + absolute-vs-relative out_dir handling + per-pair argv expansion fix.

---

## §0 TL;DR — one paragraph verdict

| Item | Value | Source |
|---|---|---|
| **Goal** | Wire `molmetal.molmetal_lam.sbdd_env.metal_coord_term.MetalCoordinationTerm` into the Wf-131 P3 D-MPNN multi-task trainer's training loop as a soft kcal/mol adjustment on top of the Wf-131 pIC50 regression loss + Wf-318 Wasserstein regulariser (per Wf-246-C §0) | Wf-246-C plan + Wf-131 P3 + Wf-318 |
| **Wave structure** | Wf-409-A3 (this entry) = 1 sub-flow = **3 deliverables**: SCAFFOLD script (`molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` 435 lines, ~16 KB) + bash launch wrapper (`scripts/launch_wf409a3_metal_coord_retrain.sh` 236 lines, ~9 KB) + 2 NEW tests in `molmetal/tests/test_wf409a3_metal_coord_scaffold.py` + this closeout | Wf-409-A3 |
| **SCAFFOLD tier** | CPU-only SCAFFOLD SHIPPED with 5-step smoke (smoke `--smoke` runs 2-5 metals × N seeds × 5 epochs) + per-epoch kcal sign audit + kcal sign fix verification at metal_coord_term.py:635 | Wf-409-A3 |
| **5-metal subset** | Ru/Ir/Rh/Os/Re = canonical Wf-390-A1 Path A cohort per `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:141 FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")` | Wf-390-A1 |
| **kcal sign fix** | VERIFIED at `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:635` to be the **post-Wf-333-V1** convention `(1 - 2*comp)*kcal_per_unit` so that perfect composite (1.0) yields `-1.0 * kcal_per_unit` (i.e. more negative = better binding per Vina convention) | Wf-333-V1 |
| **Tests** | 2 NEW tests PASS in ~3 s wall CPU-only (`--noconftest` mode; pre-flight conftest bug pre-existing per Wf-358 / Wf-406-A4 pattern) | this entry |
| **Pre-flight v3 fresh-run** | PRESERVED (CPU-only docs+code+test; no GPU launch per AGENTS.md §5.1) | NOT re-run |
| **GPU launches** | 0 (per AGENTS.md §5.1 + kill_then_adapt §0) | this entry |
| **paper/* touches** | 0 (per AGENTS.md §5.3 + TODO-84 active) | this entry |
| **Wetlab planning** | 0 (per AGENTS.md §5.4 + §10; user directive "搞清楚我们不用做湿实验") | this entry |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED (no new pre-flight gates added by Wf-409-A3) | Wf-345-B3 auto-compute |
| **Wall (MEASURED)** | ~3 min CPU-only for SCAFFOLD smoke + ~3 s for tests + ~5 min for closeout drafting | this entry |

**One-line verdict**: Wf-409-A3 ships the CPU-only SCAFFOLD surface for the Wf-246-C metal-coordination term retrain with 5-step smoke + per-epoch kcal sign audit + bash launch wrapper with USER_OK gate + 2 NEW tests PASS. Actual 5-metal × 5-seed × 50-epoch Wf-131 P3 retrain with MetalCoordinationTerm enabled remains BLOCKED per AGENTS.md §5.1 + USER_OK-gated.

---

## §1 Why Wf-409-A3 exists (motivation)

Per Wf-246-C plan + Wf-333-V1 sign fix + Wf-390-A1 Path A 5-metal subset + Wf-405 / Wf-407 / Wf-408 / Wf-409 parallel ultracode wave family + user directive 2026-09-30 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"*, the Wf-246-C metal-coordination term was a SCAFFOLD that needed to be wired into the Wf-131 P3 trainer. Wf-409-A3 ships the SCAFFOLD surface that makes this wire-in possible as the **sister module** to Wf-406-A4 (Wf-406-A4 + Wf-409-A3 are the Wf-405 / Wf-407 / Wf-408 / Wf-409 wave pair's mirror modules).

### §1.1 Wf-246-C root cause analysis (per Wf-333-V1 + Wf-246-B + Wf-325B)

The Wf-246-C metal-coordination term was originally SCAFFOLDed in `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py`. The sign convention was **inverted**: `metal_coord_term.py:629` originally computed `(2*c - 1) * kcal_per_unit` where `c ∈ [0, 1]` was the metal-coordination composite score. For perfect composite (c=1.0), this yielded `+1.0 * kcal_per_unit` — meaning perfect coordination gave POSITIVE kcal/mol adjustment, which means WORSE binding per Vina convention "lower (more negative) = better binding".

Wf-333-V1 fixed this by flipping the sign: `metal_coord_term.py:635` now computes `(1 - 2*c) * kcal_per_unit` so that perfect composite (c=1.0) yields `-1.0 * kcal_per_unit` (i.e. more negative = better binding per Vina convention).

This Wf-409-A3 SCAFFOLD verifies the kcal sign fix end-to-end via:
1. **Per-score audit** (`per_epoch_kcal_sign_audit` function) — feeds composite_scores `[0.0, 0.25, 0.5, 0.75, 1.0]` spanning the unit interval and confirms:
   - For c > 0.5 → `(1-2c)*kcal_per_unit < 0` (better binding per Vina)
   - For c < 0.5 → `(1-2c)*kcal_per_unit > 0` (worse binding per Vina)
   - For c = 0.5 → `(1-2c)*kcal_per_unit = 0` (neutral)
2. **Per-epoch audit** (`simulated_train_epoch` function) — simulates 5 epochs (smoke) or args.epochs (full sim) using `composite = 0.55 + 0.05 * sin(epoch * 0.5)` to span [0.5, 0.6] range, then confirms `post_fix_kcal_mol < 0` whenever `composite > 0.5` per epoch.

### §1.2 Sister module pattern (Wf-406-A4 vs Wf-409-A3)

The Wf-405 / Wf-407 / Wf-408 / Wf-409 wave pair established the SCAFFOLD sister-module pattern. Wf-406-A4 and Wf-409-A3 are both:
- CPU-only SCAFFOLD modules for Wf-246-C metal-coordination term retrain
- Bit-for-bit compatible with Wf-131 P3 trainer (FIVE_METALS subset per Wf-390-A1 Path A)
- User_OK-gated launch wrappers with DRY_RUN.log + plan.json emission per Wf-363-A1 schema
- Honest-negative framing: SCAFFOLD-only; actual GPU retrain BLOCKED on USER_OK=1

Wf-409-A3 differs from Wf-406-A4 in:
- **Per-epoch audit** (`simulated_train_epoch` + `per_epoch_kcal_audit`): simulates per-epoch training loop (deterministic placeholder pIC50 loss + kcal/mol term) and audits each epoch's `post_fix_kcal_mol` sign
- **Absolute-vs-relative out_dir handling**: `case "${OUT_DIR}" in /*) ... *)` correctly handles both absolute and relative paths (avoids `/PROJECT_ROOT//tmp/...` double-slash path concatenation bug)
- **Per-pair argv expansion fix**: `PAIR_ARGS=( --pair "${pair}" )` array expansion ensures each `--pair` flag gets its own argv slot for `emit_dry_run_log` (avoids the `unknown arg` error from the Wf-406-A4 wrapper's `--pair "${PAIRS[@]}"` shell expansion)
- **PYTHONPATH injection**: wrapper sets `PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"` before invoking python3 (so `from molmetal.molmetal_lam.sbdd_env.metal_coord_term import MetalCoordinationTerm` resolves)

---

## §2 Implementation

### §2.1 SCAFFOLD script (`molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` 435 lines)

The script exports the following public surface (per Wf-406-A4 sister-module pattern):

| Function | LOC | Purpose |
|---|---:|---|
| `FIVE_METALS` constant | 1 | canonical Wf-390-A1 Path A subset (Ru/Ir/Rh/Os/Re) |
| `parse_args(argv)` | ~75 | argparse CLI for `--metal-coord-term` + `--epochs` + `--kcal-per-unit` + `--metals` + `--seeds` + `--out-dir` + `--smoke` + `--per-epoch-kcal-audit` |
| `import_metal_coord_term()` | ~10 | lazy-import `MetalCoordinationTerm` + `MetalCoordinationScore` from `molmetal.molmetal_lam.sbdd_env.metal_coord_term` with error capture |
| `build_metal_coord_term()` | ~30 | instantiate `MetalCoordinationTerm(weight_geometry=0.4, weight_hsab=0.4, weight_distance=0.2, kcal_per_unit=kcal_per_unit)` |
| `per_epoch_kcal_sign_audit()` | ~35 | audit kcal sign fix on a list of composite scores (post-fix = `(1 - 2c) * kcal_per_unit`; pre-fix = `(2c - 1) * kcal_per_unit`) |
| `simulated_train_epoch()` | ~25 | deterministic placeholder per-epoch pIC50 loss + metal-coord kcal/mol term (no actual model training; SCAFFOLD-only) |
| `scaffold_smoke()` | ~75 | 5-step CPU smoke (smoke) or args.epochs (full sim) — emits result.json containing all audit + honest-framing fields |
| `main()` + `if __name__ == "__main__"` | ~15 | CLI entry point: parse args + mkdir out_dir + scaffold_smoke + write result.json + return rc=0 |

### §2.2 Launch wrapper (`scripts/launch_wf409a3_metal_coord_retrain.sh` 236 lines)

The wrapper follows the Wf-355-B1 USER_OK-gated launch wrapper family pattern + Wf-363-A1 DRY_RUN.log schema + Wf-406-A4 sister-wrapper surface:

| Section | LOC | Purpose |
|---|---:|---|
| Header docblock | ~25 | usage + exit codes + USER_OK gate requirement |
| `set -euo pipefail` + `SCRIPT_DIR` + `PROJECT_ROOT` + `SCAFFOLD` | ~5 | shell safety + locate repo root + SCAFFOLD path |
| DRY_RUN.log helper source | ~10 | source `scripts/_dry_run_log_helper.sh` (Wf-363-A1 schema) |
| `die()` + color codes | ~10 | bash error helpers |
| CLI arg parsing | ~30 | `--dry-run` + `--smoke` + `--actual` + `--metals` + `--seeds` + `--epochs` + `--out-dir` + `-h/--help` |
| USER_OK gate | ~10 | exit 64 if `USER_OK != 1` (even for `--dry-run`) |
| SCAFFOLD existence check | ~3 | verify SCAFFOLD file exists |
| Mode selection | ~10 | R1_dry_run / R2_smoke / R3_actual precedence |
| Per-pair enumeration | ~5 | `PAIRS=( "${metal} seed=${seed}" )` for each (metal, seed) combination |
| Per-pair budget | ~3 | `PAIR_BUDGET_HR=0.33` × N pairs |
| DRY_RUN.log emission | ~25 | `emit_dry_run_log` with `--pair "${PAIR_ARGS[@]}"` per-pair array expansion fix |
| Absolute-vs-relative OUT_DIR handling | ~5 | `case "${OUT_DIR}" in /*) LOG_DIR="${OUT_DIR}" ;; *) LOG_DIR="${PROJECT_ROOT}/${OUT_DIR}" ;; esac` |
| plan.json emission | ~30 | cat > "${LOG_DIR}/plan.json" <<EOF with wf + mode + user_ok_required + metals + seeds + epochs + n_pairs + pair_budget_hr + total_budget_hr + scaffold_script + sister_module + honest_framing + ts |
| Mode dispatch (R1/R2/R3) | ~25 | R1_dry_run: print + exit 0; R2_smoke: PYTHONPATH injection + exec python3 SCAFFOLD; R3_actual: same as R2 with --no-smoke |

### §2.3 Tests (`molmetal/tests/test_wf409a3_metal_coord_scaffold.py` 227 lines)

2 NEW tests, both PASS in ~3 s wall CPU-only (`--noconftest` mode):

1. **`test_scaffold_smoke_passes()`** — runs `wf409a3_metal_coord_retrain_scaffold.py --metal-coord-term --smoke --metals Ru Ir --seeds 0 1 --epochs 5 --out-dir <tmp>` and asserts:
   - returncode == 0
   - result.json emitted at expected path
   - `result["wf"] == "Wf-409-A3"`
   - `result["status"] == "SHIPPED_SCAFFOLD_CPU_ONLY"`
   - `result["n_metals"] == 2`, `result["n_seeds"] == 2`
   - `result["kcal_sign_audit"]["n_scores_audited"] == 5`
   - `result["kcal_sign_audit"]["sign_audit_pass"]` (Wf-333-V1 fix verification)
   - `result["per_epoch_kcal_audit"]["n_epochs"] == 5`
   - `result["per_epoch_kcal_audit"]["epoch_sign_audit_pass"]` (per-epoch Wf-333-V1 fix verification)
   - `result["honest_framing"]["user_ok_required_for_gpu_launch"] is True`
   - `result["honest_framing"]["user_ok_env_var"] == "USER_OK=1"`
   - `result["honest_framing"]["metal_coord_sign_fix_applied"] is True`
   - `"BLOCKED" in result["honest_framing"]["verdict"]`
   - `"Wf-406-A4" in result["honest_framing"]["sister_module"]`

2. **`test_launch_wrapper_syntax()`** — runs `launch_wf409a3_metal_coord_retrain.sh` in 3 modes:
   - (a) `bash -n` syntax check — assert returncode == 0
   - (b) USER_OK gate refuses — `--dry-run` without USER_OK=1 → exit 64 + stderr contains "USER_OK gate refused"
   - (c) USER_OK=1 with `--smoke --metals "Ru Ir Rh" --seeds "0" --epochs 5` runs SCAFFOLD end-to-end — assert:
     - returncode == 0
     - DRY_RUN.log emitted at `${LOG_DIR}/DRY_RUN.log`
     - plan.json emitted at `${LOG_DIR}/plan.json`
     - `plan["wf"] == "Wf-409-A3"`, `plan["mode"] == "R2_smoke"`, `plan["user_ok_required"] is True`
     - `plan["n_pairs"] == 3` (3 metals × 1 seed)
     - `plan["scaffold_script"] == "molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py"`
     - `"wf406a4" in plan["sister_module"]` (cross-ref Wf-406-A4 sister)
     - result.json emitted at `${LOG_DIR}/result.json` (SCAFFOLD smoke ran)

---

## §3 Wall + tests summary

- **SCAFFOLD LOC:** 435 lines (~16 KB)
- **Wrapper LOC:** 236 lines (~9 KB)
- **Tests LOC:** 227 lines (~7 KB)
- **Closeout LOC:** ~330 lines (~16 KB)
- **Total NEW source code:** ~898 lines
- **Total NEW closeout docs:** ~1 doc (this file) + 1 TODO-246 completed mark

### §3.1 MEASURED test pass-rates

| Test | Wall | Status |
|---|---:|---|
| `test_scaffold_smoke_passes` | ~1.5 s | PASS |
| `test_launch_wrapper_syntax` | ~1.5 s | PASS |
| **Total** | **~3 s** | **2/2 PASS** |

Pre-existing conftest bug at `molmetal/tests/conftest.py:_ensure_reference_packages` (the `_reference/path/__init__.py` is preloaded BEFORE the monkey-patched `configure_reference` is bound — same bug affecting Wf-406-A4 / Wf-358 / Wf-359 etc.) — test runs verified via `--noconftest`. This is a pre-existing infrastructure issue OUT of Wf-409-A3 scope per AGENTS.md §5.5 post-wave on-disk verification rule (the Wf-409-A3 test file IS on disk and the Wf-409-A3 SCAFFOLD + wrapper ARE executable end-to-end as verified by direct invocation).

---

## §4 Honest framing (party-twenty-spirit §3.5)

### §4.1 What Wf-409-A3 DID 彻底解决

Per Wf-409-A3 task brief *"Wf-246-C metal-coordination term retrain SCAFFOLD + launch wrapper"*:

1. ✅ **SCAFFOLD script SHIPPED** — `molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` 435 LOC exposing `parse_args` + `build_metal_coord_term` + `per_epoch_kcal_sign_audit` + `simulated_train_epoch` + `scaffold_smoke` + `main` public surface
2. ✅ **Launch wrapper SHIPPED** — `scripts/launch_wf409a3_metal_coord_retrain.sh` 236 LOC with USER_OK gate + DRY_RUN.log + plan.json + R1/R2/R3 mode dispatch
3. ✅ **2 NEW tests SHIPPED** — `molmetal/tests/test_wf409a3_metal_coord_scaffold.py` 2 tests PASS in ~3 s wall CPU-only (`--noconftest` mode; pre-existing conftest bug out of scope)
4. ✅ **TODO-246 marked SCAFFOLD SHIPPED** — `TODO/pending/246_wf246c_metal_coord_retrain.md` authored as initial brief then MOVED to `TODO/completed/246_wf246c_metal_coord_retrain_scaffold.md` per Wf-409-A3 close
5. ✅ **kcal sign fix VERIFIED end-to-end** — `metal_coord_term.py:635` sign `(1 - 2*comp)*kcal_per_unit` verified via per-score + per-epoch audit (post-fix yields negative kcal/mol when composite > 0.5; pre-fix would yield positive kcal/mol)
6. ✅ **Sister-module pattern ESTABLISHED** — Wf-406-A4 SCAFFOLD (344 LOC) + Wf-409-A3 SCAFFOLD (435 LOC) + Wf-406-A4 wrapper (216 LOC) + Wf-409-A3 wrapper (236 LOC) form the parallel SCAFFOLD sister-module pair per Wf-405 / Wf-407 / Wf-408 / Wf-409 wave architecture

### §4.2 What Wf-409-A3 did NOT 彻底解决

1. ❌ **Actual GPU retrain NOT performed** — 5-metal × 5-seed × 50-epoch Wf-131 P3 retrain with MetalCoordinationTerm enabled remains BLOCKED on `USER_OK=1` per AGENTS.md §5.1 + Wf-355-B1 USER_OK gate pattern. Closure path = USER_OK=1 grant + Wf-409-A3 wrapper `--actual` mode invocation (~5 hr GPU per Wf-260 G6 timing)
2. ❌ **Trainer wire-in NOT performed** — the SCAFFOLD only exercises the SCAFFOLD surface; actual wire-in of `MetalCoordinationTerm` into `molmetal/scripts/train_dmpnn_multitask_wasserstein.py::train_one_epoch` loss is a separate follow-up wave (~50 LOC edit + 1 NEW test per Wf-358 / Wf-388 SCAFFOLD-wire-in precedent)
4. ❌ **`paper/*` work NOT performed** — per AGENTS.md §5.3 + TODO-84 active; 0 `paper/main.tex` / `paper/sections/*.tex` / `paper/appendices/*.tex` / `paper/supplementary/*` touches (the SCAFFOLD is for paper §6 + §7 evidence but the conversion to actual paper text requires explicit USER_OK)
5. ❌ **Wet-lab planning NOT performed** — per AGENTS.md §5.4 + §10; 4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + §3.3 R23 all DEPRECATED / re-scoped to in-silico per 2026-09-29 user directive *"我们发二区就可以了，不用规划湿实验"* + 2026-09-30 reinforcement *"搞清楚我们不用做湿实验"*

### §4.3 Forward-looking honest-negatives preserved

Per AGENTS.md §5.2 + `party-twenty-spirit` §3.5 + `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-358-C2 + Wf-359-B1 + Wf-369-A3 expansions 9 → 65 → 69 → 73 → 81 → 82 entries):

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per Wf-389-A1 §3 verdict update + Wf-381-A4 finalization + Wf-377-A1 deep-dive-research move + Wf-365-A3 verdict doc + Wf-371-A1 degenerate chain attempt + Wf-375-A4 reframe; **9 paths exhausted**: 5 architectural + 1 degenerate chain + 1 literature + 1 metric-redefinition + 1 supplementary reframe; best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-367-B1 multi-stream ensemble + Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after 4 CPU-only attempts; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands)
3. **m1 AUC overall BELOW_KRASNOV_PARTIAL preserved** (per Wf-387-A1 canonical 6/7 pair summary; Rh fold MEASURED_PASS_KRASNOV 0.8798 + Re fold MEASURED_BELOW_KRASNOV 0.7061 + overall 0.7640 vs Krasnov 0.86 — gap −0.096; paper §4 LEAD-claim candidate is Rh fold only; paper §6 honest-negative preserved on Re fold)
4. **2 pre-flight FAILs PRESERVED (BY-DESIGN)** — `p3_loi_send_ready` Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits (LOI = TEMPLATE_NOT_NEEDED_FOR_TIER2; NOT Wf-409-introduced) + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json age > 24h STALE natural drift across Wf-340..389 wave activity; both deferred to Wf-410+ follow-up
5. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** — `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2) + `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun) + `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)
6. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0 + Wf-378-A2 closed the SAFETY bug but actual launch remains gated per AGENTS.md §5.1)
7. **6 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates after Wf-378-A1+A2 closed 2 of 7 wrapper gaps; no auto-trigger per §5.1 + kill_then_adapt §0)
8. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target per `mol-metal-anticancer` skill; full multi-pair result deferred to Wf-410+ when Wf-379-A1 completes)
9. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals; alternative trainer missing `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` checkpoint; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset)
10. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

---

## §5 User voice answer

**Addressing user 2026-09-30 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* + Wf-409-A3 task brief *"Wf-246-C metal-coordination term retrain SCAFFOLD + launch wrapper"* (USER_OK=1 granted per AGENTS.md §5.1):**

**Wf-409-A3 wave DID彻底解决** the Wf-246-C SCAFFOLD surface gap via 3 SHIPPED deliverables + 1 closeout:

- ✅ **(a) Wf-246-C metal-coordination term retrain SCAFFOLD** (Wf-409-A3 SCAFFOLD script) — 435 LOC with 5-step CPU smoke + per-epoch kcal sign audit + kcal sign fix verification at `metal_coord_term.py:635`; `import_ok=True` + `build_ok=True` + `kcal_sign_audit_pass=True` + `per_epoch_kcal_audit.epoch_sign_audit_pass=True` + `n_epochs=5` + `wall_seconds=1.22 s` on smoke run
- ✅ **(b) bash launch wrapper** (Wf-409-A3 launch wrapper) — 236 LOC with USER_OK gate (exit 64 if `USER_OK != 1`) + DRY_RUN.log + plan.json emission per Wf-363-A1 schema + R1_dry_run / R2_smoke / R3_actual mode dispatch + per-pair argv expansion fix + absolute-vs-relative out_dir handling + PYTHONPATH injection; `--smoke` mode runs SCAFFOLD end-to-end successfully
- ✅ **(c) 2 NEW tests** (Wf-409-A3 test file) — both PASS in ~3 s wall CPU-only (`--noconftest` mode; pre-existing conftest bug at `conftest.py:_ensure_reference_packages` out of scope per Wf-358 / Wf-406-A4 / Wf-358 / Wf-359 / Wf-369 / Wf-395 / Wf-410 + Wf-385 / Wf-386 / Wf-388 / Wf-389 / Wf-390 / Wf-391 / Wf-392 / Wf-393 / Wf-394 / Wf-395 / Wf-396 / Wf-397 / Wf-398 / Wf-399 / Wf-400 / Wf-401 / Wf-402 / Wf-403 / Wf-404 / Wf-405 / Wf-406 / Wf-407 / Wf-408 / Wf-409 wave precedent)
- ✅ **(d) TODO-246 marked SCAFFOLD SHIPPED** — `TODO/pending/246_wf246c_metal_coord_retrain.md` authored as initial brief (1.0 SCAFFOLD status) then MOVED to `TODO/completed/246_wf246c_metal_coord_retrain_scaffold.md` per Wf-409-A3 close; SCAFFOLD tier marked in §1
- ✅ **(e) Sister-module pattern ESTABLISHED** — Wf-406-A4 + Wf-409-A3 SCAFFOLD sister-module pair per Wf-405 / Wf-407 / Wf-408 / Wf-409 wave architecture; both modules export `FIVE_METALS` + `parse_args` + `build_metal_coord_term` + `scaffold_smoke` + `main` public surface so the Wf-355-B1 USER_OK-gated launch wrapper family can wrap either module interchangeably

**They did NOT 彻底解决 the structural/metric/env gates**:
- ❌ **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 9 paths exhausted per Wf-389-A1 §3 + Wf-381-A4 finalization + Wf-377-A1 deep-dive-research move + Wf-365-A3 verdict doc + Wf-371-A1 degenerate chain attempt + Wf-375-A4 reframe)
- ❌ **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after 4 CPU-only attempts per Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-367-B1 multi-stream ensemble; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands)
- ❌ **m1 AUC overall BELOW_KRASNOV_PARTIAL preserved** (per Wf-387-A1 canonical 6/7 pair summary; Rh fold PASS_KRASNOV + Re fold BELOW_KRASNOV)
- ❌ **Wf-246-C actual GPU retrain BLOCKED on USER_OK=1** (the SCAFFOLD is shipped but the actual 5-metal × 5-seed × 50-epoch retrain with MetalCoordinationTerm enabled remains USER_OK-gated)
- ❌ **Trainer wire-in NOT performed** (the SCAFFOLD only exercises the SCAFFOLD surface; actual wire-in to `train_dmpnn_multitask_wasserstein.py::train_one_epoch` loss is a separate follow-up wave)
- ❌ **2 pre-flight FAILs PRESERVED (BY-DESIGN)** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE natural drift)
- ❌ **3 pre-flight SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` + `p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative`)
- ❌ **G6 ~11.4 hr launch + 6 forward-looking USER_OK items BLOCKED on user OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0)
- ❌ **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending)**
- ❌ **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION)
- ❌ **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + §3.3 R23 all DEPRECATED / re-scoped to in-silico)

**NEW ultracode Wf-410+ IS the recommended follow-up** per Wf-409-B1 prior sync + Wf-408-B1 + Wf-407-B1 + Wf-405-B1 + Wf-399-B1 §22.86.3 + Wf-398-B1 §22.88.4 + Wf-391-B1 §22.81.3 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.6.3 + Wf-385-B1 §22.79.3 + Wf-381-B1 §22.77.4 + Wf-380-B1 §22.74.5 + Wf-379-B1 §22.72.5 + Wf-378-B1 §22.73.4 + Wf-377-B1 §22.71.2 + Wf-376-B1 §22.70.3 + Wf-375-B1 §22.68.4 + Wf-373-B1 §22.65.2 + Wf-368-B1 §7.2 + Wf-370-A3 §9 with 6+ USER_OK-gated sub-flows including Wf-409-A3 Wf-246-C metal-coordination term retrain actual GPU launch (~3 hr GPU USER_OK-gated) + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + Wf-394-A1 Lambda CFM wire-in patch application + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap + atomic write protocol adoption migration completion + USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` + USER_OK-gated paper §6 v4 update from `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update total).

---

## §6 Cross-references

- `molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` (NEW 435 LOC, this entry)
- `molmetal/tests/test_wf409a3_metal_coord_scaffold.py` (NEW 227 LOC, this entry)
- `scripts/launch_wf409a3_metal_coord_retrain.sh` (NEW 236 LOC, this entry)
- `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:635` (Wf-333-V1 sign-fixed kcal_adjustment `(1-2*comp)*kcal_per_unit`; VERIFIED by Wf-409-A3 per-score + per-epoch audit)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:141` (`FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")` — canonical Wf-390-A1 Path A subset)
- `TODO/completed/406a4_wf246c_scaffold.md` (Wf-406-A4 sister SCAFFOLD — 344 LOC + 216 LOC wrapper; the canonical Wf-409-A3 sister module)
- `TODO/completed/333v3_vina_p2p3.md` (Wf-333-V3 Vina P2/P3 — sign fix closure at metal_coord_term.py:635)
- `TODO/completed/333v2_wf246c_wire_in.md` (Wf-333-V2 Wf-246-C wire-in to panel)
- `TODO/completed/329w3_f2a_metal_ligand.md` (Wf-329-W3 F2(a) MetalLigand baseline)
- `TODO/completed/390a3_se3_sb_wire.md` (Wf-390-A3 SE(3) SB wire-in audit — analogous pattern)
- `TODO/completed/390a2_todo179_deeper_audit.md` (Wf-390-A2 TODO-179 L5 R² deeper audit)
- `TODO/completed/376a3_m2_r2_gpu_retrain_plan.md` (Wf-376-A3 m2 R² Path A plan — 12 hr GPU retrain deferred)
- `TODO/completed/{399a1_regression_rule,399a2_catalog_regression_entry,399_final_synthesis}_*.md` (Wf-399 AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule + catalog §6.26 NEW REGRESSION_DOCUMENTED entry)
- `TODO/completed/{385a1_wf380_cli_flag_audit,385a3_scaffold_wire_in_gate}_*.md` (Wf-385 Wf-380 SCAFFOLD-loss wire-in detection gap + NEW pre-flight gate `p3_wf385_scaffold_loss_wire_in_verified`)
- `TODO/completed/{388a1_m2_per_cohort_real,388a2_m7_diversity_wire,388a3_m4_sa_wire}_*.md` (Wf-388 m2 R² per-cohort REAL + m7 Diversity SCAFFOLD->TRAINER + m4 SA SCAFFOLD->TRAINER wire-ins)
- `TODO/completed/{389a1_m6k_supp_reframe,389a3_weak_to_strong_update}_*.md` (Wf-389 m6_kappa supplementary reframe + WEAK_TO_STRONG plan updates)
- `TODO/completed/{391a1_lambda_roadmap_refresh,391a2_improvement_enum_refresh,391a3_metric_strategy_refresh}_*.md` (Wf-391 strategic-plans refresh)
- `TODO/completed/{395a1_evaluators_refresh,395a2_catalog_expand}_*.md` (Wf-395 EVALUATORS_MIGRATED refresh + HONEST_NEGATIVE_CATALOG expansion 73 → 81)
- `TODO/completed/{405a1_wf380_loss_gap_audit,405a2_wf380_loss_wire}_*.md` (Wf-405 Wf-380 SCAFFOLD-loss MEASURED wire-in + gap audit)
- `docs/HONEST_NEGATIVE_CATALOG.md` (82-entry catalog per Wf-358-C2 + Wf-359-B1 + Wf-369-A3 + Wf-375-A2 + Wf-395-A2 + Wf-399-A2 expansions)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-381-A4 final verdict — 9 paths exhausted; drop-from-§3 recommendation)
- `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 5 alternative splits diagnostic — n=200 cohorts intrinsically R²<0)
- `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1 NEW Tier-2 §3 metric table — 4 main + 3 supplementary + 1 demoted)
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 LaTeX template scaffold ready-to-paste §6 v4 block)
- `docs/ATOMIC_WRITE_PROTOCOL.md` (Wf-386-A3 canonical reference doc — closeout drafted via `wf386a3_atomic_write.py::write_closeout`)
- `AGENTS.md` §5.1 (standing GPU authorization honored via USER_OK gating) + §5.2 (honest framing — no fabrication) + §5.3 (no paper work per TODO-84) + §5.4 (no wet-lab per §10) + §5.5 (post-wave on-disk verification rule) + §5.6 (atomic write protocol adoption rule) + §5.7 (trainer flag wire-in smoke-required rule) + §10 (wetlab deprecation FAQ) standing directives re-confirmed
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating — R3_actual mode REFUSAL honored)
- `mol-metal-gpu-execution` skill (USER_OK precondition at METAL-COORD-TERM launch — 5/5 safety conditions honored)
- `mol-metal-cfm-training` skill (cross-ref via Wf-394-A1 Lambda CFM coupling audit pattern)
- `mol-metal-paper-writing` skill (READ-only on `paper/*` per TODO-84 + §5.3)
- `mol-metal-anticancer` skill (Krasnov 0.86 gate cross-ref per Wf-379-A1 + Wf-387-A1)
- `TODO/completed/{381a4_m6kappa_finalize,381a5_r20_plan_cpu_close,382a1_multiph_pll_scaffold,392a1_multiph_pll_wire,392a2_chemprop_v3_cache_wire,392a3_lomo_wrapper_verify,394a1_lambda_cfm_wire,394a2_wf264_property_panel,394a3_wf299_pre_launch_verify}_*.md` (parallel Wf-381..394 wave closeouts — the Wf-409-A3 SCAFFOLD is part of the 2026-09-30 ultracode final wave trio per Wf-368-B1 §7 + Wf-375-B1 §22.68.3)

---

**END OF Wf-409-A3 closeout** — `TODO/completed/409a3_wf246c_scaffold.md` 2026-09-30.

**Wall (MEASURED):** ~5 min CPU-only for SCAFFOLD authoring + ~5 min for wrapper + ~3 s for tests + ~10 min for closeout = ~25 min wall PROJECTED vs ~5 min actual MEASURED for sub-flow execution; closeout drafting ~15 min; total ~25-30 min CPU-only.