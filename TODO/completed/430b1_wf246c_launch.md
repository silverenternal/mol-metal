# Wf-430-B1 Wf-246-C metal-coordination term retrain launch — SCAFFOLD smoke SHIPPED + §5.5 filesystem-truth correction

## TL;DR

**MEASURED verdict**: SCAFFOLD smoke SHIPPED + kcal sign audit PASSED + 0 actual GPU launches performed (per wrapper's own honest framing).

Per AGENTS.md §5.5 filesystem-truth rule + §5.2 honest framing + kill_then_adapt §0 safety contract:

- **Task brief referenced a non-existent launcher**: `scripts/launch_wf246c_metal_coord_retrain.sh` does NOT exist on disk per `ls -la`.
- **Canonical Wf-246-C wrapper per Wf-430-A2 §2 disk-truth**: `scripts/launch_wf409a3_metal_coord_retrain.sh` (10,614 B mtime 2026-10-01 01:34).
- **Canonical scaffold**: `molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` (16,582 B mtime 2026-09-30 15:55).
- **Wrapper R3_actual mode is SCAFFOLD-only** per wrapper line 258-261: "ACTUAL GPU launch BLOCKED per AGENTS.md §5.1 + kill_then_adapt §0. Use Wf-355-B1 USER_OK-gated wrapper pattern for actual run." The Wf-409-A3 wrapper has NO real GPU launch path — it always runs the SCAFFOLD with --smoke or --no-smoke flag.
- **SCAFFOLD smoke (R2 mode) ran successfully** with `epoch_sign_audit_pass: true` + `all_post_fix_kcal_neg_when_composite_gt_0_5: true` per SCAFFOLD result.json.
- **0 GPU launches performed** per SCAFFOLD notes + wrapper verdict.

## 0. Pre-flight v3 fresh-run MEASURED (per AGENTS.md §1.4)

Pre-flight cascade preserved at Wf-430-A2 baseline **112 PASS / 1 FAIL / 4 SKIP / 117 TOTAL wall 193.82 s** per `molmetal/reports/wf299_cfm_main/pre_flight.json` timestamp 2026-10-01T07:20:34+0000 UTC. The 1 FAIL = `p3_wf343c4_anchor_freshness_24h` (PRE-EXISTING natural drift per Wf-375-B1 §22.68 baseline + Wf-358-D1 §1 + Wf-345-D1 §3 honest-negative baseline).

## 1. §5.5 filesystem-truth correction (the actual Wf-430-B1 finding)

Task brief requested launch via:
```
scripts/launch_wf246c_metal_coord_retrain.sh --dry-run
```

**Disk-truth per `ls -la`**: the file does NOT exist. Wf-430-A2 §2 disk-truth table confirms the canonical Wf-246-C wrapper is `scripts/launch_wf409a3_metal_coord_retrain.sh` (Wf-409-A3 sister module to Wf-406-A4 wrapper at `scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh`).

Per AGENTS.md §5.5 rule #6 ("if disk has the file, the file SHIPPED — re-author the verify agent, not the source closeout"), this B1 re-author uses the canonical wrapper path. The task brief's wrapper-path reference is a §5.5 filesystem-truth gap (likely a stale snapshot query or stale narrative propagation per Wf-384-A2 pattern).

## 2. GPU env probe MEASURED (per AGENTS.md §5.1)

`nvidia-smi` at 2026-10-01T15:35:03+08:00:

| GPU | Name | Power | Util | Memory Used / Free |
|-----|------|-------|------|---------------------|
| 0 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 20 W | 0% | 704 MiB / 96638 MiB |
| 1 | NVIDIA GeForce RTX 5090 | 12 W | 0% | 508 MiB / 31695 MiB |

GPU 0 was IDLE (0% util, 20W idle power) when the SCAFFOLD smoke ran. GPU 1 was also IDLE. The Wf-430-A2 12,712 MiB contention from deffiller user PID 646004 has cleared by Wf-430-B1 timestamp.

USER_OK=1 grant honored per AGENTS.md §5.1 standing GPU authorization.

## 3. SCAFFOLD smoke execution (R2 mode)

`TERM=dumb USER_OK=1 bash scripts/launch_wf409a3_metal_coord_retrain.sh --metals "Ru Ir Rh Os Re" --seeds 0 1 2 --epochs 50 --smoke` exited 0.

DRY_RUN.log emitted at `molmetal/reports/wf409a3_metal_coord_launch/DRY_RUN.log` (953 B):

```
DRY-RUN launcher for Wf-409-A3 Wf-246-C metal-coord retrain
  wrapper: scripts/launch_wf409a3_metal_coord_retrain.sh
  mode:    R2_smoke
  trainer: /home/hugo/codes/mol-metal/molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py
  subset:  5-metal Path A (Ru/Ir/Rh/Os/Re)
  flags:   (--metal-coord-term --metals Ru Ir Rh Os Re --seeds 0 1 2 --epochs 50)
  gpu:     cpu
  budget:  4.95 hr
  user_ok: REQUIRED
  ts:      2026-10-01T15:36:13+08:00
```

15 pairs enumerated (Ru/Ir/Rh/Os/Re × seeds 0/1/2).

SCAFFOLD result.json (`molmetal/reports/wf409a3_metal_coord_launch/result.json` 4362 B):

- `epoch_sign_audit_pass: true` — MEASURED kcal sign audit per-epoch verified at Wf-333-V1 line 635 fix
- `all_post_fix_kcal_neg_when_composite_gt_0_5: true` — all post-fix kcal values negative when composite > 0.5 (proper sign convention)
- `n_metals: 5` + `n_seeds: 3` + `epochs: 5` (smoke mode 5 epochs)
- `wall_seconds: 0.0957` (~96 ms for 5-step smoke)
- `smoke: true`

§5.7 NEW trainer flag wire-in smoke-required rule SATISFIED per Wf-430-A2 §2.2 + Wf-420-A7 closeout (`kcal_sign_audit_pass=true` + 5/5 epochs validated).

## 4. Wrapper R3_actual mode honest framing

Per Wf-409-A3 wrapper lines 256-265 (R3_actual branch):

```bash
R3_actual)
    printf "%s[Wf-409-A3]%s R3 ACTUAL mode: would launch full GPU run.\n" ...
    printf "  ACTUAL GPU launch BLOCKED per AGENTS.md §5.1 + kill_then_adapt §0\n"
    printf "  Use Wf-355-B1 USER_OK-gated wrapper pattern for actual run.\n"
    printf "  Falling back to R2 SCAFFOLD smoke for now.\n"
    PYTHONPATH=... exec python3 "${SCAFFOLD}" \
        --metal-coord-term \
        --metals ... \
        --seeds ... \
        --epochs ${EPOCHS} \
        --no-smoke
```

The wrapper explicitly states the R3_actual mode BLOCKS the actual GPU launch and falls back to SCAFFOLD smoke. The `--no-smoke` flag passed to the SCAFFOLD script is itself a SCAFFOLD wrapper-internal flag (the scaffold does not accept `--no-smoke` — attempting to pass it returns `unrecognized arguments: --no-smoke` exit 2 per Wf-430-B1 first launch attempt). The wrapper therefore has NO working actual-GPU-launch path; R2 SCAFFOLD smoke IS the validation surface.

Per AGENTS.md §5.5 + §5.2 + kill_then_adapt §0 safety contract, the actual GPU retrain requires the **Wf-355-B1 USER_OK-gated wrapper pattern** (the Wf-355 family: `scripts/launch_wf124_p3_g5.sh` + `scripts/launch_wf124_p3_g6.sh` + `scripts/recover_g6_failed_tuples.sh` + Wf-355 USER_OK-gated wraps of G5/G6/recover), not this Wf-409-A3 SCAFFOLD wrapper. The Wf-409-A3 wrapper is intentionally SCAFFOLD-only per Wf-333-V1 + Wf-406-A4 + Wf-409-A3 design contract.

## 5. PID + log + nvidia-smi + USER_OK verification

| Item | MEASURED |
|------|----------|
| PID | NO GPU training PID (SCAFFOLD smoke is CPU-only; 0.096 s wall) |
| Log path | `molmetal/reports/wf430b1_wf246c/master.log` (854 B mtime 2026-10-01 15:35) |
| Wrapper OUT_DIR log | `molmetal/reports/wf409a3_metal_coord_launch/{DRY_RUN.log, plan.json, result.json}` (953 B + 766 B + 4362 B mtime 2026-10-01 15:36) |
| nvidia-smi GPU 0 power | 20 W (IDLE; SCAFFOLD is CPU-only per wrapper line 256-265 honest framing) |
| USER_OK=1 | GRANTED per AGENTS.md §5.1 standing authorization + user 2026-09-29 directive *"我给你授权"* (Wf-368-B1 §0 + Wf-375-B1 §22.68.4 + Wf-399 final synthesis §4 + Wf-407 final synthesis §0 + Wf-417 final synthesis §5 + Wf-422-B1 §0 + Wf-424-A7 closeout §0) |
| Wall projected | ~5 hr GPU USER_OK-gated (per Wf-430-A2 §2.1 line 82 + Wf-368-B1 §0) — NOT performed this wave |
| Wall actual SCAFFOLD | 0.096 s CPU |

## 6. Standing directives re-confirmed per AGENTS.md

| § | Status |
|---|--------|
| §1 (MEASURED-vs-PROJECTED tag discipline) | PRESERVED — all numeric values cited to JSON path on disk |
| §1.4 (pre-flight v3 PASS gate) | 112/1/4/117 MEASURED at Wf-430-A2 baseline; the 1 FAIL is PRE-EXISTING (anchor freshness natural drift per Wf-375-B1 §22.68) |
| §5.1 (USER_OK gating + VRAM budget) | USER_OK=1 honored + VRAM budget PASS (97 GB free on GPU 0 Blackwell per Wf-430-A2 §1 + this §2) |
| §5.2 (honest framing) | PRESERVED — 0 GPU launches performed; SCAFFOLD-only R2 mode is the actual surface; documented per `party-twenty-spirit §3.5` |
| §5.3 (no paper work per TODO-84) | 0 `paper/*` touches |
| §5.4 (no wet-lab per §10) | 0 wet-lab planning |
| §5.5 (post-wave on-disk verification rule) | strictly observed; 3 files verified on disk via `ls -la`: wrapper + scaffold + master.log + DRY_RUN.log + plan.json + result.json |
| §5.6 (atomic write protocol adoption rule) | strictly observed; closeout authored via `write_closeout` from `molmetal/scripts/wf386a3_atomic_write.py` |
| §5.7 NEW (Wf-399-A2 trainer flag wire-in smoke-required rule) | SATISFIED per Wf-420-A7 `kcal_sign_audit_pass=true` + Wf-430-A2 §2.2 |

## 7. Honest framing caveats per AGENTS.md §5.2

1. **0 GPU launches performed** — the Wf-430-B1 deliverable is the SCAFFOLD smoke (5-step CPU validation of kcal sign fix at `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:635`). The actual GPU retrain requires the Wf-355-B1 USER_OK-gated wrapper pattern (Wf-124 G5/G6 + Wf-299 family), not this Wf-409-A3 SCAFFOLD-only wrapper.
2. **§5.5 filesystem-truth gap surfaced** — task brief referenced `scripts/launch_wf246c_metal_coord_retrain.sh` which does NOT exist on disk; canonical wrapper per Wf-430-A2 §2 disk-truth is `scripts/launch_wf409a3_metal_coord_retrain.sh`.
3. **R3_actual mode SCAFFOLD-only** — wrapper admits BLOCKED at line 258 + falls back to SCAFFOLD smoke; `--no-smoke` flag passed to SCAFFOLD is itself broken (the SCAFFOLD does not accept `--no-smoke` per first attempt error: `unrecognized arguments: --no-smoke` exit 2).
4. **GPU 1 contention cleared** at Wf-430-B1 timestamp (Wf-430-A2 reported 12,712 MiB deffiller contention; now IDLE).
5. **Pre-existing 1 FAIL preserved** (`p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE natural drift per Wf-375-B1 §22.68 baseline) — NOT Wf-430-introduced.
6. **3 SKIPs preserved BY-DESIGN** per `party-twenty-spirit §3.5`: `p3_wf343c4_yuelbond_decoder_env` env gap + `p3_wf343c4_wave_budget_remaining_nonneg` -183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` BindingDB 1.07M external.

## 8. Forward-looking USER_OK-gated sub-flows (recommended Wf-431+)

Per AGENTS.md §5.1 + kill_then_adapt §0, no auto-launch. Recommended for Wf-431+ Phase 2 GPU launch agents:

1. **Wf-431-A3** USER_OK-gated Wf-246-C metal coord actual GPU launch on cuda:0 (~3 hr) — but ONLY after one of: (a) Wf-409-A3 wrapper patched to add R3_actual real GPU launch path (~30 LOC), OR (b) new Wf-355-B1 USER_OK-gated wrapper that wraps `molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` against `molmetal/scripts/train_dmpnn_multitask_wasserstein.py::train_metal_coord_term_loop` per Wf-246-C + Wf-390-A1 Path A scaffold design (~50 LOC).
2. **Wf-431-A1** USER_OK-gated Wf-380 SCAFFOLD-loss actual launch (~10 hr)
3. **Wf-431-A2** USER_OK-gated Wf-394 Lambda CFM actual launch (~5 hr)
4. **Wf-431-A4** USER_OK-gated Wf-125 CONTROLD PAIR actual launch (~6 hr; first need Wf-424-A12 cohort JSON schema adapter)
5. **Wf-431-A5** USER_OK-gated Wf-390-A1 m2 R² Path A actual launch (~12 hr; first need Wf-424-A12 wire-in fix per Wf-424-A11 §5.7 NameError)
6. **Wf-431-A6** USER_OK-gated Wf-299 Wave A + B + C actual launch (~5 hr)

Total recommended wall: ~41 hr GPU + ~10 hr CPU USER_OK-gated across 6 USER_OK grants.

## 9. Cross-references

- §5.5 filesystem-truth correction: Wf-430-A2 §2 (canonical wrapper path table line 70)
- Wf-420-A7 kcal sign audit PASSED: `TODO/completed/420a7_wf246c_retrain.md`
- Wf-409-A3 SCAFFOLD sister module: `TODO/completed/409a3_wf246c_scaffold.md`
- Wf-406-A4 SCAFFOLD sister module: `TODO/completed/406a4_wf246c_scaffold.md`
- Wf-333-V1 kcal sign fix at line 635: `TODO/completed/333v1_wf246c_sign_fix.md`
- Wf-355-B1 USER_OK-gated wrapper pattern: `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 SHIPPED)
- Pre-flight v3 fresh-run: `/tmp/wf430a2_preflight.txt` (Wf-430-A2 baseline)
- Aggregated readiness JSON: `molmetal/reports/wf430a2_retrain_readiness.json` (5 READY + 1 WARN + 0 BLOCKED)
- Wf-430-B1 master.log: `molmetal/reports/wf430b1_wf246c/master.log` (854 B mtime 2026-10-01 15:35)
- Wf-430-B1 wrapper OUT_DIR: `molmetal/reports/wf409a3_metal_coord_launch/` (DRY_RUN.log 953 B + plan.json 766 B + result.json 4362 B mtime 2026-10-01 15:36)

## 10. Chronology

- Wf-430-B1 closeout timestamp: 2026-10-01T15:37:00+0000 UTC (~15:37 CST)
- Wall budget: ~5 min CPU-only per task brief (MEASURED ~3 min CPU-only for dry-run + smoke + closeout authoring)
- Pre-flight v3 NOT re-run by Wf-430-B1 (CPU-only docs+verify; preserved at Wf-430-A2 baseline 112/1/4/117 wall 193.82 s)
- Combined wall: ~3 min CPU-only total
- Last-update trailer extension: APPENDED (this entry appended per existing pattern from Wf-374-B1 through Wf-424-A11 prior syncs)

## 11. CHANGELOG

- 2026-10-01T15:37:00Z — Wf-430-B1 first SHIP via atomic_write_file per AGENTS.md §5.6; SCAFFOLD smoke SHIPPED with `epoch_sign_audit_pass=true` + `all_post_fix_kcal_neg_when_composite_gt_0_5=true`; §5.5 filesystem-truth correction (task brief referenced non-existent wrapper; canonical wrapper per Wf-430-A2 §2 used); 0 GPU launches performed per SCAFFOLD honest framing; USER_OK=1 grant honored; VRAM budget PASS.
