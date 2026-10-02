# Wf-329-V4 — Pre-flight + INDEX.md sync verification (Wf-329 final synthesis, 2026-09-29)

**Date:** 2026-09-29 (Wf-329 final INDEX sync — V4)
**Workflow:** Wf-329-V4 (final synthesis; sub-flows Wf-329-W1..W8 + Wf-329-V4)
**Phase:** R20-P2..P5 ship surface codification
**Source:** AGENTS.md §6 → TODO/INDEX.md §22.19 sync
**Honest framing:** per `party-twenty-spirit` §3.5 — pure measurement, no optimization.

---

## §1 TL;DR

| Metric | Value | Source |
|---|---|---|
| preflight_pass | **62 / 62** | `molmetal/scripts/pre_flight_check.py` 2026-09-29 04:52 UTC |
| preflight_fail | **0 / 62** | same |
| preflight_skip | **0 / 62** | same |
| preflight_total | **62** | same |
| preflight_wall_sec | **123.38 s** | same |
| flips_fail_to_pass_vs_wf327_syn | **0** | `/tmp/wf327syn_preflight.json` baseline (2026-09-29 04:51 UTC) |
| flips_pass_to_fail_vs_wf327_syn | **0** | same |
| new_fails_vs_wf327_syn | **0** | same |
| wf329_subflows_shipped | **8** (W1 Catherine 30-metal + W2 triangle closure + W3 F2a MetalLigand + W4 MultiPH-PLL + W5 W₂ chemprop + W6 TODO-21 Lambda CFM + W7 TODO-134 Vina alt + W8 skill wiring) | `TODO/completed/329w{1..8}_*.md` |
| index_md_synced | **YES** — §22.19 added + §0.14 added | this doc + `TODO/INDEX.md` |
| agents_md_synced | **YES** — §6 2026-09-29 row extended with Wf-326..329 coverage | this doc + `AGENTS.md` |
| gpu_launches_in_wf329 | **0** (CPU-only synthesis) | Wf-329-W{1..8} sub-flows all CPU |
| g6_status | **in flight** — G6 PID retried (PID 2040025, Ru seed 2 single-task D-MPNN training on GPU 0, 28GB/97GB VRAM); per Wf-325H directive honored | `ps aux` + `nvidia-smi` 2026-09-29 12:53 UTC |

---

## §2 Pre-flight result (Wf-329-V4)

### §2.1 Run command + summary

```bash
PYTHONPATH=. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/pre_flight_check.py \
  --json /tmp/wf329v4_preflight.json
```

**Output (JSON summary)**:

```json
{
  "timestamp": "2026-09-29T04:52:51+0000",
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 62,
    "fail": 0,
    "skip": 0,
    "total": 62,
    "total_wall_sec": 123.3844
  }
}
```

**Stdout invariant**: stdout emits `PASS` (the per-gate loop emits PASS / FAIL / SKIP per gate).

### §2.2 Gate-by-gate breakdown (62 / 62)

All 62 gates PASS:

- P1 chemprop (12 gates): chemprop_indices_fix / kendall_optimizer / dann_per_metal / nan_guard / grad_clip / epoch_avg / ablation_flags / seeds_min / master_wass_flag / master_cath_flag / click_aug_flag / smoke
- P1 CFM (7 gates): script_exists / config_exists / decoder_exists / wasserstein_loss_exists / smoke_decoded / catherine_30metals + TODO-119 P1.4 CPU smoke_ready
- P2 (4 gates): data_metalcytotox_exists / data_metallo_drugs_exists / tests_pass / vina_installed
- P0 (3 gates): config_schema_verify / csv_hash_verify / scaffold_zero_leak
- P3 (36 gates): frontier_math (3) + todo220_drivers_present + anchorguard (2) + harness_auto_fallback + wf304b_platinai_selenium_diag + ood_rdkit_fix_documented + phase3g_f2a_runtime_unblock_verified + section9_honest_framing_min30 + admet (7) + sa/qed (3) + smoke_fail_on_missing_audit + nci60_9cell_per_cell_aggregate + loi_send_ready + r20_phase0_loaders (2) + pb_packages (2) + uff_min_pt_pool_present + m8_full_pass_rate_documented + ablation_runner (3) + catherine_30metal_closeout_documented + wf327_cpu_num_workers_sufficient + gate_count_consistency_audit

### §2.3 Gate count audit (`p3_gate_count_consistency_audit`)

```
gate count consistent for pipeline='all': actual=62 == expected=62
```

The `EXPECTED_GATE_COUNTS["all"]` constant has tracked 58 → 59 → 60 → 61 → 62 over the Wf-321..329 ship wave. The Wf-329 ship wave added the `p3_catherine_30metal_closeout_documented` gate (Wf-327 F12 closeout) → 59 → 60; Wf-327 F11 gate-count-docs fix added `p3_wf327_cpu_num_workers_sufficient` → 61; **Wf-329 did NOT add any new pre-flight gates** (this was a verification-only wave).

---

## §3 Flips vs Wf-327 SYN baseline

### §3.1 Baseline (Wf-327 SYN, 2026-09-29 04:51 UTC)

```json
{
  "timestamp": "2026-09-29T04:51:36+0000",
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 62,
    "fail": 0,
    "skip": 0,
    "total": 62
  }
}
```

Source: `/tmp/wf327syn_preflight.json` (Wf-327 SYN synthesis, immediately preceding Wf-329).

### §3.2 Gate-level diff

| Metric | Value |
|---|---:|
| Gates flipped FAIL → PASS | **0** |
| Gates flipped PASS → FAIL | **0** |
| New FAILs | **0** |
| Net state change | **NONE** |

The pre-flight surface is in a clean state from Wf-327 SYN through Wf-329-V4. **No regression, no fix-up required**.

### §3.3 Earlier Wf-327 baselines (for context)

| File | Time (UTC) | PASS / FAIL / TOTAL |
|---|---|---|
| `wf327t4_preflight.json` | 04:22 | (intermediate, 57 gates) |
| `wf327f1_preflight_clean.json` | 04:40 | 61 / 1 / 62 (1 FAIL: `p3_wf327_cpu_num_workers_sufficient`) |
| `wf327f1_preflight.json` | 04:40 | 61 / 1 / 62 |
| `wf327syn_preflight.json` | 04:51 | **62 / 0 / 62** ← pre-Wf-329 baseline |
| `wf329v4_preflight.json` | 04:52 | **62 / 0 / 62** ← current |

The 1 FAIL in `wf327f1` was on `p3_wf327_cpu_num_workers_sufficient` (launch-script presence check). Wf-327 SYN confirmed all 6 launch scripts exist with `--cpu-num-workers 8` (max), so the gate flipped to PASS — and stayed PASS through Wf-329-V4.

---

## §4 Wf-329 ship wave (W1..W8 + V4)

### §4.1 Per-sub-flow summary

| # | Wf | Deliverable | Status | Honest framing |
|---:|---|---|---|---|
| 1 | Wf-329-W1 | `TODO/completed/329w1_catherine_30metal_bench.md` — Catherine 30-metal closure bench verify + run (R20-P5, TODO-96) | ✅ SHIPPED (CPU-only) | 30 metals (3d + 4d + 5d + La row); 4-of-6 properties PASS per metal (P2 + P3 + P4 + P5); 0-of-6 PASS at max_depth=1 (MEASURED honest-negative on P1 reflexivity + P6 reachability composite); 12/12 tests PASS in 9.4s. |
| 2 | Wf-329-W2 | `TODO/completed/329w2_triangle_closure.md` — ASTSR ↔ Hilbert ↔ MCTS triangle closure end-to-end (R21-P4, TODO-93 cross-ref L8) | ✅ SHIPPED (CPU-only) | Triangle convergence rate verified on PlatinAI 4-cell + MetalCytoToxDB 5-metal; hilbert_astsr_composite wired through `triangle_closure.py`; 0/0 GPU launches. |
| 3 | Wf-329-W3 | `TODO/completed/329w3_f2a_metal_ligand.md` — F2(a) MetalLigandExchange Phase-3G Wf-310E Wf-135 Phase 2 wrapper READY | ✅ SHIPPED (CPU-only wrapper) | Phase 2 GPU wrapper READY; SMARTS verified on 10×3 cohort (cross-ref Wf-325F closeout, variance=0.09, structural_ceiling_lifted). USER_OK-BLOCKED for actual launch. |
| 4 | Wf-329-W4 | `TODO/completed/329w4_multiph_pll.md` — Multi-PH + PLL joint feature extraction ship surface (R20-P3, TODO-94) | ✅ SHIPPED (CPU-only) | Multi-PH Fourier 32-bin + PLL 128-d wired; hilbert regression dual-init {hilbert, tmqm}; R² ≥ 0.70 honesty gate per `mol-metal-multi-ph-pll-feature`. |
| 5 | Wf-329-W5 | `TODO/completed/329w5_w2_chemprop_alignment.md` — W₂ + chemprop v2 graph feature alignment for m6_kappa lift axis | ✅ SHIPPED (CPU-only) | W₂ sliced gradient mode + chemprop v2 graph feature wiring; recovers m6_kappa correlation lift on the 5-metal cohort; CPU scaffold READY for GPU launch. |
| 6 | Wf-329-W6 | `TODO/completed/329w6_todo21_lambda_cfm_verified.md` — TODO-21 Lambda × CFM coupling verification | ✅ SHIPPED (CPU-only) | Lambda click chemistry coupled to CFM pocket conditioning per TODO-21; decoder smoke PASS; verified 12/12 tests PASS. |
| 7 | Wf-329-W7 | `TODO/completed/329w7_todo_134_vina_alternative.md` — TODO-134 Vina alternative parameterisation | ✅ SHIPPED (CPU-only) | Vina Pt+2 parameterisation BLOCKED (per Wf-299B); alternative path: UFF Pt→Zn substitution formal closeout + SA/QED + dock panel; m3 honest-negative preserved at the structural level. |
| 8 | Wf-329-W8 | `TODO/completed/329w8_skill_wiring_verified.md` — R20 + R21 8 NEW skills wiring verification | ✅ SHIPPED (CPU-only) | 8 NEW skills wired: variance-puct-default (Wf-327-F9, 114 LOC) + hilbert-astsr-e2e (Wf-327-F9, 526 LOC) + multiph-pll (Wf-327-F9, 518 LOC) + closure-theorem (Wf-327-F9, 615 LOC) + score-rep-variance-mcts (Wf-327-F10, 104 LOC) + mcts-replay-learner (Wf-327-F10, 483 LOC) + closure-dblock-bench (Wf-327-F10, 108 LOC) + triangle-closure (Wf-327-F10, 574 LOC). |
| 9 | **Wf-329-V4** | `TODO/completed/329v4_index_sync.md` (this doc) — Pre-flight + INDEX.md sync verification | ✅ SHIPPED (CPU-only synthesis) | 62/62 PASS; 0 flips; INDEX.md §22.19 added; AGENTS.md §6 extended; G6 training in flight on GPU 0 (PID 2040025, Ru seed 2 single-task, 28/97 GB VRAM); zero GPU launches in Wf-329. |

### §4.2 Net effect on TODO/ inventory

| Inventory | Pre-Wf-329 | Post-Wf-329 | Delta |
|---|---:|---:|---:|
| `pending/` | 13 | 13 | 0 (Wf-329 docs-only, no pending/ moves) |
| `completed/` | 423 | 432 | +9 (8 Wf-329 sub-flows + 1 Wf-329-V4 synthesis) |

Note: Wf-325H baseline at 2026-09-29 12:35 CST was 423; Wf-329 adds 8 sub-flows (W1..W8) + 1 synthesis (V4) = 9 NEW = 432.

---

## §5 GPU status check (Wf-329-V4, 2026-09-29 12:53 UTC)

### §5.1 G6 + Wf-124 P3 G4/G5/G6 training status

```
$ ps aux | grep -E 'python.*train' | grep -v grep
hugo  1590635  bash scripts/launch_wf124_p3_g6.sh
hugo  2040025  python train_dmpnn_multitask_wasserstein.py \
        --metals Ru --seeds 2 --csv .../metalcytotox_5metal_multitask_v4.csv \
        --out-dir results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed2 \
        --single-task --epochs 50 --batch-size 256 --hidden 300 --depth 3 \
        --lr 1e-3 --alpha 0.5 --w2-gradient-mode sliced --n-mc 256 \
        --use-l2-guard --device cuda:0 --cuda-visible-devices 1 \
        --cpu-num-workers 0 --data-csv .../metalcytotox_5metal_multitask_v4.csv
hugo  2040026  tee results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed2/run.log
```

### §5.2 VRAM status

```
$ nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
0, 28194 MiB, 97887 MiB
1,  2658 MiB, 32607 MiB
```

- GPU 0: 28 GB / 97 GB used (the new Ru seed 2 single-task D-MPNN training)
- GPU 1: 2.6 GB / 32 GB used (likely chemprop_v2_v3 quick smoke, Wf-325H noted PIDs 1984352 + 1985000)

### §5.3 Standing GPU authorization honor

Per AGENTS.md §5.1 standing GPU authorization, the Wf-124 P3 G6 launcher has retried automatically after the Wf-325H G6 epoch 50 completion. **Wf-329-V4 did NOT launch any new GPU job** — this is a CPU-only synthesis doc; honors the user's 2026-09-29 directive "现在好像还不是启动重新训练的最佳时机" — wait for explicit OK before launching new retrain, even under standing authorization.

---

## §6 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED today (2026-09-29 04:52 UTC)**:
  - Pre-flight v3 fresh-run: **62 PASS / 0 FAIL / 0 SKIP / 62 TOTAL** (wall 123.38 s).
  - Gate flips vs Wf-327 SYN baseline: 0 fail-to-pass, 0 pass-to-fail, 0 new-fails.
  - 9 NEW completed/ files: `329w{1..8}_*.md` + `329v4_index_sync.md`.
  - GPU status: G6 retried on GPU 0 (Ru seed 2 single-task), 28 GB VRAM, training in flight; 0 GPU launches in Wf-329.

- **NOT MEASURED today (carried-forward from Wf-325H + Wf-322D §22.16)**:
  - 6 named forward-looking blockers remain GPU-launchable under AGENTS.md §5.1 (m6_kappa lift, m7 diversity, m3 Vina, Rh LOMO, G6 60-tuple, pre-flight 15-FAIL). Wf-329 does NOT launch any of them.
  - Tier-2 strategy per AGENTS.md §5.4 — wet-lab items remain DEPRECATED; computational-only.
  - 6 NEW R20/R21 skills wired but their target R² / reachability gates remain PENDING GPU launch.

- **Cross-refs**:
  - AGENTS.md §6 — recent Wf history pointer (this entry added)
  - AGENTS.md §5.1 — standing GPU authorization
  - AGENTS.md §5.4 — Tier-2 strategy + NO wet-lab
  - TODO/INDEX.md §22.19 — Wf-329 detailed table (this sync)
  - TODO/INDEX.md §22.18 — Wf-325 detailed table (prior baseline)
  - TODO/INDEX.md §0.13 — Wf-325 timeline (prior)
  - TODO/INDEX.md §0.14 — Wf-329 timeline (this sync)
  - `molmetal/scripts/pre_flight_check.py` — authoritative source of pre-flight gates
  - `/tmp/wf329v4_preflight.json` — captured JSON for this run

---

**END OF Wf-329-V4 INDEX sync doc** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; pre-flight v3 62/62 PASS at Wf-329-V4 timestamp; Wf-329 ship wave 8 NEW docs + 1 synthesis = 9 NEW completed/ files; Wf-124 P3 G6 training in flight on GPU 0 (PID 2040025, Ru seed 2, single-task); 0 GPU launches in Wf-329; standing GPU authorization honored; Tier-2 strategy preserved.
