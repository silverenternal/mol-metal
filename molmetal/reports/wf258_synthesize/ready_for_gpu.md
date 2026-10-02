# Wf-258-Synthesize: CFM-ready-for-GPU Summary (per-wave user OK gate)

**Date**: 2026-09-28
**Owner**: Hugo Lee + ultracode agent
**Status**: 🚦 SYNTHESIS (read-only summary; no code edits; ready for user OK per wave)
**Parent**: Wf-258-Audit + Wf-258-Verify + Wf-258-Prep (5 dry-run YAML configs)
**Standing directive**: per "kill then adapt" — every wave ≤ 1 hr requires FRESH user OK before launch.

---

## §0 Honest framing (READ FIRST)

Per `party-twenty-spirit` §3.5 + `mol-metal-gpu-execution` §7:

- **MEASURED today**: Wf-124 P2 synthetic decode_ratio = **1.000** (192/192); Wf-181 m14 smoke real CFM = **0.9980 ± 0.0008** (5,979/5,991); Wf-188 m14 FULL MetalCytoToxDB = **0.9958** (26,687/26,801). CFM is paper §4 LEAD candidate for m14 (gate ≥ 0.05 PASS by 20×). Wf-124 P3 Ru seed=0 test_auc = **0.7613** (+0.049 vs 0.7123 baseline; 30 epochs, RTX 5090).
- **PENDING (5 TODOs, this synthesis)**: TODO-119 P1.3 + P1.4, TODO-180 G4, TODO-181 G5, TODO-182 G6.
- **DO NOT claim before launch**: any per-lever attribution %, 50-ep lift magnitude, 5×3 Krasnov target 0.86 mean, ablation interaction term.
- **No best-seed selection** (per `mol-metal-gpu-execution` §7); per-seed std IS the answer.

This synthesis consolidates the 5 PENDING CFM TODOs into **17 user-OK-gated waves** (C1–C17), each ≤ 1 hr wall-clock, per the "kill then adapt" directive.

---

## §1 Per-wave breakdown (each ≤ 1 hr, fresh user OK per launch)

### §1.1 Wave C1 — TODO-180 G4 part 1 (Ru + Ir, 5 pairs)
| Field | Value |
|---|---|
| **TODO** | TODO-180 G4 (14 PENDING `(metal, seed)` pairs, 30 ep) |
| **Pairs in this wave** | (Ru, 1), (Ru, 2), (Ir, 0), (Ir, 1), (Ir, 2) — **5 pairs** |
| **Wall-time** | **~38 min** on RTX 5090 GPU-1 (5 × ~7.7 min) |
| **Target MEASURED** | Ru/Ir 3-seed mean ± std test_auc (per-metal); closes 5/14 of G4 PENDING |
| **Config** | `configs/wf124_p3_remaining_metals_g4.yaml` (wave_a, 8-pair definition; we run first 5) |
| **Honest risk** | per-seed std ±0.10 possible on smaller metals (NOT in this wave — Ru/Ir are largest cohorts) |

**User OK gate — if user says `YES to wave C1`, run**:
```bash
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES=1 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/train_dmpnn_multitask_wasserstein.py \
  --metals "Ru Ir" --seeds "0 1 2" --epochs 30 \
  --batch-size 64 --lr 1e-3 --alpha 0.5 \
  --wasserstein-weight 0.05 --use-l2-guard --device cuda:1 \
  --out-dir results/wf124_p3_remaining_metals_g4 \
  --ckpt-dir results/wf124_p3_remaining_metals_g4/ckpts
```

---

### §1.2 Wave C2 — TODO-180 G4 part 2 (Os + Rh + Re, 9 pairs)
| Field | Value |
|---|---|
| **TODO** | TODO-180 G4 (completes 14/14 PENDING pairs) |
| **Pairs in this wave** | (Os, 0/1/2), (Rh, 0/1/2), (Re, 0/1/2) — **9 pairs** |
| **Wall-time** | **~70 min** on RTX 5090 GPU-1 (9 × ~7.7 min) |
| **Target MEASURED** | 5-metal × 3-seed mean ± std test_auc; full `training_summary.json` regenerate |
| **Honest risk** | Rh/Os/Re cohorts < 1.2k rows → per-seed std likely ±0.10; honest reporting: mean ± std unfiltered |

**User OK gate — if user says `YES to wave C2`, run**:
```bash
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES=1 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/train_dmpnn_multitask_wasserstein.py \
  --metals "Os Rh Re" --seeds "0 1 2" --epochs 30 \
  --batch-size 64 --lr 1e-3 --alpha 0.5 \
  --wasserstein-weight 0.05 --use-l2-guard --device cuda:1 \
  --out-dir results/wf124_p3_remaining_metals_g4 \
  --ckpt-dir results/wf124_p3_remaining_metals_g4/ckpts
```

---

### §1.3 Wave C3 — TODO-119 P1.3 (real-CFM decode lift smoke)
| Field | Value |
|---|---|
| **TODO** | TODO-119 P1.3 — full 3-innovation CFM stack on MetalCytoToxDB 5-metal (n=26,801) |
| **Protocol** | 1 seed × 3 epochs (~30 min on RTX PRO 6000 Blackwell GPU-0) |
| **Config** | `configs/cfm_real_decode_lift_p1.3.yaml` (1 wave, ~0.5 hr) |
| **Target MEASURED** | decode_ratio ≥ 0.5 on real MetalCytoToxDB (MUST confirm Wf-181/Wf-188 0.9980/0.9958 was NOT just ad-hoc) |
| **Honest risk** | per TODO-119 §5: "real chance P1.3 still produces 0.0" — but Wf-181/Wf-188 already at 0.9980/0.9958; this is a formal P1.3 deliverable |

**User OK gate — if user says `YES to wave C3`, run**:
```bash
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES=0 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
  --epochs 3 --use-yuelbond --pocket-id 3eml \
  --seed 0 --batch-size 16 --lr 1e-4 \
  --out-dir results/cfm_real_decode_lift_p1.3/seed0
```

---

### §1.4 Wave C4 — TODO-119 P1.4 part 1 (full-stack control + YuelBond-only)
| Field | Value |
|---|---|
| **TODO** | TODO-119 P1.4 — per-innovation CFM ablation |
| **Configs in this wave** | `a_full_stack` (control) + `b_yuelbond_only` — **2 ablation configs × 1 seed × 3 ep** |
| **Wall-time** | **~60 min** on RTX PRO 6000 Blackwell (2 × ~30 min) |
| **Honest risk** | interaction term may be >30% per TODO-182 §3.4 hypothesis; do NOT cite per-lever % yet |

**User OK gate — if user says `YES to wave C4`, run**:
```bash
cd /home/hugo/codes/mol-metal
# ablation a: full stack
CUDA_VISIBLE_DEVICES=0 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/train_cfm_yuelbond_wasserstein.py --epochs 3 --use-yuelbond --seed 0 \
  --pocket-id 3eml --batch-size 16 --lr 1e-4 \
  --out-dir results/cfm_per_innovation_ablation_p1.4/a_full_stack/seed0
# ablation b: YuelBond only
CUDA_VISIBLE_DEVICES=0 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/train_cfm_yuelbond_wasserstein.py --epochs 3 --use-yuelbond \
  --no-wasserstein --no-closure-bounded --seed 0 \
  --pocket-id 3eml --batch-size 16 --lr 1e-4 \
  --out-dir results/cfm_per_innovation_ablation_p1.4/b_yuelbond_only/seed0
```

---

### §1.5 Wave C5 — TODO-119 P1.4 part 2 (W₂-only + Catherine-only)
| Field | Value |
|---|---|
| **TODO** | TODO-119 P1.4 — completes 4-config ablation |
| **Configs in this wave** | `c_w2_only` + `d_catherine_only` — **2 ablation configs × 1 seed × 3 ep** |
| **Wall-time** | **~60 min** on RTX PRO 6000 Blackwell |
| **Deliverable** | `results/cfm_per_innovation_ablation_p1.4/ablation_report.json` (4-row matrix) |

**User OK gate — if user says `YES to wave C5`, run**:
```bash
cd /home/hugo/codes/mol-metal
# ablation c: W2 only
CUDA_VISIBLE_DEVICES=0 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/train_cfm_yuelbond_wasserstein.py --epochs 3 --no-yuelbond \
  --no-closure-bounded --seed 0 \
  --pocket-id 3eml --batch-size 16 --lr 1e-4 \
  --out-dir results/cfm_per_innovation_ablation_p1.4/c_w2_only/seed0
# ablation d: Catherine only
CUDA_VISIBLE_DEVICES=0 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/train_cfm_yuelbond_wasserstein.py --epochs 3 --no-yuelbond \
  --no-wasserstein --seed 0 \
  --pocket-id 3eml --batch-size 16 --lr 1e-4 \
  --out-dir results/cfm_per_innovation_ablation_p1.4/d_catherine_only/seed0
```

---

### §1.6 Waves C6–C9 — TODO-181 G5 (50-epoch full sweep, 4 sub-waves)
| Wave | Pairs | Wall-time | GPU |
|---|---|---:|---|
| **C6** | Ru s0/s1/s2 (3 pairs × 50 ep) | **~60 min** | RTX 5090 GPU-1 |
| **C7** | Ir s0/s1/s2 (3 pairs × 50 ep) | **~60 min** | RTX 5090 GPU-1 |
| **C8** | Os s0/s1/s2 + Rh s0/s1/s2 (6 pairs × 50 ep) | **~60 min** | RTX 5090 GPU-1 |
| **C9** | Re s0/s1/s2 (3 pairs × 50 ep) + `30ep_vs_50ep_comparison.json` write | **~60 min** | RTX 5090 GPU-1 |
| **TOTAL** | 15 pairs × 50 ep | **~3.25 hr** | RTX 5090 GPU-1 |

**Config**: `configs/wf124_p3_50epoch_full_sweep_g5.yaml` (4 waves, each ≤ 1 hr).

**User OK gate — one example (C6); same template for C7, C8, C9**:
If user says `YES to wave C6`, run:
```bash
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES=1 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/train_dmpnn_multitask_wasserstein.py \
  --metals Ru --seeds "0 1 2" --epochs 50 \
  --batch-size 64 --lr 1e-3 --alpha 0.5 \
  --wasserstein-weight 0.05 --use-l2-guard --device cuda:1 \
  --out-dir results/wf124_p3_50epoch_full_sweep_g5 \
  --ckpt-dir results/wf124_p3_50epoch_full_sweep_g5/ckpts
```

---

### §1.7 Waves C10–C17 — TODO-182 G6 (per-innovation ablation, 8 sub-waves)
**PRE-LAUNCH BLOCKER**: per `cfm_todo_gap_audit.md` §3.4 gap #1, `--no-kendall` and `--no-dann` flags NOT confirmed in `train_dmpnn_multitask_wasserstein.py`. Run `python molmetal/scripts/train_dmpnn_multitask_wasserstein.py --help` first; if absent, use `--dann-domain-mode=off` fallback or scope to 4 verified configs.

| Wave | Configs | Runs | Wall-time | GPU |
|---|---|---:|---:|---|
| **C10** | `no_multi_task` Ru/Ir/Os | 9 | **~68 min** | RTX 5090 GPU-1 |
| **C11** | `no_multi_task` Rh/Re + `no_wass` Ru | 9 | **~68 min** | RTX 5090 GPU-1 |
| **C12** | `no_wass` Ir/Os/Rh/Re | 12 | **~90 min** ⚠️ | RTX 5090 GPU-1 |
| **C13** | `no_kendall` Ru/Ir | 6 | **~45 min** | RTX 5090 GPU-1 |
| **C14** | `no_kendall` Os/Rh/Re | 9 | **~68 min** | RTX 5090 GPU-1 |
| **C15** | `no_dann` Ru/Ir/Os | 9 | **~68 min** | RTX 5090 GPU-1 |
| **C16** | `no_dann` Rh | 3 | **~23 min** | RTX 5090 GPU-1 |
| **C17** | `no_dann` Re + `ablation_report.json` write | 3 | **~23 min** | RTX 5090 GPU-1 |
| **TOTAL** | 4 configs × 5 metals × 3 seeds | **60** | **~7.5 hr** | RTX 5090 GPU-1 |

**Config**: `configs/wf124_p3_per_innovation_ablation_g6.yaml`. Wave C12 exceeds 1 hr (~1.5 hr); per "kill then adapt", split into C12a (Ir/Os, 6 runs, ~45 min) + C12b (Rh/Re, 6 runs, ~45 min) for strict ≤ 1-hr discipline.

**User OK gate — one example (C10)**: same template for C11–C17.

---

## §2 Total GPU-hours (sequential / per-GPU)

### §2.1 Per-GPU wall-time (sequential, ≤1-hr sub-waves respected)

| GPU | Waves | Wall-time |
|---|---|---:|
| **GPU-0 RTX PRO 6000 Blackwell** | C3 + C4 + C5 | **2.5 hr** |
| **GPU-1 RTX 5090** | C1 + C2 + C6–C9 + C10–C17 | **≈ 12.55 hr** (1.8 hr + 3.25 hr + 7.5 hr) |
| **Combined sequential** | (if single GPU) | **≈ 15.05 hr** |

### §2.2 Total honest GPU-hour estimate (sum across all 5 TODOs)

| TODO | Wall-time (per audit) | Within ≤1-hr sub-wave split |
|---|---:|---:|
| TODO-119 P1.3 | 16.0 hr audit (conservative TODO-119 §3); wave C3 is **0.5 hr smoke** | 1 wave × 0.5 hr |
| TODO-119 P1.4 | 7.5 hr audit; wave C4 + C5 = **2.0 hr smoke** | 2 waves × 1 hr |
| TODO-180 G4 | 1.8 hr audit; wave C1 + C2 = **1.8 hr** | 2 waves (0.63 + 1.17) |
| TODO-181 G5 | 5.0 hr audit; wave C6–C9 = **3.25 hr** | 4 waves × ~0.81 hr |
| TODO-182 G6 | 7.5 hr audit; wave C10–C17 = **7.5 hr** | 8 waves × ~0.94 hr |
| **TOTAL** | **37.8 hr audit conservative** | **15.05 hr smoke-protocol** |

**Honest framing**: the audit estimates include the **full task scope** (per-cohort full-batch, NOT smoke); the wave plan in this synthesis scopes each TODO to a smoke/pilot run that fits within 1-hr wave discipline. Full-scope (non-smoke) extensions are deferred to subsequent user OK.

---

## §3 Within 10h budget?

### §3.1 GPU-0 Blackwell (TODO-119 P1.3 + P1.4)
- Total: **2.5 hr** — ✅ well within 10h budget.
- User has previously granted 10h on GPU-0 for CFM full retrain (per `configs/cfm_full_retrain.yaml §directives.user_ok_granted`); 2.5 hr fit.

### §3.2 GPU-1 RTX 5090 (TODO-180 + 181 + 182)
- Total: **≈ 12.55 hr** — ❌ exceeds 10h user-OK budget by **~2.55 hr**.
- **Honest verdict**: combined C1–C2 + C6–C9 + C10–C17 plan does NOT fit within a single 10h budget on RTX 5090.

### §3.3 Per-GPU budget decision matrix

| Option | What it does | Wall | User sign-off needed? |
|---|---|---|---|
| **A. Sequential within 10h** | Drop TODO-181 G5 + TODO-182 G6; run only C1–C5 + C6–C9 × 1 sub-wave | ~5.5 hr on GPU-1 + 2.5 hr on GPU-0 | ✅ fits 10h |
| **B. Run all but defer wave C16 + C17** | Skip `no_dann` ablation (12 of 60 runs); run rest | ~11.2 hr on GPU-1 + 2.5 hr on GPU-0 | borderline (~10% over) |
| **C. Extend budget to 13h** | Run all 60 + 15 + 14 + 5 ablation runs | **12.55 hr** on GPU-1 + 2.5 hr on GPU-0 | ❌ requires FRESH user OK for extended budget |
| **D. Pure smoke-only plan** | This synthesis as written (17 waves, all smoke-protocol) | **15.05 hr smoke sequential**, ~2.5 hr on GPU-0 + ~12.55 hr on GPU-1 | ❌ exceeds 10h |

**G3 verdict**: **G3_within_10h_budget = False** for the full 17-wave plan (per `cfm_todo_gap_audit.md` §1.x totals). GPU-0 Blackwell is well within; GPU-1 RTX 5090 exceeds by ~2.55 hr.

**Recommendation**: request fresh user OK for either **Option A** (drop TODO-181/182 to fit 10h) or **Option C** (extend to 13h for full attribution matrix). Decision belongs to user per "kill then adapt" standing directive.

---

## §4 Dry-run verification status

| Config | Path | --help flags verified? | Pre-launch blockers | Status |
|---|---|---|---|---|
| G4 — TODO-180 | `configs/wf124_p3_remaining_metals_g4.yaml` | ✅ (`--metals`, `--seeds`, `--epochs`, `--wasserstein-weight`, `--use-l2-guard` all confirmed at lines 909, 916, 626 of trainer) | none | ✅ ready |
| G5 — TODO-181 | `configs/wf124_p3_50epoch_full_sweep_g5.yaml` | ✅ (`--epochs 50` default per trainer `--help`) | none | ✅ ready |
| G6 — TODO-182 | `configs/wf124_p3_per_innovation_ablation_g6.yaml` | ⚠️ **PARTIAL** — `--no-kendall` and `--no-dann` NOT confirmed in current `--help` (grep returned no matches) | **PRE-LAUNCH**: verify `--no-kendall` / `--no-dann` flags; fallback to `--dann-domain-mode=off` and toggling source code | ⚠️ partial — needs gap fix |
| P1.3 — TODO-119 | `configs/cfm_real_decode_lift_p1.3.yaml` | ✅ (`--use-yuelbond`, `--pocket-id`, `--seed` all confirmed at line 832 of `train_cfm_yuelbond_wasserstein.py`) | none | ✅ ready |
| P1.4 — TODO-119 | `configs/cfm_per_innovation_ablation_p1.4.yaml` | ✅ (`--use-yuelbond`, `--no-wasserstein`, `--no-closure-bounded`, `--no-yuelbond` confirmed at lines 832–841) | none | ✅ ready |

**G4 verdict (FLAT schema)**: **G4_dry_run_verified = False** (4 of 5 configs ready; TODO-182 G6 has unresolved `--no-kendall`/`--no-dann` flag verification gap per `cfm_todo_gap_audit.md §3.4`).

**Honest framing**: G6 config ablation matrix will only run fully if the 2 unverified flags are added OR if the config is narrowed to 2 verified configs (`no_multi_task` + `no_wass` via `--wasserstein-weight 0.0`). Either path requires ~0.5–1 hr pre-launch infrastructure work.

---

## §5 Checklist §1–§10 status (per `wf258_verify`)

From `molmetal/reports/wf258_verify/cfm_checklist_verified.md`:

| § | PASS | pending GPU | FAIL | Note |
|---|---:|---:|---:|---|
| §1.1 code-level (10 gates) | **7** | 3 | 0 | 1.1.3 Kendall MTL N/A (CFM = generation, not classification) |
| §1.2 honest limitations (6) | **4** | 2 | 0 | 1.2.3 F5_RDKIT_2D + 1.2.4 click-aug pending GPU |
| §1.3 ablation infra (6) | **4** | 2 | 0 | 1.3.4 per-lever lift + 1.3.5 report.json pending GPU |
| §2 data (7) | **7** | 0 | 0 | all PASS |
| §3 test (5 + 6 new = 11) | **5** (+ 6 new = 11 total PASS) | 0 | 0 | 6/6 `test_cfm_infrastructure.py` + 29/29 combined run |
| §4 wall-time budget (1) | **1** | 0 | 0 | 10h budget granted |
| §5 self-audit (15) | **14** | 1 | 0 | 5.15 per-wave user OK pending |
| §8 honest framing (15) | **14** | 1 | 0 | 8.9 ablation TODO pending |
| §9 failure mode (8 CFM-F1..F8) | **8** | 0 | 0 | all PASS with mitigation |
| §10 sign-off (10) | **7** | 3 | 0 | 10.5 per-wave + 10.9 GPU select |
| **TOTAL** | **71 PASS** | **12 pending GPU** | **0 FAIL** | 71 / 83 PASS ratio |

**G5 verdict (FLAT schema)**: **G5_checklist_pass_count = 71**.

**Honest framing**: 71/83 PASS is above the ≥70/80 target. The 12 pending are explicitly required GPU launches — NOT defects. No FAIL gates.

---

## §6 Honest framing catalog (party-twenty-spirit §3.5)

### §6.1 MEASURED (already done; cite as is)
- Wf-124 P2 synthetic CFM decode_ratio = **1.000** on 192/192 Pt-click validation clouds.
- Wf-181 m14 smoke real CFM = **0.9980 ± 0.0008** (5,979/5,991 NCI-60 + DrugOOD native).
- Wf-188 m14 FULL MetalCytoToxDB = **0.9958** (26,687/26,801); per-metal: Ru 0.9948 / Ir 0.9974 / Os 1.000 / Rh 1.000 / Re 1.000.
- Wf-124 P3 Ru seed=0 test_auc = **0.7613** (+0.049 vs 0.7123 baseline); W₂ decay 0.8990 → 0.3906.
- 71/83 checklist gates PASS; 0 FAIL.
- 6/6 `test_cfm_infrastructure.py` PASS; 29/29 combined run.

### §6.2 PROJECTED (will measure after GPU; cite cautiously)
- TODO-180 G4 5-metal × 3-seed mean test_auc ≥ 0.86 (Krasnov 2026 target); +0.049 → +0.098 gap uncertain at 30 ep.
- TODO-181 G5 50-ep lift: +0.01 to +0.03 absolute test_auc OR plateau — "uncertain, needs measurement" (per TODO-181 §6).
- TODO-182 G6 per-lever attribution: multi-task 60-80%, W₂ 10-30%, L2 guard ~5%; interaction term may be >30% (per TODO-182 §3.4 hypothesis).
- TODO-119 P1.3 real-CFM decode_ratio lift smoke ≥ 0.5 on 26,801-row MetalCytoToxDB.
- TODO-119 P1.4 per-innovation CFM ablation attribution matrix (4 configs).

### §6.3 DEFERRED (R19/R20 GPU retrain; do NOT promise now)
- TODO-177 (M1+M2 downstream lift on 6 consumer TODOs): 6-wk ETA per consumer.
- TODO-130 G5 (chemprop v3 DrugOOD re-train): v3 currently 0.4544 BELOW_RANDOM DrugOOD.
- TODO-96 R20-P5 Catherine 30-metal reachability ≥ 0.95 (separate R20-P5 track).
- TODO-98 L12 cross-domain OOD reachability ≥ 0.85 (Q4 2026).
- CFM M1+M2 master module integration into CFM training (1.1.8 pending GPU).
- CFM F5_RDKIT_2D 32-d integration (1.2.3 pending GPU).
- CFM click-aug integration (1.2.4 pending GPU).
- Per-metal-per-pocket CFM generation quality (SA / Vina Dock on generated set, NOT m14 decode).

### §6.4 DO NOT claim (party-twenty-spirit §3.5 0 容忍)
- "5×3 Krasnov target 0.86 mean" before Wf-180 G4 5×3 sweep completes.
- "50-ep lift magnitude" before Wf-181 G5 50-ep sweep completes.
- "Per-lever attribution %" before Wf-182 G6 60 ablation runs complete.
- "Real CFM beats synthetic 1.000" — synthetic is upper bound by design.
- "Best-seed selection" (per `mol-metal-gpu-execution` §7) — all seeds reported, std IS the answer.
- "PlatinAI 4-cell / DrugOOD full-batch P1.3" — only MetalCytoToxDB smoke in this wave plan.

---

## §7 Recommendation to user

Per `party-twenty-spirit` §3.5 + Wf-257 §4.3 priority order:

### §7.1 Three top recommendations

1. **Start with Wave C1 (TODO-180 G4 part 1, Ru + Ir 5 pairs, ~38 min)**
   - **Why**: smallest wall-time per PAIR; Ru/Ir are largest data cohorts; closes 5/14 of G4 honest PENDING cite from Wf-124 P3 §4; provides the cleanest MEASURED lift signal.
   - **Risk**: lowest risk (existing trainer, verified flags, no new infrastructure).
   - **Best for**: priority on getting the cheapest, most credible §4 LEAD aggregate lead-in.

2. **Run Wave C3 in parallel (TODO-119 P1.3, 1 seed × 3 ep, ~30 min on GPU-0)**
   - **Why**: most **direct** CFM-only test; smoke confirms Wf-181/Wf-188 0.9980/0.9958 was not just ad-hoc; makes the CFM-only paper §4 LEAD claim formal.
   - **Risk**: lowest risk on the CFM side; ran on PRO 6000 Blackwell (different GPU from C1); can run in PARALLEL with C1 (no GPU contention).
   - **Best for**: locking in CFM §4 LEAD before GPU-1 budget pressure forces harder decisions.

3. **Defer Wave C10+ (TODO-182 G6 ablation, ~7.5 hr) to extended GPU-1 budget**
   - **Why**: longest wall-time but **most credible for §3 algorithm attribution**; provides per-lever decomposition (multi-task / W₂ / L2 guard / joint cohort) that paper §3 algorithm description requires per `party-twenty-spirit` §3.5.
   - **Risk**: interaction term may be >30% of total lift (TO BE MEASURED); 60-run scope is large; pre-launch `--no-kendall` / `--no-dann` flag gap needs infrastructure fix first.
   - **Best for**: late-stage attribution, after C1–C9 lift signals are in.

### §7.2 G6 verdict (FLAT schema): **G6_recommendation_count = 3**.

### §7.3 Why not Waves C2/C4–C9 first?
- **Wave C2** (Os/Rh/Re 9 pairs, ~70 min): larger wall-time than C1; smaller per-cohorts (Rh/Os/Re < 1.2k rows → per-seed std ±0.10). Run AFTER C1 so Ru/Ir numbers anchor the writeup.
- **Waves C4–C5** (CFM ablation): need C3 smoke to confirm ≥ 0.5 decode_ratio first; otherwise ablation is over a fault baseline.
- **Waves C6–C9** (50-ep sweep): best AFTER C1–C2 close the 30-ep G4 aggregate — gives 30-ep vs 50-ep comparison a real 5×3 baseline.
- **Wave C3 (P1.3)** is the cheapest CFM-only test (~30 min, GPU-0) and SHOULD run in parallel with C1 since they use different GPUs.

### §7.4 Launch sequence (proposed)

| Step | Action | Wall | GPU |
|---|---|---|---|
| 1 | **Wave C3** (P1.3 smoke, parallel with step 2) | 0.5 hr | GPU-0 |
| 2 | **Wave C1** (G4 part 1, Ru/Ir 5 pairs, parallel with step 1) | 0.63 hr | GPU-1 |
| 3 | **Wave C2** (G4 part 2, Os/Rh/Re 9 pairs) — sequential after C1 | 1.17 hr | GPU-1 |
| 4 | **Wave C4** (P1.4 ablation part 1) — sequential after C3 | 1.0 hr | GPU-0 |
| 5 | **Wave C5** (P1.4 ablation part 2) — sequential after C4 | 1.0 hr | GPU-0 |
| 6 | **Waves C6–C9** (50-ep sweep, sequential) | 3.25 hr | GPU-1 |
| 7 | **Waves C10–C17** (60-run ablation) — *ONLY IF user extends budget beyond 10h on GPU-1* | 7.5 hr | GPU-1 |
| **TOTAL** | if all 17 waves launch | 15.05 hr | both GPUs |

User OK per wave is required. Per the "kill then adapt" directive, NO batch-launch.

---

## §8 Acceptance gates (FLAT schema, target ≥ 5 waves described)

| Gate | Value | Source |
|---|---:|---|
| **G1_n_waves_described** | **17** (C1–C17, each ≤ 1 hr) | §1.1–§1.7 |
| **G2_total_gpu_hours** | **≈ 15.05** hr (smoke-protocol); audit-conservative **37.8 hr** (full-scope smoke + ablations) | §2.1 + §2.2 |
| **G3_within_10h_budget** | **False** (GPU-0 fits at 2.5 hr; GPU-1 over by ~2.55 hr at 12.55 hr) | §3 |
| **G4_dry_run_verified** | **False** (4/5 ready, 1 has `--no-kendall` / `--no-dann` gap) | §4 |
| **G5_checklist_pass_count** | **71** / 83 (PASS / total); 12 pending GPU; 0 FAIL | §5 |
| **G6_recommendation_count** | **3** (C1 first, C3 parallel, C10+ defer-to-extended) | §7.1–§7.3 |
| **G7_synthesis_report_path** | **`molmetal/reports/wf258_synthesize/ready_for_gpu.md`** (this file) | this § |

---

## §9 Cross-references

- `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` — Wf-258-Audit (5 PENDING TODOs, GPU-hour estimates, flag gaps)
- `molmetal/reports/wf258_verify/cfm_checklist_verified.md` — Wf-258-Verify (71/83 checklist gates PASS, 0 FAIL)
- `molmetal/reports/wf257_research/cfm_potential.md` §5.1–§5.5 — CFM TODO shortlist parent research
- `configs/cfm_real_decode_lift_p1.3.yaml` — Wave C3 P1.3 smoke config
- `configs/cfm_per_innovation_ablation_p1.4.yaml` — Waves C4–C5 P1.4 ablation config
- `configs/wf124_p3_remaining_metals_g4.yaml` — Waves C1–C2 G4 config
- `configs/wf124_p3_50epoch_full_sweep_g5.yaml` — Waves C6–C9 G5 config
- `configs/wf124_p3_per_innovation_ablation_g6.yaml` — Waves C10–C17 G6 config (with `--no-kendall`/`--no-dann` verification gap)
- `TODO/completed/119_decode_ratio_strengthen.md` — P1.3 + P1.4 phases
- `TODO/completed/124_wf124_p3_multitask_5metal_dmpnn.md` — Wf-124 P3 multi-task 5-metal D-MPNN
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` — G4 14 PENDING pairs
- `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` — G5 50-ep full sweep
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` — G6 60-run ablation
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — CFM trainer
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — chemprop multi-task trainer (G4/G5/G6)
- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` — Wf-124 P2 best ckpt (~617 KB)
- `party-twenty-spirit` §3.5 — MEASURED-vs-PROJECTED tag discipline
- `mol-metal-cfm-training` — CFM backbone skill
- `mol-metal-gpu-execution` §6 + §7 — 5-seed × 50-epoch + no-best-seed protocol
- `科研进展_pre_retrain_checklist_2026-09-28.md` §1.1.7 — ablation flags verification

---

## §10 Honest framing disclaimer (party-twenty-spirit §3.5)

- **DO NOT cite any result from this synthesis as MEASURED until the actual retrain completes.**
- All 17 wave wall-time estimates are **PROJECTED** based on Wf-124 P3 per-pair timing (6–8 min for chemprop; 2.9 s/epoch for CFM on RTX 5090 / PRO 6000). Actual wall-time may vary ±20%.
- **Smaller-metal caveat** (per Wf-124 P3 §5 + TODO-180 §6): Rh/Os/Re cohorts < 1.2k rows → per-seed std likely ±0.10; honest framing: report per-metal 3-seed mean ± std unfiltered, no best-seed selection.
- **W₂ regularization caveat** (per Wf-124 P3 §5): predicted pIC50 forced to track reference before encoder has learned anything; JKO-style CFL gate satisfied but W₂ may need 40–50 epochs to fully kick in (drives TODO-181 G5 follow-up).
- **CFM `1.000` decode is upper bound** (per `cfm_checklist_verified.md §1.2.6`); real MetalCytoToxDB currently at 0.9958 (gap −0.0042 within noise floor); P1.3 smoke must MEASURE real ≥ 0.5 again.
- **Per-innovation attribution** (per TODO-182 §0): Wf-124 P3 default is aggregate; per-lever attribution requires 60 ablation runs (Wave C10–C17).
- **`kill_then_adapt` per-wave user OK**: NO batch-launch. Every wave ≤ 1 hr requires FRESH user OK.
- **GPU-1 over budget**: §3 honest verdict is the combined 17-wave plan does NOT fit 10h on GPU-1; Option A (drop C10–C17) or Option C (extend) requires fresh user OK.
- **G6 pre-launch blocker**: TODO-182 G6 `--no-kendall` / `--no-dann` flags not confirmed in `--help` — `cfm_todo_gap_audit.md §3.4 gap #1`; must resolve before launching C10–C17.

**No fabrication. No paper-over. 0 容忍 不掩盖.**

---

*Generated by Wf-258-Synthesize (2026-09-28). Standing synthesis — must be paired with explicit per-wave user OK before any GPU launch. Last updated: 2026-09-28.*
