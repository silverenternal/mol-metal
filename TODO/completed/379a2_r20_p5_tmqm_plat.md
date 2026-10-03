# Wf-379-A2 — R20 §5 tmQM + PlatinAI 4-cell joint retrain launch attempt

**Date:** 2026-09-29T16:51:54Z (wall 0 s — launch REFUSED)
**Wf:** Wf-379-A2 (ultracode follow-up per Wf-368-B1 forward-looking gap)
**Status:** SCAFFOLD_NOT_FUNCTIONAL — honest-negative close (no GPU launch performed)
**Cross-references:** Wf-354-B3 (NOT-WRITTEN), Wf-355-A3 (wrapper SHIPPED), Wf-368-B1 (forward-looking gap), Wf-369-A2 (m2 R² FINAL verdict), Wf-335-SYN §3 (CEILING_BREAK on metalcytotox_combined)

---

## §0 TL;DR

Per Wf-369-A2 verdict + Wf-376-A3 plan Path B: m2 R² platinai_4cell + hilbert cohorts need GPU chemprop v3 encoder + Hilbert primal-dual to lift from HONEST_NEGATIVE_R2_DEEP. Wf-379-A2 attempted to launch the Wf-310E R20 §5 tmQM + PlatinAI 4-cell joint retrain wrapper (Wf-355-A3, canonical USER_OK-gated pattern at `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`) on GPU 0 Blackwell (70.7 GB free, fits 28 GB peak per Wf-376-A3 plan).

**Verdict:** GPU launch REFUSED per honest framing. Wrapper invocation would fail at `train_dmpnn_multitask_wasserstein.py` argparse on the FIRST (cell, seed) tuple — 14 of the wrapper's invocation args (`--mode joint_tmqm_platinai`, `--tmqm-csv`, `--platinai-csv`, `--cells`, `--tmqm-metals`, `--hidden`, `--depth`, `--use-multiph-pll`, `--multiph-fourier-bins`, `--pll-signature-dim`, `--w2-weight`, `--cpu-num-workers`, `--num-workers`, `--no-cuda-graphs`, `--out-dir`, `--device`) are NOT in the trainer's parser. The trainer's `--metals` is restricted to `{Ru,Ir,Rh,Os,Re}` (line 247 `FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")`) — wrapper wants 11 metals (Cr/Mn/Cu/Zn/Ru/Rh/Pd/Ag/Pt/Ir/Au), CHOICES CONSTRAINT VIOLATION. Alternative trainer `train_cisplatin_resistance_v2.py` requires `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` which DOES NOT EXIST on disk (only `featurise_stats.json` + `pretrain_metrics.json` present).

**No GPU launches performed.** 0 wall seconds consumed on actual GPU. Per AGENTS.md §5.2 + `mol-metal-gpu-execution` §0 `kill_then_adapt` directive, honest-negative close preferred over wasteful GPU cycle launch.

**Output:** `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json` (full evidence tree).

---

## §1 Status: SCAFFOLD_NOT_FUNCTIONAL honest-negative

### §1.1 VRAM probe

```
$ nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv
0, 70692 MiB, 26650 MiB, 19 %     # RTX PRO 6000 Blackwell, 70.7 GB free (dfine_t146 using 26.6 GB)
1, 32199 MiB,     4 MiB,  0 %     # RTX 5090, idle
```

70.7 GB free on GPU 0 — fits the 28 GB peak projected per Wf-376-A3 plan Path B with margin. VRAM is NOT the blocker.

### §1.2 Wrapper USER_OK gate (PASS)

```
$ head -122 /home/hugo/codes/mol-metal/scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
...
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    ...
    exit 64
fi
```

Canonical Wf-355-B1 USER_OK gate pattern. Exit 64 on refusal. Verified per Wf-354-A2.

### §1.3 Wrapper pre-launch checks (PASS)

```
$ USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh --dry-run
[OK] tmQM d-block verified: 108542 lines / 82296 d-block hits
[OK] PlatinAI 4-cell verified: 4079 lines / 4 cells
[OK] Multi-PH + PLL 32-bin Fourier + 128-d PLL augmentation importable (Wf-348-B1 SHIPPED)
[OK] plan emitted → /home/hugo/codes/mol-metal/molmetal/reports/wf355a3_r20_p5_tmqm_plat/plan.json
[OK] DRY_RUN.log emitted → ... (20 (cell, seed) pairs)
```

Data quality, Multi-PH + PLL import, per-wave checklist, pre-flight, wave-budget all wired. Wrapper is operationally sound. The blocker is downstream — at the trainer invocation.

### §1.4 Trainer invocation FAILS at parse_args (BLOCKER)

Wrapper's intended invocation (per DRY_RUN output):
```
CUDA_VISIBLE_DEVICES=0 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    /home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py \
  --mode joint_tmqm_platinai \
  --tmqm-csv /home/hugo/codes/mol-metal/.data/molmetal/tmQM/tmqm_parsed.csv \
  --platinai-csv /home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell.csv \
  --cells MCF7 A549 A2780 A2780cis \
  --tmqm-metals Cr Mn Cu Zn Ru Rh Pd Ag Pt Ir Au \
  --seeds 0 1 2 3 4 \
  --epochs 50 \
  --hidden 300 --depth 3 \
  --use-multiph-pll --multiph-fourier-bins 32 --pll-signature-dim 128 \
  --w2-weight 0.05 --w2-gradient-mode sort \
  --cpu-num-workers 8 --num-workers 8 --no-cuda-graphs \
  --out-dir /home/hugo/codes/mol-metal/molmetal/reports/wf355a3_r20_p5_tmqm_plat/per_seed_results \
  --device cuda:0
```

Trainer actual `parse_args` CLI surface (output of `usage:` line):
```
--metals {Ru,Ir,Rh,Os,Re} [...]   --seeds SEEDS [...]
--epochs EPOCHS                    --alpha ALPHA
--wasserstein-weight WASSERSTEIN_WEIGHT
--single-task
--ablation-config {mtl_baseline,mtl_structured,mtl_gradproj,single_task}
--n-mc N_MC                        --w2-gradient-mode {sliced,sort}
--use-l2-guard / --no-l2-guard
```

**Wrapper args NOT in trainer (14):**
- `--mode` (no such concept)
- `--tmqm-csv`, `--platinai-csv`, `--cells`, `--tmqm-metals` (no joint data flags)
- `--hidden`, `--depth` (hardcoded)
- `--use-multiph-pll`, `--multiph-fourier-bins`, `--pll-signature-dim` (no Wf-348-B1 augmentation hook)
- `--w2-weight` (named `--wasserstein-weight` in trainer)
- `--cpu-num-workers`, `--num-workers` (no Dataloader hooks)
- `--no-cuda-graphs` (no torch.compile hooks)
- `--out-dir`, `--device` (hardcoded)

**`FIVE_METALS` restriction (line 247):** `FIVE_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")`. The wrapper's 11-metal tmQM pretrain (Cr/Mn/Cu/Zn/Ru/Rh/Pd/Ag/Pt/Ir/Au) violates `--metals` `choices=FIVE_METALS` constraint — argparse will reject.

**Argparse smoke test (CONFIRMED FAIL):**
```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_dmpnn_multitask_wasserstein.py \
    --mode joint_tmqm_platinai --tmqm-csv /tmp/x --platinai-csv /tmp/y
usage: train_dmpnn_multitask_wasserstein.py [-h]
                                            [--metals {Ru,Ir,Rh,Os,Re} [{Ru,Ir,Rh,Os,Re} ...]]
                                            ...
error: unrecognized arguments: --mode joint_tmqm_platinai --tmqm-csv /tmp/x --platinai-csv /tmp/y
```

Failure mode: argparse exits with code 2 on first unrecognized arg `--mode joint_tmqm_platinai`. Wrapper would exit 1 on every (cell, seed) tuple without producing any GPU work.

### §1.5 Alternative trainer `train_cisplatin_resistance_v2.py` (BLOCKED for different reasons)

| Property | Value |
|---|---|
| Supports `--init tmqm` | YES (line 942: `tmqm / se3sb / transport / multiph`) |
| Supports `platinai_4cell.csv` | YES (line 186) |
| Supports joint tmQM + PlatinAI | NO (it's a classifier with 10-d chemoinformatic fingerprint side features, not R² regression) |
| Required pretrained encoder | `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` |
| Encoder on disk? | NO — `models/tmqm_pretrained/` only has `featurise_stats.json` + `pretrain_metrics.json` |
| Suitable for m2 R² recovery? | NO — produces classification `active[4]` not regression R² |

Even if the wrapper were rewritten to use this trainer, the encoder checkpoint would need to be pretrained first, AND the metric would be classification AUC not R² — which does not address the m2 R² recovery goal from Wf-369-A2.

---

## §2 Per-cohort R² / aggregate (NOT MEASURED)

| Cohort | Wf-335-M2 baseline | Wf-367-B1 multi-stream | Wf-379-A2 |
|---|---|---|---|
| `metalcytotox_combined` (n=2000) | R² = +0.2227 CEILING_BREAK | preserved | NOT_MEASURED (SCAFFOLD_NOT_FUNCTIONAL) |
| `platinai_4cell` (n=200) | R² = -0.1251 HONEST_NEGATIVE_R2_DEEP | R² = -0.5106 ± 0.4170 (multi-stream ensemble overfits on n=200) | NOT_MEASURED |
| `hilbert` (n=200) | R² = -0.1268 HONEST_NEGATIVE_R2_DEEP | R² = -0.4933 ± 0.3188 (same overfit) | NOT_MEASURED |
| Aggregate | HONEST_NEGATIVE_PARTIAL 1/3 cohorts | HONEST_NEGATIVE_R2_DEEP on 2/3 cohorts | NOT_MEASURED |
| Krasnov 0.86 gate | NOT HIT | NOT HIT | NOT_MEASURED |

**Krasnov gate status:** NOT_MEASURED — GPU launch REFUSED, no MEASURED R² produced by Wf-379-A2.

**`tmQM_combined` cross-dataset R²:** NOT_MEASURED — wrapper was supposed to evaluate `tmQM_dblock` 29-element cohort (per `metrics/by_metric/tmqm_combined_r2.json` Wf-71 P3 baseline macro_r2_mean=+0.2432) but launch was REFUSED.

---

## §3 Architectural signal floor analysis (preserved from Wf-316A + Wf-354-B3)

Even if the wrapper ran successfully and the trainer integration gap were closed, the R² ceiling on the 12-d RDKit 2D feature stack would cap any m2 R² result:

- **MEASURED (Wf-122 P3 + Wf-129 P3):** PlatinAI 4-cell pIC50 has `|ρ| ≤ 0.019` on every RDKit 12-d column → linear R² upper bound = (0.019)² × 12 = **0.0043** — R² ≥ 0 is mathematically infeasible on the 12-d feature stack alone.
- **Wf-122 P3 post-fix MEASURED R² = -5.66 ± 1.36** (BELOW +0.70 target)
- **Wf-129 P3 Multi-PH + PLL guarded MEASURED R² = -0.0780 ± 0.0148** (BELOW +0.70 target; Multi-PH + PLL 32-bin + 128-d PLL augmentation PREVENTS divergence but does NOT lift the architectural signal floor)
- **Wf-316A STATUS_HOLD:** "Even with PROJECTED chemprop v2 + hidden=300/depth/3 + tmQM pretrain + multi-task + multi-seed, the lift is bounded by the 12-d feature ceiling — not by the wrapper."

**Wf-348-B1 architectural-fix path (SHIPPED):** Multi-PH + PLL 32-bin Fourier + 128-d PLL augmentation module at `molmetal/molmetal_lam/reward/multiph_pll_feature.py` — IMPORTABLE per wrapper pre-launch check, but NOT WIRED into `train_dmpnn_multitask_wasserstein.py` (no `--use-multiph-pll` arg).

**Verdict:** Three layered blockers exist:
1. **Trainer integration NOT-WRITTEN** (wrapper invokes 14 unsupported args; FIVE_METALS choices constraint violated) — Wf-379-A2 finding
2. **Pretrained encoder MISSING** (models/tmqm_pretrained/tmqm_pretrained_encoder.pt DOES NOT EXIST) — Wf-354-B3 finding (partial)
3. **Architectural signal floor at R²=0.0043** on 12-d RDKit 2D stack — Wf-316A finding (PARTIALLY LIFTED by Wf-348-B1 32-bin Fourier + 128-d PLL, NOT YET EMPIRICALLY VERIFIED on PlatinAI 4-cell + Hilbert cohorts)

The m2 R² recovery goal from Wf-369-A2 + Wf-376-A3 Path B requires ALL THREE to be closed:
- Step 1: Author trainer integration (close #1) — ~200 LOC trainer work, multi-day
- Step 2: Pretrain tmQM encoder (close #2) — ~12 GPU-hr standalone
- Step 3: Empirically verify Wf-348-B1 lift on PlatinAI 4-cell + Hilbert (close #3 partially) — ~3 GPU-hr smoke + ~12 GPU-hr full sweep
- Step 4: USER_OK-gated GPU launch with full W₂+chemprop v2+MultiPH-PLL — ~12 GPU-hr

Total GPU budget: ~36 GPU-hr (3 GPU-hr smoke + 12 GPU-hr encoder pretrain + 12 GPU-hr full sweep + 9 GPU-hr multi-stream recovery axis). Significantly exceeds the Wf-355-A3 wrapper's 12-hr smoke budget alone.

---

## §4 Forward-looking fix path

Per Wf-368-B1 + Wf-369-A1 §3.1 + Wf-367-B2 forward-looking: Wf-379-A2 attempted to close one of the four forward-looking gaps from Wf-366-C1 §8. Of the four gaps:
- (a) Pre-existing `p3_wf343c2_per_wave_runtime_check` FAIL 5-LOC argparse fix → **CLOSED by Wf-367-A1+A2**
- (b) Wf-363-A2 GAP-3 `pre_flight_gpu_ready.sh` dry-run SKIP → OPEN
- (c) `launch_wf124_p3_g6.sh` USER_OK gate parity (Wf-365-B2 SKIPPED) → OPEN
- (d) Wf-310E R20-P5 wrapper + trainer integration → **PARTIALLY CLOSED by Wf-379-A2 — wrapper SCAFFOLD verified, trainer integration NOT-WRITTEN documented; recommend Wf-380 NEW ultracode for trainer**

**Wf-380 plan recommendation:**
1. **Wf-380-A1** Author trainer integration — extend `train_dmpnn_multitask_wasserstein.py` with `--mode joint_tmqm_platinai` branch (or write new trainer); lift FIVE_METALS restriction; wire `--use-multiph-pll` + `--hidden/--depth` + `--tmqm-csv` + `--platinai-csv` + `--cells` + `--tmqm-metals`. ~200-400 LOC trainer work, multi-day.
2. **Wf-380-A2** Pretrain tmQM d-block encoder at `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` using `train_fm_cytotox.py` or chemprop v2 standalone. ~12 GPU-hr.
3. **Wf-380-A3** Smoke retrain on PlatinAI 4-cell n=200 — verify Wf-348-B1 lift empirically. ~3 GPU-hr.
4. **Wf-380-B1** USER_OK-gated full sweep launch via wrapper. ~12 GPU-hr.
5. **Wf-380-C1** Update metrics + INDEX + AGENTS sync.

**Alternative (cheaper, in-silico only):** Per Wf-376-A3 Path C, skip the wrapper + tmQM pretrain entirely. Reuse the Wf-335-M2 + Wf-367-B1 multi-stream Ridge ensemble on the existing `metalcytotox_combined` n=2000 cohort where the CEILING_BREAK +0.2227 stands. Acknowledge platinai_4cell + hilbert as small-cohort intrinsic ceiling (50+ feature Ridge overfits on n=200 — this is a bias-variance tradeoff, not a model defect). No new GPU cycles required.

---

## §5 Cross-references

| Doc | Finding |
|---|---|
| `TODO/completed/354b3_r20_p5_tmqm_plat_joint_retrain_wrapper.md` | Wf-354-B3 NOT-WRITTEN finding (CLOSED by Wf-355-A3 wrapper creation) |
| `TODO/completed/354a2_wf310e_user_ok_verify.md` | Wf-354-A2 USER_OK gate verification PASS on Wf-355-A3 canonical pattern |
| `TODO/completed/355a3_r20_p5_tmqm_wrapper.md` | Wf-355-A3 wrapper creation (Wf-354-B3 closure) |
| `TODO/completed/367b1_m2_r2.py` | Wf-367-B1 multi-stream Ridge ensemble HONEST_NEGATIVE_R2_DEEP small cohort overfit |
| `TODO/completed/368b1_user_ok_final.py` | Wf-368-B1 forward-looking gap — wrapper exists, trainer integration NOT-WRITTEN |
| `TODO/completed/369a2_m2_r2_verdict.py` | Wf-369-A2 m2 R² FINAL verdict — 50+ feature Ridge overfits on n=200 |
| `TODO/completed/335_final_synthesis.md` | Wf-335-SYN §3 M2 CEILING_BREAK +0.2227 on metalcytotox_combined n=2000 |
| `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` | Wf-316A STATUS_HOLD 12-d signal floor |
| `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §5 | Phase 4 spec |
| `TODO/completed/339d2_lambda_roadmap_audit.md` §4.4 | Phase 4 NOT LAUNCHED |
| `TODO/completed/96_r20_p5_closure_catherine.md` | R20-P5 sister scope (Catherine closure, not tmQM+PlatinAI joint) |
| `molmetal/reports/wf335m2_r2/result.json` | Wf-335-M2 per-cohort R² MEASURED source |
| `molmetal/reports/wf355a3_r20_p5_tmqm_plat/DRY_RUN.log` | Wf-355-A3 DRY_RUN.log with 20 (cell, seed) pairs enumerated |
| `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json` | Wf-379-A2 evidence tree |

---

## §6 Honest framing

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.2 (honest framing mandate):

**MEASURED today (Wf-379-A2):**
- GPU 0 free VRAM = 70.7 GB on Blackwell (fits 28 GB peak per Wf-376-A3)
- Wrapper USER_OK gate = canonical Wf-355-B1 pattern, PASS
- Wrapper pre-launch checks (per-wave + pre-flight + wave-budget + data quality + Multi-PH/PLL import) = PASS
- Trainer `train_dmpnn_multitask_wasserstein.py` argparse FAILS on wrapper invocation (14 unsupported args; FIVE_METALS choices constraint violated)
- Alternative trainer `train_cisplatin_resistance_v2.py` requires pretrained encoder that DOES NOT EXIST
- 0 GPU launches performed (per `mol-metal-gpu-execution` §0 `kill_then_adapt` directive — refuse wasteful GPU cycle launch when wrapper invocation is known-fail)

**PROJECTED (if Wf-380 close all three blockers):**
- ~36 GPU-hr total (12 pretrain + 3 smoke + 12 full sweep + 9 multi-stream recovery axis)
- m2 R² platinai_4cell + hilbert NOT expected to lift above HONEST_NEGATIVE_R2_DEEP even after full effort (50+ feature Ridge overfits on n=200 — bias-variance tradeoff)
- m2 R² metalcytotox_combined n=2000 CEILING_BREAK +0.2227 stands as is (already lifted)

**STATUS_HOLD (architectural signal floor):**
- Wf-316A 12-d RDKit 2D `|ρ| ≤ 0.019` → R² upper bound = 0.0043
- Wf-348-B1 32-bin Fourier + 128-d PLL augmentation SHIPPED but NOT YET EMPIRICALLY VERIFIED on PlatinAI 4-cell + Hilbert cohorts
- Wf-122 P3 + Wf-129 P3 attempts both BELOW +0.70 target

**Honest framing preserved across wave:**
- m2 R² platinai_4cell + hilbert cohorts remain HONEST_NEGATIVE_R2_DEEP per Wf-367-B1 + Wf-369-A2 verdict
- m2 R² metalcytotox_combined CEILING_BREAK +0.2227 stands per Wf-335-M2
- Wrapper SCAFFOLD_NOT_FUNCTIONAL is the Wf-379-A2 finding (trainer integration NOT-WRITTEN)
- Recommend Wf-380 NEW ultracode OR Wf-376-A3 Path C (cheaper, in-silico only) for closure

**User voice answer:** Per Wf-369-A1 §3.1 + Wf-368-A4 §3 + Wf-367-B2 §0 — the Wf-379-A2 launch attempt was the third follow-up recommended in the Wf-368-B1 forward-looking list. Result: wrapper SCAFFOLD verified but trainer integration gap DOCUMENTED. **NOT** all 4 forward-looking gaps from Wf-366-C1 §8彻底解决 (thoroughly resolved). Recommend:
1. Wf-380 NEW ultracode for trainer integration + tmQM encoder pretrain (~36 GPU-hr total, 2-4 day gate)
2. OR Wf-376-A3 Path C in-silico only — reuse Wf-335-M2 + Wf-367-B1 multi-stream Ridge ensemble on existing n=2000 cohort; acknowledge n=200 platinai_4cell + hilbert as small-cohort intrinsic ceiling (cheaper, immediate)
3. Document decision in Wf-380-B1 OR Path C closeout

**Per AGENTS.md §5.1 standing GPU authorization (USER_OK=1 + VRAM budgeted):** Honored. Launch REFUSED per honest framing — wrapper invocation would have wasted 12 GPU-hr on argparse failure. NO `paper/*` writes per AGENTS.md §5.3 (TODO-84 active).
