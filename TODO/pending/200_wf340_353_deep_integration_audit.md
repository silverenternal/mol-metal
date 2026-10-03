# 200 — Wf-340 + Wf-353 Innovation Deep Integration Audit

**Document ID:** TODO-200 / Wf-380-INNOVATION-DEEP-INTEGRATION
**Version:** 1.0
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + agents
**Trigger:** Per `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §5 — 10 CFM innovation points are SCAFFOLDS only, not deep-integrated to trainer

---

## §0 TL;DR

**问题**: Wf-340 (6 wires, ~820 LOC) + Wf-353 (4 wires, ~480 LOC) ship 10 CFM innovation points as CPU-only SCAFFOLDS. **None are deep-integrated to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`**. GPU retraining (Wf-379) uses same D-MPNN/chemprop v3 architecture — no lift from innovation points.

**Solution**: Wire all 8 wire-able innovation points (4 medium + 4 hard) into `train_cfm_yuelbond_wasserstein.py` as CLI flags with **default ON** (post-validation), so the next GPU retrain will actually use them.

**Wall**: 10-15 days CPU-only (8 SCAFFOLD wires × 1-2 days each)

---

## §1 Background

### 10 innovation points (Wf-340 + Wf-353)

| # | Source | Innovation | Current State | Deep Integration Path |
|---|---|---|---|---|
| 1 | Wf-340-A1 | EDM σ-preconditioning (Karras 2022) | SCAFFOLD in `acceleration.py` | `--edm-preconditioning` flag (default ON) |
| 2 | Wf-340-A2 | OT-CFM audit | Audit only | `--use-ot-cfm` flag (default ON) |
| 3 | Wf-340-A3 | StableVM | SCAFFOLD stub | `--stable-vm` flag + full impl in `stable_vm.py` |
| 4 | Wf-340-A4 | SO(3)-averaged FM (Cao 2025 ICML) | SCAFFOLD | `--so3-average` flag + `so3_average.py` |
| 5 | Wf-353-A1 | Triton mini-batch OT kernel (Khainb 2025) | SCAFFOLD | `--triton-ot` flag (default ON) + `triton_mini_batch_ot()` |
| 6 | Wf-353-A2 | Shortcut distillation (Frans 2024) | SCAFFOLD | `--shortcut-distill` flag + `shortcut_distillation.py` |
| 7 | Wf-353-A3 | Consistency regularization (EDM Karras 2022) | SCAFFOLD | `--consistency-weight` flag + `consistency_regularization.py` |
| 8 | Wf-353-A4 | FlowMol unified atom-bond flow (CC-BY-NC) | SCAFFOLD | `--flowmol-unified` flag + LICENSE GATE |
| 9 | Wf-340-A3 | log-normal t-sampling | SCAFFOLD | `--log-normal-t` flag (default ON) |
| 10 | Wf-340-A3 | linear lr-warmup | SCAFFOLD | `--lr-warmup-steps` flag (default ON) |

### Why integration is shallow

- Wf-340 + Wf-353 ships focus on **research + survey + scaffold** per task brief
- Subsequent workflows (Wf-358/359/360/363) focus on **CPU-only fixes + improvements**, not on trainer CLI flag wiring
- Wf-371 + Wf-373 + Wf-375 + Wf-376 focus on **closeout + index sync + synthesis**, not on trainer wiring

**Root cause**: 没有专门 workflow 把 SCAFFOLD wire 到 trainer。

---

## §2 Per-innovation deep integration plan

### §2.1 EDM σ-preconditioning (Wf-340-A1)

**File changes**:
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--edm-preconditioning` CLI flag (default ON)
- `molmetal/adapters/flow_matching_lipman/acceleration.py`: expose `edm_preconditioned_loss(pred, target, sigma)` function
- Wire into main training loop:
  ```python
  if args.edm_preconditioning:
      loss = edm_preconditioned_loss(pred, target, sigma)
  else:
      loss = F.mse_loss(pred, target)
  ```

**Test**: 1 NEW test verifying EDM preconditioned loss matches Karras 2022 reference

**Expected lift**: +5-15% on m3 (binding affinity via EDM noise conditioning)

---

### §2.2 OT-CFM audit (Wf-340-A2)

**File changes**:
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`: add `--use-ot-cfm` flag (default ON)
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py`: expose `ot_solver(samples_ref, samples_pred)` as default W₂
- Wire into trainer:
  ```python
  if args.use_ot_cfm:
      w_distance = ot_solver(samples_ref, samples_pred)
  else:
      w_distance = wasserstein_1d_torchsort(samples_ref, samples_pred)
  ```

**Test**: 1 NEW test verifying OT solver matches scipy reference to <1e-5

**Expected lift**: +10-20% on m2 R²

---

### §2.3 StableVM (Wf-340-A3)

**File changes**:
- `molmetal/adapters/flow_matching_lipman/stable_vm.py`: NEW (full implementation, ~80 LOC)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--stable-vm` flag (default ON)
- Wire into trainer:
  ```python
  if args.stable_vm:
      loss = stable_vm_loss(pred, target, n_scales=4)
  else:
      loss = F.mse_loss(pred, target)
  ```

**Test**: 1 NEW test verifying StableVM multi-scale loss

**Expected lift**: +5-10% on m7 (diversity via multi-scale)

---

### §2.4 SO(3)-averaged FM (Wf-340-A4)

**File changes**:
- `molmetal/adapters/flow_matching_lipman/so3_average.py`: NEW (SO(3) augmentation + equivariant averaging, ~150 LOC)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--so3-average` flag (default ON)
- Wire into trainer:
  ```python
  if args.so3_average:
      rotations = sample_so3_matrices(n=4)
      losses = [loss(pred, target @ r) for r in rotations]
      loss = torch.stack(losses).mean()
  else:
      loss = F.mse_loss(pred, target)
  ```

**Test**: 1 NEW test verifying SO(3) equivariance <1e-3

**Expected lift**: +5-10% on m15 RMSD

---

### §2.5 Triton mini-batch OT (Wf-353-A1)

**File changes**:
- `molmetal/adapters/flow_matching_lipman/acceleration.py`: add `triton_mini_batch_ot()` function (~120 LOC)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`: add `--triton-ot` flag (default ON)
- Wire into trainer:
  ```python
  if args.triton_ot:
      w_distance = triton_mini_batch_ot(samples_ref, samples_pred, batch_size=64)
  else:
      w_distance = ot_solver(samples_ref, samples_pred)
  ```

**Test**: 1 NEW test verifying Triton OT matches scipy reference

**Expected lift**: +5-10× speed (Triton); same accuracy

---

### §2.6 Shortcut distillation (Wf-353-A2)

**File changes**:
- `molmetal/adapters/flow_matching_lipman/shortcut_distillation.py`: NEW (teacher/student distillation, ~120 LOC)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--shortcut-distill` flag (default ON)
- Wire into trainer:
  ```python
  if args.shortcut_distill:
      teacher_pred = teacher_ode(target, n_steps=64)
    student_pred = student_shortcut(target, n_steps=4)
    loss = shortcut_distillation_loss(student_pred, teacher_pred, n_steps_schedule)
  else:
    loss = F.mse_loss(pred, target)
  ```

**Test**: 1 NEW test verifying shortcut model matches teacher within tolerance

**Expected lift**: +10-30% on m14 CFM decode

---

### §2.7 Consistency regularization (Wf-353-A3)

**File changes**:
- `molmetal/adapters/flow_matching_lipman/consistency_regularization.py`: NEW (consistency loss, ~80 LOC)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--consistency-weight` flag (default 1.0)
- Wire into trainer:
  ```python
  loss = F.mse_loss(pred, target)
  if args.consistency_weight > 0:
    sigma = sample_edm_sigma(batch_size)
    consistency_loss = consistency_loss_fn(pred, target, sigma)
    loss = loss + args.consistency_weight * consistency_loss
  ```

**Test**: 1 NEW test verifying consistency loss

**Expected lift**: +5-15% on m14 CFM decode

---

### §2.8 FlowMol unified atom-bond flow (Wf-353-A4)

**File changes**:
- `molmetal/adapters/flow_matching_lipman/flowmol_unified.py`: NEW (license-gated, ~180 LOC)
- `molmetal/models/bond_head.py`: replace with FlowMol wrapper
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--flowmol-unified` flag (default ON post-USER_OK)

**LICENSE GATE**: FlowMol is arXiv:2412.01966 CC-BY-NC. Integration requires USER_OK + license acknowledgment.

**Test**: 1 NEW test (license-gated; SKIP if no USER_OK)

**Expected lift**: +5-15% on m13 validity + m14 decode

---

### §2.9 log-normal t-sampling (Wf-340-A3)

**File changes**:
- `molmetal/adapters/flow_matching_lipman/acceleration.py`: expose `log_normal_t_sampling(batch_size, mu=-0.5, sigma=1.0)` function
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--log-normal-t` flag (default ON)

**Test**: 1 NEW test verifying log-normal sampling distribution

**Expected lift**: +5-10% on convergence speed

---

### §2.10 linear lr-warmup (Wf-340-A3)

**File changes**:
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--lr-warmup-steps` flag (default 1000)
- Wire into scheduler:
  ```python
  if args.lr_warmup_steps > 0:
    scheduler = LinearWarmupCosineDecay(
        optimizer, warmup_steps=args.lr_warmup_steps, total_steps=args.total_steps
    )
  ```

**Test**: 1 NEW test verifying warmup scheduler

**Expected lift**: +5-10% on training stability

---

## §3 Priority order

按 lift × ease:

1. **§2.10 linear lr-warmup** (Easy; 0.5 day)
2. **§2.9 log-normal t-sampling** (Easy; 0.5 day)
3. **§2.5 Triton mini-batch OT** (Medium; 1 day)
4. **§2.1 EDM σ-preconditioning** (Medium; 1-2 days)
5. **§2.2 OT-CFM** (Medium; 1 day)
6. **§2.7 Consistency regularization** (Medium; 1-2 days)
7. **§2.3 StableVM** (Hard; 2-3 days)
8. **§2.4 SO(3)-averaged FM** (Hard; 2-3 days)
9. **§2.6 Shortcut distillation** (Hard; 2-3 days)
10. **§2.8 FlowMol unified** (Hard + LICENSE; 3-5 days + USER_OK)

---

## §4 Acceptance criteria

- All 10 innovations wired to `train_cfm_yuelbond_wasserstein.py` as CLI flags
- All 10 innovations have default ON (post-validation)
- 10 NEW tests (1 per innovation) all PASS
- Pre-flight 89/2/3/94 PRESERVED
- Integration tested end-to-end via `--dry-run` + smoke 50-step training

---

## §5 Cross-references

- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §5 — full innovation audit
- Wf-340-SYN — CFM fast-converging training research
- Wf-353 — Wf-340 deeper integration (4 sub-flows)
- Wf-358-B2 — SO(3)-equivariant D-MPNN scaffold (related but separate)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — target trainer
- `molmetal/adapters/flow_matching_lipman/acceleration.py` — SCAFFOLD home
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` — OT-CFM SCAFFOLD
- AGENTS.md §5.1 — standing GPU authorization
- TODO-207 — pre-flight PASS requirement

---

**END OF TODO-200** — 2026-09-30, 5 sections, 10 deep integration plans, 10 NEW tests expected, ~10-15 days wall
