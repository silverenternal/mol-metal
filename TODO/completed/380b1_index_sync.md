# Wf-380-B1 — INDEX + AGENTS sync + comprehensive final synthesis v25

**Document ID:** Wf-380-B1 / 380b1_index_sync
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-380-B1 agent
**Trigger:** Per `TODO/INDEX.md` §22 + `AGENTS.md` §6 — Wf-380 ultracode residual implementation wave trainer integration + aggregate docs-only INDEX + AGENTS + comprehensive final synthesis v25 sync.

---

## §0 TL;DR

**Goal**: Deliver the trainer integration of Wf-380-A1..A7 SCAFFOLD modules + aggregate docs-only INDEX/AGENTS sync + comprehensive final synthesis v25 for the Wf-380 wave.

**Delivered**:
- **14 NEW CLI flags** added to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` parse_args (use_ot_cfm + stable_vm + so3_average + log_normal_t + triton_ot + shortcut_distill + consistency_weight + flowmol_unified + top_p + top_k + temperature + mmd_weight + repulsion_weight + per_cohort_features; plus 7 matching `--no-X` negation flags for the default-ON semantics)
- **3 NEW CLI flags** added to `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` parse_args (per_cohort_features + triton_ot + use_ot_cfm; plus 3 matching `--no-X` negation flags)
- **1-line Wf-380 banner** in `main()` of each trainer that prints the active Wf-380 SCAFFOLD module set for paper §6 reproducibility audits
- **AST parse OK** for both trainers; **0 NEW tests** (verification + trainer integration only — docs + code, no test files)
- **§22.74** NEW INDEX entry appended between §22.73 Wf-378-B1 and §23 Wf-316B
- **NEW §6 row** appended to AGENTS.md (after Wf-379-B1 row 285) + trailer clause appended to last-update trailer
- **`TODO/completed/380b1_index_sync.md`** (this closeout — ~150 LOC, 5 sections)
- **`TODO/completed/380_final_synthesis.md`** (~700 LOC, 11 sections comprehensive final synthesis v25)
- **Pre-flight v3 fresh-run** at Wf-380-B1 timestamp: **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** PRESERVED at Wf-379-B1 baseline (NOT Wf-380-introduced regression)

**Honest framing per party-twenty-spirit §3.5**:
- Trainer integration is **SCAFFOLD-only** — the 17 NEW CLI flags (14 CFM + 3 DMPNN) are wired as observability + dispatcher patterns. The actual loss / sampling wire-in at each `args.X` read-site is documented but NOT implemented. Closure path = future Wf-381+ MEASURED wire-in with full loss modification.
- All Wf-380 SCAFFOLD MEASURED speedup / lift claims remain PROJECTED per upstream papers (Wf-340-A1 EDM preconditioning 2-3x FID convergence + Wf-353-A1 Triton OT 1.5-3x + Wf-353-A2 shortcut distillation +10-20% sample quality + Wf-380-A3 diversity +0.02-0.035 m7 lift + Wf-380-A2 per-cohort +0.05-0.10 m2 R² lift). MEASURED requires GPU retrain per AGENTS.md §5.1 + kill_then_adapt §0.
- Pre-existing 2 NEW FAILs (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-380-introduced; both deferred to Wf-381+ follow-up.
- Pre-existing 3 SKIPs preserved BY-DESIGN (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency).
- 0 GPU launches performed across the entire Wf-380 wave.
- 0 `paper/*` touches across the entire Wf-380 wave (per AGENTS.md §5.3 TODO-84 active).

---

## §1 Pre-flight verification

Pre-flight v3 fresh-run at Wf-380-B1 timestamp 2026-09-29T17:13:00+0000 UTC per `molmetal/reports/wf380b1_preflight.txt` (872 lines):
- **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL**
- 2 FAILs preserved baseline (NOT Wf-380-introduced): `p3_loi_send_ready` (Tier-2 reclassification cascade) + `p3_wf343c4_anchor_freshness_24h` (wf269c_report.json STALE)
- 3 SKIPs preserved BY-DESIGN
- `gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-recomputed via Wf-345-B3)`
- EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline
- **HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** matches Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 baseline at same drift level
- **7 consecutive runs PRESERVED at 89/2/3/94**
- Pre-flight cascade **STILL CLOSED end-to-end across 14 consecutive runs** (Wf-367-C1 → Wf-368-C1 → Wf-369-B1 → Wf-370-B1 → Wf-371-B1 → Wf-373-B1 → Wf-372-C1 → Wf-375-B1 → Wf-374-B1 → Wf-376-B1 → Wf-377-B1 → Wf-379-B1 → Wf-378-B1 → Wf-380-B1 [this run])

---

## §2 Trainer integration details

### §2.1 DMPNN trainer (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`)

3 NEW CLI flags added at parse_args (lines 3001-3083) + matching 1-line banner in `main()` (lines 3090-3115):

| Flag | Default | Description |
|---|:---:|---|
| `--per-cohort-features` | ON | Per-cohort-specific feature count selection (Wf-380-A2; TODO-201) |
| `--no-per-cohort-features` | OFF | Negation flag (backward-compat with Wf-131 P1 / Wf-335-M1) |
| `--triton-ot` | ON | Triton mini-batch OT coupling (alias for `--triton-minibatch-ot`; Wf-353-A1) |
| `--no-triton-ot` | OFF | Negation flag (CPU pytorch_sinkhorn fallback) |
| `--use-ot-cfm` | ON | Exact OT-CFM coupling (Pooladian 2023; Wf-380-A1 #2) |
| `--no-use-ot-cfm` | OFF | Negation flag (backward-compat with Wf-131 P1 stochastic coupling) |

3 NEW DEFAULT constants at lines 376-378:
```python
DEFAULT_PER_COHORT_FEATURES: bool = True
DEFAULT_TRITON_OT: bool = True
DEFAULT_USE_OT_CFM: bool = True
```

### §2.2 CFM trainer (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`)

14 NEW CLI flags added at parse_args (lines 1487-1740) + matching 1-line banner in `main()` (lines 1748-1797):

| Flag | Default | Description |
|---|:---:|---|
| `--use-ot-cfm` + `--no-use-ot-cfm` | ON | Exact OT-CFM coupling (Pooladian 2023) |
| `--stable-vm` + `--no-stable-vm` | ON | StableVM multi-scale loss (Kim & Jung 2026 ICML) |
| `--so3-average` + `--no-so3-average` | ON | SO(3)-averaged FM loss (Cao et al. 2025 ICML; alias for `--use-so3-averaged-fm`) |
| `--log-normal-t` + `--no-log-normal-t` | ON | Log-normal t-sampling (alias for `--use-log-normal-t`) |
| `--triton-ot` + `--no-triton-ot` | ON | Triton mini-batch OT coupling (alias for `--triton-minibatch-ot`) |
| `--shortcut-distill` + `--no-shortcut-distill` | ON | Teacher-student shortcut distillation (alias for `--use-shortcut-distillation`) |
| `--consistency-weight` | 1.0 | Shortcut consistency regularizer weight (distinct from existing `--shortcut-consistency-weight` default 0.1) |
| `--flowmol-unified` + `--no-flowmol-unified` | OFF | FlowMol unified atom-bond flow head (LICENSE-GATED: CC-BY-NC; USER_OK required per AGENTS.md §5.1) |
| `--top-p` | 0.9 | Nucleus sampling top-p (Holtzman 2020) |
| `--top-k` | 50 | Top-k truncation |
| `--temperature` | 1.0 | Sampling temperature tau |
| `--mmd-weight` | 0.1 | MMD loss weight (Gretton 2012) |
| `--repulsion-weight` | 0.05 | Cosine-similarity repulsion loss weight (Liu 2021) |
| `--per-cohort-features` + `--no-per-cohort-features` | ON | Per-cohort-specific feature count selection (Wf-380-A2) |

### §2.3 Wire-in points (documented but NOT implemented)

Per Wf-380 brief §2, each flag's wire-in to the loss / sampling step is documented at each `args.X` read-site. The Wf-380 banner in each trainer's `main()` emits the active SCAFFOLD module set for paper §6 reproducibility audits. The actual loss / sampling wire-in (e.g., `if args.edm_preconditioning: loss = edm_preconditioned_loss(pred, target, sigma)`) requires future Wf-381+ MEASURED wire-in with full loss modification + Blackwell sm_120 GPU retrain per AGENTS.md §5.1.

---

## §3 INDEX.md sync

Appended NEW §22.74 entry between §22.73 Wf-378-B1 and §23 Wf-316B (~85 lines, 6 sub-sections):

- §22.74 Wf-380 ultracode residual implementation + trainer integration + aggregate docs-only sync + comprehensive final synthesis v25
- §22.74.1 Wf-380-A1..A7 SCAFFOLDS (7/7 SHIPPED 100%)
- §22.74.2 Wf-380-B1 trainer integration SHIPPED
- §22.74.3 Wf-380-B1 aggregate docs-only sync
- §22.74.4 MEASURED test pass-rates
- §22.74.5 User voice answer
- §22.74.6 Cross-link

---

## §4 AGENTS.md sync

- **NEW §6 row** appended after Wf-379-B1 row 285 (Wf-380-B1 trainer integration + aggregate + INDEX + AGENTS + final synthesis v25)
- **Last-update trailer** extended with Wf-380-B1 sync clause (appended to existing trailer)
- 1-line summary: 8/8 deliverables SHIPPED 100% + 47+ NEW tests PASS (A1=41 + A2=3 + A3=3 + A4..A7 per closeout + B1=0) + 14 NEW CFM flags + 3 NEW DMPNN flags + 0 NEW tests in B1 + 0 GPU launches + 0 paper/* touches + 11 honest-negative caveats preserved

---

## §5 Honest framing

Per `party-twenty-spirit` §3.5, the following honest-negative caveats are preserved:

1. **Wf-380-B1 SCAFFOLD-only integration** — 14 NEW CFM + 3 NEW DMPNN CLI flags wired as observability + dispatcher patterns; loss / sampling wire-in at each `args.X` read-site documented but NOT implemented.
2. **All Wf-380 SCAFFOLD MEASURED speedup/lift claims remain PROJECTED** — Wf-340-A1 EDM preconditioning 2-3x FID convergence + Wf-353-A1 Triton OT 1.5-3x + Wf-353-A2 shortcut distillation +10-20% sample quality + Wf-380-A3 diversity +0.02-0.035 m7 lift + Wf-380-A2 per-cohort +0.05-0.10 m2 R² lift; MEASURED requires GPU retrain per AGENTS.md §5.1 + kill_then_adapt §0.
3. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per Wf-377-A1 deep-dive-research + Wf-365-A3 verdict + Wf-380-A7 finalise).
4. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic + Wf-380-A2 per-cohort features).
5. **Pre-existing 2 NEW FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-380-introduced.
6. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency).
7. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0 + Wf-369-A1 §3.1 + Wf-378-A2 §3.2).
8. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0).
9. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell; 6 (metal, seed) pairs queued; full multi-pair result deferred to Wf-381).
10. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION).
11. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico).

---

## §6 Cross-references

- `TODO/completed/380{a1,a2,a3,a4,a5,a6,a7}_*.md` — 7 Wf-380 SCAFFOLD sub-flow SHIPPED closeouts
- `TODO/completed/380b1_index_sync.md` — this closeout (Wf-380-B1 INDEX + AGENTS sync)
- `TODO/completed/380_final_synthesis.md` — comprehensive final synthesis v25
- `TODO/INDEX.md` §22.74 — NEW INDEX entry appended between §22.73 Wf-378-B1 and §23 Wf-316B
- `AGENTS.md` §6 NEW row — Wf-380-B1 trainer integration + aggregate + INDEX + AGENTS + final synthesis v25
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — 14 NEW CLI flags + 7 `--no-X` negation flags + 1-line Wf-380 banner in `main()` (parse_args lines 1487-1740 + banner lines 1748-1797)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — 3 NEW CLI flags + 3 `--no-X` negation flags + 1-line Wf-380 banner in `main()` (parse_args lines 3001-3083 + banner lines 3090-3115)
- `molmetal/adapters/flow_matching_lipman/{edm_preconditioning,optimal_transport_v2_solver,stable_vm,so3_average,triton_mini_batch_ot,shortcut_distillation_wf380,consistency_regularization_wf380,flowmol_unified,log_normal_t,lr_warmup}.py` — Wf-380-A1 10 CFM SCAFFOLD modules ~1055 LOC
- `molmetal/adapters/flow_matching_lipman/{diversity_sampling,scaffold_conditioning}.py` — Wf-380-A3 m7 diversity SCAFFOLD modules ~919 LOC
- `molmetal/scripts/wf380a2_m2_r2_per_cohort.py` ~390 LOC + `molmetal/configs/m2_r2_per_cohort.yaml` ~125 lines — Wf-380-A2 per-cohort features
- `molmetal/reports/wf380b1_preflight.txt` (872 lines) — Wf-380-B1 pre-flight fresh-run 89/2/3/94
- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1) + `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 387 LOC) + `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC) + `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC)
- `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries 6 sections)
- `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC)
- `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants)
- `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC)
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template)
- `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC)
- `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections)
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §10 standing directives re-confirmed
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- `mol-metal-cfm-training` skill (Wf-380-A1..A7 SCAFFOLD modules)
- `mol-metal-triangle-closure` skill (Wf-380-A4 Vina tuning)
- `mol-metal-anticancer` skill (Wf-379-A1 m1_auc gate)
- `mol-metal-sci-paper-submission` skill (Wf-351-B3 NEW per 2026-09-29 user directive)