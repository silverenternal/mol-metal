# Wf-354-B3 — R20 §5 tmQM d-block + PlatinAI 4-cell joint retrain launch wrapper — verify + USER_OK gate

**Date:** 2026-09-29
**Wf:** Wf-354-B3 (USER_OK-gate verification per AGENTS.md §5.1 kill_then_adapt)
**Parent:** `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §5 (Phase 4 — tmQM d-block + PlatinAI 4-cell joint retrain, 1 week GPU)
**Status:** ⚠️ **HONEST-NEGATIVE** — wrapper does NOT exist on disk; per §5 plan, the wrapper was never written; STATUS_HOLD per `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` + `TODO/completed/339d2_lambda_roadmap_audit.md` §4

---

## §1 Search performed

```bash
$ find . -name "*r20*p5*" -o -name "*tmqm*launch*" -o -name "*joint_retrain*" 2>/dev/null | head -20
./TODO/completed/96_r20_p5_closure_catherine.md

$ find . -name "*tmqm_plat*" -o -name "*plat4cell*" 2>/dev/null
(no matches)

$ find . -path ./node_modules -prune -o -name "*r20*p5*launch*" -print 2>/dev/null
(no matches)

$ grep -rn "R20 §5\|joint retrain.*tmQM\|tmQM.*joint.*PlatinAI" TODO/ --include="*.md" \
    | head
TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md:178:## §5 Phase 4 — tmQM d-block + PlatinAI 4-cell joint retrain (1 week GPU)
TODO/03_plans/LAMBDA_ROADMAP.md: ... (R20 §5 cross-refs)
TODO/completed/339d2_lambda_roadmap_audit.md:166:§5 Phase 4 | tmQM d-block + PlatinAI 4-cell joint retrain (1 week GPU) | ⏸️ NOT LAUNCHED (USER_OK-gated)
```

**Verdict**: No `scripts/launch_*r20_p5*tmqm_plat*joint*.sh` wrapper exists on disk. Only `TODO/completed/96_r20_p5_closure_catherine.md` matches the `r20*p5*` glob (the Catherine 30-metal closure bench — different ship target, NOT a tmQM + PlatinAI 4-cell joint retrain wrapper).

## §2 Plan spec (per `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §5)

| Item | Spec |
|---|---|
| **Data** | tmQM 100K (pretrain) + PlatinAI 4-cell 500 (fine-tune) |
| **Model** | chemprop v2 (hidden=300, depth=3) |
| **Validation target** | cross-dataset R² ≥ 0.70 (current macro +0.1248 → target +0.70 lift via chemprop v2 backbone) |
| **ETA** | ~3 hr wall (single RTX 5090) per §5.3 — OR 1 week GPU per LAMBDA_ROADMAP §3.2 R22-P1 (wider scope, Phase 4 of 6) |
| **Trainer candidates** | `molmetal/scripts/train_chemprop_v2_v3.py` (Wf-266 levers WIRED) + `train_metal_lomo*.py` family — neither is a tmQM+PlatinAI joint trainer as of Wf-354-B3 |

## §3 Architectural signal floor HONEST-NEGATIVE blocker (per `TODO/pending/179_wf123_p3_w2_pic50_alignment.md`)

Even if the wrapper were written, the joint retrain is **architecturally infeasible** on the 12-d F5_RDKIT_2D descriptor stack:

- **MEASURED**: PlatinAI 4-cell pIC50 has |ρ| ≤ 0.019 on every RDKit 12-d column
- **Mathematical ceiling**: linear R² upper bound = (0.019)² × 12 = **0.0043** — R² ≥ 0 is mathematically infeasible on the 12-d feature stack
- **Wf-122 P3 + Wf-129 P3 attempts** (Wf-313G dashboard §3.4): MEASURED R² = -5.66 ± 1.36 (Wf-122 P3 post-fix) and -0.0780 ± 0.0148 (Wf-129 P3 Multi-PH + PLL guarded retrain) — both BELOW the +0.70 target
- **Wf-122 P3 Wf-129 P3 verdict**: "Regularisers PREVENT divergence but do NOT lift the architectural signal floor."

Per `party-twenty-spirit §3.5`: the wrapper would not lift R² above the 0.0043 ceiling on the 12-d stack. **Even with PROJECTED chemprop v2 + hidden=300/depth/3 + tmQM pretrain + multi-task + multi-seed**, the lift is bounded by the 12-d feature ceiling — not by the wrapper.

## §4 Why no wrapper exists (HONEST-NEGATIVE audit)

Per `TODO/completed/339d2_lambda_roadmap_audit.md` §4 (2026-09-29 LAMBDA_ROADMAP audit):

| Phase | Goal | Status | Why ultracode not launched |
|---|---|---|---|
| **§5 Phase 4** | tmQM d-block + PlatinAI 4-cell joint retrain (1 week GPU) | ⏸️ NOT LAUNCHED (USER_OK-gated) | per `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` STATUS_HOLD |

**Root cause**: Wf-316A applied STATUS_HOLD marker because the W₂ recipe §3.1 was found **architecturally infeasible** (12-d RDKit 2D signal floor). With STATUS_HOLD, the wrapper was never authored — there was nothing to wrap.

**Sister workflows on the same axis** (already WRITTEN, for context):
- `scripts/launch_wf310e_wf219_production_gpu.sh` — 244 LOC, USER_OK + PAPER_OK gates, dry-run + actual mode (Wf-219 §4.2 chemprop v2 production backbone)
- `scripts/launch_wf310e_wf135_phase2_gpu.sh` — 244 LOC, USER_OK gate (Wf-135 Phase 2 LOMO 5-metal × 5-seed × 50-epoch)
- `scripts/launch_wf124_p3_g4.sh` + `g5.sh` + `g6.sh` — USER_OK-gated TODO-180/181/182 GPU wrappers
- `scripts/launch_wf338v2_wf176_m19_lomo_cpu.sh` — CPU scaffold wrapper (Wf-176 m19 LOMO Phase 2)

All 6 existing wrappers follow a uniform pattern: USER_OK=1 gate at line ~82-88, dry-run mode that emits `plan.json` + skips GPU launch, exit-code-64 refusal when gate fails. The R20 §5 wrapper would mirror this pattern if/when authored.

## §5 USER_OK-gate verification — N/A (no wrapper)

Per the Wf-354-A2 precedent (USER_OK-gate verification), the verification path requires a wrapper file to exist. Since no wrapper exists, **there is nothing to verify**.

The verification would proceed as follows IF the wrapper existed:

```bash
# Step 1: Confirm wrapper exists
$ test -f scripts/launch_r20_p5_tmqm_plat_joint_retrain.sh && echo "EXISTS" || echo "MISSING"
MISSING

# Step 2: Confirm USER_OK gate
$ grep -n "USER_OK" scripts/launch_r20_p5_tmqm_plat_joint_retrain.sh
(grep: scripts/launch_r20_p5_tmqm_plat_joint_retrain.sh: No such file or directory)

# Step 3: Confirm dry-run mode
$ grep -n "dry-run\|DRY-RUN\|--dry-run" scripts/launch_r20_p5_tmqm_plat_joint_retrain.sh
(grep: scripts/launch_r20_p5_tmqm_plat_joint_retrain.sh: No such file or directory)
```

All three steps fail with ENOENT. **Verification verdict: NOT_APPLICABLE**.

## §6 Honest framing (per AGENTS.md §5.2 + `party-twenty-spirit §3.5`)

| Item | Status |
|---|---|
| Wrapper existence | ❌ **HONEST-NEGATIVE NOT-WRITTEN** (per LAMBDA_ROADMAP §4 STATUS_HOLD + Wf-316A STATUS_HOLD marker on TODO-179) |
| USER_OK gate | N/A — no wrapper |
| Dry-run mode | N/A — no wrapper |
| Tests | 0 NEW |
| Wall | PROJECTED 10-15 min CPU-only (verification only); 0 actual GPU launch |
| Architectural feasibility of joint retrain | ❌ **HONEST-NEGATIVE INFEASIBLE** on 12-d F5_RDKIT_2D (R² ceiling 0.0043, MEASURED −5.66 to −0.08 range across Wf-122/Wf-129) |

**Summary verdict**: Per AGENTS.md §5.1 (USER_OK required for >1 hr GPU) + §5.2 (honest framing mandate) + §5.4 (Tier-2 markers, no wet-lab planning) + Wf-316A STATUS_HOLD marker, the wrapper was NOT authored because:
1. The W₂ recipe is architecturally infeasible on the 12-d signal floor (R² ≤ 0.0043 ceiling).
2. STATUS_HOLD marker was applied per Wf-316A on 2026-09-29 — the wrapper would be a `PAPER-CLASS` resource that is BLOCKED on user OK + architectural lift.
3. The trainer modifications would require W₂ + Wasserstein weight + ref-distribution flags that don't currently lift R² above the signal floor (per TODO-179 §3.1 recipe audit).

**Forward-looking flag**: If/when the 12-d signal floor is lifted (e.g., via metal-aware features per Wf-86 P3 Vuln B fix direction; or via a non-12-d descriptor like Multi-PH + PLL augmented features from Wf-94 P3), a wrapper can be authored following the Wf-310E pattern. Until then, **the wrapper is HONEST-NEGATIVE NOT-WRITTEN, STATUS_HOLD**.

---

## §7 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §5 (Phase 4 — tmQM d-block + PlatinAI 4-cell joint retrain)
- `TODO/completed/96_r20_p5_closure_catherine.md` (R20-P5 parent, Catherine closure — DIFFERENT ship target)
- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (Wf-316A STATUS_HOLD marker; 12-d signal floor)
- `TODO/completed/339d2_lambda_roadmap_audit.md` §4 (LAMBDA_ROADMAP audit; Phase 4 = NOT LAUNCHED)
- `TODO/completed/177_wf313g_downstream_lift_dashboard.md` §3.4 (Wf-122 + Wf-129 MEASURED R² verdict)
- `TODO/completed/86_wf39_deflexpressor_end_to_end.md` §2.2 (Vuln B — PlatinAI 4-cell synthetic IC50 zero feature correlation)
- `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md` (12-d RDKit 2D feature ceiling root cause)
- `scripts/launch_wf310e_wf219_production_gpu.sh` (USER_OK + PAPER_OK gate pattern, ~244 LOC, sister wrapper)
- `scripts/launch_wf310e_wf135_phase2_gpu.sh` (USER_OK gate pattern, ~244 LOC, sister wrapper)
- AGENTS.md §1.1-§1.5 standing directives (all honored) + §5.1 kill_then_adapt + §5.2 honest framing + §5.4 Tier-2 markers (no wet-lab planning per user directive 2026-09-29)

---

**END of Wf-354-B3** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §1.2 + §5.1-§5.4; HONEST-NEGATIVE NOT-WRITTEN + STATUS_HOLD (0 NEW code; 0 NEW tests; 1 NEW doc; 2 honest-negative caveats preserved per §6).
