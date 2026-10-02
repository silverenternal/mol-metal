# Wf-300E: Wf-300 Below-Avg Remediation — Synthesis

**Date**: 2026-09-29
**Owner**: Hugo Lee
**Branch**: master
**Parents**: Wf-300A / Wf-300B / Wf-300C / Wf-300D outputs

---

## 1. Per-agent summary

| Agent | Task | Status | Evidence |
|---|---|---|---|
| **Wf-300A** | `drugood_v3_scaffold` BELOW-Field-Avg remediation (chemprop v3 + DrugOOD multi-task pretrain + DANN GRL + scaffold split, target ≥ 0.55) | **PARTIAL — PLAN ONLY** | `molmetal/reports/wf300a_drugood_remediation/remediation_plan.md` ships the recipe but the GPU retrain (5 metal × 5 seed × 50 ep) has NOT been fired. Pass criterion = per-metal mean AUC ≥ 0.55 for ≥ 3/5 metals; honest framing marks the 0.55–0.65 band ASPIRATIONAL, not projected. Constant-`domain_id=0` DANN is a known weakening (TODO-143 §2.2 blocker). |
| **Wf-300B** | PlatinAI S1 (Selenium + undetected-chromedriver) prep for Rusanov 2025 Table S4-S5 IC50 µM harvest (target n_rows ≈ 4,078 × 4 cells = 16,000 IC50, vs S7's 100 rows) | **PARTIAL — PREP ONLY** | `molmetal/reports/wf300b_selenium_prep/plan.md` ships the §3.2 script skeleton + §3.5 hybrid-strategy ~2 hr wall breakdown. NOT launched; awaits USER OK per Wf-292D §5.4. CF detection reliability 60–75% — fallback = stay on S7 + honest-negative §6 paper-limitation paragraph. |
| **Wf-300C** | TODO-119 P1.4 actual GPU launch sequence (Wave-A P1.4 smoke 2 hr + Wave-B G6 C12 smoke 10 min + Wave-C G6 full sweep 7.5 hr = 9.5 hr sequential) | **PARTIAL — LAUNCH SCRIPT SHIPPED** | `scripts/launch_wf300_cfm_main.sh` shipped with `USER_OK` gate verified (auto-aborts without env var). Pre-flight 25/25 PASS, GPU state live-verified (cuda:0 PRO 6000 66.6 GB free; cuda:1 RTX 5090 29.4 GB free, G5 wrapper Ru seed2 in flight). Per-wave abort criteria 7.2.1–7.2.8 wired. Awaits `USER_OK=1 bash scripts/launch_wf300_cfm_main.sh`. |
| **Wf-300D** | Harness auto-fallback wrapper for `StructuredOutput` complex-schema returns (`error_max_structured_output_retries` → schema-shape passthrough) | **COMPLETED** | `/home/hugo/.claude/wrapper.py` (~85 LOC) + `/home/hugo/.claude/test_wrapper.py` (~115 LOC). 9/9 tests PASS. Detector: `data["type"]=="object" and "properties" in data and isinstance(data["properties"], dict)`. Extractor: 3 outcomes (`schema_shape=True`, `value=parsed`, `parse_error=...`). |

---

## 2. Critical remaining

| # | Critical blocker | Source | Why it matters |
|---:|---|---|---|
| **C1** | `USER_OK=1` gate (Wf-300C) — all 3 GPU waves (9.5 hr sequential) blocked on user sign-off per 207 §1.1 + Wf-295D §4.8 | Wf-300C §3 | Without user OK the launch script auto-aborts. Wave-A cheapest (2 hr) — could fire solo with `SKIP_WAVES=B,C` for first-pass signal. |
| **C2** | Wf-300A GPU retrain NOT YET FIRED — `drugood_v3_scaffold` 0.4574 (Ru seed 0) still BELOW_RANDOM; remediation recipe unverified | Wf-300A §3 | Per-metal mean AUC ≥ 0.55 = ship condition; current value is below-random. Need actual GPU run to validate the v3-pretrain + DANN hypothesis. |
| **C3** | Per-row `domain_id` labelling = TODO-143 §2.2 blocker (constant-0 DANN reduces to weaker adversarial regulariser) | Wf-300A §3.2 + TODO-143 | Even with the Wf-300A run firing, the DANN signal is attenuated. Lifting this blocker requires Butina clustering on DrugOOD train + 4-bin domain_id threading. |
| **C4** | PlatinAI S1 (Wf-300B) not launched — paper-grade parity with Rusanov 2025 Table S5 still gated on USER OK | Wf-300B §5 | Without IC50 floats, Wf-127 P4 / Wf-128 P3 / TODO-119 ship surface stays at coverage-only (S7). |
| **C5** | Wf-300D wrapper NOT wired into workflow runner | Wf-300D §7 + §9 | 250+ existing `wf_*.json` scripts still call `JSON.parse(agent(...))` directly; they need the wrapper imported before downstream `agent(...)` calls. Wf-300E integration task suggested but not implemented. |
| **C6** | `directives.user_ok_required: true` per-wave gate at sub-wave granularity NOT implemented (only entry-level) | Wf-300C §3 | Per `cfm_per_innovation_ablation_p1.4.yaml` the per-wave OK gate should fire mid-execution; current script only blocks at script entry. Sub-wave abort falls back to §7.2 `abort_recovery.py`. |

---

## 3. Next priorities (sequenced, MEASURED-or-not)

1. **Wf-300E integration** (cheap, ~1–2 hr, no GPU): wire `/home/hugo/.claude/wrapper.py` into the workflow runner so every `wf_*.json` script uses `extract_agent_result` before `JSON.parse`. Covers C5 and unblocks safer retries on schema-shape passthrough.
2. **Wf-300C Wave-A solo** (cheap, 2 hr GPU, gated on C1): `USER_OK=1 SKIP_WAVES=B,C bash scripts/launch_wf300_cfm_main.sh` — fires P1.4 4-ablation × 1-seed × 3-ep smoke on PRO 6000. Per-ablation `Δ_yuelbond`, `Δ_w2`, `Δ_catherine` MEASURED; validates infra before 7.5 hr G6 commit. Covers C1 partially.
3. **Wf-300A retrain fire** (medium, ~3–4 hr GPU, gated on C1 + GPU slot): 5 metal × 5 seed × 50 ep chemprop v3 multi-task pretrain on DrugOOD train + DANN GRL. If per-metal mean AUC ≥ 0.55 for ≥ 3/5 metals → ship. If PARTIAL [0.50, 0.55) → honest-negative disclosure. If FAIL < 0.50 → next-step = TODO-21 CFM retrain + per-row domain_id (lift C3 blocker).
4. **Wf-300B launch** (medium, ~2 hr CPU, gated on C4 USER OK): hybrid Selenium + requests strategy. Validate n_rows ≈ 4,000 compounds × 4 cells; drop S7 CSV as superseded; wire into Wf-127 P4 (TODO-119 P1.5).
5. **TODO-143 per-row `domain_id` lift** (small, ~0.5 hr dev + re-run): Butina cluster on DrugOOD train (k≈500), bucket into 4 bins by quartile, thread `domain_id` into `DomainAdversarialHead` forward. Re-fire Wf-300A retrain — removes the constant-0 caveat that weakens DANN. Lifts C3.
6. **Wf-300C Wave-B + Wave-C** (large, 7.5 hr GPU, gated on Wave-A PASS + C1): G6 4-cfg × 5 metal × 3 seed = 60 runs on RTX 5090. `--wasserstein-weight 0.0` flag validated by Wave-B 10-min smoke before 60-run commit. `--no-enforce-budget --cfm-yuelbond-required` per Wf-296C + LAMBDA_ROADMAP §9.AF. Closes C1 fully.
7. **Wf-300D wrapper rollout audit** (cheap, ~0.5 hr): grep all `wf_*.json` for `JSON.parse(agent(` and produce a migration manifest; prioritise the 5–10 high-frequency workflows for first-pass wiring.

---

## 4. Honest framing (party-twenty-spirit §3.5)

- **MEASURED**: Wf-300D wrapper 9/9 tests PASS; Wf-300C pre-flight 25/25 PASS; GPU state live (cuda:0 66.6 GB free, cuda:1 29.4 GB free with G5 wrapper in flight at PID 1346380).
- **ASPIRATIONAL / not projection**: Wf-300A per-metal mean AUC ≥ 0.55 (DrugOOD Tab. 7 / Ganin ICML 2015 band); Wf-300B n_rows ≈ 4,078 × 4 (Rusanov 2025 Table S5 expected shape, not validated pre-launch); Wf-300C `target_test_auc: 0.86` in G6 yaml is **PROJECTED, NOT MEASURED**.
- **Honest-negative contingency**: Wf-300A PARTIAL → honest-negative disclosure with PARTIAL verdict; Wf-300B CF block → stay on S7 + §6 paper-limitation paragraph; Wf-300C sub-wave abort → §7.2 `abort_recovery.py` (Wf-262 SHIPPED).
- **No double-launch risk**: Wf-300A (DrugOOD) and Wf-300C Wave-C (MetalCytoToxDB) target different datasets and different GPUs (PRO 6000 vs RTX 5090) when fired in parallel; sequential is safer given the 13 hr user budget.
- **Standing directive (Wf-292D §5.4 + 207 §1.1)**: any > 1 hr GPU/CPU work requires explicit USER OK before launch. Wf-300A / Wf-300B / Wf-300C all clear this bar — none of them fire without user OK gate.