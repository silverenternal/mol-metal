# Design Boundary

Single-source-of-truth for the project's FROZEN / STABLE / PROPOSED / OUT-OF-SCOPE surface.
Mirrors the `flowa` DESIGN_BOUNDARY.md pattern (~80 lines).

## FROZEN (paper-grounded, do not modify without §5.5 filesystem-truth audit)

- **Catherine closure theorem** (`molmetal/molmetal_lam/lam_chem/closure.py`) — closure-benchmarked over **30 d-block metals × 5,000 samples each** per `molmetal/scripts/verify_closure_theorem.py` (`molmetal/reports/wf130_p2_closure_30_5000/`, `gate_target 0.85`). **⚠️ Corrected 2026-10-10:** this entry previously read "6/6 properties PASS on ≥ 25 d-block metals × 1000 samples… **Per-metal reachability ≥ 0.95 on PlatinAI 4-cell OOD cohort**". The reachability figure was wrong by ~2× against every target metal, and the PlatinAI attribution has no artifact behind it:
  - closure reachability on the six target metals — **Ru 0.652 / Rh 0.6162 / Re 0.5616 / Pt 0.5348 / Os 0.4758 / Ir 0.359**, all against gate 0.85, **all `gate_pass: false`**
  - the separate `ood_per_metal_reachability.json` (scope `tier2_d_block_ood`, strict `reachability_rate`) — **Ru 0.4667 / Rh 0.4833 / Os 0.3636 / Ir 0.3333 / Re 0.2500**
  - **no PlatinAI-scoped reachability metric exists in `metrics/by_metric/`** — the earlier "≥ 0.95 on PlatinAI 4-cell" cited a cohort with no record
  - the "6/6 properties PASS" half is **UNVERIFIED** and deliberately left in place pending a check; note `wf123_p2_closure_bias/per_function_verification.json :: closure_oversample` reports `catherine_6of6_pass_rate = 0.0339` on Rh, which is the *per-compound* all-six rate and is a different quantity — do not read it as a property-level pass/fail.
  - **Paper consequence:** this cannot be presented as "Paper §3.4 main result" on reachability grounds. Any closure claim must carry the 0.85 gate and the per-metal table. See `docs/00_MODEL_NARRATIVE.md` §7 limitations.
- **chemprop v2 trainer** (`molmetal/scripts/train_chemprop_v2_v3.py`) — `v3-fallback` mode default per Wf-263A (LOMO head-collapse fix). 19/25 + 25/25 + 9/9 MEASURED-on-disk cohorts per `metrics/by_metric/chemprop_v2_*.json`.
- **MECHANISM_METRIC_MAPPING** (`TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`) — 21 mechanisms M1-M21 + 22 metrics + 7 ablation design table. Per-Wf-313H 33/33 + Wf-320G 107/107 regression tests.
- **HONEST_NEGATIVE_CATALOG** (`docs/HONEST_NEGATIVE_CATALOG.md`) — 9 → 96 entries expansion history (Wf-263B → Wf-358-C2 → Wf-359-B1 → Wf-369-A3 → Wf-395-A2 → Wf-398-A2 → Wf-399-A2 → Wf-411-A4 → Wf-415-A3 → Wf-426-A4 → Wf-430-B5).

## STABLE (R17-R19 ship surface, backwards-compat by default)

- **chemprop v3 mode** (`--chemprop-mode v3 --enhanced-head`) — opt-in via two-flag pattern per Wf-263A §1.5; re-introduces v3 head-collapse risk if user opts in without reading §9.4 honest-negative.
- **D-MPNN multitask + LOMO** (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`) — `--no-kendall --no-dann --enhanced-head` flag surface; bit-exact backward compat preserved.
- **CFM / YuelBond** (`molmetal/adapters/flow_matching_lipman/`) — `--sa-penalty --sa-target --scaffold-conditioning --top-p --mmd-weight --repulsion-weight --cfm-multiph-pll` flag surface; bit-exact backward compat preserved.
- **MCTS VariancePUCT + replay-learner** (`molmetal/molmetal_lam/search_alg/`) — per Wf-329-W6 ship; vulnerable to vuln_f + vuln_g per Wf-327-F10.
- **CFM property panel + Vina Dock** (`molmetal/scripts/cfm_property_panel.py` + `cfm_vina_dock_panel.py`) — opt-in property evaluation panel; BLOCKED on Pt+2 parameterisation per Wf-299B.

## PROPOSED (R20-R21 ship surface, scaffold stage, MEASURED-vs-PROJECTED honesty gate)

- **Hilbert primal-dual ASTSR** (`molmetal/molmetal_lam/reward/hilbert_primal_dual.py`) — R² lift on real PlatinAI 4-cell + MetalCytoToxDB 5-metal; HONEST_NEGATIVE_FINAL after Wf-377-A1 deep-dive-research move + 9 paths exhausted per Wf-389-A1 + Wf-411-A4 3-band refinement (13/15 cohort-cells MODERATE; 0 STRONG; structural ceiling LIFTED but threshold-reaching axis NOT lifted).
- **Multi-PH + PLL joint Hilbert regression** (`molmetal/molmetal_lam/lam_chem/pll_signature.py`) — R20-P3 TODO-94 scaffold; SCAFFOLD stage; no trainer wiring yet.
- **3-way ASTSR + Hilbert + MCTS triangle** (`molmetal/scripts/triangle_closure.py`) — R20-P4 TODO-95; SCAFFOLD stage.
- **Catherine 30-metal bench** (`molmetal/scripts/bench_catherine_30metals.py`) — R20-P5 TODO-96; SCAFFOLD stage.

## OUT-OF-SCOPE (per AGENTS.md §5.4 + §10 wetlab deprecation)

- **Wet-lab experiments** — 4 LOI drafts + 3 wetlab SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 ALL DEPRECATED; in-silico alternatives only.
- **Tier-1 journal submission** — paper target downgraded JCIM Tier-1 → **Tier-2 computational-only** (JCAMD / J. Mol. Graph. Model. / WIREs Comput. Mol. Sci. / Mol. Divers. / Comput. Biol. Chem.). No wet-lab data required for Tier-2 acceptance per `party-twenty-spirit` §3.5.
- **`paper/*` without explicit user authorization** — TODO-84 active; `mol-metal-paper-writing` skill is READ-only on `paper/sections/*.tex` etc. without explicit user OK per AGENTS.md §5.3.

## Maintenance policy

- **FROZEN**: any modification requires git commit + §5.5 filesystem-truth verification + §5.2 honest framing disclosure.
- **STABLE**: minor modifications OK; major API break requires major version bump + IMPROVEMENT_ENUM update.
- **PROPOSED**: SCAFFOLD stage; MEASURED-vs-PROJECTED honesty gate applies per `party-twenty-spirit` §3.5; flag surface is opt-in (defaults OFF).
- **OUT-OF-SCOPE**: REFUSE per `party-twenty-spirit` §5 (BLOCK / failure response).
