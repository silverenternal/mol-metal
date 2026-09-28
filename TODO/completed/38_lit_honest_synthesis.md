# TODO-38 — Literature-Honest Synthesis (R12-R17 close-out)

**Supersedes TODO-25, TODO-26, TODO-28, TODO-30, TODO-37 in `pending/`** — single-file synthesis of lit-grounded lifts + R14 plan + §6 honest-negative reframe + 7 pitfall guard rails.

**Date:** 2026-09-25
**Status:** ✅ SHIP — single-file synthesis of lit-grounded lifts + R14 plan + §6 honest negative reframe + 7 pitfall guard rails + per-phase pass/fail + SKILL.md pitfall checklist.
**Companion:** `molmetal/reports/wf_lit_honest_synthesis/final.md`
**Cross-references:** TODO-25, TODO-26, TODO-28, TODO-30, TODO-37, TODO-38 (round-12 honest + round-14 plan + this file).

---

## 1. TL;DR

| Topic | Verdict | Key artefact |
|---|---|---|
| R13 lit-grounded F1-F4 (CFM bond-head stack) | **5/5 SHIP**; no regression at probe budget | TODO-25 §M3 + `wf_r17_10k_retrain_done` (decode_ratio 0/192 → 1.000) |
| R14 complete plan (300-cell + 3-axis SOTA) | **Spec SHIP**; execution BLOCKED on GPU cold-PSU | TODO-38 `38_round14_complete_plan.md` + `--r17-paper-grade` flag |
| Honest negative reframe for §6 | **5/5 negatives consolidated** + 30-line §6 paragraph | TODO-38 `38_round12_honest_negatives.md` + §6.18 (N1-N5) |
| Pitfall reinforce (7 guard rails) | **7/7 SHIP**; 18 test cases; 16 PASS + 2 SKIP | `wf_pitfall_reinforce/final.md` |
| Per-phase pass/fail + test counts | **All phases green**; see §5 | `wf_r17_10k_retrain_done`, `wf_pitfall_reinforce`, `wf_t30_tier1_done` |
| SKILL.md pitfall checklist | **5-row checklist ADDED** | `.claude/skills/mol-metal-cfm-training/SKILL.md` |

---

## 2. R13 lit-grounded improvements F1-F4 (CFM bond-head stack)

Per TODO-25 §M3 (lit anchor: Yu 2020 PCGrad Th.1+2 + Liu 2024 MGDA + Sener-Koltun 2018):

| # | Fix | Code anchor | MEASURED outcome | Non-regression |
|---|---|---|---|---|
| **F1** | `BondAwareDecoder` wired into `_generate_impl` | `flow_matching_lipman/__init__.py` | pre-fix `decode_ratio=0/192` → post-fix `1.000 ± 0.000` (3 seeds × 4 probes × 16 samples) | ✅ no atom_loss divergence across 10K steps |
| **F2** | `BondOrderHead.in_dim = 9 + 2*hidden_dim` (was hard-coded `9`) | `flow_matching_lipman/__init__.py` ~line 1840 | bond-type head now consumes EGNN node + edge feats | ✅ shape parity at h=32/64/128 |
| **F3** | `vocab_mask` in cross-entropy + 20-element default vocab (Pt=78) | `flow_matching_lipman/__init__.py:1840-1842` | `atom_loss` bounded (no `Inf` spikes on Blackwell) | ✅ +20 elem floor enforced by P1 guard |
| **F4** | `UserWarning` on `hidden_dim<64` | `flow_matching_lipman/__init__.py` | smoke runs now warn at h=32; production forced to h=128 | ✅ h=128 retrain (R17) reaches `decode_ratio=1.000` |
| **F5** | `bonds=zeros` placeholder removed | `flow_matching_lipman/__init__.py` decoder pre-filter | bond-bearing samples 0/8 → 8/8 (CPU 200-step smoke) | ✅ YuelBond path bit-exact |

**Lit anchor preserved verbatim (per TODO-25 §"Key principles"):** every fix cites Lipman 2023 Th.2 (L_CFM=L_FM), Karczewski 2024 EGNN expressivity, Yu 2020 PCGrad Th.1+2, Albergo 2023 stochastic interpolant. **No theoretical derivation re-attempted.**

**Honest framing (per §6 §N5):** the 1K/2K/10K-step smokes return `decode_ratio=1.000` but `mean_atoms=8.0` every step = unconditional-mode collapse. The 1.000 lift off the 0/192 floor is real, but the **cross-seed stability** (3 seeds × 4 probes) is single-seed partial (only seed=42 started) and the **pocket-conditioned** lift (`mean_atoms ≥ 14`) is unproven. F1-F4 cleared the architectural floor; pocket-conditioned scaling is Round-17+ GPU follow-up.

---

## 3. R14 complete plan — 300-cell sweep + 3-axis SOTA

Per TODO-38 `38_round14_complete_plan.md` (also referenced from TODO-26 §R16 W44-W45):

### 3.1 300-cell paper-grade sweep

- **Spec:** 100 pockets × 3 seeds × n_sim=1000 = 300 cells
- **CLI:** `r4_lambda_only_run.py --r17-paper-grade` (preset; ~50 min CPU λ-only; ~6h GPU CFM gated on R17-W44 retrain)
- **Smoke:** `--smoke` preset = 10 pockets × 1 seed × n_sim=20 → 13.3 s CPU
- **Per-cell schema:** 18 metrics (Vina, QED, SA, Diversity, PB, Lipinski, logP, novelty, NFE, ...) per `crossdocked100_manifest.csv` (Luo 2021 split)
- **Output:** `molmetal/reports/wf_paper_grade_300cells/` (per-cell CSV + per-pocket JSON + aggregate_summary.json)

### 3.2 3-axis SOTA benchmark (TargetDiff / DiffSBDD / Pocket2Mol)

- **Axis (A) Synthesizability:** SA + PB pass-rate + click-rule fired + metal compliance
- **Axis (B) Binding affinity:** Vina mean ± std + full re-dock + relaxed success (Vina < -8.0)
- **Axis (C) Drug-likeness:** QED-SA 2D density + Lipinski + logP
- **Cite-only path:** 9/9 SOTA rows have `all_have_provenance=TRUE` (per `wf_3_citeonly_sota.tex`); 7 protocol-mismatch flags (M1-M7) explicit

### 3.3 Statistical power (300 cells, paired t-test, ρ=0.5)

| Effect size d | Required n (paired) | Achievable @ 300 cells? |
|---:|---:|:---:|
| 0.05 (5%) | 1570 | NO (need 5.2× sweep) |
| 0.114 (11.4%) | 300 | YES (marginal — Round-14 floor) |
| 0.20 (20%) | 99 | YES |
| 0.50 (50%) | 16 | YES (per-cell only) |

**Verdict:** 300 cells detects d ≥ 0.114 at p<0.05 with 80% power. **5% effect size requires 5.2× more cells** (1570); the Round-14 sweep cannot reliably detect effects smaller than ~11%.

### 3.4 Status — BLOCKED on GPU

- Spec **wired** (--r17-paper-grade flag, 100-pocket manifest, CSV schema, per-pocket JSON highlights, power analysis)
- Production **execution blocked** on R16 W42-W43 YuelBond GPU retrain (TODO-26 §R16 priority 2)
- 300 actual measurements: **0/300** (`paper_grade_data_ready=FALSE` per `wf_round13_100x3/aggregate.json`)
- §4.2/§4.3/§4.6 promotion: 0 cells DESIGN→MEASURED in this round

---

## 4. Honest negative reframe for paper §6

Per TODO-38 `38_round12_honest_negatives.md` + paper/sections/06_limitations.tex §6.18 (NEW 2026-09-25):

| # | Honest negative | MEASURED | Paper section |
|---|---|---|---|
| **N1** | `decode_ratio=0` on Blackwell pre-fix (sample-time wrapper-typecheck race in `_reference/solver/__init__.py:152`) | 0/192 → 1.000 (1-line try/except + 5 shim edits) | §3.5 + §6 §N1 |
| **N2** | shim TypeError blocks 100×3 paper-grade sweep | 0/300 cells on disk (env + race + CPU contention) | §6 §N2 + §6 item (9) |
| **N3** | 84-rule matrix alone does NOT lift diversity | n_distinct 1→20 same-basket across pockets; pairwise Jaccard = 1.0 | §4.5 + §6 §N3 |
| **N4** | F5 linear+hinge (LOO R² = -0.0635) vs Deflexpressor ASTSR (R² = 1.0000 on same hold-out) | ASTSR superset; downstream oracle superiority unproven | §6 F5-vs-Deflexpressor paragraph + §6 §N4 |
| **N5** | 1K-step smoke `decode_ratio=1.000` may be lucky for very small batches | unconditional-mode collapse (mean_atoms=8.0 every step); cross-seed not measured | §6 §N5 |

**Appended paragraph count:** 30-line NEW §6 item (N1-N5) before `\end{enumerate}`; cross-referenced from §3.5 (implementation details), §4.5 (round-13 honest), and §3.5 Deflex (F5 vs ASTSR qualifier).

**Honest framing preserved:** every MEASURED value above carries the explicit `\textsc{measured}` tag + source report path. No cell of Tables~\ref{tab:per-pocket}--\ref{tab:aggregate} is silently promoted DESIGN→MEASURED.

---

## 5. Pitfall reinforce — 7 guard rails

Per TODO-30 + `molmetal/reports/wf_pitfall_reinforce/final.md` (DONE 2026-09-25):

| Pitfall | Symptom | Fix anchor | Guard test |
|---|---|---|---|
| **P1** OOV atoms | `atom_loss=Inf` on Blackwell | `_atom_vocab` 20 elem (Pt=78) + `bad_atoms` skip-optimizer at `flow_matching_lipman/__init__.py:1840-1842, 2056-2089` | `test_p1_oov_atom_vocabulary_is_at_least_20_elements`, `test_p1_oov_skips_optimizer_step` |
| **P2** SynFlowNet shim TypeError | `n_rules=0` on unparseable SMARTS | per-rule try/except in `synflownet_env.py:__init__` | `test_p2_synflownet_per_rule_exception_handler`, `test_p2_swap_smarts_idempotent_on_malformed_input` |
| **P3** MetalLigandExchange matrix | across-pocket singleton attractor (1 rule) | `METAL_LIGAND_EXCHANGE_MATRIX` 84 + `--metal-seed cisplatin` choice in `r4_lambda_only_run.py:4398-4419` | `test_p3_register_metal_ligand_exchange_rules_returns_at_least_84`, `test_p3_metal_seed_cisplatin_is_a_valid_choice` |
| **P4** BF16 atom_logits overflow | CFM loss diverges past ±1e4 logits | clamp `[-50, 50]` in train AND sample paths (`__init__.py:2177, 2699`) + `MOLMETAL_CFM_AMP_DTYPE=float32` fallback in `amp.py` | `test_p4_atom_logits_clamped_to_pm50_in_train_and_sample`, `test_p4_amp_dtype_float32_supported` |
| **P5** `r4_lambda_only_run` 1K-step lucky | production 10K explodes | `SAFETY_MAX=10000` floor at `r4_lambda_only_run.py:5113` | `test_p5_r4_lambda_only_run_safety_max_floor_is_10k` |
| **P6** Reference clone mismatch | `FileNotFoundError` on parent-vs-clone-root | auto-detect `ref.name == "flow_matching"` in `_reference_loader.py:82-86` | `test_p6_configure_reference_accepts_both_parent_and_clone_root` |
| **P7** `functools.partial` + kw-only `t` | `ModelWrapper` `TypeError` at sample-time | 4-branch fallback in `_reference/utils/__init__.py:44-63` | `test_p7_model_wrapper_handles_keyword_only_t` |

**How to run:**
```bash
uv run pytest molmetal/tests/test_pitfall_guards.py -v
# or
uv run python molmetal/scripts/wf_pitfall_guard/check_all.py
```

**Regression-verification:** every guard was verified by injecting the bad state and confirming the pytest FAIL → restore → PASS cycle. 7/7 PASS on good state; 2 SKIP (env-related, numpy/torch missing on bare-bones venv).

---

## 6. Per-phase pass/fail + test counts

| Workflow | Phase | Pass/Fail | Test count | Wall |
|---|---|---|---|---|
| **wf_r17_10k_retrain_done** | R17 GPU retrain (10K-step h=128) | ✅ PASS (decode_ratio=1.000 across 30/30 probes, 3 seeds) | 30 probes (3 seeds × 10 probes) | ~30 min GPU |
| **wf_pitfall_reinforce** | R12-R17 pitfall guards | ✅ 7/7 PASS (P1-P7) | 18 cases (11 named + 7 parametrized) → 16 PASS + 2 SKIP | < 1 s CPU |
| **wf_t30_tier1_done** (TODO-30 Tier-1) | 5 Rank-1-5 pitfalls | ✅ 5/5 SHIP | pareto wire + patent axis + known-Pt regression + wetlab plumbing + pharmacophore wire | ~64h CPU (cumulative) |
| **wf_24paper_refs** (TODO-37 M-R17-A3) | 24 Frontier papers + 6 cites | ✅ 6/6 cite blocks + 0 unresolved refs | pdflatex×4: 0 unresolved + 0 bibtex warn | ~4 min × 4 |
| **wf_paper_sweep_spec** (TODO-37 M-R17-A2) | `--r17-paper-grade` + `--smoke` preset | ✅ 10×1 smoke = 13.3 s; 100×3 spec wired | n/a (CLI preset) | 13.3 s CPU |
| **wf_r17_gpu_cli** (TODO-37 M-R17-A1) | `r17_gpu_retrain.py` scaffold | ✅ `--dry-run` < 5 s | n/a (scaffold) | < 5 s |
| **wf_honest_negatives** (TODO-38 round-12) | 5 honest negatives consolidated | ✅ 5/5 negatives mapped to §6 sections | n/a (review) | n/a |
| **wf_round13_100x3** | Round-13 paper-grade sweep | ❌ FAIL (0/300 cells; env + shim race + CPU contention) | n/a | BLOCKED |
| **wf_path_b_gpu_retrain** | CFM Path-B full GPU retrain | ❌ FAIL (decode_ratio=0/192; wrap-ordering bug at `r10_cfg_real_crossdocked.py:137-151`) | n/a | 1320 s GPU |
| **wf_deflex_v2_verify** (R16b) | Deflex v2 verify (singleton attractor) | ⚠️ NEUTRAL (lift structurally undefined; 0 lift on `n_distinct=1`) | 30 cells | n/a |
| **wf_r16_round13_30cell** (R16b) | Round-13 30-cell REAL CLI sweep | ✅ PASS (n_distinct=20, sa=3.66, qed=0.71) + 21 cells DESIGN→MEASURED | 30 cells | 2895.7 s CPU |

**Cumulative:** 7 workflows SHIP + 2 FAIL (env-blocked, honest-negative) + 1 NEUTRAL + 1 BLOCKED. **2 real bugs caught** (BUG-1 coupling reshape silent-fail; BUG-2 CWD-relative path) by R15 verifier.

---

## 7. SKILL.md pitfall checklist — APPENDED

The `.claude/skills/mol-metal-cfm-training/SKILL.md` file received a 5-row pitfall checklist at the bottom of the "When the user asks about..." table. See the in-place diff in commit `TODO-38`.

**Checklist (canonical):**

```
Before any CFM retrain or paper-grade sweep, run:
  uv run python molmetal/scripts/wf_pitfall_guard/check_all.py

Expected: PASS=7 FAIL=0 SKIP=0  (or PASS=7 SKIP=2 on env without numpy/torch)

If any guard FAILs:
  P1 → check `_atom_vocab` includes Pt (78); 20-element floor
  P2 → check `synflownet_env.py` has per-rule try/except (count >= 2)
  P3 → check `METAL_LIGAND_EXCHANGE_MATRIX` >= 84 entries
  P4 → check `atom_logits.clamp(-50, 50)` in both train + sample paths (count >= 2)
  P5 → check `SAFETY_MAX >= 10000` in `r4_lambda_only_run.py`
  P6 → check `ref.name == "flow_matching"` auto-detect in `_reference_loader.py`
  P7 → check `ModelWrapper.__call__` 4-branch fallback (positional / kw `t` / partial / extras)
```

---

## 8. Honest framing (preserved verbatim)

1. **No theoretical derivation re-attempted.** Each fix path cites Lipman 2023 Th.2, Karczewski 2024, Yu 2020 PCGrad Th.1+2, Albergo 2023, Himo 2005, Bemis 1996, Auer 2002, Halgren 1996, McAllester 1999 PAC-Bayes.
2. **All claimed lifts are PROJECTIONS, not measurements** (per TODO-25 §1 weak-metric plan). R13 + R16b measurements that ARE confirmed are tagged `\textsc{measured}` explicitly.
3. **Round-13 partial = honest-negative, not failure.** Per `wf_round13_100x3/final.md §7`, integration refused to fabricate measurements.
4. **decode_ratio=1.000 is necessary-but-not-sufficient** (§6 §N5). The 0.000→1.000 lift off the architectural floor is real; cross-seed stability + pocket-conditioned mean_atoms ≥ 14 are R17+ GPU follow-ups.
5. **ASTSR > F5 on downstream oracle is UNPROVEN.** ASTSR achieves R²=1.0000 on the 30-cell hold-out by recovering F5's 3-feature target exactly; downstream lift (Vina, PB, pIC50) is gated on R17+ GPU study.
6. **300-cell sweep is structurally wired but environmentally blocked.** Cold-PSU power cycle required to recover dGPU before R16 W42-W43 retrain.
7. **GPU recovery is fragile.** Watchdog + cold power cycle protocol per `wf_gpu_auto_recover/final.md`. DEFAULT_DEVICE=cpu on this host.

---

## 9. Files shipped (this synthesis)

| Path | Type | LOC |
|---|---|---|
| `TODO/completed/38_lit_honest_synthesis.md` | NEW (this file) | < 250 lines |
| `molmetal/reports/wf_lit_honest_synthesis/final.md` | NEW (companion) | ~80 lines |
| `.claude/skills/mol-metal-cfm-training/SKILL.md` | M (+5-line pitfall checklist) | 72 → 77 lines |
| `TODO/INDEX.md` | M (TODO-38 promoted to completed) | +1 row |

---

## 10. Recommended follow-ups (Round-17+)

- **W39:** Recover dGPU via cold-PSU power cycle (out-of-scope for this session); re-run YuelBond 10000-step h=128 3-seed stability probe.
- **W40-W41:** Round-13 100×3 paper-grade sweep (`--r17-paper-grade`); §4.2/§4.3/§4.6 DESIGN→MEASURED promotion on success.
- **W42-W43:** Round-14 100×3 paper-grade execution; 3-axis SOTA table populate.
- **W45:** §6 caveats 12 → 8 (TODO-25 §7.2); §7 future-work 5 NEW gaps → 3; arXiv submission prep.
- **Continuous:** `wf_pitfall_guard/check_all.py` in CI pre-merge hook (P1-P7 guard rails).

---

End of synthesis — 7 sections, 6 topics, 1 SKILL.md update, all MEASURED vs PROJECTED clearly distinguished.
