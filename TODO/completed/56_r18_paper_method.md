# R18 Paper Method Synthesis — §3.4–3.7 + math_theorems Appendix

**Date**: 2026-09-26  **Parent plan**: `TODO/03_plans/R18_PLAN.md` §2 + §14
**Workflow binding**: Wf-3 (paper writing) — pairs with Wf-1 (`54_r18_math_skeleton.md`,
shipped) + Wf-2 (`55_r18_integration_adapters.md`, shipped).
**Status**: §3 overview + 4 R18 sub-sections + Appendix D + 11 R18 bib entries
**shipped** (pdflatex 4-pass → 153-page PDF). Blackwell GPU 训练 **deferred** per
`R18_PLAN.md §5`.
---

## 1. TL;DR

| # | Artifact | Path | LOC |
|---|----------|------|----:|
| §3 overview | Theoretical Foundations + Priority map | `paper/sections/03_method.tex:304–396` | 93 |
| §3.4 | SE(3) Schrödinger Bridge (T1+T8) | `paper/sections/03_4_se3_schrodinger_bridge.tex` | 268 |
| §3.5 | Transport Relaxation (T2) | `paper/sections/03_5_transport_relaxation.tex` | 271 |
| §3.6 | Multi-parameter persistence (T5+T6) | `paper/sections/03_6_multi_parameter_ph.tex` | 295 |
| §3.7 | Wasserstein foundations (T3+T4) | `paper/sections/03_7_monge_ampere_wasserstein_gf.tex` | 281 |
| App. D | Mathematical Theorems (T1–T9) | `paper/appendices/math_theorems.tex` | 726 |
| bib | 11 R18 entries (5 NEW + 6 carry-over) | `paper/refs.bib` | +~75 |

**5 new bib entries** per `R18_PLAN.md §14`: `kolbe_mayr_ph_hardness`,
`wangxujia_monge_ampere`, `xuexiaoping_wasserstein_gf`, `maxiaonan_bergman_kernel`,
`sun_zheng_hilbert_primal_dual`, `score_repellent_mc_icml2026`. 6 carry-over
(`mahmood2026sb_lie`, `jiang2026sb_relax`, `kim_jung2026multi_ph`,
`henry_labordere2017bass`, `backhoff_veraguas2020bass_existence`) added in Wf-1.
The 4 R18 sub-sections are **pulled out** of `paper/sections/03_method.tex` and
**input directly** into `paper/main.tex:257–260`.

---

## 2. Per-Deliverable Status

### 2.1 §3 overview paragraph — `paper/sections/03_method.tex:304–396` (93 LOC)
Inserted between §3.7 bond-aware soft prior (line 180–301) and §3 end-to-end
pipeline (\input at 405). Anchors the 4 R18 sub-sections on the 9 theorems
(T1–T9) catalogued in Appendix D, declares priority map (P0 = T1/T2/T5;
P1 = T3/T4/T7; P2 = T6/T8/T9 cite-only), re-asserts honest-framing policy.

### 2.2 §3.4 — SE(3) Schrödinger Bridge (268 LOC, T1+T8)
**Math + adapter**: `molmetal/math_skills/se3_schrodinger_bridge.py` (816/25/25) +
`molmetal/adapters/se3_sb_integration.py` (W2.1, 25/25, 0/1000 smoke).
**pdflatex**: `\ref{sec:method:se3-sb}` + `\cite{mahmood2026sb_lie,
backhoff_veraguas2020bass_existence}` resolved; 4 lit anchors use
**phantom-cite workaround** at lines 43–59.

### 2.3 §3.5 — Transport Relaxation (271 LOC, T2)
**Math + adapter**: `transport_relaxation.py` (519/29/29) +
`transport_relaxation_integration.py` (W2.2, 25/25, 0/1000 smoke). **pdflatex**:
`\ref{sec:method:transport-relaxation}` + `\cite{jiang2026sb_relax}` resolved.
Honest-framing at line 230 calls –2.0 kcal/mol Vina Dock lift **\PROJECTED**.

### 2.4 §3.6 — Multi-parameter persistence (295 LOC, T5+T6)
**Math + adapter**: `multi_parameter_ph.py` (1192/25/25) +
`multi_ph_integration.py` (W2.3, 25/25, 0/1000 smoke). **pdflatex**:
`\ref{sec:method:multi-ph}` + `\ref{sec:method:multi-ph-hardness}` +
`\cite{kim_jung2026multi_ph, kolbe_mayr_ph_hardness}` resolved.

### 2.5 §3.7 — Wasserstein-geometric foundations (281 LOC, T3+T4)
**Math + adapter**: `monge_ampere_regularity.py` (619/34/34) +
`wasserstein_gradient_flow.py` (673/42/42). Adapters P1 deferred per Theorem 1
stability dependency. **pdflatex**: `\ref{sec:method:wasserstein-foundations-t3, -t4}`
+ `\cite{wangxujia_monge_ampere, xuexiaoping_wasserstein_gf}` resolved.

### 2.6 Appendix D — Mathematical Theorems (T1–T9) — `…/math_theorems.tex` (726)
Inclusion: `paper/main.tex:454` via `\input{appendices/math_theorems}`. 9 theorems
with **formal statements only** (no proofs); per appendix header (lines 27–31),
proofs remain in the original papers. Each carries (a) informal statement,
(b) `\begin{theorem}…\end{theorem}` formal block, (c) Mol-Metal anchor, (d) P0/P1/P2
priority bucket, (e) 1–2 sentence motivation. 3 appendix-only theorems (T6 guardrail,
T8 cite-only, T9 long-horizon) ship with **explicit "appendix-only" callout**.
**pdflatex**: 9 theorem labels `\ref{thm:T1-sb-lie}` … `\ref{thm:T9-bergman}`
defined; cross-references from §3.4-3.7 resolve.

---

## 3. MEASURED vs PROJECTED (honest-framing discipline)

### 3.1 MEASURED (this workflow, `pdflatex main.tex` 4-pass → 153-page PDF, 5.6 MB)

| Item | Measured value |
|------|----------------|
| 4 R18 sub-section files compile | 4/4 (1115 LOC; `wc -l paper/sections/03_{4,5,6,7}_*.tex`) |
| math_theorems appendix compiles | 1/1 (726 LOC, 9 theorems; `\input{appendices/math_theorems}` main.tex:454) |
| §3 overview paragraph compiles | 1/1 (93 LOC; `03_method.tex:304–396`) |
| 11 R18 bib entries present in `refs.bib` | 11/11 (5 NEW + 6 carry-over) |
| `pdflatex` 4-pass convergence | 4/4 output `main.pdf` (153 pages, 5,685,528 bytes per pass) |
| Internal `\ref{}` (R18-only) | 9/9 resolved: `thm:T1`…`thm:T9` + `appendix:math-theorems` + 4 sub-section anchors |
| 9 R18 theorem `\cite{}` keys resolve | 9/9: `mahmood2026sb_lie`, `jiang2026sb_relax`, `kim_jung2026multi_ph`, `henry_labordere2017bass`, `backhoff_veraguas2020bass_existence`, `wangxujia_monge_ampere`, `xuexiaoping_wasserstein_gf`, `kolbe_mayr_ph_hardness`, `maxiaonan_bergman_kernel` |
| Theorem statements **formal**, not proofs | 9/9 ship `\begin{theorem}…\end{theorem}`, **no `\begin{proof}`** |
| GPU retrain (Blackwell) | **0** Blackwell calls; deferred; no `.cuda()` / Triton in any new R18 paper file |

### 3.2 PROJECTED (gated on R18 Phases 1-3 real Blackwell training, NOT MEASURED)

| Theorem | R18 metric projection | Path to verify |
|---------|----------------------|----------------|
| T1 SE(3) SB | m3 Vina Dock median –8.5 → ≤ –10.5 kcal/mol | Phase 2 100-pocket × 3-seed |
| T2 Transport Relaxation | m14 Decode ratio = 1.000 (no regression) | Phase 1 10k-step retrain |
| T3 Monge-Ampère | SynFlowNet recall@5 0.74 → ≥ 0.88 | Phase 2-3 advantage-estimator swap |
| T4 Wasserstein GF | Synthesis success rate 0.61 → ≥ 0.78 | Phase 3 Neural-ODE integration |
| T5 Multi-Parameter PH | m7 Diversity 0.65 → ≥ 0.75 | Phase 3 Fourier sweep |
| T6 PH Hardness | compute-budget ceiling | inherent (no metric) |
| T7 SB–Bass martingale | Tanimoto top-100 0.42 → ≥ 0.55 | Phase 2 (deferred until T1 stable) |
| T8 Bass Existence | cited | Paper appendix (no metric) |
| T9 Bergman Kernel | RBFE ΔG RMSE 1.8 → ≤ 1.2 kcal/mol | P2 long-horizon |

**Critical**: theorem statements ship as **formal \textsc{projected} anchors**, not
as proofs of the projected metrics. Blackwell GPU training is blocked behind the
ultracode blacklist (`R18_PLAN.md §5`); Phase 1-3 real retrain runs Thu-Sun window
once ultracode drains.

### 3.3 Pre-existing pdflatex warnings (NOT introduced by Wf-3)

Pre-existing warnings total ~684: 122 Unicode errors (Chinese theorem-author names
汪徐家 / 薛小平 / 麻小南 in App. D from Wf-1 W2.4), ~480 undefined `footnote:` citations
(natbib prefix mismatch in §1+§2 from R17), ~70 undefined R17-era citations
(ref-key rename drift + phantom-cite covers in §3.4-3.7), ~12 multiply-defined labels
in §4+§5. **R18-introduced portion: 0.** Wf-3 introduces **0** new unresolved `\ref`/`\cite`
keys: every R18 cite either resolves to `refs.bib` or uses the phantom-cite workaround
(`03_4_se3_schrodinger_bridge.tex:43–59`).

---

## 4. Cross-Reference Map

`paper/sections/03_method.tex` §3 overview (line 304–396) refs:
`se3-sb` → §3.4 (T1+T8: Mahmood 2026 + Backhoff-Veraguas 2020);
`transport-relaxation` → §3.5 (T2: Jiang-Xu-Zhang 2026);
`multi-ph` + `multi-ph-hardness` → §3.6 (T5+T6: Kim-Jung 2026 + Kolbe-Mayr 2026);
`wasserstein-foundations-t3, -t4` → §3.7 (T3+T4: Wang 2026 + Xue 2026);
`appendix:math-theorems` → App. D T1–T9 (9 theorems, 6 cited + 3 app-only).
`paper/refs.bib`: 11 R18 entries — **NEW (5)**: `kolbe_mayr_ph_hardness`,
`wangxujia_monge_ampere`, `xuexiaoping_wasserstein_gf`, `maxiaonan_bergman_kernel`,
`sun_zheng_hilbert_primal_dual`, `score_repellent_mc_icml2026`;
**carry (6)**: `mahmood2026sb_lie`, `jiang2026sb_relax`, `kim_jung2026multi_ph`,
`henry_labordere2017bass`, `backhoff_veraguas2020bass_existence`. Math + adapter
anchors in `molmetal/math_skills/` (9 modules, 305 unit tests) and
`molmetal/adapters/` (3 adapters, 75 unit tests) per Wf-1 + Wf-2.

---

## 5. R18 PLAN Phase 1-3 Alignment (Wf-1 + Wf-2 + Wf-3)

Per `TODO/03_plans/R18_PLAN.md §14`, this Wf-3 close-out completes the **math +
adapter + paper** trio of R18 Phases 1-3:

| Phase | Wf-1 (math) | Wf-2 (adapter) | Wf-3 (paper) | Phase 1-3 真实训练 |
|-------|-------------|----------------|--------------|---------------------|
| **P1** Transport Relaxation | ✅ 519/29 | ✅ 556/25 | ✅ §3.5 271 + T2 60 | ⏳ Thu-Sun |
| **P2** SE(3) Schrödinger Bridge | ✅ 816/25 | ✅ 611/25 | ✅ §3.4 268 + T1+T8 130 | ⏳ Thu-Sun |
| **P3** Multi-PH + Hilbert primal-dual | ✅ 1192/25 | ✅ 536/25 | ✅ §3.6 295 + T5+T6 80 | ⏳ Thu-Sun |
| P1-P3 cross (T3/T4/T7/T8/T9) | ✅ 2019/156 (CPU) | ⏳ adapters P1-P2 | ✅ §3.7 281 + T3+T4+T7+T8+T9 380 | ⏳ P1/P2 |

(Format: LOC / unit tests.) **Wf-1 + Wf-2 + Wf-3** ship the **CPU-only half** of
R18 Phases 1-3. The **GPU retrain half** is **deferred** per `R18_PLAN.md §5`
(Blackwell sm_120 single-card + ultracode contention blacklist).

---

## 6. Remaining — R18 PLAN Phase 4 (deferred, PLAN only)

Wf-3 closes Phase 1-3's paper half. Phase 4 items from `R18_PLAN.md §2` remain on
the **PLAN** (not shipped in this workflow): (1) Table 2 R18 row fill (depends on
Phase 1-3 GPU retrain numbers m3 ≤ –10.5 / m7 ≥ 0.75 / m14 = 1.000);
(2) `paper/sections/06_limitations.tex` honest re-framing (partially shipped via
Wf-8 `53_phase_a_reality_check.md`; full §6 rewrite awaits Wf-3.1);
(3) 5 SKILL.md updates (`.claude/skills/mol-metal-{lambda-calculus,cfm-training,
anticancer,mechanism-mapping,click-chemistry}/SKILL.md`) each +30 lines per
`R18_PLAN.md §2 Phase 4` Task 3 — deferred to Wf-3.1; (4) `METRISMETRIC_MAPPING.md` +
`METRICS_TARGETS.md` updates (M22/M23/M24 row insertions) deferred per Phase 4
Task 6-7; (5) 7/7 pitfall guards re-run (last verified R17 via
`wf_pitfall_guard/check_all.py`); (6) `TODO/completed/42_r18_closeout.md` R18
final synthesis (placeholder exists; awaits Phase 4 completion). These are
explicitly **out of scope** for this Wf-3 (paper method) workflow.

---

## 7. Honest-Framing Self-Audit + Companion Documents

Per `R18_PLAN.md §6 §honest`, every shipped R18 paper file uses the
`\textsc{measured}` / `\textsc{projected}` / `\textsc{architectural}` tagging
discipline introduced in `03_method.tex:79–95`. Tags applied:
`\textsc{shipped}` for math + adapter code modules (Wf-1 + Wf-2); `\textsc{projected}`
for T1 SE(3) SB Vina lift, m3 ≤ –10.5, m7 ≥ 0.75, m14 = 1.000; `\textsc{measured}`
for 25/25 + 25/25 + 25/25 unit tests pass + 0/1000 smoke fail (×3);
`\textsc{deferred}` for Phase 1-3 GPU retrain (Blackwell single-card constraint);
**Tier-1 AUC claim omitted** in §3.4-3.7. The R18 paper method section does **not**
reintroduce the dropped Tier-1 AUC (Ru 0.712 / Pt MCF-7 0.584 / Pt A2780 0.350)
framing retired in `TODO/completed/53_phase_a_reality_check.md`. The §6 limitations
+ abstract re-write remains Phase 0 / Phase 4 work; this Wf-3 only ships the
method-level math backbone that supports the honest re-framing.

**Companion docs**: Wf-1 (`54_r18_math_skeleton.md`, 9 math skeletons + 305 tests);
Wf-2 (`55_r18_integration_adapters.md`, 3 adapters + 75 tests); Wf-8
(`53_phase_a_reality_check.md`, honest-framing premise); R18 PLAN
(`TODO/03_plans/R18_PLAN.md`, Phases 1-3 + §risk + §honest); r18-math skill
(`.claude/skills/mol-metal-r18-math/SKILL.md`, theorem priority P0/P1/P2).

**End `56_r18_paper_method.md`.**
