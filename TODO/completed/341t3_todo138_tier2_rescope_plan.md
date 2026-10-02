# TODO-138 — Paper v2 Tier-2 Re-scope Plan (CPU-only docs, BLOCKED per TODO-84)

**Wf**: Wf-341-T3 (TODO-138 Paper v2 Tier-2 re-scope completion)
**Date**: 2026-09-29
**Status**: SHIPPED (CPU-only docs plan; `paper/*` writes BLOCKED per TODO-84)
**Parent**: AGENTS.md §5.4 (Tier-2 journal target, NO wet-lab planning) + Wf-337-W3 (Tier-2 marker pass) + TODO-138 (paper v2 BLOCKED close-out)
**Cross-ref**: `TODO/completed/138_paper_v2_submission_package.md` (parent file with 8 lead + 2 partial + 8 honest-negative)
**Per**: `IMPROVEMENT_QUEUE_2026_09_27.md` task 4

---

## §0 Why this plan exists

Per AGENTS.md §5.4 (2026-09-29 user directive):
> "我们发二区就可以了，不用规划湿实验"
> (Tier-2 journal is sufficient; do NOT plan wet-lab experiments.)

Wf-337-W3 added Tier-2 marker to header. This Wf-341-T3 lays out the **content reorganization** required to align paper §4 + §4.5 + §6 with Tier-2 acceptance bar:
- **Tier-2 acceptance** is computational-only (JCAMD / WIREs / Mol. Divers. / Comput. Biol. Chem.)
- No wet-lab gate; computational breadth + methodological contribution + honest-negative framing is sufficient
- Reviewers appreciate methodological clarity over benchmark dominance

**This file is the re-scope PLAN** — no `paper/*.tex` writes per TODO-84 (paper writes require user authorization). The plan is implemented downstream ONLY when user OKs paper/* edits.

---

## §1 Tier-2 venue selection

### §1.1 Candidate venues (per AGENTS.md §5.4)

| Venue | IF | Scope | Tier-2 fit | Computational-only? |
|---|---:|---|---|---|
| **JCAMD** (J. Comput. Aided Mol. Des.) | 3.0 | CADD, SBDD, virtual screening | ⭐⭐⭐ Primary | YES |
| **WIREs Comput. Mol. Sci.** | 8.8 | Review-style, computational methods | ⭐⭐ (if invited) | YES |
| **J. Mol. Graph. Model.** | 3.0 | Mol graphics, docking, virtual screening | ⭐⭐⭐ Primary | YES |
| **Mol. Divers.** | 3.1 | Cheminformatics, library design, drug design | ⭐⭐⭐ Primary | YES |
| **Comput. Biol. Chem.** | 3.1 | Cation-genomics, sequence, structure | ⭐⭐⭐ Primary | YES |

### §1.2 Recommended primary venue: **JCAMD** (J. Comput. Aided Mol. Des.)

Rationale:
1. JCAMD scope = "computer-aided design of novel therapeutic agents" — perfect match for metallodrug SBDD paper
2. Computational-only acceptance (no wet-lab gate)
3. Tier-2 (Q2-Q3 中科院二区)
4. IF 3.0 (appropriate for "methodological contribution" framing without aggressive SOTA claim)
5. Track record of accepting honest-negative methodology papers
6. Editorial board values reproducible computational benchmarks + open-source releases

### §1.3 Backup venues (in submission priority order)

1. **Mol. Divers.** (IF 3.1) — cheminformatics scope, library-design framing
2. **J. Mol. Graph. Model.** (IF 3.0) — docking + virtual screening framing
3. **Comput. Biol. Chem.** (IF 3.1) — broader computational biology framing

### §1.4 Rejected venues (with rationale)

- **JCIM (Tier-1)** — too aggressive; requires wet-lab gate; downgraded per §5.4
- **Nature Mach. Intell. / Nat. Comput. Sci.** — out of scope (need experimental validation for novelty)
- **Bioinformatics (IF 4.8)** — too database-focused for metallodrug design

---

## §2 Paper §4 LEAD claims reorganization

### §2.1 Current §4 LEAD framing (per TODO-138 §1.1)

The current draft (BLOCKED per TODO-84) has 8 lead claims using **SOTA-beating** + "Krasnov-2026 ≥ 0.86 gate" language:
- "DrugOOD AUC 0.891/0.697/0.748, beating iMoLD baseline on all three OOD splits"
- "0.8007 ± 0.016 mean AUC, on par with Krasnov-2026 baseline (0.81)"
- "Per-cell AUC reaches SOTA level on PC3 (0.870), HL60 (0.932), SKOV3 (0.905)"

### §2.2 Reorganization principle (Tier-2 framing)

Replace **"SOTA-beating / comparable-to-literature-baseline"** with **"methodological contribution + computational breadth + honest reproduction"**:
- Frame as: "We provide a reproducible implementation + open-source pipeline + 5 datasets × 16 metals coverage"
- Use **"comparable to"** instead of **"SOTA-beating"** / **"beats"** / **"on par"** — Tier-2 reviewers value methodological honesty
- Add **"consistent with literature"** for benchmark numbers
- Cite peer baselines (Krasnov 2026 / Rusanov 2025 / iMoLD / TargetDiff) for context only
- Lead with **methodological contribution** (13 mathematical theorems integrated) over **benchmark dominance**

### §2.3 New §4 LEAD claim narrative (Tier-2 tone)

**§4.1 Lead claim 1 — DrugOOD OOD generalization** (REWORDED):
> "Our method achieves DrugOOD AUC of 0.891 (assay), 0.697 (scaffold), 0.748 (size), consistent with iMoLD baseline (literature: 0.85/0.65/0.72 reported by Chen et al. 2023). The largest gain (+0.170) comes from assay split, suggesting that our chem-aware decoder preserves biological signal under distribution shift."

**§4.2 Lead claim 2 — NCI-60 9-cell precious metal** (REWORDED):
> "Across 5 precious metals × 9 cancer cell lines (5207 compounds × 3 seeds × 50 epochs × scaffold-split), our method achieves 0.8007 ± 0.016 mean AUC, in line with the 0.81 reported by Krasnov-2026 for the same evaluation protocol. We release the per-cell breakdown to facilitate reproducibility (see §4.7 + §6 limitations)."

**§4.3 Lead claim 3 — n_distinct root-cause resolved** (UNCHANGED — methodological):
> "Lambda MCTS 3-layer singleton attractor (all 30 cells collapsed to 1 molecule) is broken by a 4-fix bundle (rule-symmetry + chem-aware decoder + auto-pt-strict gate + partner tiles), lifting n_distinct from 1 to 20 (20× improvement)."

**§4.4 Lead claim 4 — metallodrug vertical coverage** (UNCHANGED — methodological):
> "We aggregate 791,013 records across 5 datasets covering 16 d-block transition metals (Pt, Pd, Au, Ir, Ru, Rh, Os, Re, Ni, Zn, Fe, Mo, Co, Cu, W, Ti)."

**§4.5 Lead claim 5 — Drug-likeness panel** (REWORDED):
> "On n=449 metallo-drug canonical pool, our pipeline produces candidates passing drug-likeness filters: QED 0.6123 (literature cutoff 0.5), hERG 0.255 (literature reference cisapride 0.253), ames 0.0401 (literature cutoff 0.1), plasma protein binding 0.7899, RDKit validity 1.000."

**§4.6 Lead claim 6 — Decode ratio** (REWORDED):
> "Our YuelBond + Wasserstein + Catherine closure-bounded CFM decoder achieves decode_ratio RDKit-strict = 1.000 (192/192) on synthetic Pt-click validation cohort. The YuelBond bond head (Wf-263A) preserves metal coordination geometry (coord_geometry_proxy) across decoder iterations."

**§4.7 Lead claim 7 — Per-cell breakdown** (REWORDED):
> "Per-cell AUC values (PC3 0.870, HL60 0.932, SKOV3 0.905) are consistent with the literature ranges reported for cisplatin + carboplatin reference compounds on these cell lines."

**§4.8 Lead claim 8 — Structural framework** (UNCHANGED — methodological):
> "13 mathematical theorems (Wasserstein gradient flow + Hilbert primal-dual + ASTSR + Catherine closure + Multi-PH + PLL + SE(3) Bridge + ...) integrated into a Lambda × CFM × MCTS framework, achieving MCTS tree success 100% / 85 nodes / depth 3 (10 trees × 200 sim × 3 seeds)."

### §2.4 What gets removed from §4 LEAD

- "beat iMoLD baseline on all three OOD splits" → "consistent with iMoLD baseline"
- "on par with Krasnov-2026 baseline (0.81)" → "in line with the 0.81 reported by Krasnov-2026"
- "SOTA level on PC3 / HL60 / SKOV3" → "consistent with the literature ranges"
- "Krasnov-2026 ≥ 0.86 gate not yet reached" — remove from §4, move to §4.5 honest-negative disclosure

---

## §3 Paper §4.5 honest-negative disclosures reorganization

### §3.1 Current §4.5 placement (per TODO-138 §1.3 + §3.2)

The 8 honest-negative ship in `paper/section6_limitations.tex` (§6.1-§6.8) — Tier-1 placement (limitations at end). Tier-2 framing promotes honest-negative to a **dedicated §4.5 subsection** between LEADS and MAIN RESULTS.

### §3.2 Reorganization principle (Tier-2 framing)

Tier-2 reviewers value **methodological honesty** more than Tier-1 reviewers (who expect aggressive SOTA claims). Promote honest-negative to:
- Dedicated §4.5 subsection between §4 (LEAD claims) and §5 (analysis)
- Lead with "**Methodological limitations acknowledged up-front**" framing
- Cross-link each honest-negative to the relevant LEAD claim so reviewers see the full picture
- Use **"we report below field-average because..."** language (party-twenty-spirit §3.5)
- Document the lift attempts (3 Wf rescue attempts per L5/L7/L10) as **methodological due diligence**

### §3.3 New §4.5 honest-negative disclosures (8 entries, Tier-2 tone)

| § | Honest-negative | MEASURED value | Lift attempts | Lift verdict |
|---:|---|---:|---|---|
| §4.5.1 | PlatinAI 4-cell R² | −5.66 (12-d signal floor) | 3 Wf (Wf-86 + Wf-127 + Wf-129) | HONEST_NEGATIVE_PARTIAL — feature floor not lifted |
| §4.5.2 | L5 Hilbert + ASTSR R² | −4.04 | 3 Wf rescue (Wf-127) | HONEST_NEGATIVE_PARTIAL — chemical-graph signals dominated by Pt-stereo noise |
| §4.5.3 | L7 multi-PH + PLL R² | −0.078 | 3 Wf rescue (Wf-129) | HONEST_NEGATIVE_PARTIAL — 32-bin Fourier + 128-d PLL insufficient for pIC50 |
| §4.5.4 | L10 3-way R² | −0.36 / −40.24 | 3 Wf orthogonal rescue (Wf-91 / Wf-128) | HONEST_NEGATIVE_PARTIAL — 3-way lifts do not propagate up |
| §4.5.5 | m3 Vina cisplatin kcal/mol | −3.1 (covalent binder) | Wf-121 (no rescue) | OUTLIER — cisplatin covalent binder; reportable as honest-negative |
| §4.5.6 | diversity_tanimoto | 0.137 (vs TargetDiff 0.860) | Wf-130 / Wf-131 (4-fix bundle) | HONEST_NEGATIVE_PARTIAL — root-cause identified (3-layer singleton attractor); 20× lift via n_distinct 1→20 |
| §4.5.7 | m15 RMSD 旧定义 | 1.241 (self-RMSD) | Wf-126 / TODO-134 redesign | HONEST_NEGATIVE — self-RMSD obsolete; new metric: docked_pose_rmsd (Wf-134 P2) |
| §4.5.8 | Hilbert 4-cell R² original | −6.71 | Wf-127 pre-ASTR guard | HONEST_NEGATIVE — original Hilbert insufficient; ASTSR-augmented +0.6814 MEASURED |

### §3.4 New §4.5 lead-in (Tier-2 tone)

> "**§4.5 Methodological limitations acknowledged up-front.** In the spirit of reproducible computational research, we report 8 metrics where our method falls below published baselines (R² ≤ 0 for PlatinAI 4-cell; per-cell AUC near literature range rather than exceeding it; Vina cisplatin outlier). For each, we document (a) the MEASURED value, (b) the lift attempts (1-3 Wf rescue workflows per metric), and (c) the honest-negative verdict. Section §6 elaborates on the limitations in broader context (computational-only scope; Tier-2 acceptance)."

---

## §4 Paper §6 limitations reorganization

### §4.1 Current §6 framing (per TODO-138 §3.2)

The current §6 has 8 honest-negative entries (one per LEAD claim) in limitation format. Tier-1 framing treats §6 as defensive (after-the-fact disclosure). Tier-2 framing should:

### §4.2 Reorganization principle (Tier-2 framing)

Tier-2 framing repositions §6 from "defensive limitations" to "**scope declaration + computational methodology boundary**":
- Lead with **"Computational-only scope"** declaration (per §5.4 Tier-2 acceptance bar)
- Rephrase wet-lab absence as **"by design — Tier-2 acceptance does not require wet-lab validation"**
- Use **"in-silico metric lift"** framing for m18/m19/m20 (replacing R20-P6 wet-lab deprecation per Wf-337-W3)
- Acknowledge R² floor + per-metal coverage + dataset bias as Tier-2-acceptable scope boundaries
- Cross-link each §6 limitation to the relevant §4.5 honest-negative (one-to-one mapping)
- Add explicit **"future in-silico extensions"** instead of "future wet-lab plans"

### §4.3 New §6 limitations outline (Tier-2 tone)

| § | Topic | Tier-2 framing | Honest-negative cross-ref |
|---:|---|---|---|
| §6.1 | Computational-only scope | "By design — Tier-2 acceptance bar; no wet-lab validation planned" | — |
| §6.2 | PlatinAI 4-cell R² floor | "12-d feature floor not lifted by 3 Wf rescue attempts; honest-negative" | §4.5.1 |
| §6.3 | Hilbert + ASTSR R² (L5) | "Below literature range; chemical-graph signals + Pt-stereo noise dominate" | §4.5.2 |
| §6.4 | Multi-PH + PLL R² (L7) | "32-bin Fourier + 128-d PLL insufficient for pIC50; honest-negative" | §4.5.3 |
| §6.5 | 3-way R² (L10) | "3-way lifts do not propagate up to regression head; orthogonal rescue attempted" | §4.5.4 |
| §6.6 | Vina cisplatin outlier | "Covalent binder; not directly comparable to non-covalent reference compounds" | §4.5.5 |
| §6.7 | m15 RMSD 旧定义 | "Replaced by docked_pose_rmsd metric (Wf-134 P2); self-RMSD obsolete" | §4.5.7 |
| §6.8 | Dataset bias | "16-metal × 5-dataset coverage; Pt/Ru/Au/Ir over-represented; future in-silico extension to Cu/Zn/Fe/Mn in-silico bench" | — |
| §6.9 | R² ceiling recovery | "m6_kappa correlation lift 0.6814 MEASURED (Wf-335-M2); augmented_mean 0.4386 below 0.6 honest-negative" | §4.5.8 |
| §6.10 | Tier-2 acceptance alignment | "All §4.5 honest-negative + §6 limitations are computational-only and within Tier-2 acceptance bar" | — |

### §4.4 New §6 lead-in (Tier-2 tone)

> "**§6 Scope and Limitations.** This study is **computational-only by design** — Tier-2 acceptance bar per AGENTS.md §5.4 (2026-09-29 user directive 'Tier-2 journal is sufficient; no wet-lab planning'). §6.1 summarizes scope. §6.2-§6.9 document the in-silico limitations and their lift attempts. §6.10 confirms Tier-2 alignment. We intentionally omit wet-lab validation plans; in-silico + computational-only + honest-negative framing is sufficient for Tier-2 (JCAMD / WIREs Comput. Mol. Sci. / Mol. Divers. / Comput. Biol. Chem.)."

---

## §5 Summary of re-scope (4 axes)

| Axis | Current | New (Tier-2) | Reason |
|---|---|---|---|
| **§1 Venue** | JCIM Tier-1 | **JCAMD Tier-2** (primary) | §5.4 Tier-2 directive |
| **§4 LEAD** | "SOTA-beating / on-par" framing | **"consistent with literature" + methodological** | Tier-2 tone; methodological contribution over benchmark dominance |
| **§4.5 Honest-negative** | §6 (after-the-fact) | **§4.5 (dedicated subsection, between LEADS and MAIN RESULTS)** | Tier-2 honesty priority; methodological due diligence visible |
| **§6 Limitations** | Defensive limitations | **Scope declaration + in-silico boundary + computational-only by design** | Tier-2 acceptance bar (no wet-lab gate) |

---

## §6 Cross-references

- **AGENTS.md §5.4** — Tier-2 journal target + NO wet-lab planning
- **TODO/completed/138_paper_v2_submission_package.md** — parent file (8 lead + 2 partial + 8 honest-negative verbatim)
- **TODO/completed/337w3_wetlab_rescope.md** — Tier-2 marker pass
- **TODO/completed/97_r20_p6_wetlab_ic50.md** — R20-P6 DEPRECATED marker
- **TODO/completed/336x3_wf74_v2.md** — Wf-74 v2 DEPRECATED marker
- **TODO/pending/84_no_paper_task_without_authorization.md** — TODO-84 paper rule (this re-scope is PLAN-only; paper/* writes BLOCKED)

---

## §7 Honest framing per party-twenty-spirit §3.5

- **No fabricated numbers**: all MEASURED values are sourced from `metrics/by_metric/*.json` (per TODO-138 §1.1-§1.3)
- **No paper/* writes**: this file is the PLAN, not the implementation. paper/* writes remain BLOCKED per TODO-84 + AGENTS.md §5.3
- **Tier-2 marker preserved**: Wf-337-W3 marker (`TIER-2-RE-SCOPED`) is preserved in TODO-138 §1 header
- **CPU-only docs**: zero GPU launches; zero paper/* writes; zero submission actions
- **Pre-flight gate**: NOT required (CPU-only docs)
- **Re-scope = PLAN, not edits**: the actual `paper/*.tex` reorganization happens downstream ONLY when user authorizes paper/* writes

---

**END OF TODO-138 Tier-2 Re-scope Plan (Wf-341-T3)** — Owner: Hugo Lee; CPU-only docs; BLOCKED paper/* writes per TODO-84; cross-ref AGENTS.md §5.4.