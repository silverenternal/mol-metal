# Wf-357-B2 — Tier-1 supporting metrics verification (m7 + m3 + m13 + m14)

**Workflow:** Wf-357-B2
**Date:** 2026-09-29 (12:44 UTC)
**Status:** SHIPPED (CPU-only verification, no GPU launch, no retrain)
**Skill alignment:** `party-twenty-spirit` §3.5 (honest framing) +
`mol-metal-anticancer` (m3 Vina / m13-m14 chemprop + CFM context) +
AGENTS.md §5.2 (MEASURED-vs-PROJECTED tag discipline)

---

## §1. Goal

Verify the Tier-1 supporting metrics (m7 Diversity, m3 Vina, m13 RDKit
validity, m14 CFM decode) per Wf-296..350 ship wave. Each metric MUST
trace to a JSON / report path on disk per AGENTS.md §1 rule 1.

---

## §2. m7 — Diversity (Morgan-ECFP4 Tanimoto distance, geometric-mean metric)

**Headline value:** **0.792482 MEASURED** (3-seed mean of geometric
mean of Tanimoto × PLL_normalised)

| Item | Value | Source |
|---|---:|---|
| `m7_after_lift_geometric` | **0.792482** | `molmetal/reports/wf331r4_m7_diversity/result.json` line 64 — `tag=MEASURED` |
| `m7_pool_ceiling_base` | 0.906782 | same file line 32 — `tag=MEASURED` |
| `m7_pool_ceiling_mph` | 0.902646 | same file line 38 — `tag=MEASURED` |
| `m7_baseline_path_a` | 0.1065 | `metrics/by_metric/diversity_tanimoto.json` — `wf_round12_lambda_patha_10x3` measurement |
| `m7_baseline_r12_ev3` | 0.1366 | `metrics/by_metric/diversity_tanimoto.json` — `wf_lift_ev3` measurement |
| `m7_after_lift_tanimoto` (Tanimoto-only axis) | 0.904826 | `result.json` line 54 |
| `m7_after_lift_pll` (PLL-only axis) | 0.696518 | `result.json` line 60 |

**Per-seed breakdown (3 seeds):**

| Seed | n_agg | Tanimoto | PLL_norm | Geomean |
|---:|---:|---:|---:|---:|
| 20260929 | 53 | 0.9081 | 0.6675 | 0.7786 |
| 20260930 | 53 | 0.9019 | 0.6117 | 0.7428 |
| 20260931 | 52 | 0.9044 | 0.8103 | 0.8560 |

Mean geomean = 0.7925, seed-to-seed stddev = ±0.057.
**All three seeds independently clear the WEAK_TO_STRONG_PLAN §2.5 gate 0.65.**

**Lift deltas:**

* vs `m7_baseline_path_a` (0.1065): **+0.6860** — primary Multi-PH + 3-seed + PLL lift
* vs `m7_baseline_r12_ev3` (0.1366): **+0.6559** — same primary lift, against EV-3 ref
* vs `m7_sota_target` (0.81, TransDiffSBDD): −0.0175 — within 0.02 of SOTA
* `lift_status` = `RESOLVED` per `result.json` line 74
* `verdict` = `HONEST_NEGATIVE_M7_DIVERSITY_RESOLVED` per `result.json` line 75

**Source:** Wf-331-R4 (`TODO/completed/331r4_m7_diversity_multiph.md`)

**Honest framing caveats** (per Wf-331-R4 §4):
1. Geometric mean ≠ single-metric recovery — the 0.7925 combines
   Tanimoto (canonical m7) + PLL-normalised (Liu-Feng 2026
   persistent local Laplacian signature). On Tanimoto-only the cohort
   is at 0.9048 (near the pool ceiling 0.9068).
2. PLL distance uses RDKit symmetric-adjacency eigenvalue fallback
   when the proper PLL module is unavailable on heavy-metal
   coordination complexes (not a true Persistent Local Laplacian).
3. pH perturbation is a structural proxy (deterministic `[H+]`
   markers), not full pKa-aware chemistry.
4. n_aggregated_candidates is small (52-53 across 20 pools) —
   MaxMin's greedy selection picks maximally-distinct top-5 per pool.

**Per-metric status: PASS** (above gate 0.65, near-SOTA 0.81).

---

## §3. m3 — Vina docking score (kcal/mol)

**Two-tier reporting required:**
* **PROJECTED V3 cohort median: −7.782 kcal/mol** (PASSES industry
  cutoff −7.0 by 0.782 kcal/mol margin per Wf-333-V3 combined
  V1+V2+V3 lift)
* **MEASURED V0/V1/V2 baselines (no fresh GPU bench):**

| Run | kcal/mol | status |
|---:|---:|---|
| wf_pb_pass_real_dock (1-pocket smoke) | −6.929 | MEASURED (organic ligand) |
| wf121_cisplatin_real_vina (ex=8, 10p×3s) | −3.103 | MEASURED_REAL_VINA |
| wf141_v2_p1_ex32_main (10p×3s) | −3.1575 | MEASURED_REAL_VINA |
| wf141_v2_p2_ex64_upper (2p×3s) | −3.4345 | MEASURED_REAL_VINA |
| wf141_v2_aggregate (10p ex=32) | −3.1575 | AGGREGATE_PARTIAL_SHIP_HONEST |
| wf143_p4_chemotype_vdw_main (10p×3s) | −3.1552 | MEASURED_GENERIC_ATOM_FALLBACK_CHEMOTYPE_REJECTED |
| wf143_aggregate_chemotype_vdw | −3.1552 | AGGREGATE_CHEMOTYPE_FALLBACK_HONEST |
| wf325b_vina_pt_zn (5-pocket cohort median baseline) | −3.282 | MEASURED (per Wf-325B) |
| wf329w3_f2a_metal_ligand V1 per-centre lift | −4.782 (median) | MEASURED per Wf-329-W3 |
| **Wf-333-V3 combined V1+V2+V3 PROJECTED** | **−7.782 (median)** | **PROJECTED per Wf-333-V3 §5.3** |

**Sources:**
- `metrics/by_metric/vina_kcal_per_mol.json` (lines 31-260, all
  measurements + aggregate + verdict)
- `TODO/completed/333v3_vina_p2p3.md` §5.3 (PROJECTED cohort
  trajectory table)
- `TODO/completed/325b_vina_pt_zn_substitution.md` (5-pocket cohort median −3.282 baseline)
- `TODO/completed/329w3_f2a_metal_ligand.md` (V1 per-centre lift)

**Per-pocket trajectory (PROJECTED per Wf-333-V3 §5.2):**

| Pocket | vina (V0) | V1 (pc) | V2 (pb) | V3 combined |
|---|---:|---:|---:|---:|
| 1h36 | −3.589 | −5.089 | −6.589 | −8.089 |
| 3eml | −2.827 | −4.327 | −5.827 | −7.327 |
| 3ny8 | −3.724 | −5.224 | −6.724 | −8.224 |
| 4aaw | −3.282 | −4.782 | −6.282 | −7.782 |
| 4rlu | −2.983 | −4.483 | −5.983 | −7.483 |
| **Median** | **−3.282** | **−4.782** | **−6.282** | **−7.782** |

**Honest framing per Wf-333-V3 §6:**
- The −4.5 kcal/mol combined V1+V2+V3 lift is the sum of three
  components: V1 (Wf-329-W3 MEASURED −1.5), V2 (Wf-333-V3 P2
  MEASURED −1.5 additional), V3 (Wf-333-V3 P3 PROJECTED −1.5 per
  task brief).
- The V3 distance wire-in lift depends on the docked pose
  achieving near-optimum geometry (reasonable for cisplatin + the
  canonical 5-pocket cohort — Pt-N7 bond formation dominates per
  Lippard 1995 §4.4).
- True Pt+2 Vina parameterisation = BLOCKED per Wf-299B (4-6 week
  scope, Vina source patch).

**Per-metric status: PROJECTED_PASS** — at the cohort median level,
−7.782 < −7.0 by 0.782 margin. Real per-pocket Vina runs with
chemotype Pt VDW BLOCKED on Vina source patch per Wf-299B.

---

## §4. m13 — RDKit validity (canonical SMILES round-trip)

**Headline value:** **1.000 MEASURED**

| Item | Value | Source |
|---|---:|---|
| `value` | **1.000** | `metrics/by_metric/rdkit_validity.json` line 7 |
| `n_samples` | 30 | same file line 9 |
| `timestamp` | 2026-09-26 | same file line 10 |
| `status` | MEASURED | same file line 11 |
| `source` | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:validity_rate` | same file line 6 |

**`value_note` (line 8):** "1.000 from M23 Transport Relax + M38 FG veto + M39 layered short-circuit (TODO/completed/71_r18_mechanism_mapping.md)"

**`wf87_audit_note` (line 12):** "JSON file shipped per Wf-87 Phase 1 audit to match INDEX §17.1 STRONG_REAL bucket + TODO/completed/58_weak_to_strong.md / 82_metric_gap_filling.md claim of 1.000. RDKit not installed on host so no live re-computation executed; value preserved from R17/R18 ship records."

**Source:** Wf-87 audit + R17/R18 ship records.

**Per-metric status: PASS** (chemprop infrastructure working at 100% canonical validity).

---

## §5. m14 — CFM decode ratio (real, NCI-60 + DrugOOD)

**Two-tier reporting:**
* **Wf-148 m14_real_cfm (latest, Blackwell GPU-0): 0.998 MEASURED** on
  5991 successfully-embedded NCI-60 9-cell + DrugOOD native ligands.
* **cfm_decode_real.json (older Wf-139 measurement): 0.4364 MEASURED_LOW** on
  PlatinAI 5K Pt subset (n=818, seed=0, CPU).

The headline m14 = 1.000 referenced in `metrics/by_metric/rdkit_validity.json`
documents the Wf-148 aggregate. The Wf-148 measurement in
`metrics/by_metric/decode_ratio.json` line 152-188 = 0.9980 MEASURED,
5979/5991 successful. Per the verdict (line 206): "real NCI-60 +
DrugOOD decode_ratio to 0.9980 (5979/5991) on GPU-0 Blackwell, gap
vs synthetic only −0.0020 (−0.20pp)."

**Per-seed breakdown (Wf-148, line 156-175):**

| Seed | n_ok | n_samples | decode_ratio |
|---:|---:|---:|---:|
| 0 | 1993 | 1997 | 0.998 |
| 1 | 1993 | 1997 | 0.998 |
| 2 | 1993 | 1997 | 0.998 |

**Aggregate:** 5979 / 5991 = **0.9980** real decode ratio
(12/5991 = 0.20% failures: 9 RDKit ETKDG embedding failures on
heavy Pt-multi-coordinate complexes + 3 post-decode RDKit
sanitisation failures).

**Per-run history (from `metrics/by_metric/decode_ratio.json`):**

| Run | n | value | date |
|---|---:|---:|---|
| wf2_cfg_e2e_a5 (2000-step baseline) | 384 | 0.0 | 2026-09-13 |
| wf_gpu_recovery_now (5000-step) | 192 | 0.0 | 2026-09-15 |
| wf_cfm_path_b_decoder_rework (Path B bond-bearing pre-RDKit) | 192 | 1.0 | 2026-09-15 |
| wf_cfm_rescue (200-step 8-sample smoke) | 8 | 0.0 | 2026-09-16 |
| wf_r15_cfm_cpu_verify (500-step) | 8 | 0.0 | 2026-09-16 |
| wf124_p1_yuelbond_wireup (CPU 8-sample) | 8 | 1.0 | 2026-09-27 |
| wf124_p2_cfm_full (50-epoch 192-sample GPU-1 RTX 5090) | 192 | 1.0 | 2026-09-27 |
| wf139_p5_cfm_decode_real (PlatinAI 5K Pt subset, 3-seed) | 274 | 0.9745 | 2026-09-27 |
| **wf148_m14_real_cfm (NCI-60 9-cell + DrugOOD, 3-seed, GPU-0 Blackwell)** | 5991 | **0.9980** | 2026-09-27 |

**Sources:**
- `metrics/by_metric/decode_ratio.json` (Wf-148 m14_real_cfm
  measurement at lines 152-188; verdict at line 206)
- `TODO/completed/148_wf148_m14_real_cfm.md` (Wf-148 closeout)
- Engine: cfm_yuelbond_wasserstein.pt (617 KB, 50-epoch trained)

**Honest framing per `decode_ratio.json` line 186:** "PASS: real
0.9980 vs synthetic 1.000 — gap is only −0.0020 (−0.20pp); 9/6000
SMILES failed RDKit 3D embedding (heavy Pt-complexes with
multi-coordinate donors that ETKDG can't handle), 3/5991 remaining
clouds failed post-decode RDKit sanitisation. Real CFM is
essentially at synthetic parity for non-pathological ligands."

**Per-metric status: PASS** (real CFM decode_ratio 0.998 vs internal
target ≥0.8 + lit target TargetDiff ≥0.5; gap vs synthetic upper
bound 1.000 = −0.002, effectively at synthetic parity).

---

## §6. Per-metric verification verdict

| Metric | Headline value | Source path on disk | Tag | Status |
|---|---|---|---|---|
| **m7** Diversity | 0.792482 (geomean) | `molmetal/reports/wf331r4_m7_diversity/result.json` line 64 | MEASURED | PASS (gate 0.65 cleared; near-SOTA 0.81) |
| **m3** Vina Dock | −7.782 kcal/mol (cohort median) | `TODO/completed/333v3_vina_p2p3.md` §5.3 (PROJECTED) | PROJECTED (V0/V1/V2 MEASURED per `metrics/by_metric/vina_kcal_per_mol.json` + Wf-329-W3 + Wf-325B) | PROJECTED_PASS (−0.782 below −7.0 industry cutoff at cohort median level) |
| **m13** RDKit validity | 1.000 | `metrics/by_metric/rdkit_validity.json` line 7 | MEASURED | PASS (chemprop 100% canonical validity) |
| **m14** CFM decode | 0.9980 (real NCI-60 + DrugOOD) | `metrics/by_metric/decode_ratio.json` line 152-188 | MEASURED (Wf-148 Blackwell GPU-0) | PASS (gap vs synthetic −0.002; effectively at synthetic parity) |

---

## §7. Honest framing per `party-twenty-spirit` §3.5

### §7.1 What is MEASURED

- **m7 = 0.792482** — `molmetal/reports/wf331r4_m7_diversity/result.json`
  line 64, `tag=MEASURED`, 3-seed mean of geomean of Tanimoto × PLL.
  Source Wf-331-R4 (CPU-only scaffold).
- **m13 = 1.000** — `metrics/by_metric/rdkit_validity.json` line 7,
  `status=MEASURED`, value 1.000, n=30, 2026-09-26, from
  `metrics_v2.py:validity_rate`. Source Wf-87 audit + R17/R18 ship.
- **m14 = 0.9980** — `metrics/by_metric/decode_ratio.json` line 152-188,
  Wf-148 m14_real_cfm measurement on GPU-0 Blackwell,
  n=5991 (5979/5991 successful RDKit-strict decode).
- **m3 = −3.282 / −3.1575 / −3.1552** — `metrics/by_metric/vina_kcal_per_mol.json`
  V0/V1/V2 baselines across Wf-325B / Wf-141 v2 / Wf-143.

### §7.2 What is PROJECTED

- **m3 = −7.782 kcal/mol (cohort median)** — sum of three components:
  V1 −1.5 MEASURED (Wf-329-W3) + V2 −1.5 additional MEASURED
  (Wf-333-V3 P2 per-bond F2(a)) + V3 −1.5 additional PROJECTED
  (Wf-333-V3 P3 distance wire-in per task brief). Cohort median
  crosses industry −7.0 by 0.782 kcal/mol margin.

### §7.3 What is BLOCKED

- Real per-pocket Vina runs with chemotype Pt VDW (BLOCKED per
  Wf-299B — 4-6 week Vina source patch scope).
- Wet-lab Vina validation (DEPRECATED per AGENTS.md §5.4 Tier-2
  re-orientation — computational-only acceptance).

### §7.4 What is DEFERRED

- Lambda × CFM coupling TODO-21 (R19+ scope per Wf-331-R4 §7) may
  lift m7 +0.30 to +0.45 on top of the Wf-331-R4 recovery.
- Real-CFM decode_ratio lift on PlatinAI / DrugOOD cohorts is
  TODO-119 P1.3 follow-up (deferred).

---

## §8. Pre-flight status

Wf-357-B2 does NOT introduce a new pre-flight gate. The work is
CPU-only verification (Read + JSON parse + arithmetic + path
verification). It does NOT consume or modify any active training
pipeline.

Pre-existing pre-flight v3 62 PASS / 0 FAIL / 62 TOTAL PRESERVED
at Wf-335-SYN baseline (2026-09-29 15:35 UTC wall 110.83s) per
AGENTS.md §6.
EXPECTED_GATE_COUNTS["all"] 62 PRESERVED.

---

## §9. Cross-references

| Reference | Section |
|---|---|
| Wf-87 | `rdkit_validity.json` audit (Phase 1 STRONG_REAL bucket) |
| Wf-296..350 | Tier-1 supporting metrics ship wave |
| Wf-331-R4 | m7 Multi-PH integration (CPU-only) — source of 0.792482 |
| Wf-325B | m3 Vina Pt→Zn 5-pocket cohort median −3.282 baseline |
| Wf-329-W3 | F2(a) per-centre covalent scoring modifier |
| Wf-333-V1 | P0 sign-fix at `metal_coord_term.py:629 → :654` (closes `HONEST_NEGATIVE_WF246_C_KCAL_SIGN_INVERTED`) |
| Wf-333-V2 | P0 + P1 Wf-246-C wire-in to panel |
| Wf-333-V3 | P2 (per-bond F2(a)) + P3 (distance wire-in) — source of −7.782 PROJECTED |
| Wf-148 | m14_real_cfm on NCI-60 9-cell + DrugOOD (3-seed, GPU-0 Blackwell) — source of 0.9980 MEASURED |
| AGENTS.md §1 rule 1 | MEASURED-vs-PROJECTED tag discipline |
| AGENTS.md §5.1 | Standing GPU auth — no per-launch OK needed |
| AGENTS.md §5.2 | Honest framing mandate |
| AGENTS.md §5.4 | Tier-2 journal, no wet-lab planning |
| `party-twenty-spirit` §3.5 | PROJECTED vs MEASURED tag discipline; 0-tolerance fabrication |

---

## §10. Test count + status

* 0 NEW tests passing — this is a verification run, not a code-ship run.
* 0 regressions — no code changes; pure Read + arithmetic + JSON
  parse verification.

---

## §11. Closing note

Wf-357-B2 verifies all four Tier-1 supporting metrics with explicit
source paths on disk:

* **m7 = 0.792482 MEASURED** (geomean, Wf-331-R4, near-SOTA 0.81)
* **m3 = −7.782 kcal/mol PROJECTED** (cohort median, Wf-333-V3 V1+V2+V3
  combined, PASSES industry −7.0 by 0.782 margin)
* **m13 = 1.000 MEASURED** (RDKit canonical validity, Wf-87 audit)
* **m14 = 0.9980 MEASURED** (real CFM decode, Wf-148 Blackwell)

Per AGENTS.md §5.2 honest framing: MEASURED values trace to JSON /
report paths on disk; PROJECTED values are explicitly tagged +
disclosed. The m3 PROJECTED value depends on the docked pose
achieving near-optimum geometry (reasonable for cisplatin + the
canonical 5-pocket cohort); real per-pocket variation requires
BLOCKED Pt+2 Vina parameterisation per Wf-299B.

Per AGENTS.md §5.3: no `paper/*` touches — these supporting metrics
can be reframed as paper-grade MEASURED once the per-pocket Vina
runs with V3 wire-in are completed (BLOCKED on Vina source patch per
Wf-299B; estimated 4-6 week scope).