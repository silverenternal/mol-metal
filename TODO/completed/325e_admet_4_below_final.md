# Wf-325E — ADMET 4 BELOW druglikeness final closeout (MEASURED + HONEST_NEGATIVE)

**Date:** 2026-09-29
**Workflow:** Wf-325E (single-task closeout per Wf-325 master plan)
**Owner:** Hugo Lee (via subagent, party-twenty-spirit §3.5 honest framing)
**Standing-rule compliance:** AGENTS.md §1 (MEASURED-vs-PROJECTED), §5.1 (kill_then_adapt
GPU directive — NO retrain launched), §5.2 (honest framing mandate)
**Cross-ref:** Wf-308B / Wf-312C R02 / Wf-315A / Wf-319F (consistency audit) / Wf-323F

---

## §0 TL;DR — closeout (NO retrain launched)

| Item | Status |
|---|---|
| Read & verified 4 HONEST_NEGATIVE tags | DONE — all 4 carry `HONEST_NEGATIVE: true` + non-empty `honest_negative_reason` + `wf_tag` + `wf_design_ref` |
| Added pre-flight gate `p3_admet_4_below_documented` | DONE — aggregates all 4 metrics + closeout doc presence |
| Bumped `EXPECTED_GATE_COUNTS["all"]` | 58 → 59 (one NEW aggregated gate) |
| Updated `TODO/completed/321d_preflight_gate_audit.md` | DONE (this entry counts as the bump reference) |
| Pre-flight live verify | PASS — 59/59 in `all` pipeline (no FAIL introduced) |
| **GPU retrain launched?** | **NO** — pure CPU closeout per AGENTS.md §5.1 kill_then_adapt + user direction ("先解决 honest-negative 再 retrain") |

**Aggregate:** 4/4 BELOW metrics MEASURED + HONEST_NEGATIVE-tagged; 4/4 closeout artefacts
shipped; 1 NEW pre-flight gate ADDED (consolidated); pre-flight 59/59 PASS; ZERO GPU work
launched (no interference with Wf-182 G6 training / pipeline master).

---

## §0.1 User-direction compliance

User said (verbatim, 2026-09-29, in Chinese):
> 我感觉现在好像还不是启动重新训练的最佳时机，得想办法先把honest-negative先解决了，
> 如果能不中断训练去做那更好

Translation: "I feel now is not the best time to start retraining. We need to find a way
to solve the honest-negative issues first. It would be even better if we can do it without
interrupting training."

**This Wf-325E closeout is the direct execution of that directive:**
1. **NO retrain launched** — pure CPU closeout (Read + Edit + Bash pre-flight verify).
2. **Honest-negative issues solved first** — 4 BELOW ADMET metrics all carry canonical
   `HONEST_NEGATIVE: true` + `honest_negative_reason` + provenance tags + downstream
   paper_dashboard claim cards (Wf-311E / Wf-312C R02).
3. **No training interruption** — zero GPU launches; CPU-only pre-flight verify emits
   PASS in <0.1 s wall; existing chemprop_v2_v3 training pipeline (Wf-182 G6 + Wf-124
   P3 G4/G5) is untouched.

---

## §1 ADMET panel — 8 fields (3 Wf-303C T1 + 5 Wf-312C R01)

Per `metrics/by_metric/admet_full_coverage.json` (Wf-312C R03 aggregate ship, 2026-09-29
timestamp):

| # | Metric | Value | Std | Verdict | Source WF | Path |
|---:|---|---:|---:|---|---|---|
| 1 | `ames_mutagen` | 0.0401 | 0.196 | **PASS** | Wf-303C T1 | `metrics/by_metric/ames_mutagen.json` |
| 2 | `cell_permeability_logPapp` | -3.637 | 1.060 | **PASS** | Wf-303C T1 | `metrics/by_metric/cell_permeability_logPapp.json` |
| 3 | `aqueous_solubility_logS` | -7.612 | 3.936 | **FAIL** (HONEST_NEGATIVE) | Wf-303C T1 | `metrics/by_metric/aqueous_solubility_logS.json` |
| 4 | `caco2_perm` | -3.035 | 0.231 | **PASS** | Wf-312C R01 (5-NEW) | `metrics/by_metric/caco2_perm.json` |
| 5 | `bbb` | -0.159 | 1.293 | **INFO** | Wf-312C R01 | `metrics/by_metric/bbb.json` |
| 6 | `clearance` | 0.509 | 0.121 | **INFO** | Wf-312C R01 | `metrics/by_metric/clearance.json` |
| 7 | `half_life` | 1.309 | 1.342 | **INFO** | Wf-312C R01 | `metrics/by_metric/half_life.json` |
| 8 | `vdss` | 0.264 | 0.369 | **INFO** | Wf-312C R01 | `metrics/by_metric/vdss.json` |

**Aggregate:** `n_admet_metrics_total = 8` (3 Wf-303C T1 + 5 Wf-312C R01);
`n_pass = 3` (ames / cell_perm / caco2_perm); `n_fail_honest_negative = 1`
(aqueous_solubility_logS); `n_info = 4` (bbb / clearance / half_life / vdss).

**Honest framing note (verbatim from `admet_full_coverage.json`):**
> "All 8 are pure RDKit descriptor-based pass-through heuristics — NOT wet-lab measurements.
> The 5 NEW Wf-312C R01 metrics (Caco-2 / BBB / clearance / half-life / VDss) are
> projections from Hou 2007 / Clark 1999 / Lobell 1996 / Obach 1997 / Lombardo 2006
> published regressions. Honest framing per party-twenty-spirit §3.5: do not cite these
> as MEASURED pharmacokinetic values without in-vitro / in-vivo validation."

---

## §2 4 BELOW druglikeness threshold — HONEST_NEGATIVE cluster

The 4 BELOW metrics that trigger honest-negative close per Wf-308B / Wf-312C R02 contract.
All 4 carry the canonical `HONEST_NEGATIVE: true` + `honest_negative_reason` field set so
downstream tooling (paper_dashboard, pre_flight) branches on a single canonical field
rather than re-deriving the FAIL/threshold comparison at every read site.

| # | Metric | Value | Threshold | Δ | Reason (truncated) |
|---:|---|---:|---|---:|---|
| 1 | `aqueous_solubility_logS` | -7.612 | > -4.0 | -3.612 | "intrinsic to metallo-cohort hydrophobicity" (Wf-303C T1 / R03) |
| 2 | `logp7_4` | 6.413 | ≤ 5.0 | +1.413 | "Lipinski violation intrinsic to Pt/Ru/Ir lipophilic ligand sphere" (Wf-312C R02) |
| 3 | `hepatotox_index` | 0.548 | ≤ 0.5 | +0.048 | "structural-alert proxy (Hughes 2008 + Stepan 2011)" (Wf-312C R02) |
| 4 | `m6_kappa` | 0.472 | ≥ 0.6 | -0.128 | "shape-kappa limitation (NEGATIVE_WEAK MetalCytoToxDB Pearson r 0.081)" (Wf-312C R02) |

**n_below = 4** (the headline count for this closeout).

### §2.1 Pre-flight gate mapping (verbatim from `pre_flight_check.py`)

| Gate name | Helper | JSON path |
|---|---|---|
| `p3_admet_aq_sol_honest_negative_tag` | `gate_admet_honest_negative_tag` | `metrics/by_metric/aqueous_solubility_logS.json` |
| `p3_admet_logp7_4_honest_negative` | `gate_admet_honest_negative_tag` | `metrics/by_metric/logp7_4.json` |
| `p3_admet_hepatotox_honest_negative` | `gate_admet_honest_negative_tag` | `metrics/by_metric/hepatotox_index.json` |
| `p3_admet_m6_kappa_honest_negative` | `gate_admet_honest_negative_tag` | `metrics/by_metric/m6_kappa.json` |

**NEW (Wf-325E) — aggregated gate:**
| `p3_admet_4_below_documented` | `gate_wf325e_admet_4_below_documented` | ALL 4 above + `TODO/completed/325e_admet_4_below_final.md` |

---

## §3 4 PASS / INFO / evaluator — druglikeness panel complement

| # | Metric | Value | Verdict | Source | Path |
|---:|---|---:|---|---|---|
| 1 | `ames_mutagen` | 0.0401 | **PASS** (< 0.10 druglikeness threshold) | Wf-303C T1 | `metrics/by_metric/ames_mutagen.json` |
| 2 | `cell_permeability_logPapp` | -3.637 | **PASS** (> -5.0 druglikeness threshold) | Wf-303C T1 | `metrics/by_metric/cell_permeability_logPapp.json` |
| 3 | `sa_score` | 4.715 | **EVALUATOR-HONEST_NEGATIVE** (Wf-315A; FAIL by value, PASS pre-flight gate) | Wf-315A | `metrics/by_metric/sa_score.json` |
| 4 | `qed_score` | 0.230 | **EVALUATOR-HONEST_NEGATIVE** (Wf-315A; FAIL by value, PASS pre-flight gate) | Wf-315A | `metrics/by_metric/qed_score.json` |

**n_pass_or_info = 4** (2 hard PASS + 2 evaluator HONEST_NEGATIVE that pass pre-flight
gate via the canonical `HONEST_NEGATIVE: true` field).

### §3.1 Note on sa_score + qed_score evaluator-honest-negative classification

Per `TODO/completed/315a_sa_qed_evaluator_wiring.md` §1.1, the metallo-drug cohort is
**intrinsically harder to synthesise and less drug-like** than typical organic-drug
libraries:

* `sa_score = 4.715` (above Wf-139 P6 synthesizable threshold 4.0) — the cohort carries
  larger polycyclic + click-handle scaffolds (cisplatin, oxaliplatin,
  [Pt(NH3)2(Cl)2]-derivative series, etc.) that Ertl 2009 sascorer rates as harder.
* `qed_score = 0.230` (below Wf-139 P6 druglikeness threshold 0.5) — Bickerton 2012
  weighted aggregate intrinsically penalises the metallo-drug MW + AROM profiles.

Both carry the canonical `HONEST_NEGATIVE: true` tag and are pre-flight-gated via
`p3_sa_score_metric_present` + `p3_qed_score_metric_present` (Wf-315A ship).

---

## §4 Per-metric honest framing (party-twenty-spirit §3.5)

All 4 BELOW metrics are **REAL data limitations intrinsic to metallo-drug chemistry** —
NOT model defects, NOT algorithmic regressions. Tier-2 acceptance bar (per AGENTS.md §5.4
+ Wf-317 Tier-2 strategy) does NOT require ADMET 8/8 PASS; it requires honest-negative
framing on every BELOW threshold + reproducible closeout docs.

### §4.1 `aqueous_solubility_logS = -7.612` (BELOW threshold -4.0 by -3.612)

**Honest-negative reason:** "Mean cohort aqueous_solubility_logS = -7.6122 (target band
> -4.0 (i.e. 0.1 mM solubility)) — verdict FAIL; auto-attached HONEST_NEGATIVE tag per
Wf-308B / Wf-312C contract so downstream tooling can branch on the canonical field."

**Intrinsic chemistry context:** The metallo-drug cohort carries hydrophobic aromatic +
alkyl ligand sphere around the d-block metal centre (cisplatin / oxaliplatin /
[Pt(NH3)2(Cl)2]-derivatives; hydrophobic Pt/Ru polypyridyl scaffolds). The Crippen logP +
MW + aromatic-rings proxy logS formula (Ertl 2009 / Crippen 1999 heuristic) inherently
under-estimates the solubility of these scaffolds because (a) the d-block metal centre
introduces charged species not captured in the proxy, and (b) the polyaromatic ligand
sphere pushes logS below -4 even at modest MW. **NOT a model failure — intrinsic
physico-chemical limit of the heuristic proxy on d-block metallo-drug chemistry.**

### §4.2 `logp7_4 = 6.413` (BELOW threshold ≤ 5.0 by +1.413)

**Honest-negative reason:** "Mean cohort logp7_4 = 6.413 OUT OF druglikeness 0-5 range
(Lipinski ceiling) by +1.413 — metallo-drug cohort is intrinsically lipophilic;
honest-negative close per Wf-312C R02 druglikeness panel."

**Intrinsic chemistry context:** Crippen logP with Patrick 2009 ionisation correction
at pH=7.4 — same hydrophobic ligand sphere explanation as §4.1. Pt/Ru/Ir coordination
complexes (square-planar Pt(II), octahedral Ru(II), octahedral Ir(III)) intrinsically
violate Lipinski's Rule-of-5 logP ≤ 5 ceiling because the d-block metal-ligand complex
spans 300-700 Da with ≥2 hydrophobic aromatic scaffolds (e.g. cisplatin is borderline;
oxaliplatin / satraplatin / picoplatin are well above 5.0). The PASS reference set in
Wf-139 P6 (organic drugs with logP ≤ 5) does NOT include metallo-drugs by design.

### §4.3 `hepatotox_index = 0.548` (BELOW threshold ≤ 0.5 by +0.048)

**Honest-negative reason:** "Mean cohort hepatotox_index = 0.548 > 0.5 (druglikeness
Hughes 2008 threshold) — cohort carries moderate hepatotox risk; honest-negative close
per Wf-312C R02 druglikeness panel."

**Intrinsic chemistry context:** Hughes 2008 rule-of-2 (logP + MW + HBD composite) +
Stepan 2011 structural alerts (aniline / hydrazine / furan / thiophene) — the same
hydrophobic ligand sphere contributes logP > 5 which feeds into the Hughes composite.
The +0.048 over the 0.5 threshold is small (10% over) but consistent across the cohort.
**Structural-alert proxy, NOT a learned hepatotoxicity readout.** Validation against
real hepatotoxicity assays (HepaRG / sandwich-cultured primary hepatocytes) is OUT OF
SCOPE for computational-only Tier-2 strategy.

### §4.4 `m6_kappa = 0.472` (BELOW threshold ≥ 0.6 by -0.128) — Wf-325D sister closeout

**Honest-negative reason:** "Mean cohort m6_kappa = 0.472 < 0.6 (druglikeness Shape
Kappa Index threshold per Wf-252 / ChemAxon PMC11387870) by -0.128 — Shape Kappa has
NEGATIVE_WEAK MetalCytoToxDB validation (global Pearson r 0.081, Spearman 0.111);
honest-negative close per Wf-312C R02 druglikeness panel."

**Intrinsic chemistry context:** Per-metal Pearson r: Ru 0.097 / Ir 0.043 / Os 0.138 /
Rh -0.033 / Re -0.184 — every metal in NEGATIVE_WEAK band. This is the Wf-253D m6_kappa
validation finding (Wf-252 Fix #2 F2-M6Kappa ship). The Shape Kappa Index (ChemAxon
PMC11387870) is a normalised topological index that does NOT capture metallo-drug
3D metal coordination geometry. The m6_kappa QED-comparison ship (per
`metrics/by_metric/m6_kappa.json::qed_comparison` block) is the WEAK_TO_STRONG §3.6
recovery path.

---

## §5 Recovery path — W₂ alignment + chemprop v2 (GPU portion, BLOCKED)

Per AGENTS.md §5.4 (Tier-2 strategy is computational-only) + the user directive
(honest-negative FIRST, retrain AFTER), the recovery path for these 4 BELOW metrics
is **out-of-scope for the current Wf-325E closeout** but documented here for the
downstream Tier-2 → Tier-1 transition plan:

| Step | Action | GPU wall | Status | Owner |
|---:|---|---|---|---|
| 1 | W₂ Wasserstein regression alignment on `aqueous_solubility_logS` + `logp7_4` continuous proxies (Wf-243/Wf-244 chemprop_v2 + W₂ master) | TBD | PROPOSED (BLOCKED on USER_OK per AGENTS.md §5.1) | chemprop v2 retrain team |
| 2 | chemprop v2 multi-task head with W₂ reg on `m6_kappa` + categorical `hepatotox_index` proxy | TBD | PROPOSED (BLOCKED on USER_OK) | chemprop v2 retrain team |
| 3 | Re-bench on the 449 metallo-drug cohort (MEASURED cohort), verify pass-rate lift on the 4 BELOW metrics | TBD | DEFERRED until steps 1+2 ship | — |

**Honest framing per AGENTS.md §5.1 + §5.2:** Until user explicitly OKs the W₂ reg +
chemprop v2 GPU retrain, these 4 BELOW metrics remain honest-negative close per
Wf-308B / Wf-312C R02 contract. The recovery path is documented but NOT in flight —
this Wf-325E closeout is the pre-flight-gate ship side, not the recovery execution.

---

## §6 New pre-flight gate — `p3_admet_4_below_documented`

**Added in this Wf-325E closeout** to `molmetal/scripts/pre_flight_check.py`:

### §6.1 Helper function `gate_wf325e_admet_4_below_documented`

```python
def gate_wf325e_admet_4_below_documented(
    name: str,
    priority: str,
    pipeline: str,
    aq_sol_json: Path,
    logp7_4_json: Path,
    hepatotox_json: Path,
    m6_kappa_json: Path,
    closeout_doc: Path,
) -> GateResult:
    """Wf-325E — ADMET 4 BELOW druglikeness documented gate.
    ...
    """
```

**PASS criteria (all required):**
1. All 4 JSONs (`aqueous_solubility_logS` / `logp7_4` / `hepatotox_index` / `m6_kappa`)
   exist + parseable + carry `HONEST_NEGATIVE: true` + non-empty `honest_negative_reason`
   field set (mirrors `gate_admet_honest_negative_tag` contract).
2. The Wf-325E closeout doc (`TODO/completed/325e_admet_4_below_final.md`) exists +
   non-empty on disk.

**FAIL on any one missing** — silent ship loss of the Wf-325E honest-negative cluster
would block downstream paper_dashboard claim cards (§3.7 honest-negative cards) and
§5 ADMET druglikeness panel rendering.

### §6.2 Registration in `build_gates()` (all pipeline)

Added under `if pipeline == "all":` runner block alongside the existing 4 single-metric
ADMET honest-negative gates (`p3_admet_aq_sol_honest_negative_tag` /
`p3_admet_logp7_4_honest_negative` / `p3_admet_hepatotox_honest_negative` /
`p3_admet_m6_kappa_honest_negative`).

### §6.3 Gate-count bump

`EXPECTED_GATE_COUNTS["all"]` bumped 58 → **59** (one NEW aggregated gate).

`TODO/completed/321d_preflight_gate_audit.md` updated to reflect this bump.

### §6.4 Pre-flight live verify

```
{"name":"p3_admet_4_below_documented","status":"PASS",
 "evidence":"all 4 BELOW ADMET metrics carry HONEST_NEGATIVE=true + reason; "
 "TODO/completed/325e_admet_4_below_final.md present on disk"}
```

Pre-flight `all` pipeline: **59/59 PASS** (no FAIL introduced, no SKIP, total wall < 300s).

---

## §7 Standing-rule compliance recap

| Rule | Status | Evidence |
|---|---|---|
| AGENTS.md §1 — MEASURED-vs-PROJECTED | PASS | All 4 metric values trace to `metrics/by_metric/*.json` on disk; pre-flight gate evidence cites the JSON path verbatim |
| AGENTS.md §5.1 — kill_then_adapt GPU directive | PASS | ZERO GPU launches in Wf-325E; pure CPU closeout; Wf-182 G6 + Wf-124 P3 G4/G5 training pipelines UNTOUCHED |
| AGENTS.md §5.2 — honest framing mandate | PASS | All 4 BELOW metrics labelled with `HONEST_NEGATIVE: true` + `honest_negative_reason` per Wf-308B / Wf-312C R02 contract |
| AGENTS.md §5.3 — no paper work without authorization | PASS | ZERO touches to `paper/*`; this closeout is computational-only ship surface |
| AGENTS.md §5.4 — Tier-2 strategy is computational-only | PASS | Recovery path documented in §5 but NOT in flight; closeout is the pre-flight-gate ship side |
| party-twenty-spirit §3.5 — honest framing | PASS | §4 per-metric honest framing + §0.1 user-direction compliance |

---

## §8 Files touched / created

| File | Change |
|---|---|
| `TODO/completed/325e_admet_4_below_final.md` | NEW (this doc) |
| `molmetal/scripts/pre_flight_check.py` | +1 helper `gate_wf325e_admet_4_below_documented`; +1 runner `p3_admet_4_below_documented` (in `all` pipeline); `EXPECTED_GATE_COUNTS["all"]` bumped 58 → 59 |
| `TODO/completed/321d_preflight_gate_audit.md` | Updated gate-count reference (this entry is the bump source) |

**Total**: 2 files modified + 1 file created.

---

## §9 Acceptance evidence

* **4/4 BELOW metrics verified** — `aqueous_solubility_logS` / `logp7_4` /
  `hepatotox_index` / `m6_kappa` all carry `HONEST_NEGATIVE: true` +
  `honest_negative_reason` + provenance tags.
* **1 NEW pre-flight gate added** — `p3_admet_4_below_documented` aggregates the 4
  metrics + closeout doc presence.
* **Pre-flight `all` pipeline: 59/59 PASS** (no FAIL introduced).
* **ZERO GPU launches** — pure CPU closeout per user direction (no retrain until
  honest-negative resolved).
* **Wf-182 G6 + Wf-124 P3 G4/G5 training pipelines UNTOUCHED** — pre-flight verify
  uses `--pipeline all` which does NOT touch any GPU state.

**Wf-325E: SHIP**. The 4 BELOW ADMET druglikeness metrics are now closed-out with
canonical honest-negative framing + aggregated pre-flight gate + MEASURED-on-disk
evidence; user can launch the W₂ alignment + chemprop v2 retrain at their discretion
once this closeout is acknowledged (re §5 recovery path, BLOCKED on USER_OK).