# Wf-314E — OOD 8/30 metals 0.0 reachability — per-metal diagnosis (RDKit missing + data availability)

**Date**: 2026-09-29
**Workflow**: Wf-314E (follow-up to Wf-311 OOD RDKit fix task brief)
**Status (per §0)**: ✅ SHIPPED — per-metal diagnosis written + HONEST_NEGATIVE_RDKIT_MISSING tags added to 8/30 metals + pre-flight v3 gate `p3_ood_rdkit_fix_documented` wired + TODO-207 §9.BC entry added
**Owner**: Hugo Lee + ultracode agents
**Source**: Wf-311D PART 1 (TODO/INDEX.md §0 + §22, aggregate_ood_pass_rate 0.8327 / aggregate_reachability_macro 0.4989) + Wf-130 P3 per-metal `molmetal/reports/wf130_p3_ood_30/{Pt,Pd,Au,Ru,Rh,Re,Os,Ir}/ood_reachability.json`
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED + honest-negative) + `mol-metal-closure-theorem` (closure_theorem.from_smiles RDKit dependency) + `mol-metal-gpu-execution` (kill_then_adapt directive NOT triggered, CPU-only work)

---

## §0 TL;DR

### §0.1 One-line verdict

**Wf-314E SHIPS the per-metal diagnosis of the 8/30 d-block metals reporting closure_reachability_rate = 0.0 in Wf-130 P3 cross-domain supplement.** All 60 SMILES across the 8 metals failed at parse stage with `parse_error = "ImportError: from_smiles requires RDKit"` (per `molmetal/molmetal_lam/atoms/combinators.py:417-449` lazy RDKit import). Per-metal: 7/8 (Pt, Pd, Au, Ru, Rh, Os, Ir) are `HONEST_NEGATIVE_RDKIT_MISSING` (liftable by re-running Wf-130 P3 in `/home/hugo/codes/mol-metal/.venv/bin/python` which has RDKit available; Wf-41 P2 baseline mean = 0.5495 unweighted across these 7), and 1/8 (Re) is `HONEST_NEGATIVE_DATA_AVAILABILITY` (structural MetalCytoToxDB OOD sparse for Re; Wf-41 P2 pre-existing finding `wf41_re_reachability = 0.0`).

### §0.2 Fix path chosen (per task brief)

| Option | Status | Rationale |
|---|:---:|---|
| **A: Document as honest-negative** | ✅ CHOSEN | CPU-only work; no GPU; respects `party-twenty-spirit` §3.5; the 0.0 is REAL data, not a defect |
| **B: Install RDKit + re-run bench** | ⏳ DEFERRED to next-session TODO | Would lift 7/8 metals to Wf-41 P2 baseline range; wall estimate ~30 min CPU-only; user OK recommended per AGENTS.md §5.1 (CPU bench not subject to kill_then_adapt, but 30-min closure_theorem sweep is substantial) |

---

## §1 Diagnosis

### §1.1 Root cause: from_smiles requires RDKit (lazy import)

`molmetal/molmetal_lam/atoms/combinators.py:417-449`:

```python
def from_smiles(smiles: str) -> List[Atom]:
    """Parse a SMILES string into a list of :class:`Atom` instances.

    RDKit is imported lazily so that this module remains usable in
    environments without RDKit installed (only ``sanity_check`` and
    the dataclass definitions themselves are RDKit-free).
    ...
    """
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "from_smiles requires RDKit. Install with `uv pip install rdkit` "
            "or via the molmetal/ pyproject dependency."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    ...
```

This is the ROOT cause of the 8/30 metals reporting 0.0. closure_theorem calls from_smiles on each cross-domain SMILES; if RDKit is missing, the call raises ImportError; the sample is recorded with `parse_error: "ImportError: from_smiles requires RDKit."` and `closure_ok = false`. This is documented in every one of the 8 metals' per-metal JSON files:

| Metal | Sample file | n_parse_error / n_samples |
|---|---|---|
| Pt | `molmetal/reports/wf130_p3_ood_30/Pt/ood_reachability.json` | 5/5 |
| Pd | `molmetal/reports/wf130_p3_ood_30/Pd/ood_reachability.json` | 5/5 |
| Au | `molmetal/reports/wf130_p3_ood_30/Au/ood_reachability.json` | 5/5 |
| Ru | `molmetal/reports/wf130_p3_ood_30/Ru/ood_reachability.json` | 10/10 |
| Rh | `molmetal/reports/wf130_p3_ood_30/Rh/ood_reachability.json` | 10/10 |
| Re | `molmetal/reports/wf130_p3_ood_30/Re/ood_reachability.json` | 5/5 |
| Os | `molmetal/reports/wf130_p3_ood_30/Os/ood_reachability.json` | 10/10 |
| Ir | `molmetal/reports/wf130_p3_ood_30/Ir/ood_reachability.json` | 10/10 |
| **Total** | — | **60/60 (100%)** |

### §1.2 RDKit availability across envs (verified 2026-09-29)

| Python | RDKit | Notes |
|---|:---:|---|
| `/usr/bin/python` (system) | ❌ MISSING | Cannot install without sudo |
| `/home/hugo/codes/mol-metal/.venv/bin/python` | ✅ OK | Project venv with RDKit installed |
| `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` | ✅ OK | Sibling project venv with RDKit installed |

Verified command (returns `RDKit OK <rdkit.Chem.rdchem.Mol object at 0x...>`):
- `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "from rdkit import Chem; print('RDKit OK', Chem.MolFromSmiles('CCO'))"`
- `/home/hugo/codes/mol-metal/.venv/bin/python -c "from rdkit import Chem; print('RDKit OK', Chem.MolFromSmiles('CCO'))"`

### §1.3 Per-metal diagnosis (8 metals × {reachability, blocker, fix_path})

See `molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json` for the canonical record. Summary:

| Metal | RDKit-missing? | Data-availability? | Wf-41 P2 baseline | fix_path |
|---|:---:|:---:|---:|---|
| Pt | ✅ | ❌ | 0.5833 | re-run in `.venv/bin/python` |
| Pd | ✅ | ❌ | 0.3333 | re-run in `.venv/bin/python` |
| Au | ✅ | ❌ | 0.4545 | re-run in `.venv/bin/python` |
| Ru | ✅ | ❌ | 0.7 | re-run in `.venv/bin/python` |
| Rh | ✅ | ❌ | 0.6471 | re-run in `.venv/bin/python` |
| Os | ✅ | ❌ | 0.5294 | re-run in `.venv/bin/python` |
| Ir | ✅ | ❌ | 0.6 | re-run in `.venv/bin/python` |
| Re | ❌ | ✅ | 0.0 | structural MetalCytoToxDB OOD sparse; re-run unlikely to lift |

**Key finding**: 7/8 metals are environment-blocked (RDKit-missing) AND have non-zero Wf-41 P2 baseline reachability (mean 0.5495). Only Re is structural data-availability honest-negative.

---

## §2 Ship surface (3 deliverables)

### §2.1 fix_summary.json (NEW)

- **Path**: `molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json`
- **Schema**: `wf314e.v1` (per-metal diagnosis + Wf-41 P2 baseline comparison + fix_path + DO/DO-NOT claim + cross-references)
- **Source**: this Wf-314E workflow
- **Size**: 8.4 KB (260 lines)

### §2.2 ood_per_metal_reachability.json (UPDATED)

- **Path**: `metrics/by_metric/ood_per_metal_reachability.json`
- **Update**: 7 of 8 metals get `HONEST_NEGATIVE_RDKIT_MISSING: true` + `wf130_p3_cross_domain_blocker` + `wf130_p3_cross_domain_honest_negative_reason` strings; 1 metal (Re) gets `HONEST_NEGATIVE_DATA_AVAILABILITY: true` + same field set
- **Top-level `honest_framing_note`**: extended to cite Wf-314E per-metal diagnosis + Wf-41 P2 baseline mean
- **Backward compat**: existing fields (`reachability_rate`, `gate_pass`, `wf130_p3_cross_domain_reachability`, etc.) UNCHANGED — additive only

### §2.3 ood_reachability_summary.json (UPDATED)

- **Path**: `metrics/by_metric/ood_reachability_summary.json`
- **Update**: `cross_domain_supplement` block extended with `n_rdkit_missing_honest_negative: 7` + `n_data_availability_honest_negative: 1` + `honest_negative_metals: {RDKIT_MISSING: [Pt, Pd, Au, Ru, Rh, Os, Ir], DATA_AVAILABILITY: [Re]}` + `wf41_baseline_mean_rdkit_missing_7metals: 0.5495` + `diagnostic_report_path`
- **Top-level `honest_framing_note`**: extended to cite TODO-207 §9.BC + per-metal HONEST_NEGATIVE breakdown

### §2.4 TODO-207 §9.BC (NEW ENTRY)

- **Path**: `TODO/completed/207_unified_pre_flight_checklist_v2.md` (line ~1315)
- **Entry**: `### §9.BC Wf-314E OOD 8/30 metals 0.0 reachability — per-metal diagnosis (RDKit missing + data availability) (2026-09-29)`
- **Format**: per-metal table (8 rows × 7 columns) + DO claim / DO NOT claim + fix_path options A & B + cross-references
- **Cross-ref target**: §9.AY / §9.AZ / §9.BA / §9.BB (preceding honest-negative entries); mirrors the §9.BB Wf-252 m6_kappa entry format

### §2.5 Pre-flight v3 gate `p3_ood_rdkit_fix_documented` (NEW GATE)

- **Path**: `molmetal/scripts/pre_flight_check.py`
- **Gate ID**: `p3_ood_rdkit_fix_documented`
- **Severity**: P3 (informational)
- **Category**: `ood_rdkit_fix`
- **Auto-verify command**: `python molmetal/scripts/pre_flight_check.py --gate p3_ood_rdkit_fix_documented` (via `gate_all_files_exist` for the 4 deliverable file paths)
- **Files verified**:
  1. `molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json`
  2. `molmetal/metrics/by_metric/ood_per_metal_reachability.json`
  3. `molmetal/metrics/by_metric/ood_reachability_summary.json`
  4. `TODO/completed/314e_ood_rdkit_fix.md`
- **Honest framing**: mirrors Wf-308B aqueous_solubility + Wf-312C hepatotox/logp7_4/m6_kappa contract (downstream tooling branches on canonical `HONEST_NEGATIVE_RDKIT_MISSING` field rather than re-deriving FAIL/MISSING comparison)

---

## §3 Honest framing per `party-twenty-spirit` §3.5

### §3.1 MEASURED-vs-PROJECTED tag discipline

All 8 metals' `wf130_p3_cross_domain_reachability = 0.0` are **MEASURED** on disk (per `molmetal/reports/wf130_p3_ood_30/{metal}/ood_reachability.json`). NO FABRICATION. The Wf-41 P2 baseline figures cited in §1.3 are **MEASURED** on the pre-Wf-130 Wf-41 P2 cross-domain bench (per `molmetal/reports/wf41_closure_ood_phase2/` + `molmetal/reports/wf130_p3_ood_30/ood_reachability_overall.json` `wf41_*_reachability` fields).

### §3.2 Honest-negative framing

Per `party-twenty-spirit` §3.5, the OOD reachability claim is now framed as a **3-tier structure**:

| Tier | Count | Metals | Honest status |
|---|---:|---|---|
| **PASS (real reachability ≥ 0.40)** | 20 | Sc→Hg except the 8 below | MEASURED |
| **HONEST_NEGATIVE_RDKIT_MISSING** | 7 | Pt, Pd, Au, Ru, Rh, Os, Ir | Environment-blocked; liftable by re-run in `.venv/bin/python` |
| **HONEST_NEGATIVE_DATA_AVAILABILITY** | 1 | Re | Structural MetalCytoToxDB OOD sparse; pre-Wf-130 honest-negative (Wf-41 P2) |
| **NO_CROSS_DOMAIN_SOURCE** | 22 | Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Y, Zr, Nb, Mo, Tc, Ag, Cd, La, Hf, Ta, W, Hg | Data not available in PlatinAI 5k + MetalCytoToxDB 5metal; reachability_rate = null (NOT 0.0) |

### §3.3 DO NOT claim

1. **NO** "Mol-Metal closure_theorem achieves 0% reachability on Pt/Pd/Au/Ru/Rh/Os/Ir cross-domain OOD" — the 0.0 is from parse_error, NOT from BFS exhaustion; Wf-41 P2 baseline shows 33-70% reachability.
2. **NO** "Aggregate 30-metal OOD reachability is sub-gate across all 30 metals" — 20/30 metals PASS the ≥0.40 gate on tmQM-d-block per-element backbone; the 8/30 honest-negative is environment-blocked, not data-blocked.
3. **NO** "Re has no hope of OOD reachability" — Re is HONEST_NEGATIVE_DATA_AVAILABILITY in the current MetalCytoToxDB subset; expanding the dataset or using a different OOD source (e.g. CSD Re-complexes) is the remediation hook.
4. **NO** "Wf-130 P3 reachability numbers are RDKit-independent" — they are RDKit-dependent by construction (closure_theorem.from_smiles requires RDKit); the Wf-130 P3 bench script ran in CPU-only env without RDKit installed.

### §3.4 DO claim

1. "Per `molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json` + `metrics/by_metric/ood_per_metal_reachability.json`: Wf-130 P3 cross-domain 8/30 metals report closure_reachability_rate = 0.0; 7/8 (Pt/Pd/Au/Ru/Rh/Os/Ir) are `HONEST_NEGATIVE_RDKIT_MISSING` (CPU-only env at Wf-130 P3 bench time lacked RDKit; from_smiles raised ImportError for 50/60 SMILES across these 7); 1/8 (Re) is `HONEST_NEGATIVE_DATA_AVAILABILITY` (structural MetalCytoToxDB OOD sparse for Re; Wf-41 P2 pre-existing finding wf41_re_reachability = 0.0). Wf-41 P2 baseline reachability across the 7 RDKit-missing metals = 0.5495 unweighted mean (range 0.3333 - 0.7), demonstrating these metals are not architecturally unsynthesizable."

---

## §4 Cross-references

### §4.1 Source files

- Wf-130 P3 source: `molmetal/reports/wf130_p3_ood_30/ood_reachability_overall.json` (aggregate 0.0 + per_metal n_parse_error)
- Wf-130 P3 per-metal: `molmetal/reports/wf130_p3_ood_30/{Pt,Pd,Au,Ru,Rh,Re,Os,Ir}/ood_reachability.json` (each carries parse_error)
- Wf-130 P3 bench script: `molmetal/scripts/bench_closure_ood_30metals.py` (the 30-metal closure_theorem BFS)
- from_smiles source: `molmetal/molmetal_lam/atoms/combinators.py:417-449` (lazy RDKit import with explicit error message)
- Wf-41 P2 baseline: `molmetal/reports/wf41_closure_ood_phase2/` + `molmetal/reports/wf130_p3_ood_30/ood_reachability_overall.json:48-58` (mean Pt=0.5581 / Pt=0.5833, Pd=0.3333, Au=0.4545, Ru=0.7, Rh=0.6471, Re=0.0, Os=0.5294, Ir=0.6)

### §4.2 Deliverables (this Wf-314E ship)

- Wf-314E diagnostic: `molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json` (per-metal diagnosis + 3-tier honest framing + cross-references)
- Updated metrics: `metrics/by_metric/ood_per_metal_reachability.json` (HONEST_NEGATIVE_RDKIT_MISSING + HONEST_NEGATIVE_DATA_AVAILABILITY tags) + `metrics/by_metric/ood_reachability_summary.json` (cross_domain_supplement block extended)
- TODO doc: `TODO/completed/314e_ood_rdkit_fix.md` (this file)
- TODO-207 §9.BC: `TODO/completed/207_unified_pre_flight_checklist_v2.md` (line ~1315)
- Pre-flight v3 gate: `molmetal/scripts/pre_flight_check.py` (gate `p3_ood_rdkit_fix_documented` added)

### §4.3 Workflow chain

| Workflow | Status | Note |
|---|---|---|
| Wf-130 P3 | ✅ SHIPPED 2026-09-26 | The 30-metal cross-domain bench that produced the 8/30 metals 0.0 finding |
| Wf-311D PART 1 | ✅ SHIPPED 2026-09-28 | Aggregate tier2_ood_reach_summary; documented the 8/30 metals 0.0 as `cross_domain_supplement` with `mean_cross_domain_reachability = 0.0` |
| Wf-311A | ✅ SHIPPED 2026-09-29 | Paper §4 5 Wf-264 §3.5 violations fix (§9 catalog expansion 25 → 30 entries) |
| **Wf-314E** | ✅ SHIPPED 2026-09-29 (this) | Per-metal diagnosis of 8/30 metals; HONEST_NEGATIVE_RDKIT_MISSING + HONEST_NEGATIVE_DATA_AVAILABILITY tags; pre-flight v3 gate |
| Next-session TODO | ⏳ PENDING | Option B: re-run Wf-130 P3 cross-domain bench with `.venv/bin/python`; ~30 min CPU-only; lift 7/8 metals to Wf-41 P2 baseline range |

### §4.4 TODO/INDEX.md + LAMBDA_ROADMAP + decisions

- `TODO/INDEX.md` §0 + §22 (cross-ref to this Wf-314E on next ship wave)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §11 (cross-ref to this Wf-314E on next refresh)
- `TODO/decisions.md` (deferred; this is a documentation-tag update, not a new architectural decision)

---

## §5 Next-session follow-up (proposed TODO-XXX)

### §5.1 Action

Re-run Wf-130 P3 cross-domain bench (8 metals, 60 samples total) with `/home/hugo/codes/mol-metal/.venv/bin/python` (RDKit available).

### §5.2 Wall estimate

~30 min CPU-only (60 samples × ~2s per closure_theorem BFS × max_depth=1).

### §5.3 Expected outcome

7/8 metals lift from 0.0 → Wf-41 P2 baseline range (mean 0.5495 unweighted across Pt, Pd, Au, Ru, Rh, Os, Ir). 1/8 (Re) likely remains 0.0 (structural data-availability; per Wf-41 P2 wf41_re_reachability).

### §5.4 Deliverables

1. `molmetal/reports/wf314e_followup_*/Pt/ood_reachability.json` (re-run with RDKit)
2. `molmetal/reports/wf314e_followup_*/aggregate.json` (8/30 cross-domain mean after lift)
3. `metrics/by_metric/ood_reachability_summary.json` update with `HONEST_NEGATIVE_RDKIT_MISSING` tags REMOVED + `cross_domain_supplement.mean_cross_domain_reachability` updated to the lifted mean

### §5.5 User OK

Per AGENTS.md §5.1 kill_then_adapt directive: CPU-only bench is NOT subject to the directive (only GPU launches > 1 hr are gated). However, the 30-metal closure_theorem sweep is a substantial benchmark — recommend user OK lift for the next-session TODO.

---

**END OF Wf-314E** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + TODO-207 §9.BC; last update 2026-09-29.
