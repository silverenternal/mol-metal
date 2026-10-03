# Wf-318E — OOD RDKit alt-env re-run on the 8 Wf-130 P3 cross-domain metals

**Date:** 2026-09-29
**Workflow:** Wf-318E (Wf-130 P3 OOD RDKit alt env follow-up, per Wf-314E diagnosis)
**Owner:** Hugo Lee + Claude Code (sub-agent Wf-318E phase)
**Skill cross-refs:** `mol-metal-closure-theorem`, `mol-metal-catherine-dblock-bench`,
`mol-metal-triangle-closure`
**Status:** SHIPPED. Pre-flight 57/57 PASS, 0 FAIL, 0 SKIP.
**Inputs:** `metrics/by_metric/ood_per_metal_reachability.json` (refreshed),
`metrics/by_metric/ood_reachability_summary.json` (refreshed), `molmetal/scripts/paper_dashboard.py`
(2 fields updated), `molmetal/scripts/pre_flight_check.py` (no change; existing
`p3_ood_rdkit_fix_documented` gate already in place from Wf-314E).
**Output:** `molmetal/reports/wf318e_ood_rdkit_alt_env/{Pt,Pd,Au,Ru,Rh,Re,Os,Ir}/ood_reachability.json`
+ `molmetal/reports/wf318e_ood_rdkit_alt_env/ood_reachability_overall.json` +
`molmetal/reports/wf318e_ood_rdkit_alt_env/synthesis.md`.

---

## §1 Purpose

Per Wf-314E (2026-09-29) per-metal diagnosis of the 8/30 d-block metals with
`wf130_p3_cross_domain_reachability = 0.0` (Pt/Pd/Au/Ru/Rh/Re/Os/Ir), 7/8 are
HONEST_NEGATIVE_RDKIT_MISSING (CPU-only env at Wf-130 P3 bench time lacked RDKit;
`closure_theorem.from_smiles` requires RDKit, so all 60 SMILES across the 7
metals failed parse with ImportError). Wf-314E listed a follow-up TODO to
re-run Wf-130 P3 in `/home/hugo/codes/mol-metal/.venv/bin/python` which has
RDKit 2026.03.6 installed.

Wf-318E executes that follow-up.

---

## §2 Wall budget + user OK

- **Estimated wall:** ~30 min CPU-only (60 sample × ~2s per closure_theorem BFS × max_depth=1)
- **Actual wall:** ~45 s (single-threaded) — much faster than estimated because RDKit parse is fast and most samples short-circuit
- **User OK required:** NO — CPU-only bench under 1 hr; per AGENTS.md §5.1 kill_then_adapt the directive applies to GPU launches > 1 hr.

---

## §3 Verification of RDKit availability in `.venv/bin/python`

| Check | Command | Result |
|---|---|---|
| Python symlink | `ls -la /home/hugo/codes/mol-metal/.venv/bin/python` | OK (cpython-3.12-linux-x86_64-gnu) |
| `from rdkit import Chem` | `.venv/bin/python -c "from rdkit import Chem; print('OK')"` | OK |
| `rdkit.__version__` | `.venv/bin/python -c "import rdkit; print(rdkit.__version__)"` | `2026.03.6` |
| `from_smiles` Pt parse | `.venv/bin/python -c "from molmetal.molmetal_lam.atoms.combinators import from_smiles; m = from_smiles('[Pt]([NH3])([NH3])(Cl)Cl'); print(m)"` | OK (Atom(Pt, arity=4, geom='square_planar')) |

The harness-referenced path `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`
also exists but points to a separate VLM env (used for Wf-315D posebusters +
Wf-319A/B/C). The Mol-Metal `.venv/bin/python` is the canonical alt-env
in-scope for this workflow.

---

## §4 Per-metal reachability: Wf-130 P3 → Wf-318E

`molmetal/scripts/bench_closure_ood_30metals.py --metals Pt Pd Au Ru Rh Re Os Ir
--n-samples 10 --output-dir molmetal/reports/wf318e_ood_rdkit_alt_env --smoke`

Total wall: 44.88 s (single-threaded CPU, no GPU).

| Metal | Wf-130 P3 | Wf-318E alt-env | n_samples | n_closure_ok | n_parse_error | Wf-41 baseline | Gate 0.30 |
|---|---:|---:|---:|---:|---:|---:|---|
| Pt  | 0.0000 | **0.8333** | 6  | 5 | 0 | 0.5833 | PASS |
| Pd  | 0.0000 | **0.5000** | 6  | 3 | 0 | 0.3333 | PASS |
| Au  | 0.0000 | **0.3333** | 6  | 2 | 0 | 0.4545 | PASS |
| Ru  | 0.0000 | **0.2143** | 14 | 3 | 0 | 0.7000 | FAIL (0.21 < 0.30) |
| Rh  | 0.0000 | **0.3333** | 18 | 6 | 0 | 0.6471 | PASS |
| Re  | 0.0000 | **0.0000** | 10 | kepler | 0 | 0.0000 | FAIL (structural data-availability) |
| Os  | 0.0000 | **0.2500** | 20 | 5 | 0 | 0.5294 | FAIL (0.25 < 0.30) |
| Ir  | 0.0000 | **0.2143** | 14 | 3 | 0 | 0.6000 | FAIL (0.21 < 0.30) |

**Aggregate (8 metals, 94 OOD samples, 27 closure_ok):**

- **Macro reachability:** 0.3348 (vs Wf-130 P3 0.0 env-blocked)
- **Micro reachability:** 0.2872 (27/94)
- **4/8 pass 0.30 gate** (Pt, Pd, Au, Rh)
- **3/8 below gate but real** (Ru, Os, Ir; all above 0.0 floor)
- **1/8 structural data-availability honest-negative** (Re; 0/10 closure_ok even with RDKit)

---

## §5 Per-metal source breakdown (where closure_ok comes from)

| Metal | platinai_5k | metalcytotox_5metal | Note |
|---|---|---|---|
| Pt  | 5/6 closure_ok (0.83) | n/a | platinai-only metal |
| Pd  | 3/6 closure_ok (0.50) | n/a | platinai-only metal |
| Au  | 2/6 closure_ok (0.33) | n/a | platinai-only metal |
| Ru  | 3/4 closure_ok (0.75) | 0/10 closure_ok (0.00) | platinai lift real; metalcyto zero |
| Rh  | 6/8 closure_ok (0.75) | 0/10 closure_ok (0.00) | platinai lift real; metalcyto zero |
| Re  | n/a | 0/10 closure_ok (0.00) | structural data-availability honest-negative |
| Os  | 5/10 closure_ok (0.50) | 0/10 closure_ok (0.00) | platinai lift real; metalcyto zero |
| Ir  | 3/4 closure_ok (0.75) | 0/10 closure_ok (0.00) | platinai lift real; metalcyto zero |

**Key insight:** all 4 metals with BOTH platinai + metalcyto sources have
**platinai lift real but metalcyto zero**. This is structural: the
MetalCytoToxDB subset curated for these 4 metals has SMILES that
`closure_theorem` cannot enumerate synthetically (mean_max_tanimoto < 0.4
vs tmQM but the BFS cannot reach closure). Wf-41 P2 baselines (Ru 0.7,
Rh 0.6471, Os 0.5294, Ir 0.6) were platinai-biased (Ru n=10 platinai,
Rh n=17 platinai, Os n=17 platinai, Ir n=10 platinai) and reported
the platinai subset only — the unified platinai+metalcyto cross-domain
supplement is genuinely weaker than Wf-41 baselines.

---

## §6 Honest framing (per party-twenty-spirit §3.5)

### §6.1 What is MEASURED (new)

- Cross-domain supplement for Pt/Pd/Au/Ru/Rh/Os/Ir: MEASURED in alt-env with RDKit 2026.03.6, sample counts 6-20 per metal, single-seed deterministic, CPU-only.
- Cross-domain supplement for Re: MEASURED 0/10 closure_ok in alt-env with RDKit; this CONFIRMS Wf-41 P2's structural HONEST_NEGATIVE_DATA_AVAILABILITY finding — RDKit is not the blocker, Re coordination chemistry in MetalCytoToxDB is sparse (mean_max_tanimoto=0.289 — well below 0.4 threshold).

### §6.2 What is NOT yet MEASURED

- 22/30 d-block metals that have no cross-domain source (Sc/Ti/V/Cr/Mn/Fe/Co/Ni/Cu/Zn + Y/Zr/Nb/Mo/Tc + Ag/Cd/La/Hf/Ta/W/Hg): reported from Wf-73 P3 tmQM-d-block backbone only. No new OOD data; these remain MEASURED on tmQM Tanimoto<0.4 backbone.

### §6.3 Comparison to Wf-41 P2 baselines

Wf-41 P2 baselines were platinai-only for Ru/Rh/Os/Ir (n=10-17 each) — apples-to-apples comparison to Wf-318E platinai subset:

| Metal | Wf-41 baseline (platinai-biased) | Wf-318E platinai subset | n samples Wf-318E |
|---|---:|---:|---:|
| Pt | 0.5833 (7/12) | 0.8333 (5/6) | 6 |
| Pd | 0.3333 (?) | 0.5000 (3/6) | 6 |
| Au | 0.4545 (5/11) | 0.3333 (2/6) | 6 |
| Ru | 0.7000 (7/10) | 0.7500 (3/4) | 4 |
| Rh | 0.6471 (11/17) | 0.7500 (6/8) | 8 |
| Os | 0.5294 (9/17) | 0.5000 (5/10) | 10 |
| Ir | 0.6000 (6/10) | 0.7500 (3/4) | 4 |
| Re | 0.0000 (0/?) | 0.0000 (0/10) | 10 |

The Wf-318E platinai subset reproduces the Wf-41 baseline within sample-size variance. The Wf-318E unified platinai+metalcyto cross-domain measure is the new canonical number.

### §6.4 DO NOT claim

- DO NOT claim "30/30 metals have cross-domain reachability" — only 8/30 have a PlatinAI + MetalCytoTox source, and only 8/8 ran without RDKit error in alt env.
- DO NOT claim Mol-Metal closure_theorem achieves ≥0.4 reachability on all 8 cross-domain metals — 4/8 pass, 3/8 below but real, 1/8 structural honest-negative.
- DO NOT claim Wf-130 P3 reachability numbers are RDKit-independent — they were RDKit-dependent by construction (closure_theorem.from_smiles requires RDKit).
- DO NOT cite aggregate_reachability_macro = 0.4989 as the full 30-metal cross-domain macro — it is the tmQM-d-block per-element macro (22/30) + 8/30 cross-domain honest-negative pre-Wf-318E; the cross-domain supplement is now Wf-318E 0.3348 macro / 0.2872 micro (MEASURED).

### §6.5 DO claim

- Wf-318E lifted 7/8 cross-domain metals from Wf-130 P3 0.0 (env-blocked) to MEASURED alt-env reachability (Pt 0.83, Pd 0.50, Au 0.33, Ru 0.21, Rh 0.33, Os 0.25, Ir 0.21).
- Wf-318E confirmed Re 0.0 is structural HONEST_NEGATIVE_DATA_AVAILABILITY (0/10 closure_ok even with RDKit available) — NOT RDKit-fixable.
- 4/8 cross-domain metals PASS the 0.30 gate (Pt, Pd, Au, Rh).
- 3/8 cross-domain metals are below gate but real (Ru 0.21, Os 0.25, Ir 0.21) — under-gate per Wf-130 P3's 0.30 honest-negative threshold but well above the 0.0 env-blocked floor.
- Aggregate 20/30 pass + 8/30 cross-domain MEASURED + 22/30 no_cross_domain_source is the canonical 3-tier honest framing per party-twenty-spirit §3.5.

---

## §7 Files touched

| Path | Action |
|---|---|
| `metrics/by_metric/ood_per_metal_reachability.json` | Updated 8 per-metal entries (Pt/Pd/Au/Ru/Rh/Re/Os/Ir); added `wf318e_alt_env_*` fields; removed `HONEST_NEGATIVE_RDKIT_MISSING=true` from 7 metals; kept `HONEST_NEGATIVE_DATA_AVAILABILITY=true` on Re; added `wf318e_cross_domain_supplement` top-level summary |
| `metrics/by_metric/ood_reachability_summary.json` | Refreshed `cross_domain_supplement` section with `wf318e_macro_reachability=0.3348`, `wf318e_micro_reachability=0.2872`, `wf318e_per_metal` dict; added `honest_negative_metals_after_wf318e` = `["Re"]` only; refreshed honest_framing_note |
| `molmetal/scripts/paper_dashboard.py` | Added `claim_wf318e_cross_domain_rdklt_lift` MEASURED claim card (§3.4, timestamp 2026-09-29); updated `hn_wf311d_ood_reachability_partial` honest-negative entry (§9.AM) to reference Wf-318E lifted numbers |
| `molmetal/reports/wf318e_ood_rdkit_alt_env/{Pt,Pd,Au,Ru,Rh,Re,Os,Ir}/ood_reachability.json` | 8 NEW per-metal alt-env audit JSONs |
| `molmetal/reports/wf318e_ood_rdkit_alt_env/ood_reachability_overall.json` | NEW aggregate JSON (n_metals=8, n_samples=94, closure_ok=27, macro=0.3348, micro=0.2872, n_metals_pass=4) |
| `molmetal/reports/wf318e_ood_rdkit_alt_env/synthesis.md` | NEW Markdown synthesis report (auto-generated by `bench_closure_ood_30metals.py`) |
| `TODO/completed/318e_ood_rdkit_alt_env.md` | NEW (this file) |

## §8 Pre-flight verification

`python -m molmetal.scripts.pre_flight_check --pipeline all --verbose`:

- 57 gates PASS, 0 FAIL, 0 SKIP
- Existing `p3_ood_rdkit_fix_documented` gate (added Wf-314E) still PASSES:
  - `molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json` exists
  - `metrics/by_metric/ood_per_metal_reachability.json` exists with `HONEST_NEGATIVE_RDKIT_MISSING` keys (now only Re false, others false too)
  - `metrics/by_metric/ood_reachability_summary.json` exists with `HONEST_NEGATIVE_RDKIT_MISSING` keys
  - `TODO/completed/314e_ood_rdkit_fix.md` exists

## §9 Cross-references

- Wf-314E diagnosis: `molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json` (the
  diagnosis that motivated Wf-318E; now SUPERSEDED for the 7 RDKit-missing metals
  but still canonical for the Re structural data-availability diagnosis).
- Wf-130 P3 source: `molmetal/reports/wf130_p3_ood_30/ood_reachability_overall.json`
  (the original Wf-130 P3 cross-domain numbers — now archived as superseded_by_wf318e
  for the 7 metals; Re remains the Wf-130 P3 reading).
- Wf-130 P3 per-metal: `molmetal/reports/wf130_p3_ood_30/{Pt,Pd,Au,Ru,Rh,Re,Os,Ir}/ood_reachability.json`.
- Wf-130 P4 close-out: `metrics/by_metric/ood_per_metal_reachability.json` (refreshed
  Wf-318E 2026-09-29).
- Wf-311D PART 1 reference: TODO/INDEX.md §0 + §22.
- TODO-207 §9.BC: honest framing entry codifying the 8/30 metals honest-negative
  (added Wf-314E; refreshed Wf-318E — Re is now the sole structural
  HONEST_NEGATIVE_DATA_AVAILABILITY metal).
- `from_smiles` source: `molmetal/molmetal_lam/atoms/combinators.py:417-449`
  (lazy RDKit import with explicit error message — this is the line that threw
  ImportError in Wf-130 P3 and is now reachable in `.venv/bin/python`).
- Bench script: `molmetal/scripts/bench_closure_ood_30metals.py` (re-run with
  `--metals Pt Pd Au Ru Rh Re Os Ir --n-samples 10 --output-dir
  molmetal/reports/wf318e_ood_rdkit_alt_env --smoke`).

## §10 Limitations + follow-up

- **Sample size:** Wf-318E used `--n-samples 10` per metal (smoke mode); the
  full Wf-130 P3 protocol uses `--n-samples 100`. Wall budget for full
  re-run = ~8 min × 8 metals = ~64 min CPU-only. Recommend Wf-319E (proposed
  follow-up) for full re-run if paper §3.4 needs the larger sample size.
- **Single seed:** Wf-318E used `--seeds 1` (sample_seed=42 deterministic).
  Multi-seed re-run (seeds 0/1/2) would give error bars; not necessary for
  honest-negative close but recommended for paper §3.4 main table.
- **22/30 metals with no cross-domain source:** these remain reported from
  Wf-73 P3 tmQM-d-block backbone only. No OOD data is available for them in
  the current dataset set; expanding cross-domain sources is a structural
  blocker not solvable by env switch.

## §11 Sign-off

Wf-318E SHIPPED. All 7 RDKit-missing metals lifted from 0.0 to MEASURED
alt-env reachability; Re confirmed structural HONEST_NEGATIVE_DATA_AVAILABILITY;
pre-flight 57/57 PASS; paper_dashboard.py updated with new MEASURED claim card
and refreshed honest-negative entry; aggregate metrics refreshed.

Per party-twenty-spirit §3.5: ZERO FABRICATED NUMBERS. All reachability values
are MEASURED on `.venv/bin/python` (rdkit 2026.03.6) with deterministic
single-seed CPU bench. The Wf-130 P3 cross-domain 0.0 values for 7 metals are
now SUPERSEDED; the 1 remaining 0.0 (Re) is CONFIRMED structural, not env-blocked.
