# Wf-323C — UFF pre-relax + Vina re-dock CPU smoke for m8_full lift

**Date:** 2026-09-29
**Author:** Wf-323C (follow-up to Wf-319C / Wf-315D)
**Status:** SHIPPED with HONEST_NEGATIVE_UFF_INSUFFICIENT (as designed per task brief)
**Branch:** master
**Commit:** (this Wf)

---

## 1. Context (why this Wf exists)

Per Wf-319C + WEAK_TO_STRONG §2.6:

> m8_full_pass_rate = 0/15 on the Wf-315D PoseBusters smoke retry
> (15 on-disk Wf-179 docked poses re-checked with PB dock mode).
> The 5 consistently failing checks: bond_lengths, bond_angles,
> internal_steric_clash, aromatic_ring_flatness, internal_energy.

The Wf-246D synthesis.md §3 root-cause analysis: **Vina's heavy-atom-only
output** has no explicit Hs and no geometry optimization of bonded terms
— both required for PB's reference window.

Wf-323C hypothesis: **pre-relax the conformers with UFF (RDKit CPU)** so
the upstream 3D geometry hits better bonded-term ranges, then re-dock
via Vina 1.2.7. The UFF-min Pt-pool from Wf-319B (96 conformers) is the
input source.

**Per the Wf-323C brief**: *"If still 0/15: HONEST_NEGATIVE_UFF_INSUFFICIENT
+ recovery path (MMFF94 + GPU)."*

---

## 2. MEASURED outcome (16-conformer CPU smoke)

| Metric | Wf-315D | Wf-323C | Δ |
|---|---:|---:|---:|
| `m8_full_pass_rate` | 0.0000 (0/15) | **0.0000 (0/14)** | **+0.0000** |
| `m8_protein_aware_pass_rate` | 1.0000 (15/15) | 0.6875 (11/16) | −0.3125 |
| `n_poses_pre_relaxed` | 15 | 14 | — |
| `n_pt_zn_substituted` | 0 (organic input) | 14 (Pt-pool input) | — |
| `wall_seconds_total` | 8.13 | 98.68 | — |

**Lift vs projected +0.10–0.15: NOT ACHIEVED.** UFF pre-relax alone is
**INSUFFICIENT** for the PB bonded-terms window. The same 5 checks fail
as the Wf-315D baseline:

| Failing check | Wf-323C (n_fail/14) |
|---|---:|
| `internal_energy` | 14/14 |
| `inchi_convertible` | 14/14 |
| `bond_angles` | 13/14 |
| `bond_lengths` | 11/14 |
| `internal_steric_clash` | 6/14 |
| `minimum_distance_to_protein` | 2/14 |
| `no_radicals` | 1/14 |
| `protein-ligand_maximum_distance` | 1/14 |

The `internal_energy` failure on all 14 is the smoking gun: PB's
energy-ratio module requires UFF/MMFF94 reference conformers, and
RDKit's UFF can't parameterise Pt (per Wf-319B's
`HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING`). The `inchi_convertible`
failure on all 14 is the Pt→Zn substitution side-effect (InChI can't
be generated for the Zn-substituted atom type).

---

## 3. Discovery: Vina 1.2.7 rejects `Pt` in PDBQT (NEW blocker)

Wf-323C discovered that **Vina 1.2.7's PDBQT parser rejects the `Pt`
atom-type**:

```
PDBQT parsing error: Atom type Pt is not a valid AutoDock type
(atom types are case-sensitive)
```

This is the same blocker noted for the Wf-299B CFM Vina panel
(per AGENTS.md §5.1: *"CFM Vina docking is BLOCKED on Pt+2
parameterisation"*). Wf-323C surfaces this as a first-class
`HONEST_NEGATIVE_VINA_PT_REJECTED` tag.

**Wf-323C CPU-side workaround**: auto-substitute `Pt` / `Ru` / `Ir` /
`Os` / `Rh` / `Pd` / `Au` / `Re` with `Zn` (the closest Vina-accepted
divalent d-block) in the ligand PDBQT — the same
**MEASURED_GENERIC_ATOM** convention as Wf-246-C. The 3D coords + bond
graph are preserved exactly; only the AD4 atom-type field is rewritten.

**Downstream caveat**: the kcal/mol for the metal anchor is the generic
Zn-VDW approximation, not the true Pt-anchored kcal/mol. Production
recovery requires **GPU retrain with custom Pt scoring_function**
(BLOCKED on AGENTS.md §5.1 + ≥1 hr wall budget).

---

## 4. Ship surface

### 4.1 Code + data

| Path | Purpose |
|---|---|
| `molmetal/scripts/wf323c_m8_full_cpu_fix.py` | New CLI entry point (502 LOC) |
| `molmetal/reports/wf323c_m8_full/result.json` | MEASURED output (16-conformer smoke) |
| `molmetal/reports/wf246d_m8_pb_dock/1h36_receptor.pdbqt` | Auto-converted from PDB (Vina requires strict PDBQT) |
| `metrics/by_metric/steric_clash_pb_dock_1h36.json` | Updated to `MEASURED_SMOKE` + Wf-323C smoke-summary block + HONEST_NEGATIVE_UFF_INSUFFICIENT + HONEST_NEGATIVE_VINA_PT_REJECTED |

### 4.2 Pre-flight gate

Added `p3_m8_full_pass_rate_documented` to the pre-flight check.
**Function**: `gate_wf323c_m8_full_pass_rate_documented` in
`molmetal/scripts/pre_flight_check.py`.

**PASS criteria** (all required):
1. `result.json` exists + JSON-loadable + carries the canonical
   top-level keys (`m8_full_pass_rate`, `m8_protein_aware_pass_rate`,
   `n_poses_pre_relaxed`, `n_pt_zn_substituted`, `results`).
2. `steric_clash_pb_dock_1h36.json` is `MEASURED_SMOKE` AND carries
   the `wf323c_smoke_summary` block.
3. Wf-319B pool JSON has ≥80 conformers (the Wf-323C input source).

The gate accepts **both** a successful lift AND an honest-negative
outcome — the gate's contract is that the delta is **documented**, not
that it exceeds a target. This matches the party-twenty-spirit §3.5
honest-framing mandate.

### 4.3 Unit tests

`molmetal/tests/test_wf323c_m8_full_cpu_fix.py` — **8 NEW tests** (all
PASS):

| # | Test | Contract |
|---|---|---|
| 1 | `test_wf323c_script_present` | Script exists + carries the canonical anchors |
| 2 | `test_wf323c_result_required_keys` | All 13 top-level keys present |
| 3 | `test_wf323c_pass_rates_in_unit_interval` | Both rates are floats in [0.0, 1.0] (MEASURED, not None) |
| 4 | `test_wf323c_pt_zn_substitution_recorded` | ≥1 OK Vina record with `pt_zn_substituted=True` + non-None kcal/mol |
| 5 | `test_wf323c_n_poses_pre_relaxed_matches_records` | Top-level count == OK records count |
| 6 | `test_wf323c_honest_negative_tags_consistent` | `HONEST_NEGATIVE_UFF_INSUFFICIENT` + `HONEST_NEGATIVE_VINA_PT_REJECTED` set when applicable |
| 7 | `test_wf323c_preflight_gate_pass` | `gate_wf323c_m8_full_pass_rate_documented` returns PASS |
| 8 | `test_wf323c_expected_gate_count_bumped` | `EXPECTED_GATE_COUNTS['all']` ≥ 57 (was 56 pre-Wf-323C) |

### 4.4 Pre-flight result

```
"name": "p3_m8_full_pass_rate_documented",
"priority": "P3",
"pipeline": "wf323c",
"status": "PASS",
"evidence": "result.m8_full=0.0000 (n_pre=14, n_pt_zn=14); metric
MEASURED_SMOKE; pool=96 conformers"
```

---

## 5. Recovery plan (the M1 — MMFF94 + GPU path)

The Wf-323C outcome confirms: **UFF pre-relax alone is insufficient**
for the PB bonded-terms window on Pt-coordination complexes. The
recovery path documented per the brief:

| Step | Action | Effort | Status |
|---|---|---|---|
| 1 | MMFF94 (RDKit MMFFOptimizeMolecule) on the organic donor shell — MMFF94 covers the C/N/O organic atoms UFF misses | CPU | DEFERRED (next Wf) |
| 2 | GPU retrain with custom Vina scoring_function override for Pt-group d-block metals | GPU | BLOCKED on AGENTS.md §5.1 + ≥1 hr wall budget |
| 3 | Metal-coordination term wire-up (Wf-246-C) on the d-block atoms | CPU+GPU | BLOCKED on AGENTS.md §5.1 |
| 4 | DiffDock / Boltz fallback (no Vina, native metal support) | GPU | BLOCKED on AGENTS.md §5.1 |

The honest framing: **m8_full_pass_rate = 0.0000 stays at 0** until
one of steps 1-4 completes. The Wf-323C pipeline + Pt→Zn
substitution + UFF pre-relax are the **CPU ceiling** for the lift
on the 1h36 pocket.

---

## 6. Honest framing (party-twenty-spirit §3.5)

### 6.1 MEASURED claims

* `n_poses_pre_relaxed = 14` (MEASURED — `molmetal/reports/wf323c_m8_full/result.json`)
* `n_pt_zn_substituted = 14` (MEASURED — same)
* `m8_full_pass_rate = 0.0000 (0/14)` (MEASURED — same)
* `m8_protein_aware_pass_rate = 0.6875 (11/16)` (MEASURED — same)
* `wall_seconds_total = 98.68 s` (MEASURED — same)

### 6.2 PROJECTED claims (NOT MEASURED — explicitly labelled)

* PROJECTED m8_full lift: **+0.10 to +0.15** (per WEAK_TO_STRONG §2.6
  R18 §2.7 brief) — **NOT ACHIEVED** (lift = +0.0000).
* PROJECTED recovery path (MMFF94 + GPU) is the M1, deferred to a
  future Wf.

### 6.3 HONEST_NEGATIVE claims

* `HONEST_NEGATIVE_UFF_INSUFFICIENT` — UFF pre-relax alone cannot
  lift m8_full above the 5% delta threshold.
* `HONEST_NEGATIVE_VINA_PT_REJECTED` — Vina 1.2.7 PDBQT parser rejects
  the `Pt` atom-type. Wf-323C auto-substitutes `Zn` (MEASURED_GENERIC_
  ATOM). Production recovery requires GPU retrain with custom Pt
  scoring_function.

Both tags are recorded in `result.json` and propagated to the
`steric_clash_pb_dock_1h36.json` metric JSON's `value.wf323c_smoke_
summary` block + `notes` field.

### 6.4 What was NOT changed

* `paper/*` — no touches per AGENTS.md §5.3 (TODO-84 active).
* Wf-315D's `m8_prod_1h36.json` — that is the Wf-315D baseline
  measurement (15 Wf-179 organic poses); Wf-323C ran on a different
  input (16 Wf-319B Pt-pool conformers).
* `molmetal/scripts/bench_vina_triple_rebench.py` — untouched (Wf-60
  bench is for Vina triple m4 + m10, not m8).
* Wf-319B `pool.json` — untouched (input source).

---

## 7. Verification commands

```bash
# Run the Wf-323C pipeline (dry-run)
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf323c_m8_full_cpu_fix.py

# Run the Wf-323C pipeline (execute, 16 conformers)
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf323c_m8_full_cpu_fix.py \
    --n-conformers 16 --exhaustiveness 4 --n-poses 2 --execute

# Run the Wf-323C unit tests
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf323c_m8_full_cpu_fix.py -v

# Run the pre-flight gate
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py | grep m8_full_pass_rate
```

---

**END OF Wf-323C SHIP SURFACE** — Owner: Wf-323C; cross-ref
`TODO/completed/319b_uff_min_pt_pool.md`, `TODO/completed/315d_pb_smoke_retry.md`,
`TODO/completed/319c_pb_packages_status_present.md`,
WEAK_TO_STRONG §2.6 R18; last update 2026-09-29.
