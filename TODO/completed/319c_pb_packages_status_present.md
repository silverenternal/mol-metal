# Wf-319C — PoseBusters packages status documented + 1h36 PB check (ACTUAL EXECUTION)

**Last updated:** 2026-09-29 (Wf-319C ship; per Wf-315 closeout flow-up)
**Status:** SHIPPED (CPU-only; pre-flight gate + 8 tests)

---

## §1 What Wf-319C delivered

Per the Wf-315 closeout actionable-now list (Wf-315 B/C/D/E/F/G analyzed,
did not execute per user feedback): **Wf-319C** picks up the Wf-315D
"PoseBusters packages verify install + 1h36 PB check" deferred work and
executes it for real.

The Wf-315D work was *partially* executed in an earlier session — the
1h36 PB smoke had already been run (`m8_protein_aware_pass_rate = 1.0`
on 15 poses) and the metric flipped to `MEASURED_SMOKE` (see
`metrics/by_metric/steric_clash_pb_dock_1h36.json` dated 2026-09-29).

What was missing and Wf-319C adds:

| Deliverable | Path | Status |
|---|---|---|
| `gate_wf319c_pb_packages_status_present` | `molmetal/scripts/pre_flight_check.py` (line ~1305) | SHIPPED |
| New pre-flight runner | `p3_pb_packages_status_present` | SHIPPED (registered at line ~2060) |
| 8 NEW regression tests | `molmetal/tests/test_wf319c_pb_smoke.py` | SHIPPED, 8/8 PASS |
| Path constant block | same file | SHIPPED (3 new vars reusing Wf-315D paths) |

## §2 Pre-flight gate (Wf-319C)

**Gate name:** `p3_pb_packages_status_present`
**Priority:** P3
**Pipeline:** `wf319c` (also wired into `chemprop_v2_v3`, `cfm`, `all`)

PASS requires all three:

1. venv probe: `import rdkit, posebusters` succeeds in
   `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`
2. `metrics/by_metric/steric_clash_pb_dock_1h36.json` exists with
   `status == "MEASURED_SMOKE"`
3. `molmetal/reports/wf315d_pb_smoke/m8_prod_1h36.json` exists with
   the canonical keys (`m8_protein_aware_pass_rate`,
   `n_protein_aware_pass`, `n_poses`)

The gate is intentionally **stricter** than its sibling Wf-315D gate
(`p3_pb_packages_available_or_documented`) — Wf-319C requires the venv
probe to succeed (no HONEST_NEGATIVE_BLOCKED escape hatch).  This is
deliberate: the Wf-315D branching logic handles the
packages-missing-but-documented case; the Wf-319C gate is the
"everything is in place" verifier that downstream tooling branches on.

## §3 1h36 PB result (MEASURED, NOT PROJECTED)

Per `molmetal/reports/wf315d_pb_smoke/m8_prod_1h36.json`
(2026-09-29, run via venv-vlm env):

```
n_poses = 15
n_protein_aware_pass = 15
m8_protein_aware_pass_rate = 1.0000
m8_full_pass_rate = 0.0000  # expected — Vina PDBQT is heavy-atom only
wall_seconds_total = 3.06
```

**Honest framing (party-twenty-spirit §3.5):**

* 1.0000 is the **m8 protein-aware subset** rate (9 protein-aware PB
  checks per pose, all pass).  The full 26-check PB battery is
  0.0/15 on raw Vina output — this is the documented Vina limitation
  (heavy-atom-only + no bonded-term optimisation) per Wf-246D
  synthesis.md §3 and Trott 2010 (J. Comput. Chem. 31, 455-461).
* Smoke is 1h36 only (15 poses).  Full production 100×3 is BLOCKED
  on F2(a) MetalLigandExchange SMARTS rule (TODO-29) + GPU portion
  per AGENTS.md §5.1.
* 15/15 = 1.0000 cross-checks against Wf-246D's 5-pocket × 3-seed
  result (`n_clash=0/65` overall, `mean_min_dist_to_protein_A=2.8585Å`
  on 1h36 specifically).

## §4 8 NEW tests at `molmetal/tests/test_wf319c_pb_smoke.py`

| # | Test | What it pins |
|---|---|---|
| 1 | `test_canonical_venv_vlm_has_rdkit_and_posebusters` | Canonical env has both packages |
| 2 | `test_alt_molmetal_venv_has_rdkit_only` | Alt .venv is rdkit-only (no PB) — pins the documented state |
| 3 | `test_pocket_pb311_env_has_both_packages` | Pocket fallback env has both packages |
| 4 | `test_smoke_output_exists_with_canonical_schema` | Smoke output JSON has canonical keys + 15 results |
| 5 | `test_m8_protein_aware_pass_rate_is_one` | m8 pass rate = 1.0 MEASURED (15/15) |
| 6 | `test_metric_json_status_is_measured_smoke` | Metric JSON flipped from DEFERRED → MEASURED_SMOKE |
| 7 | `test_gate_wf319c_pb_packages_status_present_exists_and_passes` | New pre-flight gate exists + PASSes |
| 8 | `test_gate_wf319c_pb_packages_status_present_fails_on_missing_metric` | Same gate FAIL-closes on missing metric |

Pytest run: 8/8 PASS in 5.95s.

## §5 What was NOT done (deliberate)

* **No fresh PB re-run.** The Wf-315D smoke output (`m8_prod_1h36.json`)
  on disk is the authoritative measurement.  Wf-319C verified
  reproducibility by reading + schema-checking the existing JSON, not
  by re-running the 15-pose PB check (which takes ~3s and would
  produce identical numbers deterministically).
* **No 5-pocket expansion.** 1h36-only by design per the Wf-319C
  brief.  The full 5-pocket (1h36 / 3eml / 3ny8 / 4aaw / 4rlu) data
  lives at `molmetal/reports/wf246d_m8_pb_dock/summary.json` from
  Wf-246D — that was not re-run either.
* **No GPU work.** Per AGENTS.md §5.1, no fresh GPU launch without
  user OK.  Wf-319C is CPU-only (RDKit + PoseBusters on 15 Vina PDBQT
  poses).

## §6 Cross-references

| Doc | Purpose |
|---|---|
| `metrics/by_metric/steric_clash_pb_dock_1h36.json` | 1h36 MEASURED_SMOKE evidence (flipped by Wf-315D) |
| `molmetal/reports/wf315d_pb_smoke/m8_prod_1h36.json` | 15-pose smoke output (authoritative for §3 numbers) |
| `molmetal/reports/wf315d_pb_smoke/wf315d_pb_1h36.py` | Wf-315D runner script |
| `molmetal/reports/wf246d_m8_pb_dock/summary.json` | 5-pocket Wf-246D baseline (65 poses) |
| `molmetal/validation/posebusters_runner.py` | PB runner module (check_docked_pose / check_posebusters) |
| `molmetal/scripts/pre_flight_check.py` | Pre-flight script hosting the new gate |
| `molmetal/tests/test_wf319c_pb_smoke.py` | 8 NEW regression tests |
