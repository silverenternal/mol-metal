# Wf-306A — drugood + Vina GPU launch plan

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Status**: READY on USER_OK gate (per Wf-294B kill_then_adapt pattern)
**Total wall PROJECTED**: ~5 hr

---

## §1  Context (per Wf-297B / Wf-299B / Wf-301A / Wf-301B / Wf-304C / Wf-305D)

| Metric | Baseline | Field-avg target | Honest framing |
|---|---|---|---|
| `drugood_v3_scaffold` `ood_test_auc` | 0.4574 (Wf-297B) | 0.55 - 0.65 (Wf-301A) | BELOW band; Wf-304C recovery PROJECTED via Wf-285B DANN axis |
| `vina_min` `kcal_mol` (PlatinAI mbfinder × 1h36) | n_docked=0/100 (Wf-299B) | -7.5 to -9.5 (Wf-301B) | Pt+2 UFF parameterisation blocker; Wf-301B shipped `--pt-strip`; Wf-305D ships the wrapper |

**Wf-305D shipped the implementation** (drugood DANN driver + Vina pt-strip wrapper) but **CPU smoke NOT run**.  Wf-306A is the follow-up that:
1. Runs CPU smoke on both implementations (PART 1 + PART 2)
2. Authors the GPU launch wrapper (PART 3)

---

## §2  Wf-306A PART 1 + PART 2 — CPU smoke results (MEASURED)

### PART 1 — drugood DANN CPU smoke
**Status**: IN-FLIGHT at report-write time (CPU featurization of 95,236 rows is the bottleneck; expected to complete in < 30 min on kanzi CPU).  Once the featurization cache finishes, the 1-epoch D-MPNN forward/backward + Wasserstein + DANN GRL pass completes in < 10 min.

**Configuration** (per Wf-306A brief):
* `wf305d_drugood_dann_train.py --seeds 0 --epochs 1 --hidden 64 --depth 2 --device cpu --use-dann --out-dir /tmp/wf306a_drugood_smoke`
* Train/val/test split: 66665/9524/19047 (Bemis-Murcko scaffold split from Wf-275B)
* D-MPNN hidden=64, depth=2 (smoke-spec)
* `--use-dann` GRL head, `lambda_grl=0.1`, 4-class domain head (train/val/test/NCI-60-reserved)
* Wasserstein pIC50 regulariser, `w_w=0.05`

**Verdict gate** (per Wf-306A brief):
* `ood_test_auc` MUST be emitted in JSON
* No exceptions
* JSON has `n_seeds_completed=1` + `per_seed[0].test_auc`

### PART 2 — Vina pt-strip CPU smoke — **PASS**
**Status**: PASS MEASURED at 01:50 UTC.

**Output** (`/tmp/wf306a_vina_smoke/panel.json`):
```
n_input=5  n_parsed=5  n_pdbqt_ok=5  n_docked=5
vina_binary_ok=True  overall_mean_kcal_mol=-4.6042
```

**Per-molecule kcal/mol** (best pose × pocket 1h36):
| SMILES (largest fragment after Pt-strip) | best kcal/mol |
|---|---|
| `[OH2][Pt]([Cl])([Cl])N12CCN(CC1)CC2` | -4.083 |
| `[OH2][Pt]([OH2])([OH2])([Cl])N12CN3CN(CN(C3)C1)C2.[Cl]` | -4.685 |
| `O=C([O])C1C[S][Pt]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[S]C1.[H]` | -4.354 |
| `CN1P(c2ccccc2)(c2ccccc2)(c2ccccc2)[Pt]2([S]CC(C(=O)[O])C[Se]2)P1(c1ccccc1)(c1ccccc1)c1ccccc1.[H]` | -4.952 |
| `CN1P(c2ccccc2)(c2ccccc2)(c2ccccc2)[Pt]2([Se]CC(C(=O)[O])C[Se]2)P1(c1ccccc1)(c1ccccc1)c1ccccc1.[H]` | -4.947 |

**Verdict gate** (per Wf-306A brief): "at least 1/5 docked successfully (vs 0/5 without pt-strip); vina_score emitted" — **5/5 docked, REAL_VINA status, vina_score emitted** — **PASS**.

**Honest framing** (party-twenty-spirit §3.5): `overall_mean_kcal_mol=-4.6042` is **ABOVE** the field-avg band -9.5 to -7.5 — `passes_field_avg_band=False` (honest-negative).  However the **recovery axis** is *being able to dock at all*: without `--pt-strip` the 0/100 baseline (Wf-299B) is `STUB_FALLBACK`, and with `--pt-strip` we now get 5/5 `REAL_VINA`.  The band-pass is the next-step improvement; this smoke establishes the **docking pipeline is functional**, which is the prerequisite for any band-pass work.

**Upstream blocker discovered + fixed**: Wf-305D smoke on the project `.venv` returned `n_parsed=0` with `note='meeko import failed: ModuleNotFoundError'`.  Re-ran with `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` (the harness's intended venv) which has meeko 0.8.0 installed → 5/5 PASS.  **Action**: the GPU launch wrapper pins `PY=${PY:-/home/hugo/codes/CadButEaas/.venv-vlm/bin/python}` per Wf-300C convention.

---

## §3  Wf-306A PART 3 — GPU launch wrapper

**File**: `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh`
**Status**: written, executable, syntax-checked, dry-run PASS, USER_OK gate verified.

### Phases

| Phase | Wall PROJECTED | GPU | Output | Trainers |
|---|---|---|---|---|
| A — drugood DANN full | ~3-4 hr | cuda:1 (RTX 5090) | `reports/wf306a_drugood_vina_launch/phase_a/{seed0,seed1,seed2}.json + training_summary.json` | `wf305d_drugood_dann_train.py --seeds 0 1 2 --epochs 50 --hidden 128 --depth 3 --use-dann` |
| B — Vina pt-strip full | ~1-2 hr | CPU | `reports/wf306a_drugood_vina_launch/phase_b/panel.json` | `wf305d_vina_pt_strip.py --n-mols 100 --pocket-id 1h36 --pt-strip` |
| **Total** | **~5 hr** | | | |

### Pre-launch gates (BLOCKER before any phase)

1. `WF306A_USER_OK=YES` env var (kill_then_adapt guard per Wf-294B + Wf-300C)
2. `pre_flight_check.py --pipeline all` — 24/24 PASS target (TODO-207 v2 §0.2)
3. `wave_budget_tracker.py --budget-hr 6 --cfm-yuelbond-required` — DEFERRED expected (drugood+Vina are not CFM)

### Per-phase abort criteria (TODO-207 v2 §7.2)

* 7.2.1  timeout:  ckpt older than 2× PROJECTED wall-time → kill_then_adapt
* 7.2.2  failure pattern:  3+ NaN/OOM/IndexError → abort_recovery.py
* 7.2.3  GPU util < 70% for 5+ min (Phase-A only) — advisory
* 7.2.4  collapse:
  * Phase-A: drugood `ood_test_auc < 0.45` on 2/3 seeds → abort
  * Phase-B: Vina `n_docked < 50/100` → abort (pt-strip regression detection)

### Usage

```bash
# Dry-run (no GPU spinup, validates CLI + paths + USER_OK)
WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --dry-run --phase A

# Real GPU launch (requires USER_OK + kanzi GPU 1 free)
WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase A
WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase all

# Wave B alone
WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase B
```

---

## §4  Honest framing (party-twenty-spirit §3.5)

* **CPU smoke = implementation verification**.  Field-avg target (drugood ≥ 0.55, Vina -7.5 to -9.5) requires **GPU + user OK**.
* **PART 1 drugood smoke**: 1-seed × 1-epoch is NOT statistically significant on the field-avg gate; the gate is a per-seed JSON emission + `n_seeds_completed=1`.  The GPU run (3-seed × 50-epoch) is the gate that matters for paper §3.
* **PART 2 Vina smoke**: 5/5 docked = **infrastructure recovery**; the band-pass on kcal/mol is honest-negative at -4.60 (above band).  The GPU run (100-mol × 4-pocket) is the next-step band-pass work.
* **GPU wrapper is READY but not yet launched** (per Wf-294B kill_then_adapt discipline).  USER_OK must be granted by Hugo before any wave starts.

---

## §5  File pointers

| Path | Role |
|---|---|
| `molmetal/scripts/wf305d_drugood_dann_train.py` | Wf-305D PART 1 driver (DANN MT-DMPNN on DrugOOD) |
| `molmetal/scripts/wf305d_vina_pt_strip.py` | Wf-305D PART 2 driver (Vina pt-strip on PlatinAI mbfinder) |
| `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` | Wf-306A PART 3 GPU launch wrapper (NEW) |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | DANN + Wasserstein + ReverseLayerF primitives (Wf-285B + Wf-131 P1) |
| `molmetal/scripts/cfm_vina_dock_panel.py` | Vina build_panel + pt-strip primitive (Wf-301B + Wf-264) |
| `molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv` | Wf-275B adapter (95,236 rows) |
| `.data/molmetal/platinai_4cell_mbfinder.csv` | Wf-295B PlatinAI mbfinder (100 Pt complexes) |
| `molmetal/reports/wf30_pocket_prep/1h36/` | Wf-30 staged 1h36 receptor |
| `/tmp/wf306a_drugood_smoke/` | CPU smoke PART 1 output (in-flight) |
| `/tmp/wf306a_vina_smoke/panel.json` | CPU smoke PART 2 output (PASS, MEASURED) |
| `molmetal/reports/wf306a_drugood_vina_launch/` | GPU launch reports dir (this file lives here) |

---

## §6  Next-step actions

1. **WAIT** for drugood CPU smoke to complete (~30 min expected).
2. **AGGREGATE** PART 1 JSON output into this launch plan §2.
3. **REQUEST USER OK** from Hugo for the 5-hr GPU launch.
4. **LAUNCH** `bash launch_wf306a_drugood_vina_gpu.sh --phase all` (or per-phase).
5. **WRITE Wf-306A closeout TODO** with MEASURED ood_test_auc + n_docked + per-pocket kcal/mol.
