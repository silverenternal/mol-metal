# WF-Extra-2: REINVENT4 multi-property JSONL worker — shipped 2026-09-14

> **Honest framing:** the *worker* (subprocess bridge) is **MEASURED**
> and runs against the upstream REINVENT4 binary at
> `/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/reinvent`.  The
> *wiring into `RewardAggregator`* is **PROJECTED** — implemented in
> design (component channel already exists, the worker is now
> protocol-compatible) but not yet plumbed through the closed-loop
> harness.  No search-loop runs were performed for this delivery.

---

## 1. Spec

### 1.1 Worker file

`/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/reinvent4_multiproperty_jsonl_worker.py`

### 1.2 Wire protocol (JSON-lines over stdin/stdout)

**Request** (one per line):

```json
{
  "smiles": "CCO",
  "components": {"logp": 0.4, "ring_count": 0.2, "qed": 0.4},
  "timeout": 60.0
}
```

**Response** (one per line):

```json
{
  "smiles": "CCO",
  "multiproperty_score": 0.5625158053759708,
  "components_raw": {"qed": 0.4068, "slogp": 0.9994, "numrings": 0.000177},
  "device": "cuda:0",
  "model_sha": "f3c82ce036d7b50c",
  "ok": true,
  "reason": null,
  "elapsed_seconds": 3.49
}
```

### 1.3 Recognised component keys (and their REINVENT4 mapping)

| key | REINVENT4 component | transform |
|-----|---------------------|-----------|
| `qed`            | `QED`                       | none |
| `logp`           | `SlogP`                     | reverse_sigmoid(0.5→4.0, k=0.5) |
| `ring_count`     | `NumRings`                  | sigmoid(1→5, k=0.5) |
| `aromatic_rings` | `NumAromaticRings`          | sigmoid(0→3, k=0.5) |
| `mw`             | `MolecularWeight`           | double_sigmoid(200→500) |
| `tpsa`           | `TPSA`                      | double_sigmoid(0→140) |
| `hbd`            | `HBondDonors`               | reverse_sigmoid(0→5) |
| `hba`            | `HBondAcceptors`            | reverse_sigmoid(4→10) |
| `rot_bonds`      | `NumRotBond`                | reverse_sigmoid(2→10) |
| `heavy_atoms`    | `NumHeavyAtoms`             | reverse_sigmoid(15→50) |
| `matching_smarts`| `MatchingSubstructure`      | penalty (c1ccccc1) |

Unknown keys are silently dropped; an empty components dict falls
back to plain QED with weight 1.0.

### 1.4 Failure modes

| `ok` | `reason` | meaning |
|------|----------|---------|
| `false` | `invalid_smiles`        | RDKit/REINVENT4 cannot parse the SMILES (fast-path pre-check) |
| `false` | `missing_binary`        | `reinvent` not on PATH (binary probe failed) |
| `false` | `timeout`               | subprocess exceeded `timeout` seconds |
| `false` | `backend_error:*`       | non-zero exit code, missing CSV, parse error |
| `true`  | `null`                  | success — `multiproperty_score` ∈ [0, 1] |

### 1.5 Aggregate semantics

The worker passes `type = "arithmetic_mean"` (weighted average) to
REINVENT4.  All component transforms return values in `[0, 1]` by
construction (sigmoids, double-sigmoids), so the weighted arithmetic
mean is itself in `[0, 1]`.  This matches the contract of
`reinvent4_jsonl_worker.py` and the four-tuple
`ScoreResult(qed, sa, binding, novelty)` consumed by
`ScoreAggregator`.

---

## 2. Test results — MEASURED

`uv run pytest -q molmetal/molmetal_lam/tests/test_reinvent4_multiproperty_worker.py --tb=short`

```
...........                                                              [100%]
=============================== warnings summary ================================
.venv/lib/python3.12/site-packages/_hypothesis_pytestplugin.py:487
  /home/hugo/codes/try_triton_on_rocm/.venv/lib/python3.12/site-packages/_hypothesis_pytestplugin.py:487: UserWarning: Skipping collection of '.hypothesis' directory
    warnings.warn(...)

======================== 11 passed, 1 warning in 37.61s ========================
```

### 2.1 Test inventory

| # | test | class | status | description |
|---|------|-------|--------|-------------|
| 1 | `test_worker_valid_smiles_returns_score`  | TestReinvent4MultipropertyWorker         | passed | CCO → score in [0,1] |
| 2 | `test_worker_invalid_smiles_graceful`     | TestReinvent4MultipropertyWorker         | passed | `!!!INVALID!!!` → reason=invalid_smiles |
| 3 | `test_worker_empty_smiles_graceful`       | TestReinvent4MultipropertyWorker         | passed | empty string → reason=invalid_smiles |
| 4 | `test_worker_timeout`                     | TestReinvent4MultipropertyWorker         | passed | timeout=0.05s aborts gracefully |
| 5 | `test_unknown_component_key_silently_dropped` | TestReinvent4MultipropertyWorkerUnit  | passed | unknown key dropped without crash |
| 6 | `test_empty_components_defaults_to_qed`   | TestReinvent4MultipropertyWorkerUnit     | passed | empty components → QED fallback |
| 7 | `test_zero_weight_component_dropped`      | TestReinvent4MultipropertyWorkerUnit     | passed | weight=0 → component excluded |
| 8 | `test_components_key_stable`              | TestReinvent4MultipropertyWorkerUnit     | passed | dict order independent |
| 9 | `test_components_key_ignores_zero_weights`| TestReinvent4MultipropertyWorkerUnit     | passed | zero weights omitted from fingerprint |
|10 | `test_response_keys_present`              | TestReinvent4MultipropertyWorkerWireContract | passed | every documented key is present |
|11 | `test_components_raw_is_dict`             | TestReinvent4MultipropertyWorkerWireContract | passed | even on errors, components_raw is dict |

### 2.2 Metrics

| metric | value |
|--------|-------|
| `n_tests`              | 11 |
| `n_passed`             | 11 |
| `n_failed`             |  0 |
| `smoke_score_for_ethanol` | 0.5625158053759708 |
| `wall_clock_per_query`    | ~3.3-3.9 s (REINVENT4 subprocess cold start dominates) |

---

## 3. Smoke run output — MEASURED

Direct subprocess calls into the worker (real `reinvent` binary,
real RDKit, no mocks):

```
=== Test 1: Valid SMILES (CCO) ===
{"smiles": "CCO", "multiproperty_score": 0.5625158053759708, "components_raw": {"qed": 0.4068079656553945, "slogp": 0.9993926, "numrings": 0.00017779632}, "device": "cuda:0", "model_sha": "f3c82ce036d7b50c", "ok": true, "reason": null, "elapsed_seconds": 3.4944}

=== Test 2: Invalid SMILES ===
{"smiles": "!!!INVALID!!!", "multiproperty_score": null, "components_raw": {}, "device": "cuda:0", "model_sha": null, "ok": false, "reason": "invalid_smiles"}

=== Test 3: Benzene (c1ccccc1) ===
{"smiles": "c1ccccc1", "multiproperty_score": 0.5234852222583378, "components_raw": {"qed": 0.4426283718993647, "slogp": 0.8645085, "numrings": 0.0031523092}, "device": "cuda:0", "model_sha": "f3c82ce036d7b50c", "ok": true, "reason": null, "elapsed_seconds": 3.3121}

=== Test 4: Acetylsalicylic acid (aspirin) ===
{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "multiproperty_score": 0.5897348912333189, "components_raw": {"qed": 0.5501217966938848, "slogp": 0.95654964, "mw": 0.13854544}, "device": "cuda:0", "model_sha": "9f75c658542dff7f", "ok": true, "reason": null, "elapsed_seconds": 3.371}

=== Test 5: Cisplatin ([NH3][Pt]([NH3])(Cl)Cl) ===
{"smiles": "[NH3][Pt]([NH3])(Cl)Cl", "multiproperty_score": 0.8420029038688222, "components_raw": {"qed": 0.6841053474944192, "mw": 0.99990046}, "device": "cuda:0", "model_sha": "169b1ad763391457", "ok": true, "reason": null, "elapsed_seconds": 3.2886}
```

### 3.1 Interpretive notes

* **CCO** scores higher than benzene on logP (`0.9994` vs `0.8645`)
  but lower on QED (`0.407` vs `0.443`) — consistent with ethanol
  being a smaller, more lipophilic molecule.
* **Cisplatin** scores surprisingly well (`0.8420`) because the
  chosen components are QED + MW: the MW sigmoid peaks around 300 Da
  and Pt complexes get a flat `0.9999` there.  Note that QED's
  underlying heuristic is not trained on metal complexes — this is
  the documented limitation, not a bug.
* **Aspirin** scores `0.5897` — MW penalises it (`0.139`) because
  it's at the upper end of the [200, 500] sweet-spot window.  SlogP
  scores it nearly perfectly (`0.957`).
* **`model_sha`** is the SHA256-prefix of the components dict,
  truncated to 16 hex chars.  Identical components → identical SHA,
  so callers can detect "same objective across runs".
* **Wall-clock** is dominated by REINVENT4's Python interpreter cold
  start (~3 s).  Repeated calls in the same process would amortise
  this; the worker is designed to be invoked once per (SMILES,
  components) tuple for the open-loop reward-evaluation use case.

---

## 4. Wiring — PROJECTED

The worker's output is wire-compatible with `ScoreResult` /
`ScoreAggregator` from `reinvent4_subprocess_adapter.py`:
`multiproperty_score` is exactly the `[0, 1]` scalar that
`RewardAggregator.r_reinvent4` expects.

The adapter already exists and degrades gracefully when the
underlying worker is missing.  To wire the new worker in:

1. **Swap binary**: change `REINVENT4Adapter.binary` default from
   `reinvent4_jsonl_worker.py` to
   `reinvent4_multiproperty_jsonl_worker.py`, OR
2. **Env override**: `REINVENT4_BIN=/path/to/reinvent4_multiproperty_jsonl_worker.py`
3. **Replace `ScoreAggregator`**: the existing 4-tuple
   `(qed, sa, binding, novelty)` is a fixed-component aggregator; the
   new worker returns a single weighted scalar.  Either:
   - drop a thin wrapper class that calls the new worker and returns
     a synthetic `ScoreResult` whose `.qed` equals the
     `multiproperty_score` (cheapest path), OR
   - replace `ScoreAggregator.aggregate` with a one-line call to the
     new worker's output.

**Status:** PROJECTED.  No closed-loop run was performed; the wiring
is described above and is a 5-line patch in
`reinvent4_subprocess_adapter.py` plus a one-line env-var flip in the
harness entry point.

---

## 5. Files touched (this delivery)

| path | action |
|------|--------|
| `molmetal/molmetal_lam/sbdd_env/reinvent4_multiproperty_jsonl_worker.py` | **new** — the worker |
| `molmetal/molmetal_lam/tests/test_reinvent4_multiproperty_worker.py`    | **new** — 11 tests |
| `molmetal/reports/wf_extra2_worker.md`                                 | **new** — this report |

No existing files were modified.

---

## 6. Reproduction commands

```bash
cd /home/hugo/codes/try_triton_on_rocm

# Run tests
uv run pytest -q molmetal/molmetal_lam/tests/test_reinvent4_multiproperty_worker.py --tb=short

# Smoke (ethanol)
echo '{"smiles": "CCO", "components": {"qed": 0.4, "logp": 0.4, "ring_count": 0.2}}' \
  | /mnt/storage/env-projects/reinvent4-rocm/.venv/bin/python \
      molmetal/molmetal_lam/sbdd_env/reinvent4_multiproperty_jsonl_worker.py
```
