# Wf-321E — ABLATION_TABLE registry integration test

**Date**: 2026-09-29
**Owner**: Wf-321E (code task)
**Status**: SHIPPED (gate wired; artefact documented)

---

## §1 任务

Verify `molmetal/scripts/ablation_runner.py` mirrors the 11 R17 ablations
declared in `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §7 (brief said
§5, but the actual ablation table sits at §7 — the §5 heading refers to
the M29-M35 mechanism block, not the ablation table).

Cross-check that:

1. Every ablation declared in the §7 table is registered in
   `ABLATION_TABLE`.
2. No spurious ablations are registered.
3. The export name + cardinality are correct (11 unique non-baseline +
   1 BASELINE reference row).

---

## §2 §7 ablation table (canonical 11)

Source: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` lines 603-617.

| ID | Ablation | Target metric | Status |
|---|---|---|---|
| M15 | shim TypeError fix | m14 | ✅ done (wf-289, 0/192) |
| M13 | logits clamp (±50) | m14, m1 | ✅ done (wf-289 F4) |
| M14 | LR warmup (linear 200 steps) | m14 | ✅ done (wf-289 F2) |
| M12 | 20-vocab → 12-vocab | m1, m2 | ✅ done (wf-289 G) |
| M11 | 4 NEW SMARTS | m11 | ⏳ TODO (wf_lambda_88rule_deflex) |
| M10 | 84-rule MLE matrix | m11 | ⏳ TODO (wf_lambda_88rule_deflex) |
| M3 | KataGo-MCTS → vanilla UCB | m7, m11 | ⏳ TODO |
| M4 | PLL heuristic → random | m7, m11 | ⏳ TODO |
| M8 | LambdaASTSR → F5 | m16 | ⏳ TODO (wf_lambda_deflexcite_coupling) |
| M7 | YuelBond → distance-only | m13, m14 | ⏳ TODO |
| M21 | random split → scaffold split | m1, m2, m17 | N/A (data hygiene) |

11 unique non-baseline ablations.

---

## §3 `ABLATION_TABLE` 镜像 check

Source: `molmetal/scripts/ablation_runner.py` lines 134-241.

```python
ABLATION_TABLE: Tuple[AblationSpec, ...] = (
    AblationSpec(ablation_id="BASELINE", ...),  # 1 reference row
    AblationSpec(ablation_id="M15", ...),
    AblationSpec(ablation_id="M13", ...),
    AblationSpec(ablation_id="M14", ...),
    AblationSpec(ablation_id="M12", ...),
    AblationSpec(ablation_id="M11", ...),
    AblationSpec(ablation_id="M10", ...),
    AblationSpec(ablation_id="M3", ...),
    AblationSpec(ablation_id="M4", ...),
    AblationSpec(ablation_id="M8", ...),
    AblationSpec(ablation_id="M7", ...),
    AblationSpec(ablation_id="M21", ...),
)
```

12 entries total (1 BASELINE + 11 §7 ablations). Order-insensitive set
equality with §7:

```
ABLATION_TABLE non-baseline: {M3, M4, M7, M8, M10, M11, M12, M13, M14, M15, M21}
§7 ablation table:           {M3, M4, M7, M8, M10, M11, M12, M13, M14, M15, M21}
```

**Exact match.** All 11 §7 ablations mirrored; no extras.

---

## §4 名称 correction (honest framing)

Brief asked for `from molmetal.scripts.ablation_runner import ABLATION_REGISTRY`.
**That symbol does not exist.** The actual export is `ABLATION_TABLE`.
Wf-321E corrected this in the gate documentation and gate body — the
external symbol must be `ABLATION_TABLE` (verified by reading lines
152-153 of the runner):

> "Ablation registry — must mirror ``TODO/MECHANISM_METRIC_MAPPING.md`` §5."

> "``ABLATION_TABLE: Tuple[AblationSpec, ...] = ( ... )``"

Run-verified:

```
$ python -c "from molmetal.scripts.ablation_runner import ABLATION_REGISTRY; print(len(ABLATION_REGISTRY))"
ImportError: cannot import name 'ABLATION_REGISTRY' from 'molmetal.scripts.ablation_runner'

$ python -c "from molmetal.scripts.ablation_runner import ABLATION_TABLE; ids=[s.ablation_id for s in ABLATION_TABLE]; print(len([i for i in ids if i!='BASELINE']))"
11
```

---

## §5 New pre-flight gate: `p3_ablation_runner_registry_complete`

**Location**: `molmetal/scripts/pre_flight_check.py` (gate
function `gate_wf321e_ablation_registry_complete`; wired into the wf319a
block under P3 pipeline).

**Behaviour**:

1. Imports `ABLATION_TABLE` from
   `molmetal/scripts/ablation_runner.py` via `importlib.util`.
2. Verifies ≥ 11 unique non-baseline IDs.
3. Compares ablation_id set against the canonical 11 §7 set
   (order-insensitive set equality).
4. Verifies BASELINE reference row is present.
5. Verifies each entry carries the §7 contract fields:
   `ablation_id`, `label`, `mechanism_module`, `target_metric`,
   `expected_effect`, `toggle`, `env_vars`, `extra_args`.

**Smoke test result** (Wf-321E ship):

```
status = PASS
evidence = ABLATION_TABLE OK: 11/11 §7 ablations mirrored (M3/M4/M7/M8/M10/M11/M12/M13/M14/M15/M21) + 1 BASELINE; all 12 entries carry the §7 contract fields
wall_sec = 0.0013
```

---

## §6 配套关系 (companion surface)

Wf-321E is the **source-side** companion to Wf-319A's
**artefact-side** `p3_ablation_runner_aggregate_csv_present` gate:

| Gate | Surface | Catches |
|---|---|---|
| Wf-319A `p3_ablation_runner_aggregate_csv_present` | Artefact: `aggregate.csv` ≥ 11 rows | Source-vs-on-disk drift at the artefact layer |
| **Wf-321E `p3_ablation_runner_registry_complete`** | Source: `ABLATION_TABLE` matches §7 set | Source-shape drift at the registry layer (e.g. someone deletes an `AblationSpec` entry without updating §7, or vice-versa) |

Together they form a **two-sided closure check**: the registry must match
the canonical ablation table, AND the aggregate.csv emitted by the
runner must have ≥ 11 rows. Either side failing ⇒ pre-flight FAIL.

---

## §7 Honest framing

* This integration test only verifies the **count + ID set + field
  contract**. It does NOT verify the runtime toggleable hooks (e.g.
  `MOLMETAL_ABLATE_M15=1`) actually fire inside
  `metallo_drug_smoke_retrain.py` — that contract is owned by
  Wf-319A's test suite (`test_wf319a_ablation_runner.py`).
* The expected `ABLATION_TABLE` set is hard-coded in
  `EXPECTED_ABLATION_IDS_S5` inside the gate (named `_S5` per the
  brief's §5 reference). When §7 grows beyond 11 entries the constant
  must be updated.
* The gate does **not** check `predicted_value` semantics (e.g. that
  M15 → m14 = 0.0) — that's the `gate_wf319a_ablation_aggregate_csv_present`
  + Wf-319A test scope.

---

## §8 Files touched (Wf-321E ship)

* `molmetal/scripts/pre_flight_check.py` —
  + `gate_wf321e_ablation_registry_complete(name, priority, pipeline, runner_module)`
  + `EXPECTED_ABLATION_IDS_S5` (canonical 11-ID set, frozen)
  + `wf321e_runner_module` variable in the wf319a block
  + new gate tuple `(p3_ablation_runner_registry_complete, ...)` wired
    into the wf319a runners section.
* `TODO/completed/321e_ablation_runner_integration_test.md` (this doc).

No paper §6 input contracts were modified; no metric JSON files touched;
no `ABLATION_TABLE` row contents modified.

---

## §9 References

* `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §7 (the 11-ablation table)
* `molmetal/scripts/ablation_runner.py` lines 90-141 (registry docstring
  + `ABLATION_TABLE` declaration)
* `molmetal/scripts/pre_flight_check.py` `gate_wf321e_ablation_registry_complete`
* Wf-319A ship surface (`p3_ablation_runner_aggregate_csv_present` —
  artefact-side companion)

---

**END OF Wf-321E** — Owner Wf-321E; cross-ref Wf-319A; last update 2026-09-29.