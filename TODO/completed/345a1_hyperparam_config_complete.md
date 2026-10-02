# Wf-345-A1 — Fix p3_wf343c4_hyperparams_config_complete gate via KEY_ALIASES

**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-345-A1 ultracode agent
**Workflow:** Wf-345 (ultracode residual closeout wave)
**Status:** SHIPPED

---

## §1 Problem statement

Per `wx63q8fwe` ultracode review (2026-09-29 15:51 UTC), the
`p3_wf343c4_hyperparams_config_complete` pre-flight gate at
`molmetal/scripts/pre_flight_check.py:2940` was FAILing with evidence:

```
6/7 YAMLs missing AGENTS.md §5.1 standing GPU grant hyperparam contract:
14 missing hyperparam(s):
['cfm_full_retrain.yaml:train.batch_size (None)',
 'cfm_per_innovation_ablation_p1.4.yaml:train.batch_size (no train dict)',
 'cfm_per_innovation_ablation_p1.4.yaml:train.lr (no train dict)', ...]
```

The gate enforces AGENTS.md §5.1 (2026-09-29 user grant: **"Hyperparameters
via config files: reference peer papers in the same domain; encode in
`--batch-size`, `--lr`, `--hidden`, `----depth` config rather than
guessing"**) by requiring the 4 canonical keys
(`train.batch_size`, `train.lr`, `model.hidden_dim`, `model.n_layers`)
in every active `configs/*.yaml`.

The contract is real, but the implementation is too strict: only
`configs/default.yaml` carries the canonical schema; the other 6
active YAMLs use **accepted** but non-canonical locations
(`protocol.batch_size`, `protocol.lr`, `model.depth`) for valid
historical reasons (the CFM p1.x YAMLs predate the Wf-343C4 gate;
the Wf-124 P3 chemprop_v2_v3 YAMLs deliberately use `model.depth`
to match the `--depth` CLI flag convention).

---

## §2 Decision: approach (b) — KEY_ALIASES extension

Two approaches were on the table:

- **(a) Rename 6 YAMLs** to canonical schema. Rejected — would break
  downstream consumers (launch scripts, ablation runners, audit
  consumers) reading those keys.
- **(b) Extend `REQUIRED_HYPERPARAMS` with alias resolution.** Chosen —
  preserves backward compat + audit-observable (resolutions named in
  evidence message) + 1-line per-alias extension.

PROJECTED safer than (a) per the brief: rename touches 6 YAMLs ×
~10 downstream consumers each = 60 LOC of rip-and-replace risk;
(b) touches 1 file (~80 LOC net) + 1 YAML (cfm_full_retrain.yaml
needed canonical `batch_size` added because it was completely missing
even from aliases — it had only `--batch-size 16` in the entry.command
string).

---

## §3 Implementation

### §3.1 Code change: `molmetal/scripts/pre_flight_check.py:2940-3131`

Added `KEY_ALIASES` dict + `_resolve_key` helper to
`gate_wf343c4_hyperparams_config_complete`:

```python
KEY_ALIASES: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
    ("train", "batch_size"): [
        ("train", "batch_size"),
        ("protocol", "batch_size"),
        ("data", "batch_size"),
    ],
    ("train", "lr"): [
        ("train", "lr"),
        ("protocol", "lr"),
        ("optimizer", "lr"),
    ],
    ("model", "hidden_dim"): [
        ("model", "hidden_dim"),
        ("model", "hidden"),
    ],
    ("model", "n_layers"): [
        ("model", "n_layers"),
        ("model", "depth"),
        ("model", "num_layers"),
    ],
}
```

The `_resolve_key(obj, canonical_section, canonical_key)` helper
walks the alias list, returns `(value, resolved_alias)` where
`resolved_alias` is the `(section, key)` tuple that produced the
value (or `None` if no match). Type validation runs against the
canonical key, but the evidence message names the alias location
used so audit consumers can see the resolution.

The PASS evidence message now appends the alias-resolutions dict:

```
all 7/7 active YAMLs carry train.batch_size + train.lr + model.hidden_dim + model.n_layers
(AGENTS.md §5.1 standing GPU grant hyperparam contract):
['cfm_full_retrain.yaml', ..., 'wf124_p3_remaining_metals_g4.yaml']
(alias-resolved: {'cfm_per_innovation_ablation_p1.4.yaml':
  'batch_size:protocol.batch_size; lr:protocol.lr', ...})
```

### §3.2 YAML fix: `configs/cfm_full_retrain.yaml:73`

The CFM full retrain config had `batch_size` only inside the
`entry.command` string (as `--batch-size 16`), with no top-level
`batch_size` key under any section.  Added canonical
`train.batch_size: 16` (matching the CLI flag value) with a
`# Wf-345-A1` provenance comment.

No other YAML edit was needed — the alias map handles the
remaining 5 YAMLs.

---

## §4 Verification

### §4.1 Gate PASS (was FAIL)

Before fix:
```
p3_wf343c4_hyperparams_config_complete: FAIL
  evidence: "6/7 YAMLs missing AGENTS.md §5.1 standing GPU grant
           hyperparam contract: 14 missing hyperparam(s)"
```

After fix:
```
p3_wf343c4_hyperparams_config_complete: PASS
  evidence: "all 7/7 active YAMLs carry train.batch_size + train.lr
           + model.hidden_dim + model.n_layers (AGENTS.md §5.1
           standing GPU grant hyperparam contract):
           [cfm_full_retrain.yaml, cfm_per_innovation_ablation_p1.4.yaml,
            cfm_real_decode_lift_p1.3.yaml, default.yaml,
            wf124_p3_50epoch_full_sweep_g5.yaml,
            wf124_p3_per_innovation_ablation_g6.yaml,
            wf124_p3_remaining_metals_g4.yaml]
           (alias-resolved: {cfm_per_innovation_ablation_p1.4.yaml:
             'batch_size:protocol.batch_size; lr:protocol.lr',
            cfm_real_decode_lift_p1.3.yaml: 'batch_size:protocol.batch_size; lr:protocol.lr',
            wf124_p3_50epoch_full_sweep_g5.yaml: 'batch_size:protocol.batch_size; lr:protocol.lr; n_layers:model.depth',
            wf124_p3_per_innovation_ablation_g6.yaml: 'batch_size:protocol.batch_size; lr:protocol.lr; n_layers:model.depth',
            wf124_p3_remaining_metals_g4.yaml: 'batch_size:protocol.batch_size; lr:protocol.lr; n_layers:model.depth'})"
  wall_sec: 0.0599
```

MEASURED gate PASS at 2026-09-29 ~16:00 UTC (Wf-345-A1 timestamp).

### §4.2 NEW tests: 4/4 PASS

`molmetal/tests/test_wf345a1_hyperparam_aliases.py`:

```
test_alias_resolution_canonical_keys_pass                  PASSED [ 25%]
test_alias_resolution_alias_keys_pass                      PASSED [ 50%]
test_all_7_active_yamls_pass_after_alias_resolution       PASSED [ 75%]
test_alias_resolutions_in_evidence_message                 PASSED [100%]
========================= 4 passed, 1 warning in 0.38s =========================
```

The 4 tests cover:
1. **Canonical-schema PASS** — backward-compat invariant (default.yaml
   path still works; no alias branch added when fully canonical).
2. **Alias-schema PASS + no-match FAIL** — confirms each documented
   alias location (`protocol.batch_size`, `protocol.lr`,
   `model.depth`) resolves, AND a YAML with neither canonical nor
   aliased locations correctly FAILs.
3. **7/7 active YAMLs PASS** — regression-test surface; runs the gate
   against the 7 on-disk `configs/*.yaml` files and asserts PASS.
4. **alias-resolutions in evidence** — confirms the audit-observable
   contract: alias resolutions are surfaced in the evidence message
   in `<key>:<resolved_section>.<resolved_key>` format.

### §4.3 Backward-compat preserved

The pre-existing Wf-343C4 test
`test_gate_wf343c4_hyperparams_config_complete_pass_on_fixture`
(PASS branch + FAIL branch) still PASSes — KEY_ALIASES is ADDITIVE.

### §4.4 Overall pre-flight state

```
TOTAL: 68 | PASS: 65 | FAIL: 2
FAIL: ['p3_wf343c4_anchor_freshness_24h', 'p3_wf343c2_per_wave_runtime_check']
```

Hyperparams gate: PASS ✓ (was FAIL; now PASS).

The 2 remaining FAILs (`anchor_freshness_24h`,
`per_wave_runtime_check`) are **pre-existing FAILs not in Wf-345-A1
scope** — `anchor_freshness_24h` is an AGENTS.md §5.4 honest-negative
(stale anchor JSONs > 24 h old by design); `per_wave_runtime_check`
is a separate contract unrelated to the hyperparam config check.

---

## §5 Honest framing (party-twenty-spirit §3.5)

### §5.1 What we MEASURED

- **6/7 YAMLs missing canonical-schema hyperparams** before the fix
  (MEASURED via gate FAIL evidence).
- **7/7 YAMLs PASS after the fix** (MEASURED via gate PASS evidence
  + 4 NEW test PASS).
- **5 YAMLs use aliases** (MEASURED via evidence message
  `alias-resolved` dict):
  - `cfm_per_innovation_ablation_p1.4.yaml` — `batch_size:protocol.batch_size; lr:protocol.lr`
  - `cfm_real_decode_lift_p1.3.yaml` — same
  - `wf124_p3_50epoch_full_sweep_g5.yaml` — `batch_size:protocol.batch_size; lr:protocol.lr; n_layers:model.depth`
  - `wf124_p3_per_innovation_ablation_g6.yaml` — same
  - `wf124_p3_remaining_metals_g4.yaml` — same
- **2 YAMLs use canonical schema**: `default.yaml`, `cfm_full_retrain.yaml`.
- **Wall time**: < 1 s per gate invocation (CPU-only, < 10 ms target).

### §5.2 What we PROJECTED

- Approach (b) preserves backward compat for downstream consumers
  reading `protocol.batch_size`, `protocol.lr`, `model.depth`
  (PROJECTED 1.5× safer than rename per the brief).
- The alias-resolutions dict in the evidence message gives downstream
  consumers an audit-observable signal which YAMLs used which alias
  (PROJECTED sufficient for `per_wave_checklist.py` integration + any
  future `mol-metal-cfm-training` skill that wants to surface the
  resolved values).
- Zero GPU launches performed (CPU-only; gate is in-process).

### §5.3 Honest-negative caveats

1. **KEY_ALIASES is permissive by design** — a future YAML using an
   un-listed alias location would FAIL the gate. The audit-consumer
   path would need an updated KEY_ALIASES entry. This is the
   intended AGENTS.md §5.1 contract: any new alias location must be
   explicitly added (no silent over-permissiveness).

2. **The Wf-343C4 integration test 4 (`EXPECTED_GATE_COUNTS['all'] == 66`)**
   is a pre-existing FAIL — `EXPECTED_GATE_COUNTS['all']` is now 68
   (2 NEW gates added by later workflows). This is unrelated to
   Wf-345-A1; it is a separate contract-drift fix candidate for a
   future Wf.

3. **`p3_wf343c4_anchor_freshness_24h` FAIL** is the AGENTS.md §5.4
   honest-negative pre-existing state — anchor JSONs are stale by
   design (24 h cutoff) and the gate correctly detects this. Not
   in Wf-345-A1 scope.

---

## §6 Cross-references

- `molmetal/scripts/pre_flight_check.py:2940-3131` — the gate
  function + KEY_ALIASES + _resolve_key implementation.
- `molmetal/scripts/pre_flight_check.py:4780-4795` — the gate
  registration in `build_gates('all')` (unchanged; gate still in
  'all'-only partition).
- `molmetal/tests/test_wf345a1_hyperparam_aliases.py` — 4 NEW tests.
- `molmetal/tests/test_wf343c4_integration.py:179-219` — pre-existing
  Wf-343C4 hyperparams test (PASS branch + FAIL branch; still passes
  post-fix, backward-compat verified).
- `configs/cfm_full_retrain.yaml:73` — added `train.batch_size: 16`
  with `# Wf-345-A1` provenance comment.
- `AGENTS.md §5.1` — the standing GPU grant hyperparam contract
  (canonical paths + alias-resolution rationale).
- `TODO/completed/343c4_integration.md` — Wf-343C4 ship summary
  (predecessor; the 4 NEW pre-retrain internal check gates including
  the hyperparams gate).

---

**END OF Wf-345-A1 SHIP SURFACE**
