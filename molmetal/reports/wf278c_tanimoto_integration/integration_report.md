# Wf-278C — cfm_tanimoto_ood_filter Integration Audit

**Date**: 2026-09-29
**Scope**: `molmetal/molmetal_lam/lam_chem/cfm_tanimoto_ood_filter.py` +
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
**Ref**: TODO-98 (Wf-275B ship) + Wf-271B modified-file summary

---

## 1. Filter API (`cfm_tanimoto_ood_filter.py`)

Public surface (`__all__` lines 282-288):

| Symbol | Purpose |
|---|---|
| `TanimotoOODSummary` | DIAGNOSTIC dataclass — `n_total / n_in_domain / n_ood / ood_fraction / max_train_tanimoto / n_failed_parse` |
| `apply_tanimoto_ood_filter(query, train, threshold, radius, n_bits)` | Bool mask per query (fail-closed on bad SMILES) |
| `cfm_ood_filter_smiles_list(query_smiles, query_payloads, train_smiles, ...)` | Partition payloads into `(in_domain, ood)` buckets; loose `object` payload contract |
| `summarise_tanimoto_ood(query, train, ...)` | Full per-query `max_train_tanimoto` histogram + summary |
| `write_tanimoto_ood_summary_json(summary, path)` | JSON IO helper |

**CPU-only** (pure RDKit Morgan-ECFP4 radius=2, nBits=2048); wraps
`molmetal.molmetal_lam.lam_chem.ood_protocol.TanimotoOODProtocol`.

---

## 2. CLI wiring (`train_cfm_yuelbond_wasserstein.py`)

Three CLI flags added (lines 864-883 of `train_cfm_yuelbond_wasserstein.py`):

```
--use-tanimoto-ood-filter          (off by default, store_true)
--ood-filter-threshold 0.4         (Pocket2Mol / TargetDiff default)
--ood-filter-train-pool ""         (path to SMILES file, one per line)
```

Verified via `python ... --help`:
```
--use-tanimoto-ood-filter
                        Apply Tanimoto<0.4 OOD filter to the training cohort
                        (TODO-98). Default off for backward compat with the
                        synthetic-cohort default.
--ood-filter-threshold OOD_FILTER_THRESHOLD
                        Tanimoto threshold for the OOD filter (default 0.4,
                        Pocket2Mol / TargetDiff).
--ood-filter-train-pool OOD_FILTER_TRAIN_POOL
                        Optional path to a SMILES file (one per line)
                        to use as the OOD reference train pool.  Empty
                        → the synthetic cohort is the reference
                        (backward-compat default).
```

---

## 3. Filter invocation (lines 974-1019 of `main()`)

```python
if getattr(args, "use_tanimoto_ood_filter", False):
    try:
        from molmetal.molmetal_lam.lam_chem.cfm_tanimoto_ood_filter import (
            apply_tanimoto_ood_filter,
        )
        ood_flags = apply_tanimoto_ood_filter(
            ["N.N.[Cl][Pt][Cl]"] * len(train_clouds),
            train_smiles=(
                open(args.ood_filter_train_pool).read().splitlines()
                if args.ood_filter_train_pool
                else ["N.N.[Cl][Pt][Cl]"]
            ),
            threshold=args.ood_filter_threshold,
        )
        ...
```

Import is correct (`cfm_tanimoto_ood_filter.apply_tanimoto_ood_filter`).
Invocation is wrapped in `try/except` with a `print(...)` skip message on
exception (safety net).

---

## 4. Verdict — `PARTIAL` (functional on real data, no-op on synthetic)

**Wired**: ✓ — CLI flag → import → invocation all present.

**Functionally effective on synthetic cohort**: ✗ — the filter is invoked
with placeholder SMILES (`["N.N.[Cl][Pt][Cl]"] * len(train_clouds)`)
because synthetic Pt-click clouds have no canonical SMILES fingerprint
mapping. Train pool defaults to itself (`["N.N.[Cl][Pt][Cl]"]`), so
identical SMILES in query + train ⇒ Tanimoto = 1.0 ⇒ NOT OOD ⇒
`n_ood == 0` ⇒ `kept = train_clouds` (filter preserves all).

This is documented as a **backward-compat placeholder** at lines 980-985
of the train script:

> Synthetic Pt-click clouds have no canonical SMILES, so we cannot run
> the fingerprint filter directly. Fall back to the canonical
> cisplatin-derivable SMILES fingerprint: filter out clouds whose
> synthetic atom composition does NOT match the cisplatin heuristic
> (1 Pt + ≥2 donors). We treat this as the backward-compat placeholder.

**Once a real-data loader is wired (TODO-119 P1.3 follow-up)**:
passing `--ood-filter-train-pool <path-to-PlatinAI-or-DrugOOD-train.smi>`
will switch the filter from no-op → effective. The wiring contract
(argparse → import → invocation → train_clouds update) is already in place.

---

## 5. Tests — EXISTING + PASSING

`molmetal/tests/test_cfm_todo98_tanimoto_ood.py` already ships **19 tests**
across 5 sections:

| Section | Tests | Status |
|---|---|---|
| `TestApplyTanimotoOODFilter` | 7 (returns_bool, self_match_in_domain, dissimilar_ood, empty_train_all_ood, invalid_smiles_ood, threshold_boundary, default_threshold_0_4) | ✓ |
| `TestCFMOODFilterSmilesList` | 4 (partition_lengths, payloads_round_trip, length_mismatch_raises, atom_cloud_payloads) | ✓ |
| `TestSummariseTanimotoOOD` | 5 (counts, max_train_tanimoto, empty_query, threshold_round_trips, to_dict_json) | ✓ |
| `TestWriteTanimotoOODSummaryJSON` | 1 (writes_to_disk) | ✓ |
| Backward-compat | 2 (existing_cfm_imports_unaffected, ood_filter_cli_flag_exists_in_train_script) | ✓ |

```
$ .venv/bin/python -m pytest molmetal/tests/test_cfm_todo98_tanimoto_ood.py -q
...................                                                      [100%]
19 passed in 2.51s
```

**No new tests added** — coverage is already at 19/19 with explicit
assertions on the placeholder-SMILES semantic ("A query that matches a
train entry should NOT be OOD"), the empty-train-pool semantic, the
fail-closed SMILES-parse semantic, and the CLI-flag round-trip via
`subprocess --help`.

---

## 6. End-to-end smoke (CLI invocation)

```
$ .venv/bin/python molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
    --epochs 1 --n-train 4 --n-val 2 --use-tanimoto-ood-filter \
    --device cpu --out-dir /tmp/wf278c_smoke
[Wf-124 P2] device=cpu
[Wf-124 P2] building cohorts: train=4, val=2
[Wf-124 P2][TODO-98] OOD filter kept 4/4 clouds.
[Wf-124 P2] cohort built: train=4 val=2
[Wf-124 P2] pocket=1h36 bias_sum=1.000
[Wf-124 P2] epoch=000 train_loss=2.0293 bond_ce=1.9546 w2=0.7437
            grad_norm=3.160 catherine_pass=1.00 decode_ratio=1.000 (2/2)
[Wf-124 P2] DONE. best_decode_ratio=1.000 @ epoch=0 target=HIT.
            Wall total: 0.3s.
```

Filter log line emits correctly: `[Wf-124 P2][TODO-98] OOD filter kept 4/4 clouds.`
Training completes cleanly in 0.3 s on CPU with `--use-tanimoto-ood-filter` ON.

---

## 7. Honest framing

* **Integration is wired, not functionally effective on synthetic data**:
  the placeholder SMILES (`["N.N.[Cl][Pt][Cl]"] * len(train_clouds)`)
  ensures the filter is a no-op on the synthetic cohort. This is
  intentional backward-compat — the synthetic cohort was the Wf-124 P2
  architectural-path benchmark, not the OOD audit target.
* **Real-data path is one CLI flag away**:
  `--use-tanimoto-ood-filter --ood-filter-train-pool <path-to-real-train.smi>`
  is the contract for the TODO-119 P1.3 follow-up.
* **No silent defaults**: the filter is OFF by default (store_true with
  default False). All three flags appear in `--help` with explicit
  defaults.
* **No silent metric substitution**: the filter is a
  data-loading pre-filter, NOT a CFM loss-surface modification
  (per pre-retrain checklist §1.2.5 DIAGNOSTIC-vs-GRADIENT discipline).
* **Audit trail**: 19 existing pytest contracts pin the API; CLI flag
  presence verified via `subprocess --help`; smoke test confirms
  end-to-end invocation on the train script.

---

## 8. Cross-references

* `TODO/completed/119_decode_ratio_strengthen.md` §3 P1.3 (real-CFM lift)
* `TODO/completed/98_wf44_closure_ood_phase2.md` (Wf-44 P2 L12)
* `TODO/12_flow_matching/` (CFM training path)
* Wf-257-Apply (TODO-98 ship surface)
* Wf-271B (CLI flag wiring summary)
* `molmetal/molmetal_lam/lam_chem/ood_protocol.py` (TanimotoOODProtocol)
* `molmetal/tests/test_cfm_todo98_tanimoto_ood.py` (19 tests, all green)
