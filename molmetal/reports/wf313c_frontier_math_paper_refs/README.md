# Wf-313C — Frontier Math Per-Adapter Research-Paper Reference (INDEX)

**Date:** 2026-09-29
**Owner:** Hugo Lee (audit) — ultracode Day 3-5 CPU deliverable
**Trigger:** Wf-313C PART 2 (per TODO-234 §6 Day 3-5 CPU plan)
**Parent doc:** `TODO/pending/234_wf234_frontier_math_audit_integration.md`

This directory holds the per-adapter research-paper reference
documentation for the 4 STANDALONE frontier-math adapters composed via
`molmetal.adapters.frontier_math_integration.FrontierMathIntegration`.

---

## 1. Per-adapter references

| Adapter | File | Paper cite key | Status |
|---|---|---|---|
| Multi-PH pocket features (W2.3) | [`multi_ph_integration_ref.md`](multi_ph_integration_ref.md) | `kim_jung2026multi_ph` | W2.3 CPU smoke shipped |
| SE(3) Lipman solver (Integration 1) | [`se3_lipman_integration_ref.md`](se3_lipman_integration_ref.md) | `mahmood2026sb_lie` | Integration 1 CPU smoke shipped |
| SE(3) SB decoder (W2.1) | [`se3_sb_integration_ref.md`](se3_sb_integration_ref.md) | `mahmood2026sb_lie` | W2.1 CPU smoke shipped |
| Transport relaxation (W2.2) | [`transport_relaxation_integration_ref.md`](transport_relaxation_integration_ref.md) | `jiang_xu_zhang2026sb_transport_relaxation` | W2.2 CPU smoke shipped |

Each ref doc follows the same 5-section template:

1. **Paper citation** — arXiv / DOI / cite key.
2. **API surface** — every public symbol + its type + purpose.
3. **Honest reproduction caveats** — CPU-only, no-modification-of-`flow_matching_lipman`, approximations, defaults.
4. **Drop-in integration points** — where in the main pipeline the adapter slots in.
5. **Cross-references** — test files, sibling adapters, math layer.

---

## 2. Honest framing (per Wf-313C brief)

* **Day 3-5 is CPU smoke.** Full paper-grade integration is the GPU
  portion (Day 7 in Wf-310F §6) and requires user OK per
  `mol-metal-gpu-execution` §0.
* **3 research-only modules are honestly excluded** from the factory
  surface (`bass_existence` / `sb_bass_martingale` /
  `monge_ampere_regularity`).  Their value is the paper §3 method
  reference text — NOT a drop-in loss / feature / velocity field.
  See Wf-305B / Wf-306B / TODO-234 §2.
* **Paper refs are placeholder citations.** Actual paper §3 prose
  text is PAPER_BLOCKED per TODO-84 (no paper work without user
  authorization).  The refs here are cite-key + API + caveats only.
* **Tests in `molmetal/tests/test_wf313c_frontier_math_integration_e2e.py`**
  exercise every adapter via the factory surface — no direct adapter
  import.  CPU-only; <1 s on a single thread.

---

## 3. Cross-doc pointers

| Document | Path |
|---|---|
| TODO-234 audit + plan | `TODO/pending/234_wf234_frontier_math_audit_integration.md` |
| Factory wrapper | `molmetal/adapters/frontier_math_integration.py` |
| Wf-310F Day 1-2 smoke tests | `molmetal/tests/test_wf310f_frontier_math_integration.py` |
| Wf-309B Day 1-2 smoke tests | `molmetal/tests/test_wf309b_frontier_math_integration.py` |
| Wf-313C Day 3-5 e2e tests | `molmetal/tests/test_wf313c_frontier_math_integration_e2e.py` |
| R18 plan (upstream) | `TODO/03_plans/R18_PLAN.md` |
| R18 math migration plan | `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` |
| W2.1 / W2.2 / W2.3 ship surface | `TODO/completed/55_r18_integration_adapters.md` |
| Honest-framing policy | `party-twenty-spirit` §3.5 |

---

## 4. Reproduction contract

```bash
# Run the 8 Day 3-5 paper-grade integration tests
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf313c_frontier_math_integration_e2e.py -v

# Run the 4 Day 1-2 build-only tests
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf310f_frontier_math_integration.py -v
```

Expected: 8 PASSED + 4 PASSED = 12 tests, all under 1 s wall on a
single CPU thread.  No GPU required.

---

**END OF INDEX** — Owner Hugo Lee; MEASURED + Wf-313C Day 3-5 status; last update 2026-09-29.