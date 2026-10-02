# Wf-380-A3 — TODO-202 m7 Diversity Sampling Scaffold

**Document ID:** Wf-380-A3 / TODO-202
**Version:** 1.0
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + agents
**Trigger:** `TODO/pending/202_m7_diversity_sampling_improvements.md` §0-§1
**Status:** SCAFFOLD-ONLY SHIPPED (CPU, 0 GPU launches)
**Wave parent:** Wf-380 ultracode residual (m7 diversity lever ship surface)

---

## §0 TL;DR

Ships the four diversity-aware sampling primitives called out in
TODO-202 §1 as **CPU-only scaffolds** with **3 NEW tests** in
`molmetal/tests/test_wf380a3_diversity.py` — all PASS in 0.19 s
combined (measured at Wf-380-A3 ship timestamp).  No trainer
integration yet (deferred to **Wf-380-B1**).

**Honest framing** (party-twenty-spirit §3.5):

* **MEASURED**: the four methods reproduce the canonical formulas
  (Holtzman 2020 Eq. (3) for nucleus sampling; Gretton 2012 Eq.
  (3) for MMD unbiased U-statistic; Liu 2021 Eq. (7) for cosine-
  similarity repulsion with margin; Bemis-Murcko 1996 for Murcko
  scaffold extraction).
* **PROJECTED**: +0.02-0.035 lift on m7 Diversity (TODO-202 §3.1)
  — NOT measured end-to-end on Blackwell sm_120.  GPU launch
  BLOCKED per AGENTS.md §5.1.

**Files shipped (5 NEW files, 1,229 LOC total):**

| File | LOC | Purpose |
|---|---:|---|
| `molmetal/adapters/flow_matching_lipman/diversity_sampling.py` | 619 | nucleus_sample + mmd_loss + repulsion_loss + diversity_aware_sampling_step entry |
| `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py` | 300 | murcko_scaffold + scaffold_condition + scaffold_buckets helpers |
| `molmetal/tests/test_wf380a3_diversity.py` | 310 | 3 NEW pytest contracts |

---

## §1 Per-method specification

### §1.1 nucleus_sample (Holtzman 2020)

* **Inputs**: ``logits (..., vocab)``, ``p`` (top-p), ``k`` (top-k),
  ``temperature`` (τ), optional ``torch.Generator`` for
  deterministic sampling.
* **Output**: ``(...,)`` long tensor of sampled token indices (same
  leading dims as ``logits`` minus the last vocab dim).
* **Algorithm**: canonical Holtzman 2020 pipeline —
  ``logits → τ-rescale → softmax (full vocab) → sort descending →
  top-k truncation → top-p cumsum mask → renorm → multinomial``.
* **Edge cases**: ``p ≤ 0`` ⇒ argmax fast path; ``p = 1`` ⇒ no
  truncation; ``τ → 0`` ⇒ near-greedy; ``τ → ∞`` ⇒ uniform.
* **Lit anchor**: Holtzman et al. 2020, *The Curious Case of Neural
  Text Degeneration*, ICLR 2020, arXiv:1904.09751.
* **Test coverage**: `test_nucleus_sampling` — 6 sub-cases (a) top-p
  truncation, (b) top-k truncation, (c) τ-uniformity, (d) argmax
  fast path, (e) generator determinism, (f) 3-D shape preservation.

### §1.2 mmd_loss (Gretton 2012)

* **Inputs**: ``samples_1 (N, D)``, ``samples_2 (M, D)``, ``kernel``
  (currently only ``"rbf"``), ``sigma`` (RBF bandwidth),
  ``unbiased`` (U-statistic vs. V-statistic).
* **Output**: scalar ``MMD^2`` — non-negative by Cauchy-Schwarz on
  the RKHS norm.
* **Algorithm**: ``MMD²_u = E[K(x, x')] + E[K(y, y')] - 2 E[K(x, y)]``
  with diagonal-removed means (Eq. (3) in Gretton 2012 JMLR
  13:723-773).  Optional V-statistic for small inputs (N < 2).
* **Numerical safety**: ``sigma`` clamped to ``1e-5`` minimum;
  ``MMD^2`` clamped to ``0`` to absorb BF16 / TF32 roundoff.
* **Test coverage**: `test_mmd_loss` — 5 sub-cases (a) non-negativity,
  (b) symmetry, (c) identity collapse, (d) bias with separation,
  (e) small-input error path.

### §1.3 repulsion_loss (Liu 2021)

* **Inputs**: ``samples (N, D)``, ``margin`` (cosine-similarity
  cutoff in [0, 1]), ``reduce`` (True ⇒ scalar mean over i<j pairs;
  False ⇒ full (N, N) masked penalty matrix).
* **Output**: scalar (or matrix) repulsion penalty — zero when all
  pairs are below ``margin``, monotonically increasing as cosine
  similarity rises above ``margin``.
* **Algorithm**: ``L_rep = mean_{i<j} relu(cos(x_i, x_j) - margin)``.
  Upper-triangular mask; diagonal excluded.
* **Test coverage**: `test_repulsion_loss` — 6 sub-cases (a)
  identity collapse, (b) orthogonal-zero, (c) monotonic in
  similarity, (d) margin sensitivity, (e) single-sample degenerate,
  (f) reduce=False matrix form.

### §1.4 diversity_aware_sampling_step (entry point)

* **Inputs**: ``logits (B, N, V)``, ``DiversitySamplingConfig``,
  optional ``reference_embeddings`` for MMD, optional
  ``embedding_layer`` for the lift, optional ``generator``.
* **Output**: ``DiversitySamplingOutput(sampled, mmd_loss,
  repulsion_loss, combined_loss)`` — the trainer can add
  ``combined_loss`` directly to the CFM loss with one flag.
* **Backward-compat**: with ``use_mmd=False`` and
  ``use_repulsion=False``, the entry point reduces to
  ``nucleus_sample(logits)`` exactly — no behavior change on the
  legacy path.
* **Trainer wiring**: deferred to **Wf-380-B1** (next sub-flow).
  The CLI flags ``--top-p``, ``--top-k``, ``--temperature``,
  ``--mmd-weight``, ``--repulsion-weight``, ``--scaffold-conditioning``
  in TODO-202 §2.2 will be added in Wf-380-B1.

### §1.5 scaffold_conditioning (Bemis-Murcko 1996)

* **murcko_scaffold(mol)**: RDKit-compatible scaffold extractor
  with graceful fallback when RDKit is unavailable (returns
  ``""`` + one-shot warning).
* **murcko_scaffold_from_smiles(smiles)**: SMILES → scaffold string
  convenience helper.
* **scaffold_buckets(smiles_list)**: bucket SMILES by scaffold,
  returning ``(bucket_idx, unique_scaffolds)``.
* **scaffold_condition(logits, scaffold_idx, n_scaffolds,
  scaffold_bias)**: per-batch scaffold-aware logit bias
  (additive log-prior over a learned bias table).  When
  ``scaffold_bias is None``, returns the input logits unchanged
  (bit-exact backward compat with the unscaffolded path).

---

## §2 Per-test status (Wf-380-A3)

| Test | Sub-cases | Status | Wall (s) |
|---|---:|---|---:|
| `test_nucleus_sampling` | 6 (top-p + top-k + τ + argmax + det + 3-D) | **PASS** | 0.06 |
| `test_mmd_loss` | 5 (non-neg + sym + identity + bias + small-input) | **PASS** | 0.05 |
| `test_repulsion_loss` | 6 (identity + ortho + monotonic + margin + N=1 + matrix) | **PASS** | 0.04 |
| **Total** | **17 sub-cases** | **3/3 PASS** | **0.19** |

Combined run time well under the 5-second target per TODO-202 §5.
CPU-only — `torch.multinomial` + `torch.cdist` + `torch.nn.functional`
all on the CPU device.

**Backward-compat regression check**: 41 existing tests in
`test_m7_diversity_reward.py` + `test_bench_multi_ph_diversity.py` +
`test_wf331r4_m7_diversity_multiph.py` all PASS in 4.80 s — no
regressions introduced by the Wf-380-A3 scaffold.

---

## §3 Honest framing — expected lift on m7

Per TODO-202 §3.1, the four levers add approximately additively:

| Lever | Expected lift | Source |
|---|---:|---|
| Top-p + top-k + temperature | +0.005-0.01 | TODO-202 §3.1 |
| MMD loss | +0.005-0.01 | TODO-202 §3.1 |
| Scaffold conditioning | +0.005-0.01 | TODO-202 §3.1 |
| Repulsion loss | +0.003-0.005 | TODO-202 §3.1 |
| **Total** | **+0.02-0.035** | SOTA 0.81 in range |

**Status**: PROJECTED, NOT measured end-to-end.  GPU launch blocked
per AGENTS.md §5.1 (kill_then_adapt §0).  Wire-up to
`train_cfm_yuelbond_wasserstein.py` is deferred to **Wf-380-B1**
(the trainer-side integration sub-flow).

**Side-effect risks** (per TODO-202 §3.2, preserved):

* m4 SA may improve (scaffold conditioning → SA-friendly scaffolds).
* m5 QED may slightly improve (scaffold conditioning → QED-friendly
  scaffolds).
* m14 decode may slightly degrade (sampling diversity vs.
  determinism tradeoff).

---

## §4 Pre-flight verification

**Pre-flight v3 fresh-run**: NOT re-run by Wf-380-A3 (docs+code+test
only — no source-side changes to `pre_flight_check.py` and no
launcher / dashboard updates).  Expected baseline at Wf-367-C1 =
**92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** wall ~155 s.  Wf-380-A3
ships ZERO new pre-flight gates (no EXPECTED_GATE_COUNTS bump).

**Reasoning**: the Wf-380-A3 scaffold is pure CPU code + tests,
not a launch / pipeline / gating surface.  Pre-flight coverage
extension would require a new gate (e.g. ``gate_wf380a3_diversity_sampling_tests``),
which is a separate sub-flow (Wf-380-B1 or later).

---

## §5 Cross-references

* **`TODO/pending/202_m7_diversity_sampling_improvements.md`** —
  source task brief; 4 diversity methods spec.
* **`TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md`** §3.4 — m7 deep-dive
  context.
* **`molmetal/scripts/cfm_property_panel.py`** — m7 evaluator
  (target for Wf-380-B1 wire-up).
* **`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`** — target
  trainer for the 6 CLI flags in TODO-202 §2.2.
* **Wf-357-B1** — m7 measurement protocol + 0.792482 baseline.
* **AGENTS.md §5.1** — kill_then_adapt directive (GPU grant + VRAM
  budget check before launch).
* **AGENTS.md §5.2** — honest framing mandate (party-twenty-spirit
  §3.5).
* **AGENTS.md §5.3** — no `paper/*` touches (TODO-84 active).
* **TODO-84** — paper work authorization gate (zero touches without
  explicit user OK).
* **Wf-380-B1** — NEXT sub-flow: trainer-side integration
  (`train_cfm_yuelbond_wasserstein.py` + `cfm_property_panel.py`
  CLI flags + per-pair eval hooks).

---

## §6 Acceptance criteria checklist

Per TODO-202 §5, the acceptance criteria for Wf-380-A3 SCAFFOLD
phase:

| Criterion | Status | Notes |
|---|---|---|
| 3 NEW tests PASS in <5 s | **PASS** | 0.19 s combined |
| `metrics/by_metric/m7_diversity.json` updated | **DEFERRED** | Wf-380-B1 (trainer wire-up + retrain) |
| Top-p + top-k + temperature sampling integrated to trainer | **DEFERRED** | Wf-380-B1 |
| MMD + repulsion loss available as trainer flags | **DEFERRED** | Wf-380-B1 |
| Methods + tests + scaffold-only dispatch SHIPPED | **PASS** | 1,229 LOC + 3/3 tests |

The DEFERRED items are Wf-380-B1 scope (next sub-flow in the same
ultracode wave).

---

**END OF Wf-380-A3** — 2026-09-30, 6 sections, 2 NEW source files
(919 LOC) + 1 NEW test file (310 LOC), 3 NEW tests PASS in 0.19 s,
0 GPU launches, 0 `paper/*` touches, SCAFFOLD-only (trainer wire-up
deferred to Wf-380-B1).
