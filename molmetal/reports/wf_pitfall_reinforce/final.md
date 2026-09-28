# WF-TODO-30 Pitfall Reinforce — Final Report

**Status**: ✅ **DONE** 2026-09-25
**Master plan**: `TODO/pending/30_pitfall_reinforce_plan.md`
**Outcome**: 7 / 7 R12-R17 pitfalls now have CI guard rails that catch
regressions at test time.

---

## 1. TL;DR

The 7 infrastructure-level pitfalls that bit the project between
Round 12 and Round 17 each have a parametrized pytest guard plus a
single-shot runner script.  Every guard was verified end-to-end by:

1. Running on the **good state** (current code) → all 7 PASS.
2. Injecting a **bad state** (deliberately reverting the fix) → the
   corresponding guard FAILS.
3. Restoring the fix → the guard PASSes again.

| Pitfall | Description (one-liner) | Test method |
|---|---|---|
| **P1** | OOV atoms → vocab_mask + 20-element default vocab | `test_p1_oov_atom_vocabulary_is_at_least_20_elements`, `test_p1_oov_skips_optimizer_step` |
| **P2** | SynFlowNet shim TypeError → per-rule try/except | `test_p2_synflownet_per_rule_exception_handler`, `test_p2_swap_smarts_idempotent_on_malformed_input` |
| **P3** | MetalLigandExchange 84-rule matrix + cisplatin seed choice | `test_p3_register_metal_ligand_exchange_rules_returns_at_least_84`, `test_p3_metal_seed_cisplatin_is_a_valid_choice` |
| **P4** | BF16 atom_logits clamp ±50 + FP32 AMP fallback | `test_p4_atom_logits_clamped_to_pm50_in_train_and_sample`, `test_p4_amp_dtype_float32_supported` |
| **P5** | `r4_lambda_only_run` SAFETY_MAX ≥ 10000 floor | `test_p5_r4_lambda_only_run_safety_max_floor_is_10k` |
| **P6** | Reference clone location auto-detect (parent vs clone root) | `test_p6_configure_reference_accepts_both_parent_and_clone_root` |
| **P7** | ModelWrapper keyword-only `t` fallback (`functools.partial` fix) | `test_p7_model_wrapper_handles_keyword_only_t` |

**Test count**: 11 named tests + 7 parametrized dispatchers = 18 test cases
**Skip count (env-related)**: 2 (require numpy/torch)
**Result**: 16 PASSED, 2 SKIPPED, 0 FAILED on the good state.

---

## 2. Pitfall catalogue (R12-R17)

Each pitfall is summarised as:

* **Symptom** — what the user observed at runtime.
* **Root cause** — the underlying bug.
* **Fix** — the patch that prevents recurrence.
* **Guard** — the CI test that fails when the fix is reverted.

### P1 — OOV atom vocabulary

* **Symptom**: training step 1 → `atom_loss = Inf`, `cfm_loss` diverges,
  `decode_ratio = 0/192` on Blackwell.
* **Root cause**: the previous 12-element default vocab excluded Pt
  (and other metallodrug atoms).  When a training molecule contained
  Pt, `vocab_mask` -inf'd the Pt logit slot, `F.cross_entropy`
  produced `Inf` for that row, and the CFM loss diverged.
* **Fix**:
  * `molmetal/adapters/flow_matching_lipman/__init__.py:1840-1842`
    expanded `_atom_vocab` to 20 elements
    (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 21, 22, 33, 34, 35, 39, 40, 53, 72, 78)
    including Pt (78).
  * Defence-in-depth: `train_step` (line ~2056-2089) detects `bad_atoms`
    and **skips** the optimizer step instead of poisoning AdamW with
    `Inf` loss.
* **Guard**:
  * Verifies `_atom_vocab` is ≥ 20 elements AND contains Pt (78).
  * Verifies `bad_atoms` and `SKIPPING optimizer step` strings exist
    in the source.

### P2 — SynFlowNet shim TypeError

* **Symptom**: `SynFlowNetEnvAdapter` fails to load → `n_rules == 0`
  → 0/192 decode.
* **Root cause**: an unparseable SMARTS in `REACTION_RULES` raises
  from `_SFNReaction.__post_init__` (line ~75 of
  `synflownet_env.py`), which aborts `__init__` mid-loop before the
  rule can be registered.
* **Fix**: wrap each rule's template construction in `try / except
  Exception:` (per-rule handlers in `SynFlowNetEnvAdapter.__init__`).
* **Guard**:
  * Regex match for `try: ... self._sfn[name] = _SFNReaction(...)`.
  * Regex match for `try: ... self._rev_sfn[name] = _SFNReaction(...)`
    (with optional pre-step like `rev_sma = _swap_smarts(sma)`).
  * `except Exception` count ≥ 2.

### P3 — MetalLigandExchange 84-rule matrix + cisplatin seed

* **Symptom**: across-pocket singleton attractor — every pocket's
  MCTS root collapses to the same cisplatin seed because the only
  available `MetalLigandExchange` rules are 5 (F2(a) curated subset)
  and `n_distinct` stays at 1.
* **Root cause**: the matrix was not extended to the full
  12 metals × 7 ligands = 84 SMARTS patterns the MCTS can fire on.
* **Fix**:
  * `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py:356`
    ships `METAL_LIGAND_EXCHANGE_MATRIX` (84 entries) AND a
    `register_metal_ligand_exchange_rules()` function that returns
    84 matrix + 17 NEW SMARTS = **101 patterns** when fully wired.
  * `molmetal/scripts/r4_lambda_only_run.py:4398-4419` exposes
    `--metal-seed cisplatin` (plus carboplatin, oxaliplatin,
    nedaplatin, ru_arene, ir_cp_star) as a CLI choice so users can
    opt in to the seeded path.
* **Guard**:
  * Loads `pt_metal_ligand_exchange.py` by file path (bypassing the
    package `__init__` which transitively imports numpy/torch) and
    asserts `METAL_LIGAND_EXCHANGE_MATRIX` has ≥ 84 entries.
  * Asserts `register_metal_ligand_exchange_rules()` returns ≥ 84 rules.
  * Asserts `"cisplatin"` is in the `--metal-seed` argparse choices.

### P4 — BF16 atom_logits overflow

* **Symptom**: untrained Blackwell BF16 matmul occasionally overflows
  `atom_logits` past `+/-1e4`, `F.cross_entropy` blows up to `Inf`,
  CFM loss diverges.
* **Root cause**: BF16 has lower dynamic range than FP32; the EGNN
  velocity head's first matmul can produce out-of-range logits before
  training has stabilised.
* **Fix**:
  * `molmetal/adapters/flow_matching_lipman/__init__.py:2177` clamps
    `atom_logits` to `[-50, 50]` in the **training** path.
  * `molmetal/adapters/flow_matching_lipman/__init__.py:2699` clamps
    `atom_logits` to `[-50, 50]` in the **sampling** path (defence
    in depth).
  * `molmetal/adapters/flow_matching_lipman/amp.py` honours
    `MOLMETAL_CFM_AMP_DTYPE=float32` for an explicit FP32 fallback
    when numerical instability is observed.
* **Guard**:
  * Counts `atom_logits.clamp(min=-50.0, max=50.0)` occurrences
    (≥ 2 = both paths clamped).
  * Sets `MOLMETAL_CFM_AMP_DTYPE=float32` and constructs a
    `CFMAMPContext(dtype=torch.float32)` to verify the fallback
    path is wired (skipped when torch is not installed).

### P5 — `r4_lambda_only_run` 1K-step is lucky-only

* **Symptom**: a 1K-step smoke run produces a good-looking loss curve
  but the 10K-step production run explodes.
* **Root cause**: the historical `--n-simulations` 100-cap silently
  clamped the flag; production-scale runs were never tested.
* **Fix**: `molmetal/scripts/r4_lambda_only_run.py:5113` introduces
  `SAFETY_MAX = 10000` so a typoed `--n-simulations 1000000` cannot
  lock the GPU on a developer box, while the flag is honoured
  verbatim up to the safety ceiling.
* **Guard**:
  * Regex match for `SAFETY_MAX = <int>` and asserts the value
    is ≥ 10000.

### P6 — Reference clone location mismatch

* **Symptom**: importing `LipmanFlowMatchingAdapter` raises
  `FileNotFoundError: cloned flow_matching not found at ...` because
  the caller passes the parent directory but the loader expected the
  clone root.
* **Root cause**: legacy convention is `references/flow_matching/`
  (caller passes the parent); canonical convention is
  `references/flow_matching/__init__.py` (caller passes the clone
  root).  The previous loader assumed one convention only.
* **Fix**: `molmetal/adapters/flow_matching_lipman/_reference_loader.py:82-86`
  auto-detects by checking `(ref / "__init__.py").is_file() and
  ref.name == "flow_matching"`.
* **Guard**:
  * Loads `_reference_loader.py` by file path.
  * Asserts `ref.name == "flow_matching"` appears in the
    `configure_reference()` source (auto-detect branch).

### P7 — `functools.partial` + keyword-only `t`

* **Symptom**: `ModelWrapper(functools.partial(model, t=0.5))`
  raises `TypeError` because partial binds `t` as a kwarg but the
  wrapper calls it positionally.
* **Root cause**: some velocity fields take `t` as keyword-only
  (`def forward(self, x, *, t): ...`).  `functools.partial` cannot
  satisfy a keyword-only argument via positional binding.
* **Fix**: `molmetal/adapters/flow_matching_lipman/_reference/utils/__init__.py:44-63`
  `ModelWrapper.__call__` tries `(x, t)` first, falls back to
  `(x, t=t)`, then `(x, t=t, **extras)`, then `(x, t, **extras)`.
* **Guard**:
  * Loads the shim by file path.
  * Calls wrapped legacy positional VF → asserts `("X", 0.5)`.
  * Calls wrapped keyword-only `t` VF → asserts `("kwonly", "Y", 0.7)`.
  * Calls wrapped `functools.partial(vf, t=0.9)` → asserts
    `("full", "Z", 0.9, None)`.

---

## 3. Regression-verification matrix

For every pitfall the guard was verified against a synthetic
regression.  Each row shows: (a) the regression introduced; (b)
the resulting pytest outcome; (c) the restore.

| Pitfall | Regression injected | pytest outcome |
|---|---|---|
| P1 | shrink vocab to 12 elements | FAIL `len(vocab) >= 20` |
| P1 | swap Pt(78) for Au(79) in 20-elem vocab | FAIL `78 in vocab` |
| P2 | remove reverse-template try/except | FAIL `rev_match is not None` |
| P3 | shrink matrix to 1 rule | FAIL `len(matrix) >= 84` |
| P4 | remove sampling-path clamp | FAIL `n >= 2` clamps |
| P5 | `SAFETY_MAX = 100` | FAIL `val >= 10000` |
| P6 | corrupt auto-detect branch | FAIL `ref.name == "flow_matching"` |
| P7 | remove except TypeError pass in branch 1 | FAIL kw-only VF raises TypeError |

Restoration sequence (in order): each injected regression was
reverted via `cp /tmp/<backup>` and the full test suite was
re-run to confirm 16 / 16 PASS.

---

## 4. How to run

### Direct pytest (canonical)

```bash
cd /home/hugo/codes/mol-metal
uv run pytest molmetal/tests/test_pitfall_guards.py -v
```

### Single-shot runner (ultracode dispatch entry point)

```bash
cd /home/hugo/codes/mol-metal
uv run python molmetal/scripts/wf_pitfall_guard/check_all.py
```

Output (ASCII table):

```
Pitfall  Description                                                   Status
-------------------------------------------------------------------------------
P1       OOV atoms → vocab_mask + 20-element default vocab             PASS
P2       SynFlowNet shim TypeError → per-rule try/except               PASS
P3       MetalLigandExchange 84-rule matrix + cisplatin seed default   PASS
P4       BF16 atom_logits clamp ±50 + FP32 AMP fallback                PASS
P5       r4_lambda_only_run SAFETY_MAX ≥ 10000 floor                   PASS
P6       Reference clone location auto-detect (parent vs clone root)   PASS
P7       ModelWrapper keyword-only t fallback (functools.partial fix)  PASS
-------------------------------------------------------------------------------
PASS=7 FAIL=0 SKIP=0
```

### JSON output (for harness ingestion)

```bash
uv run python molmetal/scripts/wf_pitfall_guard/check_all.py --json
```

The runner returns exit code 0 when all 7 guards pass; 1 when any
guard fails; 2 when pytest itself crashes (env missing).

---

## 5. Output files (this workflow)

* `molmetal/tests/test_pitfall_guards.py` — 11 named tests + 7
  parametrized dispatchers (18 test cases total; 16 PASS, 2 SKIP in
  the bare-bones test venv).
* `molmetal/scripts/wf_pitfall_guard/check_all.py` — single-shot
  runner with ASCII + JSON output.
* `molmetal/reports/wf_pitfall_reinforce/final.md` — this report.

---

## 6. Honest framing

* **Test-env skips**: 2 tests skip when numpy/torch are not installed
  (`test_p2_swap_smarts_idempotent_on_malformed_input` and
  `test_p4_amp_dtype_float32_supported`).  These env-gates are
  intentional — the guards still verify the *source-level*
  invariants (string literals, regex matches), and skip only the
  runtime smoke checks that need numpy/torch.
* **Source-only assertions**: the majority of the guards are
  *static* (regex / source-file lookups) rather than runtime
  smoke.  This is intentional: static guards run in < 1 second
  and don't require GPU / ROCm / network.  The runtime checks
  that DO require heavy deps (numpy / torch / RDKit) are guarded
  by `pytest.skip` so they don't false-fail on a minimal venv.
* **No GPU retrain claimed**: this is a CPU-only workflow.  No
  CFM training was executed; the guards merely verify the
  *plumbing* (vocab literals, clamp call counts, exception
  handlers, module attributes) is in place.
* **P3 metal-seed default**: the canonical `--metal-seed` default
  is `None` (legacy behaviour preserved; the search falls back
  to the pocket reference).  The guard verifies that `cisplatin`
  is **available** as a choice, not that it is the default —
  changing the default would break existing pipelines that pass
  `--metal-seed none` explicitly.

---

## 7. Reference

* TODO-30 plan: `TODO/pending/30_pitfall_reinforce_plan.md`
* Pitfall master audit: `molmetal/reports/wf_pitfall_audit/MASTER.md`
* Per-layer audits: `molmetal/reports/wf_pitfall_audit/p{1..6}_*.md`

End of report.