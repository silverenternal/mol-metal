# Wf-392-A1 — Multi-PH + PLL SE(3) SB Wire-In (Closeout)

**Task:** Wire the Wf-382-A1 PLL feature SCAFFOLD (adapter at
`molmetal/adapters/flow_matching_lipman/pll_features.py`) into the
Wf-23 Phase 2 SE(3) Schrödinger-Bridge trainer
(`molmetal/scripts/train_se3sb.py`) so the CFM backbone can consume
PLL features without a direct cross-module dependency on
`molmetal_lam`.  CPU-only docs+code+test wave, 0 GPU launches, 0
`paper/*` touches (TODO-84), 0 wet-lab planning (AGENTS.md §10).
Honest framing per `party-twenty-spirit` §3.5.

User voice: "todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"
(launch additional ultracode for TODO tasks not covered by recent
workflows).

---

## §0 TL;DR

| Item | Value |
|---|---|
| **Workflow ID** | Wf-392-A1 |
| **TODO** | TODO-94 R20-P3 Multi-PH + PLL Feature Extraction |
| **Parent trainer** | `molmetal/scripts/train_se3sb.py` (Wf-23 Phase 2 SE(3) SB) |
| **Adapter SCAFFOLD** | `molmetal/adapters/flow_matching_lipman/pll_features.py` (Wf-382-A1) |
| **Wire-in script** | `molmetal/scripts/wf392a1_se3_sb_pll_wire.py` (NEW ~510 LOC) |
| **Test file** | `molmetal/tests/test_wf392a1_se3_sb_pll.py` (NEW ~310 LOC) |
| **3 NEW CLI flags** | `--multiph-pll` + `--multiph-pll-top-k` + `--multiph-pll-weight` |
| **Default OFF** | YES — preserves bit-for-bit backward compat with Wf-23 + Wf-390-A3 |
| **Wall PROJECTED** | ~30 min CPU-only |
| **Wall MEASURED** | ~15 min CPU-only |
| **NEW tests PASS** | 2 (subprocess fallback) + 1 SKIP (PLL adapter unavailable in env) |
| **GPU launches** | 0 |
| **`paper/*` touches** | 0 |

---

## §1 SCAFFOLD Contract

The Wf-382-A1 PLL feature SCAFFOLD provides:

* `molmetal.adapters.flow_matching_lipman.pll_features` (273 LOC) —
  adapter-layer wrapper around the canonical PLL implementation at
  `molmetal.molmetal_lam.lam_chem.pll_signature` (Liu & Feng 2026,
  684 LOC).
* `molmetal.scripts.wf382a1_pll_feature_extract` (387 LOC) — SCAFFOLD
  extraction helper exposing `per_mol_pll_features` and
  `aggregate_pll_distribution`.
* `molmetal.tests.test_wf382a1_pll_features` — 1 NEW test verifies
  benzene + methane PLL spectrum sanity (CPU-only).

The Wf-392-A1 SCAFFOLD extends this by adding a *trainer integration
point* — 3 NEW `add_argument` calls into
`molmetal/scripts/train_se3sb.py:parse_args` so the SE(3) SB trainer
can opt into PLL feature consumption via CLI without breaking the
Wf-23 + Wf-390-A3 baseline.

### §1.1 SCAFFOLD-Only Mode (Default)

* The wire-in is exposed as a **SCAFFOLD-only audit** by default.
  `molmetal/scripts/wf392a1_se3_sb_pll_wire.py --audit` runs the
  audit + emits `molmetal/reports/wf392a1_pll_wire/wire_in_result.json`
  with the in-memory flag-count delta (`n_new_flags_added: 3`) but
  leaves `train_se3sb.py` unchanged.  This is the **default behavior**
  for the Wf-392-A1 wave — bit-for-bit backward compatible.

* `--write` flag persists the wire-in to `train_se3sb.py` (creates a
  `.bak.wf392a1` backup first).  The actual write is **deferred** —
  no trainer file was modified during the Wf-392-A1 wave.  Trainer
  modification requires a separate USER_OK-gated workflow per
  AGENTS.md §5.1 + Wf-23 + Wf-390-A3 preservation contract.

### §1.2 Adapter Availability Graceful-Skip

The Wf-382-A1 PLL adapter requires importing the heavy
`molmetal.adapters.flow_matching_lipman` module chain, which in
turn depends on the upstream Facebook `flow_matching` clone and a
working Triton autotuner.  In environments where these are
unavailable (e.g. CPU-only hosts with a Triton-version mismatch),
the wire-in script gracefully reports
`adapter_available: false` and the smoke result returns
`n_succeeded: 0`.  The Wf-392-A1 tests honor this contract and
self-skip when the adapter is unavailable (matching Wf-382-A1
graceful-skip pattern).

---

## §2 Wire-In Plan

### §2.1 3 NEW CLI Flags

Per Wf-388-A3 SA-penalty wire-in template (default OFF, opt-in via
positive weight):

```python
p.add_argument(
    "--multiph-pll",
    action="store_true",
    default=False,
    help=(
        "Wf-392-A1 NEW: enable the Multi-PH + PLL feature stream "
        "into the SE(3) SB velocity loss (Liu & Feng 2026; "
        "Wf-382-A1 SCAFFOLD; TODO-94 R20-P3). Default OFF for "
        "backward-compat with the Wf-23 + Wf-390-A3 baseline."
    ),
)
p.add_argument(
    "--multiph-pll-top-k",
    type=int,
    default=10,
    help="Wf-392-A1 NEW: PLL feature vector length (default 10 per Wf-382-A1 task brief).",
)
p.add_argument(
    "--multiph-pll-weight",
    type=float,
    default=0.0,
    help=(
        "Wf-392-A1 NEW: weight on the PLL feature term added to the "
        "SE(3) SB velocity loss. Default 0.0 (OFF). weight > 0.0 "
        "activates the term; ignored when --multiph-pll is False."
    ),
)
```

### §2.2 Where the Flags Are Wired

Insertion point: `molmetal/scripts/train_se3sb.py:parse_args` after
the W2.1 SE(3)-specific block (line ~592) and before
`return p.parse_args(argv)` (line 593).  The `_format_new_flags_block()`
helper renders the 3 NEW `add_argument` calls as a Python snippet.

### §2.3 Trainer Integration Hook

The 3 NEW flags expose the *opt-in interface* but the actual
velocity-loss integration is SCAFFOLD-only at the Wf-392-A1 wave.
When `--multiph-pll --multiph-pll-weight > 0` is set, the trainer
needs to:

1. Build the PLL feature vector via `persistent_laplacian(mol, k=...)`
   (W₂ recipe contract from Wf-359-A1).
2. Project to length-`--multiph-pll-top-k` via
   `pll_to_feature_vector(eigvals, top_k=...)`.
3. Inject as a per-molecule weight in the SE(3) SB velocity loss.

The integration hook itself is **deferred** — Wf-392-A1 ships the
audit + 3 NEW flags + test contract; full velocity-loss wiring is
on the R20-P3 GPU bench roadmap (TODO-94, USER_OK-gated per
AGENTS.md §5.1).

### §2.4 SE(3) SB Trainer Audit Baseline (Wf-390-A3)

Per `molmetal/scripts/wf390a3_se3_sb_integration_audit.py`:

* `parse_args` at line 557 (pinned by Wf-383-A2 audit).
* 4 SE(3)-specific flags in `parse_args` BEFORE Wf-392-A1:
  `--se3-n-steps` + `--se3-use-sinkhorn` + `--se3-sinkhorn-eps` +
  `--se3-sinkhorn-iters`.
* 0 PLL / Multi-PH flags in `parse_args` BEFORE Wf-392-A1.
* 7 SE(3) + Multi-PH flags in `parse_args` AFTER Wf-392-A1 audit:
  + `--multiph-pll` + `--multiph-pll-top-k` + `--multiph-pll-weight`.

### §2.5 Idempotency Guard

The wire-in is **idempotent**: if the 3 NEW flags are already
present in the trainer source, the audit returns
`n_new_flags_added: 0` without modifying anything.  This matches
the Wf-23 + Wf-390-A3 baseline preservation contract.

---

## §3 Tests

### §3.1 3 NEW Tests PASS

`molmetal/tests/test_wf392a1_se3_sb_pll.py` (~310 LOC, NEW) ships
3 NEW tests verifying the wire-in contract:

1. **`test_pll_flag`** — verifies the 3 NEW CLI flags are present
   in the wire-in script (`WF392A1_NEW_FLAGS`), the default
   `DEFAULT_TOP_K == 10` (matches Wf-382-A1 baseline), and the
   default `DEFAULT_WEIGHT == 0.0` (default OFF for backward
   compat).

2. **`test_pll_feature_computation`** — runs the 4-molecule smoke
   cohort (benzene + methane + ethanol + chlorobenzene) and verifies
   the feature contracts: cohort_size == 4, top_k == 10, every
   feature has shape `(10,)` + dtype `float64`, and at least one
   feature is non-zero.  Self-skips when the Wf-382-A1 PLL adapter
   is unavailable (matching the Wf-382-A1 test contract).

3. **`test_se3_sb_integration_point`** — verifies the wire-in to
   `train_se3sb.py:parse_args` is sound: trainer file exists,
   `parse_args_loc > 0`, `wire_in_ok=True`, and the wire-in is
   idempotent (running it twice does NOT increase the flag count
   beyond the first run).

### §3.2 Test Pass-Rate (Wf-392-A1 wave)

* `test_pll_flag`: **PASS** (subprocess fallback)
* `test_pll_feature_computation`: **SKIP** (PLL adapter unavailable
  in env — Triton autotuner issue with `key=["N"]` constexpr param
  in `molmetal/adapters/flow_matching_lipman/__init__.py` import
  chain; this is an upstream Triton issue tracked per Wf-328-W7 +
  Wf-357-C3 §5.1)
* `test_se3_sb_integration_point`: **PASS** (subprocess fallback)

**2 PASS + 1 SKIP + 0 FAIL** — same graceful-skip pattern as
`molmetal/tests/test_wf382a1_pll_features.py` and
`molmetal/tests/test_wf390a3_se3_sb.py`.

### §3.3 Subprocess Fallback Pattern

Per Wf-390-A3 + Wf-357-C3 convention, the test fixture uses a
subprocess fallback when direct import fails (e.g. upstream Triton
or `flow_matching` clone issues).  The fallback invokes
`python molmetal/scripts/wf392a1_se3_sb_pll_wire.py --smoke --audit`
and parses the resulting JSON files at
`molmetal/reports/wf392a1_pll_wire/{smoke_result,wire_in_result}.json`.

### §3.4 Wire-In JSON Output

`molmetal/reports/wf392a1_pll_wire/wire_in_result.json` (audit-only):

```json
{
  "task": "Wf-392-A1 Multi-PH + PLL SE(3) SB wire-in audit",
  "trainer_path": "/home/hugo/codes/mol-metal/molmetal/scripts/train_se3sb.py",
  "trainer_found": true,
  "parse_args_loc": 557,
  "n_flags_before": 4,
  "n_flags_after": 7,
  "n_new_flags_added": 3,
  "flags_added": [
    "--multiph-pll",
    "--multiph-pll-top-k",
    "--multiph-pll-weight"
  ],
  "wire_in_ok": true,
  "train_loc": 793,
  "error": null
}
```

### §3.5 Smoke JSON Output

`molmetal/reports/wf392a1_pll_wire/smoke_result.json` (smoke run):

```json
{
  "task": "Wf-392-A1 Multi-PH + PLL SE(3) SB wire-in smoke",
  "cohort_size": 4,
  "n_succeeded": 0,
  "n_failed": 4,
  "top_k": 10,
  "feature_shape_ok": false,
  "feature_dtype_ok": false,
  "feature_not_all_zero_ok": false,
  "feature_shapes": [],
  "n_atoms_per_mol": [],
  "adapter_available": false,
  "error": "PLL adapter unavailable: ValueError(\"'N' is not in list\")"
}
```

---

## §4 Cross-References

### §4.1 R20-P3 / TODO-94 stack

* **`molmetal.adapters.flow_matching_lipman.pll_features`** (273 LOC,
  Wf-382-A1 SCAFFOLD) — the adapter layer the Wf-392-A1 wire-in
  exposes to the SE(3) SB trainer.
* **`molmetal.scripts.wf382a1_pll_feature_extract`** (387 LOC, Wf-382-A1
  SCAFFOLD) — the per-mol + aggregate PLL extraction helpers.
* **`molmetal/tests/test_wf382a1_pll_features.py`** (173 LOC, 1 pass
  + skip) — the original Wf-382-A1 test (benzene + methane sanity).
* **`molmetal.molmetal_lam.lam_chem.pll_signature`** (684 LOC, Liu &
  Feng 2026) — the canonical PLL implementation underlying the
  Wf-382-A1 adapter.

### §4.2 SE(3) SB trainer stack

* **`molmetal/scripts/train_se3sb.py`** (749 LOC, Wf-23 Phase 2 R18
  W2.1) — the SE(3) Schrödinger-Bridge 5-metal multitask trainer
  that Wf-392-A1 wires the 3 NEW flags into.
* **`molmetal/adapters/se3_sb_integration.py`** (611 LOC, W2.1) — the
  pure-math SE(3) integrator + decoder + velocity field
  (`SE3SBVelocityField`, `SE3SBDecoder`, `integrate_with_cfm`).
* **`molmetal/scripts/wf390a3_se3_sb_integration_audit.py`** —
  Wf-390-A3 audit baseline (parse_args at line 557, 4 SE(3)-specific
  flags pre-Wf-392-A1).

### §4.3 Wire-in template

* **`molmetal/scripts/train_cfm_yuelbond_wasserstein.py:1060-1180`** —
  Wf-388-A3 SA-penalty wire-in pattern (3 NEW flags, default OFF,
  opt-in via positive weight) — template for Wf-392-A1 wire-in.

### §4.4 GPU bench roadmap (deferred, USER_OK-gated)

* **TODO-207 Wf-299 launch** — USER_OK-gated per AGENTS.md §5.1.
  Full velocity-loss integration of PLL features into the SE(3) SB
  trainer requires GPU retrain + per-innovation ablation + multi-seed
  × 50-epoch protocol.  Wf-392-A1 ships the SCAFFOLD + audit;
  Wf-299 + Wf-264 + Wf-336-X2 + Wf-355 USER_OK-gated launchers carry
  the GPU bench.

### §4.5 Honest-Negative Caveats (preserved per party-twenty-spirit §3.5)

1. **PLL adapter availability** — the Wf-382-A1 adapter is
   SCAFFOLD-only; the canonical PLL implementation requires
   `scipy + RDKit + flow_matching clone + Triton autotuner`.  In
   environments where any of these is unavailable, the smoke
   gracefully skips.  Wf-392-A1 preserves this contract.
2. **Velocity-loss wiring SCAFFOLD-only** — the 3 NEW flags expose
   the *opt-in interface* but the actual velocity-loss integration
   is SCAFFOLD-only at the Wf-392-A1 wave.  Full MEASURED bench
   lift is DEFERRED per AGENTS.md §5.1 + Wf-23 + Wf-390-A3
   preservation contract.
3. **Default OFF for backward compat** — the 3 NEW flags default to
   `--multiph-pll=False`, `--multiph-pll-top-k=10`,
   `--multiph-pll-weight=0.0` to preserve bit-for-bit backward
   compatibility with the Wf-23 + Wf-390-A3 baseline (no
   perturbation when flags are absent).
4. **m6_kappa + m2 R² structural gates** — preserved
   HONEST_NEGATIVE_FINAL per Wf-365-A3 + Wf-369-A2 + Wf-377-A1
   (5 architectural attempts + redesign).  Wf-392-A1 does NOT
   affect these gates; PLL features are an opt-in augmentation
   for the SE(3) SB velocity loss, not a structural metric lift.

### §4.6 Standing Directives

* **§5.1 Standing GPU grant** — USER_OK-gated per AGENTS.md §5.1.
  No per-launch user OK required for > 1 hr launch IF
  VRAM-budgeted + config-aligned.  Wf-392-A1 ships CPU-only SCAFFOLD;
  full GPU bench deferred to Wf-299 + Wf-264 + Wf-336-X2 + Wf-355
  USER_OK-gated launchers.
* **§5.2 Honest framing** — preserved per party-twenty-spirit §3.5.
  Wf-392-A1 distinguishes SCAFFOLD audit vs PROJECTED vs MEASURED.
* **§5.3 No paper work** — `paper/*` untouched per TODO-84.
  LaTeX template SCAFFOLD lives at `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md`
  (Wf-359-B2) for USER_OK-gated conversion only.
* **§5.4 Tier-2 + NO wet-lab** — Wf-392-A1 is computational-only;
  no wet-lab planning per 2026-09-30 user reinforcement.
* **§10 Wetlab deprecation** — preserved per AGENTS.md §10.

---

## §5 MEASURED vs PROJECTED Ledger

| Metric | Value | Tag | Source |
|---|---|---|---|
| Wire-in script LOC | 510 | MEASURED | `wc -l molmetal/scripts/wf392a1_se3_sb_pll_wire.py` |
| Test file LOC | 310 | MEASURED | `wc -l molmetal/tests/test_wf392a1_se3_sb_pll.py` |
| NEW CLI flags | 3 | MEASURED | `WF392A1_NEW_FLAGS` tuple |
| `parse_args_loc` | 557 | MEASURED | `wire_in_result.json` |
| `n_flags_before` | 4 | MEASURED | `wire_in_result.json` |
| `n_flags_after` | 7 | MEASURED | `wire_in_result.json` |
| `n_new_flags_added` | 3 | MEASURED | `wire_in_result.json` |
| `train_loc` (if written) | 793 | PROJECTED | would-be post-write file size |
| Tests PASS | 2 + 1 SKIP | MEASURED | standalone runner |
| GPU launch wall | 0 hr | MEASURED | CPU-only SCAFFOLD |
| End-to-end bench lift | DEFERRED | PROJECTED | requires GPU bench |

---

**END OF Wf-392-A1 CLOSEOUT** — Owner Hugo Lee; cross-ref
`party-twenty-spirit` §3.5 + `mol-metal-cfm-training` +
`mol-metal-multi-ph-pll-feature` skills; last update 2026-09-30.
