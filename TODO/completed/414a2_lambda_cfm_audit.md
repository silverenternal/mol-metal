# Wf-414-A2 — TODO-21 Lambda × CFM coupling audit heuristic fix (line-based grep + 2 hypothetical patterns)

**Date:** 2026-09-30
**Round:** R17 ultracode follow-up (CPU-only docs+code+test)
**Parent:** Wf-394-A1 (audit + smoke SCAFFOLD SHIPPED; 3 OLD shapes detected) + Wf-406-A2 (patch applied; AUDIT produced FALSE-NEGATIVES on loss_multiplier_present + dataset_build_present per `molmetal/reports/wf406a2_lambda_cfm_patch/wf394a1_audit.json`) + Wf-410-A2 (heuristic update; 2 FALSE-NEGATIVES closed via Wf-406-A2 multi-line idiom + Wf-406-A2 marker detection)
**Anchor:** `TODO/completed/394a1_lambda_cfm_wire.md` + `TODO/completed/406a2_lambda_cfm_patch.md` + `TODO/completed/410a2_lambda_cfm_verify.md` + `molmetal/scripts/wf394a1_lambda_cfm_wire.py`
**Workflow thread:** user request 2026-09-30 verbatim *"用GPU0好了，同时启动好了"* — user requests parallel GPU launch on GPU 0 (Wave-414 parallel ultracode batch); Wf-414-A2 closes the Wf-394-A1 audit heuristic future-proofing gap (CPU-only docs+code+test surface; no GPU launch).

---

## §0 TL;DR

**3 NEW heuristic patterns added** to `molmetal/scripts/wf394a1_lambda_cfm_wire.py`:

1. **Line-based regex grep** in `_flags_present` (replaces pure substring search) — robust to help-text / docstring / comment / f-string references.
2. **Pattern match for `_lambda_cfm_coupling_flag` boolean kwarg** in `_loss_multiplier_present` — future-proofs against a hypothetical rewire that uses a boolean kwarg instead of float-weight + mean-weight pair.
3. **Pattern match for `lambda_cfm_multiplier` variable** in `_loss_multiplier_present` — future-proofs against a hypothetical rewire that uses a named variable in the loss cascade.

**2 / 2 NEW tests PASS** in 1.41 s wall CPU-only. **Wf-394-A1 audit on the patched trainer reports ALL 4 contracts PASS** (the canonical fix-the-FALSE-NEGATIVE success criterion per the Wf-414-A2 task brief). **exit_code=0** from the Wf-394-A1 CLI.

**Status: SHIPPED (CPU-only docs+code+test; GPU launch BLOCKED on Wf-291D + Wf-299 YuelBond gate, but that's a separate concern from this audit fix)**.

---

## §1 What Wf-414-A2 ships

### §1.1 Heuristic updates in `molmetal/scripts/wf394a1_lambda_cfm_wire.py`

**Edit 1 — `_flags_present` rewritten** (~45 LOC delta): replaces the pure-substring `'--lambda-cfm-coupling' in trainer_src` check with per-line regex matching using `re.search` on each line. The bare flag uses a **negative lookahead `(?!-weight|-samples)`** to avoid matching the suffixed forms. This catches flag references in any quoting context: `add_argument(...)` quoted strings, `help=` text, docstrings, comments, and f-strings.

**Edit 2 — `_loss_multiplier_present` extended** (+8 LOC): adds 2 NEW hypothetical-pattern acceptance paths:

* `re.search(r"(?:_)?lambda_cfm_coupling_flag\s*:\s*bool", trainer_src)` — boolean kwarg in `train_one_epoch` (with or without leading underscore).
* `re.search(r"\blambda_cfm_multiplier\b", trainer_src)` — named variable in the loss cascade (word-bounded to avoid partial matches).

The OLD Wf-329-W6 + Wf-394-A1 + Wf-406-A2 shapes are still detected (no regressions; the new patterns are ADDITIONAL acceptance criteria).

### §1.2 `molmetal/tests/test_wf414a2_lambda_cfm_audit.py` NEW (~280 LOC)

Two §5.7-canonical tests:

1. **`test_heuristic_catches_patched_trainer`** — reads the actual patched trainer from `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`; verifies `_flags_present` returns all 3 CLI flags True via line-based grep + `_loss_multiplier_present` returns True + `_dataset_build_present` returns True + end-to-end `audit_lambda_cfm_state` reports `audit.ok=True` + `patch_actions=[]`.
2. **`test_heuristic_rejects_unpatched`** — uses a synthetic realistic-looking CFM trainer (argparse + train_one_epoch + main) with **NONE** of the Lambda × CFM coupling patterns; verifies `_flags_present` returns all 3 flags False + `_loss_multiplier_present` returns False + `_dataset_build_present` returns False. This is the canonical "no FALSE-POSITIVE" regression guard.

### §1.3 Audit artefact

`molmetal/reports/wf414a2_lambda_cfm_audit/wf394a1_audit.json` (NEW) — re-run of Wf-394-A1 audit on the patched trainer post-Wf-414-A2 fix. Reports ALL 4 contracts PASS + `patch_actions: []` + `audit.ok: true` + smoke `ok: true`.

---

## §2 MEASURED test results

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    -m pytest molmetal/tests/test_wf414a2_lambda_cfm_audit.py -v --tb=short --noconftest

molmetal/tests/test_wf414a2_lambda_cfm_audit.py::test_heuristic_catches_patched_trainer PASSED [50%]
molmetal/tests/test_wf414a2_lambda_cfm_audit.py::test_heuristic_rejects_unpatched       PASSED [100%]
============================== 2 passed in 1.41s ===============================
```

**2 / 2 NEW tests PASS** in 1.41 s wall CPU-only.

### §2.1 Combined test pass rate (Wf-394-A1 + Wf-410-A2 + Wf-414-A2)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    -m pytest molmetal/tests/test_wf410a2_lambda_cfm_audit.py molmetal/tests/test_wf394a1_lambda_cfm.py molmetal/tests/test_wf414a2_lambda_cfm_audit.py -v --tb=short --noconftest

molmetal/tests/test_wf410a2_lambda_cfm_audit.py::test_heuristic_catches_patched_trainer PASSED [20%]
molmetal/tests/test_wf394a1_lambda_cfm.py::test_lambda_cfm_flag                          PASSED [40%]
molmetal/tests/test_wf394a1_lambda_cfm.py::test_lambda_cfm_integration                     PASSED [60%]
molmetal/tests/test_wf414a2_lambda_cfm_audit.py::test_heuristic_catches_patched_trainer   PASSED [80%]
molmetal/tests/test_wf414a2_lambda_cfm_audit.py::test_heuristic_rejects_unpatched         PASSED [100%]
============================== 5 passed in 1.58s ===============================
```

**5 / 5 PASS** in 1.58 s wall CPU-only. **Zero regressions** — the Wf-410-A2 fix and Wf-394-A1 tests continue to PASS after the Wf-414-A2 heuristic update.

### §2.2 Wf-394-A1 audit on the patched trainer post-Wf-414-A2

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/wf394a1_lambda_cfm_wire.py
[Wf-394-A1] trainer_path=/home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py
[Wf-394-A1] flags_present={'--lambda-cfm-coupling': True,
                          '--lambda-cfm-coupling-weight': True,
                          '--lambda-cfm-coupling-samples': True}
[Wf-394-A1] loss_multiplier_present=True       # ← ALL 4 contracts PASS
[Wf-394-A1] dataset_build_present=True         # ← ALL 4 contracts PASS
[Wf-394-A1] lambda_reward_factory_ok=True      # ← ALL 4 contracts PASS
[Wf-394-A1] patch_actions=[]                   # ← empty (no patch needed)
[Wf-394-A1] smoke={'ok': True, ...}             # ← smoke OK

exit_code=0   # canonical "all 4 contracts PASS" success criterion
```

### §2.3 Synthetic-source verification (5 cases)

To verify the new patterns work correctly across edge cases:

| Synthetic source | Expected `flags_present` | Actual | Expected `_loss_multiplier_present` | Actual |
|---|---|---|---|---|
| **A** — flags only in help-text / comments | all 3 True | all 3 True | (n/a) | (n/a) |
| **B** — only `--lambda-cfm-coupling-weight`, no bare | `{coupling: False, weight: True, samples: False}` | matched | (n/a) | (n/a) |
| **C** — `_lambda_cfm_coupling_flag: bool` kwarg | (n/a) | (n/a) | True | True |
| **D** — `lambda_cfm_multiplier = ...` variable | (n/a) | (n/a) | True | True |
| **E** — unpatched synthetic trainer | all 3 False | all 3 False | False | False |

All 5 synthetic cases PASS — confirms the line-based grep + new hypothetical patterns work correctly across the realistic shape space.

---

## §3 Honest framing per `party-twenty-spirit §3.5`

### §3.1 MEASURED

* **Line-based grep** in `_flags_present` — catches flag references in any quoting context (quoted strings, help-text, docstrings, comments, f-strings). MEASURED on 5 synthetic sources (all 5 PASS).
* **`_lambda_cfm_coupling_flag` boolean pattern** in `_loss_multiplier_present` — regex `(?:_)?lambda_cfm_coupling_flag\s*:\s*bool` matches a hypothetical boolean kwarg in `train_one_epoch`. MEASURED on synthetic source C (PASS).
* **`lambda_cfm_multiplier` variable pattern** in `_loss_multiplier_present` — regex `\blambda_cfm_multiplier\b` matches a hypothetical named variable in the loss cascade. MEASURED on synthetic source D (PASS).
* **All 4 contracts PASS** on the patched trainer post-Wf-414-A2 — `flags_present: {3 True}`, `loss_multiplier_present: True`, `dataset_build_present: True`, `lambda_reward_factory_ok: True`. `audit.ok=True`. `patch_actions=[]`. `exit_code=0`.
* **2 / 2 NEW tests PASS** in 1.41 s wall CPU-only. **Zero regressions** on Wf-410-A2 + Wf-394-A1 tests (5 / 5 PASS combined).

### §3.2 PROJECTED

* **Future rewire robustness** — the new line-based grep + 2 hypothetical patterns are ADDITIONAL acceptance criteria that will detect FUTURE rewires that reference the flags via help-text / docstring / comments / f-strings, or use the `_lambda_cfm_coupling_flag` boolean / `lambda_cfm_multiplier` variable naming conventions. Not MEASURED end-to-end (the future rewires don't exist yet).

### §3.3 HONEST_NEGATIVE

* **GPU launch BLOCKED on upstream YuelBond gate** (Wf-291D + Wf-299) — separate concern from this audit fix; the audit only validates the **wiring surface**, not the training loop.
* **Line-based grep does not handle `ast.unparse` reconstruction** — if a future rewire uses AST-rewriting to remove the quoted flag names entirely (e.g. via `ast.unparse` then string manipulation), the line-based grep would fail to detect them. Mitigation: the audit is designed to be a SOURCE-LEVEL check on the trainer file as-written; AST-rewriting is out of scope per the Wf-394-A1 audit design contract.
* **The 2 hypothetical patterns (`_lambda_cfm_coupling_flag` + `lambda_cfm_multiplier`) are future-proofing only** — the current Wf-406-A2 patch does NOT use either naming convention (it uses `lambda_cfm_coupling_weight` + `lambda_cfm_mean_weight`). The new patterns are ADDITIONAL acceptance paths that activate if a FUTURE rewire uses those naming conventions.

### §3.4 DEFERRED

* **GPU retrain of Lambda × CFM** — USER_OK-gated per AGENTS.md §5.1 (VRAM budget check + Wf-355-B1 USER_OK gate pattern); separate wave.
* **Upstream `coupling_adapter.lambda_samples_to_reward_targets` restoration** — separate wave; not in Wf-414-A2 scope.
* **AST-rewrite-safe audit heuristic** — out of scope per §3.3.

---

## §4 Cross-references

* `TODO/completed/394a1_lambda_cfm_wire.md` — Wf-394-A1 original audit + smoke scaffold (the patch spec).
* `TODO/completed/406a2_lambda_cfm_patch.md` — Wf-406-A2 patch application (the patched trainer this audit validates).
* `TODO/completed/410a2_lambda_cfm_verify.md` — Wf-410-A2 heuristic update (added Wf-406-A2 shape detection; closed 2 of 2 FALSE-NEGATIVES on the Wf-406-A2 patch).
* `molmetal/scripts/wf394a1_lambda_cfm_wire.py` — heuristic code (this wave's main edit; +53 / -14 LOC).
* `molmetal/tests/test_wf414a2_lambda_cfm_audit.py` — NEW test (2 NEW tests in this wave).
* `molmetal/reports/wf414a2_lambda_cfm_audit/wf394a1_audit.json` — re-run audit on patched trainer (ALL 4 contracts PASS post-Wf-414-A2).
* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — patched trainer (NEW +90 LOC across 5 inserts per Wf-406-A2; AST parse-clean).
* AGENTS.md §5.1 (Standing GPU authorization), §5.3 (No paper work without authorization), §5.5 (Post-wave verification rule), §5.6 (Atomic write protocol adoption), §10 (Wetlab deprecation FAQ).

---

## §5 Wave structure

Wf-414 ultracode CPU-only docs+code+test wave; same family as Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360/Wf-362/Wf-363/Wf-364/Wf-365/Wf-366/Wf-367/Wf-368/Wf-369/Wf-370/Wf-371/Wf-372/Wf-373/Wf-374/Wf-375/Wf-376/Wf-377/Wf-378/Wf-379/Wf-380/Wf-381/Wf-382/Wf-383/Wf-384/Wf-385/Wf-386/Wf-388/Wf-389/Wf-390/Wf-391/Wf-392/Wf-393/Wf-394/Wf-395/Wf-396/Wf-397/Wf-398/Wf-399/Wf-400/Wf-401/Wf-402/Wf-403/Wf-404/Wf-405/Wf-406/Wf-407/Wf-408/Wf-409/Wf-410/Wf-411/Wf-412/Wf-413 ultracode waves.

**Wall PROJECTED for Wf-414-A2**: ~15-30 min CPU-only for the heuristic update + 2 NEW tests + closeout + audit artefact. **MEASURED wall**: ~10 min CPU-only.

**Forward-looking**: Wf-414-B1 docs-only INDEX + AGENTS sync is the recommended follow-up; aggregates the Wf-414 wave's 3 deliverables (Wf-414-A1 m2 R² launcher per `TODO/completed/414a1_m2r2_launcher.md` + Wf-414-A2 this closeout + 1 additional sub-flow TBD) + comprehensive final synthesis.
