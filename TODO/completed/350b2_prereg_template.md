# Wf-350-B2 — Pre-Registration Record Template (prereg.md)

**Date:** 2026-09-29
**Author:** Wf-350-B2 (autonomous ultracode unit)
**Scope:** Per the 2026-09-29 user request ("基于刚才的洞察，可以启动ultracode，覆盖所有发现的问题把残余的问题解决了") + Wf-343a-R1 finding #2 (DeepMind §3.1 pre-registration) + AGENTS.md §5.2 honest framing mandate.

---

## §0 TL;DR

Created `docs/PREREG_TEMPLATE.md` (~165 LOC) covering 7 required sections + 5 canonical schema fields. Updated 3 launch scripts (`scripts/launch_wf124_p3_g5.sh` + `scripts/launch_wf124_p3_g6.sh` + `scripts/recover_g6_failed_tuples.sh`) to optionally accept `--prereg <path>` and emit WARNING when paper-grade launches proceed without one. **2 NEW tests** (4 test cases with parametrization) PASS at Wf-350-B2 timestamp wall 0.52s.

**Honest framing per AGENTS.md §5.2 + party-twenty-spirit §3.5:**

- **Template only** — not auto-enforced; USER_OK-gated per AGENTS.md §5.4 Tier-2 marker.
- **Launch script integration** is additive — `--prereg` flag is OPTIONAL; existing launch invocations without `--prereg` continue to work, just emit a WARNING.
- **Tests: 2 NEW test functions** (4 cases with parametrization, since `test_launch_scripts_accept_prereg_arg` is parametrized over 3 launch scripts).
- **Wall:** MEASURED **0.52s** for 4 test cases (CPU-only).
- **Backward-compat:** 0 existing tests modified; 0 existing launchers modified in a breaking way.

---

## §1 Files Created

| File | LOC | Purpose |
|---|---:|---|
| `docs/PREREG_TEMPLATE.md` | ~165 | Pre-registration record template with 7 sections (§0-§9) + 5 canonical schema fields |
| `molmetal/tests/test_wf350b2_prereg.py` | ~175 | 2 NEW test functions (4 cases via parametrize) covering template schema + launch-script arg parsing |
| `TODO/completed/350b2_prereg_template.md` | ~200 | This ship doc |

---

## §2 Files Modified

| File | Change | LOC delta |
|---|---|---:|
| `scripts/launch_wf124_p3_g5.sh` | Added `--prereg <path>` parser + WARNING emitter + schema verification block; switched dry-run gate from `${1:-}` to `${DRY_RUN}` variable | +60 / -3 |
| `scripts/launch_wf124_p3_g6.sh` | Added `--prereg <path>` parser + WARNING emitter + schema verification block; switched dry-run gate from `${1:-}` to `${DRY_RUN}` variable | +60 / -3 |
| `scripts/recover_g6_failed_tuples.sh` | Added `--prereg <path>` parser + WARNING emitter + schema verification block | +60 / -1 |

---

## §3 Template schema (docs/PREREG_TEMPLATE.md)

### §3.1 Section structure

| § | Title | Required for |
|---|---|---|
| §0 | Purpose | Always (template preamble) |
| §0.1 | Canonical schema fields (machine-parseable) | Machine-checked by launch scripts |
| §1 | Research question + hypothesis | Always |
| §2 | Dataset(s) + cohort definition | Always |
| §3 | Hyperparameters + model architecture | Always |
| §4 | Evaluation metric(s) + acceptance gate | Always |
| §5 | Pre-registered ablation plan | If launch is an ablation |
| §6 | Pre-registered analysis plan | Always |
| §7 | Anti-patterns: NO post-hoc cherry-picking | Always |
| §8 | Sign-off | Always |
| §9 | References | Always |

### §3.2 Canonical schema fields (DeepMind §3.1; Wf-343a-R1 finding #2)

The launch scripts verify these 5 snake_case tokens are present:

1. `hypothesis` (§1)
2. `primary_metric` (§4)
3. `baseline` (§4)
4. `expected_effect_size` (§4)
5. `falsification_criterion` (§4)

### §3.3 Anti-pattern prohibitions (§7)

The template explicitly enumerates 7 anti-patterns that would invalidate the pre-registration:

1. Excluding metals/seeds post-hoc
2. Re-running with different hyperparameters
3. Re-framing the primary metric after seeing results
4. Cherry-picking cohorts
5. Reporting PROJECTED as MEASURED
6. Silently changing the config file
7. Selective reporting of ablation results

---

## §4 Launch script integration

### §4.1 `--prereg <path>` argument

All 3 launch scripts now accept an OPTIONAL `--prereg <path>` argument:

- **Path form A (with equals):** `--prereg=docs/prereg/<file>.md`
- **Path form B (separate args):** `--prereg docs/prereg/<file>.md`

### §4.2 Behavior

| State | Behavior |
|---|---|
| `--prereg <path>` supplied + file exists + has all 7 sections | Emits `=== PREREG verification: <path> ===` + `OK: prereg.md schema valid; hypothesis + metrics sections present` + checks hyperparameter table for epochs=50 + w2_gradient_mode sort (G5/G6) + proceeds with launch |
| `--prereg <path>` supplied + file MISSING | Emits `FAIL: --prereg file not found: <path>` + `exit 1` (pre-launch abort) |
| `--prereg <path>` supplied + missing §1 or §4 | Emits `FAIL: prereg.md missing §N ...` + `exit 1` (pre-launch abort) |
| `--prereg` NOT supplied | Emits `WARNING: --prereg <path> not supplied. Paper-grade launches should provide a pre-registration record per AGENTS.md §5.2 + Wf-350-B2 PREREG_TEMPLATE.md.` + proceeds with launch (warning is non-blocking per Tier-2 marker) |

### §4.3 Backward compat

The existing dry-run path (`--dry-run`) is preserved by switching from positional `${1:-}` to a stateful `DRY_RUN` variable that any flag order can set. The existing `--prereg` arg-position-independence is critical for the recovery script which has a separate DRY_RUN setup at line 164 (the test passes `--dry-run` first for the recovery script to short-circuit the USER_OK gate).

---

## §5 Tests (molmetal/tests/test_wf350b2_prereg.py)

### §5.1 `test_prereg_template_schema_validity`

Verifies:
- `docs/PREREG_TEMPLATE.md` exists
- All 7 required sections (§1 Research question, §2 Dataset(s), §3 Hyperparameters, §4 Evaluation metric, §5 Pre-registered ablation plan, §6 Pre-registered analysis plan, §7 Anti-patterns) are present (regex match on `^## §N ...` lines)
- All 5 canonical fields (hypothesis, primary_metric, baseline, expected_effect_size, falsification_criterion) are present (string match)
- Template references `AGENTS.md §5.2` (honest framing)
- Template references `Wf-350-B2` (self-reference)

### §5.2 `test_launch_scripts_accept_prereg_arg`

Parametrized over 3 launch scripts:
- `scripts/launch_wf124_p3_g5.sh`
- `scripts/launch_wf124_p3_g6.sh`
- `scripts/recover_g6_failed_tuples.sh`

For each, verifies:
- Source code references `--prereg` parsing logic
- Source code has PREREG verification block
- Subprocess: `--prereg docs/PREREG_TEMPLATE.md --dry-run` emits `OK: prereg.md schema valid` line + exits rc=0
- Subprocess: `--dry-run` (no --prereg) emits `WARNING: --prereg <path> not supplied` line + exits rc=0

### §5.3 Test results

```
molmetal/tests/test_wf350b2_prereg.py::test_prereg_template_schema_validity PASSED
molmetal/tests/test_wf350b2_prereg.py::test_launch_scripts_accept_prereg_arg[scripts/launch_wf124_p3_g5.sh] PASSED
molmetal/tests/test_wf350b2_prereg.py::test_launch_scripts_accept_prereg_arg[scripts/launch_wf124_p3_g6.sh] PASSED
molmetal/tests/test_wf350b2_prereg.py::test_launch_scripts_accept_prereg_arg[scripts/recover_g6_failed_tuples.sh] PASSED

========= 4 passed, 1 warning in 0.52s =========
```

Wall: **MEASURED 0.52s** for 4 test cases.

---

## §6 Tier-2 honest framing (AGENTS.md §5.4)

Per the 2026-09-29 user directive ("我们发二区就可以了，不用规划湿实验"), the pre-registration template is **NOT auto-enforced**:

- The template is a recommendation for paper-grade work, NOT a hard prerequisite.
- Existing launch invocations (`bash scripts/launch_wf124_p3_g5.sh --dry-run`) continue to work unchanged — they just emit a WARNING line about pre-registration.
- The WARNING is non-blocking; it does not gate the launch.
- If a future user wants to make `--prereg` mandatory for paper-grade launches, that's a separate Wf (TODO-138 paper §3 hookup + a launch-script policy gate).

This is consistent with the Wf-343a-R1 finding #2 + Wf-343a6 audit recommendation row "Pre-registration record" being **DEFERRED under Tier-2 re-orientation** (per Wf-343a6 §8.2.5). The template is shipped because it's lightweight + useful for users who want it, NOT because Tier-2 acceptance requires it.

---

## §7 Forward-looking honest-negatives

1. **Template is not auto-enforced** — paper-grade launches can still proceed without a prereg.md. This is intentional per Tier-2 + the Wf-343a6 deferral.
2. **Schema verification is shallow** — only checks 7 sections + 5 snake_case tokens are present. Does NOT verify the content matches the actual launch config (e.g. that `epochs=50` in the table matches the launcher's `--epochs 50`). Deeper semantic verification is a future Wf.
3. **No pre-registration registry** — there is no central index of prereg.md files. Future workflows could add a `docs/prereg/INDEX.md` to track which launches have pre-registrations on file.
4. **No integration with paper §3 / §4** — the prereg.md template references paper §3/§4/§5 as the downstream consumer but does not auto-link the two. Future work could add a TODO/INDEX.md column linking each prereg.md to its paper section.
5. **No version control hook** — the prereg.md is a static file; if the user edits it after launch, there is no audit trail. Future work could enforce that prereg.md files are committed to git before launch (per the kill_then_adapt directive).

---

## §8 Cross-references

- AGENTS.md §5.1 standing GPU authorization (2026-09-29 user grant)
- AGENTS.md §5.2 honest framing mandate (party-twenty-spirit §3.5)
- AGENTS.md §5.4 Tier-2 journal target + NO wet-lab planning
- TODO/completed/207_unified_pre_flight_checklist_v2.md (pre-flight gate catalog)
- TODO/completed/343a1_industry.md §4 recommendation #2 (prereg.md origin)
- TODO/completed/343a5_audit_207.md row 8 (pre-research framing DeepMind §3.1)
- TODO/completed/343a6_audit_per_wave.md row "Pre-registration record" (DEFERRED under Tier-2)
- TODO/completed/350b2_prereg_template.md (this file — Wf-350-B2 ship doc)
- DeepMind responsible research framework §3.1 (pre-registration)

---

**END OF 350b2_prereg_template.md** — 2 NEW tests PASS; 4 test cases PASS total at Wf-350-B2 timestamp wall 0.52s; pre-flight v3 62/62 PASS PRESERVED (template + launch-script edits are non-blocking, no pre-flight gates touched).
