# Wf-A13 — single-venv migration, gate integrity, and two silent-degradation fixes

**Date:** 2026-10-03
**Status:** SHIPPED (working tree; committed separately)
**Authority:** user directives 2026-10-03 — *"全部修到这个项目下专用的虚拟环境（之前明明cuda版本的torch以及其他依赖都是装好的）"* and *"你提到的另一点也要修"*

---

## §0 TL;DR

Four things, in the order the user asked for them.

1. **Every live reference to the legacy `CadButEaas/.venv-vlm` now points at the
   repo-local `.venv`** — 203 lines across 141 files, then 9 more across 7, then
   97 across 35 for the identifier/prose residue. Final sweep: **0 executable
   paths left**.
2. **The single-venv guard is now wired into the pre-commit hook** (it had been
   an orphan for weeks) — verified in both directions.
3. **SyntheMol's path was hardcoded to a checkout that does not exist**, causing
   the reaction-rule registry to silently hold 5 rules instead of 22.
4. **The 43 GB pre-featurized duplicate was deleted**, and a 43 GB memory
   landmine in the test suite was defused.

A fifth item — the `.apply()` API gap in `SyntheMolReactionRule` — was
**investigated and deliberately NOT changed**; see §5.

---

## §1 Single-venv migration

### §1.1 The two regimes

Two mutually contradictory, both-live venv regimes existed:

| source | says canonical | says forbidden |
|---|---|---|
| **AGENTS.md §5.9** (2026-10-02) | `mol-metal/.venv` | CadButEaas |
| `.gitignore` L9-15 + `env.sh` + `Makefile` + `ci.yml` (Sep 25) | `CadButEaas/.venv-vlm` | `mol-metal/.venv` |

The second even shipped an executable guard that **printed
`Remove it: rm -rf /home/hugo/codes/mol-metal/.venv`** — i.e. it told the reader
to delete the venv §5.9 designates as canonical. §5.9 wins (newer, and it is the
user's own directive).

### §1.2 Method — a script, not a sed

203 sites across 141 files is not hand-editable, but a blind `sed` is worse. The
migration ran as `/tmp/wfa13_venv_migration.py` with a **dry-run default**, and
the dry run earned its keep twice:

- **Trap 1 — the rule documents.** Nine references are *negative examples*:
  `AGENTS.md:517` `# WRONG: CadButEaas venv`, `env.sh:12`, and
  `check_no_stale_venv.sh:35` — the guard's own rejection message. Rewriting
  them would have turned *"# WRONG: CadButEaas"* into *"# WRONG: .venv"* and
  left the guard unable to say what it rejects. Guarded by whole-file exclusion
  plus a per-line negative-example regex.
- **Trap 2 — `..venv`.** A bare `("venv-vlm", ".venv")` rule turns `.venv-vlm`
  into `.` + `.venv` = **`..venv`**, silently corrupting every path it touched.
  Caught in the dry run, never applied. The safe form keeps the dots.
- **Trap 3 — BioLM.** A catch-all `CadButEaas` → prose rule rewrote the pure-data
  path `/home/hugo/codes/CadButEaas/references/BioLM-Score` into
  `/home/hugo/codes/the project venv/references/BioLM-Score`. Caught, guarded.

### §1.3 The symlink did most of the work

`molmetal/.venvs/mol-metal-venv` was a symlink to `CadButEaas/.venv-vlm`, and
**7 scripts use it as their interpreter path**
(`master_verify_r18.py:80`, `verify_train_affinity_regression.py:102`,
`wf47_boltz_real_runs.py:169`, `integration_verify_scoring.py:93`,
`wf47_pb_aizynth_real_runs.py:96`, and two more). Repointing the symlink fixed
all seven with one `ln -snf`, no code change.

MEASURED after: `molmetal/.venvs/mol-metal-venv/bin/python` →
`torch 2.14.1+cu130 | cuda True`, and `readlink -f` equals the canonical `.venv`.

### §1.4 Also found

- A **typo'd path** in 15 places: `/home/hugo/cades/CadButEaas/...` — `cades`,
  not `codes`. It cannot ever have resolved.
- Two shapes the first pass missed and a second pass caught:
  `Path(".../.venv-vlm")` (no trailing slash) and `Path(".../.venv-vlm/bin")`.
- Variable names left pointing correctly but named wrongly
  (`CADBUTEAAS_PY = Path(".../mol-metal/.venv/...")`) — renamed, together with
  the sibling test that asserts `"CADBUTEAAS_PY" in text`, so both sides moved
  in step.

### §1.5 Verification

```
executable paths still pointing at CadButEaas (code+sh)  : 0
same, across code + configs + sh + toml                  : 0
configs parse                                            : 9/9
test_script_import_health.py                             : 14 passed
test_env_setup.py (asserts the symlink target)           : 10 passed
```

## §2 The guard is now wired

`check_no_stale_venv.sh` appeared **0 times** in `.git/hooks/pre-commit` and
**0 times** in `.pre-commit-config.yaml`. Its only "wiring" was a comment in
`env.sh` telling the reader to run it.

It is now step **(3/3)** of the pre-commit hook, with a `SKIP_VENV_GUARD=1`
escape hatch. Verified both directions — a guard that cannot fail is decoration:

```
clean env              -> exit 0
UV_PROJECT_ENVIRONMENT=<legacy path>  -> exit 1
```

**CAVEAT, and it matters:** `.git/hooks/pre-commit` lives inside `.git/`, which
git never tracks. **This wiring exists only in this clone.** Any other checkout
(or a fresh clone) has the guard script but does not run it. The durable fix is a
`.pre-commit-config.yaml` `repo: local` hook, which is version-controlled —
**not done here, and it should be the follow-up.**

## §3 SyntheMol — a silent capability loss

`molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py` hardcoded:

```python
_SYNTHEMOL_PATH = Path(
    "/home/hugo/codes/try_triton_on_rocm/molmetal/references/SyntheMol/synthemol"
)
```

— a path from a **different checkout** (the ROCm-era `try_triton_on_rocm` tree)
that no longer exists. The repo-local `references/SyntheMol/` was present and
complete the whole time.

**Why it mattered more than a broken path usually does.** `beta_reductions`
catches the resulting error, emits **one RuntimeWarning**, and continues:

```
[beta_reductions] SyntheMol registration failed: [Errno 2] No such file or
directory: '.../try_triton_on_rocm/.../Synthemol/utils.py'.
REACTION_RULES will contain only the 5 click rules.
```

Nothing fails. The model simply reasons over a fraction of its reaction space.
The only trace is a warning in a log nobody reads.

FIX: derive the path from `__file__` (repo-local first), with the legacy
absolute path kept as a fallback, and a clear error rather than a silent
degrade. MEASURED after:

```
registered : True
SyntheMol-contributed rules : 13   (matches the module's documented 13)
total REACTION_RULES        : 22   (was 5 while registration failed)
```

The same defect class was checked in `biolm_score_{oracle,full_adapter}.py`.
**They are already correct** — both use a candidate tuple with
repo-local-first and first-existing-wins, so the dead `CadButEaas` entry there is
a fallback that never fires. (I initially reported BioLM as broken; that was
wrong, and the distinction is exactly the one that mattered for SyntheMol:
single hardcoded path → broken; candidate list with a good head → fine.)

## §4 Housekeeping along the way

- **`molmetal/data/pre_featurized/_stale_2026_09_27/` deleted — 43 GB.**
  Not a duplicate: it carried the pre-regeneration **stub domain mapping**
  (`0=metal_cytotox 1=nci60 2=binding_db 3=drugood`). MEASURED labels, live vs
  deleted: `nci60` 2 vs 1, `binding_db` 3 vs 2, `drugood` 1 vs 3, and
  `metal_cytotox` 0 vs 0 — the one file identical under both mappings, which is
  what rules out a reading artifact. Confirmed independently by SHA-256 (4/4
  agreeing, including which single file is identical). See
  `TODO/completed/pre_featurized_stale_batch_deletion.md`.
- **A 43 GB memory landmine defused.** `test_pre_featurized_domain_labels.py`
  called `torch.load(...)` with no `mmap`, reading `all_pretrain.pt` (43 GB) into
  RAM. Now `mmap=True`. MEASURED under a hard 8 GB cap:
  `9 passed in 0.54s`, **peak RSS 0.78 GB** (~55x less).
- **The kanzi-era GPU guidance is gone.** The user confirmed the kanzi
  environment is no longer on this GPU. `.claude/skills/mol-metal-gpu-execution`
  had a whole §5 built on avoiding it — including *"for any job expected to run
  > 1 hour, **default to GPU 1 regardless of probe**"* — which also contradicted
  the standing *"只用gpu0吧"*. Rewritten to GPU 0. Six sibling skills carried
  `CUDA_VISIBLE_DEVICES=1`; all now `=0`. (`TODO/environment.md`'s
  `CUDA_VISIBLE_DEVICES=1` is a **measurement** showing the CUDA var works where
  the HIP one does not — left alone.)

## §5 Investigated, deliberately NOT changed

`molmetal/tests/test_f2a_metal_ligand_exchange.py` has **50 AttributeError + 4
AssertionError** failures. It was tempting to add an `apply()` alias to
`SyntheMolReactionRule` and close 50 of them. **That would have been wrong.**

The module exposes `can_apply / fire / predict_yield / rate_predictor /
reaction_id / reduce / requires_catalyst / reverse_can_apply` — and `fire` takes
a single `List[str]`. But existing call sites use `apply` with **two** arguments:

```python
# molmetal/molmetal_lam/lam_chem/medchem_gen_integration.py:650-662
product = rule.apply(copartner)
product = rule.apply(smiles, copartner)
# and lambda_combinators.py:379 documents the convention as
#   "rule.apply(scaffold, tile)"
```

So `apply` is expected to mean `fire(scaffold, tile)` — a **different signature**.
Aliasing `apply = fire` would silently mis-bind every two-argument call. This is
a genuine API mismatch that needs a design decision, not a mechanical alias.

The remaining gap is pre-existing and already documented by the pre-flight gate
itself, which excludes this file on purpose:
> *"its 33 syntemol-rule tests assume `SyntheMolReactionRule.apply()` which the
> current sbdd_env.syntemol_reactions module does NOT expose … Tracked for Wf-297
> follow-up; not blocking preflight."*

The SyntheMol path fix **improved** this file's state (rule count 0 → 13) without
making it pass: the canonical-count test wants `>= 18` and the module provides 13.

## §6 Honest framing

- **MEASURED**: 0 remaining executable `CadButEaas` paths; 9/9 configs parse;
  14 + 10 tests pass; guard exit 0 clean / 1 poisoned; 13 SyntheMol rules, 22
  total; 0.78 GB peak RSS under an 8 GB cap; 43 GB reclaimed.
- **NOT MEASURED / NOT CLAIMED**: no GPU training was run; no shipped metric was
  recomputed. The SyntheMol rule-count change affects future runs only.
- **KNOWN GAP**: the hook wiring is clone-local (§2). The `.apply()` API gap is
  open and needs a decision (§5).
- Prose residue remains in a handful of comments/messages (e.g.
  `cfm_vina_dock_panel.py:1840`, `m6_wf320h_vina_phase2.py:16`). No executable
  path remains; these are labels, not behaviour.
