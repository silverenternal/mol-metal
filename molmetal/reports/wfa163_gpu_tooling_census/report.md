# Wf-A163 — GPU-tooling census: every repo path that can touch GPU memory when the session forbids it

**Branch:** G of Wf-A163 (parallel "ultracode" wave).
**Date:** 2026-11-01.
**Artefact:** `molmetal/reports/wfa163_gpu_tooling_census/gpu_tooling_census.json` (MEASURED).
**Scanner:** `molmetal/scripts/wfa163_gpu_tooling_census.py` (stdlib only; torch is NEVER imported).

## 0. What was and was not done

* **0 GPU.** No CUDA query of any kind was executed. Every python/pytest/subprocess call
  in this branch ran with `CUDA_VISIBLE_DEVICES=""` in the environment.
* The installed torch's **source** was read
  (`.venv/lib/python3.12/site-packages/torch/cuda/__init__.py`) to establish what each
  API actually does. Reading source is not executing a query.
* **No `os.walk`.** Enumeration is `git ls-files -z` + `git ls-files -z --others
  --exclude-standard`, filtered to the scope prefixes. (The PATH `grep` here is ugrep
  with `-I`, which skips the repo's non-UTF-8 tracked text files, so the scan is
  AST/regex based, not grep based.)
* The census artefact and this report are written through
  `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file`.

## 1. The denominator (every count below is over this set)

| Scope | Files | Hits |
|---|---|---|
| `molmetal/<package>` | 716 | 118 |
| `molmetal/scripts` | 897 | 694 |
| `molmetal/tests` | 1307 | 1091 |
| `scripts` | 110 | 72 |
| `tests` | 38 | 23 |
| `molmetal/reports` (archived snapshots) | 3271 | 27 |
| **total enumerated in scope** | **6339** | — |

* **files read and scanned: 3011** = 2915 `.py` + 96 `.sh`. The other 3328 files in scope
  are `.md`/`.json`/… report archives that carry no executable GPU path.
* **0 unreadable, 0 parse failures.** Every `.py` in scope parsed.
* **666 distinct files** carry ≥ 1 hit.
* Each category was scanned over the whole `.py` set (2915) — `cuda_probe`,
  `device_autoselect`, `child_spawn`, `device_override` — and `session_lock` over the
  whole `.py`+`.sh` set (3011).

> The file counts move while other Wf-A163 branches land files. This is a snapshot of
> the working tree at the time of the run; the artefact records the exact denominator
> it was produced against, so a later re-run can be compared rather than assumed.

## 2. The census — 2025 hits

| Kind | Hits | Breakdown |
|---|---|---|
| `cuda_probe` | 274 | 242 `is_available`, 22 `device_count`, 8 `current_device`, 2 `set_device` |
| `device_autoselect` | 293 | `"cuda" if torch.cuda.is_available() else "cpu"`-shaped selection |
| `child_spawn` | 1389 | 1318 inherit the parent env · 38 force blank · **28 drop the lock** · 3 parameterised · 2 set a non-blank |
| `session_lock` | 58 | 43 preserve the blank · 10 force blank · 2 guarded by an existing value · 1 dynamic · 1 sets a non-blank · 1 passthrough |
| `device_override` | 11 | the `MOLMETAL_SMOKE_DEVICE` / `MOLMETAL_DEVICE` idiom (the good pattern) |

By surface: **test 1129 · production 673 · gate 181 · launcher 42**.
115 of the 1389 child spawns run a *workload* (train/decode/sweep), not a tool.

**Mask accounting:** 48 sites **force** the blank, 3 sites **defeat** it, 2 sites write
the mask but are **guarded by an existing value**.

### 2.1 Which probes create a GPU *context* — and which create a *primary context*

This distinction is the load-bearing one for the 550 MiB watchdog observation:

* `torch.cuda.is_available()` (242 sites) and `torch.cuda.device_count()` (22 sites) are
  **device-count queries**. From the installed torch source: `is_available()`
  (`torch/cuda/__init__.py:209-229`) takes the `else` branch unless
  `PYTORCH_NVML_BASED_CUDA_CHECK=1`, and that branch calls
  `torch._C._cuda_getDeviceCount()`, whose own comment says it *"uses the CUDA Runtime
  API cudaGetDeviceCount which in turn initializes the CUDA Driver API via cuInit"*.
  `cuInit` is a **driver initialisation, not a primary-context creation**.
  `device_count()` (`:1212-1238`) prefers `_device_count_nvml()` — NVML, no `cuInit` at
  all — and falls back to the `cuInit` path only when NVML discovery fails.
  **Under `CUDA_VISIBLE_DEVICES=""` all 264 of these return "no device" and create
  nothing at all.**
* `torch.cuda.current_device()` (8 sites) and `torch.cuda.set_device()` (2 sites) go
  through `_lazy_init()` → `torch._C._cuda_init()` / `torch._C._cuda_setDevice()`, which
  **retain the device's primary context** — the ~500-600 MiB class of allocation.
  **10 sites total; 0 of them at import time.**
  Under a blank ambient they *raise* instead (`torch/cuda/__init__.py:589-591`: *"This
  function throws if there's a driver initialization error, no GPUs are found or any
  other error occurs"*) — they do not allocate.
* **5 probes run at import time**, all of them `is_available` (driver-init only, no
  primary context):
  `molmetal/utils/device.py:51`, `molmetal/scripts/infer_cfm.py:37`,
  `molmetal/tests/test_cfm_bridge_components_tolist.py:50`,
  `molmetal/tests/test_ot_effective_backend.py:16`,
  `molmetal/tests/test_wf333t2_triton_kernels_full.py:93`.

## 3. Answer to (B): *with `CUDA_VISIBLE_DEVICES=""` in the environment, can any of the repo's own tooling still allocate GPU memory?*

**YES — but only through two mechanisms, and neither is a probe.**

1. **A child that DROPS the lock.** 28 child spawns in 18 files (all under
   `molmetal/tests/`) build a **hand-made env dict** that carries no
   `CUDA_VISIBLE_DEVICES` key. Such a child inherits **nothing** from the parent —
   including the CPU lock that `conftest.py` and `molmetal/tests/conftest.py` write at
   import time — so it sees **every** GPU. Worse, the trainers it invokes call
   `molmetal/scripts/_cuda_default.py::ensure_cuda_visible_devices_default()`, which
   **sets** `CUDA_VISIBLE_DEVICES="0"` when the variable is *unset* — and in these
   children it *is* unset. The child therefore ends up with **GPU 0 visible**. This is
   the chain most consistent with the watchdog's 550 MiB observation: a process that
   never trained but held a primary context on GPU 0.
   *The full list of 28 (file:line) is in the artefact under
   `question_b.lock_droppers`.* Per file: `test_four_domain_sampler.py` (1),
   `test_refactored_train_scripts.py` (2), `test_registry_generate_import_path.py` (1),
   `test_wf125_controlled_pair_wrapper.py` (4), `test_wf330c1_c25_25metals.py` (1),
   `test_wf335r2_wireup_reverify.py` (1), `test_wf337g1_g6_recovery.py` (1),
   `test_wf341c2_edm_precond.py` (3), `test_wf349a1_edm_preconditioning.py` (2),
   `test_wf353b2_wf343c2_c3_verify.py` (1), `test_wf356a2_anchor_refresh.py` (1),
   `test_wf357d1_fused_adam.py` (1), `test_wf361c1_top_m6_kappa_improvement.py` (1),
   `test_wf414a1_m2r2_launcher.py` (3), `test_wf432a2_w246_real_trainer.py` (1),
   `test_wf_blkc_geom_classifier.py` (2), `test_wf_smoke_freshness_gate.py` (1),
   `test_wfa156_text_scan.py` (1).
2. **A launcher that WRITES the mask.** The repo's GPU launchers are a deliberate
   opt-in surface (AGENTS.md §5.1), and `scripts/start.sh` derives the mask from the
   yaml's `compute.gpu`. **That one was already repaired by the concurrent Wf-A165
   branch** (working tree: an explicitly-set ambient mask now wins over the yaml, and
   an empty ambient is kept verbatim with a WARN). After the census classifier fix
   (§2.1 note below) exactly **one** launcher write remains a genuine defeater:
   `scripts/launch_wf308c_todo220_gpu.sh:125`
   (`CUDA_VISIBLE_DEVICES="${WF308C_CUDA_VISIBLE_DEVICES:-0}"`).

**The shell layer.** A wave agent's shell inherits the session environment, so the
blank reaches the shell and every command it runs — the exposure at this layer is a
command that sets the mask *itself* (`CUDA_VISIBLE_DEVICES=0 cmd`), which is a
per-command operator override, not a leak.

**What does NOT allocate under a blank ambient:** every one of the 274 probes (264
mask-respecting queries return "no device"; the 10 primary-context creators raise
rather than allocate), and the 1318 child spawns that inherit the parent's env — a
child inherits the lock **only if the parent had it**, which is why the exposure lives
at the *entry points that build a fresh env*, not at the probes.

**Honest negative:** with the classifier bug fixed, the repo has **zero** tooling sites
that defeat an ambient blank other than (a) the 28 child spawns above and (b) one GPU
launcher. The other two sites the first version of the census flagged as "defeaters"
are **not** defects: `molmetal/tests/test_wfa125_root_conftest_gpu_lock.py:124,146`
deliberately set `CUDA_VISIBLE_DEVICES="0"` — that test's whole purpose is to prove
the conftest blanks it — and 29 shell sites that the first classifier mis-read (see
§5.1).

## 4. The landed Wf-A162 pre-flight repair — four claims, all TRUE

`molmetal/scripts/pre_flight_check.py` is **read-only to this branch**. Its four
claimed properties were checked against the file on disk:

| Claim | Verdict | Evidence |
|---|---|---|
| The session lock short-circuits the smoke device to `"cpu"` **without spawning torch** | **TRUE** | `_session_cuda_is_locked()` at `:13056-13063` (reads the env only, never imports torch); `_smoke_device()` at `:13136-13138` returns `"cpu"` before the `subprocess.run` at `:13144-13150`. |
| A mask that hides every device also short-circuits | **TRUE** | `_visible_device_count_hint()` at `:13066-13080` counts only non-negative entries (a negative index *excludes* a device) → `0`; `_smoke_device()` at `:13139-13141` returns `"cpu"`. |
| The probe carries `PYTORCH_NVML_BASED_CUDA_CHECK=1` | **TRUE** | `_smoke_device()` at `:13144-13150`, `env={**os.environ, "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}` — the NVML branch, no `cuInit`. |
| The two consuming gates FAIL loudly instead of running the smoke | **TRUE** | D-MPNN smoke `:13373-13384` and chemprop smoke `:16953-16964` both call `_smoke_device()` → `_device_authorisation_error()` (`:13083-13100`) and return `GateResult(..., "FAIL", f"{auth_err} (smoke NOT run)", ...)`. |

**No diff is proposed for `pre_flight_check.py`.** It is read-only to this branch and
its repair is correct as landed.

## 5. The uncovered paths

### 5.0 Paths the landed repair does not cover

| # | Path | Why it is uncovered | Exposure |
|---|---|---|---|
| U1 | 28 child spawns in 18 `molmetal/tests/` files | The pre-flight repair is *inside* the pre-flight; it says nothing about a test's own child | The child sees every GPU; the trainers it calls set `CVD="0"` themselves |
| U2 | `molmetal/scripts/infer_cfm.py:37` | A DECODE entry point, not a gate; the Wf-A156 guard deliberately excludes `molmetal/scripts/` (a launch surface) | Import-time `torch.cuda.is_available()` — driver init on import; creates a primary context only if a device is visible |
| U3 | `molmetal/utils/device.py:51` | Library import-time probe; re-exported by `molmetal/utils/__init__.py:8`, imported at module level by `train_fm_pocket.py:74`, `train_hybrid.py:47`, `train_metal_hybrid_v4.py:85`, `train_pocket_flow.py:65`, `train_metal_hybrid_v4_mb1.py:76`, `molmetal/adapters/egnn_predictor.py:26` | Driver init on import; **no** primary context |

U3 is deliberately **not** patched (see §6.2).

### 5.1 A defect in the census itself, found and fixed (self-reported)

The first version of the shell scanner reported **32** "lock defeaters". Reading the
source lines showed most of them were **false positives**:

* an `echo`/`log` message that merely *mentions* `CUDA_VISIBLE_DEVICES` was counted as a
  mask write (`scripts/start.sh:729,739,741`; `scripts/launch_wf131_p3_deferred_runs.sh:241`;
  `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:420`; …);
* a trailing `# comment` that mentions the variable was counted as a write
  (`scripts/launch_wf124_p3_g5.sh:152`; `scripts/launch_wf310b_g6_final.sh:54`; …);
* the nested-brace form `${CUDA_VISIBLE_DEVICES-${GPU}}` was not recognised as the
  `${VAR-default}` family, so three launchers that **preserve** an empty ambient were
  reported as defeating it
  (`scripts/launch_cfm_round7_f5f6_retrain.sh:118`,
  `scripts/launch_dmpnn_clean5v4_loso_d1d3.sh:149`,
  `scripts/launch_pa17_chemprop_v2_moleculenet.sh:62`).

The scanner now strips comments quote-aware, ignores the variable when it appears
*inside* a quoted string, and parses `${NAME<op>word}` with a brace-depth counter.
**A guard that over-reports is worse than no guard**, so the fix is part of the census.
After the fix the count is **3**: **29 of the 32 sites the first classifier reported were
false positives**. Of the 3 that survive, **2 are a deliberate negative control** (the
`test_wfa125` proof that the conftest blanks a caller's mask) and **1 is a GPU
launcher**. The two regression tests that pin this are in
`molmetal/tests/test_wfa163_gpu_tooling_census.py`
(`test_the_shell_classifier_does_NOT_call_an_echo_a_write`,
`test_the_shell_classifier_does_NOT_call_a_comment_a_write`).

### 5.2 What is deliberately NOT patched, and why

* **`molmetal/scripts/pre_flight_check.py`** — read-only to this branch; the landed
  repair is correct (§4).
* **`molmetal/utils/device.py`** — its import-time `ROCM_AVAILABLE` probe is a
  device-count query, so it does **not** create a context; and making it lazy would
  break `molmetal/molmetal_lam/tests/test_device_selection.py:219`
  (`assert isinstance(ROCM_AVAILABLE, bool)`), which lives under
  `molmetal/molmetal_lam/**` — **read-only to this branch**.
* **`scripts/launch_*.sh`** — a deliberate GPU opt-in surface (AGENTS.md §5.1).
  `scripts/start.sh` was already repaired by the concurrent Wf-A165 branch.
* **`molmetal/tests/test_wfa156_text_scan.py`**, **`molmetal/tests/test_wfa125_root_conftest_gpu_lock.py`** —
  `test_wfa*` files, read-only to this branch. Their runtime behaviour is covered anyway
  by the conftest fix (U1), which needs no per-file edit.

## 6. The proposed patch — `patch_smoke_device_lock.diff` (PROPOSED, **NOT APPLIED**)

Applying it is the main thread's job. The diff touches **3 files**, none of them
read-only:

### 6.1 `molmetal/tests/conftest.py` + `conftest.py` (repo root) — closes U1

One choke point: `subprocess.Popen.__init__` is the single funnel through which
`subprocess.run` / `call` / `check_call` / `check_output` / `Popen` all start a child.
When the session lock is ON, the patch injects `CUDA_VISIBLE_DEVICES=""` into any
`env=` that does **not** already carry the key.

* An env that **does** carry the key keeps it **verbatim** (no silent override).
* A spawn with **no** `env=` is unchanged (it already inherits the parent's blank).
* When the lock is OFF (`MOLMETAL_TEST_ALLOW_GPU=1`), nothing is patched at all.

This closes all 28 sites with **one** edit instead of 28.

### 6.2 `molmetal/scripts/infer_cfm.py` — closes U2

The device default moves from an import-time probe to a lazy, lock-aware
`_default_device()`: the session lock wins (`CUDA_VISIBLE_DEVICES=""` → `"cpu"`, with
no probe at all), `MOLMETAL_SMOKE_DEVICE=cpu|cuda` makes the choice explicit (the
convention already used by `molmetal/scripts/metallo_drug_smoke_retrain.py`), and the
default in every other case is unchanged — the GPU whenever one is available.

### 6.3 Verification of the diff

| Check | Result |
|---|---|
| `git apply --check` (repo root, working tree) | **clean** on all 3 files |
| `patch -p1 --dry-run` on a **scratch copy** | **clean** |
| `patch -p1` on the scratch copy, then a functional check | a hand-made env without the key gains `CUDA_VISIBLE_DEVICES=""`; an env that carries the key keeps it verbatim; a spawn with no `env=` is unchanged |
| `infer_cfm.py` on the scratch copy, imported with `torch.cuda.is_available()` booby-trapped | imports cleanly (the import-time probe is gone); `_default_device()` → `"cpu"` under a blank ambient; `parse_args().device` → `"cpu"` |

The repo-root `conftest.py` is **untracked** in git, so applying the patch leaves it
modified-but-untracked — it needs `git add conftest.py` before it can be committed.

**State note.** `molmetal/tests/conftest.py` is *already modified* in the working tree by
another branch (the Wf-A123 published-tree write guard), so the patch's hunk context is
against that working-tree state — it must be applied on top of the current working tree,
not onto a pristine HEAD checkout. `molmetal/scripts/infer_cfm.py` is clean vs HEAD.

## 7. Honest limitations of this census

1. **It is a STATIC census.** It reads source; it does not run the tooling. "This site
   *can* allocate" is a statement about the code path, not a measurement of VRAM. The
   only MEASURED GPU number in this branch remains the watchdog's 550 MiB (2026-10-31),
   which this census can describe but **cannot attribute** to a specific API — the
   ~500-600 MiB class of allocation belongs to a primary context, and the census shows
   10 such sites, none of them at import time and all of them raising under a blank
   ambient.
2. **It covers context creation, not tensor placement.** `.cuda()` / `.to("cuda")`
   calls are the training paths; they are deliberately out of scope ("tooling").
3. **The scope prefixes are `molmetal/`, `scripts/`, `tests/`.** Anything outside them
   (e.g. `paper/`, `docs/`, `tools/` at the root) was not enumerated. 6339 files were
   enumerated in scope; 3011 were decoded.
4. **The file counts are a snapshot.** Other Wf-A163 branches land files while this
   branch runs; a re-run will see a different denominator. The artefact records the
   denominator it was produced against.
5. **Two `test_wfa*` files could not be edited** (read-only to this branch). Their
   behaviour is covered by the conftest fix, but their *source* was not changed.
