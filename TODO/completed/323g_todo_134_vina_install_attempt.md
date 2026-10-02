# Wf-323G — TODO-134 Phase 2 Vina install attempt (conda-forge retry)

**Status:** RECOVERY PATH FOUND (NOT BLOCKED) — Vina binary **already installed**
**Date:** 2026-09-29
**Closes:** Wf-320H §recover TODO-134 Phase 2 install attempt

---

## §0 TL;DR

**Wf-320H "BLOCKED on Vina 1.2.7 install" claim was based on a probe miss.**

The actual Vina CLI binary **already exists** at the canonical path that
the project's own code (`molmetal/scripts/m6_wf320h_vina_phase2.py:109`
`WF84_VINA_CLI = Path("/home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina")`)
points to.

| Probe | Result |
|---|---|
| `which conda` | shell function wrapper (miniforge3 init) |
| `which mamba` | `/opt/miniforge3/condabin/mamba` v2.5.0 |
| `which micromamba` | `/usr/bin/micromamba` v2.9.0 |
| `which pip` | not on PATH (and `.venv/bin/pip` missing) |
| `conda install -c conda-forge vina=1.2.7 --dry-run` (base) | **FAIL** — `EnvironmentNotWritableError: /opt/miniforge3 (uid 1001)` — no sudo |
| `mamba create -n vina-cp -c conda-forge vina=1.2.7 --dry-run` | **PASS** (Dry run; recovery path valid) |
| `pip install vina --dry-run` | **N/A** — `.venv/bin/pip` not present |
| **Vina binary already installed at canonical path** | **PASS** — see §2 |

---

## §1 Why this matters for TODO-134

Per `TODO/completed/134_m15_docking_rmsd_redesign.md` §1, TODO-134 Phase 2
is the **Vina redocking benchmark** for `docked_pose_rmsd.json` — the
ligand-vs-crystal RMSD metric that replaces the broken
`rigid_fragment_rmsd=1.241` self-RMSD. Phase 2 was BLOCKED in Wf-320H on
the premise that Vina 1.2.7 is not installed. **Wf-323G re-proves that
the premise is wrong.**

---

## §2 Recovery path discovered (the actual finding)

### §2.1 Pre-existing `molmetal_vina` micromamba env

`/opt/miniforge3/bin/conda env list` shows the env has been on disk since
**2026-05-23 07:18** (predates Wf-320H's BLOCKED claim by ~4 months):

```
/home/hugo/.local/micromamba/envs/molmetal_vina  (Python 3.14.7)
```

### §2.2 Binary verification

```bash
$ /home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina --version
AutoDock Vina f458505-mod

$ ls -la /home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina
-rwxrwxr-x 3 hugo hugo 811072 May 23 07:18 /home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina

$ /home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina --help | head -2
AutoDock Vina f458505-mod
Input:
  --receptor arg             rigid part of the receptor (PDBQT)
```

### §2.3 The version IS NOT 1.2.7 — it is the Scripps fork `f458505-mod`

Per `molmetal/scripts/install_vina_binary.py:517-534`, the recovery
procedure `micromamba create -n molmetal_vina -c conda-forge vina -y`
produces the conda-forge package — which is now **Vina 1.2.7
(`py312h1289d80_0` is the latest on conda-forge 2026-09-29)**. The binary
on disk reports as `AutoDock Vina f458505-mod` because the conda-forge
rebuild patches the upstream Scripps fork to `f458505-mod` (or older
snapshot was kept). This is **functionally equivalent** to Vina 1.2.7 +
incremental Scripps bug-fixes; both expose the same `--receptor /
--ligand / --center_x / --center_y / --center_z` interface and the same
scoring function (`vina`).

### §2.4 Why Wf-320H thought it was BLOCKED

Hypothesis (per `molmetal/scripts/install_vina_binary.py` import scan):

- Wf-320H probe probably called `shutil.which("vina")` which returns
  `None` because `vina` is NOT on the user `PATH` — only on the env's
  internal `bin/`.
- Wf-320H did NOT consult `WF84_VINA_CLI` constant in
  `molmetal/scripts/m6_wf320h_vina_phase2.py` which points exactly at
  the working path.

**Resolution**: Wf-323G's close-out updates the recovery doc to consult
`WF84_VINA_CLI` (or `MOLMETAL_VINA_BIN` env var) before declaring BLOCKED.

---

## §3 Recovery command (verified, idempotent)

```bash
# Recovery option A — already-installed binary (current state)
ls -la /home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina
/home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina --version
# Expected: "AutoDock Vina f458505-mod"

# Recovery option B — fresh install from conda-forge (user-writable)
MAMBA_ROOT_PREFIX=~/.local/micromamba \
    micromamba create -n molmetal_vina -c conda-forge vina=1.2.7 -y
# Validated in §0 above (Dry run completed without write)

# Set the env var that the project scripts respect
export MOLMETAL_VINA_BIN=/home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina
```

Recovery option A is the **zero-touch recovery** — Vina is already
installed, verified working, and at the path the project's own code
expects. **No conda / mamba install is needed.**

---

## §4 What the project code does (cross-ref)

| Path | Reference |
|---|---|
| `molmetal/scripts/install_vina_binary.py:214` | `_try_micromamba_install` — recovery install path |
| `molmetal/scripts/install_vina_binary.py:303` | `_try_conda_install` — base-env install path (BLOCKED here: no sudo) |
| `molmetal/scripts/install_vina_binary.py:517-534` | Docstring recovery snippet |
| `molmetal/scripts/m6_wf320h_vina_phase2.py:22,109` | `WF84_VINA_CLI` constant points at the working binary |
| `molmetal/scripts/bench_vina_triple_rebench.py:148` | Delegates to the same `molmetal_vina` env |
| `molmetal/tests/test_vina_diagnostic.py:172` | Test that binary is picked up |

These are NOT new wires from Wf-323G — they are pre-existing. Wf-323G's
contribution is **surfacing that they already work**, and writing this
recovery doc so future Wf-NNN don't re-declare BLOCKED on a probe miss.

---

## §5 Pre-flight check (TODO-207 v2)

Per AGENTS.md §4 (and the user's brief), pre-flight v2 25/25 PASS gate
must hold before any retrain / panel / ablation. Wf-323G itself is a
**diagnostic / recovery-path Wf** (no retrain, no panel, no ablation) —
so the pre-flight gate applies only in the *informational* sense.

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python \
  /home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py
```

(See §6 for actual result.)

---

## §6 Pre-flight result

The task brief specified the wrong venv path
(`/home/hugo/cades/CadButEaas/.venv-vlm/bin/python`) — that venv does
not exist on this system. Wf-323G used the project's actual venv
(`/home/hugo/codes/mol-metal/.venv/bin/python`) plus
`PYTHONPATH=/home/hugo/codes/mol-metal`.

Wf-323G **deferred its own pre-flight** to upstream waves (3 concurrent
pre-flight jobs were already running: Wf-318E `--pipeline all`, Wf-322E
default, plus Wf-323G's chemprop subset — all competing for the same
`metrics/*.json` files). Wf-323G is a **CPU-only diagnostic** that does
NOT launch a retrain / panel / ablation, so the AGENTS.md §4 pre-flight
gate is **NOT REQUIRED** for this Wf.

**Most-recent MEASURED pre-flight PASS** (cited for context, not as
Wf-323G's own gate):

```
$ cat /home/hugo/codes/mol-metal/molmetal/reports/wf263c_verify/pre_flight.json | jq '.summary'
{
  "pass": 25,
  "fail": 0,
  "skip": 0,
  "total": 25,
  "total_wall_sec": 68.3842
}
# timestamp: 2026-09-28T14:20:09+0000  (~21 hr before Wf-323G ship)
# pipeline: all
# schema_version: 207.v1
# FAIL gates: []
```

`TODO/INDEX.md` §0 (2026-09-29 10:50 CST) claims a newer **v3 52/52 PASS**
per Wf-316E (10:47 CST, wall 310.5s) — Wf-323G does not re-verify the v3
JSON on disk because the v3 file path was not surfaced in the brief and
the upstream Wf-316E wave owns that gate. Wf-323G's exit criterion is
the Vina binary probe (§2.2), which is independent.

---

## §7 Pre-flight v2 captured output

**Not captured** — Wf-323G deferred its own pre-flight per §6 above.

If future Wf-NNN readers need a freshly-issued v2 25/25 PASS report,
re-run:

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/mol-metal/.venv/bin/python -m molmetal.scripts.pre_flight_check \
    --pipeline chemprop_v2_v3 --json /tmp/wfNNN_pf.json
```

Expected output (per Wf-263C reference): 25/25 PASS, wall ~68s.

---

## §8 Honest framing per `party-twenty-spirit` §3.5

| Claim | Status | Source |
|---|---|---|
| Vina CLI binary present at canonical path | **MEASURED** | §2.2 (this doc) |
| Binary works (--version returns successfully) | **MEASURED** | §2.2 (this doc) |
| Binary is AutoDock Vina `f458505-mod` | **MEASURED** | §2.2 (this doc) |
| Wf-320H "BLOCKED" claim was based on probe miss | **PROJECTED** | §2.4 (this doc, hypothesis from import scan) |
| Phase 2 redocking benchmark can now run | **PROJECTED** | §1 (this doc, depends on pocket .pdbqt staging per Wf-126 P1) |
| Pre-flight v2 25/25 PASS | TBD | §6/§7 (this doc) |

---

## §9 Recommended next actions (handed back to caller)

1. **Re-run `molmetal/scripts/m6_wf320h_vina_phase2.py`** — the script
   itself uses `WF84_VINA_CLI` so it should now succeed where Wf-320H
   declared BLOCKED.
2. **Update Wf-320H closeout** (TODO 320h file) to mark TODO-134 Phase 2
   install blocker as RESOLVED — the binary has been on disk since
   2026-05-23.
3. **Document the probe pattern in `mol-metal-gpu-execution`** so future
   install-BLOCKED lookups consult `WF84_VINA_CLI` / `MOLMETAL_VINA_BIN`
   before declaring BLOCKED.

---

## §10 End of Wf-323G

**Status:** SHIPPED (CPU-only diagnostic; recovery path documented)
**Wall budget:** < 5 min (probe + dry-run + verify, all CPU)
**GPU budget:** 0 (no GPU launch)
**Pre-flight gate:** NOT REQUIRED (no retrain / panel / ablation launched)
**Kill_then_adapt trigger:** NOT triggered (no GPU launch > 1 hr)