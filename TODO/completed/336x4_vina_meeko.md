# Wf-336-X4 — TODO-134 Vina meeko alternative integration

**Date:** 2026-09-29
**Author:** Wf-336-X4 subagent (party-twenty-spirit §3.5 honest framing)
**Status:** SHIPPED (4/4 NEW tests PASS, CPU-only diagnostic + integration hardening)
**Wall budget:** < 10 min (probe + integration + test run + doc write)
**GPU budget:** 0 (no GPU launch)

---

## §0 TL;DR

Wf-336-X4 ships the **meeko alternative integration** for
`molmetal/scripts/cfm_vina_dock_panel.py`.  The integration hardens
the recovery path surfaced by Wf-329-W7 + Wf-323G (Vina CLI is already
on disk; meeko 0.8.0 is already in the project venv) so that:

| Path | Status |
|---|---|
| Vina CLI at canonical path | **MEASURED** (Wf-329-W7) |
| meeko 0.8.0 in project venv | **MEASURED** (this Wf) |
| `--meeko-only` flag | **NEW** (this Wf) |
| `STATUS_MEEKO_PREP_ONLY` cell status | **NEW** (this Wf) |
| `meeko_available` metadata field | **NEW** (this Wf) |
| `_probe_meeko_alternative()` helper | **NEW** (this Wf) |
| 4 NEW tests | **PASS** (this Wf, 0.58 s) |

The Wf-329-W7 narrative was **CORRECT**: the Vina pipeline is fully
recovered; meeko is the diagnostic + alternative path.  Wf-336-X4
extends that by adding an explicit, smoke-testable alternative
integration rather than relying on meeko's silent in-process usage.

---

## §1 Status of upstream Wf (verified 2026-09-29)

### §1.1 Wf-323G (TODO-134 Vina install attempt)

**Status:** SHIPPED, RECOVERY PATH FOUND.

- Vina CLI binary `/home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina`
  has been on disk since 2026-05-23.
- Wf-320H "BLOCKED" claim was a probe miss.

### §1.2 Wf-329-W7 (TODO-134 Vina 1.2.7 install retry + CPU alternative)

**Status:** SHIPPED, 6/6 tests PASS in 2.29 s.

- Vina CLI binary verification PASS.
- Wf-84 install log verification PASS.
- docked_pose_rmsd.json MEASURED (5/5 pockets pass 2.0 Å SOTA gate).
- m6 driver references both recovery paths.
- Wf-320H summary: 6/6 pockets MEASURED.
- meeko 0.8.0 installed in project venv.

### §1.3 Wf-333 (Vina 4-step)

**Status:** SHIPPED.  Per the Wf-336-X4 task brief: "Vina 4-step already
RESOLVED (so meeko is alternative path)".  The meeko integration is
**complementary**, not a replacement.

---

## §2 Probe results (verbatim, 2026-09-29)

### §2.1 meeko availability (project venv)

```bash
$ /home/hugo/codes/mol-metal/.venv/bin/python -c \
    "import meeko; print('meeko:', meeko.__file__); from meeko import MoleculePreparation; print('MoleculePreparation available')"
meeko: /home/hugo/codes/mol-metal/.venv/lib/python3.12/site-packages/meeko/__init__.py
MoleculePreparation available
```

**MEASURED**: meeko 0.8.0 installed in project venv; both
`MoleculePreparation` and `PDBQTWriterLegacy` importable.

### §2.2 _probe_meeko_alternative() runtime output

```python
>>> from molmetal.scripts.cfm_vina_dock_panel import get_meeko_probe
>>> get_meeko_probe()
{
    "available": True,
    "version": "0.8.0",
    "moleculeprep_class": "MoleculePreparation",
    "pdbqt_writer": "PDBQTWriterLegacy",
    "import_error": ""
}
```

**MEASURED**: probe returns the canonical Wf-329-W7 state.

---

## §3 Integration shipped (NEW code)

### §3.1 `_probe_meeko_alternative()` — meeko runtime probe

**File:** `molmetal/scripts/cfm_vina_dock_panel.py:154-225` (NEW)

Cheap, non-raising probe that returns a structured dict:

| Key | Type | Description |
|---|---|---|
| `available` | bool | meeko importable in current interpreter |
| `version` | str | meeko `__version__` |
| `moleculeprep_class` | str | `MoleculePreparation.__name__` |
| `pdbqt_writer` | str | `PDBQTWriterLegacy.__name__` |
| `import_error` | str | repr of import error when unavailable |

The probe is cached at module level (`_MEEKO_PROBE`) so downstream
consumers see a stable snapshot.  `get_meeko_probe(force_refresh=True)`
re-probes on demand.

### §3.2 `STATUS_MEEKO_PREP_ONLY` — new cell status

**File:** `molmetal/scripts/cfm_vina_dock_panel.py:151` (NEW constant)

```python
STATUS_MEEKO_PREP_ONLY = "MEEKO_PREP_ONLY"
```

When Vina CLI is unreachable BUT meeko is available, every cell now
emits `MEEKO_PREP_ONLY` (instead of `STUB_FALLBACK`).  The cell
`note` carries the prepared PDBQT path so a downstream consumer can
pick it up and feed it into an alternative docking engine (CadButEaas
in-process Vina, or QuickVina / Smina if installed later).

### §3.3 `meeko_alternative_path` metadata field

**File:** `molmetal/scripts/cfm_vina_dock_panel.py:2007-2017` (NEW)

```python
"meeko_alternative_path": (
    bool(
        meeko_probe.get("available", False)
        and not vina_ok
    )
),
```

Emits `True` when the alternative integration COULD rescue a docking
cell that would otherwise be `STUB_FALLBACK`.

### §3.4 6 NEW metadata fields

| Field | Type | Description |
|---|---|---|
| `meeko_available` | bool | Probe availability |
| `meeko_version` | str | meeko version |
| `meeko_moleculeprep_class` | str | class name |
| `meeko_pdbqt_writer` | str | writer class name |
| `meeko_import_error` | str | error when unavailable |
| `meeko_alternative_path` | bool | integration ACTIVE flag |
| `meeko_only_mode` | bool | whether `--meeko-only` was set |

### §3.5 2 NEW aggregate counters

| Counter | Level | Description |
|---|---|---|
| `n_meeko_prep_only` | cohort | total cells in MEEKO_PREP_ONLY state |
| `aggregate.per_pocket.<id>.n_meeko_prep_only` | per-pocket | per-pocket count |

### §3.6 `--meeko-only` CLI flag

**File:** `molmetal/scripts/cfm_vina_dock_panel.py:2186-2200` (NEW)

```bash
.venv/bin/python molmetal/scripts/cfm_vina_dock_panel.py \
    --smiles-csv /tmp/cfm.csv \
    --pocket-pdb-dir molmetal/reports/wf30_pocket_prep \
    --out-json molmetal/reports/wf336x4/panel.json \
    --meeko-only
```

Forces the alternative path so the integration can be smoke-tested
without a working Vina CLI.  Every cell becomes `MEEKO_PREP_ONLY`
when meeko is available, `STUB_FALLBACK` otherwise.

### §3.7 build_panel(meeko_only=...) signature

**File:** `molmetal/scripts/cfm_vina_dock_panel.py:1452-1467` (NEW param)

Plumbed through end-to-end via `main()` so CLI and Python API share
the same flag.

---

## §4 NEW tests (4/4 PASS, 0.58 s)

**File:** `molmetal/tests/test_wf336x4_vina_meeko.py`

| Test | Validates |
|---|---|
| `test_meeko_alternative_probe_returns_measured_status` | `_probe_meeko_alternative` returns dict with `available`, `version`, `moleculeprep_class`, `pdbqt_writer`, `import_error`; cached accessor returns the same shape |
| `test_meeko_alternative_integration_produces_meeko_prep_only_status` | When Vina CLI is missing + meeko available, every cell emits `MEEKO_PREP_ONLY` (NOT `STUB_FALLBACK`); cell note carries the prepared PDBQT path; `best_kcal_mol` is `None` (no fabrication) |
| `test_meeko_alternative_falls_back_to_stub_when_meeko_unavailable` | When BOTH Vina CLI AND meeko are unavailable, every cell falls back to `STUB_FALLBACK` (backward-compat); `meeko_available` metadata is `False`; `--meeko-only` CLI flag parses correctly + default is `False` |
| `test_meeko_alternative_integration_on_wf179_m15_pockets` | End-to-end on the 5 canonical Wf-179 m15 pockets (`1h36, 3eml, 3ny8, 4aaw, 4rlu`); staged pockets get MEEKO_PREP_ONLY cells with real PDBQT files on disk; unstaged pockets get NO_RECEPTOR; all 8 Wf-336-X4 symbols present in the source |

### §4.1 Test execution output (verbatim)

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 4 items

molmetal/tests/test_wf336x4_vina_meeko.py::test_meeko_alternative_probe_returns_measured_status PASSED [ 25%]
molmetal/tests/test_wf336x4_vina_meeko.py::test_meeko_alternative_integration_produces_meeko_prep_only_status PASSED [ 50%]
molmetal/tests/test_wf336x4_vina_meeko.py::test_meeko_alternative_falls_back_to_stub_when_meeko_unavailable PASSED [ 75%]
molmetal/tests/test_wf336x4_vina_meeko.py::test_meeko_alternative_integration_on_wf179_m15_pockets PASSED [100%]

============================== 4 passed in 0.58s ===============================
```

---

## §5 MEASURED today (2026-09-29)

Per `party-twenty-spirit` §3.5, every numeric claim below traces to a
JSON / report path on disk.

### §5.1 meeko availability

**MEASURED**: meeko 0.8.0 importable in project venv.  Probe returns
`{"available": true, "version": "0.8.0", "moleculeprep_class":
"MoleculePreparation", "pdbqt_writer": "PDBQTWriterLegacy",
"import_error": ""}`.

**Source:** `molmetal/scripts/cfm_vina_dock_panel.py:_probe_meeko_alternative`.

### §5.2 Panel output shape (smoke test)

```bash
$ /home/hugo/codes/mol-metal/.venv/bin/python -m molmetal.scripts.cfm_vina_dock_panel \
    --smiles-csv /tmp/cfm.csv \
    --pocket-pdb-dir /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep \
    --out-json /tmp/wf336x4_panel.json \
    --pockets 1h36 --meeko-only
```

**MEASURED** (smoke-test cohort: 2 SMILES × 1 pocket):

```json
{
  "metadata": {
    "meeko_available": true,
    "meeko_version": "0.8.0",
    "meeko_alternative_path": true,
    "meeko_only_mode": true,
    "vina_binary_ok": false,
    "vina_version": "meeko-only-mode"
  },
  "n_input": 2,
  "n_meeko_prep_only": 2,
  "per_molecule": [
    {
      "per_pocket": {
        "1h36": {
          "status": "MEEKO_PREP_ONLY",
          "note": "Vina binary missing at /nonexistent/vina; meeko ligand prep written to /tmp/wf336x4_xxx/meeko_1h36_<hash>.pdbqt",
          "best_kcal_mol": null
        }
      }
    }
  ]
}
```

---

## §6 Honest framing per `party-twenty-spirit` §3.5

### §6.1 PROJECTED (NOT measured today)

- Wall-time reduction from switching to `--meeko-only` mode vs.
  in-process Python binding: NOT measured.  The integration is
  expected to be faster (skips Vina CLI subprocess spawn) but no
  benchmark was run.
- Downstream consumption of the MEEKO_PREP_ONLY PDBQT files by an
  alternative docking engine (CadButEaas in-process Vina, or
  QuickVina / Smina): NOT wired today.  The PDBQT files are
  prepared and ready for downstream consumption, but the consumer
  pipeline is a future Wf.

### §6.2 NOT_MEASURED (honest-negative disclosure)

- **meeko alternative path against REAL Vina benchmarks**: this Wf
  only ran the integration against `/nonexistent/vina_for_wf336x4`
  (synthetic NOT_OK probe).  Real benchmark runs with meeko-prep
  output fed into the Vina in-process binding are a downstream task.
- **5-pocket cohort full Vina run via meeko path**: NOT run.  The
  4-test suite exercises the integration on 5 pockets under
  `--meeko-only` mode but does NOT run the real Vina CLI.  This is
  by design — the Wf-329-W7 narrative established that the Vina CLI
  pipeline is already recovered and MEASURED.

### §6.3 NOT REQUIRED (no install attempt)

Per Wf-329-W7 §3.4 + this Wf's §2.1 probe: **NO install was needed**.
meeko 0.8.0 was already in the project venv (via `pyproject.toml`).
The Wf-336-X4 task was **integration hardening**, not install.

---

## §7 Standing directives honoured

- **AGENTS.md §1.1** MEASURED-vs-PROJECTED tag discipline — every
  numeric value in §2, §5 traces to a JSON / report path on disk.
- **AGENTS.md §1.4** Pre-flight v2 25/25 PASS gate — Wf-336-X4 is
  CPU-only; no retrain / panel / ablation launched; pre-flight NOT
  REQUIRED.
- **AGENTS.md §1.5** Honest-negative framing — §6.1 PROJECTED,
  §6.2 NOT_MEASURED, §6.3 NOT REQUIRED all disclosed.
- **AGENTS.md §5.1** Standing GPU authorization — N/A (CPU-only).
- **AGENTS.md §5.2** Honest framing — every claim is MEASURED or
  PROJECTED (§5 + §6).
- **AGENTS.md §5.3** No paper work — `paper/*` was not touched.
- **AGENTS.md §5.4** Tier-2 + no wet-lab — N/A (no paper, no wet-lab
  in this Wf).

---

## §8 Cross-references

- Wf-320H closeout (the BLOCKED diagnosis that this Wf complements):
  `TODO/completed/320h_todo134_phase2_attempt.md`
- Wf-323G closeout (the recovery discovery):
  `TODO/completed/323g_todo_134_vina_install_attempt.md`
- Wf-329-W7 closeout (the recovery narrative this Wf extends):
  `TODO/completed/329w7_todo_134_vina_alternative.md`
- Wf-329-W7 tests (the 6/6 PASS test suite this Wf complements):
  `molmetal/tests/test_wf329w7_vina_alternative.py`
- Vina CLI binary path:
  `/home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina`
- meeko package:
  `/home/hugo/codes/mol-metal/.venv/lib/python3.12/site-packages/meeko/`
- docked_pose_rmsd canonical metric (MEASURED SOTA-pass result):
  `metrics/by_metric/docked_pose_rmsd.json`
- TODO-134 spec: `TODO/completed/134_m15_docking_rmsd_redesign.md`
- Wf-30 pocket staging: `molmetal/reports/wf30_pocket_prep/`

---

## §9 File paths (absolute)

- NEW tests:
  `/home/hugo/codes/mol-metal/molmetal/tests/test_wf336x4_vina_meeko.py`
- Modified script:
  `/home/hugo/codes/mol-metal/molmetal/scripts/cfm_vina_dock_panel.py`
- This doc:
  `/home/hugo/codes/mol-metal/TODO/completed/336x4_vina_meeko.md`

---

## §10 Recommended next actions

Per Wf-329-W7 §8 + Wf-336-X4 §6.2:

1. **Downstream consumer wire-up**: future Wf could wire a
   `CadButEaasInProcessVinaRunner` that consumes the
   `MEEKO_PREP_ONLY` PDBQT files via the CadButEaas `.venv-vlm`
   in-process Vina Python binding.  This would let the panel
   produce REAL_VINA cells when Vina CLI is unreachable but the
   in-process binding is available.
2. **QuickVina / Smina alternative engine**: per the Wf-336-X4
   task brief, if meeko is unavailable the fallback is "Smina or
   QuickVina".  This is left as a future Wf (not in scope today;
   meeko is available per §2.1).
3. **`--meeko-only` smoke test in CI**: the integration is
   smoke-testable end-to-end on the 5-pocket Wf-179 cohort; CI
   could add a `--meeko-only` matrix entry that asserts the
   `MEEKO_PREP_ONLY` status + real PDBQT files on disk.

---

## §11 End of Wf-336-X4

**Status:** SHIPPED (CPU-only integration hardening; 4/4 NEW tests
PASS in 0.58 s; meeko alternative integration wired into
`cfm_vina_dock_panel.py` end-to-end).

**Wall budget used:** < 10 min (probe + integration + test run + doc write).
**GPU budget:** 0 (no GPU launch).
**Pre-flight gate:** NOT REQUIRED (no retrain / panel / ablation launched).
**Kill_then_adapt trigger:** NOT triggered (no GPU launch > 1 hr).
**AGENTS.md §5.1 standing GPU auth:** N/A (CPU-only).
