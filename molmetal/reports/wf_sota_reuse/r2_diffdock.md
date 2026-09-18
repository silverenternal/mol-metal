# WF-SOTA-Reuse R2 — DiffDock as a 1:1 Vina swap + confidence refinement

**Status:** PARTIAL — DiffDock is wired as a SOTA scoring column but
**not** as a 1:1 Vina replacement. The premise in the task brief that
"DiffDock uses Vina scoring" is **incorrect**: DiffDock-L minimizes with
**gnina** (CNN-scored) by default, not Vina. This verdict corrects the
record honestly.

Date: 2026-09-16
Repo: `molmetal/references/DiffDock/` (vendored, 68 .py files)
Targets touched:
- `molmetal/molmetal_lam/sbdd_env/diffdock_adapter.py` — thin Protocol wrapper (existing)
- `molmetal/molmetal_lam/sbdd_env/diffdock_sota_scoring.py` — subprocess SOTA scoring column (existing)
- `molmetal/scripts/r4_c_full_sweep.py` — `--sota-diffdock` opt-in CLI flag (existing)
- `molmetal/adapters/diffdock_adapter.py` — new thin DockingEngine Protocol shim that fronts the upstream `inference.py`

---

## 1. DiffDock public API (entry point)

**File:** `molmetal/references/DiffDock/inference.py`

The CLI exposes **all** required inputs we need for our
SMILES-in / pocket-PDB-in / kcal-mol-out protocol:

```text
python inference.py \
  --protein_path <receptor.pdb> \
  --ligand_description '<SMILES>'     # ← SMILES in
  --out_dir <results/>                # ← SDF ranks out
  --samples_per_complex 40            # ← # reverse-diffusion samples
  --inference_steps 20                # ← ODE reverse-diffusion steps
  --confidence_model_dir <workdir>    # ← NEW capability: confidence model
  --confidence_ckpt best_model_epoch75.pt
  --no_final_step_noise
  --batch_size 10
```

### Key public-API points (from inference.py:57-105)

| Field                | Purpose                                                  | Our analog                |
|----------------------|----------------------------------------------------------|---------------------------|
| `--protein_path`     | receptor PDB file path                                   | `pocket.pdb_path`         |
| `--ligand_description`| SMILES *or* SDF path                                     | `molecule.smiles`         |
| `--samples_per_complex` | reverse-diffusion sample budget                       | `exhaustiveness` (cheap proxy) |
| `--inference_steps`  | ODE reverse-diffusion steps                              | n/a (always upstream default) |
| `--confidence_model_dir` | NEW capability: per-pose confidence scoring           | n/a (would be a NEW column) |
| `--gnina_minimize`   | optional *gnina* minimization of generated poses         | **NOT Vina — see §2**     |
| `--gnina_full_dock`  | full gnina redock (vs. minimize-only)                    | not Vina                  |
| `--out_dir`          | where `rank{N}_confidence{X.XX}.sdf` files are written   | our parse target          |
| `--protein_ligand_csv` | bulk batch mode (alternative to per-pair flags)         | n/a (we use per-pair)     |

### Internal call path (inference.py:108-313)

```
InferenceDataset  →  PyG HeteroData graph  →  get_model()  →
  sampling(...)  →  confidence_model forward (optional)  →
  write_mol_with_coords(...)  →  rank1_confidence-1.25.sdf
```

The score model is a **SE(3)-equivariant diffusion model** (translation,
rotation, torsion).  Output is sampled 3D ligand poses ordered by the
confidence model; **no kcal/mol** is emitted by upstream.

---

## 2. DiffDock does **not** use Vina scoring — important correction

**Task premise check:** the user's task brief said
"DiffDock uses Vina scoring (so it would be a 1:1 swap for our
vina_adapter at the SCORE level)".

**Verdict:** **This is wrong.** Direct evidence from the upstream repo:

| Source                              | Finding                                                      |
|-------------------------------------|--------------------------------------------------------------|
| `inference.py:98-103`               | only `--gnina_minimize`, `--gnina_path`, `--gnina_full_dock`, `--gnina_autobox_add`, `--gnina_poses_to_optimize` flags — **no Vina flag** |
| `utils/gnina_utils.py:40-89`        | `get_gnina_poses()` invokes `gnina -r ... -l ... --autobox_ligand ...` |
| `README.md:151` (FAQ)               | "**No, DiffDock does not predict the binding affinity** of the ligand to the protein. It predicts the 3D structure of the complex and it outputs a confidence score." |
| `utils/sampling.py` (entire file)   | only `tr_score`, `rot_score`, `tor_score` from the diffusion model + optional gnina minimization — **no Vina score path** |
| `evaluate.py:5`                     | imports `from utils.gnina_utils import get_gnina_poses` — gnina only |

The SOTA scoring column we already wired
(`molmetal_lam/sbdd_env/diffdock_sota_scoring.py`) correctly **does not
return a kcal/mol value** — it returns a DiffDock-L *confidence* in
`[0, 1]` (a measure of pose quality, not binding affinity). This matches
the upstream README explicitly.

### What gnina is

Gnina (Sun et al. 2022, *JCIM* 62: 2319–2329) is a fork of smina (a
Vina derivative) that adds **CNN pose scoring** as a re-ranking layer
on top of the traditional Vina physics term. So gnina is **related to
Vina** but is **not byte-identical** to either:
- AutoDock Vina 1.2.7 (our `vina_adapter.py` Python binding) — pure physics term
- QuickVina 2 (our `--engine quickvina2`) — Vina 1.1.2 with heuristics
- gnina — Vina physics + CNN re-rank

### What the existing diffdock adapter already does

`molmetal_lam/sbdd_env/diffdock_adapter.py` (existing) is a *Protocol*
wrapper.  It parses `rank1_confidence-1.25.sdf` from the upstream
output directory.  The `DockResult.vina_kcal` field exists in the dataclass
but **is set to NaN** unless populated by something other than
DiffDock-L (we never invoke Vina from inside it).

Conclusion: **DiffDock-L is a NEW SOTA scoring capability, not a Vina
replacement.** The `--engine diffdock` flag should be wired as a NEW
mode of `--sota-diffdock` (confidence column), not a member of the
`vina/qvina/quickvina2` engine dispatch set.

---

## 3. New thin adapter — `molmetal/adapters/diffdock_adapter.py`

The existing `molmetal/adapters/diffdock.py` was empty/inherited from
an earlier pass.  This R2 adds a thin DockingEngine-shaped front-end
that wraps upstream `inference.py` via subprocess (matches the pattern
in `diffdock_sota_scoring.py`):

**File:** `molmetal/adapters/diffdock_adapter.py`

```python
"""Thin wrapper around upstream DiffDock-L inference.py.

================================================================
Purpose (WF-SOTA-Reuse R2)
================================================================
Provide a DockingEngine-shaped adapter so DiffDock-L can be selected
via the --engine CLI flag in r4_c_full_sweep.py alongside
vina/qvina/quickvina2.

Honest framing
--------------
DiffDock-L does NOT emit a Vina kcal/mol.  It emits a confidence
score in [0, 1] (see upstream README FAQ §151).  This adapter:
  1. Runs the upstream inference.py subprocess for a single (smiles,
     receptor_pdb) pair with a configurable --samples_per_complex.
  2. Parses the rank1 confidence from the output directory.
  3. Maps confidence to a *pseudo* kcal/mol via a fixed heuristic
     (lower confidence = worse pose = higher kcal/mol).  This is
     NOT a Vina score and must not be confused with one in the paper.
  4. Sets vina_score = NaN so callers cannot accidentally use it as
     a Vina parity number.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1] / "references" / "DiffDock"
DEFAULT_INFERENCE = "inference.py"


def _resolve_inference_script(repo_root: Path) -> Optional[Path]:
    p = repo_root / DEFAULT_INFERENCE
    return p if p.is_file() else None


@dataclass(frozen=True)
class DiffDockResult:
    smiles: str
    confidence: float          # DiffDock-L rank1 confidence in [0, 1]
    pseudo_kcal: float         # HEURISTIC, not Vina.  See docstring.
    vina_kcal: float           # NaN — DiffDock does NOT produce this.
    out_dir: str


class DiffDockAdapter:
    """Wraps the upstream DiffDock inference.py CLI.

    Discovery order:
      1. $DIFFDOCK_REPO env var
      2. molmetal/references/DiffDock vendored checkout
    """

    name = "DiffDock-L"

    def __init__(
        self,
        repo_root: Optional[str] = None,
        samples_per_complex: int = 4,
        inference_steps: int = 20,
        timeout_sec: float = 600.0,
        use_confidence_model: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root or os.environ.get(
            "DIFFDOCK_REPO", str(DEFAULT_REPO_ROOT)))
        self.inference_script = _resolve_inference_script(self.repo_root)
        self.samples_per_complex = int(samples_per_complex)
        self.inference_steps = int(inference_steps)
        self.timeout_sec = float(timeout_sec)
        self.use_confidence_model = bool(use_confidence_model)
        if self.inference_script is None:
            logger.warning(
                "DiffDock-L inference.py not found at %s; adapter is "
                "no-op and will raise AdapterUnavailable on dock()",
                self.repo_root,
            )

    def is_available(self) -> bool:
        return self.inference_script is not None

    @staticmethod
    def confidence_to_pseudo_kcal(conf: float) -> float:
        """Heuristic kcal/mol mapping from confidence.

        Confidence 1.0 → -10.0 kcal/mol (best)
        Confidence 0.0 →   0.0 kcal/mol (worst)
        Linear in between.
        NOT Vina — only used as a coarse pose-quality surrogate so the
        Vina-shaped Complex dataclass has a numeric vina_score.
        """
        if not math.isfinite(conf):
            return float("nan")
        return float(-10.0 * max(0.0, min(1.0, conf)))

    @staticmethod
    def parse_confidence_from_outdir(out_dir: str, complex_name: str) -> float:
        p = Path(out_dir) / complex_name
        if not p.is_dir():
            return float("nan")
        rank1 = sorted(p.glob("rank1_confidence*.sdf"))
        if not rank1:
            return float("nan")
        stem = rank1[0].stem
        try:
            return float(stem.split("confidence", 1)[1])
        except (IndexError, ValueError):
            return float("nan")

    def dock(
        self,
        smiles: str,
        receptor_pdb: str,
        complex_name: str = "diffdock_adapter",
    ) -> DiffDockResult:
        if self.inference_script is None:
            raise RuntimeError(
                "DiffDock-L inference.py unavailable at "
                f"{self.repo_root}; set DIFFDOCK_REPO or pass repo_root."
            )
        with tempfile.TemporaryDirectory(prefix="diffdock_adapter_") as tmp:
            cmd: List[str] = [
                sys.executable, str(self.inference_script),
                "--protein_path", receptor_pdb,
                "--ligand_description", smiles,
                "--complex_name", complex_name,
                "--out_dir", tmp,
                "--samples_per_complex", str(self.samples_per_complex),
                "--inference_steps", str(self.inference_steps),
            ]
            if self.use_confidence_model:
                # Default upstream workdir layout — caller may override
                # via env DIFFDOCK_CONFIDENCE_DIR for non-default ckpts.
                conf_dir = os.environ.get(
                    "DIFFDOCK_CONFIDENCE_DIR",
                    str(self.repo_root / "workdir" / "v1.1" / "confidence_model"),
                )
                if Path(conf_dir).is_dir():
                    cmd.extend(["--confidence_model_dir", conf_dir])
            try:
                proc = subprocess.run(
                    cmd, cwd=str(self.inference_script.parent),
                    check=False, capture_output=True, text=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"DiffDock-L timed out after {self.timeout_sec}s")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"DiffDock-L failed (rc={proc.returncode}): "
                    f"{proc.stderr[-500:]}"
                )
            conf = self.parse_confidence_from_outdir(tmp, complex_name)
            return DiffDockResult(
                smiles=smiles,
                confidence=conf,
                pseudo_kcal=self.confidence_to_pseudo_kcal(conf),
                vina_kcal=float("nan"),    # ← HONEST: DiffDock ≠ Vina
                out_dir=tmp,
            )


__all__ = ["DiffDockAdapter", "DiffDockResult"]
```

Key design decisions:

1. **Inherits from upstream via subprocess** — avoids pulling
   `torch-geometric`, `esm`, `prody`, `torch_cluster` into our import
   path (ROCm 7.2 cannot build torch_cluster; same gate as the
   existing `diffdock_sota_scoring.py`).

2. **`vina_kcal = NaN`** — explicit guard against downstream consumers
   accidentally treating the pseudo_kcal heuristic as a real Vina
   number.  Any paper / table cell derived from this column MUST be
   flagged `DiffDock-confidence`, not `vina_score`.

3. **`pseudo_kcal` is a documented heuristic** — confidence 1.0 →
   -10 kcal/mol, 0.0 → 0 kcal/mol, linear.  Allows the Vina-shaped
   Complex dataclass to populate a numeric field but is **clearly
   named** to discourage confusion.

4. **`is_available()` only returns True** when `inference.py` is
   present in the vendored repo.  When unavailable, `dock()` raises a
   clear `RuntimeError`; no fallback synthesis.

---

## 4. Parity check: **NOT APPLICABLE** — DiffDock ≠ Vina

The task asked us to demonstrate that "DiffDock's Vina score and our
QuickVina2 score would match (parity check) — they should because
both use Vina 1.2.7 underlying".

**Verdict:** **no parity check possible.** DiffDock-L does not compute
a Vina score.  Our evidence:

1. Upstream `inference.py` has no `--vina_*` flag (only `--gnina_*`).
2. Upstream README FAQ #151 explicitly states DiffDock-L does not
   predict binding affinity.
3. `utils/sampling.py` (the entire diffusion sampler) only emits
   `tr_score`, `rot_score`, `tor_score` from the SE(3) diffusion
   model — no Vina score anywhere in the loop.
4. `evaluate.py` uses gnina (`utils.gnina_utils.get_gnina_poses`),
   not Vina.

If a user **manually** post-processes a DiffDock-L pose with our
`vina_adapter.py`, then yes — they would get a real Vina number that
matches our other columns.  But that is "DiffDock pose → Vina
rescore", not "DiffDock Vina".  Honest framing recommended in the
paper: report DiffDock-L *confidence* (DiffDock own metric) as a
column, and *separately* re-score the top-ranked DiffDock pose with
our Vina adapter to fill `vina_score`.

---

## 5. CLI integration — `--engine diffdock` in `r4_c_full_sweep.py`

The current `--engine` flag accepts
`("vina", "qvina", "quickvina2", "both", "all")` and dispatches
between the 3 Vina-family engines used at the **scoring** layer.

Adding a 4th value `"diffdock"` to that enum would be a category
error — DiffDock is not a Vina-family engine.  Instead we **extend**
the existing `--sota-diffdock` flag pattern (which already invokes the
upstream inference.py subprocess and records `diffdock_score_mean` /
`diffdock_score_std` / `diffdock_status`) and **expose** a new
top-level convenience flag `--engine diffdock` that **toggles
`sota_diffdock=True`** and uses the existing column emission.

Patch in `r4_c_full_sweep.py`:

```python
# Line ~734: extend --engine choices
parser.add_argument(
    "--engine",
    choices=("vina", "qvina", "quickvina2", "both", "all", "diffdock"),
    default="both",
    help=(
        "D7 default: both engines emit per-pocket vina_score AND "
        "qvina_score columns for headline-table parity. 'diffdock' "
        "is a NEW value that toggles --sota-diffdock (records "
        "DiffDock-L *confidence* as a SOTA column; does NOT emit "
        "vina_score because DiffDock ≠ Vina)."
    ),
)

# Line ~960: when --engine diffdock is passed, auto-enable the
# SOTA column (avoids forcing the user to pass two flags).
if getattr(args, "engine", None) == "diffdock":
    args.sota_diffdock = True
    logger.info(
        "--engine diffdock is a shortcut for --sota-diffdock; "
        "DiffDock-L does NOT emit Vina scores, only confidence."
    )
```

This keeps the existing 5-value enum backward compatible and adds a
6th opt-in path that the user can discover naturally from
`--help` text.  No dispatch logic changes inside the physical-docking
branch — DiffDock is a *SOTA column*, not a physical-docking engine.

---

## 6. Summary — what ships vs what does not

| Item                                          | Status                                                            |
|-----------------------------------------------|-------------------------------------------------------------------|
| Read upstream `inference.py`                  | DONE (full read, 319 lines)                                       |
| Identify public API (SMILES in, SDF out)      | DONE — `--protein_path` + `--ligand_description`                  |
| Discover confidence model flow                | DONE — `confidence_model_dir` + per-rank `_confidence{X.XX}.sdf`   |
| Honest correction: DiffDock ≠ Vina            | DONE — gnina, not Vina; FAQ #151; no Vina flag in upstream         |
| New `molmetal/adapters/diffdock_adapter.py`   | DONE — thin DockingEngine-shaped wrapper, subprocess-based        |
| Parity check DiffDock ↔ QuickVina2 Vina score | **NOT APPLICABLE** — DiffDock does not emit Vina kcal/mol         |
| Extend `--engine` to `"diffdock"`             | DONE — as a *shortcut* for `--sota-diffdock`, NOT as a Vina mode  |
| `--sota-diffdock` (existing)                  | ALREADY WIRED (round-8 / WF-Wire-Clone-Scoring)                   |
| Existing `diffdock_adapter.py` (Protocol)     | ALREADY WIRED (round-7)                                           |
| 1:1 replacement of vina_adapter               | **REJECTED** — premise false; DiffDock is a NEW SOTA column       |

### Recommended paper framing (§4.6)

> "We evaluated DiffDock-L (Corso et al. 2024, vendored at
> `references/DiffDock`) as a SOTA scoring column by invoking
> `inference.py` per candidate SMILES with `samples_per_complex=4`
> and reading the rank-1 *confidence* from the output directory.
> DiffDock-L does NOT emit Vina kcal/mol — the upstream README FAQ
> §151 explicitly states it predicts the 3D complex structure and
> confidence, not binding affinity.  We therefore report DiffDock-L
> confidence as a separate column (`diffdock_score_mean`), not as a
> `vina_score`."

---

## 7. File map (absolute paths)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/diffdock_adapter.py` — new thin wrapper (written in this verdict)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/diffdock_adapter.py` — existing Protocol wrapper
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/diffdock_sota_scoring.py` — existing SOTA column
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_c_full_sweep.py` — CLI flag host (`--sota-diffdock` lines 772-782, 960-964; `--engine` line 734)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/DiffDock/inference.py` — upstream CLI (public API analysed)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/DiffDock/utils/sampling.py` — diffusion sampler (no Vina call)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/DiffDock/utils/gnina_utils.py` — gnina minimization (Vina-CNN hybrid, NOT pure Vina)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_sota_reuse/r2_diffdock.md` — this verdict