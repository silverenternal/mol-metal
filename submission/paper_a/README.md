# Paper A — reproducibility package

Self-contained kit for the CPU-reproducible numbers in the Paper-A manuscript
(*metal-specific activity does not transfer across metals*). It regenerates each
cited artifact from a committed command and diffs the result against the shipped
copy, so a reviewer can confirm the paper's CPU numbers are reproducible without
a GPU.

## Contents

| Path | What it is |
|---|---|
| `reproduce.sh` | driver — runs the CPU subset into `./.repro_out`, diffs, exits 0 iff all match |
| `paper_a/repro/metric_manifest.json` | the SSOT: every cited artifact, its producer, exact command, inputs, cost class |
| `scripts/` | the driver (`reproduce_paper_a.py`), the tolerance diff (`_metric_diff.py`) |
| `molmetal/` | the producer scripts and their import closure (vendored) |
| `molmetal/reports/` | the GPU-produced input reports the CPU syntheses read |
| `metrics/by_metric/` | the shipped artifacts the paper cites |
| `configs/` | the YAML configs for the GPU retrains |
| `folds/` | the DOI-split 10-fold files for the LightGBM entry |
| `env/requirements.txt` | pinned Python dependencies (MEASURED) |
| `data/README.md` | where to put the one raw database file |
| `LICENSE` | Apache-2.0 |

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt

# MetalCytoToxDB.csv is third-party — see data/README.md
mkdir -p .data/molmetal
cp /path/to/MetalCytoToxDB.csv .data/molmetal/

./reproduce.sh              # ~10 min CPU; prints PASS/FAIL/SKIP per entry
```

Each entry prints one line: `PASS <name>: matches metrics/by_metric/<file>`. A
non-zero exit means at least one CPU-reproducible entry did not regenerate.

## Cost classes

The manifest tags every entry. The driver runs only `cpu_seconds` / `cpu_minutes`
by default.

- **cpu_seconds / cpu_minutes** — run here; deterministic given the seed.
- **gpu_minutes / gpu_hours** — skipped unless `--include-gpu-long`. These
  retrain a model and are *not* bit-reproducible (bf16/cuDNN); the manifest says
  which ones pin their command and which do not.
- **blocked** — the producer is not committed (a superseded artifact); kept only
  as a historical record.

## Tolerance, not byte equality

XGBoost and LightGBM reduce over threads in a nondeterministic order, so two
runs of the *same* command differ in the last bits. The driver therefore compares
structurally with `abs_tol 1e-6` (`scripts/_metric_diff.py`): same keys, same
list lengths, floats equal within tolerance. A byte diff would fail on a correct
run; the tolerance diff fails only on a real change.

A few artifacts echo an input *path* into their output (e.g.
`metalcytotox_ru_paper_lightgbm_cv.json::fold_source`, which is the verbatim
`--folds-dir` argument). Such environment-specific labels are excluded per entry
via the manifest's `diff_ignore`; the underlying inputs are still checked
transitively by the measured values (for that entry, the 10 per-fold AUCs).

## What is *not* reproducible here (stated, not hidden)

- **GPU entries.** Retraining the D-MPNN / chemprop backbones needs a GPU and is
  not bit-reproducible; the shipped artifacts are the measured runs. Where the
  exact command is not pinned in the artifact, the manifest marks the entry
  `reproducible_from_committed_code: false` and says so.
- **The PA3 within-metal CI.** The PA3 report stores per-fold AUCs but not the
  per-ligand out-of-fold scores a within-metal bootstrap needs, so
  `within_metal_bootstrap_ci` / `transfer_penalty_ci` are absent for the backbone
  column; only its cross column carries a bootstrap interval. Recorded in the
  manifest and the artifact.

## Provenance

`paper_a/repro/metric_manifest.json::source_commit` records the repository
commit this package was cut from. The vendored `molmetal/` tree is the exact
import closure of the CPU producers at that commit (measured, not guessed).
