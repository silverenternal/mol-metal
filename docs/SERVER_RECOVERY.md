# Restoring a working Mol-Metal checkout

> **Current recovery procedure — 2026-11-11.** This section supersedes the
> older instructions below. The asset archives were built from the existing
> server files; no source dataset was re-downloaded for this recovery.

A full (not shallow) clone preserves the Git history required by provenance
checks. Install Python 3.12, `uv`, GitHub CLI (`gh`), `zstd`, `unzip`, `tar`
and `sha256sum`; ensure the destination has enough space for the release
assets, the extracted ChEMBL and BindingDB files, and the environment.

```bash
git clone https://github.com/silverenternal/mol-metal.git
cd mol-metal
uv sync --locked
mkdir -p ../molmetal-recovery-assets
gh release download server-recovery-20261111 \
  --repo silverenternal/mol-metal --dir ../molmetal-recovery-assets
scripts/restore_server_assets.sh ../molmetal-recovery-assets
CUDA_VISIBLE_DEVICES='' .venv/bin/python molmetal/scripts/registry.py check
CUDA_VISIBLE_DEVICES='' .venv/bin/python molmetal/scripts/pre_flight_check.py --pipeline all
```

The restore script verifies every release asset against
[`data/recovery/assets.sha256`](../data/recovery/assets.sha256), refuses to
replace a local file with different contents, extracts the selected training
inputs and referenced evidence, reconstructs both large databases, and checks
their hashes. It recreates two relative symlinks needed by historical tools.
The original server also stores pocket and cross-docking inputs in a separate
`pocket/data` tree; those external data are **not** inside this repository or
the release assets. When that existing tree is available, recreate the links
without downloading data:

```bash
.venv/bin/python scripts/restore_pocket_links.py /path/to/pocket/data
```

The tool checks every target before creating links and refuses to overwrite
existing paths. If the external tree is unavailable, pocket/cross-docking work
remains blocked until it is supplied.

The release retains `chemrxiv-2025-pp32k.zip` and `v1.zip` as original
supplementary source archives. They are verified by the restore script but
are left in the asset directory for manual inspection rather than extracted.

The release also contains `repository-refs.bundle`, holding the old server's
reachable Git refs, and `worktrees-changes.tar.gz`, holding patches and
untracked source files from its secondary worktrees. Those patches are **not**
automatically applied. Inspect that archive's `manifest.json` and apply only
changes you need. To inspect the old branch history without replacing the
current branch:

```bash
git bundle verify ../molmetal-recovery-assets/repository-refs.bundle
git fetch ../molmetal-recovery-assets/repository-refs.bundle \
  '+refs/heads/*:refs/remotes/server-backup/*'
```

If this repository is restored at a different path, legacy YAML files may
contain the former server's absolute root. Create a relocated copy before
using such a config; keep the original for provenance:

```bash
.venv/bin/python scripts/relocate_config.py configs/cfm_5metal_default.yaml \
  /tmp/molmetal-relocated-cfm.yaml
```

The tool replaces only the old repository root in YAML string values and
refuses to overwrite an existing destination. Choose the relocated config in
the appropriate launcher and still run pre-flight before training. Shell
scripts and non-YAML files may have independent host-specific paths.

A clean clone, a new virtual environment, local restoration from the archives,
and the core data-loader, predictive-index and path-reference tests were
verified on the source server. The full pre-flight still reports failures in
historical budget, configuration and run-ledger provenance checks; do not
launch training until those are resolved. A separate machine has not been
verified. Some deliberately deleted diagnostics and intermediate checkpoints
are not included; the large pre-featurized tensors must be regenerated if
needed. Read [`reports/server_recovery_validation.json`](../reports/server_recovery_validation.json)
for the dated measurements and remaining limits.

---

## Earlier recovery notes (historical)


This document describes the files preserved in Git and the gaps that remain.
Do not delete the original server copy until a fresh clone has been checked
against the data and experiments you need.

## Checkout and environment

```bash
git clone https://github.com/silverenternal/mol-metal.git
cd mol-metal
uv sync --locked
source .venv/bin/activate
```

The project targets Python 3.12. The dependency configuration specifies CUDA
13 PyTorch and Triton; the appropriate PyTorch build and optional packages
may differ on another machine. `uv sync --locked` and the console entry point
were verified in a new virtual environment in a GitHub clone on the source
server; that is not an independent test on a different host. Run
`python -m molmetal --help`, followed by the
project's relevant CPU tests. Before any retraining, run
`molmetal/scripts/pre_flight_check.py` and resolve its reported failures.

## Data and checkpoints in this snapshot

- `data/msrs/measurements.parquet` and `measurements_dedup.parquet` are tracked
  for the predictive line.
- `.data/molmetal/MetalCytoToxDB.csv`, the CFM cohort CSVs, and the v5 split manifest are included despite the general `.data` ignore rule. The
  source dataset is MetalCytoToxDB by Krasnov et al.; cite the authors and
  https://doi.org/10.1021/acs.jmedchem.5c02755 when using it. The dataset is
  available at https://zenodo.org/records/17106822.
- `molmetal/checkpoints/wf194_lomo_v2/` and
  `molmetal/checkpoints/wf219_krasnov_aligned/` preserve complete on-disk
  multi-seed sets for their respective historical experiments.
- `.ckpt/tmqm_pretrained_dmpnn.pt` preserves a pretrained encoder;
  `results/wfa108j_gcv36/checkpoints/cfm_yuelbond_wasserstein_seed0.pt` and
  `results/wfa108j_budget36/checkpoints/cfm_yuelbond_wasserstein_seed0.pt`
  preserve two CFM checkpoints. These are retained research artifacts, not a
  claim that they are the best models across all protocols.
- Other ignored checkpoints, cached features, downloaded datasets, external
  models, and ignored reports are not comprehensively preserved by Git.
  Git-tracked data and model paths should be checked against the task's own
  config and metric artifact before use. Do not assume an old report's path
  still exists.

The large `molmetal/data/pre_featurized/*.pt` caches were deliberately removed.
The D-MPNN pretraining loader fails without them; regenerate with
`.venv/bin/python molmetal/scripts/pre_featurize_dmpnn.py` only after restoring
all of its input datasets and checking disk/memory requirements. This rebuild
has not been validated on a clean host. Large external datasets such as BindingDB,
ChEMBL and NCI-60 GI50 and optional tools need to be obtained separately; consult
`DATA_STATUS.md` and the corresponding loaders. Some historical paths and
symlinks are host-specific, and some documentation describes older hardware or
superseded metrics.

This snapshot does not prove that every experiment is reproducible or every
historical metric's checkpoint is available. See the cleanup reports in
`reports/` and `docs/PREDICTIVE_LINE_INDEX.md` before treating a result as
current.

## Flow Matching reference files

The local Flow Matching reference files used by this snapshot are included at
`molmetal/references/flow_matching/`. Their symlinks are repository-relative.
They preserve the files used in this working tree; they are not a complete
upstream checkout. Some optional integrations may require separate upstream
software and must be checked against the task being run.

## Restore the external registry data archive

The GitHub release `server-recovery-20261111` contains the NCI-60 GI50 file
and the registered Rusanov supplementary files that are too large or unsuitable
for ordinary Git blobs. Restore from the repository root:

```bash
gh release download server-recovery-20261111 \
  --repo silverenternal/mol-metal \
  --pattern external-registry-data.tar.gz --dir /tmp
sha256sum /tmp/external-registry-data.tar.gz
# Compare with archive_sha256 in data/recovery/external-registry-data.json.
tar -xzf /tmp/external-registry-data.tar.gz -C .
```

The manifest records every archived path and SHA-256. Do not extract over
locally modified files without checking them first. The archive supplements
Git; it does not include BindingDB, ChEMBL, raw tmQM, every historical
checkpoint, or all ignored reports. The data and original publications carry
their own attribution and reuse terms.

## Validation and remaining blockers

`reports/server_recovery_validation.json` records the dated verification of a
GitHub checkout and downloaded release archive using the old server's Python
environment. Its full pre-flight did **not** pass. In particular, the lock
file has since been installed in a new environment on the same server; two
metric registry entries remain stale and several gates still fail. This is not
proof that it is safe to discard the source server. Check the report and make
an independent backup of any ignored data or weights needed by your workflows.
