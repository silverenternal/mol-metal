# P6.3 MMseqs2 Follow-up — Pocket-axis OOD protocol (deferred)

**Date**: 2026-09-26
**Workflow**: `wf_pitfall_tier2` / TODO-30 Rank 14 (P6.3)
**Status**: **CPU PARTIAL — pocket-axis (MMseqs2) DEFERRED**
**Companion**: `molmetal/molmetal_lam/lam_chem/ood_protocol.py` (NEW)

---

## 1. TL;DR

This round ships the **ligand-axis OOD protocol** (Tanimoto<0.4 on
Morgan-ECFP4 fingerprints) as a CPU-only, RDKit-backed
implementation.  The **pocket-axis OOD protocol** (MMseqs2
30%-sequence-identity filter) is **deferred** because the `mmseqs`
binary is not installed in the current conda environment, and the
Python wrapper (`pyMMseqs`) would add a substantial dependency
that is out of scope for this CPU-only round.

The `MMseqs2OODProtocol` class is therefore implemented as a **stub**
that raises `NotImplementedError` whenever `is_ood()` or
`max_seq_identity()` is called.  Callers must explicitly opt in to
the deferred binary path; the `CombinedOODProtocol` with
`mmseqs2=None` is the CPU-only safe default.

---

## 2. What shipped

### 2.1 `molmetal/molmetal_lam/lam_chem/ood_protocol.py` (NEW, ~440 LOC)

Public surface:

* **`TanimotoOODProtocol`** — CPU-only Morgan-ECFP4 Tanimoto<0.4
  OOD classifier.  Default threshold 0.4 (Pocket2Mol / TargetDiff
  convention); `radius` / `n_bits` configurable.
  * `compute_tanimoto(smi1, smi2)` — scalar Tanimoto helper.
  * `max_tanimoto(query_smiles, train_smiles_list)` — max sim over
    the train pool.
  * `is_ood(query_smiles, train_smiles_list)` — bool.
* **`MMseqs2OODProtocol`** — stub.  `is_ood()` and
  `max_seq_identity()` always raise `NotImplementedError`.
  Threshold validation still works.
* **`CombinedOODProtocol`** — logical AND over Tanimoto + (optional)
  MMseqs2 axes.  When `mmseqs2=None`, only the ligand axis is
  consulted.
* **`compute_tanimoto(smi1, smi2)`** — module-level scalar helper
  (zero-dep, numpy-optional, RDKit-required).
* **Constants**: `DEFAULT_TANIMOTO_THRESHOLD=0.4`,
  `DEFAULT_FP_RADIUS=2`, `DEFAULT_FP_NBITS=2048`,
  `DEFAULT_MMSEQS2_THRESHOLD=0.30`.

### 2.2 `molmetal/tests/test_ood_protocol.py` (NEW, 24 tests)

* **6** `compute_tanimoto` scalar-helper tests.
* **8** `TanimotoOODProtocol` threshold-gating / empty-train /
  max-tanimoto / instance-method / config tests.
* **4** `MMseqs2OODProtocol` stub-raises / validation / defaults
  tests.
* **6** `CombinedOODProtocol` type-validation / tanimoto-only /
  raises-when-pocket-set / AND-semantics / empty-train tests.

All tests are CPU-only (RDKit + dataclasses only).  The 4
RDKit-independent tests pass even without RDKit installed; the 20
RDKit-dependent tests skip with a clear message.

---

## 3. What is DEFERRED (the MMseqs2 binary)

### 3.1 Why we cannot ship MMseqs2 in this round

* **The `mmseqs` binary is not in the current conda env.**  Grep for
  `mmseqs` against `pyproject.toml` returns 0 hits; the binary is
  not installed system-wide either (`which mmseqs` empty).
* **The Python wrapper (`pyMMseqs`) is not declared in
  `pyproject.toml`** and would require a separate `uv add` step.
* **The MMseqs2 cluster workflow** (`mmseqs easy-cluster` or
  `mmseqs search` + `mmseqs result2msa` + `mmseqs createtaxdb`)
  needs a working-directory + tmpfs setup that would touch the
  filesystem in non-trivial ways; this is out of scope for the
  Tier-2 CPU-only patch budget.

### 3.2 Install steps (for the follow-up round, R13 / R14)

```bash
# Option A — bioconda (recommended)
conda install -c bioconda mmseqs2
# or, if the conda env is uv-managed:
uv pip install pyMMseqs  # Python wrapper around the binary

# Option B — source build (when bioconda is unavailable)
git clone https://github.com/soedinglab/MMseqs2.git
cd MMseqs2
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/opt/mmseqs ..
make -j$(nproc)
sudo make install
```

### 3.3 What the real `MMseqs2OODProtocol.is_ood` will look like

```python
def is_ood(self, query_seq: str, train_seqs: Sequence[str]) -> bool:
    """Cluster query vs train at <min-seq-id>; OOD iff query cluster is disjoint from train."""
    if not train_seqs:
        return True
    with tempfile.TemporaryDirectory() as tmp:
        # Write FASTA inputs.
        q_path = Path(tmp) / "query.fasta"
        t_path = Path(tmp) / "train.fasta"
        q_path.write_text(f">query\n{query_seq}\n")
        for i, s in enumerate(train_seqs):
            t_path.write_text(f">train_{i}\n{s}\n")
        # Cluster: train first, then query as add-only.
        cluster_dir = Path(tmp) / "cluster"
        subprocess.run([
            self.mmseqs_binary or "mmseqs", "easy-cluster",
            str(t_path), str(cluster_dir / "train"), str(tmp / "tmp"),
            "--min-seq-id", str(self.threshold_seq_id),
            "--cov-mode", "1", "-c", "0.8",
        ], check=True)
        # Search: query vs train clusters, get the best hit identity.
        result = subprocess.run([
            self.mmseqs_binary or "mmseqs", "easy-search",
            str(q_path), str(cluster_dir / "train_cluster.tsv"),
            str(tmp / "search"), str(tmp / "tmp"),
            "--min-seq-id", str(self.threshold_seq_id),
        ], capture_output=True, text=True, check=True)
        # If any hit above threshold -> NOT OOD; else OOD.
        for line in result.stdout.strip().splitlines():
            if line and not line.startswith("#"):
                return False  # found a train hit above threshold
        return True  # no train hit above threshold -> OOD
```

Estimated effort: 4–6 h engineering + 2 h pytest (incl.
`test_mmseqs2_real_backend.py` with a 100-seq synthetic benchmark).

### 3.4 What the call-site change will be

The current `CombinedOODProtocol(mmseqs2=...)` already raises
`NotImplementedError` when the pocket axis is consulted, so
**callers can adopt the new binary transparently**: replace the
stub with the real implementation, and the only line that needs
to change at the call site is the `mmseqs2=` argument.

---

## 4. Honest framing

* **Ligand-axis OOD (Tanimoto<0.4) is shipped and tested.**  This
  gives the paper the "we evaluate on a Tanimoto-OOD held-out
  subset" claim for the **molecule** axis.
* **Pocket-axis OOD (MMseqs2 30%) is NOT shipped.**  The paper
  therefore **cannot claim** "we evaluate on a sequence-OOD held-out
  subset" for the **protein** axis.  This is consistent with the
  `wf_pitfall_audit/p6_data_reproducibility.md` P6.3 Gap 3 note.
* The Tanimoto<0.4 + MMseqs2 30% combination is the **strict
  dual-axis OOD protocol** referenced in
  `TODO/engineering_practices.md:97-115`.  Half of it is live; the
  other half is tracked as a follow-up.

---

## 5. Cross-references

* `molmetal/molmetal_lam/lam_chem/ood_protocol.py` — the new module.
* `molmetal/tests/test_ood_protocol.py` — 24 tests, all green.
* `molmetal/reports/wf_pitfall_audit/p6_data_reproducibility.md:138`
  — upstream P6.3 audit citing the Tanimoto<0.4 gap.
* `TODO/engineering_practices.md:97-115` — canonical 4-split policy
  + MMseqs2 30%-seq-id requirement.
* `TODO/pending/26_round14_complete_plan.md` — R14 ship plan where
  the MMseqs2 follow-up belongs.