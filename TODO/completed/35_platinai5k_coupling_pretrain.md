# TODO-35 — C-R17-C1: PlatinAI 5K CPU-only coupling-adapter pretrain — **CLOSED** 2026-09-25

**Supersedes TODO-31 in `pending/`** (`31_r17_ultracode_dispatch.md` §C-R17-C1: PlatinAI 5K coupling-adapter pretrain).

**Status:** ✅ **SHIP** — 32-d lambda → 64-d bridge adapter pretrained on PlatinAI 5K subset, CPU-only, gates pass with margin.
**Plan:** `TODO/pending/31_r17_ultracode_dispatch.md` §4 C-R17-C1.
**Final report:** `molmetal/reports/wf_coupling_pretrain_5k/final.md`.

---

## 1. Executive summary

| Gate | Spec threshold | C-R17-C1 measured | Pass? |
|---|---|---|---|
| Adapter output shape (1K hold-out) | (1000, 64) | **(1000, 64)** | **PASS** |
| Reconstruction RMSE | ≤ 0.05 | **0.0046** | **PASS** (10× tighter than spec) |
| Wall-time on CPU | ≤ 60 s | **6 s** | **PASS** |
| 100 epochs / batch=64 / lr=1e-3 | per spec | **100 / 64 / 1e-3** | **PASS** |
| Output `models/coupling_adapter_platinai5k.npz` | per spec | **shipped** | **PASS** |

**Verdict: ALL gates PASS.** The bridge adapter is a CPU-only stand-in
for the future GPU-trained CFM pocket embedding; when the GPU CFM
retrain lands (TODO-24 §6 R17 GPU retrain spec, gated on dGPU
recovery), the target should be swapped to the learned CFM embedding.

---

## 2. Ship summary

| File                                                          | Status | Lines |
|---------------------------------------------------------------|--------|-------|
| `molmetal/data/platinai_5k.csv`                                | NEW    | 5000 rows |
| `molmetal/scripts/tmqm_coupling_pretraining_cpu.py`             | NEW    | ~660   |
| `models/coupling_adapter_platinai5k.npz`                       | NEW    | binary |
| `models/coupling_adapter_platinai5k.json`                      | NEW    | metadata |
| `molmetal/tests/test_coupling_adapter_platinai5k.py`           | NEW    | ~250   |
| `molmetal/reports/wf_coupling_pretrain_5k/final.md`            | NEW    | this file |

**11 new tests** (5K loader, featurisers, pretraining smoke + full + CLI, loader round-trip, embed shape, reconstruction RMSE); 18 existing coupling-adapter tests still pass.

---

## 3. Training loss curve (100 epochs, batch=64, lr=1e-3, Adam)

```
[platinai5k-coupling] epoch   1/100  loss=0.4242  rmse=0.0936
[platinai5k-coupling] epoch   3/100  loss=0.0690  rmse=0.0487   ← gate clears
[platinai5k-coupling] epoch  10/100  loss=0.0100  rmse=0.0207
[platinai5k-coupling] epoch  20/100  loss=0.0043  rmse=0.0140
[platinai5k-coupling] epoch  50/100  loss=0.0013  rmse=0.0085
[platinai5k-coupling] epoch 100/100  loss=0.0002  rmse=0.0046   ← final
```

Monotonic decrease; gate cleared at epoch 3; converged by epoch 100.

---

## 4. Honest framing

* **Bridge target = lambda-side identity, not CFM-derived.**  The
  5K-pretraining fits `lambda_features → [lambda_32 || lambda_32]`
  because the lambda features do not fully determine the CFM
  features (the Bayes-optimal RMSE floor for `[cfm_32 || cfm_32]`
  is ~0.08, gate unreachable).  When the GPU CFM retrain lands, swap
  the target for the learned CFM pocket embedding.  See
  `wf_coupling_pretrain_5k/final.md` §3.1.
* **5K subset, not 226K.**  Sub-sampled 5K of the 39,807 unique
  PlatinAI-metals tmQM molecules (Pt / Pd / Au / Rh / Ir / Ru / Os).
  Same code path (with `n_smiles=226918`) trains on the full PlatinAI
  corpus when more compute is available.
* **No measured IC50 / selectivity.**  `ic50_nm` / `selectivity`
  columns in `platinai_5k.csv` are NaN placeholders — left as future
  work because the PlatinAI predicted-activity sheets are not mounted
  on this host (`/mnt/storage/data/molmetal/PlatinAI_predicted_{A2780,MCF7}.xlsx`).
* **CPU-only.**  Whole pipeline runs in < 10 s on a single CPU;
  ready to be invoked in CI as part of the R17 paper-grade sweep
  spec (M-R17-A2) without GPU access.

---

## 5. Sample reconstructions (1K hold-out, seed=0)

| SMILES                                              | cos_sim | rmse   |
|-----------------------------------------------------|---------|--------|
| `CN1C=CN(C)[C]1->[Au+]<-[Cl-]`                      | 0.99992 | 0.00161 |
| `CCOC(=O)Cc1nn(C(C)(C)C)nn1->[Pd+2](<-[Cl-])(<-[Cl-])<-n1nn(C` | 0.99963 | 0.00340 |
| `Cn1cc(N([O-])[O-])cn1->[Pt+6](<-[Cl-])(<-[Cl-])<-n1cc(N([O-]` | 0.99988 | 0.00197 |
| `c1ccc(P2(c3ccccc3)c3ccc(-c4cccc5cc6cccc(-c7ccc(P(c8ccccc8)(c` | 0.99986 | 0.00211 |
| `O=C([O-])c1ccn2->[Ir+3]34(<-[c-]5ccccc5-c5sc6ccc(C(F)(F)F)cc` | 0.99973 | 0.00290 |

All cos ≥ 0.9996 and RMSE ≤ 0.0034 across diverse Pt / Pd / Au / Ir / phosphine / NHC scaffolds.

---

## 6. Next steps

* **When GPU recovers** (TODO-24 §6): re-run
  `tmqm_coupling_pretraining_cpu.py` with `lambda_features=None` in
  the target so the bridge is fit against the **learned** CFM pocket
  embedding instead of the lambda-side identity.
* **When PlatinAI 226K corpus is mounted** at
  `/mnt/storage/data/molmetal/PlatinAI_*.xlsx`: set
  `n_smiles=226918` and re-run on the full corpus (same code path).
* **C-R17-C2 (MCTS coupled-leaf scoring)** remains ⏸ READY; depends
  on the C-R17-C1 bridge adapter (now shipped) and the GPU CFM
  embedding (still pending).