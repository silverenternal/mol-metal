# Wf-361-A1 — Verify Wf-358-B2 equivariant MP scaffold (VERIFIED)

**Workflow**: Wf-361 ultracode residual verification wave / Wf-361-A1 verification
**Date**: 2026-09-29
**Status**: VERIFIED (Wf-358-B2 closeout COMPLETE; no gap; verification only)
**Honest framing**: VERIFICATION only — 0 NEW tests; 0 NEW code; PROJECTED 5-10 min CPU wall

## Summary

Wf-358-B2 was previously reported as SHIPPED at
[`TODO/completed/358b2_equivariant_mp.md`](358b2_equivariant_mp.md). This verification
pass confirms the closeout is COMPLETE with no truncation / missing work:

- Closeout file exists at `TODO/completed/358b2_equivariant_mp.md` (163 lines, full
  details including summary, files shipped, architecture, CLI flag, tests, wall budget,
  honest framing, acceptance gate, follow-up, references).
- Model implementation `molmetal/models/equivariant_dmpnn.py` exists (347 LOC, per
  closeout claim of ~347 with ~250 code).
- Test file `molmetal/tests/test_wf358b2_equivariant_mp.py` exists (369 LOC, per
  closeout claim of 369 with 4 NEW + 1 bonus = 5 tests).
- CLI flag `--equivariant-mp` is wired into
  `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` at lines 2786-2798 with
  `dest="equivariant_mp"`, `action="store_true"`, default OFF (preserves
  legacy `DirectedMPNN` numerical signature).
- All 5 tests PASS: `5 passed, 1 warning in 1.33s` on CPU-only run.

## Files verified

| Path | LOC | Status |
|---|---:|---|
| `TODO/completed/358b2_equivariant_mp.md` | 163 | OK (closeout) |
| `molmetal/models/equivariant_dmpnn.py` | 347 | OK (model) |
| `molmetal/tests/test_wf358b2_equivariant_mp.py` | 369 | OK (tests) |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | +50 LOC (lines 2786-2798) | OK (CLI flag wired) |

## Test results (CPU-only)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_wf358b2_equivariant_mp.py -v
...
collected 5 items

molmetal/tests/test_wf358b2_equivariant_mp.py::test_rbf_expansion_invariant_under_so3 PASSED [ 20%]
molmetal/tests/test_wf358b2_equivariant_mp.py::test_forward_and_gradient_flow PASSED [ 40%]
molmetal/tests/test_wf358b2_equivariant_mp.py::test_so3_rotation_equivariance_probe PASSED [ 60%]
molmetal/tests/test_wf358b2_equivariant_mp.py::test_cli_flag_equivariant_mp_wired PASSED [ 80%]
molmetal/tests/test_wf358b2_equivariant_mp.py::test_backward_compat_with_dmpnn PASSED [100%]

========================= 5 passed, 1 warning in 1.33s =========================
```

## Architecture verification (from model source)

Per `molmetal/models/equivariant_dmpnn.py`:

1. **RBF expansion** (`_rbf_expansion` at line 78) — SchNet §3.1 form
   `e_k(d) = exp(−β · (d − μ_k)²)` with `n_rbf=32`, `cutoff=5.0 Å`,
   `β = 0.5 · (n_rbf − 1) / cutoff`. **SO(3)-invariant scalar** by construction.

2. **Equivariant edge MLP** (`_EquivariantEdgeMLP` at line 114) — per-edge MLP
   `Linear(D + n_rbf) → Linear(D) → Linear(1)` + `tanh`, emits scalar
   `scalar_ij ∈ [-1, 1]`. Used by EGNN-style coord update.

3. **EquivariantDMPNN** (line 160) — wraps legacy `DirectedMPNN` with
   state-dict compatibility:
   - `self.base = DirectedMPNN(MPNNConfig(...))` — legacy backbone
   - `self.coord_mlps = nn.ModuleList([_EquivariantEdgeMLP(...) for _ in n_layers])`
   - `self.edge_rbf_proj = nn.Linear(edge_feat_dim + n_rbf, edge_feat_dim)`

4. **Forward pass** (line 249) — D-MPNN message passing + EGNN coord update:
   ```
   h_v = self.base.atom_embed(h_atom)
   dist = self._pairwise_distances(x, edge_index)
   rbf = _rbf_expansion(dist, n_rbf=self.n_rbf, cutoff=self.rbf_cutoff)
   edge_attr_aug = self.edge_rbf_proj(torch.cat([edge_attr, rbf], dim=-1))
   h_e = self.base.edge_embed(edge_attr_aug)
   for layer in range(self.n_layers):
       h_src = torch.tanh(self.base.atom_to_edge[layer](h_v[src]))
       msg = self.base.edge_mlp[layer](torch.cat([h_e, h_src], dim=-1))
       edge_in = torch.cat([h_e, rbf], dim=-1)
       scalar_ij = self.coord_mlps[layer](edge_in)
       diff = x[dst] - x[src]
       coord_msg = -diff * scalar_ij.unsqueeze(-1)
       # ... GRU update + coord update
       x = x + coord_accum
   ```

5. **Env-var opt-in** (`EQUIVARIANT_MP_ENABLED` at line 337) — matches
   Wf-328-W9 / Wf-334-U1 style (`EQUIVARIANT_MP=1` env-var).

## CLI flag verification (from trainer source)

Per `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` lines 2786-2798:

```python
p.add_argument(
    "--equivariant-mp",
    dest="equivariant_mp",
    action="store_true",
    default=DEFAULT_EQUIVARIANT_MP,
    help=(
        "Wf-358-B2 NEW: enable SO(3)-equivariant D-MPNN message "
        "passing via molmetal.models.equivariant_dmpnn.EquivariantDMPNN "
        "(SchNet RBF + EGNN coord update). Default OFF to preserve "
        "the published DirectedMPNN numerical signature. PROJECTED "
        "+10-20% m3 Vina lift (not measured on Mol-Metal yet)."
    ),
)
```

`DEFAULT_EQUIVARIANT_MP = False` (default OFF preserves the published
baseline numerical signature for Wf-124 P3 / Wf-181 P1 / Wf-131 P1 reproducibility audits).

## Test-by-test verdict

| Test | Verdict | What it verifies |
|---|---|---|
| `test_rbf_expansion_invariant_under_so3` | PASS | `_rbf_expansion(dist)` is identical (max abs diff < 1e-5) for 4 random SO(3) rotations |
| `test_forward_and_gradient_flow` | PASS | Forward returns `(N, D)`; legacy D-MPNN params receive gradient through pooled-output loss; coord_mlps params receive gradient through coord-aware loss |
| `test_so3_rotation_equivariance_probe` | PASS | Per-atom / pooled outputs are SO(3)-invariant under 4 random SO(3) rotations (max abs diff < 1e-3) |
| `test_cli_flag_equivariant_mp_wired` | PASS | `parse_args` accepts `--equivariant-mp`; default is OFF |
| `test_backward_compat_with_dmpnn` | PASS | Legacy `DirectedMPNN` instantiates + forward + state-dict round-trip unchanged; `EquivariantDMPNN` wraps the legacy backbone with strict state-dict loadability |

## Acceptance gate (Wf-358-B2) — VERIFIED

- [x] `molmetal/models/equivariant_dmpnn.py` created (~347 LOC code).
- [x] SO(3) rotation-aware message passing implemented (RBF + EGNN coord update).
- [x] CLI flag `--equivariant-mp` (default OFF) wired into trainer.
- [x] 4 NEW tests + 1 bonus = 5 tests, all PASS (5/5 = 100%).
- [x] Backward compat verified (legacy `DirectedMPNN` path unchanged).
- [x] CPU-only correctness gate passed (1.33s wall).

## Wall budget — ACTUAL

* Verification: ~5 min CPU (Bash + Read + Edit + 1 pytest run)
* **Total**: ~5 min CPU (well under the PROJECTED 5-10 min CPU-only budget)

## Honest framing per `party-twenty-spirit` §3.5

### What IS verified

* Closeout document EXISTS at the canonical path with full content (163 lines).
* Model file EXISTS with full implementation (347 LOC; RBF + EGNN coord update).
* Test file EXISTS with 5 tests, all PASS (5/5 = 100%) in 1.33s.
* CLI flag `--equivariant-mp` IS wired into trainer with default OFF.
* `EQUIVARIANT_MP=1` env-var opt-in IS supported (Wf-328-W9 / Wf-334-U1 style).
* Architecture matches closeout description (SchNet RBF + EGNN coord update + D-MPNN wrapper).

### What is NOT verified (out of scope)

* **No GPU launch performed** (verification only; CPU-only per AGENTS.md §5.1).
* **No G5/G6 retrain** — PROJECTED +10-20% m3 Vina lift is *projected* not measured
  (consistent with Wf-358-B2 closeout honest framing).
* **No chemprop / BondHead / DMPNN-Cache integration** — Wf-358-B2 ships the
  scaffold only; production wiring is a separate Wf (per closeout follow-up).
* **0 NEW tests added by this verification** — re-ran existing 5 tests on closeout.

### Verification verdict

**Wf-358-B2 is SHIPPED + VERIFIED.** No truncation / missing work detected.
Verification CPU wall: ~5 min (under 5-10 min budget).

## Follow-up (NOT in scope of Wf-361-A1)

* G5 / G6 retrain with `--equivariant-mp` to MEASURE +10-20% m3 Vina lift.
  BLOCKED on user OK per AGENTS.md §5.1.
* Coord-regression loss head to train EGNN coord update end-to-end (current
  scaffold leaves `coord_mlps` params under-trained when loss is on `out`).
* chemprop v2 / v3 + BondHead integration into multi-task Wasserstein trainer.
* GPU autograd correctness (BF16 / AMP / CUDA Graphs) — deferred per §5.1.

## Reference

* Wf-358-B2 closeout: [`TODO/completed/358b2_equivariant_mp.md`](358b2_equivariant_mp.md)
* Wf-357-C2 architecture improvement #2 spec (Catherine).
* Satorras et al. 2021, "E(n) Equivariant Graph Neural Networks", ICML 2021.
* Schütt et al. 2017, "SchNet: A continuous-filter convolutional neural network
  for modeling quantum interactions".