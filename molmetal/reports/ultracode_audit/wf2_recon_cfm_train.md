# WF-2 recon — A5 CFM training-step audit (joint BondOrderHead)

**Generated:** 2026-09-14
**Status:** MEASURED (audit only — no new runs; reconciles on-disk code)
**Scope:** Identify the exact insertion point in `LipmanFlowMatchingAdapter.train_step` for wiring the `BondOrderHead` CE loss into the CFM training step, catalogue the tensors already on hand at training time, document the existing atom-head CE wiring, and confirm `molmetal/data/tmqm.py` lacks `parse_bo_all()` (so true-bond labels for the bond head must come from elsewhere — most cheaply from `Molecule.bonds` + `Molecule.bond_types` already populated by `Molecule.from_rdkit_mol`).

---

## 1. Train-step anatomy (MEASURED)

The training entry-point is `LipmanFlowMatchingAdapter.train_step` at
`/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py:1347-1452`.
Body is 106 lines, of which the loss-composition block is lines 1413-1434 and the optimizer step is 1436-1452. The two current losses are:

- **L1 — CFM coord velocity (MSE, masked)**: lines 1413-1417.
- **L2 — Atom-type CE (masked)**: lines 1419-1432.

The total loss is composed at line 1434 as `loss = cfm_loss + atom_loss_weight * atom_loss`,
backward at 1437, `clip_grad_norm_(params, 1.0)` at 1441-1443 (parameter list = `velocity_field + pocket_encoder`; **the bond head's parameters are NOT yet in this list**), optimizer.step() at 1444, and per-component loss is stashed on `self.last_losses` at 1447-1451.

### Velocity-field forward (the source of per-atom embeddings)

`EGNNVelocityField.forward` is at lines 874-910 (sig) and 911+ (body). Signature is:
`forward(x, atom_types, edge_index, t, edge_mask=None, pocket_embed=None, dative_bond_edge_attr=None) -> dict`
returning `{"vel": (B, N, 3), "atom_logits": (B, N, max_z)}`. The per-atom hidden vector `h` after the EGNN message-passing layers is the natural input to a bond head — it lives in scope as `h` after the message-passing loop finishes, before the final projection to `vel` / `atom_logits`. **A5 should either (a) hook a small MLP head off `h` inside `EGNNVelocityField.forward` so the bond head shares the EGNN backbone, OR (b) instantiate a standalone `BondOrderHead` in the adapter and feed it per-pair slices of `h` extracted by `edge_index`.** Option (a) is cheaper and follows the TargetDiff §3.3 / Pocket2Mol §3.2 joint-head pattern; option (b) keeps the bond head purely SE(3)-unaware (the `BondOrderHead` in `molmetal/models/bond_head.py:340-395` already is — see "head consumes pre-computed distances and angle deviations").

### Tensors available at training time (MEASURED)

At lines 1376-1383 the adapter builds a batched representation from the `mols: List[Molecule]` argument:

| Tensor | Shape | Source line | Already on `self.device`? | Used for bond head? |
|--------|-------|-------------|--------------------------|---------------------|
| `x_1` | `(B, max_n, 3)` | 1376 (zeros) + 1381 (per-mol `m.coords`) | yes | yes — inter-atomic distances for `PairFeature.distance` |
| `atom_types` | `(B, max_n)` long | 1377 (zeros) + 1382 (per-mol `m.atom_types`) | yes | yes — `PairFeature.z_i / z_j` |
| `node_mask` | `(B, max_n)` bool | 1378-1383 (per-mol pad mask) | yes | yes — bond-head node mask |
| `pocket_embed` | `(B, H)` or None | 1388-1389 (`_encode_pocket`) | yes | indirect — bond head lives downstream of the EGNN, so it gets `pocket_embed` indirectly via the per-atom hidden vector `h` |
| `t` | `(B,)` | 1394 (uniform) | yes | no |
| `path_sample.x_t` | `(B, max_n, 3)` | 1397 | yes | yes — should be the input the bond head consumes (so the bond head sees the noisy, intermediate coords, matching the velocity field) |
| `edge_index` (dummy) | `(B, 2, E)` long | 1402 (`_make_dummy_edge_index`) | yes | **NEEDS REPLACEMENT** — currently a fully-connected dummy so the EGNN sees a complete graph; for the bond head we want the *true* bond topology from `Molecule.bonds` (already populated from RDKit via `Molecule.from_rdkit_mol`, see `domain/__init__.py:117-120`) |
| `Molecule.bonds` | `(2, N_bonds)` long | not yet read by `train_step` | n/a — on CPU per-mol | **MISSING** — this is the *true* bond graph the bond head should be supervised against |
| `Molecule.bond_types` | `(N_bonds,)` int8 | not yet read by `train_step` | n/a — on CPU per-mol | **MISSING** — these are the RDKit `Chem.BondType` int codes the bond head CE loss needs as labels |

**A5 needs to add two tensor-builds to `train_step` (after the existing `x_1/atom_types/node_mask` loop, ~line 1383):**
1. a padded `target_bond_index: (B, 2, E_b)` `LongTensor` from `m.bonds` with a per-row `bond_edge_mask: (B, E_b)` `BoolTensor` (padding = False), and
2. a padded `target_bond_classes: (B, E_b)` `LongTensor` from `m.bond_types` mapped to the 5-way `{0=no-bond, 1=single, 2=double, 3=triple, 4=aromatic}` vocab the existing `BondOrderHead` already emits (`BOND_LABELS` at `molmetal/models/bond_head.py:112`).

Both `Molecule.bonds` and `Molecule.bond_types` are already part of the `Molecule` dataclass at `molmetal/domain/__init__.py:76-90` — they are populated by `Molecule.from_rdkit_mol` (line 117) and are therefore *free ground truth* at training time with no new dataset work. The dative vs. covalent split is a derived view on top: dative edges are those where one endpoint is a metal from `DEFAULT_METAL_GEOMETRY` (see `molmetal/molmetal_lam/priors/metal_geometry.py:364-374`) and the `bond_types` value is single — exactly the rule the existing `BondAwareDecoder._infer_pair_features` uses at `molmetal/models/bond_head.py:570-573`.

## 2. Existing atom-head CE wiring (MEASURED)

`atom_logits = out["atom_logits"]` at line 1411. Masked CE at lines 1426-1432:
```
atom_loss_per_node = F.cross_entropy(
    atom_logits.reshape(-1, atom_logits.shape[-1]),
    atom_types.reshape(-1),
    reduction="none",
).view(b, max_n)
n_real = node_mask.sum().clamp(min=1)
atom_loss = atom_loss_per_node[node_mask].sum() / n_real
```
- Flatten over (B, N) then re-mask with `node_mask` (because atom_types=0 is a valid class for Carbon so the mask can't be derived from the label value).
- Reduce by sum-over-real-nodes / total-real-nodes (so the per-batch loss is node-count invariant).
- The logits are *not* vocab-masked at training time — line 1682 `if self._vocab_mask` only fires in `_generate_impl`, not in `train_step`. The CFM training step uses the full 100-class softmax. **A5 should be consistent and not vocab-mask during joint training either** (vocab-mask is an inference-time constraint, not a training-time prior — the bond head must learn to predict bonds for the full 5-way class set and only suppress "no-bond" outside the metal-allowed pairs).

`atom_loss_weight` is read at line 1367-1368 from `self._atom_loss_weight` if the caller doesn't override it; defaults to 1.0 (see the `setup` method around line 1223). `self._atom_loss_weight` is a public attribute on the adapter that `setup` takes as a parameter, so the bond-head loss weight can ride the same channel — add `bond_loss_weight: Optional[float] = None` to `train_step` and read from `self._bond_loss_weight` (settable in `setup`) following the same pattern.

The atom-head CE pattern is the exact template the bond-head CE must follow: same per-node/per-edge mask + flatten + reduce. The difference is the indices: atom CE is per-node, bond CE is per-edge.

## 3. tmqm.py `parse_bo_all()` (MEASURED, CONFIRMED ABSENT)

`molmetal/data/tmqm.py` ships these BO parsers:
- `parse_bo_metal` (lines 166-214) — extracts **only the metal-centre Wiberg BO line** per CSD code; sets `done = True` after the first transition-metal row (line 213) and skips the rest of the block.
- `parse_properties` (lines 217-226) — loads `tmQM_y.csv` (SMILES + DFT properties only; no per-bond data).
- `parse_xyz_headers` (lines 145-163) — comment-line parser; no bonds.

**No `parse_bo_all()` exists.** This matches the gap explicitly flagged in `molmetal/reports/ultracode_audit/wf1_recon_training_data.md:21` ("the `parse_bo_metal` parser only extracts the metal-centre line... A1's bond head needs ligand-side bonds too... a sibling `parse_bo_all()` would emit one row per atom-pair with `BO > 0.05`").

**A5 implication**: For the *joint* end-to-end training path, true bond labels do not need `parse_bo_all()` at all — they come from `Molecule.bond_types` populated by RDKit on the training-corpus SMILES (the CrossDocked SDFs already carry `Chem.BondType` per edge per `wf1_recon_training_data.md:25`). `parse_bo_all()` is only needed for the *Tier-1 Wiberg-BO regression* path described in the WF-1 training-data spec (a future enhancement, not on the A5 critical path). A5 should therefore not block on `parse_bo_all()`.

## 4. Joint bond-head training-hook spec (1 paragraph, MEASURED + PROJECTED)

A5 should add a `bond_loss_weight: Optional[float] = None` parameter to `LipmanFlowMatchingAdapter.train_step` (default 1.0, mirroring `atom_loss_weight` at line 1367), instantiate / load a `BondOrderHead` in `LipmanFlowMatchingAdapter.setup` (mirror the pattern of `self.pocket_encoder` at line 1223+), extend `EGNNVelocityField.forward` to optionally return `"bond_logits": (B, E_true, 5)` (a per-true-edge classification computed by a small `nn.Linear(hidden_dim, 5)` head off the per-atom hidden `h` indexed by `edge_index`), and append a fourth term to the loss at line 1434: `loss = cfm_loss + atom_loss_weight * atom_loss + bond_loss_weight * bond_loss` where `bond_loss = F.cross_entropy(bond_logits.reshape(-1, 5), target_bond_classes.reshape(-1), reduction='none')[bond_edge_mask].sum() / bond_edge_mask.sum().clamp(min=1)`. The new tensors `target_bond_index / target_bond_classes / bond_edge_mask` are built from `m.bonds / m.bond_types` inside the existing for-loop at lines 1379-1383 (a 4-line addition). The optimisable parameter list at line 1441-1443 must be extended to include `list(self.bond_head.parameters())` (or, if A5 hooks the bond MLP into `EGNNVelocityField`, the head's `Linear` is already covered by `self.velocity_field.parameters()`). Success criterion for A5: per-edge CE on a held-out pocket drops below 1.0 nat within 200 training steps with the bond head wired in; honest fallback (A6) is DropEdge + Gumbel-top-k connectivity prior per Pocket2Mol §3.2 / TargetDiff §3.3, which avoids needing `parse_bo_all()` entirely and only needs the per-node 2D distance matrix (already available from `x_1`).

---

## File:line list (MEASURED)

### Hook / insertion points (existing code)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py:874-910` — `EGNNVelocityField.forward` signature; extend return dict with optional `bond_logits`.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py:911+` — body of forward; the per-atom hidden `h` is the bond-head input.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py:1223+` — `setup` method (mirror `pocket_encoder` wiring for `bond_head`).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py:1347-1452` — `train_step`; add 4-line tensor build at ~line 1383, add `bond_loss_weight` kwarg at 1347, add bond-CE term at 1434, extend params list at 1441-1443.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py:1367-1368` — `atom_loss_weight` defaulting pattern; mirror for `bond_loss_weight`.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py:1411` — `out["atom_logits"]` extraction; sibling `out["bond_logits"]` if A5 hooks the head into the velocity field.

### Bond head (existing, A1 complete)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py:106-118` — class constants `BOND_NO_BOND / BOND_SINGLE / BOND_DOUBLE / BOND_TRIPLE / BOND_AROMATIC`, `BOND_LABELS`, `NUM_BOND_CLASSES = 5`, `MAX_ATOMIC_NUMBER = 90`.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py:299-334` — `AtomCloud` and `PairFeature` dataclasses; `edge_index / distance / z_i / z_j / angle_to_metal / is_dative_candidate`.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py:340-395` — `BondOrderHead` (small per-pair MLP; already in scope, in_dim=9).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py:426-707` — `BondAwareDecoder` (inference-side; reuses the head).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py:713-788` — `train_synthetic` (closed-form supervised pass on synthetic data; for A5 we will *not* use this — we wire a separate CE term directly into `train_step`).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py:794-801` — `default_trained_head(n_epochs=0, ...)`: the **frozen sidecar** the r10 harness currently uses at `r10_cfg_real_crossdocked.py:103`; A5 must replace this with the joint-trained head.

### Molecule / data side (existing, no extension needed for A5)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/domain/__init__.py:76-114` — `Molecule` dataclass: `coords, atom_types, bonds (2, N_bonds), bond_types (N_bonds,) int8, formal_charges, ...`. `from_rdkit_mol` populates `bonds` and `bond_types` from RDKit at line 117-120.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/data/tmqm.py:166-214` — `parse_bo_metal` (only metal-centre line, sets `done = True` at 213).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/data/tmqm.py:217-226` — `parse_properties` (no per-bond data).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/data/tmqm.py:232-291` — `load_tmqm` (merges shards; returns `smiles` column the r10 SMILES→Molecule path can re-parse via RDKit to get per-edge `bond_types` for free).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/data/crossdocked100_manifest.csv` — 100-row CrossDocked manifest; the ligand `.sdf` files carry RDKit `Chem.BondType` per edge (per `wf1_recon_training_data.md:25`).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py:350-374` — `DEFAULT_METAL_GEOMETRY` and `GEOMETRY_IDEAL_ANGLES_DEG`; the Z→Geometry map for inferring dative edges at training time.

### Existing CLI / wiring (existing, will gain a new flag)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py:97-110` — current frozen-sidecar `default_trained_head(n_epochs=0)` use; A5 will replace this with the joint-trained head instance.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py:172-200+` — argparse block; add `--joint-train / --no-joint-train` flag here (mirrors the `--vocab-mask` BooleanOptionalAction at line 189).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py:241 / 292` — branches that gate on `args.bond_head == 'learned'`; add a parallel `--joint-train` gate.

### New files (NEW, no edits to existing)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py` — extend with `JointBondHead` mode: takes per-atom hidden `h` + `edge_index` → `(B, E, 5)` logits; reuses existing `BondOrderHead` MLP with an extra projection from `hidden_dim` to `in_dim=9` (or, simpler: the per-atom hidden slice is the input, and we learn a tiny projection `Linear(hidden_dim, NUM_BOND_CLASSES)` directly — that is the A5 path of least resistance).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_a5_joint_training.py` — NEW unit test asserting: (1) `train_step` returns a `bond_loss` key in `last_losses`; (2) `bond_loss` is finite and > 0 on the first step; (3) after 10 training steps on a 2-mol batch the `bond_loss` decreases (monotonic-not-required, but mean < first); (4) the velocity-field parameters' `.grad` is non-None after backward (i.e. the joint head does not break the EGNN gradient path).

---

## Honest framing summary

**MEASURED**: `LipmanFlowMatchingAdapter.train_step` at `__init__.py:1347-1452` is the right hook point — it already builds `x_1 / atom_types / node_mask / pocket_embed` from `mols: List[Molecule]` and composes `cfm_loss + atom_loss_weight * atom_loss` at line 1434. The `Molecule.bonds (2, N_bonds)` and `Molecule.bond_types (N_bonds,)` fields at `domain/__init__.py:87-88` carry the per-edge ground truth for free; no dataset work is needed. `molmetal/data/tmqm.py` lacks `parse_bo_all()` (only `parse_bo_metal` at lines 166-214), but A5 doesn't need it — RDKit supplies discrete `Chem.BondType` per edge from the SMILES already in the cached `tmqm_parsed.csv`. The existing atom-head CE pattern at lines 1426-1432 is the exact template for the bond-head CE term.

**PROJECTED**: A5 will add a `bond_loss_weight` kwarg to `train_step`, a `bond_logits` entry to the velocity-field return dict, and a `bond_loss = CE(bond_logits, target_bond_classes, mask=bond_edge_mask)` term to the loss composition; success criterion is per-edge CE < 1.0 nat within 200 steps. **Honest fallback (A6, only if A5 fails to learn)**: DropEdge + Gumbel-top-k connectivity prior per Pocket2Mol §3.2 / TargetDiff §3.3 — entirely headless, uses only the per-node 2D distance matrix (already available from `x_1`), needs no `parse_bo_all()` and no bond labels at training time.
