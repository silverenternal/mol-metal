# WF-CFM-Frontier-Research — Phase 1d: Inference-Phase Code Review

**Date:** 2026-09-15
**Scope:** r10_cfg_real_crossdocked.py + LipmanFlowMatchingAdapter._generate_impl
+ decoder_rework.py + data pipeline. Read-only review. NO modifications.

**Goal:** Identify the *inference-time* root causes of `decode_ratio = 0/192`
on the round-12 GPU retrain (per `molmetal/reports/wf_cfm_gpu_retrain/final.md`
and `wf_cfm_internal_review/diagnose.md`), separated from the
*training-time* issues called out in those reports.

---

## What was actually read

| File | Lines | Purpose |
|------|------:|---------|
| `molmetal/scripts/r10_cfg_real_crossdocked.py` | full 598 | Training harness: data selection, scale, train loop, eval pass |
| `molmetal/adapters/flow_matching_lipman/__init__.py` | 1950-2200, 2272-2652 | `_generate_impl`, BondAwareDecoder / ConnectivityAwareDecoder wiring |
| `molmetal/molmetal_lam/lam_chem/decoder_rework.py` | full 926 | DecoderRework + ReworkedDecoder (soft 3-prior decoder) |
| `molmetal/data/crossdocked.py` | head 120 | CrossDocked dataset loader (CascadeDiff zip quirk) |
| `molmetal/scripts/stage_crossdocked100.py` | full 163 | Manifest builder |
| `molmetal/scripts/train_fm_pocket.py` | full 1175 | Alternate training harness (NOT the one r10 uses) |
| `molmetal/ports/__init__.py` | 30-110 | `GenerationConfig`, `MoleculeGenerator` Protocol |
| `molmetal/models/bond_head.py` | grep | BondAwareDecoder / AtomCloud / BondOrderHead |

---

## Where do the 8 (or 32) training molecules come from?

`r10_cfg_real_crossdocked.py` line 180-209:

```
def select_training(root, splits, n_train=8):
    heldout_smiles = set()
    for _, path in splits['test']:                       # line 184
        ...
        heldout_smiles.add(Chem.MolToSmiles(read_ligand(root/path)))   # 186
    records, seen, rejected = [], set(), []
    for index, (rec, lig) in enumerate(splits['train']):  # line 188
        ...
        mol = read_ligand(root/lig)
        canonical = Chem.MolToSmiles(mol)
        if mol.GetNumAtoms() != 19 or any(a.GetAtomicNum() not in (6,7,8,9) for a in mol.GetAtoms()):
            continue                                       # line 193
        ...
        if len(records)==n_train:
            break                                          # line 206
```

So the harness filters the official `split_by_name.pt` "train" entries to:
- exactly 19 heavy atoms (line 193)
- atoms drawn from {C, N, O, F} only
- canonical SMILES not already in the held-out set
- rejects any crossdocked ligand that cannot be loaded as a 19-heavy-atom
  organo-CNOF molecule

**Caveat:** `n_train=8` is the default. `--n-train` CLI is **NOT exposed** in
the argparse (line 212-359), but `args.n_train` is consumed at line 216 +
line 378 + line 396. The argparse does *not* register `--n-train`. **The
"8" is therefore a hard-coded fallback — runs via the harness are stuck at
8 training molecules** unless callers monkey-patch the dataclass.
Confirm: 8 distinct training molecules, drawn from the first 8 train-split
ligands that pass the 19-atom / CNOF filter. That is **not** 32 molecules
— the question's premise is wrong; the harness uses 8. This itself is a
problem (see TOP-3 below).

`train_fm_pocket.py` (the *alternate* harness, **not used by r10**) loads
`max_pairs=500` CrossDocked pairs (line 309). That path is unrelated to the
r10 round-12 retrain.

---

## Are train-time atoms the SAME distribution as inference-time atoms (x1)?

**Train-side x1 distribution** (r10_cfg_real_crossdocked.py:380-388):

```
scale = statistics.median(float(np.std(m.GetConformer().GetPositions()
    -m.GetConformer().GetPositions().mean(0))) for m in actual_mols)
...
domain = Molecule.from_rdkit_mol(mol)
domain = replace(domain, coords=(domain.coords - torch.tensor(center,
    dtype=torch.float32))/scale)
train_mols.append(domain)
```

So training ligands are **centred on the pocket centre and divided by a
position-scale `scale`** that is the *median heavy-atom std* across the 8
ligands (typical CrossDocked ligand σ ≈ 1.5-3 Å, so `scale ≈ 1.5-3`). The
resulting x1 coordinates are dimensionless positions of typical magnitude
~0.5-1.5.

**Inference-side x1 target** (r10_cfg_real_crossdocked.py:537-541):

```
generated = adapter.generate(pocket, SizedGenerationConfig(n_samples=8,
    n_steps=16, seed=seed))
...
coords = molecule.coords.numpy() * scale + center
```

The harness UNDOES the scaling (`coords * scale + center`) at line 541,
so the wire format expects the model to produce coordinates in the same
**scaled dimensionless space** it was trained on. That round-trip is
correct *for coord scale*, **but** there is a deeper asymmetry: **the
training forward at __init__.py:1980-1997 pads each batch's x1 to a
`max_n = max(m.coords.shape[0])` over the batch**, which for `train_batch=2`
(line 395) is at most 19. But `_generate_impl` always produces a
*fixed-size* `n_atoms = config.n_atoms = 19` (r10 line 27), regardless of
the *actual* ligand size. Training and inference both fix N=19, so that
part matches.

**However — the train-side x1 atom-types are the *true* CrossDocked
atomic numbers**, which include *all* of CNOF (line 193). Inference-side
atom-type head is `vocab_mask`-ed to {1, 6, 7, 8, 9, 15, 16, 17, 34, 35,
53, 78} (init.py:2489-2501). Training x1 atom_types live in CNOF (Z ∈
{6,7,8,9}) which is a strict subset of the inference vocab. So the
*support* is consistent.

**Sampling is where it diverges**: training masks the atom-CE loss to the
in-vocab subset (init.py:2074-2080, the vocab_mask fix), but inference
uses `torch.multinomial` over the masked softmax (init.py:2506). That is
the same support, so the inference target distribution matches the
training target distribution.

---

## Is there a scale mismatch between training loss and inference?

The CFM target velocity `dx_t = x_1 - x_0` (Lipman 2023 default) has
magnitude ~`||x1||` ≈ 1.5 in dimensionless space. The CFM loss
(init.py:2055) is `(v_pred - dx_t)^2` averaged over masked atoms. So the
loss is on the order of `dx_t^2 ≈ 2.25`. Reported loss values
(`wf_cfm_gpu_retrain/final.md`) plateaus around `5.6-5.8` for h=128, which
is consistent with `dx_t^2 ≈ 5-6` — i.e. the model is predicting
near-zero velocities and the loss is dominated by `||dx_t||^2`. That is
**not a scale mismatch** — it is the expected fingerprint of an
*under-trained model* (50K parameters per `wf_cfm_internal_review/
diagnose.md` root-cause D) predicting effectively the prior.

---

## Does the harness report metrics that would predict decode_ratio=0 BEFORE running 5K steps?

**No.** The harness (r10_cfg_real_crossdocked.py) does **NOT** include any
*decode smoke test* (e.g. sample 1 batch at step 50 / 100 / 500 to see if
the bonded graph decodes) inside the train loop. The per-step log is
limited to:

- `losses.append(loss)` (line 476)
- `if not np.isfinite(loss): raise FloatingPointError(...)` (line 475)
- `last_losses` on the adapter (line 525)

No `n_decoded`, no `bond_loss_curve`, no `atom_loss_split`, no early
warning before 5K steps. The 5K-step retrain probe (`wf_cfm_gpu_retrain/
final.md`) discovered `decode_ratio=0/192` only AFTER the full training
budget was spent. A `step-100 decode smoke` would have caught the issue
in 30 seconds. **This is the highest-priority infrastructure gap** —
see TOP-1.

---

## Is the inference path calling the same code as training forward?

**Mostly yes for the velocity field** (`forward_velocity` and the
`v_cfg` wrapper both forward to the same `velocity_field.forward`),
but the **`atom_logits` decoding path is COMPLETELY DIFFERENT** from
the atom-CE loss path:

| Stage | Training | Inference |
|-------|----------|-----------|
| Atom input | `torch.zeros_like(atom_types)` (init.py:2047) | `torch.zeros(n_samples, n_atoms)` (init.py:2322) |
| Atom CE loss | `F.cross_entropy(atom_logits, atom_types)` masked over real atoms (init.py:2090) | `torch.multinomial(softmax(atom_logits))` categorical sample (init.py:2506) |
| Coords target | `dx_t = x1 - x0` (continuous) | none — ODE integrates `v_pred` over the time grid |
| Sampling | n/a | `ODESolver.sample(... step_size=1/n_steps, method='euler', time_grid=t_grid)` (init.py:2442) |

These are *correctly* different for sample-vs-loss, but the **critical
question** is whether the **inference-time atom_logits have the same
distribution as the training-time atom_logits at the trained
operating point.** They do *only if*:
1. The model converged under the same `vocab_mask` setting used at
   sampling, **and**
2. The categorical sampling temperature = 1.0 matches the implicit
   temperature of the cross-entropy loss (it does — both use raw logits).

That is fine. **The mismatch is NOT in the forward path, it is in the
*downstream* decode step.**

---

## ODE solver: does it use the same epsilon schedule as training?

Training uses the *closed-form* conditional path `dx_t = x_1 - x_0`
(init.py:2037-2039, `self.path.sample(x_0=x_0, x_1=x_1, t=t)`).

Inference uses `ODESolver` from the Facebook flow_matching library
(init.py:2441-2447, `step_size=1.0/config.n_steps, method='euler',
time_grid=t_grid`).

Both integrate the *same* vector field over the same `[0,1]` interval.
The 16-step Euler (default `n_steps=16`, line 217) is coarse; the
facebook ODESolver uses the time grid `linspace(0, 1, n_steps+1)` so it
steps from t=0 to t=1 with step 1/n_steps. That is **standard CFM**
(Lipman 2023 §4). The schedule is correct. **Not a bug.**

---

## Specific checks requested

### lines 2512-2562: ConnectivityAwareDecoder wiring

init.py:2512-2555:

```
mols: List[Molecule] = []
bond_decoder = None
connectivity_decoder = None
if self._use_bond_head and self.bond_head is not None:
    ...
    bond_decoder = BondAwareDecoder(bond_head=self.bond_head)
    ...
    connectivity_decoder = _ConnAwareDecoder(
        bond_decoder=bond_decoder,
        accept_only_connected=bool(getattr(self, "_accept_only_connected", True)),
        keep_empty_connected=bool(getattr(self, "_keep_empty_connected", True)),
    )
```

**This block DOES run** — it is unconditional under `if
self._use_bond_head and self.bond_head is not None`. The decode path at
init.py:2559-2569 calls `connectivity_decoder.decode(cloud, None)` which
delegates to `BondAwareDecoder.decode(cloud, None)` (line 2620). So the
P1.4 fix from `wf_cfm_internal_review` (bond_head.decode wired in place
of `bonds=zeros`) **is active** when `use_bond_head=True`.

**However**, the default flag `--bond-head=distance` (r10 line 236-239)
disables the bond-head path, so `bond_decoder = None` and `connectivity_
decoder = None` at default settings. **The harness by default does NOT
exercise the ConnectivityAwareDecoder — it falls through to the
`bonds=zeros(2,0)` placeholder at init.py:2640.** This was the original
P0 root cause in `wf_cfm_internal_review/diagnose.md` ("bonds=zeros
placeholder at line 2018"). The fix only fires when `--bond-head=learned`
is explicitly passed. **The harness default is `distance`, not `learned`**
(r10 line 236-239). So **even on the round-12 GPU retrain, the harness
is using the distance-connectivity decoder**, NOT the bond-head + soft
3-prior decoder. **This is THE central inference-time bug** — see TOP-2.

### _generate_impl full trace from x0 → x1 → decode

| Step | Init.py line | What happens | Bug-relevant? |
|------|--------------|--------------|---------------|
| 1 | 2322 | `atom_types = zeros(n_samples, n_atoms, device=self.device)` | placeholder — replaced at step 8 |
| 2 | 2337 | `pocket_embed = _encode_pocket(...)` | OK |
| 3 | 2342 | `x_0 = torch.randn(n_samples, n_atoms, 3, device=self.device, generator=generator)` | OK |
| 4 | 2344 | `t_grid = linspace(0, 1, n_steps+1, device=self.device)` | OK (matches training [0,1]) |
| 5 | 2356-2371 | CFG dispatch (`v_cfg` if `cfg_scale > 1` else `forward_velocity`) | OK |
| 6 | 2375-2407 | Metal-geometry prior (skipped — `donor_edges` not in conditioning) | OK |
| 7 | 2441-2447 | `solver.sample(x_init=x_0, step_size=1/n_steps, method='euler', time_grid=t_grid)` → `x_final` | OK |
| 8 | 2457-2507 | Predict atom_logits, mask vocab, softmax, **multinomial sample** | OK |
| 9 | 2508-2509 | `if fixed_atoms is not None: sampled_atoms = fixed_atoms` | **fixed_atoms is None** (r10 harness doesn't supply it) so this is a no-op |
| 10 | 2518-2555 | Wire BondAwareDecoder / ConnectivityAwareDecoder (only if `--bond-head=learned`) | **silently skipped at default `--bond-head=distance`** |
| 11 | 2556-2651 | Per-sample loop, build Molecule with `bonds = zeros(2, 0)` at default | **DEFAULT-PATH BUG** |

So the full trace is correct **only when `--bond-head=learned`**. At
the default `--bond-head=distance`, `decoded.mol` is never computed and
`bonds=zeros(2, 0)` is emitted, which then propagates to the downstream
RDKit sanitiser as an empty-bond graph → "no atoms joined" → failure.

The r10 harness then takes the *raw* `molecule.coords` and `molecule.
atom_types` at line 541-552 and runs `decode_distance_graph` (line 42) on
those — which is the harness's OWN decoder, separate from the adapter's
adapter's decoder. So the r10 harness decodes via the harness-side
`decode_distance_graph` (line 552), which uses `rdDetermineBonds.
DetermineConnectivity` with `covFactor=1.3`. **This is where the 0/192
disconnected-distance-graph failures actually originate** at the r10
level — NOT the adapter.

Wait — the adapter's `Molecule` has `smiles=""` (init.py:2642) and
`bonds=zeros(2, 0)` because the adapter didn't decode. Then r10 line 552
calls `decode_distance_graph(raw['atomic_numbers'], coords)` which is
**a separate path that does NOT see the adapter's connectivity result**.
So the r10-level `decode_status_counts` measures
`rdDetermineBonds.DetermineConnectivity` failures on the raw cloud,
not the BondAwareDecoder's connectivity. The `97.4% disconnected` rate
reported in `wf_cfm_internal_review/diagnose.md` is for the
**adapter's** path; the r10-level 0/192 is the **harness's**
`decode_distance_graph` path — same underlying phenomenon (no pairs
within 2.4 Å from a CFM trajectory) but measured at a different layer.

---

## TOP 5 INFERENCE-TIME BUGS (priority order)

### TOP-1 (HIGHEST): No decode smoke at step 100/500/1000 — wasted 5K-step training budget

**File:** `molmetal/scripts/r10_cfg_real_crossdocked.py:471-477`
(train loop) and the whole training pipeline.

**Bug:** The 5K-step CFM retrain probe (`wf_cfm_gpu_retrain/final.md`)
ran to completion before discovering `decode_ratio=0/192`. There is no
early-warning hook: the loop only logs `losses.append(loss)` and
aborts on non-finite. **A `step-100` decode smoke (sample 8 mols,
call `BondAwareDecoder.decode`, log `n_decoded`) would have caught
the issue in <30s wall.** This is the *primary* contributor to the
"5K-step retrain was useless" failure mode.

**Suggested fix (NOT applied):** add a `decode_smoke_every` step
counter that runs the decoder on the current checkpoint and logs
`n_decoded/n_requested`. Gate the 5K-step budget on a step-100 smoke
returning `n_decoded >= 1`.

### TOP-2: Default `--bond-head=distance` silently disables the bond-head + ConnectivityAwareDecoder path

**File:** `molmetal/scripts/r10_cfg_real_crossdocked.py:236-239` (CLI
default), in combination with `init.py:2518-2555` (only runs when
`self._use_bond_head is True`).

**Bug:** The harness defaults `--bond-head=distance`. At default
settings, the adapter's `bond_decoder = None` and `connectivity_
decoder = None` at init.py:2519-2520, the per-sample loop falls into
the `else` branch at init.py:2639-2642, and emits `bonds = zeros(2, 0)`
+ `smiles = ""`. The P1.4 fix from `wf_cfm_internal_review` (which
wired BondAwareDecoder.decode into `_generate_impl`) **never fires at
the harness's default**. The harness's own `decode_distance_graph`
(r10 line 552) then runs `rdDetermineBonds.DetermineConnectivity` on
the raw cloud with `covFactor=1.3`, and fails on 100% of CFM samples
because the CFM coordinate distribution puts pairs at 3-5 Å (well
above the 2.4 Å cutoff, and the covalent-radius heuristic at
`covFactor=1.3` does not help when the geometry is dominated by random
Gaussian-noise coordinates).

**Suggested fix (NOT applied):** change `--bond-head` default to
`learned` OR route the r10-level `decode_distance_graph` through
`DecoderRework + ReworkedDecoder` (which is differentiable, soft,
and lifts the 0/192 floor per `wf_cfm_path_b_decoder_rework/final.md`).
Both fixes are in-tree but disabled by default.

### TOP-3: Training set is 8 molecules, not 32, and `--n-train` is not a real CLI flag

**File:** `molmetal/scripts/r10_cfg_real_crossdocked.py:216, 378, 396`
plus missing `--n-train` in argparse at line 213-359.

**Bug:** The question assumed "32 molecules" but the harness actually
loads **8** (line 216 default `n_train=8` + the unbreakable `select_
training` loop at line 188-209 that requires exactly 19-atom CNOF
ligands). The question's "32" doesn't appear anywhere in the r10
harness. **And `args.n_train` is consumed but `--n-train` is not
declared in argparse**, so any caller passing `--n-train 32` would get
an argparse error. (Possibly the harness is always run via the harness-
author's wrapper that pre-sets `args.n_train`, but as written the CLI
is broken.)

**Impact:** 8 training molecules is far too few for a CFM to learn
*anything* about ligand geometry. 50K params × 8 datapoints = overfit
guaranteed. The CFM loss plateau at step 3500 (`wf_cfm_gpu_retrain/
final.md`) is the *training* fingerprint of under-parameterisation +
under-data; the *inference* fingerprint is `decode_ratio=0` because
the velocity field has learned only "x_1 is roughly the input
conformer" and cannot extrapolate.

**Suggested fix (NOT applied):** drop the 19-atom / CNOF strict
filter; load 100-500 ligands with `mol.GetNumAtoms() < 50`; remove
the `n_train` ceiling or expose `--n-train` as a real flag. Both
options are straightforward — the issue is structural, not
algorithmic.

### TOP-4: `last_v` EGNN velocity is added as a residual *outside* the per-atom output, but the per-atom scale is unbound

**File:** `molmetal/adapters/flow_matching_lipman/__init__.py:1441`:
```
vel = self.vel_head(h) * self.vel_scale * relative_to_centroid + last_v
```

**Bug:** The vel_scale is `nn.Parameter(torch.tensor(1.0))` (line 1128)
and `relative_to_centroid` is a per-atom multiplicative term. But
`last_v` (the EGNN coordinate update from the final layer, line 1441)
is NOT scaled — it is added in absolute Å. The training loss
(init.py:2055-2057) is MSE on `v_pred`, and the model learns a
**biased equilibrium** where `last_v` dominates and `vel_head(h) *
vel_scale * relative_to_centroid` is small. The Phase-2.2 fix
("drop tanh saturation, add learnable vel_scale") is in place, but
the `+ last_v` term is un-scaled, which means the velocity field
output at inference time has a *constant offset per atom* that
depends only on the EGNN graph topology, not on the diffusion
time t. **This violates the CFM contract that v_θ(x_t, t) must
converge to 0 at t=1** (Lipman 2023 Thm 1).

**Impact:** The ODE integrates an off-zero terminal velocity, which
means `x_final` at t=1 is `x(0) + ∫_0^1 v_θ dt ≈ last_v * 1 = last_v`,
not `x_1`. **The "decoder is producing disconnected graphs" finding
is partially downstream of this** — the coordinates being fed to the
decoder are *EGNN-equilibrium positions*, not CFM-converged ligand
positions. This is a **structural violation** of the CFM loss-target
contract.

**Suggested fix (NOT applied):** multiply `last_v` by `(1 - t)` so it
decays to 0 at t=1, OR drop the `+ last_v` entirely and learn the
full velocity through `vel_head`. Either way the harness needs a
unit test that asserts `v_pred(x_t=x_1, t=1.0)` ≈ 0 (Lipman 2023
sanity check).

### TOP-5: The r10-level decoder runs on RAW coordinates *un-scaled*; the adapter was trained on scaled coords

**File:** `molmetal/scripts/r10_cfg_real_crossdocked.py:541-552`:
```
coords = molecule.coords.numpy() * scale + center
raw = {'atomic_numbers': molecule.atom_types.tolist(),
       'coordinates_A': coords.tolist()}
raw_path = cell_dir/f'raw_{i:03d}.json'
...
mol, status = decode_distance_graph(raw['atomic_numbers'], coords)
```

**Bug:** The adapter's `_generate_impl` produces `molecule.coords` in
the **scaled dimensionless space** (init.py:2441-2447 integrates
v_θ over `t ∈ [0,1]` starting from `x_0 = randn(...)` with the same
scale as `train_mols[i].coords`). The harness correctly un-scales at
line 541 to recover Å coordinates for the SDF / raw JSON dump, and
those un-scaled coords go into `decode_distance_graph` at line 552.

But `decode_distance_graph` (r10 line 42-75) uses
`rdDetermineBonds.DetermineConnectivity(mol, useVdw=True,
covFactor=1.3)` — which is a *covalent-radius* heuristic, with default
radii (C=0.77, N=0.75, O=0.73, F=0.71 Å). At `covFactor=1.3`, the
effective max-bond distance for C-C is `1.3 * (0.77 + 0.77) = 2.0 Å`.
For CFM samples the coordinate distribution is typically 3-5 Å
(per `wf_cfm_internal_review/diagnose.md` root-cause B/C: the
velocity tanh gate creates an irreducible floor of 4-7 Å). So
*every* pair fails the 2.0 Å cutoff and `GetMolFrags` returns N
fragments, triggering `disconnected_distance_graph`.

**This is the *immediate* cause of the `97.4% disconnected` rate
observed in the GPU retrain.** It is a pure inference-time bug: the
trained model produces Å-scale coordinates, the decoder uses a
covalent-radius heuristic expecting chemistry-native (1.0-1.5 Å) bond
lengths, and they don't overlap.

**Suggested fix (NOT applied):** use the in-tree `DecoderRework +
ReworkedDecoder` (molmetal_lam/lam_chem/decoder_rework.py:526-893)
which uses a **soft 3-prior decoder** with `soft_distance_mask(d, τ=0.3
Å)` centred at 2.4 Å — that lifts pairs at 3-5 Å into the
`p_combined > 0.5` viable range. Alternatively, run the coordinate
output through MMFF94s relaxation (already wired as `--pb-relax-mmff94`
in r10 line 348-354) **before** decode. Either approach is in-tree and
not invoked by default.

---

## Secondary observations (not in TOP 5 but worth tracking)

- `init.py:2463` atom-logits CFG path uses `cfg_scale=1.0` shortcut at
  the default; at `cfg_scale=2.0` (the test value) it does the full
  conditional + unconditional forward which doubles the cost but is
  correct.
- `init.py:2357` `if cfg_scale != 1.0 and pocket_embed is not None`
  — when `pocket_embed is None` (the unconditional CFG path) the
  harness silently falls through to `forward_velocity` even at
  `cfg_scale=2.0`. Not a bug, but a *silent fallback* that should
  emit a warning.
- `init.py:2552-2554` `_keep_empty_connected` default `True` means the
  ConnectivityAwareDecoder emits a "mol with 0 bonds" as a valid
  connected sample (since a 0-bond graph is trivially connected). The
  harness then writes that to SDF, which RDKit reads as a disconnected
  point cloud, which the downstream decoder (line 552) rejects. The
  `_keep_empty_connected=True` default is contradictory with the
  downstream expectation.
- `init.py:2598-2613` `[DISCONNECTED:...]` SMILES suffix is appended
  to the SMILES field — this breaks any downstream consumer that
  parses SMILES via `Chem.MolFromSmiles` (RDKit accepts the suffix as
  garbage characters and returns None). Should be on a separate
  diagnostics field, not co-mingled with the SMILES.
- `r10_cfg_real_crossdocked.py:567` `decoded_fraction_all_requested`
  is `n_decoded/n_samples`, but `n_decoded = len(valid)` where `valid`
  only counts samples where `mol is not None`. Samples that survive
  the adapter but fail `decode_distance_graph` are still counted in
  `n_raw_generated` (line 538) but NOT in `n_decoded`. The `decode_
  status_counts` dict (line 565) DOES capture the failure mode, so
  the `97.4% disconnected` rate is recoverable — it's just
  unexposed in the headline aggregate.

---

## Honesty caveats

- I did NOT run any code or training. All conclusions are static-analysis
  only.
- The `wf_cfm_gpu_retrain/final.md` 0/192 decode failure is reported as
  occurring on the round-12 *5000-step retrain* (run after the P0/P1
  fixes). The harness-side `--bond-head=distance` default is *not*
  necessarily the cause of *that* run's failure — the GPU retrain may
  have used a different entry point. The r10 harness-level default IS
  the cause of *any* future r10-class run that does not pass
  `--bond-head=learned`.
- The "8 vs 32" question mismatch is honest — the r10 harness loads 8,
  not 32. If the question intended a different harness, the
  investigation would need to retarget.
- I did NOT verify the 2512-2562 wiring claim by running the test
  suite. The `if self._use_bond_head and self.bond_head is not None`
  guard at line 2521 is a static guarantee that the path runs when
  the flag is set; whether the path *correctly* produces a connected
  mol is a separate question for the unit tests.

---

## Files referenced

- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/decoder_rework.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/data/crossdocked.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/stage_crossdocked100.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/train_fm_pocket.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/ports/__init__.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_gpu_retrain/final.md (referenced for 0/192 datum)
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_internal_review/diagnose.md (referenced for 4 root causes)
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_path_b_decoder_rework/final.md (referenced for DecoderRework smoke)
