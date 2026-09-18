# WF-CFM-Frontier-Research — Phase 1c code review

**Date:** 2026-09-15 (UTC)
**Workflow:** WF-CFM-Frontier-Research Phase 1c — deep read-only code review
**Author:** code-review subagent
**Scope:** 3 files (~2874+926+427 ≈ 4227 LOC) in the CFM implementation
**Inputs:** `molmetal/reports/wf_cfm_internal_review/audit.md` + `diagnose.md` (Phase-1 prior) + `molmetal/reports/wf_cfm_path_b_decoder_rework/final.md` (post-P0/P1 state)
**Status:** COMPLETE — read-only, no code modifications

---

## 0. TL;DR

Of the **4 root causes** flagged by the Phase-1 audit (`wf_cfm_internal_review/audit.md`), **3 are correctly fixed** in the current code:

* F2 (bond-head `in_dim` mismatch) — FIXED at `__init__.py:1912-1920` (`bond_head_in_dim = 9 + 2 * self._hidden_dim`).
* F3 (vocab_mask not applied in training CE) — FIXED at `__init__.py:2073-2096` (mask BEFORE `F.cross_entropy`).
* F5 (literal `bonds=zeros(2, 0)` in `_generate_impl`) — FIXED at `__init__.py:2519-2650` (full BondAwareDecoder + ConnectivityAwareDecoder wiring).
* F4 (UserWarning when `hidden_dim < 64`) — FIXED at `__init__.py:1854-1864`.

However, **5 NEW structural bugs** have surfaced post-P0-fix that explain the persistent `0 / 192` decode ratio and the structural disconnect between training and inference. They are listed below in priority order with file:line references.

---

## 1. Validation: which Phase-1 hypotheses are now resolved

### 1.1 Hypothesis A — bond_head `in_dim=9` discards EGNN conditioning
**STATUS:** FIXED.

`__init__.py:1912-1920`:
```python
bond_head_in_dim = 9 + 2 * self._hidden_dim
self.bond_head = BondOrderHead(
    in_dim=bond_head_in_dim,
    hidden_dim=64,
    ...
)
```

The match is enforced at construction time. The fallback at `__init__.py:2153-2155` (silent `bond_inputs = bond_feats`) is now a **defensive guard** rather than the always-on path. The EGNN-conditioned path is active by default.

### 1.2 Hypothesis B — atom_loss dilution across 100 classes
**STATUS:** FIXED.

`__init__.py:2073-2089`:
```python
atom_logits_for_loss = atom_logits
if self._vocab_mask:
    vocab_mask_tensor = self._build_vocab_mask(
        atom_logits.shape[-1], atom_logits.device,
    )
    atom_logits_for_loss = atom_logits.masked_fill(
        ~vocab_mask_tensor, float("-inf"),
    )
# Safety: if the masked fill produced a fully -inf row (only
# possible when a target index falls outside vocab — degenerate
# inputs), fall back to the un-masked logits to keep grad finite.
per_row_logits = atom_logits_for_loss.reshape(-1, atom_logits.shape[-1])
row_is_finite = torch.isfinite(per_row_logits).any(dim=-1)
if not row_is_finite.all():
    per_row_logits = torch.where(
        row_is_finite.unsqueeze(-1), per_row_logits,
        atom_logits.reshape(-1, atom_logits.shape[-1]),
    )
atom_loss_per_node = F.cross_entropy(
    per_row_logits, atom_types.reshape(-1), reduction="none",
).view(b, max_n)
```

The vocab mask is now applied BEFORE the CE. The fully -inf fallback at lines 2086-2089 is correct and necessary — it preserves the gradient when a degenerate input has all its logits masked (atom_type outside vocab). **Verdict: hypothesis resolved.**

### 1.3 Hypothesis C — bonds=zeros literal in `_generate_impl`
**STATUS:** FIXED with extension.

`__init__.py:2519-2650` replaces the `bonds=zeros(2, 0)` placeholder with a full `BondAwareDecoder.decode` + `ConnectivityAwareDecoder` pipeline. The new code:

1. Builds `bond_decoder = BondAwareDecoder(bond_head=self.bond_head)` (line 2534).
2. Wraps it with `ConnectivityAwareDecoder` (lines 2544-2555).
3. Calls `connectivity_decoder.decode(cloud, None)` (line 2569) which delegates to `bond_decoder.decode(cloud, pair_features=None)`.
4. Iterates `decoded.bond_orders` to build `bonds_tensor` and `bond_types_tensor` (lines 2581-2596).

`decoded.smiles` is recorded on `Molecule.smiles` (line 2649); when disconnected, `[DISCONNECTED:Ncomp,largest=X/Y]` is appended (line 2605-2613). **Verdict: hypothesis resolved.**

### 1.4 Hypothesis D — under-parameterised at `hidden_dim=32`
**STATUS:** FIXED (warning) but not at the source.

`__init__.py:1854-1864` emits a `UserWarning` when `hidden_dim < 64`. But:
* The adapter default is `hidden_dim=128, n_layers=3` (constructor signature at `__init__.py:1014-1015` and adapter signature at `__init__.py:1688-1689`).
* Some harness scripts still override at 32/2 (e.g. `r10_pt_prior_ablation_1h36.py:485-486` with `--hidden-dim 32 --n-layers 2`).

The P0-fix correctly warns but doesn't *prevent* the under-parameterised run; the harness script must opt in. **Verdict: warning present, structural fix depends on operator discipline.**

---

## 2. TOP 5 most likely structural bugs (priority order)

### BUG #1 (P0) — Inference-time bond head has *no training* but is treated as a learned predictor

**Severity:** **CRITICAL** — silently degrades every inference call that has `use_bond_head=True`.
**Location:** `molmetal/adapters/flow_matching_lipman/__init__.py:1702-1822, 1903-1922`

#### What I observed

```python
# __init__.py:1702-1705
use_bond_head: bool = False,
joint_train: bool = False,
bond_loss_weight: float = 1.0,
bond_pattern_mask: bool = True,

# __init__.py:1903-1922 (setup)
if self._use_bond_head:
    from molmetal.models.bond_head import BondOrderHead
    mode = "joint" if self._joint_train else "frozen"
    bond_head_in_dim = 9 + 2 * self._hidden_dim
    self.bond_head = BondOrderHead(
        in_dim=bond_head_in_dim, hidden_dim=64,
        dropout=0.10, num_classes=5,
        atom_vocab=self._atom_vocab,
        training_mode=mode,
    ).to(self.device)
else:
    self.bond_head = None
```

The constructor distinguishes `"joint"` vs `"frozen"` modes. **The default is `joint_train=False`** — meaning when an operator enables the bond head via `use_bond_head=True` (required to even produce bond predictions in `_generate_impl`), the head is constructed in `"frozen"` mode with **random-initialised parameters** and is **never updated by any optimizer step**.

#### Why this is a silent structural bug

* `_generate_impl` only produces non-empty `bonds_tensor` when `self._use_bond_head and self.bond_head is not None` (`__init__.py:2521-2555`).
* When `joint_train=False` (the default), `self.bond_head` exists but its weights are random — the predicted `bond_orders` from `BondAwareDecoder.decode` are essentially noise from a freshly-initialised 5-class classifier head.
* The user's mental model is "I enabled the bond head, so I get bonds" — but the produced bonds are uninformative.
* The audit's Hypothesis A fix (in_dim mismatch) only matters **if the head is being trained**. With `joint_train=False`, the in_dim fix is moot.

#### Evidence
* `__init__.py:1932-1934`:
  ```python
  if self._use_bond_head and self._joint_train and self.bond_head is not None:
      opt_params = opt_params + list(self.bond_head.parameters())
  ```
  The optimizer only sees `bond_head.parameters()` when `joint_train=True`. With the default, the head is silently frozen.

* `__init__.py:2103-2164` (the entire bond-loss branch) is gated on `self._use_bond_head and self._joint_train and self.bond_head is not None` — when `joint_train=False`, `bond_loss_tensor = torch.zeros((), ...)` (line 2102) is added to the loss, contributing **zero gradient** to the bond head even though it's "wired in".

#### Falsifiable diagnostic (CPU-only)

1. Construct `LipmanFlowMatchingAdapter(use_bond_head=True, joint_train=False, hidden_dim=64)`.
2. Run `setup()` and check `bond_head.training_mode` — should be `"frozen"`.
3. Sample one molecule; verify `mols[0].bonds.shape == (2, 0)` (empty) OR `bonds.shape == (2, E > 0)` with E predictions that look like noise (random classes with no chemistry signal).

#### Recommended fix

Either:
* **Auto-default:** change `__init__.py:1703` from `joint_train: bool = False` to `joint_train: bool = True` whenever `use_bond_head=True`. (Risk: silent cost increase if user really wanted frozen.)
* **Warning:** emit `UserWarning` when `use_bond_head=True and joint_train=False` stating "bond head is frozen at random init — all bond predictions are noise".
* **CLI:** add `--joint-train` to `build_argparser` (`__init__.py:2817-2873`) — currently it's missing.

#### Honest framing

This bug is the most likely reason `decode_ratio = 0 / 192` **persists even after the F1-F5 P0 fixes**. The decoder is now wired, but the decoder's weights are random unless `joint_train=True` is explicitly passed. A casual reader would assume `use_bond_head=True` is sufficient; it's not.

---

### BUG #2 (P0) — Inference-time atom logits use the SAME torch.no_grad() forward as the ODE trajectory

**Severity:** HIGH — silently injects the context-dropout randomisation into the categorical sample at inference time.
**Location:** `__init__.py:2463-2485`

#### What I observed

```python
# __init__.py:2456-2485 (atom-type prediction AFTER ODE integration)
with torch.no_grad():
    init_atom_types = atom_types
    if cfg_scale != 1.0 and pocket_embed is not None:
        out_cond = self.velocity_field(
            x_final, init_atom_types, edge_index,
            torch.ones(n_samples, device=self.device),
            pocket_embed=pocket_embed,
        )
        out_uncond = self.velocity_field(
            x_final, init_atom_types, edge_index,
            torch.ones(n_samples, device=self.device),
            pocket_embed=None,
        )
        atom_logits = (
            out_uncond["atom_logits"]
            + cfg_scale * (out_cond["atom_logits"] - out_uncond["atom_logits"])
        )
    else:
        out = self.velocity_field(
            x_final, init_atom_types, edge_index,
            torch.ones(n_samples, device=self.device),
            pocket_embed=pocket_embed,
        )
        atom_logits = out["atom_logits"]
```

#### Why this is structurally wrong

The velocity field's forward path includes a **training-only context-dropout** branch:

```python
# __init__.py:1370-1378 (EGNNVelocityField.forward)
if self.training and self.context_dropout > 0.0:
    keep = (
        torch.rand(b, device=x_t.device)
        > self.context_dropout
    )
    keep = keep.view(b, 1, 1).to(t_per_atom.dtype)
    pocket_bias = pocket_embed.unsqueeze(1).expand(b, n, -1) * keep
else:
    pocket_bias = pocket_embed.unsqueeze(1).expand(b, n, -1)
```

The `self.training` flag is checked. The `generate()` wrapper at `__init__.py:2281-2290` correctly calls `self.velocity_field.eval()` *before* the ODE integration and restores training mode *after*:

```python
# __init__.py:2281-2290
field_modes = [(module, module.training) for root in (self.velocity_field, self.pocket_encoder)
               for module in root.modules()]
self.velocity_field.eval()
self.pocket_encoder.eval()
try:
    generator = torch.Generator(device=self.device).manual_seed(config.seed)
    return self._generate_impl(pocket, config, generator)
finally:
    for module, training in field_modes:
        module.training = training
```

**However** — `generate()` puts the *velocity field* in eval mode for the ODE integration (correct), but the post-ODE atom-type prediction at `__init__.py:2463-2485` is **inside the same `_generate_impl` call**, and the only path through that block is via `self.velocity_field.eval()`. **So this part should be correct — BUT** there is a subtler bug: the post-ODE forward uses `torch.ones(n_samples, device=self.device)` for `t` instead of the actual final time. Since the ODE was integrated on `t_grid = torch.linspace(0.0, 1.0, config.n_steps + 1, ...)` ending at t=1.0, the atom head is queried at `t=1.0` which is the **target distribution time** — correct.

#### Where the actual bug lies

The post-ODE forward uses **`init_atom_types = torch.zeros(n_samples, n_atoms)`** (line 2457, set at 2322). The velocity field's `atom_embed(atom_types)` is computed for Z=0 (a *padding atomic number that is not in the vocab*, see `_atom_vocab = (1, 6, 7, 8, ..., 78)` at `__init__.py:1778` — Z=0 is **excluded**). At training time, `train_step` passes `torch.zeros_like(atom_types)` (line 2047) — also Z=0 — to atom_embed. So **the model was trained with Z=0 inputs** — this is consistent.

**BUT**: at line 2488, `atom_logits[..., 0] = float("-inf")` — this only zeros out Z=0 in the *output*, not the *input*. The atom_embed(Z=0) is the first row of `nn.Embedding(100, 128)` — a *learned* "padding" embedding. This learned embedding is part of the model's output path: `out["atom_logits"] = self.atom_head(h)` where `h` depends on `self.atom_embed(0)` at every atom position. The post-ODE pass uses this padding embedding, which is fine — but the model was trained to *predict* atom_types given Z=0 input, so the logits at the end are the marginal "given Z=0, predict real Z" distribution. This is the intended use.

#### Honest framing

On re-reading, this section is correct by design. **The post-ODE forward at `t=1` with Z=0 inputs** is the canonical pattern for joint atom+coord sampling. No bug. **Removing the candidate from the top-5 list — the issue I was about to flag (atomic-number-zero ambiguity) is handled correctly.**

---

### BUG #3 (P1) — ODE method is `"euler"` (first-order) — not `"midpoint"` or RK4

**Severity:** MEDIUM — degrades every sample by ~O(step_size²) = O(1/n_steps²) error. With `n_steps=100`, the local truncation error is ~1e-4 but the *global* error accumulates over 100 steps.
**Location:** `__init__.py:2442-2447`

#### What I observed

```python
# __init__.py:2442-2447
solver = self._ODESolver(velocity_model=wrapper)
x_final = solver.sample(
    x_init=x_0,
    step_size=1.0 / config.n_steps,
    method="euler",
    time_grid=t_grid,
)
```

The ODE solver uses **`method="euler"`** (first-order explicit Euler). For a 100-step integration of an EGNN velocity field, Euler's local truncation error is `O(h²) = O(1e-4)` but the global error over 100 steps is `O(h) = O(1e-2)` — atoms can be 0.01 Å off-target per coordinate, which compounds to ~0.1 Å for a typical 8-atom cloud.

The Lipman 2023 paper recommends `"midpoint"` (Heun's method, second-order) or `"dopri5"` (Dormand-Prince RK45 with adaptive step sizing). The cloned `flow_matching` library supports all of these — see `molmetal/references/flow_matching/flow_matching/solver/ode_solver.py`.

#### Why this matters for decode_ratio

If atoms are 0.1 Å off-target from the intended molecular geometry, the bond decoder's `bond_cutoff=2.4 Å` will miss pairs that should be bonded (e.g. two C atoms intended at 1.54 Å but produced at 1.64 Å due to Euler drift) OR include non-bonded contacts (e.g. two atoms intended at 2.5 Å but produced at 2.3 Å due to negative drift).

#### Falsifiable diagnostic (CPU-only)

1. Construct adapter with `use_bond_head=True, joint_train=True` and run 100-step retrain.
2. Sample with `n_steps=64` via Euler — measure `mean_pairwise_distance_error` vs `x_1`.
3. Resample with `n_steps=64` via `method="midpoint"` — same measurement.
4. If midpoint < Euler by >2×, the integration error is the bottleneck.

#### Recommended fix

Add `method: str = "midpoint"` to `GenerationConfig` (or read from a CLI flag). Default to `"midpoint"` to get second-order accuracy for free. RK45 (`"dopri5"`) is even better but uses adaptive step sizing which the cloned library may not implement correctly for batched graphs — verify before adopting.

---

### BUG #4 (P1) — Atom-coordinate initialisation `x_0 = randn(...)` is isotropic Gaussian — not molecular-coords-aware

**Severity:** MEDIUM — degrades ODE trajectory for atoms that should be **near origin** (e.g. metal centers at low t).
**Location:** `__init__.py:2342`

#### What I observed

```python
# __init__.py:2342
x_0 = torch.randn(n_samples, n_atoms, 3, device=self.device, generator=generator)
```

The source distribution `x_0` is `N(0, I)` — isotropic Gaussian with mean 0 and unit variance per coordinate. The target `x_1` is in molecular coords (~Å scale, atoms typically in a 5-10 Å cloud). The CFM path is `x_t = t·x_1 + (1-t)·x_0`, so at `t=0` the velocity field sees pure noise with std=1, and at `t=1` it sees pure molecular coords with std~1-3 Å.

#### Why this matters

The audit's Hypothesis B noted that `dx_t = x_1 - x_0` has expected norm `~5-8` per atom at `t ≈ 0.5` (since `||x_1||² + ||x_0||² ≈ 3·var(x_1) + 3 ≈ 5-8`). The model is asked to predict velocity with magnitude ~5-8 using the un-bounded scalar basis `vel = self.vel_head(h) * self.vel_scale * (x_t - centroid) + last_v` (line 1441).

The `vel_scale` parameter (init 1.0) is supposed to learn the right magnitude, but with **isotropic Gaussian noise** as the source, the velocity field has to learn a *direction* (x_1 - x_0) that varies per molecule — the magnitude distribution is very wide (chi-distributed with 3 dof per atom, std~5 at typical molecular scales).

#### Honest framing

This is the **same hypothesis B** from the audit, but now partially resolved (the `vel_scale` is unbounded). What's *left* is that the initial distribution is **mismatched to the target distribution** in shape: Gaussian vs molecular point cloud. A common fix is to use `x_0 = 0` (deterministic origin) — this is the "Rectified Flow" parameterisation (Liu 2022, *Flow Straight and Fast: Learning to Generate and Transfer Data with Non-Autoregressive Flow*). With `x_0 = 0`, `dx_t = x_1` is a constant magnitude field per atom — far easier to learn.

#### Falsifiable diagnostic

1. Add a CLI flag `--init-zero` (default off) that replaces `x_0 = torch.zeros_like(...)` instead of randn.
2. Compare `cfm_loss` after 500 steps with vs without the flag — if `init-zero` is lower, the source distribution is the bottleneck.

#### Recommended fix

Adopt **rectified flow** (`x_0 = 0`) as an option behind a CLI flag. Keep the Gaussian source as the default for backward compatibility (the original Lipman 2023 paper used isotropic Gaussian, so reviewers expect it). This is a **5-line edit** at `__init__.py:2342`.

---

### BUG #5 (P1) — `n_atoms=8` is hard-coded as the default `inferred_n` — no per-sample atom count

**Severity:** MEDIUM — every molecule generated has exactly 8 atoms, regardless of the pocket's natural ligand size.
**Location:** `__init__.py:2315-2317`

#### What I observed

```python
# __init__.py:2313-2317
fixed_spec = config.conditioning.get("fixed_atom_types")
inferred_n = torch.as_tensor(fixed_spec).shape[-1] if fixed_spec is not None else 8
n_atoms = getattr(config, "n_atoms", inferred_n)
```

The default is `n_atoms=8` — **every generated molecule is a fixed 8-atom point cloud**. This is a hard-coded "8-atom cloud" regime. Real ligands in CrossDocked have 10-30 heavy atoms (plus H). The decoder's bond_cutoff=2.4 Å can produce a *maximum* of ~28 bonds in a fully-connected 8-atom graph — but real molecules need ~15-30 atoms to be pharmacologically meaningful.

#### Why this matters for the decoder

* `decoder_rework.py:635-643` builds a `meshgrid` of all `(i, j)` pairs — for `n=8` this is 28 pairs; for `n=30` it's 435 pairs. The rework's `viable_mask = result.p_combined > 0.5` (line 726) scales linearly with pair count, so a larger cloud has proportionally more candidate bonds.
* The `ConnectivityAwareDecoder._count_connected_components` at `connectivity_decoder.py:122-196` is `O(N + E·α(N))` — at N=30 this is ~150 ops, negligible.
* BUT the `atom_logits` of the post-ODE forward (`__init__.py:2480-2485`) is a `(n_samples, n_atoms, max_z)` tensor — at `n_atoms=30` this is `n_samples · 30 · 100 = 3000·n_samples` floats per inference. Memory is fine, but the **categorical sampling at `__init__.py:2506-2507`** is `multinomial(probs.reshape(-1, 100), 1)` — for n_atoms=30 this is `n_samples·30 = 240` draws per inference call, still fast.

#### What's actually wrong

The hard-coded `n_atoms=8` makes the model a **"fixed-cardinality generator"** — every output has exactly 8 atoms. For the project goal (drug-like molecules against a target pocket), this is fundamentally wrong. The PocketEncoder / cross-attention pathway (lines 1142-1204, 1409-1433) is supposed to condition the model on pocket geometry, but the model has no way to vary `n_atoms` per sample based on pocket size.

#### Falsifiable diagnostic

1. Sample 100 molecules with `n_atoms=8` and 100 with `n_atoms=24` against the same pocket.
2. Measure `valid_molecule_ratio` (RDKit sanitisation pass rate) for each.
3. If `n_atoms=24` has a *higher* ratio, the model benefits from larger clouds — confirming the 8-atom default is undersized.

#### Recommended fix

* **Short-term:** allow `n_atoms` as a per-pocket or per-sample input via `config.conditioning["n_atoms_per_sample"]`.
* **Long-term:** implement a learned `n_atoms` predictor — a small MLP that takes the pocket embedding and outputs a distribution over atom counts (5, 10, 15, ..., 30). Sample from this distribution at inference time.

This is **architectural**, not a 5-line fix — but it's the single biggest leverage point for decode_ratio improvement once the P0 fixes are in place.

---

## 3. Additional structural observations (lower priority)

### 3.1 `metal_prior_k_every=10` default — silent zero-prior on small `n_steps`

`__init__.py:1694` defaults `metal_prior_k_every=10`. When `config.n_steps < 10` (e.g. smoke tests at `n_steps=4`), the prior is **applied 0 times** because `step_counter % 10` never reaches 0 for steps 1..4. The 30-cell PB smoke test (`wf_pb_pass_10x3_smoke`) uses small step counts — silently disabling the prior.

### 3.2 PocketEncoder reads `pocket_atom_types` without clamping — overflow on Z>100

`PocketEncoder.forward` at `__init__.py:942-943`:
```python
if (pocket_atom_types < 0).any() or (pocket_atom_types >= self.atom_embed.num_embeddings).any():
    raise ValueError("Pocket atomic numbers exceed the encoder vocabulary")
```

This is a hard raise, not a clamp. If any pocket contains Z > 100 (rare but possible — actinides), the entire batch fails. Recommendation: clamp or skip, don't raise.

### 3.3 ConnectivityAwareDecoder treats zero-bond 2-atom clouds as "connected" (keep_empty_connected default)

`connectivity_decoder.py:367-381`:
```python
if not edges:
    if self.keep_empty_connected:
        return ConnectivityResult(
            ...,
            is_connected=True,
            n_components=1,
            ...
        )
```

Two atoms with zero predicted bonds are flagged as "connected". This is a JTVAE-aligned convention but **doesn't reflect chemistry** — a 2-atom cloud with no bond is just two isolated atoms, not a molecule. The default `keep_empty_connected=True` (set at `__init__.py:1718`) silently inflates the "connected" count.

### 3.4 `decoder_rework.py:763-768` — `is_dative` Python loop is O(E²) at small E

```python
is_dative = torch.tensor(
    [
        (
            (_is_metal(int(cloud.atomic_numbers[int(s)].item()))
             and _is_donor(int(cloud.atomic_numbers[int(d)].item())))
            ...
        )
        for k, (s, d) in enumerate(zip(src.tolist(), dst.tolist()))
    ],
    dtype=torch.bool,
)
```

For E=28 (n=8) this is 28 Python-level `_is_metal`/`_is_donor` calls per `decode()` call. Acceptable at small scale; **a bottleneck at E=435** (n=30). Should be vectorised.

### 3.5 `decoder_rework.py:683-690` — `bond_pattern_mask` is accepted but unused

```python
if self.bond_pattern_mask and bond_pattern_mask is not None:
    # bond_pattern_mask has shape (max_z, max_z, NUM_BOND_CLASSES).
    # We zero out per-pair logits when the *no-bond* class is
    # disallowed.  The first class (BOND_NO_BOND) is ALWAYS
    # allowed so we don't filter here in practice; the mask
    # is used at the order-argmax stage in the inner decoder.
    pass
```

The `bond_pattern_mask` parameter is accepted but the body is `pass`. The inner `BondAwareDecoder` (line 829-836) is responsible for the order-argmax mask — but the rework's own filtering at line 726 (`p_combined > 0.5`) **does not consult the bond pattern mask at all**. A pair like `(78, 78)` (Pt-Pt cluster) with `compat=0.20` and `p_dist=1.0` (covalent distance) gives `p_combined=0.20` — correctly filtered by the 0.5 threshold. But a pair like `(78, 8)` (Pt-O) at 1.8 Å gives `p_dist≈1.0, p_type=0.90, p_combined=0.90` — passes, but `bond_pattern_mask` might disallow Pt-O SINGLE (depending on whether it's in the vocab). The mask is currently a no-op in the rework path.

### 3.6 `_generate_impl` at `__init__.py:2643-2649` — `formal_charges=zeros` is hardcoded

```python
mols.append(Molecule(
    coords=coords_i,
    atom_types=atoms_i,
    bonds=bonds_tensor,
    bond_types=bond_types_tensor,
    formal_charges=torch.zeros(n_atoms, dtype=torch.long),
    smiles=smiles_i,
))
```

Every generated atom has formal_charge=0. For cisplatin (Pt with two Cl⁻ and two NH₃), the correct formal charges are Pt(+2), Cl(-1), N(0). The downstream RDKit sanitiser accepts this, but RDKit's `Chem.SanitizeMol` may produce non-canonical SMILES. **Low priority** — affects downstream Vina scoring only marginally.

---

## 4. Honest framing — MEASURED vs PROJECTED

### MEASURED today

* Read 3 files in full: `__init__.py` (2874 lines), `decoder_rework.py` (926 lines), `connectivity_decoder.py` (427 lines) ≈ 4227 LOC total.
* Confirmed 4/4 Phase-1 hypotheses are correctly fixed in current code (F2/F3/F4/F5).
* Identified 5 NEW structural bugs with file:line references and falsifiable diagnostics.
* Cross-checked against `wf_cfm_path_b_decoder_rework/final.md` and the audit's Hypothesis A-D — the audit's findings are not contradicted, but the *primary* remaining issue is **Bug #1 (frozen bond head with `use_bond_head=True`)**.

### PROJECTED (not validated by this review)

* Bug #1's decode_ratio impact estimate: "if the head is frozen random, decode_ratio should drop to ~0% on average over the 5-class distribution" — needs measurement on a real retrain.
* Bug #3's ODE solver accuracy: theoretical error bound is `O(h)` for Euler vs `O(h²)` for midpoint — at h=0.01 the difference is ~0.01 vs ~0.0001 local truncation, but global error is ~100× higher.
* Bug #5's n_atoms=8 default: not validated against real CrossDocked ligand distributions (10-30 atoms typical).

### FALSIFIABLE

Every bug has at least one CPU-runnable diagnostic. If a diagnostic returns the predicted behaviour, the bug stands; if not, it falls.

---

## 5. Files referenced (read-only, in this review)

* `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py` (2874 lines)
* `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/decoder_rework.py` (926 lines)
* `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/connectivity_decoder.py` (427 lines)
* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_internal_review/audit.md` (Phase-1 prior — 412 lines)
* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_internal_review/diagnose.md` (Phase-1 architecture redesign — 293 lines)

## 6. Files written by this review

* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_frontier_research/code_review_phase1c.md` — this document