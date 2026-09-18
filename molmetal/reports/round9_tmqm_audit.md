# Round 9 — tmQM Pretrained Checkpoint ↔ EGNNVelocityField Audit

Date: 2026-09-13
Author: sub-agent audit (read-only)
Scope: explain the **0/42 key-match gap** when loading
`molmetal/checkpoints/dmpnn_tmqm_pretrained.pt` into
`EGNNVelocityField` via `load_tmQM_pretrained(...)`.

---

## (a) Checkpoint `mpnn_config` schema (full dict)

Source: `molmetal/checkpoints/dmpnn_tmqm_pretrained.pt`
(top-level keys: `encoder`, `mpnn_config`, `meta`).
Dumped with:

```bash
uv run python -c "import torch; ckpt=torch.load(\
'molmetal/checkpoints/dmpnn_tmqm_pretrained.pt',\
map_location='cpu', weights_only=False);\
print(ckpt['mpnn_config']); print(ckpt['meta'])"
```

`ckpt['mpnn_config']`:

```python
{
    'atom_feat_dim': 39,
    'edge_feat_dim': 6,
    'hidden_dim':   128,
    'n_layers':     3,
    'dropout':      0.1,
}
```

`ckpt['meta']` (relevant fields only):

| field | value |
|---|---|
| source | `tmQM (Balcells & Skjelstad, JCIM 2020) TPSSh-D3BJ/def2-SVP` |
| metals | `['Pt', 'Ru', 'Ir']` |
| n_pretraining_samples | 21615 |
| n_train / n_val | 19454 / 2161 |
| n_failed_featurization | 2 |
| tasks | `['coord_number(MND)', 'metal_wiberg_bo_total']` |
| target_mean / target_std | `[5.992, 4.811]` / `[1.767, 0.911]` |
| epochs | 10 |
| batch_size | 32 |
| lr | 1e-3 |
| optimizer | `AdamW` |
| n_params | 624,706 |
| best_val_loss | 0.04504 |
| final_val_loss | 0.04554 |
| final_train_loss | 0.04554 |

**Architectural pedigree** (resolved via grep):
- `molmetal/models/dmpnn.py:130` — `@dataclass class MPNNConfig`
- `molmetal/models/dmpnn.py:173` — `class DirectedMPNN(nn.Module)`
- `molmetal/models/dmpnn.py:193` — `def __init__(self, config: Optional[MPNNConfig] = None)`
- The `DirectedMPNN` consumes the 5 `mpnn_config` fields above: `atom_feat_dim`, `edge_feat_dim`, `hidden_dim`, `n_layers`, `dropout`. It produces a per-atom hidden vector (NOT an equivariant 3-D velocity).

**State-dict namespace under `ckpt['encoder']`**:
- All parameters are owned by the `DirectedMPNN` message-passing stack:
  `atom_embed.0.weight/bias`, `atom_embed.2.weight/bias`,
  `edge_embed.0.weight/bias`, `edge_embed.2.weight/bias`,
  `edge_mlp.<i>.0.weight/bias` for `i ∈ [0, n_layers)`,
  `W_aroma`, plus message / readout heads trained on
  `(coord_number, wiberg_bo_total)`.
- **42 total keys**, none of which are SE(3)-equivariant, none of which
  correspond to an EGNN `x` coord graph, `vel_head`, `time_mlp`, or
  `atom_head`.

---

## (b) Current `EGNNVelocityField.__init__` signature

Source: `molmetal/adapters/flow_matching_lipman/__init__.py`
(the EGNN module lives here; there is no separate
`egnn_velocity.py` — grep confirms the file does not exist).
Resolved lines: class at `:355`, constructor at `:365`.

```python
class EGNNVelocityField(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 3,
        max_atomic_number: int = 100,
        edge_mlp_hidden: Optional[int] = None,
        tmqm_init_path: Optional[str] = None,
        use_tmqm_init: bool = False,
    ) -> None:
        ...
        self.time_mlp   = _MaybeFusedSiLUMLP(1, hidden_dim, hidden_dim)
        self.atom_embed = nn.Embedding(max_atomic_number, hidden_dim)
        self.layers     = nn.ModuleList(
            [EGNNLayer(hidden_dim=hidden_dim,
                       edge_mlp_hidden=edge_mlp_hidden or hidden_dim)
             for _ in range(n_layers)]
        )
        self.vel_head   = nn.Linear(hidden_dim, 3, bias=False)   # zero-init
        self.atom_head  = nn.Linear(hidden_dim, max_atomic_number, bias=False)
        # zero-init
        ...
        if use_tmqm_init:
            load_tmQM_pretrained(self, ckpt)
```

**Parameters actually wired from `mpnn_config`** (today, via
`LipmanFlowMatchingAdapter.__init__` at `:785` and its hard-coded
defaults — there is **no `**ckpt_kwargs` pass-through**):

| Adapter kwarg | EGNN kwarg | value |
|---|---|---|
| `hidden_dim=128` | `hidden_dim=128` | **matched** |
| `n_layers=3` | `n_layers=3` | **matched** |
| `max_atomic_number=100` | `max_atomic_number=100` | matched (default) |
| — | `edge_mlp_hidden` | None → `hidden_dim` |
| `tmqm_init_path=None` | `tmqm_init_path` | matched |
| `use_tmqm_init=False` | `use_tmqm_init` | matched (default False) |

**`mpnn_config` keys NOT consumed by the adapter / EGNN ctor at all**:
- `atom_feat_dim` (39) — meaningless to EGNN (uses atomic-number embedding)
- `edge_feat_dim` (6)   — meaningless to EGNN (uses radial / coord MLP)
- `dropout` (0.1)       — no dropout knob on EGNNVelocityField at all

---

## (c) Bridge plan — closing the 0/42 key-match gap

### Why current load fails (0/42)

`load_tmQM_pretrained(...)` (delegated from `EGNNVelocityField.__init__`)
and its fallback `_load_tmqm_checkpoint(path, strict=False)` both call:

```python
self.load_state_dict(ckpt['encoder'], strict=False)
```

`strict=False` means **missing keys in `self` and unexpected keys in
`ckpt['encoder']` are both reported but do not raise**. Result:
every one of the 42 `DirectedMPNN` keys (e.g. `atom_embed.0.weight`,
`edge_mlp.0.0.bias`, `W_aroma`) shows up as `unexpected_keys` because
the EGNNVelocityField parameter set is disjoint (it has
`time_mlp.*`, `atom_embed.weight` (Embedding, different shape),
`layers.<i>.mlp_*.*`, `vel_head.weight`, `atom_head.weight`). **Zero
of the 42 keys find a `state_dict` slot on the EGNN side → 0/42 match.**

This is a **structural mismatch**, not a bug:
- `DirectedMPNN` is a **directed message-passing** network on
  (atom_features, edge_features). It is not equivariant to rotations /
  translations and it has no concept of `x` (3-D coords).
- `EGNNVelocityField` is an **SE(3)-equivariant** graph net on
  `(x, h, edge_index)` with a coordinate update `x ← x + vel_head(...)`.
- They share two field names (`hidden_dim`, `n_layers`) and roughly the
  same parameter count (624k vs ~600k EGNN), but their **parameter
  names and tensor shapes are disjoint**.

### Three bridge options (ordered by invasiveness)

#### Option 1 — Accept the gap, document it (zero code change)
- Keep `strict=False` and **log the actual key-match count**
  (currently swallowed as a warning).
- Surface `unexpected_keys` + `missing_keys` counts in the load log so
  operators see "loaded 0/42 keys" and can make an informed call.
- Update the F2 / Round 7 / Round 8 narrative to be honest: "tmQM
  pretraining warm-starts are **architecturally impossible** for the
  EGNN flow head; only `hidden_dim=128, n_layers=3` are inherited."
- Cheap, but does not buy any transfer-learning signal.

#### Option 2 — Project through a learned linear adapter (small change)
- Add a `DirectedMPNN → EGNN` projection layer that maps the 42-key
  DirectedMPNN state-dict into the EGNN's `atom_embed.weight` shape
  `(max_atomic_number, hidden_dim) = (100, 128)`. Specifically:
  - Take `ckpt['encoder']['atom_embed.2.weight']` (shape
    `(hidden_dim, hidden_dim) = (128, 128)`) as the initialiser for an
    EGNN-side `nn.Embedding(100, 128)` slice `[0:atomic_number]`.
  - Discard the rest (`edge_embed.*`, `edge_mlp.*`, `W_aroma`, heads)
    — those are 39/6-dim DirectedMPNN semantics and have no EGNN
    analogue.
- Net effect: **42 → 1** key used (the final atom-embed MLP weight).
- Pro: actually transfers signal (the tmQM atom embeddings carry
  chemistry of Pt/Ru/Ir).
- Con: still a one-tensor init; "warm start" is generous wording.

#### Option 3 — Re-pretrain / co-train the EGNN on tmQM (right thing)
- Add an EGNN head that **predicts the same 2 tmQM tasks**
  (`coord_number(MND)`, `metal_wiberg_bo_total`) so the encoder-side
  pretraining loss can be back-prop'd into `EGNNVelocityField`.
- Specifically:
  1. Extend `EGNNVelocityField.forward` with an optional
     `pretrain_tasks: Literal["coord_number", "wiberg_bo"]` arg.
  2. Add a small `(N, 2)` MLP head on top of the atom-level hidden
     state.
  3. Pretrain on `21,615` tmQM samples for 10 epochs (matches the F2
     config: AdamW lr=1e-3, batch=32, target_mean/std
     `[5.99, 4.81] / [1.77, 0.91]`).
  4. Save `state_dict` + `mpnn_config`-shaped meta (rename to
     `egnn_config`) at `molmetal/checkpoints/egnn_tmqm_pretrained.pt`.
  5. Wire `LipmanFlowMatchingAdapter.__init__(use_egnn_init=True,
     egnn_init_path=...)` → `EGNNVelocityField` with `strict=True`.
- Pro: **42/42** matches possible, real transfer learning.
- Con: ~30–60 min on RX 7800 XT (TMQM ~22k samples × 10 epochs ×
  small EGNN) + breaks `load_tmQM_pretrained` symmetry — the existing
  DirectedMPNN ckpt is left unused (could be kept for the discrete
  atom-type prior, separately).

### Recommended path (combines 1 + 2 for round-9 ship, queues 3 for round-10)

1. **Round-9 (this week) — Option 1 + minimal Option 2:**
   - Make `load_tmQM_pretrained` log the `unexpected_keys` /
     `missing_keys` counts explicitly (so the "0/42" is no longer
     invisible).
   - Add a `pretrain_atom_embed_init: bool = False` flag that, when
     `True`, copies `ckpt['encoder']['atom_embed.2.weight']` into a
     slice of `EGNNVelocityField.atom_embed.weight`. This converts the
     "0/42 match" into "1/42 used, rest discarded", which is honest
     and gives us real transfer on the atomic-embedding subspace.
   - Keep `strict=False` and the try/except safety net.

2. **Round-10 (queue) — Option 3:**
   - Pretrain the EGNN itself on tmQM with a `(coord_number, bo)`
     multi-task head, save an EGNN-shaped ckpt, and switch the warm
     start to `strict=True` for a clean 42/42 (or N/N) match.

### Files that would change (no edits made — read-only audit)

- `molmetal/adapters/flow_matching_lipman/__init__.py`
  - `_load_tmqm_checkpoint` (around `:442`) — add key-count logging
    and the optional `pretrain_atom_embed_init` slice copy.
- `molmetal/models/dmpnn.py`
  - (no change to `DirectedMPNN`, but reference for shape / names).
- (Round-10) New file:
  `molmetal/scripts/pretrain_egnn_tmqm.py` — pretrain loop emitting
  `molmetal/checkpoints/egnn_tmqm_pretrained.pt`.

### Quick verification recipe (for the next agent that picks this up)

```python
import torch
from molmetal.adapters.flow_matching_lipman import (
    EGNNVelocityField, load_tmQM_pretrained,
)
ckpt = torch.load('molmetal/checkpoints/dmpnn_tmqm_pretrained.pt',
                  map_location='cpu', weights_only=False)
vf = EGNNVelocityField(hidden_dim=128, n_layers=3, max_atomic_number=100)
# After patching load_tmQM_pretrained to log key counts:
# expected today:  matched=0, unexpected=42, missing=~12
```

---

## TL;DR

- **0/42 is real and structural**, not a bug:
  DirectedMPNN (39-dim atom / 6-dim edge) ↔ EGNNVelocityField
  (atomic-number embed + radial coord MLP) are disjoint
  `state_dict` namespaces.
- The only `mpnn_config` fields that currently flow into the EGNN
  ctor are `hidden_dim` and `n_layers`. `atom_feat_dim`, `edge_feat_dim`,
  `dropout` are unused.
- Round-9 close-out: log the gap honestly + salvage the atom-embed
  linear slice. Round-10: pretrain the EGNN itself on tmQM and ship
  an EGNN-shaped ckpt for a clean warm start.
