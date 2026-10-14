# Pocket Encoder Pipeline — A31/A32 Architecture

> **Status (2026-10-14):** A31 prep COMPLETE (literature pool declared). A32
> wire-in PENDING (PocketGen + PerLayerPocketInjector not yet instantiated in
> the live EGNN loop). Wf-A30 audit surfaced this as a critical cascade risk:
> the CFM pocket-conditioning slot `c` was running on `c=None` for every
> PlatinAI 449-mol eval. See `HONEST_NEGATIVE_CATALOG.md` §6.51 for the
> composite honest-negative chain.

This document describes the **complete pocket-encoder pipeline** that the
CFM (Conditional Flow Matching) model uses to condition on the protein
pocket of the cancer target. The pipeline is the answer to the audit
finding that `c` was being silently underused: it now has a named,
auditable, pluggable end-to-end path.

---

## 1. The big picture

```
                              ┌────────────────────────────────────┐
                              │ CFM trainer (R17+ model)           │
                              │                                    │
                              │ pocket_per_layer = True (Wf-A32)   │
                              │ ┌──────────────────────────────┐  │
                              │ │ EGNNVelocityField.forward     │  │
                              │ │   for layer in range(L):      │  │
                              │ │     h += PerLayerPocketInj(   │  │ ← Gap #2 fix
                              │ │            c, n_atoms)        │  │
                              │ │                              │  │
                              │ │ _encode_pocket(...)           │  │
                              │ └──────────────┬───────────────┘  │
                              └────────────────┼──────────────────┘
                                               │ c: (B, pocket_embed_dim)
              ┌────────────────────────────────┴────────────────────────┐
              │                                                            │
              ▼                                                            ▼
   ┌────────────────────────┐                                ┌───────────────────────┐
   │ PocketGenEncoder       │  ← Wf-A32 wire-in              │ PlatinAI / MetalCyto  │
   │ (molmetal/adapters/    │                                │ 449 mol cohort        │
   │  pocketgen/)           │                                │ (no 1:1 pocket anno)  │
   │                        │                                └───────────┬───────────┘
   │ BilevelEncoder (MIT)   │                                            │
   │ ESM-2 adapter          │                                            │
   │ mean+max pool → z      │                                            │
   └───────────┬────────────┘                                            │
               │                                                         │
               ▼                                                         │
   ┌────────────────────────┐                                            │
   │ fpocket_wrapper        │  ← Wf-A33 v2 wire-in (DONE — no Java)      │
   │ (molmetal/adapters/    │                                            │
   │  fpocket_wrapper.py)   │                                            │
   │                        │                                            │
   │ full PDB → residue crop│                                            │
   │ 10 Å sphere            │                                            │
   └───────────┬────────────┘                                            │
               │                                                         │
               ▼                                                         │
   ┌────────────────────────┐                                ┌───────────────────────┐
   │ platiny_pocket_pool    │  ← A31 ship (10 entries, A33 │ crossdocked_pool       │
   │ (molmetal/data/)       │    v2 10/10 extracted)      │ (100K crossdocked 2020)│
   │                        │                                │                       │
   │ 10 literature Pt / Au  │                                │ ready-to-use (R21 GPU) │
   │ / Ru target PDBs       │                                │                       │
   └────────────────────────┘                                └───────────────────────┘
                                                ▲
                                                │
                                ┌───────────────┴───────────────┐
                                │ pocket_pool_unified           │  ← A32 prep
                                │ (molmetal/data/)              │
                                │                               │
                                │ single dispatch API over both │
                                │ pools (source='crossdocked'  │
                                │  | 'platiny' | 'all')         │
                                └───────────────────────────────┘
```

## 2. The 5 modules in detail

### 2.1 `molmetal/adapters/pocketgen/` — PocketGen encoder (vendored)

**Upstream**: [zaixizhang/PocketGen](https://github.com/zaixizhang/PocketGen) (MIT, 2024)
**Paper**: Zhang et al., *Accurate Protein Pocket Design with a Bilevel Graph Transformer
and an n-gram Decoding Scheme*, Nat. Mach. Intell. 2024.

**What we keep**: encoder half only (BilevelEncoder / HierEncoder / GET) — the
generation head + chemistry + relaxation pipeline are excluded. Total ~3,500 LOC
across 9 files. `models/PD.py` (the full decoder) and `models/data.py` (their
dataloader) are removed.

**What we add**: `__init__.py` exposes a `PocketGenEncoder` wrapper class with:
- lazy import (heavy deps `torch_scatter` / `torch_geometric` / `fair-esm` / `omegaconf`
  load only on `.load()`);
- mean+max pool → fixed-size `(B, 2*hidden_channels)` embedding;
- optional learnable Linear projection to a `pocket_embed_dim` that may differ from
  PocketGen's `hidden_channels`;
- pretrained weights via `.from_pretrained(checkpoint_path)`.

**Why PocketGen and not ESM-2 / GearNet**: PocketGen is **pocket-specific** SOTA
(Nat. MI 2024); ESM-2 / GearNet are general-protein encoders and need finetuning.
PocketGen also has an ESM-2 adapter internally, so we get the best of both.

**License**: MIT (permissive; safe for commercial metallodrug IP — unlike DecompDiff
which is CC BY-NC 4.0 and would block).

### 2.2 `molmetal/adapters/p2rank_wrapper.py` — Pocket detection (CPU-only stub)

**Upstream**: [rdk/p2rank](https://github.com/rdk/p2rank) (MIT, J Cheminform 2018)

**What we ship NOW** (Wf-A33 prep):
- `P2RankResult` dataclass (rank / score / probability / centre / residue list).
- `parse_p2rank_csv(csv_path)` — pure-Python parser, no subprocess.
- `crop_pdb_to_residues(pdb_path, residues, out_path, sphere_radius_A=10.0)` — uses
  `Bio.PDB` (already a project dep) to crop a full PDB to a 10 Å sphere around
  the given residues. Tested with the Bio 1.88 + `PDBIO` API.
- `is_p2rank_available()` / `_resolve_p2rank_cmd()` — checks for the Java CLI
  (system dep; not in `.venv`).
- `top_pockets(csv_path, n, min_probability)` — convenience filter.
- `run_p2rank(...)` — raises `NotImplementedError` with install instructions.
  **Wf-A33 wire-in target** (user OK required for the Java dep).

**Why this exists**: the platiny_pocket_pool declares 10 literature PDB IDs but
the pocket files are not extracted. P2Rank is the upstream detector that turns
a full PDB into a residue range we can then crop to a 10 Å sphere and feed to
PocketGenEncoder.

### 2.3 `molmetal/data/platiny_pocket_pool.py` — Literature Pt-target pool

**What**: 10 hand-curated precious-metal anticancer drug target proteins, each
with:
- PDB ID + target name + metal specificity (Pt_II / Au_I / Ru_II / etc.)
- Literature DOI (real references, no fabrication per the "不要自己造表" memory rule)
- `pocket_path` populated for all 10 (populated 2026-10-14 in Wf-A33 v2; was empty in A31 prep; column now holds absolute paths under `.data/molmetal/platiny_pockets/`)

**Why this exists**: PlatinAI 449-mol has only `smiles,source` columns, no
`pocket_id`. The CFM pocket-conditioning slot `c` was therefore either
fed `None` (degenerate to unconditioned) or a random surrogate pocket (no
causal binding truth). The platiny pool is the **honest answer**: a small,
named, literature-anchored set of real Pt / Au / Ru target proteins that we
can use to MEASURE pocket-conditioned generative quality with proper §5.2
honest framing.

**Status as of 2026-10-14**: 10 entries shipped, **10 with extracted pockets** (Wf-A33 v2 / fpocket, no Java).
Per `molmetal/data/platiny_pocket_pool.csv` the `pocket_path` column is populated
for all 10 rows and `list_pocket_pool(only_extracted=True)` returns **10** entries
(verified 2026-10-14 at fixup-2 via venv python import); pocket files live at
`.data/molmetal/platiny_pockets/<pdb>_pocket10.pdb` (extracted via
`molmetal/scripts/wf_a33_extract_platiny_pockets.py`, commit `f51c3836b1`,
catalogued at HONEST_NEGATIVE_CATALOG.md §6.60b).

### 2.4 `molmetal/data/pocket_pool_unified.py` — Single dispatch API

**What**: thin wrapper that exposes a single `load_pocket_pool(source=..., ...)`
function over the two backends (`crossdocked_pool` and `platiny_pocket_pool`).

```python
from molmetal.data.pocket_pool_unified import load_pocket_pool, source_of

# 100K CrossDocked (in-domain, ready-to-use)
train = load_pocket_pool(source="crossdocked2020", split="train", max_n=10000)

# 10 literature Pt / Au / Ru targets (eval-time honest pool)
platiny = load_pocket_pool(source="platiny")  # all 10, pocket_path populated

# Pt_II filter on platiny
pt_ii = load_pocket_pool(source="platiny", metal="Pt_II")  # 4 rows

# Combined (caller's responsibility to know they're not equivalent)
all_rows = load_pocket_pool(source="all", max_n=100)

# Source discrimination
for r in all_rows:
    if source_of(r) == "platiny":
        # ... eval-time framing
        pass
    else:
        # ... in-domain training
        pass
```

**Why two pools, not one**: the two pools have different semantics — crossdocked
has 100K real pockets ready-to-use; platiny has 10 declared targets with no
1:1 binding annotation. Merging them into one CSV would lose this distinction
(cf. audit finding #6: "no shared helper" — the unified loader is the fix
without merging).

### 2.5 `molmetal/adapters/egnn_extensions.py::PerLayerPocketInjector`

**What**: a `nn.Module` that projects a per-sample pocket vector `(B, pocket_dim)`
to `(B, N, hidden)` and adds it to the EGNN hidden state at every layer.
**Status**: class exists + works (12/12 unit tests PASS) but is **dead code**
in the live CFM (audit finding #2: never instantiated). **Wf-A32 wire-in
target**: instantiate `self.pocket_injector = PerLayerPocketInjector(...)` in
`EGNNVelocityField.__init__` and call it inside the per-layer loop.

**Why per-layer injection matters**: at present, the pocket signal enters the
EGNN at line 1717 of `flow_matching_lipman/__init__.py` as a one-shot additive
bias on `h` and never re-enters the loop. The bilevel GraphTransformer in
PocketGen does per-layer injection by design; without it, our model only sees
the pocket signal once at the input, not at every message-passing step.

---

## 3. Honest framing — the §6.51 / §6.53 chain

Per `AGENTS.md` §5.2 honest-framing + `party-twenty-spirit` §3.5, every numeric
claim in this pipeline must carry a MEASURED-vs-PROJECTED tag.

| Claim | Tag | Why |
|---|---|---|
| PocketGen can encode pockets for our 14-metal vocab | **PROJECTED** | The wrapper is wired + lazy-import OK, but no real weights loaded yet; no run on PlatinAI 449 mol with `c=PocketGen(...)` |
| 10 literature Pt targets are the right pool | **MEASURED** | CSV on disk, 10 rows, schema compatible with crossdocked_pool |
| Pool has 10 extracted pockets (v2 / fpocket, no Java) | **MEASURED** | CSV col `pocket_path` populated on all 10 rows + `list_pocket_pool(only_extracted=True)` returns 10 entries (verified at fixup-2 2026-10-14); 10 pocket PDBs at `.data/molmetal/platiny_pockets/<pdb>_pocket10.pdb` (§6.60b) |
| P2Rank crops to 10 Å sphere | **MEASURED** | crop_pdb_to_residues tested with Bio 1.88 + PDBIO; 18 unit tests PASS |
| PerLayerPocketInjector projects (B, dim) → (B, N, hidden) | **MEASURED** | 12 unit tests PASS; zero-init bit-exact legacy preserved |
| PerLayerPocketInjector is wired into the CFM EGNN loop | **NOT SHIPPED** | audit finding #2; Wf-A32 work |
| PocketGen weights vendored | **NOT SHIPPED** | deferred to Wf-A32; ~250 MB download |
| Heavy deps (torch_scatter / torch_geometric / esm) installed | **NOT SHIPPED** | pip install deferred to A32 (user OK) |
| Pocket-conditioned generative quality lift on PlatinAI | **PROJECTED** | not yet run |

---

## 4. Open follow-ups

| Wf | Task | Status |
|---|---|---|
| **A32.1** | pip install `pocketgen` extras (torch_scatter + torch_geometric + fair-esm + omegaconf) on cu130 host | BLOCKED (user OK) |
| **A32.2** | vendor PocketGen pretrained weights (~250 MB) to `models/pocketgen/pocketgen_pretrained.pt` | BLOCKED (A32.1) |
| **A32.3** | instantiate `PerLayerPocketInjector` in `EGNNVelocityField.__init__` (flow_matching_lipman/__init__.py:1429-1452 region) | UNBLOCKED (CPU-only) |
| **A32.4** | wire `_encode_pocket` to return both `embed` and `atom-types-onehot` + mask for cross-attn path | UNBLOCKED (CPU-only) |
| **A32.5** | §6.54 honest-negative for the pre-A32 paper claims that ran on c=None | READY (A30-2 §6.51 chain) |
| **A33.1** | `fpocket_wrapper.py` (NEW, no Java) — C-binary `fpocket 4.2.3` detector + `run_fpocket()` wire-in (per user voice 2026-10-14 *"为什么还有java?"*) | **DONE** (commit `04de1dbf13`, §6.60a) |
| **A33.2** | 10 platiny targets extracted via `fpocket` + Biopython sphere-crop (no P2Rank, no Java) — `wf_a33_extract_platiny_pockets.py` writes `.data/molmetal/platiny_pockets/<pdb>_pocket10.pdb` | **DONE** (commit `f51c3836b1`, §6.60b) |
| **A33.3** | 5-seed × 50-epoch re-run of `r_platinai` with `c=PocketGen(platiny_pocket)` for honest eval-time comparison | BLOCKED (GPU + A32.3 — user 2026-10-14 *"gpu先别动"*) |

## 5. Cross-references

- `docs/HONEST_NEGATIVE_CATALOG.md` §6.51 (Wf-A30-2 composite, 8 sub-entries) —
  the audit chain that surfaced every gap this pipeline fixes.
- `docs/HONEST_NEGATIVE_CATALOG.md` §6.53 (deferred) — for the pocket-binding-
  truth gap on PlatinAI 449 mol.
- `docs/HONEST_NEGATIVE_CATALOG.md` §6.54 (deferred) — for the pre-A32 c=None
  paper claims.
- `docs/WF_HISTORY.md` — Wf-A28 + A29 + A30 + A31 prep + A32 prep timeline.
- `AGENTS.md` §5.11 (YAML-only-launch + DECLARED_INERT_PATHS) — every new
  PocketGen / P2Rank flag must round-trip through a YAML before being
  launched.
- `TODO/INDEX.md` §0 (last-updated trailer) — sync'd after Wf-A31 + A32 ship.
- `ATTRIBUTION.md` (in `molmetal/adapters/pocketgen/`) — MIT attribution +
  what we vendored / what we excluded.
