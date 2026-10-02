# Wf-392-A2 — chemprop v3 featurization cache wire-in scaffold

**Date:** 2026-09-30 (Wf-392-A2 ship)
**Author:** Hugo Lee (via Wf-392-A2 ultracode sub-flow)
**Status:** SHIPPED — wire-in scaffold complete; trainer integration follow-up
deferred to a separate wave (see §4 below).
**Standing directives:** §5.1 (GPU grant) + §5.2 (honest framing) + §5.3 (no
paper/*) + §5.4 (Tier-2 re-orientation, NO wet-lab) re-confirmed.

---

## §1 TL;DR

Wf-392-A2 wires the Wf-382-A2 chemprop v3 featurization cache into the
``train_chemprop_v2_v3.py`` trainer via a `--chemprop-v3-cache` CLI flag
(default OFF; preserves bit-for-bit backward compat). The wire-in is
**idempotent** (re-running is a no-op via the ``WIRED_MARKER`` check) and
**safe** (writes a ``.bak.wf392a2`` snapshot of the un-wired trainer before
insertion).

Shipped artifacts:

* ``molmetal/scripts/wf392a2_chemprop_v3_cache_wire.py`` (~290 LOC) — wire
  + smoke harness with CLI surface.
* ``molmetal/tests/test_wf392a2_chemprop_v3_cache.py`` (~115 LOC) — 2 NEW
  tests (test_cache_flag + test_cache_hit_rate).
* ``TODO/completed/392a2_chemprop_v3_cache_wire.md`` — this closeout.

**2 NEW tests PASS** in ~0.04 s wall (CPU-only); 0 paper/* touches; 0
wet-lab planning; 0 GPU launches.

---

## §2 cache — recap of Wf-382-A2 scaffold

Per ``molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py`` (§0):

* **Tier 1 — per-SMILES** (``precompute_smiles``): runs chemprop v3's
  ``SimpleMoleculeMolGraphFeaturizer`` (default) OR the Wf-320D
  ``FastAtomFeaturizer`` (opt-in ``use_fast_featurizer=True``) once per
  canonical SMILES. Stores ``{"V", "E", "edge_index", "rev_edge_index"}``.
* **Tier 2 — per-batch** (``get_or_build_batch``): caches the assembled
  ``BatchMolGraph`` per ``batch_idx``. Cache key is just ``batch_idx``
  because ``BatchMolGraph`` doesn't depend on chemprop v3 model weights
  (cross-epoch reuse is the actual saving).
* Hit/miss tracked via ``ChempropV3CacheStats`` (n_hit, n_miss, hit_rate,
  build_wall_s, lookup_wall_s).
* Cache is **stateless** with respect to chemprop v3 model parameters —
  the per-batch tier is safe to reuse across epochs.

Honest framing per ``party-twenty-spirit`` §3.5: cache savings PROJECTED
5-10× on featurisation speed on cross-epoch reuse (per Wf-320D §0 +
Wf-328-W10 §0 MEASURED baseline for chemprop v2 + D-MPNN caches; chemprop
v3 cache uses the same two-tier pattern).

---

## §3 wire_in — Wf-392-A2 design

### §3.1 Public functions

```python
# molmetal/scripts/wf392a2_chemprop_v3_cache_wire.py
def wire_cache_to_trainer(trainer_path: str) -> bool:
    """Add --chemprop-v3-cache CLI flag to trainer_path.

    Idempotent: returns False if already wired (WIRED_MARKER present).
    Side effect: writes {trainer_path}.bak.wf392a2 on first wire.
    """

def smoke_cache_hit_rate(
    trainer_path: Optional[str] = None,
    cohort_size: int = 10,
    epochs: int = 2,
    batches_per_epoch: int = 5,
    batch_size: int = 4,
) -> Dict[str, object]:
    """Exercise the Wf-382-A2 cache on a synthetic cohort.

    Returns dict with cache_info, hit_rate, build_wall_s,
    lookup_wall_s, n_added, tier2_exercised, tier2_error.
    """
```

### §3.2 Insertion anchor (verified at Wf-392-A2 author time)

The trainer's argparse ends with ``--allow-single-seed`` at lines
1411-1417, followed by ``args = p.parse_args(argv)`` at line 1418. The
wire-in inserts a new ``p.add_argument("--chemprop-v3-cache", ...)`` block
between them. The exact anchor is the literal string:

```python
    p.add_argument(
        "--allow-single-seed", action="store_true",
        help=(
            "Bypass the --n-seeds-min check (for smoke tests; not for "
            "paper claims).  Wf-245 Fix."
        ),
    )
    args = p.parse_args(argv)
```

### §3.3 Inserted block (what `--chemprop-v3-cache` looks like in the trainer)

```python
    # Wf-392-A2 wired: --chemprop-v3-cache CLI flag inserted.
    p.add_argument(
        "--chemprop-v3-cache", action="store_true", default=False,
        help=(
            "Enable chemprop v3 featurization cache (default: OFF; "
            "preserves bit-for-bit backward compat). When ON, the "
            "trainer initializes a ChempropV3FeaturizationCache "
            "and uses it for per-SMILES + per-batch featurization "
            "reuse across epochs. SCAFFOLD: full training-loop "
            "wire-in is a follow-up wave; this flag is wired + "
            "smoke-tested only."
        ),
    )
    args = p.parse_args(argv)
```

### §3.4 Idempotency

* First wire: snapshot trainer to ``{trainer_path}.bak.wf392a2`` + insert
  flag block + return ``True``.
* Subsequent wires: detect ``# Wf-392-A2 wired: --chemprop-v3-cache CLI
  flag inserted.`` marker in trainer file + return ``False`` (no-op).
* Backup file is bit-for-bit identical to the un-wired source (verified
  in test_cache_flag).

### §3.5 Default OFF — bit-for-bit backward compat

The trainer's existing ``_batch_molgraphs_to(bmols, device)`` calls at
lines 751 + 1031 are NOT modified by this wave. The cache only activates
when `--chemprop-v3-cache` is passed (default OFF). All pre-existing
trainer behaviour is preserved when the flag is not passed.

---

## §4 integration_tests — what the tests verify

### §4.1 test_cache_flag

Verifies the wire-in:

* Copies ``train_chemprop_v2_v3.py`` to a temp file (no mutation of real
  trainer).
* Calls ``wire_cache_to_trainer`` → returns ``True`` (new wiring).
* Verifies backup ``{path}.bak.wf392a2`` exists.
* Calls ``wire_cache_to_trainer`` again → returns ``False`` (no-op
  idempotency).
* Verifies the wired file contains `--chemprop-v3-cache` (2 occurrences:
  dest + help text).
* Verifies the wired file is valid Python via ``ast.parse`` (no syntax
  errors introduced).
* Parses `--chemprop-v3-cache --metals Ru --epochs 1` via a local argparse
  to verify the flag accepts CLI input correctly.

### §4.2 test_cache_hit_rate

Verifies the cache works on a synthetic cohort:

* Uses ``pytest.importorskip("rdkit")`` to skip the test if RDKit is not
  available.
* Calls ``smoke_cache_hit_rate`` with ``cohort_size=10``, ``epochs=2``,
  ``batches_per_epoch=5``, ``batch_size=4``.
* Verifies tier-1 precompute completes (n_added >= 1).
* On hosts WITH chemprop installed: tier-2 exercises → hit_rate > 0
  (epoch 1 builds all → 0% hit; epoch 2 hits all → ~50% total).
* On hosts WITHOUT chemprop: tier-2 fails gracefully → tier2_error
  populated + hit_rate == 0.
* Verifies result dict is JSON-serializable.

### §4.3 Honest framing

* Both tests are CPU-only; no GPU launches.
* The wire-in function does not import rdkit / chemprop / torch — pure
  text manipulation.
* The smoke function requires rdkit (uses ``Chem.MolFromSmiles``); chemprop
  is optional for tier-2.
* Test results are deterministic: same cohort + same trainer → same wired
  content (verified via byte-equal backup check).

---

## §5 cross_references

* **Wf-382-A2** — ``molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py``
  (the cache itself, 467 LOC). Wf-392-A2 is the trainer wire-in follow-up
  requested by Wf-382-A2 §0 SCAFFOLD note ("Wf-380-B1 trainer integration
  follow-up required for end-to-end wire-in").
* **Wf-328-W10** — ``molmetal/scripts/train_dmpnn_multitask_wasserstein.py``
  ``_featurize_with_cache`` (on-disk pickle cache for D-MPNN, lines 178-209).
  Different cache mechanism (on-disk pickle) — chemprop v3 cache uses
  in-memory two-tier pattern from Wf-320D.
* **Wf-320D** — ``molmetal/scripts/wf320d_featurization_fix.py``
  ``PerEpochFeatureCache`` (chemprop v2 in-memory two-tier cache, line 292+).
  Wf-392-A2 mirrors the chemprop v2 API exactly so the same integration
  pattern drops in.
* **TODO-155** §3 — chemprop v2 cache pivot. Chemprop v3 cache is the v3
  equivalent of the v2 cache wired in TODO-155.
* **AGENTS.md §5.1** — Standing GPU authorization (this wave is CPU-only;
  no GPU launches performed).
* **AGENTS.md §5.2** — Honest framing per ``party-twenty-spirit`` §3.5
  (PROJECTED vs MEASURED tag discipline; this wave cites cache savings as
  PROJECTED 5-10× per Wf-320D + Wf-328-W10 MEASURED baseline).
* **AGENTS.md §5.3** — No paper/* touches (per TODO-84 active; this wave
  has 0 paper/* touches).
* **AGENTS.md §10** — Wetlab deprecation FAQ (this wave has 0 wetlab
  planning).

### §5.1 Prior Wf chronology

* Wf-382-A2 (cache SCAFFOLD) → Wf-392-A2 (wire-in scaffold) →
  follow-up wave (training-loop integration, deferred).

### §5.2 Forward-looking gaps

* **Full training-loop integration** — replace ``_batch_molgraphs_to``
  calls at trainer lines 751 + 1031 with ``cache.get_or_build_batch(...)``.
  Closure path: ~50 LOC edit + 1 NEW test. Deferred to a follow-up wave.
* **Smoke on chemprop v3 install** — current smoke runs on tier-1 SMILES
  + degraded tier-2 if chemprop missing. End-to-end tier-2 cache hit_rate
  requires chemprop v3 install at smoke time.
* **Wave budget** — this wave is CPU-only docs+code+test, ~10-15 min wall.
  Pre-flight + tests not gated by ``pre_flight_check.py`` (docs+light code
  pattern per Wf-356 + Wf-359 + Wf-371 + Wf-377 baseline).

---

**END OF Wf-392-A2 CLOSEOUT** — owner Hugo Lee; cross-ref
``party-twenty-spirit`` §3.5; standing directives §5.1 + §5.2 + §5.3 +
§5.4 + §10 re-confirmed.