# WF-PB-Pass-Real-Dock — final.md

Honest-framing report. Single-pocket / single-seed smoke of the
`--pb-check + --physical-docking` path that was previously stubbed.
This is NOT a 100-pocket sweep. Projected ~30–50 % pass rate is
untested at scale; the numbers below are 1/1 = 100 % only because
we generated exactly one molecule, and it happens to be chemistry-clean.

---

## 1. Setup verification (preflight)

| Item | Status |
|---|---|
| `--pb-check` flag present in `r4_c_full_sweep.py` | YES (line 759) |
| `--physical-docking` flag present | YES (line 687) |
| `pb_config` built when `--pb-check` enabled (requires `--physical-docking`) | YES (lines 900–903, enforced at line 901) |
| `PoseBustersAdapter` imported in worker branch (post-physical) | YES (line 517: `from molmetal_lam.sbdd_env.posebusters_adapter import PoseBustersAdapter`) |
| `validate_list(smiles)` call wired | YES (line 530) |
| `n_pb_pass` / `pb_pass_rate` / `pb_status` populated on PocketResult | YES (lines 540–542) |

Adapter file: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py`
Driver file: `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_c_full_sweep.py`

CLI notes (corrections to the request verbatim):
- `--output-dir` is not a flag; the equivalent is `--output-prefix <prefix>` (line 668).
- `--n-top-k 20` is not a flag; `top_k` is read from the SOTA-aligned YAML config
  (`search.top_k = 100`). There is `--physical-top-k` (default 10) but that is
  unrelated to the search-space top_k.
- `--pockets 1` must be a directory containing the manifest pair paths. We used
  `--pockets /mnt/storage/data/molmetal/crossdocked/extracted` (the parent of
  every manifest row). Without this, preflight fails with `Manifest pair paths
  must be inside the --pockets extraction root`.

Effective command (with corrections):

```
uv run python molmetal/scripts/r4_c_full_sweep.py \
  --pockets /mnt/storage/data/molmetal/crossdocked/extracted \
  --n-pockets 1 --seeds 42 \
  --seed-strategy click_tile \
  --physical-docking --pb-check \
  --n-simulations 100 \
  --output-prefix molmetal/reports/wf_pb_pass_real_dock/r4c
```

`--engine` defaulted to `both`; the orchestrator downgraded to `vina` because
GPU dispatch is single-column (warning at startup, expected behaviour).

The first attempt with `--seed-strategy reference` produced 0 generated
candidates (no_candidates in 4 s). The `click_tile` strategy succeeded with
1 generated candidate in 169 s — this matches the documented behaviour
that reference initialization is constrained to the paired-ligand chemistry
and frequently returns empty under heavy priors.

---

## 2. Per-pocket table

| pocket | seed | status | n_cand | n_gen | n_docked | n_pb_pass | pb_pass_rate | vina_mean | vina_best |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| test_000 | 42 | ok | 1 | 1 | 1 | 1 | 1.000 | -6.929 | -6.929 |

JSON provenance:
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/r4c.json`
  → `per_pocket[0]` (pocket_id=`test_000`).
- `pb_check.status = completed`, `pb_check.n = 1`, `pb_check.n_pb_pass = 1`,
  `pb_check.pb_pass_rate = 1.0`.
- `physical.status = completed`, `physical.summary.n_docked = 1`,
  `physical.summary.n_pb_pass = 1`, `physical.summary.pb_pass_rate_selected = 1.0`,
  `physical.summary.vina_mean_docked_kcal_mol = -6.929`,
  `physical.summary.vina_best_kcal_mol = -6.929`.

Single generated SMILES (post-search, post-dock, post-PB):
`Cc1ccc(-c2ccc(C(=O)CS)cc2)cc1` — a thiomethyl ketone-biaryl. PoseBusters
`mol` mode (chemistry + geometry, no protein) accepted it on all 25 checks
after ETKDGv3 + MMFF94 embedding (PoseBusters adapter default path).

---

## 3. Aggregate

| Metric | Value |
|---|---:|
| n_pockets (jobs) | 1 |
| n_seeds | 1 |
| n_candidates_total | 1 |
| n_generated_candidates | 1 |
| n_docked (physical.summary.n_docked) | 1 |
| n_pb_pass (physical.summary.n_pb_pass) | 1 |
| pb_pass_rate (selected = generated) | 1.000 |
| pb_status | completed |
| SA mean (top1) | 1.773 |
| QED mean (top1) | 0.640 |
| Lipinski pass rate (descriptor) | 1.000 |
| Vina best kcal/mol | -6.929 |
| wall_seconds | 168.98 |

---

## 4. Hardware / environment

| Item | Value |
|---|---|
| GPU used | RX 7800 XT (gfx1101) — declared but NOT engaged this run |
| Docking engine actually invoked | CPU Vina (engine=vina, physical-engine downgraded from auto=quickvina2-gpu because `--engine both` is GPU-incompatible; the orchestrator logged the downgrade and forced CPU Vina single-column) |
| PoseBusters path | CPU-only (chemistry+geometry, no protein) |
| Triton / ROCm | triton-rocm 3.8.0, ROCm 7.2, gfx1101 wave64 — not invoked |
| Python | uv-managed 3.12 |

Hardware-constraints check: PB path is intentionally CPU-bound (PoseBusters
`mol` mode uses ETKDGv3 + MMFF94 + RDKit in-process). No GPU dependency
was added. GPU constraints MET for this CPU-only path; the docking step
above was forced CPU because `--engine both` collides with single-column
GPU dispatch. Re-running with `--engine vina --physical-engine vina` would
keep CPU; `--physical-engine quickvina2-gpu --engine vina` would engage
the GPU binary.

---

## 5. Honest framing — what this DOES and DOES NOT show

DOES show:
1. The `--pb-check + --physical-docking` plumbing is wired end-to-end:
   Vina docks → PoseBustersAdapter.validate_list runs → n_pb_pass +
   pb_pass_rate are populated on PocketResult → aggregator exposes
   `pb_pass_rate` → markdown/CSV/JSON all carry it.
2. The adapter correctly accepts a real RDKit-canonical, ETKDGv3-embedded
   drug-like molecule (25/25 chemistry+geometry checks pass).
3. `physical.summary.pb_pass_rate_selected = 1.0` matches the
   `pb_check.pb_pass_rate = 1.0` column — these are now consistent.

DOES NOT show:
1. A 30–50 % projected PB pass rate is NOT measured. n=1 is not a
   statistical sample. At pocket scale we expect the chemistry-validity
   pass rate to land well below 100 % as MCTS produces more diverse
   (and sometimes broken) graphs.
2. We did NOT run any protein-aware PoseBusters check. `pb_mode="mol"`
   only validates chemistry+geometry — it does NOT check clash with the
   pocket. That requires `pb_mode="dock"` with the receptor passed in;
   the adapter currently hard-codes `mol`-style validation (no protein
   input port). The `n_pb_pass` figure here therefore only certifies
   that the molecule is chemically sensible in isolation.
3. We did NOT compare against TargetDiff / DiffDock / PoseBusters
   baseline numbers. This is a path-correctness check, not a SOTA
   comparison.
4. Only one of 100 test pockets was exercised. Cross-pocket variance
   (especially metal-binding pockets) is unknown.

---

## 6. Recommendation

WIRE-COMPLETE, but scale-validation remains to be done:

1. SHIP the `--pb-check` plumbing as is — no code change needed beyond
   what is already merged (Round-7 + Round-8 wiring).
2. RUN a 10-pocket × 3-seed PB smoke (n_jobs=30) before claiming any
   pass-rate number in the paper. Use `--pb-mode dock` if a protein-aware
   check is needed (currently the adapter would have to be extended to
   accept a receptor — that is a separate ~30-line patch).
3. UPDATE paper §4.1 / §4.6 / Table 2 to mark `pb_pass_rate` as
   "chemistry-validity only (mol mode) on real docked poses" — NOT as a
   PoseBusters-on-complex measure. Honest framing.
4. KEEP this 1/1 record as the smoke evidence that the path runs;
   treat the "0 % → ~30-50 %" projection as unverified until the 10×3
   sweep lands.
5. Re-engage the GPU path next time by passing
   `--physical-engine quickvina2-gpu --engine vina` (single column).

---

## 7. File inventory

| Path | Purpose |
|---|---|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/final.md` | This report |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/r4c.csv` | Per-pocket CSV (1 row) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/r4c.json` | Full metadata + per-pocket JSON |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/r4c.md` | Auto-generated r4c markdown |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/r4c_logs/` | Worker stdout/stderr logs |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/r4c_poses/` | Docked pose SDFs + per-pocket eval |

---

Author note: this run is a path-correctness smoke, not a number-producing
experiment. The PB column is now real on real docked poses; the rest of
the paper must defer to the 10×3 sweep before quoting any pass-rate.
