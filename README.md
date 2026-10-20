# Mol-Metal

> **TL;DR.** Mol-Metal is a precious-metal anticancer drug design framework (Pt/Ru/Ir/Au/Pd) on top of Flow Matching + Structure-Based Drug Design + Lambda Calculus MCTS proof-search.

## Quickstart

See [`QUICKSTART.md`](QUICKSTART.md) — install + console-script entry point + 5-min smoke.

## Develop

Agent-facing entry point: [`AGENTS.md`](AGENTS.md) — project rules + skill catalog + standing directives. Project TOC + overview: [`TODO/README.md`](TODO/README.md). Single-source-of-truth per file status: [`TODO/INDEX.md`](TODO/INDEX.md).

## Canonical trainers (Wf-A37-8)

For **precious-metal anticancer drug design** (Pt/Ir/Ru/Au/Os/Re/Rh), the canonical
config + trainer pair is:

| Axis | Entry point | Default |
|---|---|---|
| CFM decoder | [`configs/cfm_5metal_default.yaml`](configs/cfm_5metal_default.yaml) | cohort pinned to `metalcytotox_5metal_multitask_v3.csv` (Ru/Ir/Os/Rh/Re), `wasserstein_weight: 0.0` (Wf-A37-7) |
| Chemprop property | [`molmetal/scripts/train_chemprop_v2_v3.py`](molmetal/scripts/train_chemprop_v2_v3.py) | `--metals` default = `Ru Ir Os Rh Re Pt` (Wf-A37-8 extended; `METALS_5_PLUS_PT`) |
| D-MPNN property | [`configs/dmpnn_within_metal_auc_cap512_d4_ens_gpu.yaml`](configs/dmpnn_within_metal_auc_cap512_d4_ens_gpu.yaml) | per-metal ensemble, cap 512, depth 4 |

Use `cfm_5metal_default.yaml` as the CFM entry-point and chemprop `--metals=auto`
(or `--metals=RU_IR_OS_RH_RE_PT`) for chemprop when the 7-metal cohort is not yet
available on disk.

## Documentation map

- [`docs/HONEST_NEGATIVE_CATALOG.md`](docs/HONEST_NEGATIVE_CATALOG.md) — MEASURED-vs-PROJECTED honesty ledger (111 entries, Wf-A30-4 close-out)
- [`docs/DESIGN_BOUNDARY.md`](docs/DESIGN_BOUNDARY.md) — FROZEN / STABLE / PROPOSED / OUT-OF-SCOPE surface
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 21-mechanism M1-M21 phase diagram
- [`docs/POCKET_ENCODER_PIPELINE.md`](docs/POCKET_ENCODER_PIPELINE.md) — Pocket encoder (c) pipeline: PocketGen + P2Rank + platiny_pocket_pool + PerLayerPocketInjector (Wf-A31/A32)
- [`metrics/`](metrics/) — single source of truth for MEASURED numbers (393 files, counted 2026-10-20)
- [`molmetal/reports/`](molmetal/reports/) — per-workflow JSON/MD reports

## Wave-versioning policy (2026-10-01)

All version history lives in `git log`. No per-wave file naming convention is in use going forward — every Wf ships a closeout in `TODO/completed/` referenced by git commit, not by chained wave-name prefixes.

## License

Tier-2 computational-only paper target (AGENTS.md §5.4). No wet-lab planning. Per-measure MEASURED-vs-PROJECTED label discipline per `party-twenty-spirit` §3.5.
