# Mol-Metal

> **TL;DR.** Mol-Metal is a precious-metal anticancer drug design framework (Pt/Ru/Ir/Au/Pd) on top of Flow Matching + Structure-Based Drug Design + Lambda Calculus MCTS proof-search.

## Quickstart

See [`QUICKSTART.md`](QUICKSTART.md) — install + console-script entry point + 5-min smoke.

## Develop

Agent-facing entry point: [`AGENTS.md`](AGENTS.md) — project rules + skill catalog + standing directives. Project TOC + overview: [`TODO/README.md`](TODO/README.md). Single-source-of-truth per file status: [`TODO/INDEX.md`](TODO/INDEX.md).

## Documentation map

- [`docs/HONEST_NEGATIVE_CATALOG.md`](docs/HONEST_NEGATIVE_CATALOG.md) — MEASURED-vs-PROJECTED honesty ledger
- [`docs/DESIGN_BOUNDARY.md`](docs/DESIGN_BOUNDARY.md) — FROZEN / STABLE / PROPOSED / OUT-OF-SCOPE surface
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 21-mechanism M1-M21 phase diagram
- [`metrics/`](metrics/) — single source of truth for MEASURED numbers (113+ files)
- [`molmetal/reports/`](molmetal/reports/) — per-workflow JSON/MD reports

## Wave-versioning policy (2026-10-01)

All version history lives in `git log`. No per-wave file naming convention is in use going forward — every Wf ships a closeout in `TODO/completed/` referenced by git commit, not by chained wave-name prefixes.

## License

Tier-2 computational-only paper target (AGENTS.md §5.4). No wet-lab planning. Per-measure MEASURED-vs-PROJECTED label discipline per `party-twenty-spirit` §3.5.
