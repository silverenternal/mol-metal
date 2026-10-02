# Mol-Metal — Quickstart

> **TL;DR.** Mol-Metal is a precious-metal anticancer drug design framework on top of Flow Matching + Structure-Based Drug Design + Lambda Calculus MCTS proof-search. Run pre-flight in <2 min, launch a benchmark in <1 min.

## Install

```bash
# Already installed as a uv-managed package — see pyproject.toml
uv pip install -e .              # editable install (mirrors OmegaFold setup.py:25-37)
```

## Console-script entry point

After `uv pip install -e .`, the `molmetal` command is on PATH:

```bash
molmetal --version                # print package version
molmetal preflight                # run pre-flight v3 gate suite (~75 s, CPU-only)
molmetal demo                     # run the MolFlow-Triton synthetic batch + EGNN + ODE smoke
```

## 5-min smoke

```bash
# 1) Pre-flight (CPU-only; <2 min wall)
python -m molmetal preflight

# 2) Dry-run a benchmark launcher (no GPU consumed)
bash scripts/launch_wf124_p3_g5.sh --dry-run

# 3) Configure + launch (USER_OK-gated GPU retrain per AGENTS.md §5.1)
USER_OK=1 bash scripts/launch_wf124_p3_g5.sh
```

## Documentation map

- `README.md` — top-level project status
- `TODO/README.md` — TODO directory status TOC + skill ecosystem
- `AGENTS.md` — agent-facing rules + skill catalog + standing directives (§5)
- `docs/HONEST_NEGATIVE_CATALOG.md` — per-metric MEASURED-vs-PROJECTED honesty ledger
- `docs/DESIGN_BOUNDARY.md` — FROZEN / STABLE / PROPOSED / OUT-OF-SCOPE surface
- `docs/ARCHITECTURE.md` — 21-mechanism M1-M21 phase diagram

## Wave-versioning policy

All version history lives in `git log`. No per-wave file naming convention is in use going forward — every Wf ships a closeout in `TODO/completed/` referenced by git commit.

## Closing notes

Tier-2 computational-only paper target (AGENTS.md §5.4). No wet-lab planning — every entry is in-silico. Per-measure MEASURED-vs-PROJECTED label discipline per `party-twenty-spirit` §3.5.
