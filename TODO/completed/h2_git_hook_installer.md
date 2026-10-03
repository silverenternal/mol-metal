# H2 — Clone-survivable git-hook installer (hole 9)

**Status:** SHIPPED to working tree 2026-10-03 (no commit — a commit is in flight in another lane per lane rules; see "Commit state" below).
**Lane:** H2. Scope: `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, NEW installer under `scripts/`, this closeout. Did NOT touch `molmetal/scripts/pre_flight_check.py` (contested file, other lane) or `paper/*`.

## Problem (one line)

`molmetal/scripts/check_no_stale_venv.sh` (and the whole 3-step commit-gate chain that calls it) was wired only into `.git/hooks/pre-commit` — and `.git/hooks/` is never tracked by git, so a fresh clone received every gate script and executed none of them.

## What shipped

| File | Change | What it does |
|---|---|---|
| `scripts/git_hooks/pre-commit` | NEW (3985 B, +x) | Byte-identical tracked copy of the hand-written commit gate (3-step chain: pre-flight v3 `--pipeline all` / `_validate_metrics.py` / `check_no_stale_venv.sh`; bypasses SKIP_PREFLIGHT / SKIP_METRICS / SKIP_VENV_GUARD). sha256 `1cd1f326082fac2df28b2ed97e5f532ea9ea8c8ef090ddbf350ca232d04cb747`. |
| `scripts/install_git_hooks.sh` | NEW (5000 B, +x) | Idempotent, safe installer: copies `scripts/git_hooks/*` into the clone's hook dir (`git rev-parse --git-path hooks`, honours `core.hooksPath`), chmod +x. |
| `.pre-commit-config.yaml` | MODIFIED | Added `repo: local` entry `no-stale-venv` (`bash molmetal/scripts/check_no_stale_venv.sh`, `language: system`, `always_run`) so the framework path also covers the §5.9 guard. Inert until the framework is installed. |
| `.github/workflows/ci.yml` | MODIFIED | NEW job `hook-install-gate` (ubuntu-latest, 5 min): asserts fresh checkout has no hook, installs from tracked source, asserts executable + `cmp`-identical, asserts second run is a no-op, runs the venv guard directly. Deliberately does NOT run `--pipeline all`. Added to `ci-summary` needs + loop. |

### Installer contract (MEASURED, see proof below)

- **Idempotent** — identical destination means no write at all (content and mtime preserved); second run prints "already installed and identical", `installed=0 unchanged=1`.
- **Non-clobbering** — a differing non-framework hook is backed up to `<hook>.backup.<UTCstamp>` with a NOTICE before replacement.
- **Framework-safe** — a destination containing `INSTALL_PYTHON`/`hook-impl` markers (pre-commit-framework-managed) is REFUSED with exit 1 unless `--force`; a later `pre-commit install` still migrates our hook to `.legacy` and chains both.
- **Framework-agnostic** — needs no pre-commit package (plain copy). Works whether or not the framework is installed.
- Flags: `--dry-run` (change nothing), `--force`, `--help`.

## Proof (§5.5 — ls -la evidence)

Real-repo install is a no-op in effect; live hook byte-identical pre/post (sha unchanged, mtime 10:10 untouched):

```
$ sha256sum .git/hooks/pre-commit            # BEFORE (snapshot) and AFTER: identical
1cd1f326082fac2df28b2ed97e5f532ea9ea8c8ef090ddbf350ca232d04cb747
$ bash scripts/install_git_hooks.sh
OK: pre-commit already installed and identical — no change
install_git_hooks.sh: installed=0 unchanged=1 backed_up=0 refused=0   # exit 0
$ cmp /tmp/h2_hook_before.copy .git/hooks/pre-commit && diff -u ...   # empty; zero behavioral change
$ ls -la scripts/install_git_hooks.sh scripts/git_hooks/pre-commit .git/hooks/pre-commit
-rwxr-xr-x 1 hugo hugo 5000 Oct  3 10:27 scripts/install_git_hooks.sh
-rwxr-xr-x 1 hugo hugo 3985 Oct  3 10:27 scripts/git_hooks/pre-commit
-rwxr-xr-x 1 hugo hugo 3985 Oct  3 10:10 .git/hooks/pre-commit
```

Scratch-repo batteries (all in /tmp; scripts `/tmp/h2_test_hooks.sh` + `/tmp/h2_test_behavior.sh`):

- 21/21 PASS — fresh-clone install (installed sha == live-hook sha), idempotency (byte + mtime), foreign-hook backup, framework refusal + `--force`, `--dry-run`, `core.hooksPath`.
- 12/12 PASS — the INSTALLED hook executed end-to-end with stub dependencies in a scratch repo: all 3 steps invoked, `ALL CHECKS PASS` on success, exit 1 + "single-venv guard FAILED" on guard failure, `SKIP_VENV_GUARD=1` bypass honoured.

YAML: both edited files parse via PyYAML (`yaml.safe_load`); job list confirmed.

## What a fresh clone must run

```
uv sync                              # (or `uv pip install -e .`)
bash scripts/install_git_hooks.sh    # once per clone; idempotent, safe to re-run
```

Setup-docs note (per task): `QUICKSTART.md` §Install should gain that one line after the `uv pip install -e .` line (line 9). **Not edited by this lane** (file not owned) — flagged for the setup-docs lane. `AGENTS.md` NOT edited (kept small per task + memory).

## Open items / honest framing

- CI job `hook-install-gate` is YAML-validated and its command sequence was rehearsed on the host, but NOT yet executed by a live CI runner — **PROJECTED** until its first run.
- The `repo: local` entry is inert until `pre-commit` is installed in the venv — **PROJECTED** path.
- **USER DECISION still open** (from review): installing the `pre-commit` package into `.venv` + first-run hook-env network fetch. Not done here; ruff/ruff-format/mypy/mdformat staged enforcement (review Stage 2/3) not touched.
- HOLE 8 (commit-path gate mutating tracked files; `pre_flight_check.py` line 3371 non-atomic write) remains owned by the other lane; `pre_flight_check.py` untouched here (verified `git status` clean for it).
- Commit state: new files are **untracked working-tree files** (`?? scripts/git_hooks/`, `?? scripts/install_git_hooks.sh`) until the next commit includes them; no `git add`/commit performed per lane rule ("Do NOT commit"). They are version-controlled from that commit onward.
