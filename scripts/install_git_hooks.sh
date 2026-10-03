#!/usr/bin/env bash
# install_git_hooks.sh — install Mol-Metal's version-controlled git hooks into this clone.
#
# WHY THIS EXISTS
#   .git/hooks/ is never tracked by git, so a fresh clone starts with NO commit
#   gate — even though every script the gate calls (molmetal/scripts/
#   pre_flight_check.py, scripts/_validate_metrics.py, molmetal/scripts/
#   check_no_stale_venv.sh) IS tracked. The gate that chains them lived only in
#   the maintainer's untracked .git/hooks/pre-commit. This installer reproduces
#   it from the tracked source in scripts/git_hooks/, making the commit gate
#   clone-survivable.
#
# USAGE (run once per clone, after `uv sync` / `uv pip install -e .`):
#
#   bash scripts/install_git_hooks.sh             # install / update hooks
#   bash scripts/install_git_hooks.sh --dry-run   # show what would happen
#   bash scripts/install_git_hooks.sh --force     # replace a foreign hook
#                                                 # (backed up first)
#
# BEHAVIOUR
#   * Idempotent: a second run sees identical content and changes nothing
#     (no rewrite, mtime preserved); exit 0.
#   * Safe: an existing hook that DIFFERS from the tracked source is never
#     silently clobbered — it is backed up to <hook>.backup.<UTC timestamp>
#     and the installer says so.
#   * Refuses (without --force) to replace a hook that looks managed by the
#     pre-commit framework (contains INSTALL_PYTHON / hook-impl markers) —
#     run `pre-commit install` for that clone instead, or pass --force.
#   * Framework-agnostic: needs no pre-commit package. If the framework is
#     installed LATER, `pre-commit install` migrates the hook this installer
#     placed to pre-commit.legacy and chains both (the framework's own
#     migration mode) — the two coexist.
#
# TRACKED SOURCE
#   scripts/git_hooks/<hook-name>. Currently: pre-commit, the 3-step gate
#   (pre-flight v3 --pipeline all / metric schema validation / single-venv
#   guard — AGENTS.md §1.4 + §5.9). Add new hooks by dropping files here.

set -euo pipefail

usage() {
    cat <<'EOF'
usage: bash scripts/install_git_hooks.sh [--dry-run] [--force]

Installs the version-controlled git hooks (scripts/git_hooks/*) into this
clone's hooks directory (git rev-parse --git-path hooks; honours
core.hooksPath). Run once per fresh clone.

  --dry-run   print what would be done, change nothing
  --force     replace a differing existing hook (it is backed up first);
              without this, a hook that looks pre-commit-framework-managed
              is refused rather than replaced
  -h, --help  this text
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/scripts/git_hooks"

DRY_RUN=0
FORCE=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        --force)   FORCE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "install_git_hooks.sh: unknown argument '${arg}' (try --help)" >&2; exit 2 ;;
    esac
done

if [ ! -d "${SRC_DIR}" ]; then
    echo "ERROR: tracked hook source directory missing: ${SRC_DIR}" >&2
    exit 1
fi

# Where THIS clone looks for hooks — honours core.hooksPath when set.
HOOKS_DIR="$(git -C "${REPO_ROOT}" rev-parse --git-path hooks)"
case "${HOOKS_DIR}" in
    /*) : ;;
    *)  HOOKS_DIR="${REPO_ROOT}/${HOOKS_DIR}" ;;
esac

installed=0
unchanged=0
backed_up=0
refused=0

shopt -s nullglob
for src in "${SRC_DIR}"/*; do
    [ -f "${src}" ] || continue
    name="$(basename "${src}")"
    dst="${HOOKS_DIR}/${name}"

    if [ -f "${dst}" ] && cmp -s "${src}" "${dst}"; then
        echo "OK: ${name} already installed and identical — no change"
        unchanged=$((unchanged + 1))
        continue
    fi

    if [ -f "${dst}" ]; then
        if grep -qE "INSTALL_PYTHON|hook-impl" "${dst}" && [ "${FORCE}" -ne 1 ]; then
            echo "REFUSING: ${dst} looks managed by the pre-commit framework." >&2
            echo "  Run 'pre-commit install' for this clone, or re-run with --force to replace it (a backup is taken)." >&2
            refused=$((refused + 1))
            continue
        fi
        if [ "${DRY_RUN}" -eq 1 ]; then
            echo "DRY-RUN: would back up ${dst} -> ${dst}.backup.<UTC-timestamp>"
        else
            backup="${dst}.backup.$(date -u +%Y%m%dT%H%M%SZ)"
            cp -p "${dst}" "${backup}"
            echo "NOTICE: existing ${name} differed from tracked source; backed up to ${backup}"
            backed_up=$((backed_up + 1))
        fi
    fi

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "DRY-RUN: would install ${src} -> ${dst} (chmod +x)"
    else
        mkdir -p "${HOOKS_DIR}"
        cp "${src}" "${dst}"
        chmod +x "${dst}"
        echo "INSTALLED: ${src} -> ${dst}"
        installed=$((installed + 1))
    fi
done

echo "install_git_hooks.sh: installed=${installed} unchanged=${unchanged} backed_up=${backed_up} refused=${refused}"
if [ "${refused}" -gt 0 ]; then
    exit 1
fi
exit 0
