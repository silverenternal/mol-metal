# H1 — HOLE 3 (honest hook staging) + HOLE 8 (gate-written tracked files) fix

**Status:** SHIPPED to working tree 2026-10-03. **No commit** (lane rule — a commit is in flight in another lane; the staged untrack entries below ride the next commit).
**Lane:** H1. Files touched: `.pre-commit-config.yaml` (HOLE 3), `.gitignore` (HOLE 8), this closeout.
**Did NOT touch:** `molmetal/scripts/pre_flight_check.py` (contested file, another running workflow — required follow-up documented in §4), `paper/*`, wet-lab items, or any HOLE-9 file.

## §0 Coordination note (config file contention, resolved by ordering)

At H1 start, `.pre-commit-config.yaml` was already modified and `scripts/git_hooks/` + `scripts/install_git_hooks.sh` appeared untracked — lane **H2** (HOLE 9) had written them seconds earlier (config mtime 10:27:38; H2 closeout `TODO/completed/h2_git_hook_installer.md` mtime 10:28:49, i.e. H2 closed out after its last write). H1 then applied 5 surgical `Edit`s on the merged content, never a whole-file `Write`; H2's `repo: local` no-stale-venv block and its sibling comment are intact (YAML map in §1). No write-write overlap occurred.

## §1 HOLE 3 — what changed (`.pre-commit-config.yaml`, +47 lines vs HEAD incl. H2's block)

1. NEW header comment block ("HONEST-STAGING NOTE"): states what IS enforced today (the hand-written gate chain only), what enforces once the framework is installed (5 cheap hygiene hooks + the local no-stale-venv guard), and what is declared-but-manual — so the file no longer implies enforcement it does not deliver.
2. `stages: [manual]` added to `ruff`, `ruff-format`, `mypy`, `mdformat`, `codespell` (they fail or are unmeasured on this tree today). Run explicitly via `pre-commit run --hook-stage manual --all-files`.

Measurement backing (MEASURED 2026-10-03):

```
$ /home/hugo/codes/mol-metal/.venv/bin/python -c "import pre_commit"
ModuleNotFoundError: No module named 'pre_commit'        # rc=1
$ grep -c -E "INSTALL_PYTHON|hook-impl" .git/hooks/pre-commit
0                                                        # rc=1 -> hand-written hook only, no framework markers
$ command -v ruff codespell mypy mdformat pre-commit
ruff: /usr/bin/ruff (0.16.8, NOT pinned v0.6.9)
codespell: ABSENT | mypy: ABSENT | pre-commit: ABSENT | mdformat: /home/hugo/.local/bin/mdformat
```

(Review-measured with system ruff 0.16.8: `ruff check molmetal/scripts` -> 18,369 findings; `ruff format --check` -> 540 files would be reformatted. PROJECTED for the pinned v0.6.9.)

Post-edit YAML validation (venv python + PyYAML, `yaml.safe_load` OK):

```
local                      no-stale-venv            default(commit)
pre-commit-hooks           trailing-whitespace      default(commit)
pre-commit-hooks           end-of-file-fixer        default(commit)
pre-commit-hooks           check-yaml               default(commit)
pre-commit-hooks           check-json               default(commit)
pre-commit-hooks           detect-private-key       default(commit)
ruff-pre-commit            ruff                     ['manual']
ruff-pre-commit            ruff-format              ['manual']
mirrors-mypy               mypy                     ['manual']
mdformat                   mdformat                 ['manual']
codespell                  codespell                ['manual']
repos=6 hooks=11
```

## §2 HOLE 8 — what changed (untrack + gitignore, 3 paths)

Choice: **untrack + gitignore** (not redirect), because the redirect requires editing `pre_flight_check.py` (default `out_dir` at ~:5051; the non-atomic JSONL write at ~:3371) — a contested file explicitly off-limits to this lane. Untracking is implementable now, non-destructive (AGENTS.md §5.5: files stay on disk; committed blobs remain in history), and immediately stops commits from recording transient gate writes.

```
$ git rm --cached -- molmetal/reports/wf350a3_hp_provenance.jsonl \
    molmetal/reports/wf385a3_scaffold_wire_in/train/events.jsonl \
    molmetal/reports/wf385a3_scaffold_wire_in/train/synthesis.md
rm 'molmetal/reports/wf350a3_hp_provenance.jsonl'
rm 'molmetal/reports/wf385a3_scaffold_wire_in/train/events.jsonl'
rm 'molmetal/reports/wf385a3_scaffold_wire_in/train/synthesis.md'
```

Files still on disk (§5.5 evidence, MEASURED):

```
-rw-r--r-- 5294 Oct  3 10:23 molmetal/reports/wf350a3_hp_provenance.jsonl
-rw-r--r-- 18323 Oct  3 10:23 molmetal/reports/wf385a3_scaffold_wire_in/train/events.jsonl
-rw-r--r-- 2656 Oct  3 10:23 molmetal/reports/wf385a3_scaffold_wire_in/train/synthesis.md
-rw-r--r-- 7100 Oct  3 10:29 .gitignore
-rw-r--r-- 3610 Oct  3 10:30 .pre-commit-config.yaml
```

`.gitignore` block added (lines 186-198): the 2 patterns + a reserved `results/gate_scratch/` ignore for the pending §4(b) redirect.

Ignore-rule proof:

```
$ git check-ignore -v <3 paths + synthetic probe>
.gitignore:195:molmetal/reports/wf350a3_hp_provenance.jsonl	...
.gitignore:196:molmetal/reports/wf385a3_scaffold_wire_in/train/	(all 3 + probe)
```

Synthetic-touch proof (appended one line to each of the 3 files, then byte-exact restore):

```
events.jsonl:   sha 1abed12203110c8f -> 1abed12203110c8f restored=YES
hp_provenance:  sha 5d92a9ea181f8411 -> 5d92a9ea181f8411 restored=YES
synthesis.md:   sha 26fabc1ed524c5e6 -> 26fabc1ed524c5e6 restored=YES
porcelain after each append: ONLY the one-time staged "D  <path>" — no " M", no "??".
```

Synthetic sibling `train/_h1_synthetic_probe`: `git status --porcelain` -> empty (ignored). Final `git status --porcelain`: the 3 paths no longer appear as modified/untracked; the three `D ` staged entries ARE the untrack and clear on the next commit (H1 was told not to commit). Staged stat: `3 files changed, 138 deletions(-)`.

## §3 What H1 deliberately did NOT do

1. **No pre-commit framework install** — USER DECISION (changes the user environment + first-run hook-env network fetch). Evidence in §1.
2. **No tree-wide ruff/ruff-format/mdformat cleanup** — requires the scoped, directory-limited cleanup first (review Stage 2/3).
3. **No commit** (lane rule). The staged `D` untracks ride the next commit from the commit-in-flight lane.
4. **No `pre_flight_check.py` edits** — required follow-up for its owner in §4.
5. **HOLE 9 untouched** — shipped by H2; H1 did not modify `scripts/git_hooks/`, `scripts/install_git_hooks.sh`, or `.github/workflows/ci.yml`.

## §4 REQUIRED FOLLOW-UP (for the pre_flight_check.py owner — not applied by H1)

- **(a) `molmetal/scripts/pre_flight_check.py` ~:3371** — the hp-provenance JSONL is rewritten via plain `open(jsonl_path, "w", ...)` (non-atomic; itself a §5.6 gap). Route through `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file` (or drop once (b) lands).
- **(b) `molmetal/scripts/pre_flight_check.py` ~:5051** — `out_dir: Path = MOLMETAL_ROOT / "reports" / "wf385a3_scaffold_wire_in"` -> point the default at the ignored scratch root `results/gate_scratch/wf385a3_scaffold_wire_in` so the trainer subprocess stops writing into the repo report tree. `.gitignore` already reserves that root (this change). The redirect is the durable end-state; the untrack here is the immediate mitigation.
- Line numbers are as of the 2026-10-03 review; the file was being edited concurrently — re-verify before applying.

## §5 Honest framing / consequences

- All 11 hooks remain **unenforced** until the framework is installed (USER DECISION open). This change enforces nothing new — it only stops the config from implying enforcement it does not deliver.
- The 3 paths will not exist in a fresh clone until the first gate run regenerates them. Tests checked: none hard-fail on absence — `molmetal/tests/test_wf350b3_repro_manifest.py` SKIPs when the JSONL is absent (lines 185-191, MEASURED); `test_wf350a3_hp_provenance.py` uses `tmp_path` only. Last committed blobs stay retrievable via `git show <old-HEAD>:<path>`.
- The `D ` staged entries are visible in `git status` until the next commit.
- Once §4(b) lands, consumers reading `molmetal/reports/wf385a3_scaffold_wire_in/train/*` must switch to the scratch root.

## §6 §5.6 compliance

This closeout was written via `molmetal/scripts/wf386a3_atomic_write.py::write_closeout` (atomic temp-file + rename, overwrite guard, post-write truncation check), invoked with the project venv python. End of closeout.
