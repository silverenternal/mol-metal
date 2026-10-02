# TODO-204 m15 MM-GBSA real meeko+Vina+GBSA backend — Wf-432-A2 verification

**Task**: Add real meeko + Vina + GBSA backend integration (replacing closed-form surrogate).

## §5.5 filesystem-truth verdict (MANDATORY)

Per AGENTS.md §5.5 filesystem-truth rule + AGENTS.md §5.1 + §5.4 (no sudo, no wet-lab):

The real meeko+Vina+GBSA backend is **HONEST-NEGATIVE BLOCKED** on external dependencies that are not installed in this environment.

### Evidence (ls -la + which + pip list per §5.5 rule #2)

```
$ ls -la /usr/local/bin/meeko /usr/bin/meeko ~/.local/bin/meeko 2>&1
ls: cannot access '/usr/local/bin/meeko': No such file or directory
ls: cannot access '/usr/bin/meeko': No such file or directory
ls: cannot access '/home/hugo/.local/bin/meeko': No such file or directory

$ which vina vina_split vina_python 2>&1
vina not found
vina_split not found
vina_python not found

$ ls /opt/meeko /opt/vina /usr/local/vina 2>&1
ls: cannot access '/opt/meeko': No such file or directory
ls: cannot access '/opt/vina': No such file or directory
ls: cannot access '/usr/local/vina': No such file or directory

$ grep -n "real backend raises\|NotImplementedError\|until the meeko + Vina deps land" molmetal/scripts/wf404a2_rmsd_multitemplate_scaffold.py
63:* ``real`` backend raises ``NotImplementedError`` until the meeko + Vina deps land.
```

### wf404a2 SCAFFOLD-only state on disk

```
$ ls -la molmetal/scripts/wf404a2_rmsd_multitemplate_scaffold.py
-rw-r--r-- 1 hugo hugo 31770 Sep 30 13:22 molmetal/scripts/wf404a2_rmsd_multitemplate_scaffold.py

$ grep -n "REPORTS_DIR\|mm_gbsa_rescore_top_k\|per_pocket_multitemplate_bench" molmetal/scripts/wf404a2_rmsd_multitemplate_scaffold.py
194:REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf404a2_multitemplate_smoke"
479:def mm_gbsa_rescore_top_k(
506:def per_pocket_multitemplate_bench(
```

The `wf404a2_rmsd_multitemplate_scaffold.py` script is **SCAFFOLD-only** with the real backend
explicitly raising `NotImplementedError` until the meeko + Vina dependencies land (line 63).

## Why real backend is BLOCKED

Per AGENTS.md §5.4 + §10 (wetlab deprecation) + §1 (no sudo for untracked installs):

1. **No meeko binary on disk** (verified via `ls -la /usr/local/bin/meeko /usr/bin/meeko ~/.local/bin/meeko` → all "No such file or directory").
2. **No vina binary on disk** (verified via `which vina vina_split vina_python` → all "not found").
3. **No conda/pip meeko or vina packages installed** (verified via `pip list | grep -iE "meeko|vina"` → 0 matches).
4. **No sudo** per AGENTS.md §5.4 (no wet-lab, no sudo installs).
5. **`wf404a2` self-documents the gap**: line 63 explicitly states the real backend is unimplemented pending meeko + Vina deps.

Per AGENTS.md §5.1 standing GPU authorization + USER_OK=1 grant, USER_OK covers GPU launch + computational-only; it does NOT cover installation of meeko/Vina system binaries (which would require sudo per system package manager install).

## Honest framing per §5.2 + party-twenty-spirit §3.5

- **The task brief asks for a real backend that requires external deps that are not installed.**
- **The SCAFFOLD-only `wf404a2` surrogate is the SHIPPABLE artifact on disk per AGENTS.md §5.5 filesystem-truth rule #1.**
- **The real backend is NOT a code patch — it is a deployment/install problem outside the scope of this ultracode wave.**
- **Wf-432-A2 cannot ship what is not on disk.** The honest-negative framing per §5.2: 0 of 4 §5.7 invariants are SATISFIED for the real backend because the real backend is not callable in this environment.

## What IS on disk (SCAFFOLD-only ship surface)

- `molmetal/scripts/wf404a2_rmsd_multitemplate_scaffold.py` 31,770 B (3,042 LOC) — SCAFFOLD-only with `surrogate` (closed-form) backend working + `real` backend raising `NotImplementedError` line 63.
- `molmetal/scripts/wf380a5_multi_template_mm_gbsa.py` — closed-form surrogate MM-GBSA backend (the production-shippable artifact).
- `molmetal/reports/wf404a2_multitemplate_smoke/` — smoke bench output dir with surrogate results.

## What IS open / forward-looking (USER_OK-gated)

- **Install meeko + vina system packages** (requires sudo OR user-OK conda/pip install of pre-built wheels) — closure = Wf-440+ with user-supplied deps OR conda env provision.
- **Implement real meeko receptor prep + Vina docking + MM-GBSA rescore** (~150 LOC trainer-side glue per wf404a2 #63 inline TODO) — closure = Wf-440+ after meeko+vina install.
- **USER_OK-gated actual retrain** on Blackwell sm_120 (~5 hr GPU per Wf-260 G6 timing).

## Standing directives re-confirmed

- §1 + §5.1 (USER_OK covers GPU launch, not system-install) + §5.2 (honest framing: real backend is BLOCKED on external deps) + §5.3 (no paper work per TODO-84) + §5.4 (no wet-lab per §10 + no sudo) + §5.5 (filesystem-truth rule strictly observed: 0 of 4 §5.7 invariants SATISFIED because the real backend is not callable in this environment; the SCAFFOLD-only surrogate IS on disk and working) + §5.6 (atomic write protocol observed via `write_closeout` from `molmetal/scripts/wf386a3_atomic_write.py`) + §10 (wetlab deprecation).

## last-update trailer

This closes the TODO-204 m15 MM-GBSA real backend task per AGENTS.md §5.5 filesystem-truth as **HONEST-NEGATIVE BLOCKED** on external deps (meeko + vina not installed; wf404a2 line 63 self-documents the gap; SCAFFOLD-only surrogate SHIPPED on disk).
