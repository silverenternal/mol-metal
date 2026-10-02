# Wf-284A — argparse %u crash fix in train_dmpnn_multitask_wasserstein.py

**Date:** 2026-09-29
**Status:** ALREADY FIXED (no edit required; verification pass only)

## 1. Investigation

`grep -n '%' molmetal/scripts/train_dmpnn_multitask_wasserstein.py` returned only
two literal `%` characters in the file:

| Line | Context                                                     | Status                                 |
| ---- | ----------------------------------------------------------- | -------------------------------------- |
| 370  | docstring inside `_JointGraphDataset` ("GPU 15-30% util.")   | safe — Python `argparse` does NOT parse  |
|      |                                                             | docstrings for `%u`/format-spec.         |
| 901  | `if (epoch + 1) % WASSERSTEIN_LOG_EVERY == 0 ...`            | safe — valid Python modulo operator.   |

**No literal `%` character exists inside any `argparse.ArgumentParser` help
text.** Line 1190 (the alleged crash site) already reads
`"at 15-30 percent util on RTX PRO 6000)."`. The fix described in the Wf-282C
finding — replace `%` with `percent` — has already been applied to the source.

## 2. Verification

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_dmpnn_multitask_wasserstein.py --help 2>&1 \
    | grep -A 1 "cpu-num-workers" | head -5

                        DataLoader num_workers for parallel batch collate
                        (default 4 = forked workers; set 0 for main-process
                        manual loop, slower). Pre-Wf-260 default was 0 (157
                        §1.2 honest gap: --cpu-num-workers flag existed but
                        was never wired to a DataLoader, leaving GPU at 15-30
                        percent util on RTX PRO 6000).

$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_dmpnn_multitask_wasserstein.py --help >/dev/null 2>&1
$ echo $?
0
```

- Help text rendered correctly with literal word `percent`, not `%`.
- `--help` exits with status `0` — no `TypeError: not all arguments converted
  during string formatting` crash.
- `argparse`-`%`-scan verdict: PASS (no unescaped `%` in any help= string).

## 3. Conclusion

Wf-282C's underlying bug — argparse interpreting `%` in `--cpu-num-workers`
help text as a `%u` format spec — was already remediated at line 1190 before
this Wf-284A pass ran. No edit was needed; the verification command proves
the fix is live. Wf-284A closes as a **no-op confirmation**.