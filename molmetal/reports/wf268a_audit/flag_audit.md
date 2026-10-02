# Wf-268A Audit: CFM Ablation Flag Verification

**Target file**: `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
**Spec reference**: TODO/completed/207_unified_pre_flight_checklist_v2.md §1.1.7
**Audit date**: 2026-09-28
**Auditor**: Wf-268A (read-only verification)

## Acceptance gates

| Gate | Flag | PASS/FAIL | Line(s) | Evidence |
|------|------|-----------|---------|----------|
| G1 | `--no-yuelbond` | **PASS** | 832-833 | `p.add_argument("--no-yuelbond", dest="use_yuelbond", action="store_false", help="Disable YuelBond (use the production decoder only).")` |
| G2 | `--no-wasserstein` | **PASS** | 836-838 | `p.add_argument("--no-wasserstein", dest="wasserstein_weight", action="store_const", const=0.0, help="Disable Wasserstein regularization.")` |
| G3 | `--no-closure-bounded` | **PASS** | 841-843 | `p.add_argument("--no-closure-bounded", dest="closure_bounded", action="store_false", help="Disable Catherine closure-bounded generation.")` |

## Detailed findings

### G1 — `--no-yuelbond` (PASS)

The flag is registered at line 832 and pairs with the affirmative `--use-yuelbond` at line 830 (default `True`). It uses `action="store_false"` with `dest="use_yuelbond"`, so passing `--no-yuelbond` sets `args.use_yuelbond = False`. Default remains `True`, preserving backward compatibility.

### G2 — `--no-wasserstein` (PASS)

The flag is registered at line 836 and pairs with `--wasserstein-weight 0.1` (line 834). It uses `action="store_const", const=0.0` to bind to the same `dest="wasserstein_weight"`, so `--no-wasserstein` is equivalent to `--wasserstein-weight 0`. The training loop guards on `wasserstein_weight > 0.0` (line 624), so a zero weight cleanly disables the W₂ branch.

### G3 — `--no-closure-bounded` (PASS)

The flag is registered at line 841 and pairs with `--closure-bounded` at line 839 (default `True`). It uses `action="store_false"` with `dest="closure_bounded"`, so passing `--no-closure-bounded` sets `args.closure_bounded = False`. Both `train_one_epoch` (line 567) and `evaluate_decode_ratio` (line 711) accept `closure_bounded: bool`, and the call sites at lines 1052 and 1064 thread `args.closure_bounded` through correctly. When disabled, `evaluate_decode_ratio` falls through to the plain `_decode_with_rdkit_strict` path (line 768).

## Verdict

All three ablation flags required by TODO-207 v2 §1.1.7 are present and wired through to their respective code paths.

**Overall: 3/3 PASS**

## Honest framing note

The flags are syntactic and semantic — argparse parses them and the values flow into `train_one_epoch` / `evaluate_decode_ratio` parameter lists. No behaviour has been measured (no benchmark re-run executed); this audit is a static read of the CLI surface and the function-signature wiring, per the Wf-268A read-only constraint.