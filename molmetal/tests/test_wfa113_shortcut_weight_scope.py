"""Wf-A113 (T33) — the shortcut-distillation term's weight, and the defect it hit.

Background
----------
``shortcut_distill_loss`` used to enter the objective with no coefficient at
all: ``+ shortcut_distill_loss``.  On c8 it measured 11.5779 / 10.9638 /
10.5906 against ``fm_term`` 11.5367 / 10.8346 / 10.3499, i.e. roughly half the
objective, and no config key could scale it.  Adding a weight is the easy part.

The hard part is where the weight was read.  ``train_one_epoch`` is a
module-level function; ``args`` is ``main()``-scoped.  The first attempt read
``getattr(args, "shortcut_distill_weight", 1.0)`` inside the term block, which
raised ``NameError`` on every batch.  The block's own ``except`` caught it,
warned once, and set the term to ``torch.zeros(())`` — so the term silently
stopped training while the only trace was a ``RuntimeWarning``.  MEASURED, on
the same 1-epoch probe the liveness guard uses:

    before:  shortcut_distill_loss  requires_grad=False  value=0.0
             RuntimeWarning: [cfm-fm] shortcut_distill fell back to 0:
                             NameError("name 'args' is not defined")
    after:   shortcut_distill_loss  requires_grad=True   value=0.000636186
             (bit-identical to the pre-weight HEAD value 0.000636186, so a
             weight of 1.0 is a true no-op on the objective)

This is the same shape as Wf-A30's ``wasserstein_pocket_term``, whose weight
was read via ``locals().get("batch_ctx")`` in a scope that never had it.  The
lesson is not "remember that args is out of scope"; it is that a loss term
wrapped in ``try/except → warn → zero`` degrades silently, so the scope must be
checked mechanically.  ``test_no_free_args_identifier`` does that for the whole
trainer, with a positive control proving the detector can fail.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


# --------------------------------------------------------------------------
# Scope analysis
# --------------------------------------------------------------------------


def _param_names(fn: ast.AST) -> set:
    a = fn.args
    names = {x.arg for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)}
    if a.vararg is not None:
        names.add(a.vararg.arg)
    if a.kwarg is not None:
        names.add(a.kwarg.arg)
    return names


def _body_stores(fn: ast.AST) -> set:
    """Names assigned anywhere in ``fn``'s own body (nested scopes excluded)."""
    names: set = set()
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            # A nested scope binds its own names; it does not bind ours.
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return names


def _binds_args(fn: ast.AST) -> bool:
    return "args" in _param_names(fn) or "args" in _body_stores(fn)


def free_args_loads(tree: ast.AST) -> list:
    """``[(lineno, enclosing_function), ...]`` for ``args`` used out of scope.

    "Out of scope" means: the nearest enclosing function neither takes ``args``
    as a parameter, nor assigns it in its own body, nor is nested inside a
    function that does.
    """
    hits: list = []

    def walk(node: ast.AST, stack: list) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, stack + [(child.name, _binds_args(child))])
                continue
            if (
                isinstance(child, ast.Name)
                and child.id == "args"
                and isinstance(child.ctx, ast.Load)
                and not any(bound for _, bound in stack)
            ):
                hits.append((child.lineno, stack[-1][0] if stack else "<module>"))
            walk(child, stack)

    walk(tree, [])
    return hits


def test_no_free_args_identifier_in_the_trainer() -> None:
    """``args`` must not be read where nothing bound it.

    A violation here is not a style problem: the read raises ``NameError``, and
    every loss block in this file catches that, warns once and substitutes
    ``torch.zeros(())``.  The term then sits inside the objective training
    nothing, which is the failure class ``test_cfm_loss_gradient_liveness``
    exists to catch — but that guard can only see it when the relevant flag is
    ON in its smoke, and a term whose flag defaults to off is invisible.
    """
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    hits = free_args_loads(tree)
    assert not hits, (
        "`args` is read outside any scope that binds it: "
        + ", ".join(f"line {ln} in {fn}()" for ln, fn in hits)
        + ". Thread the value through the function signature instead; the "
        "surrounding `except` will otherwise zero the term silently."
    )


def test_the_detector_can_fail() -> None:
    """Positive control — without this the test above could pass on a bug."""
    bad = ast.parse(
        "def f(x):\n"
        "    try:\n"
        "        return getattr(args, 'w', 1.0) * x\n"
        "    except Exception:\n"
        "        return 0.0\n"
    )
    assert free_args_loads(bad) == [(3, "f")]

    # ...and it must not fire on the shapes that are legitimately in scope.
    ok = ast.parse(
        "def g(args):\n"
        "    return args.w\n"
        "def h():\n"
        "    args = make()\n"
        "    def inner():\n"
        "        return args.w\n"
        "    return inner\n"
    )
    assert free_args_loads(ok) == []


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def _find_function(tree: ast.AST, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {TRAINER.name}")


def test_shortcut_weight_is_a_train_one_epoch_kwarg() -> None:
    """The term's consumer must receive the weight, not look it up."""
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    fn = _find_function(tree, "train_one_epoch")
    a = fn.args
    # train_one_epoch is keyword-only, so the parameter lives in kwonlyargs.
    defaults = {
        name.arg: default
        for name, default in zip(a.kwonlyargs, a.kw_defaults)
        if default is not None
    }
    defaults.update(
        dict(zip([x.arg for x in a.args[len(a.args) - len(a.defaults):]], a.defaults))
    )
    assert "shortcut_distill_weight" in defaults, (
        "train_one_epoch has no shortcut_distill_weight parameter — the term "
        "block would have to reach for `args` and hit the NameError above."
    )
    default = ast.literal_eval(defaults["shortcut_distill_weight"])
    assert default == 1.0, (
        f"default shortcut_distill_weight is {default!r}, expected 1.0: the "
        "term entered the objective unweighted before this change, so 1.0 is "
        "the only value that leaves archived runs reproducible."
    )


def test_main_threads_the_weight_from_args() -> None:
    """The call site must read the flag, not hardcode the default."""
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train_one_epoch"
    ]
    assert calls, "no train_one_epoch(...) call found"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        assert "shortcut_distill_weight" in kw, (
            f"the train_one_epoch(...) call at line {call.lineno} does not pass "
            "shortcut_distill_weight, so the YAML key would be inert."
        )
        # Expected shape: float(getattr(args, "shortcut_distill_weight", 1.0))
        value = kw["shortcut_distill_weight"]
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name), (
            f"line {call.lineno}: unexpected expression for the weight"
        )
        inner = value.args[0] if value.args else None
        assert (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "getattr"
            and isinstance(inner.args[0], ast.Name)
            and inner.args[0].id == "args"
            and ast.literal_eval(inner.args[1]) == "shortcut_distill_weight"
        ), (
            f"line {call.lineno}: the weight is not read from "
            "args.shortcut_distill_weight, so the YAML key reaches nothing."
        )


def test_the_term_is_multiplied_by_the_weight() -> None:
    """``shortcut_distill_loss`` must be ``weight * adapter(...)``.

    Reads the assignment rather than the value: a term that ignores the weight
    is exactly what this change exists to fix, and it would still pass every
    other test in the suite.
    """
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    fn = _find_function(tree, "train_one_epoch")
    assigns = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "shortcut_distill_loss" for t in node.targets)
    ]
    assert assigns, "no assignment to shortcut_distill_loss found"
    products = [
        node
        for node in assigns
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult)
    ]
    assert products, (
        "shortcut_distill_loss is assigned without a multiplication by its "
        "weight — it would re-enter the objective unweighted."
    )
    ok = False
    for node in products:
        for side in (node.value.left, node.value.right):
            for sub in ast.walk(side):
                if isinstance(sub, ast.Name) and sub.id == "shortcut_distill_weight":
                    ok = True
    assert ok, "the multiplication does not reference shortcut_distill_weight"
