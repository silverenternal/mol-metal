"""Wf-A269 wave-20 — every shipped config must be **LAUNCHABLE**.

THE DEFECT CLASS (a new one)
---------------------------
A config can satisfy every §5.11 gate and still be **un-launchable**, because
the gates check that a YAML key resolves to a dest — they do not check that
the trainer **ACCEPTS** the value it resolves to.

MEASURED 2026-11-04 (the instance that opened the class)
---------------------------------------------------------
``molmetal/scripts/train_cfm_yuelbond_wasserstein.py:16160-16173`` (Wf-A113
T70, 2026-10-28) raises ``SystemExit`` at start-up when a config asks for a
non-default bond-candidate trio:

    _bc_src = str(getattr(args, "bond_candidate_source", "truth_only"))
    _bc_cut = float(getattr(args, "bond_candidate_cutoff", 2.4))
    _bc_ratio = int(getattr(args, "subsample_neg_ratio", 0))
    if _bc_src != "truth_only" or abs(_bc_cut - 2.4) > 1e-9 or _bc_ratio != 0:
        raise SystemExit(...)

because this trainer's model is ``CFMJointModel`` and it never calls
``LipmanFlowMatchingAdapter.train_step``, where the trio was wired.

``configs/cfm_5metal_wfa69_smoke.yaml`` asked for exactly that
(``radius_graph`` / ``4``) ⇒ the config **could not launch the trainer at
all**, and it made **2 of the repo's OWN tests RED** — the §6.55g
bond-decision wire-in guard ``molmetal/tests/test_wfA50_bond_decision_yaml7.py``
— i.e. the guard that proves "``flow.bond_decision`` lands in
``args.bond_decision``" was **unverified for 7 days** (2026-10-28 → 2026-11-04).

⇒ the class is: **a config value that the trainer REFUSES, which no gate
sees, because the gates only resolve keys, they never ask the trainer.**

WHAT THIS TEST DOES
-------------------
Source-derived and CPU-only (no torch, no model, no GPU): the trainer's own
inert defaults are read off its AST — the same three ``getattr(args, ...,
<default>)`` calls the guard uses — and every shipped config that declares
the trio is required to declare it at those defaults.

* **T1** every shipped config's trio == the trainer's own defaults
  (the un-launchable outlier is gone);
* **T2** anti-vacuity — the trainer still REFUSES a non-default trio
  (if the ``raise SystemExit`` guard is deleted, this test fails, so the
  test cannot be satisfied by removing the protection).

MUTATION-PROVABLE: setting any shipped config's ``bond_candidate_source``
back to ``radius_graph`` makes **T1 FAIL**; deleting the trainer's guard
makes **T2 FAIL**.

APPEND-ONLY (§5.5 rule #6): the pre-repair rendering of
``configs/cfm_5metal_wfa69_smoke.yaml`` (``radius_graph`` / ``4``) is
preserved VERBATIM in that file's dated comments.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"

#: The trio, and the local variable each one is read into by the trainer's
#: own guard.  The DEFAULTS are NOT hard-coded here — they are read off the
#: trainer's AST in :func:`_trainer_defaults`.
TRIO = {
    "bond_candidate_source": "_bc_src",
    "bond_candidate_cutoff": "_bc_cut",
    "subsample_neg_ratio": "_bc_ratio",
}

CONFIG_ROOTS = (REPO / "configs", REPO / "molmetal" / "configs")


def _trainer_source() -> ast.Module:
    return ast.parse(TRAINER.read_text(encoding="utf-8"), filename=str(TRAINER))


def _trainer_defaults() -> dict[str, object]:
    """The trainer's own inert defaults for the trio, read off its AST.

    Read from the ``getattr(args, "<name>", <default>)`` calls the guard
    uses, so the test follows the trainer instead of restating it.
    """
    defaults: dict[str, object] = {}
    for node in ast.walk(_trainer_source()):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        local = node.targets[0].id
        if local not in TRIO.values():
            continue
        # The guard wraps the getattr in a coercion call —
        #     _bc_src = str(getattr(args, "bond_candidate_source", "truth_only"))
        # — so the getattr is NESTED inside the assignment's value, not the
        # value itself.  Find it by walking the value.
        for call in ast.walk(node.value):
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "getattr"
                    and len(call.args) == 3):
                continue
            key = call.args[1]
            default = call.args[2]
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if not isinstance(default, ast.Constant):
                continue
            defaults[key.value] = default.value
    return defaults


def _trainer_still_refuses() -> bool:
    """Does the trainer still ``raise SystemExit`` for a non-default trio?"""
    for node in ast.walk(_trainer_source()):
        if not isinstance(node, ast.If):
            continue
        text = ast.dump(node.test)
        if not all(local in text for local in TRIO.values()):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Raise)
                    and isinstance(sub.exc, ast.Call)
                    and isinstance(sub.exc.func, ast.Name)
                    and sub.exc.func.id == "SystemExit"):
                return True
    return False


def _configs_declaring_the_trio() -> dict[str, dict[str, object]]:
    """``{config path (repo-relative): {key: value}}``."""
    found: dict[str, dict[str, object]] = {}
    for root in CONFIG_ROOTS:
        if not root.is_dir():
            continue
        for cfg in sorted(root.rglob("*.yaml")):
            try:
                text = cfg.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "bond_candidate_source" not in text and "subsample_neg_ratio" not in text:
                continue
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError:
                continue
            if not isinstance(data, dict):
                continue
            flow = data.get("flow")
            if not isinstance(flow, dict):
                continue
            vals = {k: flow[k] for k in TRIO if k in flow}
            if vals:
                found[str(cfg.relative_to(REPO))] = vals
    return found


# --------------------------------------------------------------------------
# T1 — every shipped config declares the trio at the trainer's defaults
# --------------------------------------------------------------------------


def test_t1_no_shipped_config_asks_for_a_refused_trio() -> None:
    defaults = _trainer_defaults()
    assert defaults, ("the trainer's inert defaults could not be read off "
                      "its AST — the extraction is broken, which would make "
                      "this test vacuous")

    declared = _configs_declaring_the_trio()
    assert declared, ("no shipped config declares the trio — the extraction "
                      "is broken, which would make this test vacuous")

    offenders: dict[str, dict[str, object]] = {}
    for path, vals in declared.items():
        for key, value in vals.items():
            want = defaults.get(key)
            if want is None:
                continue
            if isinstance(want, float):
                bad = abs(float(value) - want) > 1e-9
            else:
                bad = value != want
            if bad:
                offenders.setdefault(path, {})[key] = value

    assert not offenders, (
        f"shipped configs ask for bond-candidate values the trainer REFUSES "
        f"at start-up (trainer guard: train_cfm_yuelbond_wasserstein.py"
        f":16160-16173): {offenders} (the trainer's inert defaults: "
        f"{defaults}). Such a config cannot launch the trainer at all, and "
        "no §5.11 gate sees it — the gates resolve keys, they never ask the "
        "trainer."
    )


# --------------------------------------------------------------------------
# T2 — anti-vacuity: the protection is still there
# --------------------------------------------------------------------------


def test_t2_trainer_still_refuses_a_non_default_trio() -> None:
    assert _trainer_still_refuses(), (
        "the trainer no longer refuses a non-default bond-candidate trio — "
        "the guard at train_cfm_yuelbond_wasserstein.py:16160-16173 was "
        "removed or weakened. T1 would then be satisfiable by deleting the "
        "protection, which is the trade this repo forbids."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
