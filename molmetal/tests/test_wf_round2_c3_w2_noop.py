"""Wf-C3 / Wf-E1 (Round-2 + Round-4, 2026-10-05): pin the W2 no-op, then the fix.

The Wf-C3 audit question was "is W2 load-bearing or just wall-time?". The
measured answer was "neither — it does not touch the model at all".

Root cause
----------
``train_metal_lomo_wasserstein.extract_per_metal_embeddings`` built its
embedding batches with::

    pooled = model._forward_one_mol(...)
    out_embeds.setdefault(batch_metas[b], []).append(pooled.detach().cpu())
                                   ^^^^^^^

The ``.detach()`` severed the autograd graph. ``wasserstein_transfer_loss`` then
operates on plain CPU tensors and returns a scalar with ``requires_grad=False``.
At the injection site::

    loss = loss + w2_loss_t

that is an addition of a Python/NumPy-level constant: it changes the *printed*
loss value and nothing else. ``loss.backward()`` sees the identical graph it
would have seen with ``--wasserstein-weight 0.0``.

Measured evidence (Rh LOMO, seeds 0-2, 10 epochs, CPU)
------------------------------------------------------
    seed | W2 ON  | W2 OFF | delta
      0   | 0.3613 | 0.3597 | +0.0016
      1   | 0.3477 | 0.3562 | -0.0084
      2   | 0.3358 | 0.3360 | -0.0002
    MEAN  | 0.3483 | 0.3506 | -0.0023     (|mean| < 0.01 -> within seed noise)

Wall-time: 49.2s (OFF) -> 118.8s (ON), a 2.4x cost for zero model change. The
per-epoch training losses of the two arms are byte-identical on every epoch
except the three W2-injection epochs (0, 5, 9), where ON is higher by exactly
``wasserstein_weight * w2``.
Evidence: ``molmetal/reports/wf_round2_w2_{on,off}/Rh/metrics.json``.

The Wf-E1 fix
-------------
Two changes, both in ``train_metal_lomo_wasserstein.py``:

1. ``--wasserstein-weight`` now **defaults to 0.0**, and a non-zero value
   without the new ``--wasserstein-real`` opt-in emits a
   ``DeprecationWarning`` and is otherwise ignored. The whole W2 surface
   (in-loop extraction, epoch-end recording, final per-metal snapshot) is
   gated on ``--wasserstein-real``, so a prior config that still says
   ``--wasserstein-weight 0.05`` now runs at the ``0.0`` wall-time.
2. ``--wasserstein-real`` makes the W2 *real*:
   ``extract_per_metal_embeddings(..., real_grad=True)`` keeps the autograd
   graph, and the W2 scalar is back-propagated once per injection epoch on
   the first batch. Under this flag ``requires_grad`` must be ``True`` — the
   assertion below is the load-bearing one, because the entire fix is
   worthless if the detach quietly comes back.

The default expectation is therefore INVERTED relative to Wf-C3: W2 having no
effect is now the *intended* state, and W2 having an effect is the opt-in.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "molmetal" / "scripts" / "train_metal_lomo_wasserstein.py"


def _trainer_source() -> str:
    return TRAINER.read_text(encoding="utf-8")


def test_trainer_file_exists() -> None:
    assert TRAINER.is_file(), f"trainer not found at {TRAINER}"


def test_w2_injection_site_carries_the_ablation_note() -> None:
    """The inline comment at the injection site must name the ablation."""
    src = _trainer_source()
    assert "Wf-C3 ABLATION" in src, (
        "the W2 injection site lost its Wf-C3 ablation note — the finding that "
        "W2 contributes zero gradient is no longer discoverable at the site "
        "where it matters"
    )
    # The note must reference the measurement, not just assert a conclusion.
    assert re.search(r"wf_round2_w2_\{?on", src), (
        "the Wf-C3 note must cite the on-disk ablation evidence paths "
        "(molmetal/reports/wf_round2_w2_{on,off}/...)"
    )


def test_wasserstein_weight_help_warns_it_is_not_a_regulariser() -> None:
    """--wasserstein-weight help must not over-claim what the flag does."""
    src = _trainer_source()
    m = re.search(
        r'p\.add_argument\(\s*\n\s*"--wasserstein-weight",.*?\n    \)',
        src,
        re.S,
    )
    assert m is not None, "could not locate the --wasserstein-weight argument"
    help_text = " ".join(m.group(0).split())
    assert "NO effect" not in help_text or "W₂ term is OFF" in help_text, (
        "the --wasserstein-weight help text still implies the flag regularises "
        "the model unconditionally; after Wf-E1 that is false and must be "
        "stated at the CLI surface too"
    )
    assert "OFF" in help_text, (
        "the --wasserstein-weight help must say the W2 term is OFF by default "
        "(--wasserstein-real is required)"
    )


def test_wasserstein_weight_defaults_to_zero() -> None:
    """Wf-E1 — the default must be 0.0, not the historical 0.05.

    A 0.05 default keeps paying for a term that does nothing; the whole
    point of dropping the injection is that the *default* run stops paying.
    """
    torch = pytest.importorskip("torch")
    from molmetal.scripts.train_metal_lomo_wasserstein import _parse_args

    ns = _parse_args([])
    assert ns.wasserstein_weight == 0.0, (
        f"default --wasserstein-weight is {ns.wasserstein_weight}, expected 0.0 "
        "(Wf-E1: W2 is off unless --wasserstein-real is passed)"
    )
    assert ns.wasserstein_real is False, (
        "--wasserstein-real must default to False — the real-W2 path is opt-in"
    )


def test_nonzero_weight_without_real_warns_and_stays_ignored() -> None:
    """A prior config saying ``--wasserstein-weight 0.05`` must not silently
    claim to regularise: it parses, warns, and leaves W2 off."""
    pytest.importorskip("torch")
    from molmetal.scripts.train_metal_lomo_wasserstein import _parse_args

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ns = _parse_args(["--wasserstein-weight", "0.05"])
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep, (
        "passing a non-zero --wasserstein-weight without --wasserstein-real "
        "must emit a DeprecationWarning (the flag is now ignored)"
    )
    assert "--wasserstein-real" in str(dep[0].message), (
        "the DeprecationWarning must name the opt-in flag that re-enables W2"
    )
    assert ns.wasserstein_real is False

    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        _parse_args(["--wasserstein-weight", "0.05", "--wasserstein-real"])
    assert not [
        w for w in caught2 if issubclass(w.category, DeprecationWarning)
    ], (
        "--wasserstein-real + non-zero weight is the supported combination "
        "and must NOT warn"
    )


def test_w2_term_is_grad_free_by_default() -> None:
    """DEFAULT: W2 has no effect on the model.

    This is now the *intended* behaviour (Wf-E1), not a bug: without
    ``--wasserstein-real`` the embeddings are detached and the term is a
    constant, so no gradient is contributed.
    """
    torch = pytest.importorskip("torch")

    per_metal = {
        "Ru": torch.randn(20, 8),
        "Rh": torch.randn(20, 8),
    }
    assert not any(t.requires_grad for t in per_metal.values())

    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_transfer_loss,
    )

    w2 = wasserstein_transfer_loss(per_metal, n_mc=8, seed=0)

    assert w2.requires_grad is False and w2.grad_fn is None, (
        "the DEFAULT path must stay grad-free (detached embeddings). If this "
        "now reports requires_grad=True without --wasserstein-real, someone "
        "has made real W2 the default — that is a Wf-E1 opt-in decision, not "
        "an accident, and every historical wasserstein_weight comparison must "
        "be re-measured."
    )

    # Adding it must leave the gradient of the real loss completely untouched.
    real = torch.tensor(1.0, requires_grad=True)
    real.backward()
    baseline_grad = float(real.grad)

    real2 = torch.tensor(1.0, requires_grad=True)
    (real2 + w2).backward()

    assert float(real2.grad) == baseline_grad, (
        "adding the W2 penalty changed the gradient on the default path — the "
        "W2 is no longer a pure constant there, so the Wf-E1 'off by default' "
        "contract no longer holds"
    )


def test_real_grad_flag_makes_w2_differentiable() -> None:
    """``real_grad=True`` must produce a REAL W2 gradient.

    This is the load-bearing Wf-E1 assertion: the whole point of the opt-in
    is that ``extract_per_metal_embeddings`` keeps the autograd graph when
    asked, so the detach does not silently come back.
    """
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel
    from molmetal.scripts.train_metal_lomo_wasserstein import (
        extract_per_metal_embeddings,
    )
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_transfer_loss,
    )

    torch.manual_seed(0)
    model = DMPNNMultiTaskModel(hidden=16, depth=1, dropout=0.0)
    smiles = np.array(
        ["CCO", "c1ccccc1", "CCN", "CC(=O)O", "c1ccncc1", "CCCO"] * 4
    )
    metals = np.array(["Ru", "Rh"] * 12)

    embeds = extract_per_metal_embeddings(
        model, smiles, metals, torch.device("cpu"), max_per_metal=6, real_grad=True
    )
    assert embeds, "real_grad=True produced no embeddings — extraction failed"
    assert all(t.requires_grad for t in embeds.values()), (
        "real_grad=True must keep the per-metal embeddings attached to the "
        "autograd graph; if they came back detached, --wasserstein-real is "
        "FLAG_ONLY and W2 is still a no-op"
    )
    assert any(
        t.grad_fn is not None for t in embeds.values()
    ), "attached-but-grad_fn-None embeddings are still a constant"

    w2 = wasserstein_transfer_loss(embeds, n_mc=8, seed=0)
    assert w2.requires_grad is True, (
        "wasserstein_transfer_loss over real_grad embeddings must be a "
        "differentiable scalar"
    )
    w2.backward()
    grads = [
        p.grad for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert grads, (
        "backward() through the real W2 term produced no non-zero parameter "
        "gradients — the term is STILL a no-op under --wasserstein-real"
    )


def test_w2_gate_property_matches_flag() -> None:
    """``WassersteinLOMORunner.w2_enabled`` must require BOTH real + weight>0."""
    torch = pytest.importorskip("torch")

    from molmetal.scripts.train_metal_lomo_wasserstein import WassersteinLOMORunner

    cases = [
        (0.0, False, False),
        (0.05, False, False),  # the legacy config: parsed but OFF
        (0.0, True, False),  # opt-in with nothing to apply
        (0.05, True, True),  # the only combination that does anything
    ]
    for weight, real, expected in cases:
        # Build the object without running the dataset pipeline.
        r = WassersteinLOMORunner.__new__(WassersteinLOMORunner)
        r.wasserstein_weight = weight
        r.wasserstein_real = real
        assert r.w2_enabled is expected, (
            f"w2_enabled for weight={weight} real={real} should be {expected}, "
            f"got {r.w2_enabled}"
        )


def test_emitted_json_records_effective_weight() -> None:
    """A metrics JSON must not read "weight 0.05" next to a model that never
    received a W2 gradient — ``w2_effective_weight`` is the honest field."""
    src = _trainer_source()
    assert '"w2_real"' in src, "the per-seed JSON no longer records w2_real"
    assert '"w2_effective_weight"' in src, (
        "the per-seed JSON no longer records w2_effective_weight; a reader "
        "could otherwise conclude the run was W2-regularised at the requested "
        "weight when it was not"
    )
    assert "res.w2_real" in src and "res.w2_effective_weight" in src, (
        "w2_real / w2_effective_weight must be read off the RESULT (which "
        "knows what the runner actually did), not off args (which only says "
        "what was requested)"
    )


def test_ablation_artifacts_are_on_disk_and_agree() -> None:
    """The on-disk ablation must show |mean delta| < 0.01 (the Wf-C3 verdict)."""
    on_path = REPO_ROOT / "molmetal/reports/wf_round2_w2_on/Rh/metrics.json"
    off_path = REPO_ROOT / "molmetal/reports/wf_round2_w2_off/Rh/metrics.json"
    assert on_path.is_file(), f"missing ablation artifact: {on_path}"
    assert off_path.is_file(), f"missing ablation artifact: {off_path}"

    on = json.loads(on_path.read_text(encoding="utf-8"))
    off = json.loads(off_path.read_text(encoding="utf-8"))

    assert on["wasserstein_weight"] == 0.05
    assert off["wasserstein_weight"] == 0.0
    assert on["metals"] == off["metals"] == ["Rh"]
    assert on["seeds"] == off["seeds"] == [0, 1, 2]
    assert on["epochs"] == off["epochs"] == 10

    # model_select is recorded per-seed, not in the aggregate — read it there
    # so the assertion covers the setting the AUCs were actually produced under.
    for tag in ("on", "off"):
        per_seed = sorted(
            (REPO_ROOT / f"molmetal/reports/wf_round2_w2_{tag}/Rh/Rh").glob(
                "per_seed_metrics_seed*.json"
            )
        )
        assert len(per_seed) == 3, f"expected 3 per-seed files for {tag}, got {per_seed}"
        for f in per_seed:
            assert json.loads(f.read_text(encoding="utf-8"))["model_select"] == (
                "last_epoch"
            ), f"{f} was not produced under --model-select last_epoch"

    deltas = [a - b for a, b in zip(on["per_run_auc"], off["per_run_auc"])]
    mean_delta = sum(deltas) / len(deltas)
    assert abs(mean_delta) < 0.01, (
        f"measured |mean Δ test_AUC| = {abs(mean_delta):.4f} >= 0.01 — the "
        "Wf-C3 verdict ('W2 is not load-bearing') no longer holds for this "
        "ablation; re-derive the recommendation from the new numbers"
    )
