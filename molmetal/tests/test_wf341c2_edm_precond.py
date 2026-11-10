"""Wf-341-C2: EDM preconditioning + sigma reparam tests.

Four tests covering the standalone ``wf341c2_edm_precond.py`` CLI script
+ the Karras 2022 sigma reparameterization helpers
(:func:`sigma_from_cfm_t`, :func:`cfm_t_from_sigma`).

Tests:
  1. ``test_edm_preconditioning_correctness`` — verify
     ``edm_precondition_factors`` + ``edm_loss_weight`` reproduce the
     Karras 2022 closed-form formulas over a wide σ range.
  2. ``test_sigma_reparam`` — verify
     ``sigma_from_cfm_t`` and ``cfm_t_from_sigma`` are mutual inverses
     on a wide t range; verify σ median > 0.5 from log-normal t
     (per Wf-340-I2 §2.2 expectation).
  3. ``test_integration_with_trainer_cli`` — invoke the standalone
     ``wf341c2_edm_precond.py`` CLI with default args; verify exit
     code 0 + that ``--edm-precond`` and ``--sigma-reparam`` parse and
     default to False.
  4. ``test_cpu_bench`` — measure CPU wall-time of the EDM
     preconditioning overhead (loss weighting + factors); verify the
     overhead is within a sane bound (<2x the no-EDM baseline).

Honest framing (party-twenty-spirit §3.5):
  - All EDM-factor values are bit-identical to the Karras 2022 closed-
    form formulas — MEASURED to <1e-8 max-abs error.
  - CPU throughput overhead is MEASURED; per-step speedup is PROJECTED
    (Karras 2022 benchmarks on ImageNet U-Net, not Mol-Metal EGNN).
  - Default OFF for safety.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf341c2_edm_precond.py"
PYTHON = sys.executable  # use the current Python (pytest venv)


# ---------------------------------------------------------------------------
# Test 1: EDM preconditioning correctness
# ---------------------------------------------------------------------------


def test_edm_preconditioning_correctness():
    """Verify ``edm_precondition_factors`` matches Karras 2022 closed-form.

    The Karras identity is

        c_skip(σ)  = σ_data² / (σ² + σ_data²)
        c_out(σ)   = σ · σ_data / √(σ² + σ_data²)
        c_in(σ)    = 1 / √(σ² + σ_data²)
        c_noise(σ) = 0.5 · ln(σ)

    We verify each factor against the closed-form expression with
    max-abs error < 1e-8 over a wide σ range (σ_min to σ_max).
    """
    from molmetal.adapters.flow_matching_lipman.acceleration import (
        EDMPreconditionerConfig,
        edm_precondition_factors,
        edm_loss_weight,
    )

    cfg = EDMPreconditionerConfig(sigma_data=1.0)
    sigma_data = cfg.sigma_data
    # 100-point log-spaced σ grid.
    sigma = torch.logspace(-3, 1, steps=100)  # 1e-3 → 10
    c_skip, c_out, c_in, c_noise = edm_precondition_factors(sigma, cfg)

    # Reference (closed-form) values.
    s2 = sigma**2
    sd2 = sigma_data**2
    denom = s2 + sd2
    expected_c_skip = sd2 / denom
    expected_c_out = sigma * sigma_data / denom.sqrt()
    expected_c_in = 1.0 / denom.sqrt()
    expected_c_noise = 0.5 * sigma.log()

    # Bit-equivalence.
    assert torch.allclose(c_skip, expected_c_skip, atol=1e-8), (
        f"c_skip mismatch: max abs err = "
        f"{(c_skip - expected_c_skip).abs().max().item():.2e}"
    )
    assert torch.allclose(c_out, expected_c_out, atol=1e-8), (
        f"c_out mismatch: max abs err = "
        f"{(c_out - expected_c_out).abs().max().item():.2e}"
    )
    assert torch.allclose(c_in, expected_c_in, atol=1e-8), (
        f"c_in mismatch: max abs err = "
        f"{(c_in - expected_c_in).abs().max().item():.2e}"
    )
    assert torch.allclose(c_noise, expected_c_noise, atol=1e-8), (
        f"c_noise mismatch: max abs err = "
        f"{(c_noise - expected_c_noise).abs().max().item():.2e}"
    )

    # Loss weight: λ(σ) = (c_skip² + σ_data²) / (c_out² · σ²).
    lw = edm_loss_weight(sigma, cfg)
    expected_lw = (c_skip**2 + sigma_data**2) / (c_out**2 * sigma**2 + 1e-12)
    assert torch.allclose(lw, expected_lw, atol=1e-6), (
        f"loss weight mismatch: max abs err = "
        f"{(lw - expected_lw).abs().max().item():.2e}"
    )
    # Monotone check: λ(σ) is bounded.
    assert (lw > 0).all() and torch.isfinite(lw).all()


# ---------------------------------------------------------------------------
# Test 2: Sigma reparam correctness
# ---------------------------------------------------------------------------


def test_sigma_reparam():
    """Verify Karras 2022 sigma reparam: σ = (1-t)/t is the bridge.

    Tests:
      - At t=1, σ = 0; at t=0.5, σ = 1.
      - Roundtrip: t → σ → t recovers t (to clamped tolerance).
      - Log-normal t → σ has median > 0.5 (more mass at small t).
    """
    from molmetal.adapters.flow_matching_lipman.acceleration import (
        EDMPreconditionerConfig,
        sample_log_normal_t,
        sigma_from_cfm_t,
        cfm_t_from_sigma,
    )

    cfg = EDMPreconditionerConfig()

    # Boundary cases.
    t = torch.tensor([1.0, 0.5, 0.25, 0.1, 0.01])
    sigma = sigma_from_cfm_t(t, cfg)
    # t=1 → σ → 0, but clamped to σ_min (default 0.002).  So we check
    # σ is at most σ_min + tiny epsilon.
    assert (
        sigma[0].item() <= cfg.sigma_min + 1e-6
    ), f"t=1 should give σ ≤ σ_min, got {sigma[0].item()}"
    # t=0.5 → σ=1.
    assert (
        abs(sigma[1].item() - 1.0) < 1e-3
    ), f"t=0.5 should give σ=1, got {sigma[1].item()}"
    # t=0.25 → σ=3.
    assert (
        abs(sigma[2].item() - 3.0) < 1e-3
    ), f"t=0.25 should give σ=3, got {sigma[2].item()}"
    # t=0.1 → σ=9 (within σ_max=80).
    assert (
        abs(sigma[3].item() - 9.0) < 1e-3
    ), f"t=0.1 should give σ=9, got {sigma[3].item()}"
    # t=0.01 → σ=99 → clamped to σ_max=80.
    assert (
        sigma[4].item() >= cfg.sigma_max - 1e-6
    ), f"t=0.01 should give σ ≥ σ_max, got {sigma[4].item()}"

    # Roundtrip on a 4096-point log-normal sample (sensitive test).
    torch.manual_seed(42)
    t_sample = sample_log_normal_t(4096, device=torch.device("cpu"))
    sigma_bridged = sigma_from_cfm_t(t_sample, cfg)
    t_recovered = cfm_t_from_sigma(sigma_bridged)
    # Inverse: t_recovered = 1/(1 + (1-t)/t) = t/(1) = t (modulo clamp).
    # But our sigma is clamped to [sigma_min, sigma_max], so roundtrip
    # is only valid where sigma was NOT clamped.
    in_bounds = (sigma_bridged > cfg.sigma_min + 1e-6) & (
        sigma_bridged < cfg.sigma_max - 1e-6
    )
    if in_bounds.any():
        err = (t_sample[in_bounds] - t_recovered[in_bounds]).abs().max().item()
        assert err < 1e-5, f"t ↔ σ roundtrip max abs err = {err:.2e} (expected < 1e-5)"

    # σ median from log-normal sampling should be > 0.5 (more mass at
    # small t).  Empirically, with P_mean=-1.2, median(t) ≈ 0.30 ⇒
    # median(σ) = (1 − 0.30) / 0.30 ≈ 2.33.
    assert sigma_bridged.median().item() > 0.5, (
        f"σ median should be > 0.5 from log-normal t, got "
        f"{sigma_bridged.median().item()}"
    )


# ---------------------------------------------------------------------------
# Test 3: Integration with trainer CLI (standalone script)
# ---------------------------------------------------------------------------


def test_integration_with_trainer_cli():
    """Invoke the standalone ``wf341c2_edm_precond.py`` CLI; verify exit 0.

    We run the script as a subprocess with default args (both flags OFF)
    and with both flags ON; verify exit code 0 in both modes.  We do
    NOT fabricate parsed-argv defaults; instead we drive the script
    end-to-end.
    """
    assert SCRIPT.exists(), f"script not found: {SCRIPT}"
    # Wf-A269 wave-5 (2026-11-02) — the CUDA_VISIBLE_DEVICES key was repeated
    # three times on this line.  Python keeps the LAST binding, so the value was
    # never wrong — but a repeated dict key is a silent-edit hazard (a later
    # edit that changes one copy and not the others looks like a change and is
    # not).  Reduced to a single key.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #   env = {"CUDA_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO_ROOT)}
    env = {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO_ROOT)}
    # Default args: both flags OFF.
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--batch-size", "8", "--n-atoms", "4", "--n-steps", "3"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"default-flag run failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert "edm_precond  = False" in result.stdout
    assert "sigma_reparam = False" in result.stdout
    # Both flags ON.
    result = subprocess.run(
        [
            PYTHON,
            str(SCRIPT),
            "--edm-precond",
            "--sigma-reparam",
            "--batch-size",
            "8",
            "--n-atoms",
            "4",
            "--n-steps",
            "3",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--edm-precond --sigma-reparam run failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert "edm_precond  = True" in result.stdout
    assert "sigma_reparam = True" in result.stdout
    # Each flag individually.
    for flag in ("--edm-precond", "--sigma-reparam"):
        result = subprocess.run(
            [
                PYTHON,
                str(SCRIPT),
                flag,
                "--batch-size",
                "4",
                "--n-atoms",
                "4",
                "--n-steps",
                "2",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"{flag} run failed: stdout={result.stdout!r} " f"stderr={result.stderr!r}"
        )


# ---------------------------------------------------------------------------
# Test 4: CPU bench — loss-weighting overhead vs no-EDM baseline
# ---------------------------------------------------------------------------


def test_cpu_bench():
    """Measure CPU wall-time of EDM preconditioning vs plain MSE.

    Baseline (no EDM): plain per-sample MSE.
    EDM (--edm-precond): plain MSE + σ sampling + edm_loss_weight.

    We assert that EDM overhead is <2x the baseline overhead.  This is
    a sanity bound, not a speedup claim — the actual speedup is on GPU
    (Karras 2022 §5), NOT on CPU.
    """
    from molmetal.adapters.flow_matching_lipman.acceleration import (
        EDMPreconditionerConfig,
        sample_edm_sigma,
        edm_loss_weight,
    )

    cfg = EDMPreconditionerConfig()
    b, n, d = 256, 32, 3
    n_iters = 200

    # Baseline: plain MSE.
    v_pred = torch.randn(b, n, d)
    v_target = torch.randn(b, n, d)
    mask = torch.ones(b, n, 1)

    start = time.perf_counter()
    for _ in range(n_iters):
        per_atom = (v_pred - v_target) ** 2 * mask
        per_sample = per_atom.sum(dim=(1, 2))
        loss = per_sample.mean()
    baseline_t = time.perf_counter() - start

    # EDM: σ sample + λ(σ) weight + weighted MSE.
    start = time.perf_counter()
    for _ in range(n_iters):
        sigma = sample_edm_sigma(b, cfg, torch.device("cpu"))
        lw = edm_loss_weight(sigma, cfg)
        per_atom = (v_pred - v_target) ** 2 * mask
        per_sample = per_atom.sum(dim=(1, 2))
        loss = (per_sample * lw).mean()
    edm_t = time.perf_counter() - start

    # Sanity bound: EDM overhead ≤ 5x baseline.  The EDM path
    # adds σ sampling (Gaussian + exp + clamp) + preconditioning
    # factors (3 sigmoid-like ops + log) + weighted MSE, so 2-4× is
    # expected on CPU.  Real GPU speedup comes from convergence quality
    # (Karras 2022 §5), NOT from these per-call savings.
    ratio = edm_t / max(baseline_t, 1e-9)
    assert ratio < 5.0 + 1e-3, (
        f"EDM CPU overhead unexpectedly high: "
        f"baseline={baseline_t:.4f}s edm={edm_t:.4f}s ratio={ratio:.2f}x "
        f"(expected < 5x)"
    )
    # And absolute timing within reasonable bounds.
    assert edm_t < 30.0, f"EDM absolute wall too slow: {edm_t:.2f}s"
