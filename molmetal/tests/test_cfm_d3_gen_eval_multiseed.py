"""Round-3 D3 — the generation eval must be big enough and split enough to
be read as a number.

Why this file exists.  ``gen_validity`` at n=16 was measured at 0.375 /
0.312 / 0.562 / 0.688 / 0.750 over five CONSECUTIVE epochs of a single run
(``results/cfm_tmqm_all6``).  That is a 44pp swing on a monotone trajectory.
A 16-sample binomial at p~0.5 has a 1-sigma band of +-12.5pp, so nearly all
of the swing is the draw, not the model — and the "best epoch" the run
selected on it was selected on noise.  D3 makes the eval 3 seeds x n=64 over
DISJOINT composition windows (192 distinct chemistries) and records the
per-seed split, so a pooled number is never the only thing on offer.

Three properties are pinned here, each of which is a way this could silently
regress:

1. **A decode-only run stays decode-only.**  The obvious implementation —
   give ``--gen-eval-n`` a hard default of 64 — switches the sampler on for
   every run that never asked for it, which changes both the wall time and
   what a run reports.  The default is ``None``, and ``None`` means "inherit
   ``--gen-validity-samples``", so an unset flag leaves the switch alone.
2. **The seeds must cover DIFFERENT compositions.**  Seeding only re-draws
   the noise; three seeds over the same first-64 templates would be three
   measurements of one narrow slice.  ``template_offset`` is what makes the
   windows disjoint, and the test asserts disjointness.
3. **The chemistry summary pools rows rather than averaging averages.**
   Averaging per-seed summaries weights a seed that produced no parseable
   SMILES exactly as heavily as a full one, and divides a ratio of ratios.
   The test catches that with a deliberately lopsided seed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load(
        PROJECT_ROOT
        / "molmetal"
        / "scripts"
        / "train_cfm_yuelbond_wasserstein.py",
        "_t_cfm_d3_gen_eval",
    )


def _resolve_n(args) -> int:
    """Mirror of the resolution the eval block performs inline.

    Duplicated here on purpose: the trainer does it inside the epoch loop, so
    a test that called the trainer's own helper would be testing the helper
    against itself.  If this and the trainer disagree, this one is the
    specification.
    """
    switch = int(getattr(args, "gen_validity_samples", 0) or 0)
    raw = getattr(args, "gen_eval_n", None)
    return switch if raw is None else int(raw)


# ---------------------------------------------------------------------------
# 1. The switch.  --gen-eval-n must not switch the sampler ON by existing.
# ---------------------------------------------------------------------------
class TestEvalSwitch:
    def test_decode_only_stays_decode_only(self, trainer, monkeypatch):
        """No flags at all -> n=0 -> the eval does not run.

        This is the regression the `default=None` choice exists to prevent.
        """
        monkeypatch.setattr(sys, "argv", ["trainer"])
        assert _resolve_n(trainer.parse_args()) == 0

    def test_legacy_flag_alone_is_unchanged(self, trainer, monkeypatch):
        """A pre-D3 invocation keeps its exact per-seed n."""
        monkeypatch.setattr(
            sys, "argv", ["trainer", "--gen-validity-samples", "16"]
        )
        assert _resolve_n(trainer.parse_args()) == 16

    def test_gen_eval_n_alone_turns_the_eval_on(self, trainer, monkeypatch):
        """Setting the new flag is an explicit request to sample."""
        monkeypatch.setattr(sys, "argv", ["trainer", "--gen-eval-n", "8"])
        assert _resolve_n(trainer.parse_args()) == 8

    def test_gen_eval_n_overrides_the_legacy_count(self, trainer, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["trainer", "--gen-validity-samples", "16", "--gen-eval-n", "64"],
        )
        assert _resolve_n(trainer.parse_args()) == 64

    def test_gen_eval_n_zero_disables(self, trainer, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["trainer", "--gen-validity-samples", "16", "--gen-eval-n", "0"]
        )
        assert _resolve_n(trainer.parse_args()) == 0

    def test_default_gen_eval_n_is_the_noise_floor_not_16(self, trainer):
        """The RECOMMENDED default must be the one the measurement supports.

        n=16 has a 1-sigma band of +-12.5pp at p=0.5, which is wider than
        the effect the run is trying to resolve.  n=64 gives +-6.2pp.
        """
        assert trainer.DEFAULT_GEN_EVAL_N == 64
        half = 0.5 / (trainer.DEFAULT_GEN_EVAL_N**0.5)
        assert half < 0.07, f"1-sigma band {half:.3f} is not under 7pp"


# ---------------------------------------------------------------------------
# 2. Seed parsing.  An eval knob must not be able to cost a GPU run.
# ---------------------------------------------------------------------------
class TestSeedParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, [0, 1, 2]),
            ("0 1 2", [0, 1, 2]),
            ("0,1,2", [0, 1, 2]),
            ("3 4 5", [3, 4, 5]),
            ([7, 8], [7, 8]),
            (9, [9]),
            ("", [0, 1, 2]),
            ((1, 2), [1, 2]),
        ],
    )
    def test_accepted_forms(self, trainer, raw, expected):
        assert trainer.parse_gen_eval_seeds(raw) == expected

    def test_garbage_falls_back_instead_of_raising(self, trainer):
        """A typo in an eval flag must not abort a 2.7 hr run."""
        assert trainer.parse_gen_eval_seeds("0 1 oops") == [0, 1, 2]

    def test_default_seed_count(self, trainer):
        assert trainer.DEFAULT_GEN_EVAL_SEEDS == (0, 1, 2)
        assert len(trainer.DEFAULT_GEN_EVAL_SEEDS) == 3


# ---------------------------------------------------------------------------
# 3. The seeds must score DIFFERENT compositions.
# ---------------------------------------------------------------------------
class TestTemplateWindows:
    def test_offsets_give_disjoint_windows(self):
        """The property the offsets exist to provide.

        Pure arithmetic on the windows the eval builds, with no model in the
        loop: seed i scores templates[i*n : (i+1)*n].  Three seeds at n=64
        must touch 192 DIFFERENT compositions, not 64 compositions three
        times over — otherwise the extra samples buy precision on a narrow
        slice and nothing else.
        """
        templates = list(range(1000))
        n = 64
        windows = []
        for i in range(3):
            off = i * n
            windows.append(templates[off : off + n])
        assert [len(w) for w in windows] == [64, 64, 64]
        seen = [t for w in windows for t in w]
        assert len(set(seen)) == 192, "seed windows overlap"
        assert not (set(windows[0]) & set(windows[1]))
        assert not (set(windows[1]) & set(windows[2]))

    def test_offset_beyond_the_pool_falls_back_rather_than_returning_nothing(
        self, trainer, monkeypatch
    ):
        """An offset past the end must still score something.

        A val pool smaller than seeds*n (a small smoke run, a tiny debug
        split) used to be a silent 0/0 if the slice were trusted.  It
        wraps to the head of the pool instead, and says so by returning
        fewer than n_samples rather than nothing.
        """
        captured = {}

        def _fake_gen(model, pool, device, *, method, n_steps, seed):
            captured["pool"] = list(pool)
            captured["seed"] = seed
            # The real generator returns one cloud per template, one that
            # carries that template's composition.
            return list(pool)

        monkeypatch.setattr(
            trainer, "generate_clouds_from_noise", _fake_gen, raising=True
        )
        templates = [object() for _ in range(5)]
        n_ok, n_tot = trainer.evaluate_generation_validity(
            model=object(),
            decoder=object(),
            templates=templates,
            device="cpu",
            n_samples=4,
            seed=3,
            template_offset=4,  # window would be [4:8] -> only 1 element
        )
        assert n_ok == 0 and n_tot == 1
        assert len(captured["pool"]) == 1


# ---------------------------------------------------------------------------
# 4. The chemistry summary pools rows; it does not average averages.
# ---------------------------------------------------------------------------
class TestChemistryPooling:
    def test_pooled_summary_is_row_weighted_not_seed_weighted(self, trainer):
        """Two seeds of very different yield must pool by SAMPLE, not by SEED.

        Averaging per-seed summaries would report 0.5 here (mean of 1.0 and
        0.0) over 5 real samples that contain 1 success — a different number
        about a different denominator, reported identically.
        """
        seed_a = [
            {"n_fragments": 1, "n_heavy": 10, "largest_fragment_frac": 1.0,
             "has_metal": True},
        ] * 1
        seed_b = [
            {"n_fragments": 4, "n_heavy": 10, "largest_fragment_frac": 0.4,
             "has_metal": False},
        ] * 4
        pooled = trainer.summarise_decode_chemistry(list(seed_a) + list(seed_b))
        assert pooled["n_chemistry_scored"] == 5
        # Row-weighted: (1.0 + 4*0.4) / 5.  Seed-weighted would be (1.0+0.4)/2.
        assert pooled["largest_fragment_frac_mean"] == pytest.approx(0.52)
        assert pooled["metal_ratio"] == pytest.approx(0.2)
        assert pooled["single_fragment_ratio"] == pytest.approx(0.2)

    def test_unparseable_rows_do_not_count_as_molecules(self, trainer):
        """A `None` row is an unparsed decode, not a zero-fragment molecule."""
        rows = [
            None,
            {"n_fragments": 1, "n_heavy": 6, "largest_fragment_frac": 1.0,
             "has_metal": True},
        ]
        out = trainer.summarise_decode_chemistry(rows)
        assert out["n_chemistry_scored"] == 1
        assert out["metal_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. The per-seed split has to be RECORDED, not just computed.
# ---------------------------------------------------------------------------
class TestPerSeedRecording:
    def test_gen_eval_is_a_registered_event_type(self, trainer):
        """Per-seed rows go to events.jsonl under a NEW type.

        A new type (rather than overloading `train_epoch`) is what keeps the
        existing TUI parsers — which filter on `train_epoch` / `lr_step` /
        `resume` — working untouched.
        """
        assert "gen_eval" in trainer._EVENT_TYPES
        assert "train_epoch" in trainer._EVENT_TYPES

    def test_epoch_metrics_carries_the_split(self, trainer):
        """The dataclass must be able to hold the per-seed numbers.

        `gen_validity` alone cannot say whether a 0.75 was (0.74, 0.75, 0.76)
        or (0.55, 0.75, 0.95), and those support completely different
        conclusions about the run.
        """
        m = trainer.EpochMetrics(
            epoch=3,
            wall_time_s=1.0,
            train_loss=0.1,
            bond_ce_loss=0.1,
            wasserstein_loss=0.1,
            catherine_audit_pass=False,
            decode_ratio_rdkit_strict=0.4,
            decode_ratio_val=2,
            decode_ratio_n=5,
            decoding_attempted=True,
            lr=1e-3,
            grad_norm=0.0,
        )
        # Defaults must be the "did not run" state, not a measured zero.
        assert m.gen_eval_per_seed == []
        assert m.gen_seed_spread == 0.0
        assert m.gen_eval_n == 0
        m2 = trainer.EpochMetrics(
            epoch=4,
            wall_time_s=1.0,
            train_loss=0.1,
            bond_ce_loss=0.1,
            wasserstein_loss=0.1,
            catherine_audit_pass=False,
            decode_ratio_rdkit_strict=0.4,
            decode_ratio_val=2,
            decode_ratio_n=5,
            decoding_attempted=True,
            lr=1e-3,
            grad_norm=0.0,
            gen_eval_seeds="0,1,2",
            gen_eval_n=64,
            gen_eval_per_seed=[{"seed": 0, "gen_validity": 0.5}],
            gen_seed_spread=0.4,
        )
        assert m2.gen_eval_n == 64
        assert m2.gen_seed_spread == pytest.approx(0.4)
        # Distinct instances: a mutable default shared across epochs would
        # accumulate every earlier epoch's seeds.
        assert m.gen_eval_per_seed is not m2.gen_eval_per_seed


# ---------------------------------------------------------------------------
# 6. The four shipped configs must actually carry the new eval size.
# ---------------------------------------------------------------------------
class TestShippedConfigs:
    @pytest.mark.parametrize(
        "name",
        [
            "cfm_tmqm_all6",
            "cfm_full_retrain",
            "cfm_per_innovation_ablation_p1.4",
            "cfm_real_decode_lift_p1.3",
        ],
    )
    def test_config_resolves_to_192_samples(self, trainer, name, monkeypatch):
        """Each config must yield 3 x 64 = 192 DISTINCT scored samples.

        A config that silently kept n=16 would leave the shipped run exactly
        as noisy as Round-2 while looking configured.
        """
        cfg = PROJECT_ROOT / "configs" / f"{name}.yaml"
        if not cfg.is_file():
            pytest.skip(f"{name}.yaml not present")
        monkeypatch.setattr(
            sys, "argv", ["trainer", "--config", str(cfg)]
        )
        args = trainer.parse_args()
        n = _resolve_n(args)
        seeds = trainer.parse_gen_eval_seeds(
            getattr(args, "gen_eval_seeds", None)
        )
        assert n == 64, f"{name}: per-seed n is {n}, expected 64"
        assert len(seeds) == 3, f"{name}: {len(seeds)} seeds, expected 3"
        assert n * len(seeds) == 192
