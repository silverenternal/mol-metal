"""Wf-A114E L3 — the per-epoch record must not be clobbered.

MEASURED 2026-10-30 (`molmetal/reports/wfa114e/census_population_conflict.md`):
the trainer writes `per_epoch_metrics.json` to a FIXED path at the END of
training, so two runs launched into the same `--out-dir` overwrite each
other's record and the survivor is whichever finished LAST.  Four of the 59
census runs carry a TRUNCATED PREFIX of a longer run in the same directory,
and two of them flip across the 0.50 gate when the true series is restored.

The guard is a RUN IDENTITY (the whole config block) plus an EPOCH-EXTENT
check, because neither alone can separate every pair MEASURED on disk:

  * `cfm_fullcorpus_2026_10_10` — the 1-epoch CPU smoke that truncated a
    14-epoch run declares a DIFFERENT config (epochs=1, n_train=64,
    n_val=32, device=cpu), so the identity separates them.
  * `cfm_round3_retrain/run` and the two `cfm_round7_f5f6_retrain` dirs —
    the 4-epoch run that truncated a 20-epoch run shares that run's config
    EXACTLY (same declared budget, same seed, same device), so only the
    epoch EXTENT separates them.

Three gaps closed on 2026-10-30 (the Wf-A114E adversarial verifier):

  * D1 — the identity was SIX keys, so two runs with a different lr /
    batch_size / hidden_dim / wasserstein_weight could clobber each other
    silently.  It is now the WHOLE config block.
  * D2 — "nothing is deleted either way" was FALSE: an ALLOWED write that
    replaced a record of a DIFFERENT population destroyed it.  It is now
    preserved under a run-scoped filename.
  * D3 — a budget-extending RESUME was falsely refused.  The declared epoch
    budget is no longer a difference by itself when the incoming epoch
    range CONTINUES the existing one.

CPU-only, no training, no GPU: every case below is a pure function call
over a synthetic config dict.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: The config blocks MEASURED on disk for the four class-A census runs and
#: for the resume run.  Embedded so the test does not depend on the
#: gitignored report artifacts; `test_the_embedded_configs_match_the_ones_
#: on_disk` re-checks each against the file when the file is present.
CLASS_A_CONFIGS: dict[str, dict] = {
    # the 1-epoch CPU smoke's config — what the canonical file holds today
    "molmetal/reports/cfm_fullcorpus_2026_10_10": {
        "batch_size": 8,
        "bond_decision": "threshold",
        "bond_decision_threshold": 0.5,
        "closure_bounded": True,
        "connectivity_prior": "on",
        "connectivity_prior_min_logit": 0.0,
        "device": "cpu",
        "epochs": 1,
        "hidden_dim": 128,
        "lr": 0.0003,
        "lr_live": [4.542857142857142e-05],
        "n_layers": 3,
        "n_train": 64,
        "n_val": 32,
        "pocket_id": "1h36",
        "seed": 0,
        "target_decode_ratio": 0.8,
        "use_yuelbond": True,
        "wasserstein_weight": 0.1,
    },
    # the 4-epoch population's config — identical in the three round dirs
    # except for `bond_decision`
    "molmetal/reports/cfm_round3_retrain/run": {
        "batch_size": 64,
        "bond_decision": "threshold",
        "bond_decision_threshold": 0.5,
        "closure_bounded": True,
        "connectivity_prior": "on",
        "connectivity_prior_min_logit": 0.0,
        "device": "cuda:0",
        "epochs": 20,
        "hidden_dim": 128,
        "lr": 0.0003,
        "n_layers": 3,
        "n_train": 256,
        "n_val": 192,
        "pocket_id": "1h36",
        "seed": 0,
        "target_decode_ratio": 0.8,
        "use_yuelbond": True,
        "wasserstein_weight": 0.1,
    },
}
CLASS_A_CONFIGS["molmetal/reports/cfm_round7_f5f6_retrain/run"] = dict(
    CLASS_A_CONFIGS["molmetal/reports/cfm_round3_retrain/run"]
)
CLASS_A_CONFIGS[
    "molmetal/reports/cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529"
] = dict(
    CLASS_A_CONFIGS["molmetal/reports/cfm_round3_retrain/run"],
    bond_decision="argmax",
)

#: `cfm_c7_fullgen_2026_10_20` — the resume run: 14 rows, epochs 4..17.
RESUME_CONFIG: dict = {
    "batch_size": 64,
    "bond_decision": "threshold",
    "bond_decision_threshold": 0.5,
    "closure_bounded": True,
    "connectivity_prior": "on",
    "connectivity_prior_min_logit": 0.0,
    "device": "cuda:0",
    "epochs": 18,
    "hidden_dim": 128,
    "lr": 0.0003,
    "lr_live": [5.705654986801928e-06],
    "n_layers": 3,
    "n_train": 26715,
    "n_val": 192,
    "pocket_id": "1h36",
    "seed": 0,
    "target_decode_ratio": 0.8,
    "use_yuelbond": True,
    "wasserstein_weight": 0.1,
}

#: The four class-A census runs (Wf-A114E §2.3, MEASURED).
CLASS_A_DIRS = [
    "molmetal/reports/cfm_fullcorpus_2026_10_10",
    "molmetal/reports/cfm_round3_retrain/run",
    "molmetal/reports/cfm_round7_f5f6_retrain/run",
    "molmetal/reports/cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529",
]


@pytest.fixture(scope="module")
def trainer():
    path = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    spec = importlib.util.spec_from_file_location("_t_cfm_fp_guard", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_cfm_fp_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


def _config(**over):
    """A config block shaped like the trainer's own `config` payload."""
    cfg = {
        "epochs": 20,
        "n_train": 256,
        "n_val": 192,
        "pocket_id": "1h36",
        "seed": 0,
        "device": "cuda:0",
        # the hyperparameters Wf-A114E D1 measured as invisible to the
        # six-key fingerprint
        "lr": 0.0003,
        "batch_size": 64,
        "hidden_dim": 128,
        "wasserstein_weight": 0.1,
    }
    cfg.update(over)
    return cfg


def _payload(trainer, cfg, epochs, *, with_fingerprint=True):
    """A synthetic per-epoch record, shaped like the trainer's own."""
    out = {
        "config": cfg,
        "per_epoch_metrics": [{"epoch": int(e)} for e in epochs],
    }
    if with_fingerprint:
        out["run_fingerprint"] = trainer.compute_run_fingerprint(cfg)
        out["run_fingerprint_note"] = (
            "the config keys that identify this run."
        )
    return out


def _write(trainer, out_dir, cfg, epochs, *, with_fingerprint=True):
    """Do what main() does: resolve the record paths, then write."""
    payload = _payload(trainer, cfg, epochs, with_fingerprint=with_fingerprint)
    fingerprint = trainer.compute_run_fingerprint(cfg)
    record = trainer.resolve_record_paths(out_dir, fingerprint, epochs)
    trainer.write_text_atomic(
        record["json_path"], json.dumps(payload, indent=2) + "\n"
    )
    return record, payload


# ---------------------------------------------------------------------------
# The three cases the guard must get right
# ---------------------------------------------------------------------------


def test_a_missing_file_allows_the_write(trainer, tmp_path):
    target = tmp_path / "per_epoch_metrics.json"
    allowed, reason = trainer.may_write_per_epoch_metrics(
        target, trainer.compute_run_fingerprint(_config())
    )
    assert allowed is True
    assert "no existing" in reason


def test_the_same_fingerprint_allows_the_write(trainer, tmp_path):
    target = tmp_path / "per_epoch_metrics.json"
    cfg = _config()
    target.write_text(json.dumps(_payload(trainer, cfg, range(20))))
    allowed, reason = trainer.may_write_per_epoch_metrics(
        target, trainer.compute_run_fingerprint(cfg)
    )
    assert allowed is True
    assert "same run fingerprint" in reason


def test_a_different_fingerprint_is_refused_and_names_the_keys(trainer, tmp_path):
    target = tmp_path / "per_epoch_metrics.json"
    # The 1-epoch CPU smoke's config, MEASURED on
    # molmetal/reports/cfm_fullcorpus_2026_10_10.
    target.write_text(
        json.dumps(_payload(trainer, _config(epochs=1, device="cpu"), range(1)))
    )
    allowed, reason = trainer.may_write_per_epoch_metrics(
        target, trainer.compute_run_fingerprint(_config())
    )
    assert allowed is False
    assert "DIFFERENT run" in reason
    assert "differing keys" in reason
    assert "epochs" in reason and "device" in reason


def test_the_refusal_does_not_destroy_the_existing_file(trainer, tmp_path):
    """The other run's record must survive the refusal, byte for byte."""
    cfg_a = _config(epochs=20)
    record_a, payload_a = _write(trainer, tmp_path, cfg_a, range(20))
    canonical = tmp_path / "per_epoch_metrics.json"
    assert record_a["canonical"] is True
    before = canonical.read_bytes()

    # A different run (the 1-epoch CPU smoke) arrives into the same dir.
    cfg_b = _config(epochs=1, n_train=64, n_val=32, device="cpu")
    record_b, payload_b = _write(trainer, tmp_path, cfg_b, range(1))

    assert record_b["canonical"] is False
    assert record_b["json_path"] != canonical
    assert record_b["json_path"].is_file()
    assert canonical.read_bytes() == before, (
        "the refusal destroyed the other run's record"
    )
    assert json.loads(canonical.read_text())["config"] == cfg_a
    written_b = json.loads(record_b["json_path"].read_text())
    assert written_b["config"] == cfg_b
    assert written_b["per_epoch_metrics"] == [{"epoch": 0}]


# ---------------------------------------------------------------------------
# The MEASURED case the identity alone cannot catch
# ---------------------------------------------------------------------------


def test_a_truncated_prefix_with_the_same_config_is_refused(trainer, tmp_path):
    """MEASURED: `cfm_round3_retrain/run` — the 4-epoch run that truncated a
    20-epoch run shares that run's config EXACTLY."""
    cfg = _config(epochs=20)
    _write(trainer, tmp_path, cfg, range(20))
    canonical = tmp_path / "per_epoch_metrics.json"
    before = canonical.read_bytes()

    record, _written = _write(trainer, tmp_path, cfg, range(4))

    assert record["canonical"] is False
    assert "MORE epochs" in record["reason"]
    assert canonical.read_bytes() == before


def test_a_longer_same_config_rerun_is_allowed(trainer, tmp_path):
    cfg = _config(epochs=20)
    _write(trainer, tmp_path, cfg, range(4))
    record, payload = _write(trainer, tmp_path, cfg, range(20))
    assert record["canonical"] is True
    written = json.loads((tmp_path / "per_epoch_metrics.json").read_text())
    assert len(written["per_epoch_metrics"]) == 20


def test_a_resume_continuation_is_allowed(trainer, tmp_path):
    """MEASURED: `cfm_c7_fullgen_2026_10_20` — the canonical record holds
    epochs 4..17, the epochs a RESUMED run recorded.  The pre-resume record
    (epochs 0..3) must still be overwritable by it."""
    cfg = _config(epochs=18)
    _write(trainer, tmp_path, cfg, range(4))
    record, _written = _write(trainer, tmp_path, cfg, range(4, 18))
    assert record["canonical"] is True
    written = json.loads((tmp_path / "per_epoch_metrics.json").read_text())
    assert [r["epoch"] for r in written["per_epoch_metrics"]] == list(range(4, 18))


def test_an_empty_record_does_not_replace_a_longer_one(trainer, tmp_path):
    """A resumed run that has nothing left to train writes NO epoch rows.
    Under the old behaviour that empty record replaced the run's own
    history, and the census dropped the run (`load_run` needs a row)."""
    cfg = _config(epochs=18)
    _write(trainer, tmp_path, cfg, range(4, 18))
    canonical = tmp_path / "per_epoch_metrics.json"
    before = canonical.read_bytes()
    record, _written = _write(trainer, tmp_path, cfg, [])
    assert record["canonical"] is False
    assert canonical.read_bytes() == before


def test_two_populations_sharing_a_config_get_two_filenames(trainer, tmp_path):
    """MEASURED: the 4-epoch and the 20-epoch run in one directory share a
    config, so the run-scoped name must carry the epoch extent too."""
    cfg = _config(epochs=20)
    record_a, _ = _write(trainer, tmp_path, cfg, range(20))
    record_b, _ = _write(trainer, tmp_path, cfg, range(4))
    assert record_a["json_path"] != record_b["json_path"]
    assert record_b["json_path"].is_file()
    assert record_a["json_path"].is_file()


# ---------------------------------------------------------------------------
# D1 — the identity must cover the WHOLE config block
# ---------------------------------------------------------------------------


def test_the_identity_covers_the_whole_config_block(trainer):
    """D1: the identity is every config key, not a hand-picked six."""
    cfg = _config(lr_live=[1e-5], fm_prior_scale={"ok": 1.0})
    fp = trainer.compute_run_fingerprint(cfg)
    assert set(fp) == set(cfg) - set(trainer.DERIVED_CONFIG_KEYS)
    # the six keys the first version compared are a strict SUBSET of it
    assert set(trainer.RUN_FINGERPRINT_KEYS) < set(fp)
    for key in trainer.RUN_FINGERPRINT_KEYS:
        assert fp[key] == cfg[key]


def test_the_derived_keys_are_excluded_from_the_identity(trainer):
    """`lr_live` and `fm_prior_scale` are consequences of a run, not causes:
    comparing them would refuse a legitimate resume."""
    cfg = _config(lr_live=[1e-5], fm_prior_scale={"ok": 1.0})
    fp = trainer.compute_run_fingerprint(cfg)
    assert "lr_live" not in fp
    assert "fm_prior_scale" not in fp
    other = _config(lr_live=[9e-5], fm_prior_scale={"ok": 0.0})
    assert trainer.fingerprint_diff(fp, trainer.compute_run_fingerprint(other)) == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("lr", 0.001),
        ("batch_size", 16),
        ("hidden_dim", 64),
        ("wasserstein_weight", 0.0),
    ],
)
def test_a_run_that_differs_only_in_a_hyperparameter_is_a_different_run(
    trainer, tmp_path, key, value
):
    """D1: MEASURED over the 80 on-disk records — 29 distinct fingerprints,
    13 groups with more than one artifact, and every colliding group
    differed in exactly the keys the six-key fingerprint ignored."""
    cfg_a = _config()
    _write(trainer, tmp_path, cfg_a, range(20))
    canonical = tmp_path / "per_epoch_metrics.json"
    before = canonical.read_bytes()

    cfg_b = _config(**{key: value})
    record, _written = _write(trainer, tmp_path, cfg_b, range(20))

    assert record["canonical"] is False
    assert key in record["reason"]
    assert canonical.read_bytes() == before


def test_the_six_key_fingerprint_would_have_allowed_that_clobber(trainer):
    """The regression witness: the OLD identity was blind to exactly this."""
    cfg_a = _config(lr=0.001)
    cfg_b = _config(lr=0.0001)
    old = {
        k: cfg_a[k]
        for k in trainer.RUN_FINGERPRINT_KEYS
        if k in cfg_a
    }
    old_b = {
        k: cfg_b[k]
        for k in trainer.RUN_FINGERPRINT_KEYS
        if k in cfg_b
    }
    assert old == old_b, "the six-key identity would have called these the same"
    # the new identity does not
    assert "lr" in trainer.fingerprint_diff(
        trainer.compute_run_fingerprint(cfg_a),
        trainer.compute_run_fingerprint(cfg_b),
    )


# ---------------------------------------------------------------------------
# D2 — an allowed write must not DESTROY the record it replaces
# ---------------------------------------------------------------------------


def test_the_replaced_record_is_preserved_not_destroyed(trainer, tmp_path):
    """D2: a SHORT population writes first, a LONGER one arrives second.
    The write is allowed (the longer range is not a truncation), and the
    shorter record must survive on disk under a run-scoped filename."""
    cfg = _config(epochs=20)
    record_a, payload_a = _write(trainer, tmp_path, cfg, range(4))
    canonical = tmp_path / "per_epoch_metrics.json"
    short_bytes = canonical.read_bytes()

    record_b, _written = _write(trainer, tmp_path, cfg, range(20))

    assert record_b["canonical"] is True
    assert record_b["displaced"], "the replaced record was not preserved"
    kept = Path(record_b["displaced"][0])
    assert kept.is_file()
    assert kept.read_bytes() == short_bytes, (
        "the preserved copy is not the record that was replaced"
    )
    # and the canonical now holds the longer record
    assert len(
        json.loads(canonical.read_text())["per_epoch_metrics"]
    ) == 20


def test_a_refused_write_leaves_the_canonical_record_alone(trainer, tmp_path):
    """D2 + the class-A constraint: a refused write displaces nothing."""
    cfg = _config(epochs=20)
    _write(trainer, tmp_path, cfg, range(20))
    canonical = tmp_path / "per_epoch_metrics.json"
    before = canonical.read_bytes()

    record, _written = _write(trainer, tmp_path, cfg, range(4))

    assert record["canonical"] is False
    assert record["displaced"] == []
    assert canonical.read_bytes() == before


def test_the_preserved_copy_uses_the_run_scoped_naming_convention(
    trainer, tmp_path
):
    """D2: the preserved file is named the way the refusal path names its
    own — `per_epoch_metrics.{tag}.{json,csv}`."""
    cfg = _config(epochs=20)
    _write(trainer, tmp_path, cfg, range(4))
    (tmp_path / "per_epoch_metrics.csv").write_text("epoch,loss\n0,1.0\n")

    record, _written = _write(trainer, tmp_path, cfg, range(20))

    names = {Path(p).name for p in record["displaced"]}
    assert names, "nothing was preserved"
    for name in names:
        assert name.startswith("per_epoch_metrics.")
        assert name.endswith((".json", ".csv"))
    assert any(n.endswith(".csv") for n in names), (
        "the CSV of the replaced population was destroyed"
    )


# ---------------------------------------------------------------------------
# D3 — a budget-extending resume is not a clobber
# ---------------------------------------------------------------------------


def test_a_budget_extending_resume_is_allowed(trainer, tmp_path):
    """D3: run 1 writes epochs 4..17 with config.epochs=18; run 2 resumes
    with --epochs 20.  The budget change must not make them different."""
    cfg = _config(epochs=18)
    _write(trainer, tmp_path, cfg, range(4, 18))
    canonical = tmp_path / "per_epoch_metrics.json"

    resumed = _config(epochs=20)
    record, _written = _write(trainer, tmp_path, resumed, range(18, 20))

    assert record["canonical"] is True
    assert "resume" in record["reason"]
    written = json.loads(canonical.read_text())
    assert [r["epoch"] for r in written["per_epoch_metrics"]] == [18, 19]


def test_a_budget_change_with_a_different_hyperparameter_is_refused(
    trainer, tmp_path
):
    """D3: 'a budget change that comes with a different set of
    hyperparameters is still a different population'."""
    cfg = _config(epochs=18)
    _write(trainer, tmp_path, cfg, range(4, 18))
    canonical = tmp_path / "per_epoch_metrics.json"
    before = canonical.read_bytes()

    other = _config(epochs=20, lr=0.001)
    record, _written = _write(trainer, tmp_path, other, range(18, 20))

    assert record["canonical"] is False
    assert "lr" in record["reason"]
    assert canonical.read_bytes() == before


def test_a_budget_change_with_a_different_cohort_is_refused(trainer, tmp_path):
    """D3: 'a budget change that comes with a DIFFERENT cohort ... is still
    a different population'."""
    cfg = _config(epochs=18)
    _write(trainer, tmp_path, cfg, range(4, 18))
    canonical = tmp_path / "per_epoch_metrics.json"
    before = canonical.read_bytes()

    other = _config(epochs=20, n_train=4096)
    record, _written = _write(trainer, tmp_path, other, range(18, 20))

    assert record["canonical"] is False
    assert "n_train" in record["reason"]
    assert canonical.read_bytes() == before


def test_a_budget_change_without_a_continuation_is_refused(trainer, tmp_path):
    """The budget only excuses itself when the incoming epoch range
    CONTINUES the existing one."""
    cfg = _config(epochs=18)
    _write(trainer, tmp_path, cfg, range(4, 18))
    canonical = tmp_path / "per_epoch_metrics.json"
    before = canonical.read_bytes()

    other = _config(epochs=20)
    record, _written = _write(trainer, tmp_path, other, range(4, 6))

    assert record["canonical"] is False
    assert "epochs" in record["reason"]
    assert canonical.read_bytes() == before


# ---------------------------------------------------------------------------
# The four REAL class-A census runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_dir", CLASS_A_DIRS)
def test_a_class_a_run_is_still_refused(trainer, tmp_path, run_dir):
    """Wf-A114E §2.3: the four class-A census runs must STILL be refused,
    and the canonical record must survive BYTE-IDENTICAL."""
    cfg = CLASS_A_CONFIGS[run_dir]
    if run_dir.endswith("cfm_fullcorpus_2026_10_10"):
        # the canonical file holds the 1-epoch CPU smoke's record; the
        # 14-epoch run (n_epochs_run: 14, MEASURED from events.jsonl) is
        # the population that was clobbered.
        _write(trainer, tmp_path, cfg, range(1))
        incoming = dict(cfg, epochs=14)
        incoming_epochs = range(14)
    else:
        # the 20-epoch population wrote first, the 4-epoch one truncated it
        _write(trainer, tmp_path, cfg, range(20))
        incoming = dict(cfg)
        incoming_epochs = range(4)
    canonical = tmp_path / "per_epoch_metrics.json"
    before = canonical.read_bytes()

    record, _written = _write(trainer, tmp_path, incoming, incoming_epochs)

    assert record["canonical"] is False, f"{run_dir} was not refused"
    assert "differing keys" in record["reason"]
    assert "DIFFERENT run" in record["reason"]
    assert canonical.read_bytes() == before, (
        f"{run_dir}: the refusal did not preserve the canonical record"
    )


@pytest.mark.parametrize("run_dir", CLASS_A_DIRS)
def test_a_class_a_run_refusal_names_the_differing_keys(
    trainer, tmp_path, run_dir
):
    """The reason names WHAT differs, so a reader can tell the two
    populations apart without opening the files."""
    cfg = CLASS_A_CONFIGS[run_dir]
    if run_dir.endswith("cfm_fullcorpus_2026_10_10"):
        _write(trainer, tmp_path, cfg, range(1))
        incoming = dict(cfg, epochs=14, n_train=256, device="cuda:0")
        incoming_epochs = range(14)
    else:
        _write(trainer, tmp_path, cfg, range(20))
        incoming = dict(cfg)
        incoming_epochs = range(4)

    record, _written = _write(trainer, tmp_path, incoming, incoming_epochs)

    reason = record["reason"]
    if run_dir.endswith("cfm_fullcorpus_2026_10_10"):
        assert "epochs" in reason
        assert "n_train" in reason
        assert "device" in reason
    else:
        # the two populations share the config EXACTLY — the epoch extent
        # is what separates them, and the reason must say so
        assert "epoch_extent" in reason
        assert "MORE epochs" in reason


def test_the_resume_run_c7_is_still_allowed(trainer, tmp_path):
    """`cfm_c7_fullgen_2026_10_20` writes epochs 4..17 into the canonical
    record — a RESUME, and it must not be refused."""
    _write(trainer, tmp_path, RESUME_CONFIG, range(4))
    canonical = tmp_path / "per_epoch_metrics.json"

    record, _written = _write(trainer, tmp_path, RESUME_CONFIG, range(4, 18))

    assert record["canonical"] is True
    written = json.loads(canonical.read_text())
    assert [r["epoch"] for r in written["per_epoch_metrics"]] == list(
        range(4, 18)
    )


def test_the_embedded_configs_match_the_ones_on_disk():
    """The embedded fixtures are copies of the on-disk records' config
    blocks; this keeps the copies honest when the artifacts are present."""
    for run_dir, cfg in CLASS_A_CONFIGS.items():
        path = PROJECT_ROOT / run_dir / "per_epoch_metrics.json"
        if not path.is_file():
            pytest.skip(f"{run_dir} is not on disk")
        on_disk = json.loads(path.read_text(encoding="utf-8"))["config"]
        assert on_disk == cfg, f"{run_dir}: the embedded config drifted"
    path = PROJECT_ROOT / "molmetal/reports/cfm_c7_fullgen_2026_10_20" / (
        "per_epoch_metrics.json"
    )
    if path.is_file():
        on_disk = json.loads(path.read_text(encoding="utf-8"))["config"]
        assert on_disk == RESUME_CONFIG


# ---------------------------------------------------------------------------
# The identity itself
# ---------------------------------------------------------------------------


def test_a_legacy_record_without_a_fingerprint_is_read_from_its_config(
    trainer, tmp_path
):
    """Every artifact on disk predates the explicit fingerprint; its identity
    is recovered from its own `config` block."""
    target = tmp_path / "per_epoch_metrics.json"
    target.write_text(
        json.dumps(
            {
                "config": _config(epochs=1, device="cpu"),
                "per_epoch_metrics": [{"epoch": 0}],
            }
        )
    )
    fingerprint, epochs, source = trainer.read_existing_record(target)
    assert source == "config"
    assert fingerprint["epochs"] == 1
    assert fingerprint["device"] == "cpu"
    assert epochs == [0]
    allowed, _reason = trainer.may_write_per_epoch_metrics(
        target, trainer.compute_run_fingerprint(_config())
    )
    assert allowed is False


def test_an_unreadable_record_is_never_overwritten(trainer, tmp_path):
    target = tmp_path / "per_epoch_metrics.json"
    target.write_text("{not json")
    allowed, reason = trainer.may_write_per_epoch_metrics(
        target, trainer.compute_run_fingerprint(_config())
    )
    assert allowed is False
    assert "not a readable JSON" in reason


def test_the_written_record_carries_its_own_fingerprint(trainer, tmp_path):
    _record, payload = _write(trainer, tmp_path, _config(), range(3))
    assert payload["run_fingerprint"] == trainer.compute_run_fingerprint(_config())
    assert payload["run_fingerprint_note"]


def test_a_fingerprint_diff_names_only_the_keys_that_differ(trainer):
    a = trainer.compute_run_fingerprint(_config())
    b = trainer.compute_run_fingerprint(_config(epochs=1, seed=7))
    assert trainer.fingerprint_diff(a, b) == ["epochs", "seed"]


def test_no_argparse_flag_was_added_for_the_guard(trainer):
    """AGENTS.md §5.11 — YAML-only launch: the guard is not a knob."""
    src = Path(trainer.__file__).read_text(encoding="utf-8")
    adds = [ln for ln in src.splitlines() if "add_argument" in ln]
    for word in ("fingerprint", "clobber", "run_scoped", "run-scoped",
                 "displace", "preserve", "guard"):
        assert not [ln for ln in adds if word in ln.lower()], (
            f"the guard added a CLI flag mentioning {word!r}"
        )
