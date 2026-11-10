"""Cross-dataset transfer-learning validation — F fix.

Wf-A36-1 (2026-10-14) E1 + E2 — wire the PocketEncoder into the eval path
and apply the EMA snapshot to the FULL CFM velocity field, not just the
YuelBond bond head.  At default (``use_pocket_encoder=False`` and no EMA
in the checkpoint) this module is bit-exact identical to the pre-Wf-A36-1
path for every shipped CFM YAML; see :mod:`molmetal.eval._pocket_encoder`
for the helper definitions and the ~~§5.11 backward-compat contract~~
→ AGENTS.md §3.3 audit-ledger; corrected 2026-10-31.

Targets the F weakness from ``docs/CFM_WEAKNESSES_2026_10_13.md``:

    F — 0/149 generated molecules contain Pt (real_tmqm Pt fraction 25.2%);
    root cause was a reservoir-luck failure.  The min_per_metal
    sampler patch closed the eval-time gap, but cross-dataset
    TRANSFER LEARNING (PlatinAI / NCI-60 / MetalCytoToxDB) was never
    systematically evaluated.  Pretrain + 0.046 (Ru fine-tune) on
    random scaffold was REFUTED on Bemis–Murcko scaffold split
    (paired delta -0.0094, t(4) = -0.839, 95% CI [-0.0280, +0.0116]).

This script is a SCRIPT TEMPLATE — invoking it requires:
  1. A trained CFM checkpoint on tmQM (we have one — the seed-0
     pretrain at molmetal/checkpoints/).
  2. PlatinAI 4-cell labels (already loaded by
     molmetal.molmetal_lam.lam_chem.platinai_dataset.PlatinAIDataset).
  3. NCI-60 GI50 9-cell labels (per todo.json::legacy_identifiers, TIER_1.m21).
  4. MetalCytoToxDB (per todo.json::legacy_identifiers, TIER_1.m1).

Default YAML flag ``eval.cross_dataset_validation: false`` keeps the
9-wave trainer sum bit-exact.  When set to true, the trainer's eval
loop emits a JSON with one row per (dataset, fold, cell_line) so
reviewers can verify the transfer claim.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

# Wf-A37-1 (NEW-4, 2026-10-15) — the NCI-60 9-cell SMILES loader used to
# call ``list(NCI60_GI50_DEFAULT.smiles)[:64]`` on a ``pathlib.Path``,
# which has no ``.smiles`` attribute.  AttributeError was silently caught
# at the outer try/except and the branch returned ``[]``.  The fix points
# at the project-shipped precious-metal cytotoxicity CSV
# (``molmetal/data/nci60_precious_9cell.csv`` — a 7,824-row join of
# MetalCytoToxDB compounds against the 9 precious-metal-active NCI-60
# cell lines), which is the on-disk source of NCI-60 SMILES that
# ``molmetal/data/four_domain_combined.py`` already uses as
# ``DEFAULT_NCI60_CSV``.  The legacy ``NCI60_GI50_DEFAULT`` path in
# ``platinai_dataset.py`` points at the raw GI50 measurement CSV, which
# carries NSC identifiers + AVERAGE/ASTDDEV columns but no SMILES — so
# there is nothing to read there.  We do NOT re-introduce the bare
# ``NCI60_GI50_DEFAULT`` path here.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_NCI60_PRECIOUS_9CELL_CSV = _PROJECT_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"

LOG = logging.getLogger("molmetal.eval.cross_dataset")

# Wf-A30-1 (2026-10-14) A7 — import the canonical helper from the eval
# package so a single definition covers every eval module.  The local
# ``_ckpt_is_smoke_stub`` alias keeps legacy callers working without
# changing the public surface; both names resolve to the same code path.
from molmetal.eval import ckpt_is_smoke_stub as _ckpt_is_smoke_stub

# Wf-A36-1 (2026-10-14) E1 + E2 — the shared helpers for the eval-side
# PocketEncoder instantiation (E1) and the full-model EMA round-trip
# (E2).  See :mod:`molmetal.eval._pocket_encoder` for the
# ~~§5.11 bit-exact default~~ → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31
# (``use_pocket_encoder=False`` ⇒ no import, no warn,
# no behaviour change).
from molmetal.eval._pocket_encoder import (
    build_pocket_encoder as _build_pocket_encoder,
    try_build_full_cfm_model as _try_build_full_cfm_model,
)

def _load_pretrained_checkpoint(ckpt_path: Path, device: str = "cpu") -> Dict:
    """Load a CFM checkpoint into a flat dict.

    Returns a dict with ``model_state_dict``, ``optimiser_state_dict``,
    ``epoch``, ``metrics``.  Tolerates files saved by both the
    legacy ``torch.save`` (single dict) and the new ``atomic_write``
    path (same shape).

    Wf-A30-1 (2026-10-14) A7 — refuse the smoke-stub checkpoint up
    front, with a clear error, rather than silently passing through.
    """
    if _ckpt_is_smoke_stub(ckpt_path):
        raise ValueError(
            f"[cross-dataset] refusing smoke-stub checkpoint {ckpt_path}; "
            f"rename to a non-`.smoke_stub.pt` path after a real retrain."
        )
    import torch
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        # Bare state_dict — wrap for the rest of the script.
        return {"model_state_dict": obj, "epoch": -1, "metrics": {}}
    raise ValueError(f"Unrecognised checkpoint layout at {ckpt_path}: {type(obj)}")


def _load_nci60_smiles(limit: int = 64) -> List[str]:
    """Read NCI-60 9-cell SMILES from the project-shipped precious-metal
    cytotoxicity CSV (``molmetal/data/nci60_precious_9cell.csv``).

    Wf-A37-1 (NEW-4, 2026-10-15) — replaces the legacy
    ``list(NCI60_GI50_DEFAULT.smiles)[:64]`` call which silently
    raised AttributeError (the constant is a ``pathlib.Path``, not a
    dataset) and produced an empty ``nci60_9cell`` record for every
    cross-dataset run.  The on-disk GI50 measurement CSV at
    ``.data/molmetal/NCI60_GI50/GI50.csv`` carries NSC identifiers +
    AVERAGE / STDDEV columns but no SMILES column, so there is nothing
    to read there.  The precious-metal cytotoxicity CSV
    (``molmetal/data/nci60_precious_9cell.csv``) is the actual on-shipment
    source of NCI-60 SMILES — a 7,824-row join of MetalCytoToxDB
    compounds against the 9 precious-metal-active NCI-60 cell lines
    (A2780, A2780cis, A549, DU145, HCT116, HL60, MCF7, PC3, SKOV3);
    ``molmetal/data/four_domain_combined.py`` already uses it as
    ``DEFAULT_NCI60_CSV``.

    Order-preserving dedup, then slice to ``limit``.  Raises
    ``FileNotFoundError`` if the project CSV is missing and
    ``ValueError`` if the CSV header has no ``smiles`` column.  The
    caller (``_load_smiles`` in :func:`run_cross_dataset_validation`)
    wraps this in a try/except that logs and returns ``[]`` — so a
    missing CSV no longer silently produces an empty cross-dataset
    record without any log evidence.
    """
    if not _NCI60_PRECIOUS_9CELL_CSV.exists():
        raise FileNotFoundError(
            f"NCI-60 9-cell precious CSV not found at {_NCI60_PRECIOUS_9CELL_CSV}"
        )
    seen: set = set()
    out: List[str] = []
    with _NCI60_PRECIOUS_9CELL_CSV.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        if "smiles" not in (reader.fieldnames or []):
            raise ValueError(
                f"{_NCI60_PRECIOUS_9CELL_CSV} has no 'smiles' column; "
                f"present columns: {reader.fieldnames}"
            )
        for row in reader:
            s = (row.get("smiles") or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                break
    return out


def _predict_smiles(
    state: Dict,
    dataset_smiles: List[str],
    device: str = "cpu",
    max_n: int = 64,
    ema_state: Optional[Dict] = None,
    use_pocket_encoder: bool = False,
) -> List[str]:
    """Decode dataset SMILES through the loaded CFM YuelBond head.

    Accepts the FULL checkpoint ``state`` dict returned by
    :func:`_load_pretrained_checkpoint` (which carries the
    ``model_state_dict`` + ``ema_state_dict`` + the trainer's
    ``args`` sub-dict written at save time per the trainer's
    ``_ckpt_state_a30["args"] = vars(args)``).  Backward-compat
    passthrough: a flat ``model_state_dict`` (a plain dict of
    parameter tensors) is also accepted; on that path the bond-decision
    helper degrades to canonical defaults ``("argmax", 0.5, None)``
    because no ``"args"`` sub-dict is present.

    Wf-A36-1 (2026-10-14) E1 + E2 — when ``use_pocket_encoder=True``,
    instantiate the eval-side :class:`PocketGenEncoder` (a structural
    wiring change; the cross-dataset path currently has no pocket data
    to feed it, but the encoder is now in the eval module so a future
    pocket-conditioned decode can attach it without touching this
    function).  When ``ema_state`` is non-empty, the helper tries to
    apply the snapshot to the FULL :class:`CFMJointModel` (not just
    the YuelBond head) so the CFM trajectory / bond head see the
    EMA-smoothed weights.  Both paths are bit-exact no-ops at the
    ~~§5.11 default~~ → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31
    (``use_pocket_encoder=False`` and a pre-Wf-A20
    checkpoint with no ``ema_state_dict``).
    """
    # Wf-A35-FIXUP-1 (2026-10-14) — accept the full ckpt dict so
    # ``bond_decision_settings`` below can read the trainer's ``args``
    # sub-dict.  Pre-fix the function took a FLAT ``model_state_dict``
    # (parameter tensors only), so the helper only ever saw a flat
    # dict-of-tensors and the saved yaml's ``flow.bond_decision`` /
    # ``bond_decision_threshold`` / ``class_logit_bias`` were silently
    # dropped at cross-dataset decode time.  Backward compat: a flat
    # state_dict (no ``"args"`` key) is forwarded directly.
    if isinstance(state, dict) and isinstance(state.get("model_state_dict"), dict):
        model_state: Dict = state["model_state_dict"]  # type: ignore[assignment]
        _ckpt_state: Dict = state
    else:
        model_state = state
        _ckpt_state = state

    try:
        import torch
        from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
            YuelBondDecoderHead,
        )
        from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
            YuelBondBondDecoder,
        )
        from molmetal.eval.linear_probe import smiles_to_atom_cloud
        # Wf-A35-1 (2026-10-14) C2 — read the (decision, threshold,
        # class_logit_bias) triplet from the loaded ckpt so the
        # ``YuelBondBondDecoder`` constructed below uses the SAME rule
        # the trainer's :func:`evaluate_bond_accuracy` used when it
        # reported bond F1. Until this fix the kwargs were dropped
        # (``decision="argmax"``, ``decision_threshold=0.5``,
        # ``class_logit_bias=None`` were always assumed) regardless of
        # what the saved yaml carried.  Helper signature is
        # ``bond_decision_settings(state) -> (decision, threshold,
        # class_logit_bias)``; it accepts the full ckpt dict from
        # ``_load_pretrained_checkpoint`` (which has the trainer's
        # ``args`` sub-dict) or a flat ``args`` mapping.  Defaults are
        # the historical triple, so every shipped CFM YAML produces
        # bit-identical output here.
        from molmetal.eval.bond_decision import (
            bond_decision_settings as _bond_decision_settings,
        )
    except Exception as exc:  # pragma: no cover - import path
        LOG.warning("[cross-dataset] decode import failed: %s; pass-through", exc)
        return list(dataset_smiles)

    head_sd = {
        k[len("head."):]: v
        for k, v in model_state.items()
        if k.startswith("head.")
    }
    if not head_sd:
        LOG.warning(
            "[cross-dataset] model_state has no 'head.' keys; pass-through"
        )
        return list(dataset_smiles)

    # Wf-A30-1 (2026-10-14) A5 — read `atom_encoder.weight`, NOT
    # `head.atom_encoder.weight`. The ``k[len("head."):]`` strip above
    # already removes the prefix, so the correct key is just
    # ``atom_encoder.weight``. The legacy spelling caused a KeyError
    # on every ckpt that did NOT pre-strip the prefix in the writer,
    # which silently took the warn-and-pass-through branch above and
    # emitted a 0/0 cross-dataset record.
    try:
        atom_w = head_sd["atom_encoder.weight"]
        hidden_dim = int(atom_w.shape[0])
        n_layers = sum(
            1
            for k in head_sd
            if k.startswith("msg_mlps.") and k.endswith(".0.weight")
        )
        if n_layers < 2:
            n_layers = 3
    except Exception as exc:
        LOG.warning(
            "[cross-dataset] ckpt head shape unrecognised: %s; pass-through",
            exc,
        )
        return list(dataset_smiles)

    try:
        head = YuelBondDecoderHead(hidden_dim=hidden_dim, n_layers=n_layers)
        # The head_sd keys do NOT include the 'head.' prefix after the
        # strip above; pass strict=False so any future layer additions
        # do not break the loader.
        head.load_state_dict(head_sd, strict=False)
        LOG.info(
            "[cross-dataset] loaded head from ckpt (hidden_dim=%d, n_layers=%d)",
            hidden_dim,
            n_layers,
        )
    except Exception as exc:
        LOG.warning(
            "[cross-dataset] head load_state_dict failed: %s; pass-through",
            exc,
        )
        return list(dataset_smiles)

    # Wf-A30-1 (2026-10-14) A6 — EMA round-trip.  When the saved
    # checkpoint carries an EMA snapshot AND the head's stubs accept
    # it, swap it in before any forward pass so the decode runs on
    # the EMA-averaged weights.  Without this, every cross-dataset
    # metric was measured against the post-final-step live weights,
    # which on a 50-epoch run is dominated by the SGD noise of the
    # last few batches.
    #
    # Wf-A36-1 (2026-10-14) E2 — try the FULL :class:`CFMJointModel`
    # first.  The head-only stub (the legacy code path) just sets a
    # flag — the live weights were never actually swapped out, so the
    # decode silently ran on noisy post-final-step weights.  Building
    # the full model + applying ``model.ema_swap_in()`` swaps the EMA
    # snapshot into every parameter (vel_probe, atom_embed, head, etc.)
    # and the decoder then consumes the head from inside the model.
    # Falls back to the head-only stub when CFMJointModel construction
    # fails (e.g. the model was saved with a flag combination this
    # checkpoint reader does not understand).
    _ema_applied = False
    _full_model = None
    if isinstance(ema_state, dict) and ema_state:
        _full_built, _full_model = _try_build_full_cfm_model(
            model_state, ema_state
        )
        if _full_built and _full_model is not None:
            # Re-bind the decoder's head to the model-internal one so
            # the EMA-smoothed weights are what the bond head sees.
            head = _full_model.head
            _ema_applied = True
            LOG.info(
                "[cross-dataset] applied full-model EMA (%d tensors); "
                "decode uses model.head",
                len(ema_state),
            )
        else:
            # Fallback: head-only stub (legacy pre-Wf-A36-1 behaviour).
            try:
                head.load_ema_state(ema_state)
                _ema_applied = bool(head.ema_swap_in())
                if _ema_applied:
                    LOG.info(
                        "[cross-dataset] applied EMA snapshot to head "
                        "only (%d tensors) — full-model build failed",
                        len(ema_state),
                    )
            except Exception as exc:
                LOG.warning(
                    "[cross-dataset] EMA round-trip failed: %s; "
                    "using live weights",
                    exc,
                )
    else:
        LOG.info("[ckpt-load] no ema_state_dict; using live weights")

    # Wf-A36-1 (2026-10-14) E1 — instantiate the eval-side
    # :class:`PocketGenEncoder` when the caller opts in.  The
    # cross-dataset path does not have pocket point-clouds to feed the
    # encoder (the input is SMILES, not (ligand, pocket) pairs), so
    # the encoder is currently a structural no-op on this decode path;
    # the helper is wired in so a future pocket-conditioned decode
    # variant can attach it without touching this function.  At the
    # ~~§5.11 default~~ → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31
    # (``use_pocket_encoder=False``) the helper
    # short-circuits before any import and emits no log / warning, so
    # every shipped CFM YAML produces bit-exact identical eval output.
    _pocket_req, _pocket_enc, _pocket_dim = _build_pocket_encoder(
        bool(use_pocket_encoder), _ckpt_state, int(hidden_dim)
    )
    if _pocket_req and _pocket_enc is not None:
        LOG.info(
            "[cross-dataset] PocketGenEncoder wired in "
            "(pocket_embed_dim=%d)",
            _pocket_dim,
        )

    # Wf-A35-1 (2026-10-14) C2 — same triplet the trainer's
    # :func:`evaluate_bond_accuracy` reads from live state.  At the
    # canonical defaults (``argmax``, ``0.5``, ``None``) this is a
    # bit-exact no-op pass-through against the pre-Wf-A35-1
    # constructor call.  When the saved yaml set
    # ``flow.bond_decision: threshold`` (or its bias counterpart),
    # the pre-fix decoder silently ignored it; the metrics and the
    # assembled molecules now share one rule.
    #
    # Wf-A35-FIXUP-1 (2026-10-14) — pass the FULL ckpt ``state``
    # (with ``"args"`` sub-dict carrying the trainer's
    # ``vars(args)`` saved at ``_ckpt_state_a30["args"]``) instead
    # of the flat ``model_state_dict``.  Pre-fix the helper only ever
    # saw a flat dict-of-tensors, so the saved yaml's
    # ``flow.bond_decision`` was silently dropped on this decode path
    # even though the trainer's metrics side read it.  At the
    # canonical defaults the helper still returns
    # ``("argmax", 0.5, None)``, so the post-fix decode is bit-exact
    # identical to the pre-fix decode for every shipped CFM YAML
    # whose ``flow.bond_decision`` is unset.
    _decision, _threshold, _bias = _bond_decision_settings(_ckpt_state)
    decoder = YuelBondBondDecoder(
        head=head,
        decision=_decision,
        decision_threshold=_threshold,
        class_logit_bias=_bias,
    )
    out: List[str] = []
    n_fallback = 0
    for smi in dataset_smiles[: int(max_n)]:
        try:
            cloud = smiles_to_atom_cloud(smi)
            if cloud is None:
                out.append(smi)
                n_fallback += 1
                continue
            res = decoder.decode(cloud)
            decoded_smi = res.decoded.smiles if res.decoded.sanitized else ""
            if not decoded_smi:
                out.append(smi)
                n_fallback += 1
            else:
                out.append(decoded_smi)
        except Exception as exc:
            LOG.warning(
                "[cross-dataset] decode failure on %r: %s",
                smi[:40], exc,
            )
            out.append(smi)
            n_fallback += 1
    # Wf-A30-1 A6 — restore the live weights if we swapped the EMA in,
    # so the head's caller (e.g. linear_probe.py, which shares the
    # same checkpoint-loader code path) observes the same state we'd
    # see had no EMA existed.
    #
    # Wf-A36-1 (2026-10-14) E2 — when the full-model EMA path was used
    # (``_full_model`` is not None), call ``ema_swap_out`` on the model
    # instead of the head, so the vel_probe / atom_embed weights also
    # get restored.  The head-only fallback keeps the legacy call.
    if _ema_applied:
        try:
            if _full_model is not None:
                _full_model.ema_swap_out()
            else:
                head.ema_swap_out()
        except Exception:
            pass
    if n_fallback:
        LOG.info(
            "[cross-dataset] decode: %d/%d fell back to input SMILES",
            n_fallback, min(len(dataset_smiles), int(max_n)),
        )
    return out


def _score_oracle(smiles_batch: List[str], oracle_name: str) -> List[float]:
    """Per-`eval/scoring/<oracle>.py` score on a decoded SMILES batch.

    Returns one float per SMILES (NaN-safe: unparseable SMILES get
    the corpus median per dataset — measured at the time the script
    is invoked, not hard-coded here).
    """
    try:
        if oracle_name == "pocket2mol_score":
            from molmetal.eval.scoring.pocket2mol_score import score
            return [float(score(s)) if score(s) is not None else float("nan") for s in smiles_batch]
        if oracle_name == "softmol_score":
            from molmetal.eval.scoring.softmol_score import score
            return [float(score(s)) if score(s) is not None else float("nan") for s in smiles_batch]
        if oracle_name == "targetdiff_score":
            from molmetal.eval.scoring.targetdiff_score import score
            return [float(score(s)) if score(s) is not None else float("nan") for s in smiles_batch]
    except Exception as exc:
        LOG.warning("oracle %s failed: %s", oracle_name, exc)
    return [float("nan")] * len(smiles_batch)


def run_cross_dataset_validation(
    ckpt_path: str,
    out_json: str,
    datasets: Optional[List[str]] = None,
    device: str = "cpu",
    do_linear_probe: bool = False,
    linear_probe_epochs: int = 20,
    linear_probe_max_n: int = 200,
    use_pocket_encoder: bool = False,
) -> Dict:
    """Run transfer-learning validation against the 3 cross-datasets.

    Parameters
    ----------
    ckpt_path : str
        Path to a trained CFM checkpoint (per
        ``molmetal.checkpoints``).  Required.
    out_json : str
        Output JSON path.  Will be overwritten if it exists.
    datasets : list of str, optional
        One or more of ``{"platinai_4cell", "nci60_9cell",
        "metalcytotoxdb"}``.  Default = all three.
    device : str
        "cpu" or "cuda".  Default "cpu" so the script can run on the
        smoke CPU without GPU.
    do_linear_probe : bool
        When ``True`` (default ``False``), additionally run the
        :func:`molmetal.eval.linear_probe.run_cross_dataset_linear_probe`
        path: freeze the YuelBond encoder, train a small logistic head
        on the per-cell labels, and attach the per-label AUC + mean
        AUC to every per-(dataset, oracle) record.  Default off so the
        pre-Wf-A14 JSON shape is preserved when the flag is absent.
    linear_probe_epochs : int
        Logistic-regression epochs for the probe head.  Default 20.
    linear_probe_max_n : int
        Cap on rows used by the probe (smoke-friendly).  Default 200.
    use_pocket_encoder : bool
        Wf-A36-1 (2026-10-14) E1 — when ``True``, instantiate the
        eval-side :class:`PocketGenEncoder` so a future
        pocket-conditioned decode variant can attach it.  Default
        ``False`` keeps the ~~§5.11 bit-exact path~~ → AGENTS.md §3.3
        audit-ledger; corrected 2026-10-31: every shipped CFM
        YAML produces identical eval output regardless of this flag.

    Returns
    -------
    dict with keys ``{"results", "config", "wall_time_s"}`` where
    ``results`` is a list of per-(dataset, fold, oracle) score records
    (each record optionally has ``linear_probe`` and
    ``linear_probe_reason`` keys when ``do_linear_probe=True``).
    """
    if datasets is None:
        datasets = ["platinai_4cell", "nci60_9cell", "metalcytotoxdb"]
    t0 = time.time()
    LOG.info("[cross-dataset] loading checkpoint: %s", ckpt_path)
    state = _load_pretrained_checkpoint(Path(ckpt_path), device=device)
    LOG.info("[cross-dataset] loaded checkpoint at epoch %s", state.get("epoch", -1))

    # Wf-A37-1 (NEW-4, 2026-10-15) — the NCI-60 9-cell SMILES loader
    # now lives at module scope as ``_load_nci60_smiles`` (defined just
    # above :func:`_load_pretrained_checkpoint`).  See that function's
    # docstring for the full rationale (the legacy code called
    # ``list(NCI60_GI50_DEFAULT.smiles)[:64]`` on a ``pathlib.Path``,
    # raised AttributeError, and the branch returned ``[]`` silently).

    # Map dataset name → (SMILES loader, oracle to score on).
    def _load_smiles(name: str) -> List[str]:
        try:
            if name == "platinai_4cell":
                from molmetal.molmetal_lam.lam_chem.platinai_dataset import PlatinAIDataset
                ds = PlatinAIDataset()
                # Wf-A36-2 F2 (2026-10-14) — explicit per-row SMILES contract
                # (Wf-A14 dataset protocol).  Previously ``getattr(ds,
                # "smiles", [])`` silently dropped dataset content when
                # the attribute was missing; the benchmark then ran with
                # an empty cohort and reported a meaningless score.
                # Backward-compat: datasets that DO expose .smiles behave
                # bit-exact identically (we slice [:64] the same way).
                if not hasattr(ds, "smiles"):
                    raise AttributeError(
                        f"Dataset {ds!r} has no .smiles attribute; "
                        f"cross_dataset._load_smiles requires a per-row "
                        f"SMILES list. Did you mean to use a different "
                        f"loader?"
                    )
                return list(ds.smiles)[:64]
            if name == "nci60_9cell":
                # Wf-A37-1 (NEW-4, 2026-10-15) — was the buggy
                # ``list(NCI60_GI50_DEFAULT.smiles)[:64]`` line, where
                # ``NCI60_GI50_DEFAULT`` is a ``pathlib.Path`` (see
                # :mod:`molmetal.molmetal_lam.lam_chem.platinai_dataset`).
                # ``Path`` has no ``.smiles`` attribute so the call
                # raised AttributeError that was silently swallowed at
                # the outer try/except, and the nci60_9cell branch
                # returned ``[]`` for every cross-dataset run.  We now
                # read the project-shipped precious-metal cytotoxicity
                # CSV directly via the helper below.
                return _load_nci60_smiles(64)
            if name == "metalcytotoxdb":
                from molmetal.molmetal_lam.lam_chem.platinai_dataset import MetalCytoToxDataset
                ds = MetalCytoToxDataset()
                # Wf-A36-2 F2 — same explicit-contract fix; see comment above.
                if not hasattr(ds, "smiles"):
                    raise AttributeError(
                        f"Dataset {ds!r} has no .smiles attribute; "
                        f"cross_dataset._load_smiles requires a per-row "
                        f"SMILES list. Did you mean to use a different "
                        f"loader?"
                    )
                return list(ds.smiles)[:64]
        except Exception as exc:
            LOG.warning("[cross-dataset] dataset %s loader failed: %s", name, exc)
        return []

    out_records: List[Dict] = []
    for ds_name in datasets:
        smiles = _load_smiles(ds_name)
        if smiles:
            # Wf-A30-1 (2026-10-14) A6 — pull the EMA snapshot out of the
            # checkpoint (the trainer wrote ``ema_state_dict`` since
            # Wf-A20 when ``--use-ema=True``) and forward it to the
            # decoder.  A pre-Wf-A20 ckpt or an EMA-disabled run leaves
            # this empty; the decoder logs "no ema_state_dict; using
            # live weights" and continues.
            #
            # Wf-A35-FIXUP-1 (2026-10-14) — forward the FULL ``state``
            # (which carries the trainer's ``args`` sub-dict saved as
            # ``_ckpt_state_a30["args"] = vars(args)``) to
            # ``_predict_smiles`` so the bond-decision helper at the
            # :class:`YuelBondBondDecoder` construction site reads the
            # saved yaml's ``flow.bond_decision`` /
            # ``bond_decision_threshold`` / ``class_logit_bias``
            # instead of always defaulting to
            # ``("argmax", 0.5, None)``.  Pre-fix this call passed
            # ``state["model_state_dict"]`` (a flat parameter tensor
            # dict), so the helper's ``isinstance(state, dict) and
            # isinstance(state.get("args"), dict)`` branch never fired
            # — only the flat-lookup fallback ran.  At canonical
            # defaults (every shipped CFM YAML) the helper still
            # returns ``("argmax", 0.5, None)``, so this is a
            # bit-exact no-op for shipped runs and a real fix for any
            # future ckpt with non-default bond-decision settings.
            decoded = _predict_smiles(
                state,
                smiles,
                device=device,
                ema_state=state.get("ema_state_dict", {}),
                use_pocket_encoder=use_pocket_encoder,
            )
            for oracle in ("pocket2mol_score", "softmol_score", "targetdiff_score"):
                scores = _score_oracle(decoded, oracle)
                valid = [s for s in scores if s == s]  # NaN-safe filter
                if not valid:
                    continue
                mean_score = sum(valid) / len(valid)
                out_records.append(
                    {
                        "dataset": ds_name,
                        "n_input": len(smiles),
                        "n_decoded": len(decoded),
                        "n_valid_oracle": len(valid),
                        "oracle": oracle,
                        "mean_score": mean_score,
                    }
                )
        else:
            LOG.info(
                "[cross-dataset] %s: legacy _load_smiles returned empty; "
                "skipping oracle decode, but linear-probe will still run "
                "from the CSV loader.",
                ds_name,
            )
        # Wf-A14 (2026-10-13) — linear-probe transfer (freeze encoder,
        # train head).  Gated on do_linear_probe so the default path
        # is bit-exact identical to the pre-Wf-A14 output.
        # Note: the linear probe uses its OWN CSV loader
        # (``_load_dataset_for_probe`` in linear_probe.py), so it can
        # run even when the legacy ``_load_smiles`` returns empty.
        if do_linear_probe:
            try:
                from molmetal.eval.linear_probe import (
                    run_cross_dataset_linear_probe,
                )
                probe_res = run_cross_dataset_linear_probe(
                    ckpt_path=str(ckpt_path),
                    dataset_name=ds_name,
                    device=device,
                    n_epochs=int(linear_probe_epochs),
                    max_n=int(linear_probe_max_n),
                    use_pocket_encoder=use_pocket_encoder,
                )
                # Attach the probe record to the per-oracle records so
                # the JSON has a single entry per (dataset, oracle) +
                # one probe entry.  Reviewers can grep the JSON for
                # "linear_probe" to filter.
                if out_records:
                    for rec in out_records:
                        if rec["dataset"] == ds_name and "linear_probe" not in rec:
                            rec["linear_probe"] = probe_res.get("linear_probe")
                            rec["linear_probe_reason"] = probe_res.get("reason")
                else:
                    # No oracle records — emit a probe-only record so
                    # the JSON is non-empty.
                    out_records.append(
                        {
                            "dataset": ds_name,
                            "n_input": 0,
                            "n_decoded": 0,
                            "n_valid_oracle": 0,
                            "oracle": "<linear-probe-only>",
                            "mean_score": float("nan"),
                            "linear_probe": probe_res.get("linear_probe"),
                            "linear_probe_reason": probe_res.get("reason"),
                        }
                    )
            except Exception as exc:
                LOG.warning(
                    "[cross-dataset] linear probe failed for %s: %s",
                    ds_name, exc,
                )
                for rec in out_records:
                    if rec["dataset"] == ds_name:
                        rec["linear_probe_reason"] = f"exception: {exc}"
    payload = {
        "config": {
            "ckpt_path": str(ckpt_path),
            "datasets": datasets,
            "device": device,
            "out_json": out_json,
        },
        "results": out_records,
        "wall_time_s": time.time() - t0,
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    LOG.info("[cross-dataset] wrote %s (%d records, %.1fs)", out_json, len(out_records), payload["wall_time_s"])
    return payload


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Wf-A14 cross-dataset validation")
    parser.add_argument("--ckpt", required=True, help="Path to CFM checkpoint")
    parser.add_argument("--out", default="metrics/by_metric/cfm_cross_dataset_transfer.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["platinai_4cell", "nci60_9cell", "metalcytotoxdb"],
        help="Datasets to validate against",
    )
    parser.add_argument(
        "--do-linear-probe",
        action="store_true",
        help="Also run frozen-encoder linear probe (Wf-A14 transfer protocol)",
    )
    parser.add_argument(
        "--linear-probe-epochs",
        type=int,
        default=20,
        help="Logistic-regression epochs for the probe head (default 20)",
    )
    parser.add_argument(
        "--linear-probe-max-n",
        type=int,
        default=200,
        help="Max rows used by the linear probe (smoke-friendly)",
    )
    parser.add_argument(
        "--pocket-encoder",
        action="store_true",
        help=(
            "Wf-A36-1 (2026-10-14) E1 — instantiate the eval-side "
            "PocketGenEncoder so a future pocket-conditioned decode "
            "variant can attach it.  Default off (was: §5.11 bit-exact) "
            "→ AGENTS.md §3.3 audit-ledger; corrected 2026-10-31."
        ),
    )
    args = parser.parse_args()
    run_cross_dataset_validation(
        ckpt_path=args.ckpt,
        out_json=args.out,
        datasets=args.datasets,
        device=args.device,
        do_linear_probe=args.do_linear_probe,
        linear_probe_epochs=args.linear_probe_epochs,
        linear_probe_max_n=args.linear_probe_max_n,
        use_pocket_encoder=args.pocket_encoder,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_cross_dataset_validation",
    "_load_pretrained_checkpoint",
    "_predict_smiles",
    "_score_oracle",
]
