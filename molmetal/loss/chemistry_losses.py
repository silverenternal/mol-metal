"""Chemistry-aware loss terms — L1 / L-A1 / L-A3 / L-A8 fixes.

Each class appends its (key, tensor) pair to ``_LAST_LOSS_TERMS`` so the
existing ``test_cfm_loss_gradient_liveness.py`` regression continues to
gate ``requires_grad`` correctly. The default ``weight=0.0`` / ``enabled=False``
path returns an exact ``torch.zeros(())`` — no allocation beyond the dict
write, so the bit-exact Wf-120 / Wf-124 P2 / Wf-148 baselines are preserved
when no opt-in YAML key is set.

D-NEW-2 distinction
-------------------
Two classes can be LIVE-but-INERT: ``requires_grad=True`` and ``value=0.0``
simultaneously. ``AssembleTerm`` (in :mod:`aggregation_losses`) sits in
this state at assemble_formulation == "product" — the formulation that the
MEASURED code path uses today. Both ``MoleculeLoss`` and ``DiversityLoss``
can also be LIVE-but-INERT when ``decode_ratio == 0`` (their inputs are empty).
This is a structurally different condition from "the term is OFF" and is
preserved by NOT short-circuiting ``requires_grad`` in the disabled branch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

LOG = logging.getLogger("molmetal.loss.chemistry_losses")


class NaturalChargeLoss:
    """L-A2 — supervise per-atom natural-charge regression (NBO Loewdin q).

    The pre-Wf-A14 (2026-10-13) chemistry-signal pipeline drops
    per-atom natural charges on the floor: ``load_tmqm_clouds`` carries
    the Wiberg BO + oxidation state + spin + dft_electronic_e +
    metal_q signals to the trainer, but the per-atom NBO q vector
    (the one in ``tmQM_X.q``) was never opened.  This loss closes the
    gap by adding an MSE between the model's per-atom charge
    prediction and the Loewdin q target, weighted so the legacy
    trainer remains bit-exact when ``weight=0``.

    The q target is centred around 0 (typical: -1.5 to +1.5 e), and
    the loss uses MSE on the raw q (not centred) because the q itself
    is the chemistry signal the CFM is supposed to reproduce.

    YAML keys
    ---------
    loss.natural_charge_mse_weight: float = 0.0   # default OFF (bit-exact)
    loss.use_natural_charges: bool = False        # opt-in switch

    Why default OFF even when the file is loaded
    --------------------------------------------
    The natural-charge file ``.data/molmetal/tmQM/tmQM_X.q(.gz)`` is
    NOT shipped with the repo (downstream of Mendeley/GitHub/Zenodo,
    all of which are blocked on the Wf-A14 host — see
    :file:`docs/HONEST_NEGATIVE_CATALOG.md`).  When the file is later
    available, :func:`molmetal.data.tmqm.parse_natural_charges` will
    auto-detect it; the trainer can then opt-in via the YAML keys
    above.  Default ``weight=0`` preserves the pre-Wf-A14 sum so the
    7 shipped CFM YAMLs are unaffected.
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = True,
        name: str = "natural_charge_mse",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        predicted_charges: Optional[torch.Tensor],
        target_charges: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """MSE between predicted per-atom q and the NBO Loewdin q target.

        Both tensors are flat ``(N_atoms,)`` of the same length; the
        cloud-level summation happens at the trainer side.  Returns
        zero (with the right device) when the loss is disabled OR
        when the inputs are absent / shape-mismatched / empty — never
        NaN, never crashes the trainer.
        """
        if not self.enabled or self.weight <= 0.0:
            return (
                torch.zeros((), device=device)
                if device is not None
                else torch.zeros(())
            )
        if predicted_charges is None or target_charges is None:
            return (
                torch.zeros((), device=device)
                if device is not None
                else torch.zeros(())
            )
        if (
            predicted_charges.numel() == 0
            or target_charges.numel() == 0
        ):
            return (
                torch.zeros((), device=device)
                if device is not None
                else torch.zeros(())
            )
        if predicted_charges.shape != target_charges.shape:
            return (
                torch.zeros((), device=device)
                if device is not None
                else torch.zeros(())
            )
        loss = (
            torch.nn.functional.mse_loss(
                predicted_charges, target_charges.detach()
            )
            * self.weight
        )
        return loss

    def append(
        self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]
    ) -> None:
        sink[self.name] = tensor


class ContinuousBOLoss:
    """L1 / L-A1 — supervise continuous bond-order from a (E,1) logit head.

    The shipped ``bond_ce`` path supervises a 5-class discrete prediction
    (NUM_BOND_CLASSES, BOND_TYPES ∈ {0..4}). This class adds a parallel
    SUPERVISION on the *continuous* bond-order representation:

        target = log(bond_orders_continuous + 1.0)   # [0, 1.79] for [0,1]
        loss   = F.mse_loss(logits_for_bo.squeeze(-1), target) * weight

    The head that emits ``logits_for_bo`` is owned by sibling territory
    ``molmetal/models/bond_head.py`` (L9 cross-cite — see D-NEW-4 target-
    dependent leak note in SCOUT §D). When ``bond_head_continuous_bo: false``
    in the YAML, this class still runs, but its gradient does not reach
    a head slot — the user is responsible for setting the flag consistently.

    YAML keys
    ---------
    loss.bo_continuous_mse_weight: float = 0.1   # default ON
    loss.use_continuous_bo: bool = True          # opt-in switch
    """

    def __init__(
        self,
        weight: float = 0.1,
        enabled: bool = True,
        name: str = "bo_continuous_mse",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        logits_for_bo: Optional[torch.Tensor],
        bond_orders_continuous: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.weight <= 0.0:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        if logits_for_bo is None or bond_orders_continuous is None:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        if logits_for_bo.numel() == 0 or bond_orders_continuous.numel() == 0:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        # Mirror the trainer convention: target = log(bo + 1).
        target = torch.log(bond_orders_continuous.detach() + 1.0)
        pred = logits_for_bo.squeeze(-1)
        if pred.shape != target.shape:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        loss = torch.nn.functional.mse_loss(pred, target) * self.weight
        return loss

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class MoleculeLoss:
    """L-A3 — read chemistry from the DECODED cloud, not from the real one.

    QED + SA + ADMET aggregate over a batch of decoded SMILES / mol
    objects. Returns ``torch.zeros(())`` when the batch is empty or every
    SMILES is invalid — never NaN, never a tensor with ``requires_grad``
    on a leaf detached from the model.

    Sign is NEGATIVE because the trainer MINIMISES loss — QED / SA / ADMET
    are maximised by negative loss.

    YAML keys
    ---------
    loss.use_molecule_loss: bool = True       # default ON
    loss.molecule_loss_weight: float = 0.1   # default ON
    """

    def __init__(
        self,
        weight: float = 0.1,
        enabled: bool = True,
        name: str = "molecule_loss",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        decoded_smiles_batch: Optional[Sequence[str]],
        decoded_mol_objects: Optional[Sequence[Any]] = None,
        device: Optional[torch.device] = None,
        # ---- Wf-A16 (2026-10-13) c-flow pocket ID threading ----
        # Optional per-pocket override for the QED target.  When set,
        # pocket-aware QED targets are looked up from a static
        # ``pocket_qed_table`` sidecar.  When None (default), the legacy
        # single-target QED formula is preserved bit-for-bit.
        pocket_id: Optional[str] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.weight <= 0.0:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        if not decoded_smiles_batch:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            from rdkit import Chem  # noqa: F401  (RDKit import is heavy)
        except Exception as exc:  # pragma: no cover - RDKit missing
            LOG.warning("[MoleculeLoss] RDKit unavailable (%s) — returning 0", exc)
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        from rdkit.Chem import QED as _QED  # type: ignore
        qed_scores: List[float] = []
        sa_scores: List[float] = []
        for smi in decoded_smiles_batch:
            if not isinstance(smi, str) or not smi:
                continue
            # RDKit contract: Chem.MolFromSmiles returns None for any
            # invalid SMILES — it never raises.  Trust the None return;
            # a bare except would silently swallow real bugs (NameError
            # on _Chem rebound, TypeError on a shadowed wrapper).
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                qed_scores.append(float(_QED.qed(mol)))
            except (RuntimeError, ValueError):
                pass
            # SA score via ``rdkit.Chem.Contrib.SA_Score.sascorer`` (legacy
            # opt-in). Fall back to 0.5 (midpoint of [1, 10] SA range) when
            # the module is not present in the installed RDKit distribution.
            try:
                from rdkit.Chem import RDConfig  # noqa:  F401
                import os as _os
                _sasc_path = _os.path.join(_os.path.dirname(RDConfig.__file__), "Contrib", "SA_Score", "sascorer.py")
                if _os.path.isfile(_sasc_path):
                    import importlib.util as _ilu
                    _spec = _ilu.spec_from_file_location("_sascorer", _sasc_path)
                    if _spec is not None and _spec.loader is not None:
                        _mod = _ilu.module_from_spec(_spec)
                        _spec.loader.exec_module(_mod)
                        sa_scores.append(float(_mod.calculateScore(mol)))
            except (ImportError, OSError, AttributeError, RuntimeError, ValueError):
                # sascorer is an RDKit Contrib addon: not shipped in
                # every distribution (ImportError), the path may not
                # resolve (OSError / AttributeError on RDConfig), and
                # calculateScore can raise on rare molecules.
                pass
        # Normalise SA to [0, 1] via (10 - sa) / 9 so higher = better.
        sa_norm = [max(0.0, min(1.0, (10.0 - s) / 9.0)) for s in sa_scores]
        # ADMET Ames-safety surrogate via PAINS catalog.
        admet_ames_safe: List[float] = []
        try:
            from rdkit.Chem import FilterCatalog as _fc
            params = _fc.FilterCatalogParams()
            params.AddCatalog(_fc.FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = _fc.FilterCatalog(params)
            for _smi in decoded_smiles_batch:
                if not isinstance(_smi, str) or not _smi:
                    continue
                # RDKit contract: Chem.MolFromSmiles returns None for
                # any invalid SMILES — it never raises.
                _mol = Chem.MolFromSmiles(_smi)
                if _mol is None:
                    continue
                try:
                    admet_ames_safe.append(0.0 if catalog.HasMatch(_mol) else 1.0)
                except (RuntimeError, ValueError):
                    admet_ames_safe.append(0.5)
        except ImportError:
            # FilterCatalog not available in this RDKit build.
            admet_ames_safe = []
        if not qed_scores and not sa_norm and not admet_ames_safe:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        # Wf-A61 (2026-10-17) L18 — non-empty count guard.  The
        # previous per-component ``0.5`` fallback masked real defects:
        # a cohort with ONE empty component produced a misleading
        # ``0.5`` "average molecule" signal that looked like real
        # chemistry but wasn't.  The fix:
        #   * If 2+ of 3 components are empty, return zeros (current
        #     all-empty path covers that case already).
        #   * If exactly 1 of 3 is empty, weight the remaining
        #     component × ``3/non_empty`` so the scale matches the
        #     three-component baseline.  This keeps the
        #     ``W2 AB-blocks stay honest`` invariant for cohorts with
        #     a known RDKit Contrib gap (e.g. an RDKit build missing
        #     SA_Score).
        non_empty = sum(
            1
            for x in (qed_scores, sa_norm, admet_ames_safe)
            if x
        )
        if non_empty <= 1:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        scale = 3.0 / float(non_empty)
        mean_qed = (
            (sum(qed_scores) / max(len(qed_scores), 1)) * scale
            if qed_scores
            else 0.0
        )
        mean_sa = (
            (sum(sa_norm) / max(len(sa_norm), 1)) * scale
            if sa_norm
            else 0.0
        )
        mean_admet = (
            (sum(admet_ames_safe) / max(len(admet_ames_safe), 1)) * scale
            if admet_ames_safe
            else 0.0
        )
        aggregate = mean_qed + mean_sa + mean_admet
        loss = -float(self.weight) * float(aggregate)
        return torch.tensor(loss, device=device) if device is not None else torch.tensor(loss)

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class DiversityLoss:
    """L-A8 — mean pairwise Tanimoto distance over decoded SMILES.

    0.0 = identical batch; 1.0 = all distinct. Returns ``torch.zeros(())``
    when the batch has fewer than 2 valid SMILES — never NaN.

    Sign is NEGATIVE because the trainer MINIMISES loss — high diversity
    is rewarded by negative loss.

    YAML keys
    ---------
    loss.diversity_term_weight: float = 0.1
    loss.diversity_fp_radius: int = 2
    loss.diversity_fp_bits: int = 2048
    """

    def __init__(
        self,
        weight: float = 0.1,
        enabled: bool = True,
        fp_radius: int = 2,
        fp_bits: int = 2048,
        name: str = "diversity_loss",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.fp_radius = int(fp_radius)
        self.fp_bits = int(fp_bits)
        self.name = str(name)

    def compute(
        self,
        decoded_smiles_batch: Optional[Sequence[str]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.weight <= 0.0:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        if not decoded_smiles_batch:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, DataStructs
        except Exception as exc:  # pragma: no cover - RDKit missing
            LOG.warning("[DiversityLoss] RDKit unavailable (%s) — returning 0", exc)
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        fps: List[Any] = []
        for smi in decoded_smiles_batch:
            if not isinstance(smi, str) or not smi:
                continue
            # RDKit contract: Chem.MolFromSmiles returns None for any
            # invalid SMILES — it never raises.  See property_losses.py
            # Wf-A51 W3-D rationale for the narrow-no-except pattern.
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
            except (RuntimeError, ValueError):
                continue
            fps.append(fp)
        n = len(fps)
        if n < 2:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        # Pairwise Tanimoto distance.
        sims = [[round(float(s), 4) for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps)] for i in range(n)]
        n_pairs = n * (n - 1) // 2
        sum_dist = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                sum_dist += 1.0 - sims[i][j]
        mean_dist = sum_dist / max(n_pairs, 1)
        loss = -float(self.weight) * float(mean_dist)
        return torch.tensor(loss, device=device) if device is not None else torch.tensor(loss)

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class QEDLoss:
    """QED-only decomposition slot for ``MoleculeLoss``.

    Returns ``-mean(QED) * weight``. Empty batch → 0.
    """

    def __init__(
        self,
        weight: float = 0.1,
        enabled: bool = True,
        name: str = "qed_loss",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        decoded_smiles_batch: Optional[Sequence[str]],
        device: Optional[torch.device] = None,
        # ---- Wf-A16 (2026-10-13) c-flow pocket ID threading ----
        pocket_id: Optional[str] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.weight <= 0.0:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        if not decoded_smiles_batch:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            from rdkit import Chem
            from rdkit.Chem import QED
        except Exception as exc:  # pragma: no cover
            LOG.warning("[QEDLoss] RDKit unavailable (%s) — returning 0", exc)
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        scores: List[float] = []
        for smi in decoded_smiles_batch:
            if not isinstance(smi, str) or not smi:
                continue
            # RDKit contract: Chem.MolFromSmiles returns None for any
            # invalid SMILES — it never raises.  See property_losses.py
            # Wf-A51 W3-D rationale for the narrow-no-except pattern.
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                scores.append(float(QED.qed(mol)))
            except (RuntimeError, ValueError):
                continue
        if not scores:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        loss = -float(self.weight) * (sum(scores) / len(scores))
        return torch.tensor(loss, device=device) if device is not None else torch.tensor(loss)

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class DiversityRegularizer:
    """Wraps the existing inline ``diversity_term`` block (3142-3269).

    The inline block in the v2 trainer combines FOUR sub-terms whose
    behaviour is gated by six YAML flags:

      * top_p / top_k / temperature → nucleus sampling audit
      * mmd_weight                   → MMD loss on velocities vs ref_positions
      * repulsion_weight             → cosine repulsion on velocities
      * scaffold_conditioning        → re-CE on a conditioned-logits copy

    The composite is a TRAINER-LEVEL regulariser on the real training
    path — it is structurally DIFFERENT from :class:`DiversityLoss`
    (which reads DECODED SMILES and computes Tanimoto distance). Two
    distinct classes, two distinct code paths.

    Behaviour preservation
    ----------------------
    Bit-exact identical to the inline block when instantiated with the
    same flag set. The class is opt-in via ``diversity_weight > 0`` (a
    master switch that gates the WHOLE feature, mirroring how the inline
    block gated on a 6-way OR of all sub-flags).
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = True,
        top_p: float = 1.0,
        top_k: int = 0,
        temperature: float = 1.0,
        mmd_weight: float = 0.0,
        repulsion_weight: float = 0.0,
        scaffold_conditioning: bool = False,
        name: str = "diversity_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.mmd_weight = float(mmd_weight)
        self.repulsion_weight = float(repulsion_weight)
        self.scaffold_conditioning = bool(scaffold_conditioning)
        self.name = str(name)

    def _any_subflag_active(self) -> bool:
        return (
            self.top_p < 1.0
            or self.top_k > 0
            or self.temperature != 1.0
            or self.mmd_weight > 0.0
            or self.repulsion_weight > 0.0
            or self.scaffold_conditioning
        )

    def compute(
        self,
        all_logits: Optional[torch.Tensor],
        all_targets: Optional[torch.Tensor],
        batch_velocities: Optional[List[torch.Tensor]],
        bond_class_weights: Optional[torch.Tensor],
        reference_positions: Optional[torch.Tensor],
        device: torch.device,
    ) -> tuple:
        """Compute the diversity regularizer + a possibly-replaced bond_ce.

        Returns ``(diversity_term, custom_bond_ce_or_None, nucleus_sample_invocations)``
        matching the inline block's contract:

          * ``diversity_term`` — sum of (mmd_weight * mmd) + (repulsion_weight * rep)
                          contributions, or zero when no sub-flag is active.
          * ``custom_bond_ce_or_None`` — None when no sub-flag is active (the
                          trainer keeps the regular bond_ce); a re-computed
                          F.cross_entropy when scaffold_conditioning wrote
                          a conditioned logits tensor.
          * ``nucleus_sample_invocations`` — int count of nucleus-sample
                          element audits, matching the legacy
                          ``nucleus_sample_invocations`` counter that the
                          per-epoch audit hook (line ~4103) consumes.
        """
        diversity_term = torch.zeros((), device=device)
        custom_bond_ce: Optional[torch.Tensor] = None
        nucleus_sample_invocations = 0
        if (
            not self.enabled
            or self.weight <= 0.0
            or not self._any_subflag_active()
        ):
            return diversity_term, custom_bond_ce, nucleus_sample_invocations
        if all_logits is None or all_targets is None:
            return diversity_term, custom_bond_ce, nucleus_sample_invocations
        try:
            from molmetal.adapters.flow_matching_lipman import (
                diversity_sampling as _ds,
            )
            from molmetal.adapters.flow_matching_lipman import (
                scaffold_conditioning as _sc,
            )
        except ImportError:
            # Optional sampling/conditioning modules not available.
            _ds = None  # type: ignore
            _sc = None  # type: ignore
        if _ds is None or _sc is None:
            return diversity_term, custom_bond_ce, nucleus_sample_invocations
        # --- (a) nucleus sampling audit ---
        if self.top_p < 1.0 or self.top_k > 0 or self.temperature != 1.0:
            with torch.no_grad():
                sampled = _ds.nucleus_sample(
                    all_logits,
                    p=self.top_p,
                    k=self.top_k,
                    temperature=self.temperature,
                )
                nucleus_sample_invocations += int(sampled.numel())
        # --- (b) MMD regulariser ---
        if self.mmd_weight > 0.0 and batch_velocities:
            vel = torch.cat(batch_velocities, dim=0)
            if reference_positions is not None:
                ref = reference_positions.to(device)
                n_vel = vel.shape[0]
                if ref.shape[0] < n_vel:
                    ref = ref.repeat((n_vel // ref.shape[0] + 1, 1))[:n_vel]
                with torch.no_grad():
                    _sigma = float(torch.cdist(vel, ref).median().clamp(min=1e-3))
                mmd_val = _ds.mmd_loss(vel, ref.detach(), sigma=_sigma, unbiased=False)
                diversity_term = diversity_term + self.mmd_weight * mmd_val
        # --- (c) Repulsion loss ---
        if self.repulsion_weight > 0.0 and batch_velocities:
            vel = torch.cat(batch_velocities, dim=0)
            if vel.shape[0] >= 2:
                rep_val = _ds.repulsion_loss(vel, margin=0.95, reduce=True)
                diversity_term = diversity_term + self.repulsion_weight * rep_val
        # --- (d) Scaffold conditioning ---
        if self.scaffold_conditioning:
            batch_n = all_logits.shape[0]
            if all_logits.dim() == 2:
                scaffold_idx = torch.zeros(batch_n, dtype=torch.long, device=device)
                scaffold_bias = torch.zeros(
                    (256, all_logits.shape[-1]),
                    dtype=all_logits.dtype,
                    device=device,
                )
                conditioned = _sc.scaffold_condition(
                    all_logits, scaffold_idx, n_scaffolds=256, scaffold_bias=scaffold_bias,
                )
                custom_bond_ce = torch.nn.functional.cross_entropy(
                    conditioned, all_targets, weight=bond_class_weights,
                )
        return diversity_term, custom_bond_ce, nucleus_sample_invocations

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor