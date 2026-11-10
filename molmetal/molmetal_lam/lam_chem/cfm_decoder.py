"""CFM decoder module — ``use_yuelbond`` flag + decoder factory.

Wf-124 Phase 1 (2026-09-27) — central decoder dispatch for the CFM
generate path.  Two decoders are exposed via the
``use_yuelbond`` flag:

* ``use_yuelbond=False`` (default) — wraps the production
  :class:`~molmetal.molmetal_lam.lam_chem.decoder_rework.ReworkedDecoder`
  (Path-B soft prior + BondOrderHead).  This is the bit-exact same
  decoder that produced the ``wf_cfm_path_b_decoder_rework`` row in
  ``metrics/by_metric/decode_ratio.json``.
* ``use_yuelbond=True`` — routes through the new
  :class:`~molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond.YuelBondBondDecoder`
  (Wang & Dokholyan 2025 GNN head + valence-aware selector).  The
  smoke test on 8 synthetic Pt-click clouds lifts decode_ratio from
  ``0/8`` (current 4-fix bundle) to ``8/8 = 1.000``.

The dispatch logic lives in :func:`build_cfm_decoder` — a single
function the upstream ``_generate_impl`` path can call when it wants
to A/B the YuelBond decoder against the production decoder.

Why a thin dispatcher module (not a wrapper class)?
--------------------------------------------------
The existing ``_generate_impl`` in
``molmetal/adapters/flow_matching_lipman/__init__.py`` already wires
the production decoder (``BondAwareDecoder`` +
``ConnectivityAwareDecoder``).  Adding a parallel A/B path requires
two minimal additions:

1. A ``use_yuelbond`` flag on ``GenerationConfig`` (or the adapter's
   setup kwargs) — currently deferred to Wf-124 P1.3 retrain.
2. A factory function (this module) that returns a decoder object
   whose ``.decode(cloud)`` contract matches the production decoder.

This module ships the factory + smoke runner.  The flag plumbing in
``_generate_impl`` follows in Wf-124 P1.3 when the GPU retrain starts
(per :file:`TODO/pending/119_decode_ratio_strengthen.md` §3 phase 1.3).

Honest framing
--------------
The decode_ratio lift from 0/8 to 8/8 is on *synthetic* Pt-click
clouds (the canonical CFM smoke input).  Real-CFM samples (trained
velocity field on PlatinAI / tmQM) still report 0/8 (the
``wf_cfm_rescue/phase5_200step_smoke.json`` row).  The YuelBond head
is **untrained**; we ship the architectural wire-up so the GPU
retrain (Wf-124 P1.3) can swap heads without changing the smoke
runner.  This module's contribution is the *decoder swap path*, not
a trained model.

Lit anchors
-----------
* Wang, J.; Dokholyan, N. V. (2026). *Multimodal Bond Reconstruction
  toward Generative Molecular Design.*  J. Chem. Inf. Model. 66(2),
  1003-1012.  DOI 10.1021/acs.jcim.5c03052  (preprint: bioRxiv
  10.1101/2025.05.06.652517, 2025-05-11).  F1 = 92.7 % on the CDG set
  (worst case, sigma = 0.2 A); F1 = 98.4 % on standard 3-D.
  (Citation corrected 2026-10-31, Wf-A143 — this bullet previously
  printed the title "YuelBond: a graph neural network for recovering the molecular bond network from distorted 3D structures", the author initials "Wang, S., Dokholyan, P." and the attribution of the 783/1000 RDKit failure count to ``DetermineConnectivity``.  See
  `docs/HONEST_NEGATIVE_CATALOG.md` §6.257.)
  Correction note (2026-10-31, Wf-A148) — the Wf-A143 note above is
  PARTLY FALSE AT THIS SITE.  Its third clause claims that this file's
  pre-repair bullet carried "the attribution of the 783/1000 RDKit failure
  count to ``DetermineConnectivity``".
  It did not: neither "783" nor "DetermineConnectivity" ever occurred
  in this file.  Evidence (MEASURED against git on 2026-10-31):
      git show HEAD:molmetal/molmetal_lam/lam_chem/cfm_decoder.py | grep -n "783\\|DetermineConnectivity"
  -> NO MATCH (grep exit status 1).  The SAME grep on HEAD's
  `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` DOES match
  (lines 20-21), so the Wf-A143 note was accurate at THAT site and is
  wrong at THIS one.  The Wf-A143 note's REMAINING claims ARE true
  at this site and stand unchanged: the pre-repair bullet really did
  print the title "YuelBond: a graph neural network for recovering the
  molecular bond network from distorted 3D structures" and the author
  initials "Wang, S., Dokholyan, P.", and it really did carry only the
  bioRxiv preprint identifier with no journal DOI; the current bullet
  corrects all three to the published record — Wang, J.; Dokholyan,
  N. V. (2026), *Multimodal Bond Reconstruction toward Generative
  Molecular Design*, J. Chem. Inf. Model. 66(2), 1003-1012, DOI
  10.1021/acs.jcim.5c03052, whose preprint (bioRxiv
  10.1101/2025.05.06.652517, 2025-05-11) is titled "Multimodal Bonds
  Reconstruction Towards Generative Molecular Design".  Superseded
  wording preserved verbatim (AGENTS.md §5.5 rule #6) — the Wf-A143
  note is retained above, unaltered and undeleted:
      "(Citation corrected 2026-10-31, Wf-A143 — this bullet previously
      printed the title "YuelBond: a graph neural network for recovering the molecular bond network from distorted 3D structures", the author initials "Wang, S., Dokholyan, P." and the attribution of the 783/1000 RDKit failure count to ``DetermineConnectivity``.  See
      `docs/HONEST_NEGATIVE_CATALOG.md` §6.257.)"
  (Catalog entry pending — this wave does not own
  `docs/HONEST_NEGATIVE_CATALOG.md`.)
* Jin, W., Barzilay, R., Jaakkola, T. (2018). *Junction Tree
  Variational Autoencoder for Molecular Graph Generation.*  ICML
  2018, arXiv:1802.04364.  Valence-aware greedy selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from molmetal.models.bond_head import (
    AtomCloud,
    BondAwareDecoder,
    BondOrderHead,
    DecodedMol,
)


__all__ = [
    "CFMDecoder",
    "CFMDecodeResult",
    "build_cfm_decoder",
    "smoke_decode_ratio",
    "write_smoke_results_json",
]


# ---------------------------------------------------------------------------
# CFMDecodeResult — unified result container for both decoder paths.
# ---------------------------------------------------------------------------
@dataclass
class CFMDecodeResult:
    """Output of a CFM decoder pass — same shape for both paths.

    Attributes
    ----------
    decoded : DecodedMol
        The RDKit ``Mol`` plus bond_orders / dative_bonds.
    used_yuelbond : bool
        Echo of the ``use_yuelbond`` flag (so callers can route
        results through different post-check pipelines).
    decoder_name : str
        String identifier for the decoder that produced the result
        (``"rewired_decoder_rework"`` or ``"yuelbond"``).
    overvalent_rejected : int, default 0
        Number of candidate bonds dropped by the valence-cap filter.
        Only meaningful for the YuelBond path.
    """

    decoded: DecodedMol
    used_yuelbond: bool
    decoder_name: str
    overvalent_rejected: int = 0


# ---------------------------------------------------------------------------
# CFMDecoder — the unified decoder wrapper.
# ---------------------------------------------------------------------------
class CFMDecoder:
    """Unified CFM decoder — production or YuelBond, selected by flag.

    Parameters
    ----------
    use_yuelbond : bool, default ``False``
        When ``True``, the :meth:`decode` method routes through
        :class:`~molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond.YuelBondBondDecoder`.
        When ``False`` (default), it routes through the production
        :class:`~molmetal.molmetal_lam.lam_chem.decoder_rework.ReworkedDecoder`
        with a :class:`~molmetal.models.bond_head.BondAwareDecoder` and
        a default-trained :class:`~molmetal.models.bond_head.BondOrderHead`.
    yuelbond_decoder : optional
        Pre-built ``YuelBondBondDecoder``.  When ``None`` and
        ``use_yuelbond=True`` a fresh decoder is constructed with the
        default hidden_dim=64, n_layers=3.
    rework_decoder : optional
        Pre-built ``ReworkedDecoder`` (the production decoder).  When
        ``None`` and ``use_yuelbond=False`` a fresh decoder is built
        with a default-trained :class:`BondOrderHead`.

    Notes
    -----
    The two decoder paths return slightly different result containers
    (the YuelBond path emits
    :class:`~molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond.YuelBondDecodeResult`;
    the production path emits :class:`DecodedMol`).  :meth:`decode`
    normalises both into :class:`CFMDecodeResult` so callers don't
    need to branch on the flag.
    """

    def __init__(
        self,
        use_yuelbond: bool = False,
        yuelbond_decoder: Any = None,
        rework_decoder: Any = None,
        bond_head: Optional[BondOrderHead] = None,
        connectivity_prior: bool = True,
    ) -> None:
        self.use_yuelbond = bool(use_yuelbond)
        self._yuelbond_decoder = yuelbond_decoder
        self._rework_decoder = rework_decoder
        self._bond_head = bond_head
        # Round-3 D2 — only reaches the lazily-built YuelBond decoder below;
        # a pre-built `yuelbond_decoder` was configured at its own site.
        self.connectivity_prior = bool(connectivity_prior)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decode(self, cloud: AtomCloud) -> CFMDecodeResult:
        """Run the selected decoder on an :class:`AtomCloud`.

        Returns
        -------
        :class:`CFMDecodeResult`
            Normalised output container.
        """
        if self.use_yuelbond:
            yd = self._get_yuelbond_decoder()
            yres = yd.decode(cloud)
            return CFMDecodeResult(
                decoded=yres.decoded,
                used_yuelbond=True,
                decoder_name="yuelbond",
                overvalent_rejected=yres.bonds_rejected_overvalent,
            )
        # Production path — ReworkedDecoder (Path-B soft prior).
        rd = self._get_rework_decoder()
        decoded = rd.decode(cloud)
        return CFMDecodeResult(
            decoded=decoded,
            used_yuelbond=False,
            decoder_name="rewired_decoder_rework",
            overvalent_rejected=0,
        )

    # ------------------------------------------------------------------
    # Lazy decoders
    # ------------------------------------------------------------------
    def _get_yuelbond_decoder(self) -> Any:
        """Lazy-construct a YuelBondBondDecoder on first use."""
        if self._yuelbond_decoder is None:
            from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
                YuelBondBondDecoder,
            )

            self._yuelbond_decoder = YuelBondBondDecoder(
                hidden_dim=64,
                n_layers=3,
                connectivity_prior=self.connectivity_prior,
            )
        return self._yuelbond_decoder

    def _get_rework_decoder(self) -> Any:
        """Lazy-construct a ReworkedDecoder on first use."""
        if self._rework_decoder is None:
            from molmetal.molmetal_lam.lam_chem.decoder_rework import (
                DecoderRework,
                ReworkedDecoder,
            )

            if self._bond_head is None:
                from molmetal.models.bond_head import (
                    default_trained_head,
                )

                self._bond_head = default_trained_head()
            inner = BondAwareDecoder(bond_head=self._bond_head)
            rework = DecoderRework()
            self._rework_decoder = ReworkedDecoder(
                inner=inner,
                rework=rework,
            )
        return self._rework_decoder


# ---------------------------------------------------------------------------
# Factory — the entry point the upstream ``_generate_impl`` will call.
# ---------------------------------------------------------------------------
def build_cfm_decoder(
    use_yuelbond: bool = False,
    bond_head: Optional[BondOrderHead] = None,
    yuelbond_decoder: Any = None,
    rework_decoder: Any = None,
    connectivity_prior: bool = True,
) -> CFMDecoder:
    """Build a :class:`CFMDecoder` with the right path selected.

    This is the single entry point the production ``_generate_impl``
    calls when it wants to A/B YuelBond vs the production decoder.
    The default (``use_yuelbond=False``) is bit-exact with the current
    CFM smoke output (``decode_ratio = 0/8`` on the freshly-init h=64
    adapter); the ``use_yuelbond=True`` path lifts the smoke to
    ``8/8 = 1.000`` on synthetic Pt-click clouds (per
    ``yuelbond_smoke_results.json``).

    Parameters
    ----------
    use_yuelbond : bool, default ``False``
        Switch decoder family.
    bond_head : optional
        Pre-built bond head (used by the production path).  When
        ``None`` a default-trained head is built on first decode.
    yuelbond_decoder : optional
        Pre-built :class:`YuelBondBondDecoder`.  Used when
        ``use_yuelbond=True``.  When ``None`` a fresh decoder is built
        on first decode.
    rework_decoder : optional
        Pre-built :class:`ReworkedDecoder`.  Used when
        ``use_yuelbond=False``.  When ``None`` a fresh decoder is
        built on first decode.
    """
    return CFMDecoder(
        use_yuelbond=use_yuelbond,
        yuelbond_decoder=yuelbond_decoder,
        rework_decoder=rework_decoder,
        bond_head=bond_head,
        connectivity_prior=connectivity_prior,
    )


# ---------------------------------------------------------------------------
# Smoke runner — the canonical ``decode_ratio`` probe.
# ---------------------------------------------------------------------------
def smoke_decode_ratio(
    n_clouds: int = 8,
    jitter: float = 0.20,
    use_yuelbond: bool = True,
    seed: int = 0,
) -> Tuple[int, int, List[CFMDecodeResult]]:
    """Run the canonical CFM decode smoke on synthetic Pt-click clouds.

    Returns
    -------
    (n_decoded, n_clouds, results)
        ``n_decoded`` is the number of clouds where the decoder
        returned ``sanitized=True``; ``n_clouds`` is the total count;
        ``results`` is the per-cloud :class:`CFMDecodeResult` list
        (useful for diagnostics).

    Notes
    -----
    The smoke mirrors ``wf_cfm_rescue/phase5_200step_smoke.json`` in
    size (8 samples, 200-step CPU) but uses the synthetic Pt-click
    cloud generator from
    :mod:`molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond` instead
    of a freshly-init h=64 adapter.  This isolates the *decoder*
    contribution to ``decode_ratio`` from the *velocity field*
    contribution — the EGNN velocity field is the bottleneck on real
    CFM samples (see ``metrics/by_metric/decode_ratio.json`` row
    ``wf_cfm_rescue (200-step 8-sample smoke)``); the decoder swap
    here only proves the wire-up is correct.
    """
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        build_synthetic_pt_click_clouds,
    )

    clouds = build_synthetic_pt_click_clouds(
        n_clouds=n_clouds,
        jitter=jitter,
        seed=seed,
    )
    decoder = build_cfm_decoder(use_yuelbond=use_yuelbond)
    results: List[CFMDecodeResult] = []
    n_decoded = 0
    for cloud in clouds:
        res = decoder.decode(cloud)
        results.append(res)
        if res.decoded.sanitized:
            n_decoded += 1
    return n_decoded, n_clouds, results


# ---------------------------------------------------------------------------
# Results JSON writer (Wf-124 Phase 1 deliverable).
# ---------------------------------------------------------------------------
def write_smoke_results_json(
    output_path: str,
    n_decoded: int,
    n_clouds: int,
    use_yuelbond: bool = True,
    jitter: float = 0.20,
    seed: int = 0,
    extra: Optional[dict] = None,
) -> None:
    """Write the canonical smoke results JSON.

    Parameters
    ----------
    output_path : str
        Path to write the JSON.  Parent dirs are created as needed.
    n_decoded : int
        Number of clouds where ``sanitized=True``.
    n_clouds : int
        Total number of clouds in the smoke.
    use_yuelbond : bool, default ``True``
        Echo of the ``use_yuelbond`` flag the smoke was run with.
    jitter : float, default ``0.20``
        Position jitter used by the smoke.
    seed : int, default ``0``
        RNG seed used by the smoke.
    extra : dict, optional
        Extra fields to include in the JSON (e.g. test counts,
        wall-clock timings).
    """
    import json
    from pathlib import Path

    decode_ratio = n_decoded / max(1, n_clouds)
    payload = {
        "wf": "wf124",
        "phase": 1,
        "task": "yuelbond_decoder_wireup",
        "smoke_kind": "synthetic_pt_click_clouds",
        "n_decoded": int(n_decoded),
        "n_clouds": int(n_clouds),
        "decode_ratio": float(decode_ratio),
        "use_yuelbond": bool(use_yuelbond),
        "jitter": float(jitter),
        "seed": int(seed),
        "baseline_wf_cfm_rescue_decode_ratio": "0/8 = 0.000",
        "path_b_rdkit_strict_baseline": 0.5,
        "method": (
            "Wang+Dokholyan2025 YuelBond head + valence-aware greedy "
            "selector (K=3 message-passing layers, hidden_dim=64, "
            "5k params)"
        ),
        "device": "cpu",
        "notes": (
            "Wf-124 Phase 1 smoke — wire-up of YuelBond decoder into "
            "CFM decode path. decode_ratio > 0 vs the current 0/8 "
            "baseline with the 4-fix bundle (wf_cfm_rescue/phase5 "
            "_200step_smoke.json) demonstrates the architectural "
            "wire-up is bit-correct. Real-CFM lift requires a GPU "
            "retrain with the trained YuelBond head (deferred to "
            "Wf-124 P1.3 per TODO/pending/119_decode_ratio_strengthen.md)."
        ),
    }
    if extra:
        payload["extra"] = dict(extra)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
