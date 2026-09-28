"""SoftMolAdapter — soft-block diffusion molecule generator port.

Wraps the cloned SoftMol repo (``molmetal/references/SoftMol`` — paper
arXiv:2601.21964, Yang/Xu/Yuan/Zhu/Li/Ji 2026, "From Tokens to Blocks: A
Block-Diffusion Perspective on Molecular Generation") into the same
``MoleculeGenerator`` contract as ``facebook_fm_wrapper`` /
``rxnflow_adapter``:

  - ``name``                       -> "SoftMol_v1"
  - ``setup(device=None)``         -> locate SoftMol, load checkpoint
  - ``generate(pocket, config)``   -> SoftBD block-diffusion sampling
  - ``train_step(pocket, mols)``   -> no-op (SoftMol training lives upstream)
  - ``get_metadata()``             -> paper / code / weight info

SoftMol vs hand-rolled CFM:

    Hand-rolled CFM (facebookresearch/flow_matching + EGNN velocity field)
    produces *continuous* 3D coordinates; the SMILES is then decoded by
    RDKit round-trip.  SoftMol produces *token sequences* (SMILES-ish,
    plus BOS/EOS specials) using SoftBD block-diffusion — a discrete
    diffusion over an MDLM-style absorbing-mask vocabulary with
    "Adaptive Confidence Decoding" (first-hitting + top-1 position
    update).  Output is therefore SMILES-shaped; 3D coords must be
    obtained post-hoc (RDKit ``EmbedMolecule`` with ETKDGv3).  Use this
    adapter when you want a fully-discrete, vocab-native generator (no
    continuous space needed) and you accept the upstream weight
    dependency (≈600 MB checkpoint on Hugging Face Hub
    ``SZU-ADDG/SoftMol``).

Heavy imports (hydra / lightning / datasets / tokenizers / transformers)
are guarded; if any are missing the adapter degrades gracefully:

  * ``is_softmol_available()`` returns False
  * ``generate()`` returns ``[]`` (honest empty list — caller decides
    fail vs skip)
  * ``get_metadata()["fallback_active"]`` is True

The upstream repo (``molmetal/references/SoftMol/``) is NOT modified —
this adapter is additive.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from molmetal.domain import Molecule, Pocket
from molmetal.ports import GenerationConfig

logger = logging.getLogger(__name__)
_HERE = Path(__file__).resolve()
# SoftMol is cloned at the project root, NOT under molmetal/.
# (molmetal/references/ holds symlinked Python packages like flow_matching;
#  references/ holds cloned repos like SoftMol / RxnFlow / SynFlowNet.)
_REPO = _HERE.parents[2] / "references" / "SoftMol"
# Fallback: also check molmetal/references/ for backward compat
_REPO_FALLBACK = _HERE.parents[1] / "references" / "SoftMol"


# ---------------------------------------------------------------------------
# SoftMol is a discrete-vocab block-diffusion model, NOT a coordinate model.
# To keep the ``MoleculeGenerator`` contract uniform with the hand-rolled
# CFM (which emits (B, N, 3) coords + (B, N) atom_types), we set a stable
# placeholder constant for the atom-type arity used by the post-hoc RDKit
# embedding path.  SoftMol itself does not predict atom types — it predicts
# SMILES tokens, which we then round-trip through RDKit.
# ---------------------------------------------------------------------------
_SOFTMOL_ATOM_TYPES = 11  # mirror FacebookFMWrapper: {1,6,7,8,9,15,16,17,34,35,53}
_SOFTMOL_DEFAULT_LENGTH = 72
_SOFTMOL_DEFAULT_BLOCK_SIZE = 8
_SOFTMOL_DEFAULT_STEPS = 128
_SOFTMOL_DEFAULT_NUCLEUS = 0.95
_SOFTMOL_DEFAULT_TEMPERATURE = 1.0
_SOFTMOL_HF_REPO = "SZU-ADDG/SoftMol"
_SOFTMOL_DEFAULT_CKPT = "weights/89M-epoch6-best.ckpt"


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------
def _resolve_repo() -> Optional[Path]:
    """Find the SoftMol cloned repo on disk.  Checks project-root first
    (the canonical location), then falls back to ``molmetal/references/``
    for backward compat.
    """
    for candidate in (_REPO, _REPO_FALLBACK):
        if candidate.is_dir() and (candidate / "sample.py").is_file():
            return candidate
    return None


def is_softmol_available() -> bool:
    """Return True iff the cloned SoftMol repo is on disk.

    Heavy-dep probes (hydra / lightning / tokenizers) are checked lazily
    in :func:`_try_import_upstream` because just trying to import the
    package from this module would crash on a default molmetal venv that
    does NOT have those deps installed.
    """
    return _resolve_repo() is not None


def _try_import_upstream():
    """Try to import the upstream SoftMol modules.  Returns dict or None.

    Required upstream modules (verified 2026-09-16):
        - ``dataloader.get_tokenizer`` — builds the SoftMol tokenizer
        - ``diffusion.Diffusion`` — the SoftBD LightningModule

    Optional upstream modules (graceful skip if missing):
        - ``utils`` — utilities used by sample.py
        - ``metrics`` — used by upstream evaluate()
    """
    repo = _resolve_repo()
    if repo is None:
        return None
    src = str(repo)
    if src not in sys.path:
        sys.path.insert(0, src)
    out: dict = {}
    try:
        import dataloader as _dataloader  # noqa: F401
        out["dataloader"] = _dataloader
    except Exception as exc:
        logger.debug("SoftMol dataloader import failed: %s", exc)
        return None
    try:
        import diffusion as _diffusion  # noqa: F401
        out["diffusion"] = _diffusion
    except Exception as exc:
        logger.debug("SoftMol diffusion import failed: %s", exc)
        return None
    return out


def _resolve_ckpt(ckpt: Optional[str]) -> Optional[str]:
    """Resolve the checkpoint path, optionally downloading from HF Hub.

    SoftMol's :func:`sample.generate` auto-downloads from
    ``SZU-ADDG/SoftMol`` if the local path is missing.  We mirror that
    contract here so the adapter degrades gracefully on offline hosts.
    """
    repo = _resolve_repo()
    if ckpt and os.path.isabs(ckpt) and os.path.exists(ckpt):
        return ckpt
    if ckpt and repo is not None:
        # Relative to the cloned repo, e.g. "weights/89M-epoch6-best.ckpt"
        local = repo / ckpt
        if local.exists():
            return str(local)
    # Auto-download fallback (mirrors SoftMol sample.generate)
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        target = ckpt or _SOFTMOL_DEFAULT_CKPT
        return hf_hub_download(repo_id=_SOFTMOL_HF_REPO, filename=target)
    except Exception as exc:
        logger.warning("SoftMol ckpt auto-download failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class SoftMolAdapter:
    """MoleculeGenerator port backed by SoftMol SoftBD block-diffusion.

    Public surface (matches the canonical :class:`molmetal.ports.MoleculeGenerator`):

      - ``name`` : str  — ``"SoftMol_v1"``
      - ``setup(device=None)`` : probe upstream, build config + model
      - ``generate(pocket, config)`` : sample SMILES via block-diffusion
      - ``train_step(pocket, mols)`` : 0.0 (no-op; SoftMol training is upstream)
      - ``get_metadata()`` : paper / code / config info

    When upstream is unavailable, ``generate()`` returns an honest empty
    list and ``get_metadata()["fallback_active"]`` is True.
    """

    name = "SoftMol_v1"

    def __init__(
        self,
        ref_repo_path: str = "molmetal/references/SoftMol",
        ckpt: Optional[str] = None,
        vocab: str = "vocab_V2.txt",
        length: int = _SOFTMOL_DEFAULT_LENGTH,
        block_size: int = _SOFTMOL_DEFAULT_BLOCK_SIZE,
        steps: int = _SOFTMOL_DEFAULT_STEPS,
        nucleus_p: float = _SOFTMOL_DEFAULT_NUCLEUS,
        temperature: float = _SOFTMOL_DEFAULT_TEMPERATURE,
        first_hitting: bool = True,
        top1: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        self._repo_path = Path(ref_repo_path) if Path(ref_repo_path).is_absolute() else _HERE.parents[1] / ref_repo_path
        self._ckpt_arg = ckpt
        self._ckpt_resolved: Optional[str] = None
        self._vocab = vocab
        self._length = int(length)
        self._block_size = int(block_size)
        self._steps = int(steps)
        self._nucleus = float(nucleus_p)
        self._temperature = float(temperature)
        self._first_hitting = bool(first_hitting)
        self._top1 = bool(top1)
        self._explicit_device = device
        self._device: Optional[str] = None
        self._upstream: Optional[dict] = None
        self._model = None
        self._tokenizer = None
        self._config = None
        self._using_fallback = False

    # ------------------------------------------------------------------
    def setup(self, device: Optional[str] = None) -> None:
        """Probe upstream + load checkpoint + build the SoftBD model.

        Heavy imports are guarded: if any required module is missing the
        adapter degrades to fallback mode (``_using_fallback = True``)
        so callers can detect via :meth:`get_metadata`.
        """
        self._device = device or ("cuda" if _torch_cuda_available() else "cpu")
        self._upstream = _try_import_upstream()
        if self._upstream is None:
            self._using_fallback = True
            logger.info("SoftMolAdapter running in fallback mode (upstream unavailable)")
            return

        # Build the Hydra-style config (mirror of sample._build_config)
        try:
            self._config = self._build_config()
        except Exception as exc:
            logger.warning("SoftMol config build failed: %s", exc)
            self._using_fallback = True
            return

        # Build the tokenizer
        try:
            self._tokenizer = self._upstream["dataloader"].get_tokenizer(self._config)
        except Exception as exc:
            logger.warning("SoftMol tokenizer build failed: %s", exc)
            self._using_fallback = True
            return

        # Resolve checkpoint (with HF fallback)
        self._ckpt_resolved = _resolve_ckpt(self._ckpt_arg)
        if not self._ckpt_resolved:
            logger.warning(
                "SoftMol checkpoint unavailable (offline + no local ckpt). "
                "Falling back to honest empty generate()."
            )
            self._using_fallback = True
            return

        # Load the LightningModule
        try:
            self._model = self._upstream["diffusion"].Diffusion.load_from_checkpoint(
                self._ckpt_resolved,
                tokenizer=self._tokenizer,
                config=self._config,
                strict=False,
                weights_only=False,
            ).to(self._device)
        except Exception as exc:
            logger.warning("SoftMol model load failed: %s", exc)
            self._using_fallback = True
            return

    def _build_config(self):
        """Hydra-free replica of ``SoftMol.sample._build_config``.

        We don't import hydra here — too heavy.  Instead we merge the
        same YAMLs (sample.yaml + model/<name>.yaml + overrides) using
        OmegaConf directly.  If omegaconf is missing, we fall back.
        """
        try:
            import omegaconf  # type: ignore
        except Exception:
            logger.debug("omegaconf missing — SoftMolAdapter config unavailable")
            raise
        repo = _resolve_repo()
        if repo is None:
            raise RuntimeError("SoftMol cloned repo not on disk")
        base_cfg = omegaconf.OmegaConf.load(str(repo / "configs" / "sample.yaml"))
        model_yaml = repo / "configs" / "model" / "small.yaml"
        if not model_yaml.is_file():
            model_yaml = repo / "configs" / "model" / "small-89M.yaml"
        model_cfg = omegaconf.OmegaConf.load(str(model_yaml))
        sampling_cfg = {
            "logdir": str((repo / "sample_logs" / "adapter_").resolve()),
            "nucleus_p": self._nucleus,
            "first_hitting": self._first_hitting,
            "top1": self._top1,
            "prefix": "",
            "next_block_only": False,
            "temperature": self._temperature,
        }
        overrides = omegaconf.OmegaConf.create({
            "seed": 42,
            "block_size": self._block_size,
            "algo": {"T": self._steps},
            "model": {"length": self._length, "attn_backend": "sdpa"},
            "loader": {"eval_batch_size": 1},
            "sampling": sampling_cfg,
            "eval": {"checkpoint_path": str(self._ckpt_resolved or _SOFTMOL_DEFAULT_CKPT)},
            "data": {"tokenizer_name_or_path": str((repo / self._vocab).resolve())},
        })
        merged = omegaconf.OmegaConf.merge(base_cfg, {"model": model_cfg}, overrides)
        return merged

    # ------------------------------------------------------------------
    def generate(
        self,
        pocket: Optional[Pocket],
        config: GenerationConfig,
    ) -> List[Molecule]:
        """Sample ``config.n_samples`` SMILES via SoftBD block-diffusion.

        ``pocket`` is currently IGNORED — SoftMol's published 5-target
        pipeline uses ``gated_mcts/run_mcts.py`` for pocket conditioning,
        which is NOT in scope of this adapter.  The adapter exposes
        *unconditional* generation; a future wrapper that fronts the
        gated-MCTS path can be added separately (the channel is honest-
        framed per ``party-twenty-spirit §3.5``: we do NOT claim pocket-
        conditioning support here).

        Returns a list of :class:`Molecule` whose ``smiles`` is the raw
        SoftMol output.  3D coords are not generated by SoftMol; we set
        ``coords = atom_types = bonds = empty tensors`` and rely on the
        caller to round-trip through RDKit if 3D is needed.
        """
        if self._using_fallback or self._model is None:
            return []
        try:
            n_samples = int(config.n_samples)
        except Exception:
            n_samples = 1
        try:
            raw_texts = self._model.restore_model_and_sample(num_steps=self._steps)
        except Exception as exc:
            logger.warning("SoftMol restore_model_and_sample failed: %s", exc)
            return []
        # Truncate to exact count + strip [BOS]/[EOS] markers
        cleaned = self._clean_text(raw_texts[:n_samples])
        out: List[Molecule] = []
        for smi in cleaned:
            out.append(self._smiles_to_molecule(smi))
        return out

    @staticmethod
    def _clean_text(samples: List[str]) -> List[str]:
        drop = ("[BOS]", "[EOS]")
        out = []
        for s in samples:
            t = s
            for tk in drop:
                t = t.replace(tk, "")
            out.append(t)
        return out

    @staticmethod
    def _smiles_to_molecule(smi: str) -> Molecule:
        """Build a :class:`Molecule` from a raw SoftMol SMILES string.

        SoftMol emits only SMILES — no coords, no atom-types, no bonds
        (in the tensor sense).  We therefore return a degenerate
        ``Molecule`` whose tensors are empty (shape (0, 3) etc.).  Any
        downstream consumer that wants 3D coords can call
        :func:`molmetal.domain.Molecule.from_smiles` to round-trip via
        RDKit ETKDGv3.
        """
        import torch  # local import — heavy
        empty3 = torch.zeros((0, 3), dtype=torch.float32)
        empty1 = torch.zeros((0,), dtype=torch.long)
        return Molecule(
            coords=empty3,
            atom_types=empty1,
            bonds=empty1.reshape(2, 0) if empty1.numel() else torch.zeros((2, 0), dtype=torch.long),
            bond_types=empty1,
            formal_charges=empty1,
            smiles=str(smi),
        )

    # ------------------------------------------------------------------
    def train_step(self, pocket: Optional[Pocket], mols: List[Molecule]) -> float:
        """No-op — SoftMol's training loop lives upstream in ``main.py``.

        Returns 0.0 to satisfy the :class:`MoleculeGenerator` protocol
        contract (cf. RxnFlowAdapter.train_step).
        """
        return 0.0

    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "paper": "Yang et al. 2026, arXiv:2601.21964",
            "code": "molmetal/references/SoftMol/",
            "weight_repo": _SOFTMOL_HF_REPO,
            "uses_3d": False,             # SMILES-only output
            "uses_pocket_conditioning": False,  # gated-MCTS not wired
            "vocab": self._vocab,
            "length": self._length,
            "block_size": self._block_size,
            "steps": self._steps,
            "nucleus_p": self._nucleus,
            "temperature": self._temperature,
            "first_hitting": self._first_hitting,
            "top1": self._top1,
            "ckpt_arg": self._ckpt_arg,
            "ckpt_resolved": self._ckpt_resolved,
            "fallback_active": self._using_fallback,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def list_softmol_configs() -> List[str]:
    """Return the names of the upstream ``configs/model/*.yaml`` files.

    Useful for letting callers choose between small / medium / large /
    tiny / 50M / 70M / 89M / 110M model configs without hard-coding.
    Empty list if the cloned repo is missing.
    """
    repo = _resolve_repo()
    if repo is None:
        return []
    cfg_dir = repo / "configs" / "model"
    if not cfg_dir.is_dir():
        return []
    return sorted(p.stem for p in cfg_dir.glob("*.yaml"))


__all__ = [
    "SoftMolAdapter",
    "is_softmol_available",
    "list_softmol_configs",
]
