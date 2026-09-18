"""Lightweight optional synthesis gate, usable without the search/GPU stack."""
from pathlib import Path


def build_synthesis_gate(request, config_path=None):
    """Return (checker, metadata); learned mode never falls back to SMARTS."""
    if request is None or request is False or request == 0:
        return None, {"status": "disabled", "backend": None, "applied": False}
    if request == "smarts":
        from molmetal_lam.sbdd_env.aizynth_adapter import _smarts_fallback
        return _smarts_fallback, {"status": "ready", "backend": "smarts_heuristic",
                                 "learned": False, "applied": True}
    if request is not True and request not in ("aizynthfinder", "aizynthfinder_isolated"):
        return None, {"status": "unsupported_request", "backend": None, "applied": False}
    if config_path is None or not Path(config_path).is_file():
        return None, {"status": "missing_aizynth_config", "backend": "aizynthfinder",
                      "applied": False, "learned": True}
    if request == "aizynthfinder_isolated":
        from .aizynth_isolated import IsolatedAiZynthChecker
        checker = IsolatedAiZynthChecker(config_path)
        metadata = {"backend": "aizynthfinder", "learned": True,
                    "execution": "isolated_process", "seed": checker.seed}
        try:
            details = checker.probe()
        except Exception as exc:
            return None, {**metadata, "status": "aizynth_models_unavailable", "applied": False,
                          "error": f"{type(exc).__name__}: {exc}"}
        return checker, {**metadata, "status": "ready", "applied": True, "adapter": details}
    from molmetal_lam.sbdd_env.aizynth_adapter import AiZynthAdapter
    adapter = AiZynthAdapter(config_path=str(config_path), max_iterations=200, time_limit_s=30)
    if adapter._finder is None:
        return None, {"status": "aizynth_models_unavailable", "backend": "aizynthfinder",
                      "applied": False, "learned": True,
                      "error": getattr(adapter, "_load_error", None)}
    return adapter.check, {"status": "ready", "backend": "aizynthfinder",
                           "applied": True, "learned": True}


def gate_candidates(candidates, checker, metadata):
    if checker is None:
        return candidates, {**metadata, "n_checked": 0, "n_passed": 0, "reports": []}
    passed, reports = [], []
    batch_results = None
    batch_error = None
    if hasattr(checker, "check_many"):
        try:
            batch_results = checker.check_many([c.canonical_smiles() for c in candidates])
            if len(batch_results) != len(candidates):
                raise ValueError("Synthesis batch report count mismatch")
        except Exception as exc:
            batch_error = exc
    for index, candidate in enumerate(candidates):
        smiles = candidate.canonical_smiles()
        try:
            if batch_error is not None:
                raise batch_error
            result = batch_results[index] if batch_results is not None else checker(smiles)
            correct_engine = result.engine == ("aizynthfinder" if metadata["learned"] else "smarts_fallback")
            verdict = bool(correct_engine and result.synthesizable)
            report = {"smiles": smiles, "passed": verdict, "engine": result.engine,
                      "depth": result.depth,
                      "route_smiles": list(getattr(result, "route_smiles", ())),
                      "status": "checked" if correct_engine else "backend_fallback_rejected"}
        except Exception as exc:
            verdict = False
            report = {"smiles": smiles, "passed": False, "status": "error", "error": str(exc)}
        reports.append(report)
        if verdict:
            passed.append(candidate)
    if isinstance(getattr(checker, "metadata", None), dict):
        metadata = {**metadata, "adapter": checker.metadata}
    return passed, {**metadata, "n_checked": len(reports), "n_passed": len(passed), "reports": reports}
