"""Real AMD prior RPC smoke, including invalid inputs and batch invariance."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from molmetal.molmetal_lam.sbdd_env.reinvent_prior_adapter import REINVENT4PriorAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("molmetal/configs/reinvent_prior_amd.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("molmetal/reports/reinvent_prior_adapter_amd"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "scope": "real pretrained AMD likelihood RPC; no multiproperty reward integration"}
    client = REINVENT4PriorAdapter.from_config(args.config)
    client.log_path = str(args.out_dir / "worker.log")
    report["config"] = json.loads(args.config.read_text())
    try:
        with client:
            inputs = ["CCO", "c1ccccc1", "", "invalid", None, "[Xe]", "CCO"]
            rows = client.likelihood(inputs)
            report["batch_results"] = [asdict(row) for row in rows]
            report["batch_metadata"] = client.last_metadata
            report["batch_error"] = client.last_error
            pid = client._proc.pid if client._proc else None
            single = client.likelihood(["CCO"])
            report["single_result"] = [asdict(row) for row in single]
            report["same_worker_reused"] = bool(pid and client._proc and client._proc.pid == pid)
            all_invalid = client.likelihood(["", "invalid", None, "[Xe]"])
            report["all_invalid_results"] = [asdict(row) for row in all_invalid]
            report["all_invalid_metadata"] = client.last_metadata
            report["empty_batch"] = client.likelihood([])
        good = [row for row in rows if row.status == "ok"]
        delta = (abs(rows[0].prior_nll - single[0].prior_nll)
                 if rows and single and rows[0].prior_nll is not None and single[0].prior_nll is not None else None)
        report["batch_vs_single_absolute_nll_delta"] = delta
        report["n_requested"], report["n_scored"], report["n_failed"] = len(inputs), len(good), len(rows)-len(good)
        report["passed"] = (
            len(rows) == 7 and [r.index for r in good] == [0, 1, 6]
            and rows[0].prior_nll == rows[6].prior_nll
            and delta is not None and delta < 1e-4 and report["same_worker_reused"]
            and len(all_invalid) == 4 and all(r.status == "error" for r in all_invalid)
            and report["all_invalid_metadata"].get("neural_forwards") == []
            and report["empty_batch"] == []
            and report["batch_metadata"].get("device") == "cuda:0"
            and report["batch_metadata"].get("architecture") == "gfx1101")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}:{exc}"
        client.close()
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ("passed", "error", "batch_error", "n_requested", "n_scored", "n_failed", "batch_vs_single_absolute_nll_delta")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
