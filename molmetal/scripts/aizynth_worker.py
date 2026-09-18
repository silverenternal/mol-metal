#!/usr/bin/env python3
"""One bounded JSON batch for the isolated learned AiZynth process."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "molmetal")]


def main():
    request = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        import numpy as np
        from molmetal_lam.sbdd_env.aizynth_adapter import AiZynthAdapter
        seed = int(request.get("seed", 20260913))
        random.seed(seed)
        np.random.seed(seed)
        adapter = AiZynthAdapter(
            request["config_path"], max_iterations=int(request.get("max_iterations", 200)),
            time_limit_s=int(request.get("time_limit_s", 30)),
        )
        ready = adapter._finder is not None
        reports = [report.to_dict() for report in adapter.check_list(request.get("smiles", []))] if ready else []
        response = {"schema_version": 1, "ready": ready, "metadata": adapter.get_metadata(),
                    "reports": reports, "seed": seed}
    print(json.dumps(response))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
