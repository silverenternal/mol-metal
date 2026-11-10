"""Wf-A175 refutation -- independent re-measurement of the BAND census.

READ-ONLY on the repo. Writes only inside this directory. CPU-only.
No os.walk: enumeration via `git ls-files` only (per repo memory note).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/home/hugo/codes/mol-metal")
OUT = ROOT / "molmetal/reports/wfa175_band_provenance/refutation"

CANON = re.compile(r"92\.7\s*[–—-]\s*94")          # catalog §6.266.8 method
BAND = re.compile(r"92\.7\s*[-‐‑‒–—―~]\s*94")     # BAND branch v1_dash_any


def git_ls_files(extra):
    p = subprocess.run(["git", "ls-files", "-z", *extra], cwd=str(ROOT),
                       capture_output=True, check=True)
    return [x for x in p.stdout.decode("utf-8", "replace").split("\0") if x]


def population():
    out = git_ls_files(["--cached"])
    seen = set(out)
    for p in git_ls_files(["--others", "--exclude-standard"]):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def read_safe(rel):
    p = ROOT / rel
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    return raw.decode("utf-8", "replace")


def census(rx):
    per = {}
    for rel in population():
        text = read_safe(rel)
        if text is None:
            continue
        nlines = 0
        nocc = 0
        for line in text.splitlines():
            ms = rx.findall(line)
            if ms:
                nlines += 1
                nocc += len(ms)
        if nocc:
            per[rel] = (nlines, nocc)
    files = len(per)
    lines = sum(v[0] for v in per.values())
    occ = sum(v[1] for v in per.values())
    return {"files": files, "lines": lines, "occurrences": occ, "per": per}


def main():
    res = {"canonical_regex": census(CANON), "band_regex": census(BAND)}
    # split out the two Wf-A171 reports
    newf = [f for f in res["band_regex"]["per"] if "/wfa171_" in f]
    per_b = res["band_regex"]["per"]
    res["wfa171_files"] = {f: res["band_regex"]["per"][f] for f in newf}
    res["band_minus_wfa171"] = {
        "files": res["band_regex"]["files"] - len(newf),
        "lines": res["band_regex"]["lines"] - sum(per_b[f][0] for f in newf),
        "occurrences": res["band_regex"]["occurrences"] - sum(per_b[f][1] for f in newf),
    }
    # canonical table from docs/HONEST_NEGATIVE_CATALOG.md §6.266.8
    canon_table = {
        "molmetal/reports/wfa156_bond_threshold_frontier/report.md": 1,
        "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md": 4,
        "molmetal/reports/wfa158_population_reconciliation/census.md": 1,
        "molmetal/reports/wfa161_frontier_labelling/report.md": 1,
        "molmetal/reports/wfa161_paper_unqualified_numbers/report.md": 1,
        "molmetal/reports/wfa164_bond_threshold_narrative/report.md": 1,
        "molmetal/reports/wfa164_bond_threshold_narrative/c7_pool_check/report.md": 1,
        "molmetal/reports/wfa164_c_refutation/refutation.md": 2,
        "molmetal/reports/wfa164_coordination_sphere_refutation/refutation.md": 2,
        "molmetal/reports/wfa164_posebusters_narrative/pre_repair_script_snapshot.py.txt": 1,
        "molmetal/reports/wfa164_synthesis/synthesis.md": 3,
        "molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md": 7,
        "molmetal/scripts/wf_a155_ligand_connected_rate.py": 3,
        "molmetal/scripts/wf_a155_metal_coordination_sphere.py": 3,
        "molmetal/scripts/wfa156_bond_threshold_frontier.py": 2,
        "molmetal/scripts/wfa158_frontier_postproc_matrix.py": 6,
        "molmetal/scripts/wfa158_population_reconciliation.py": 3,
        "molmetal/scripts/wfa161_frontier_labelling.py": 2,
        "molmetal/scripts/wfa161_paper_unqualified_numbers.py": 2,
        "molmetal/scripts/wfa163_narrative_truth_scan.py": 1,
        "molmetal/tests/test_wfa155_decoder_bond_candidacy.py": 1,
        "molmetal/tests/test_wfa155_ligand_connected_rate.py": 2,
        "molmetal/tests/test_wfa158_frontier_postproc_matrix.py": 1,
        "molmetal/tests/test_wfa164_coordination_sphere_narrative.py": 2,
        "metrics/by_metric/wfa155_ligand_connected_rate.json": 3,
        "metrics/by_metric/wfa155_metal_coordination_sphere.json": 2,
        "metrics/by_metric/wfa155_posebusters_validity.json": 1,
        "docs/HONEST_NEGATIVE_CATALOG.md": 8,
        "AGENTS.md": 3,
        "TODO/INDEX.md": 1,
        "TODO/README.md": 1,
        "molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py": 2,
    }
    res["canonical_table_files"] = len(canon_table)
    res["canonical_table_sum"] = sum(canon_table.values())
    can_per = res["canonical_regex"]["per"]
    diff = []
    for f, n in canon_table.items():
        got_lines, got_occ = can_per.get(f, (0, 0))
        if got_lines != n:
            diff.append({"path": f, "canonical_table": n,
                         "today_lines": got_lines, "today_occurrences": got_occ})
    res["canonical_vs_today_diffs"] = diff
    # extra files today not in the canonical table
    res["extra_today"] = {f: can_per[f] for f in can_per if f not in canon_table}
    (OUT / "_recheck.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "canonical_regex"}, indent=2,
                     ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
