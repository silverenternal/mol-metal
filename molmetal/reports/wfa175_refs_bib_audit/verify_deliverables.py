#!/usr/bin/env python3
"""Wf-A175 — self-check: (1) every VERBATIM quoted original really appears in paper/refs.bib;
(2) every corrected entry's new DOI resolves via the doi.org handle registry;
(3) the 3 required keys are present in the STAGED file; (4) detect_truncation on both files."""
from __future__ import annotations
import json, re, sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
RD = REPO / "molmetal" / "reports" / "wfa175_refs_bib_audit"
sys.path.insert(0, str(REPO / "molmetal" / "scripts"))
from wf386a3_atomic_write import detect_truncation  # noqa: E402

BIB = REPO / "paper" / "refs.bib"
STAGED = RD / "refs_corrected_STAGED.bib"
REPORT = RD / "report.md"

src = BIB.read_text(encoding="utf-8", errors="replace")
staged = STAGED.read_text(encoding="utf-8")


def entries(text):
    out = {}
    for m in re.finditer(r"^@(\w+)\{([^,\n]+),", text, re.M):
        ob = text.index("{", m.start())
        depth, j = 0, ob
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[m.group(2).strip()] = text[m.start():j + 1]
    return out


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def decomment(text):
    """Strip BibTeX comment prefixes ('% ' / '%') so quoted originals can be matched."""
    out = []
    for line in text.splitlines():
        t = line.strip()
        if t.startswith("%"):
            t = t[1:]
            if t.startswith(" "):
                t = t[1:]
            out.append(t)
        else:
            out.append(line)
    return "\n".join(out)


src_e = entries(src)
staged_e = entries(staged)
print("source entries:", len(src_e), "| staged entries:", len(staged_e))
print("staged cite keys unique:", len(staged_e) == len(re.findall(r"^@\w+\{", staged, re.M)))

# --- 1. VERBATIM quoted originals must exist in paper/refs.bib -----------------
QUOTED = [
    "rosettafold_aa2024", "attentivefp2020", "posebusters2023", "buttenschoen2024posebusters",
    "fontana2006", "footnote:rosettafoldaa", "footnote:attentivefp", "footnote:fontana2006",
    "footnote:fep_metallo", "footnote:metallodrug_review", "openmm",
]
staged_decommented = norm(decomment(staged))
bad = []
for k in QUOTED:
    o = src_e.get(k)
    if o is None:
        bad.append((k, "NOT IN paper/refs.bib"))
        continue
    body = norm(o)
    body_no_brace = body[:-1].strip() if body.endswith("}") else body
    if norm(body_no_brace) not in staged_decommented:
        bad.append((k, "verbatim quote NOT found in STAGED"))
    else:
        print(f"  OK verbatim quoted: {k}")
print("VERBATIM-CHECK failures:", bad if bad else "NONE")

# --- 2. STAGED DOIs: registered? --------------------------------------------
import requests  # noqa: E402
H = {"User-Agent": "molmetal-wfa175-verify/1.0"}
s = requests.Session()
dois = []
for k, v in staged_e.items():
    m = re.search(r"doi\s*=\s*\{([^}]+)\}", v, re.I)
    if m:
        dois.append((k, m.group(1).strip()))
print("staged entries with doi:", len(dois))
for k, d in dois:
    try:
        r = s.get("https://doi.org/api/handles/" + d, headers=H, timeout=45)
        rc = r.json().get("responseCode")
    except Exception as e:  # pragma: no cover
        rc = f"ERR {e}"
    flag = "OK " if rc == 1 else "NOT-REGISTERED"
    print(f"  {flag} {k}: {d} (responseCode={rc})")

# --- 3. required keys present in STAGED -------------------------------------
for k in ("krasnov2026metalcytotoxdb", "rusanov2025platinai", "ji2022drugood"):
    print(f"required key in STAGED: {k} -> {'YES' if k in staged_e else 'NO'}")

# --- 4. truncation detection ------------------------------------------------
for p in (STAGED, RD / "refs_sci_paper_corrected_STAGED.bib", REPORT):
    t = p.read_text(encoding="utf-8")
    print(f"truncation {p.name}: {detect_truncation(t)}")
print("staged bytes:", STAGED.stat().st_size, "| report bytes:", REPORT.stat().st_size)

# --- 5. sci-paper STAGED file: verbatim originals + registered DOIs ----------
sp = RD / "refs_sci_paper_corrected_STAGED.bib"
sp_src = (REPO / ".tools" / "sci-paper" / "refs" / "references.bib").read_text(encoding="utf-8", errors="replace")
sp_e = entries(sp_src)
sp_decommented = norm(decomment(sp.read_text(encoding="utf-8")))
for k in ("krasnov2026metalcytotoxdb", "rusanov2025platinai"):
    o = norm(sp_e[k])
    body = o[:-1].strip() if o.endswith("}") else o
    print(f"  sci-paper verbatim quoted: {k} -> {'OK' if body in sp_decommented else 'MISSING'}")
    m = re.search(r"doi\s*=\s*\{([^}]+)\}", sp_e[k], re.I)
    r = s.get("https://doi.org/api/handles/" + m.group(1).strip(), headers=H, timeout=45)
    print(f"    original doi {m.group(1).strip()} responseCode={r.json().get('responseCode')}")
sp_staged = entries(sp.read_text(encoding="utf-8"))
for k, v in sp_staged.items():
    m = re.search(r"doi\s*=\s*\{([^}]+)\}", v, re.I)
    r = s.get("https://doi.org/api/handles/" + m.group(1).strip(), headers=H, timeout=45)
    print(f"  sci-paper staged {k}: {m.group(1).strip()} responseCode={r.json().get('responseCode')}")
