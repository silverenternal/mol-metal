#!/usr/bin/env python3
"""Independent band census: HEAD tree (all blobs) + working tree, narrow + broad regex."""
import json, os, re, subprocess, sys
ROOT="/home/hugo/codes/mol-metal"
sys.path.insert(0, os.path.join(ROOT,"molmetal","scripts"))
from wfa156_text_scan import iter_repo_files, scan

NARROW = re.compile(r"92\.7\s*(?:-|–|—|to)\s*94(?:\.0)?\s*%?", re.I)   # the branch's regex
BROAD   = re.compile(r"92[.,]7\b[^0-9\n]{0,40}?94(?:\.0)?\b", re.I)      # any separator incl "%"

out={}
# ---------- HEAD: ALL blobs in the HEAD tree ----------
head_paths = subprocess.run(["git","-C",ROOT,"ls-tree","-r","--name-only","HEAD"],capture_output=True,text=True).stdout.split("\n")
head_paths=[p for p in head_paths if p]
n_hits_n=[]; n_hits_b=[]
for p in head_paths:
    r=subprocess.run(["git","-C",ROOT,"show","HEAD:"+p],capture_output=True)
    if r.returncode!=0: continue
    t=r.stdout.decode("utf-8","replace")
    if NARROW.search(t): n_hits_n.append(p)
    if BROAD.search(t): n_hits_b.append(p)
out["head"]={"n_blobs":len(head_paths),"narrow_hits":len(n_hits_n),"broad_hits":len(n_hits_b),
             "narrow_list":n_hits_n,"broad_list":n_hits_b}

# ---------- working tree ----------
all_files=iter_repo_files(ROOT, untracked=True)
paper=[p for p in all_files if p.startswith("paper/")]
def census(paths, rx):
    hits=[]
    for r in scan(ROOT, rx.pattern, files=paths):
        if r.matches: hits.append(r.path)
    return hits
wn=census(all_files,NARROW); wb=census(all_files,BROAD)
pn=census(paper,NARROW);     pb=census(paper,BROAD)
out["working_tree"]={"n_files":len(all_files),"narrow":len(wn),"broad":len(wb),
                     "narrow_only":[p for p in wn if p not in wb],
                     "broad_only":[p for p in wb if p not in wn],
                     "broad_list":wb}
out["paper"]={"n_files":len(paper),"narrow":len(pn),"broad":len(pb),"broad_list":pb}
# paper/ files on disk that git enumeration may have missed
disk=[]
for dirpath,dirs,files in os.walk(os.path.join(ROOT,"paper")):
    for f in files:
        disk.append(os.path.relpath(os.path.join(dirpath,f),ROOT))
disk=sorted(disk)
missing=[p for p in disk if p not in set(all_files)]
out["paper_disk"]={"n_on_disk":len(disk),"n_in_git_enumeration":len(set(paper)),
                   "not_in_enumeration":missing}
json.dump(out, open("tools/census_indep.json","w"), indent=1)
print(json.dumps({k:{kk:(vv if not isinstance(vv,list) else len(vv)) for kk,vv in v.items()} for k,v in out.items()},indent=1)[:3000])
print("BROAD-ONLY working-tree files:", out["working_tree"]["broad_only"])
print("BROAD-ONLY paper files:", [p for p in out["paper"]["broad_list"] if p not in pn])
