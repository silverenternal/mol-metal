import gzip, sys, re
CACHE="/home/hugo/codes/mol-metal/molmetal/reports/wfa175_frontier_comparability/cache/"
def load(arxiv):
    p=CACHE+arxiv+".txt.gz"
    try:
        raw=gzip.open(p,'rt',encoding='utf-8',errors='replace').read()
    except Exception as e:
        return None
    files={}
    cur=None; buf=[]
    for line in raw.split("\n"):
        m=re.match(r'^=====FILE (.+)=====$', line)
        if m:
            if cur: files[cur]="\n".join(buf)
            cur=m.group(1); buf=[]
        elif cur is not None:
            buf.append(line)
    if cur: files[cur]="\n".join(buf)
    return files
if __name__=="__main__":
    arxiv=sys.argv[1]; f=sys.argv[2]
    lo=int(sys.argv[3]); hi=int(sys.argv[4])
    files=load(arxiv)
    if files is None: print("NOFILE"); sys.exit(1)
    if f not in files:
        print("MISSING FILE. available:", list(files)[:40]); sys.exit(2)
    lines=files[f].split("\n")
    for i in range(lo, min(hi+1, len(lines)+1)):
        print(f"{f}:{i}: {lines[i-1]}")
