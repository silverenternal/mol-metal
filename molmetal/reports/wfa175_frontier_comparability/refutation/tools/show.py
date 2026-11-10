import sys, gzip, re
CACHE="/home/hugo/codes/mol-metal/molmetal/reports/wfa175_frontier_comparability/cache/"
def load(a):
    raw=gzip.open(CACHE+a+".txt.gz",'rt',encoding='utf-8',errors='replace').read()
    files={};cur=None;buf=[]
    for line in raw.split("\n"):
        m=re.match(r'^=====FILE (.+)=====$',line)
        if m:
            if cur: files[cur]="\n".join(buf)
            cur=m.group(1);buf=[]
        elif cur is not None: buf.append(line)
    if cur: files[cur]="\n".join(buf)
    return files
def show(a,f,lo,hi):
    fs=load(a)
    if f not in fs:
        print(f"  !! MISSING {f}; have: {sorted(fs)[:25]}"); return
    L=fs[f].split("\n")
    for i in range(lo,min(hi+1,len(L)+1)): print(f"  {f}:{i}: {L[i-1][:400]}")
if __name__=="__main__":
    # args: arxiv:file:lo:hi  (repeatable)
    for spec in sys.argv[1:]:
        a,f,lo,hi=spec.split(":")
        print(f"##### {a} {f} {lo}-{hi}")
        show(a,f,int(lo),int(hi))
