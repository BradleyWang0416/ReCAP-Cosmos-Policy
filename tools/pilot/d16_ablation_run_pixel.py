"""NONFORMAL_DIAGNOSTIC: ablation incl. the dense pixel descriptor."""
import numpy as np, pickle
D=pickle.load(open("/tmp/h2r_abl2.pkl","rb")); pool=D["pool"]; queries=D["queries"]
H=8
def seg_ok(seg,s,n):
    w=seg[s:s+n]; return len(w)==n and w.min()==w.max()
def table(task,K,ch):
    G=[];V=[];P=[];X=[];PX=[]
    for e in pool[task]:
        S=e["A"] if ch=="A" else e["B"]
        gf=e["gA"] if ch=="A" else e["gB"]
        pf=e["posA"] if ch=="A" else e["posB"]
        m=np.array([(s+H+K)<=len(S) and seg_ok(e["seg"],s,H+K) for s in e["starts"]])
        if not m.any(): continue
        idx=e["starts"][m][:,None]+np.arange(H,H+K)[None,:]
        X.append(S[idx][:,:,:3]); G.append(gf[m]); V.append(e["V"][m])
        P.append(pf[m][:,:3]); PX.append(e["Pix"][m])
    return (np.concatenate(G),np.concatenate(V),np.concatenate(P),
            np.concatenate(X),np.concatenate(PX))
def pmask(task,K,q):
    keep=[];off=0; qph=(q["start"]+H)/q["frames"]
    for e in pool[task]:
        S=e["A"]; m=np.array([(s+H+K)<=len(S) and seg_ok(e["seg"],s,H+K) for s in e["starts"]])
        n=m.sum()
        if n:
            ph=(e["starts"][m]+H)/e["frames"]; keep.append(off+int(np.argmin(np.abs(ph-qph))))
        off+=n
    return np.array(keep,dtype=int)
def resid(chunks,cur,tgt):
    al=cur[None,None,:3]+chunks-chunks[:,0:1,:]
    return np.linalg.norm(al-tgt[None,:,:3],axis=2).mean(1)*1e3
FEATS=("geom","vis","geom+vis","abs-pos","pix","geom+pix")
CELLS=[(c,w,f) for c in ("A","B") for w in ("phase","search") for f in FEATS]
for K in (8,32,64):
    acc={c:[] for c in CELLS}; hold=[];orc=[];n=0
    for q in queries:
        R=q["R"]; qs=q["start"]
        if qs+H+K>len(R) or not seg_ok(q["seg"],qs,H+K): continue
        cur=R[qs+H-1]; tgt=R[qs+H:qs+H+K]; n+=1
        hold.append(np.linalg.norm(tgt[:,:3]-cur[:3],axis=1).mean()*1e3); best=np.inf
        for ch in ("A","B"):
            G,V,P,X,PX=table(q["task"],K,ch)
            if not len(G): continue
            r=resid(X,cur,tgt)
            if ch=="A": best=min(best,float(r.min()))
            gq=q["gA"] if ch=="A" else q["gB"]
            d={"geom":np.linalg.norm(G-gq,axis=1),
               "vis":np.linalg.norm(V-q["V"],axis=1),
               "geom+vis":np.linalg.norm(np.concatenate((G,V),1)-np.concatenate((gq,q["V"])),axis=1),
               "abs-pos":np.linalg.norm(P-cur[None,:3],axis=1),
               "pix":np.linalg.norm(PX-q["Pix"],axis=1),
               "geom+pix":np.linalg.norm(np.concatenate((G,PX),1)-np.concatenate((gq,q["Pix"])),axis=1)}
            pm=pmask(q["task"],K,q)
            for f in FEATS:
                for w in ("phase","search"):
                    sub=pm if w=="phase" else np.arange(len(G))
                    if not len(sub): continue
                    o=sub[np.argsort(d[f][sub],kind="stable")]
                    acc[(ch,w,f)].append(r[o[0]])
        orc.append(best)
    hb=np.mean(hold)
    print("\n"+"="*80)
    print("K=%-3d (≈%.2fs)  n=%d  无检索基线 %.1f mm  oracle %.1f mm"%(K,K/30.0,n,hb,np.mean(orc)))
    print("="*80)
    rows=sorted((np.mean(v),c) for c,v in acc.items() if v)
    for m,c in rows:
        print("  %-2s %-7s %-9s %8.1f mm  %6.2fx%s"%(c[0],c[1],c[2],m,m/hb," ✅" if m<hb else ""))
