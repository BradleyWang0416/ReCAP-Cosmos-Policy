"""NONFORMAL_DIAGNOSTIC (read-only, CPU): retrieval ablation on the frozen 160 dev queries.
Metric = |robot_future - aligned_plan| position error, vs the no-retrieval baseline."""
import numpy as np, pickle, itertools, json
D=pickle.load(open("/tmp/h2r_abl.pkl","rb")); pool=D["pool"]; queries=D["queries"]
H=8
def seg_ok(seg,s,n): 
    w=seg[s:s+n]; return len(w)==n and w.min()==w.max()
def cand_table(task,K,channel):
    """all legal candidate windows of the task's pool, valid for horizon K"""
    G=[];V=[];P=[];CH=[];EP=[]
    for ei,e in enumerate(pool[task]):
        X=e["A"] if channel=="A" else e["B"]
        gf=e["gA"] if channel=="A" else e["gB"]
        pf=e["posA"] if channel=="A" else e["posB"]
        m=np.array([ (s+H+K)<=len(X) and seg_ok(e["seg"],s,H+K) for s in e["starts"] ])
        if not m.any(): continue
        idx=e["starts"][m][:,None]+np.arange(H,H+K)[None,:]
        CH.append(X[idx][:,:,:3]); G.append(gf[m]); V.append(e["V"][m]); P.append(pf[m][:,:3])
        EP.append(np.full(m.sum(),ei))
        gidx=e["starts"][m][:,None]+np.arange(H,H+K)[None,:]
        if channel=="A": pass
    return (np.concatenate(G),np.concatenate(V),np.concatenate(P),
            np.concatenate(CH),np.concatenate(EP))
def phase_mask(task,K,q):
    """the frozen rule: one window per episode, closest normalised phase"""
    keep=[];off=0
    qph=(q["start"]+H)/q["frames"]
    for e in pool[task]:
        X=e["A"]; m=np.array([ (s+H+K)<=len(X) and seg_ok(e["seg"],s,H+K) for s in e["starts"] ])
        n=m.sum()
        if n:
            ph=(e["starts"][m]+H)/e["frames"]
            keep.append(off+int(np.argmin(np.abs(ph-qph))))
        off+=n
    return np.array(keep,dtype=int)
def resid(chunks,cur,tgt):
    al=cur[None,None,:3]+chunks-chunks[:,0:1,:]
    return np.linalg.norm(al-tgt[None,:,:3],axis=2).mean(1)*1e3

CELLS=[(c,w,f) for c in ("A","B") for w in ("phase","search") for f in ("geom","vis","geom+vis","abs-pos")]
for K in (8,16,32,64):
    acc={c:[] for c in CELLS}; acc3={c:[] for c in CELLS}
    hold=[];orc=[];n=0
    for q in queries:
        R=q["R"]; qs=q["start"]
        if qs+H+K>len(R) or not seg_ok(q["seg"],qs,H+K): continue
        cur=R[qs+H-1]; tgt=R[qs+H:qs+H+K]; n+=1
        hold.append(np.linalg.norm(tgt[:,:3]-cur[:3],axis=1).mean()*1e3)
        best=np.inf
        for ch in ("A","B"):
            G,V,P,CH,EP=cand_table(q["task"],K,ch)
            if len(G)==0: continue
            r=resid(CH,cur,tgt)
            if ch=="A": best=min(best,float(r.min()))
            dg=np.linalg.norm(G-(q["gA"] if ch=="A" else q["gB"]),axis=1)
            dv=np.linalg.norm(V-q["V"],axis=1)
            dc=np.linalg.norm(np.concatenate((G,V),axis=1)/np.sqrt(2)
                              -np.concatenate((q["gA"] if ch=="A" else q["gB"],q["V"]))/np.sqrt(2),axis=1)
            dp=np.linalg.norm(P-cur[None,:3],axis=1)
            dist={"geom":dg,"vis":dv,"geom+vis":dc,"abs-pos":dp}
            pm=phase_mask(q["task"],K,q)
            for f in ("geom","vis","geom+vis","abs-pos"):
                for w in ("phase","search"):
                    sub=pm if w=="phase" else np.arange(len(G))
                    if len(sub)==0: continue
                    o=sub[np.argsort(dist[f][sub],kind="stable")]
                    acc[(ch,w,f)].append(r[o[0]])
                    acc3[(ch,w,f)].append(r[o[:3]].mean())
        orc.append(best)
    hb=np.mean(hold)
    print("\n"+"="*86)
    print("K = %-3d (≈%.2f s @30fps)   有效 query %d   无检索基线 %.1f mm   oracle 上界 %.1f mm"
          %(K,K/30.0,n,hb,np.mean(orc)))
    print("="*86)
    print("  %-9s %-7s %-9s %10s %10s %9s"%("通道","窗口","特征","top1 mm","top3均 mm","vs基线"))
    rows=[]
    for c in CELLS:
        if not acc[c]: continue
        m1,m3=np.mean(acc[c]),np.mean(acc3[c]); rows.append((m1,c,m1,m3,m1/hb))
    for _,c,m1,m3,rr in sorted(rows):
        flag=" ✅" if rr<1.0 else ""
        print("  %-9s %-7s %-9s %10.1f %10.1f %8.2fx%s"%(c[0],c[1],c[2],m1,m3,rr,flag))
