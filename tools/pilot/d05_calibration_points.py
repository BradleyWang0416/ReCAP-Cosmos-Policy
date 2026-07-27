"""NONFORMAL_DIAGNOSTIC (read-only): find the right human->robot correspondence point."""
import json, h5py, numpy as np, os
np.set_printoptions(precision=4, suppress=True)
SPLIT="/home/wxs/ReCAP-Cosmos-Policy/data/Human2Robot/derived/v04/source_split_manifest.json"
SRC="/DATA1/wxs/DATASETS/Human2Robot/data/v1"
rec=json.load(open(SPLIT))["records"]
seen=[r for r in rec if r["source_partition"]=="seen_train"]
by={}
for r in sorted(seen,key=lambda x:x["source_sha256"]): by.setdefault(r["task"],[]).append(r)
sample=[r for t in sorted(by) for r in by[t][:3]]

def load(r):
    with h5py.File(os.path.join(SRC,r["source_relative_path"]),"r") as f:
        return (np.asarray(f["transformed_hand_frames"][:],dtype=np.float64),
                np.asarray(f["transformed_hand_coords"][:],dtype=np.float64),
                np.asarray(f["end_position"][:,:3],dtype=np.float64)*1e-3)

def umeyama(P,Q):
    mp,mq=P.mean(0),Q.mean(0); X,Y=P-mp,Q-mq
    S=X.T@Y/len(P); U,D,Vt=np.linalg.svd(S); W=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: W[2,2]=-1
    R=Vt.T@W@U.T; s=float(np.trace(np.diag(D)@W)/X.var(0).sum()); t=mq-s*R@mp
    return s,R,t

def fit_stats(points_fn, label, lag=0):
    Ps,Qs,perep=[],[],[]
    for r in sample:
        hf,hc,ep=load(r)
        P=points_fn(hf,hc); Q=ep
        if lag>0: P,Q=P[:-lag],Q[lag:]
        elif lag<0: P,Q=P[-lag:],Q[:lag]
        Ps.append(P); Qs.append(Q)
        si,Ri,ti=umeyama(P,Q)
        perep.append(np.linalg.norm((si*(Ri@P.T).T+ti)-Q,axis=1).mean()*1e3)
    P=np.concatenate(Ps); Q=np.concatenate(Qs)
    s,R,t=umeyama(P,Q); res=np.linalg.norm((s*(R@P.T).T+t)-Q,axis=1)
    r2=1-((s*(R@P.T).T+t-Q)**2).sum()/((Q-Q.mean(0))**2).sum()
    print("  %-34s lag=%+d  global: R2=%.3f res=%5.1fmm scale=%.3f | per-ep res=%5.1fmm"
          %(label,lag,r2,res.mean()*1e3,s,np.mean(perep)))
    return r2, np.mean(perep)

I,J=16,20
defs={
 "wrist origin (row0)":            lambda hf,hc: hf[:,0],
 "world-rel fingertip mid (16,20)":lambda hf,hc: hf[:,0]+0.5*(hc[:,I]+hc[:,J]),
 "hand-local fingertip mid":       lambda hf,hc: hf[:,0]+np.einsum('tij,tj->ti',np.transpose(hf[:,1:4],(0,2,1)),0.5*(hc[:,I]+hc[:,J])),
 "hand-local fingertip mid (R^T)": lambda hf,hc: hf[:,0]+np.einsum('tij,tj->ti',hf[:,1:4],0.5*(hc[:,I]+hc[:,J])),
 "world-rel palm (pt 2)":          lambda hf,hc: hf[:,0]+hc[:,2],
}
print("== correspondence-point search (48 seen-train episodes) ==")
best=None
for lab,fn in defs.items():
    r2,pe=fit_stats(fn,lab,0)
    if best is None or r2>best[0]: best=(r2,lab,fn)
print("\n== temporal lag scan on best point: %s =="%best[1])
for lag in (-6,-3,-1,0,1,3,6):
    fit_stats(best[2],"lag scan",lag)
