"""NONFORMAL_DIAGNOSTIC: is the human stream actually paired/synchronised with the robot stream?"""
import json, h5py, numpy as np, os
SPLIT="/home/wxs/ReCAP-Cosmos-Policy/data/Human2Robot/derived/v04/source_split_manifest.json"
SRC="/DATA1/wxs/DATASETS/Human2Robot/data/v1"
rec=json.load(open(SPLIT))["records"]
seen=[r for r in rec if r["source_partition"]=="seen_train"]
by={}
for r in sorted(seen,key=lambda x:x["source_sha256"]): by.setdefault(r["task"],[]).append(r)
sample=[r for t in sorted(by) for r in by[t][:4]]
I,J=16,20
def load(r):
    with h5py.File(os.path.join(SRC,r["source_relative_path"]),"r") as f:
        hf=np.asarray(f["transformed_hand_frames"][:],dtype=np.float64)
        hc=np.asarray(f["transformed_hand_coords"][:],dtype=np.float64)
        ep=np.asarray(f["end_position"][:,:3],dtype=np.float64)*1e-3
    return hf[:,0]+0.5*(hc[:,I]+hc[:,J]), ep
def umeyama(P,Q):
    mp,mq=P.mean(0),Q.mean(0); X,Y=P-mp,Q-mq
    S=X.T@Y/len(P); U,D,Vt=np.linalg.svd(S); W=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: W[2,2]=-1
    R=Vt.T@W@U.T; s=float(np.trace(np.diag(D)@W)/X.var(0).sum()); t=mq-s*R@mp
    return s,R,t
def r2fit(P,Q):
    s,R,t=umeyama(P,Q); pred=s*(R@P.T).T+t
    return 1-((pred-Q)**2).sum()/((Q-Q.mean(0))**2).sum(), np.linalg.norm(pred-Q,axis=1).mean()*1e3
data=[(r,)+load(r) for r in sample]
print("== per-episode similarity fit: correct pairing vs mismatched control ==")
print("   (P = human fingertip-mid, Q = robot EEF, same episode index)")
ok_r2,ok_res,sh_r2,sh_res=[],[],[],[]
rng=np.random.default_rng(0)
for k,(r,P,Q) in enumerate(data):
    a,b=r2fit(P,Q); ok_r2.append(a); ok_res.append(b)
    # control: same task, different episode, truncated to common length
    same=[d for d in data if d[0]["task"]==r["task"] and d[0]["source_sha256"]!=r["source_sha256"]]
    if not same: continue
    r2,Pc,Qc=same[rng.integers(len(same))]
    n=min(len(P),len(Qc))
    # resample the other episode's robot track to this length, mimicking phase alignment
    idx=np.round(np.linspace(0,len(Qc)-1,len(P))).astype(int)
    a2,b2=r2fit(P,Qc[idx]); sh_r2.append(a2); sh_res.append(b2)
print("  correct pairing   : R2 mean %.3f median %.3f | residual mean %.1f mm"%(np.mean(ok_r2),np.median(ok_r2),np.mean(ok_res)))
print("  mismatched control: R2 mean %.3f median %.3f | residual mean %.1f mm"%(np.mean(sh_r2),np.median(sh_r2),np.mean(sh_res)))
print("  n=%d episodes"%len(ok_r2))

# same test for the CURRENT 'human' channel (action) as a sanity ceiling
print("\n== same test for the current pipeline's 'human' channel (raw action) ==")
ok2,sh2=[],[]
acts=[]
for r in sample:
    with h5py.File(os.path.join(SRC,r["source_relative_path"]),"r") as f:
        acts.append((np.asarray(f["action"][:,:3],dtype=np.float64)*1e-3,
                     np.asarray(f["end_position"][:,:3],dtype=np.float64)*1e-3))
for k,(A,Q) in enumerate(acts):
    ok2.append(r2fit(A,Q)[0])
    j=(k+1)%len(acts); Qc=acts[j][1]
    idx=np.round(np.linspace(0,len(Qc)-1,len(A))).astype(int)
    sh2.append(r2fit(A,Qc[idx])[0])
print("  correct pairing   : R2 mean %.3f"%np.mean(ok2))
print("  mismatched control: R2 mean %.3f"%np.mean(sh2))
