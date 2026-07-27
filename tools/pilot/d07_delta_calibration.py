"""NONFORMAL_DIAGNOSTIC: can a single seen-train-fitted rotation+scale align human deltas to robot deltas?
   (align_pool_chunk only ever uses deltas, so translation is irrelevant.)"""
import json, h5py, numpy as np, os
SPLIT="/home/wxs/ReCAP-Cosmos-Policy/data/Human2Robot/derived/v04/source_split_manifest.json"
SRC="/DATA1/wxs/DATASETS/Human2Robot/data/v1"
rec=json.load(open(SPLIT))["records"]
seen=[r for r in rec if r["source_partition"]=="seen_train"]
by={}
for r in sorted(seen,key=lambda x:x["source_sha256"]): by.setdefault(r["task"],[]).append(r)
train=[r for t in sorted(by) for r in by[t][:4]]     # fit
test =[r for t in sorted(by) for r in by[t][4:6]]    # held-out-from-fit seen episodes
I,J=16,20; K=8
def load(r):
    with h5py.File(os.path.join(SRC,r["source_relative_path"]),"r") as f:
        hf=np.asarray(f["transformed_hand_frames"][:],dtype=np.float64)
        hc=np.asarray(f["transformed_hand_coords"][:],dtype=np.float64)
        ep=np.asarray(f["end_position"][:,:3],dtype=np.float64)*1e-3
        ac=np.asarray(f["action"][:,:3],dtype=np.float64)*1e-3
    return hf[:,0]+0.5*(hc[:,I]+hc[:,J]), ep, ac
def deltas(X,k=K):
    return X[k:]-X[:-k]
def fit_rot_scale(dP,dQ):
    S=dP.T@dQ/len(dP); U,D,Vt=np.linalg.svd(S); W=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: W[2,2]=-1
    R=Vt.T@W@U.T; s=float(np.trace(np.diag(D)@W)/ (dP**2).sum(1).mean())
    return s,R
def score(dP,dQ,s,R):
    pred=s*(R@dP.T).T
    return 1-((pred-dQ)**2).sum()/ (dQ**2).sum(), np.linalg.norm(pred-dQ,axis=1).mean()*1e3
dPs,dQs,dAs=[],[],[]
for r in train:
    P,Q,A=load(r); dPs.append(deltas(P)); dQs.append(deltas(Q)); dAs.append(deltas(A))
dP=np.concatenate(dPs); dQ=np.concatenate(dQs); dA=np.concatenate(dAs)
s,R=fit_rot_scale(dP,dQ)
ang=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
print("== global K=8 delta calibration fitted on %d seen-train episodes =="%len(train))
print("   scale=%.3f  rotation=%.1f deg  (fit set: R2=%.3f, mean err %.1f mm)"%((s,ang)+score(dP,dQ,s,R)))
print("   robot K=8 displacement magnitude: mean %.1f mm"%(np.linalg.norm(dQ,axis=1).mean()*1e3))
# held-out seen episodes
dPs2,dQs2=[],[]
for r in test:
    P,Q,A=load(r); dPs2.append(deltas(P)); dQs2.append(deltas(Q))
dP2=np.concatenate(dPs2); dQ2=np.concatenate(dQs2)
print("   held-out-from-fit episodes (%d): R2=%.3f, mean err %.1f mm"%((len(test),)+score(dP2,dQ2,s,R)))
print()
sA,RA=fit_rot_scale(dA,dQ)
print("== same calibration for the current 'human' channel (action) ==")
r2a,ea=score(dA,dQ,sA,RA)
print("   scale=%.3f  rotation=%.2f deg  R2=%.3f  mean err %.1f mm"%(sA,np.degrees(np.arccos(np.clip((np.trace(RA)-1)/2,-1,1))),r2a,ea))
np.save("/tmp/h2r_delta_calib.npy",{"s":s,"R":R,"pair":(I,J)},allow_pickle=True)
