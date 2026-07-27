import json, h5py, numpy as np, os
SPLIT="/home/wxs/ReCAP-Cosmos-Policy/data/Human2Robot/derived/v04/source_split_manifest.json"
SRC="/DATA1/wxs/DATASETS/Human2Robot/data/v1"
rec=json.load(open(SPLIT))["records"]
seen=[r for r in rec if r["source_partition"]=="seen_train"]
by={}
for r in sorted(seen,key=lambda x:x["source_sha256"]): by.setdefault(r["task"],[]).append(r)
sample=[r for t in sorted(by) for r in by[t][:4]]
I,J=16,20; K=8
dA=[];dQ=[];dP=[]
for r in sample:
    with h5py.File(os.path.join(SRC,r["source_relative_path"]),"r") as f:
        hf=np.asarray(f["transformed_hand_frames"][:],dtype=np.float64)
        hc=np.asarray(f["transformed_hand_coords"][:],dtype=np.float64)
        ep=np.asarray(f["end_position"][:,:3],dtype=np.float64)*1e-3
        ac=np.asarray(f["action"][:,:3],dtype=np.float64)*1e-3
    P=hf[:,0]+0.5*(hc[:,I]+hc[:,J])
    dA.append(ac[K:]-ac[:-K]); dQ.append(ep[K:]-ep[:-K]); dP.append(P[K:]-P[:-K])
dA=np.concatenate(dA); dQ=np.concatenate(dQ); dP=np.concatenate(dP)
print("K=8 displacement magnitudes (mm): robot %.1f | action %.1f | human %.1f"
      %(np.linalg.norm(dQ,axis=1).mean()*1e3, np.linalg.norm(dA,axis=1).mean()*1e3, np.linalg.norm(dP,axis=1).mean()*1e3))
print("per-axis corr(action_delta, robot_delta):", [round(float(np.corrcoef(dA[:,i],dQ[:,i])[0,1]),3) for i in range(3)])
print("best axis-pair corr(human_delta, robot_delta):")
C=np.array([[float(np.corrcoef(dP[:,i],dQ[:,j])[0,1]) for j in range(3)] for i in range(3)])
print(np.round(C,3))
print("action_delta - robot_delta residual (mm): mean %.1f"%(np.linalg.norm(dA-dQ,axis=1).mean()*1e3))
# 1-step
dQ1=[];dA1=[]
for r in sample[:8]:
    with h5py.File(os.path.join(SRC,r["source_relative_path"]),"r") as f:
        ep=np.asarray(f["end_position"][:,:3],dtype=np.float64)*1e-3
        ac=np.asarray(f["action"][:,:3],dtype=np.float64)*1e-3
    dQ1.append(ep[1:]-ep[:-1]); dA1.append(ac[1:]-ac[:-1])
dQ1=np.concatenate(dQ1); dA1=np.concatenate(dA1)
print("1-step: robot %.2f mm, action %.2f mm, corr %s"
      %(np.linalg.norm(dQ1,axis=1).mean()*1e3, np.linalg.norm(dA1,axis=1).mean()*1e3,
        [round(float(np.corrcoef(dA1[:,i],dQ1[:,i])[0,1]),3) for i in range(3)]))
