"""NONFORMAL_DIAGNOSTIC (read-only, CPU only): cache everything the ablation needs."""
import json, h5py, numpy as np, os, pickle
from scipy.spatial.transform import Rotation
ROOT="/home/wxs/ReCAP-Cosmos-Policy"; DER=f"{ROOT}/data/Human2Robot/derived/v04"
FEAT="/DATA1/wxs/ReCAP_M5B_V04_RUNS/features"; SRC="/DATA1/wxs/DATASETS/Human2Robot/data/v1"
H,I,J=8,16,20
def host(p): return p.replace("/workspace",ROOT)
split=json.load(open(f"{DER}/source_split_manifest.json"))["records"]
plan=json.load(open(f"{DER}/stage4_smoke_plan.json"))
g=json.load(open(f"{FEAT}/geometry_statistics.json"))
MEAN_A=np.asarray(g["mean_10d"]); STD_A=np.asarray(g["std_10d"])
prev=np.load("/tmp/h2r_cmp.npy",allow_pickle=True).item()
MEAN_B,STD_B,(AP_LO,AP_HI)=prev["MEAN_B"],prev["STD_B"],prev["AP"]
CAL=np.load("/tmp/h2r_delta_calib.npy",allow_pickle=True).item(); SC,RC=CAL["s"],CAL["R"]
widx={}
for sh in json.load(open(f"{FEAT}/wan_cache_index.json"))["shards"]:
    widx.setdefault((sh["partition"],sh["role"],sh["task"]),[]).append(sh["path"])
def wan(part,role,task):
    out={}
    for p in sorted(widx[(part,role,task)]):
        z=np.load(p); m=json.loads(str(z["manifest_json"].item()))
        key=m.get("source_sha256") or m.get("episode_id")
        out.setdefault(key,{}).update(dict(zip(z["starts"].tolist(),z["features"])))
    return out
def to10d(p6,gr):
    m=Rotation.from_euler("XYZ",p6[:,3:6],degrees=True).as_matrix()
    return np.concatenate((p6[:,:3]*1e-3,m[:,:,0],m[:,:,1],np.asarray(gr).reshape(-1,1)),axis=1)
def humanB(hf,hc):
    pos=(SC*(RC@(hf[:,0]+0.5*(hc[:,I]+hc[:,J])).T).T)
    rot=np.einsum('ij,tjk->tik',RC,np.transpose(hf[:,1:4],(0,2,1)))
    ap=np.linalg.norm(hc[:,I]-hc[:,J],axis=1)
    return np.concatenate((pos,rot[:,:,0],rot[:,:,1],np.clip((ap-AP_LO)/(AP_HI-AP_LO+1e-9),0,1)[:,None]),axis=1)
def geomfeat(X,starts,mean,std):
    idx=starts[:,None]+np.arange(H)[None,:]
    W=X[idx]                                   # (n,H,10)
    rel=(W-W[:,-1:,:]-mean)/std
    v=rel.reshape(len(starts),-1)
    return v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-12)

pool={}
for r in split:
    if r["source_partition"]!="v04_human_pool": continue
    with h5py.File(host(r["projection"]["path"]),"r") as f:
        d=f["data/demo_0"]
        a7=np.asarray(d["human/hand_action_7d"][:],dtype=np.float64)
        hf=np.asarray(d["human/hand_frames"][:],dtype=np.float64)
        hc=np.asarray(d["human/hand_coords"][:],dtype=np.float64)
        starts=np.asarray(d["time/legal_window_start"][:],dtype=np.int64)
        seg=np.asarray(d["time/segment_id"][:],dtype=np.int64)
    A=to10d(a7[:,:6],a7[:,6]); B=humanB(hf,hc)
    wf=wan("v04_human_pool","human",r["task"])
    wd=wf.get(r["source_sha256"]) or wf.get(r["episode_id"])
    V=np.stack([wd[int(s)] for s in starts]).astype(np.float64)
    V/= (np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
    pool.setdefault(r["task"],[]).append(dict(
        sha=r["source_sha256"],A=A,B=B,starts=starts,seg=seg,frames=int(r["frame_count"]),V=V,
        gA=geomfeat(A,starts,MEAN_A,STD_A),gB=geomfeat(B,starts,MEAN_B,STD_B),
        posA=A[starts+H-1],posB=B[starts+H-1]))
print("pool episodes:",sum(len(v) for v in pool.values()),"windows:",sum(len(e["starts"]) for v in pool.values() for e in v))

queries=[]
qw={t:wan("v04_robot_dev","robot",t) for t in pool}
for q in plan["queries"]:
    qr=q["query_record"]; qs=int(q["query_start"]); task=qr["task"]
    with h5py.File(host(qr["projection"]["path"]),"r") as f:
        d=f["data/demo_0"]
        pose=np.asarray(d["robot/observed_eef_pose_6d"][:],dtype=np.float64)
        grip=np.asarray(d["robot/gripper_state"][:],dtype=np.float64)
        seg=np.asarray(d["time/segment_id"][:],dtype=np.int64)
    R=to10d(pose,grip)
    wd=qw[task].get(qr["source_sha256"]) or qw[task].get(qr["episode_id"])
    v=np.asarray(wd[qs],dtype=np.float64); v/=np.linalg.norm(v)+1e-12
    st=np.asarray([qs])
    queries.append(dict(task=task,start=qs,R=R,seg=seg,frames=int(qr["frame_count"]),V=v,
                        gA=geomfeat(R,st,MEAN_A,STD_A)[0],gB=geomfeat(R,st,MEAN_B,STD_B)[0],
                        pos=R[qs+H-1]))
print("queries:",len(queries))
pickle.dump(dict(pool=pool,queries=queries),open("/tmp/h2r_abl.pkl","wb"))
print("cached -> /tmp/h2r_abl.pkl")
