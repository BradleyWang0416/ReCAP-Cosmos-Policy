"""NONFORMAL_DIAGNOSTIC (read-only). Variant A (current: `action` as 'human') vs
Variant B (true human hand from transformed_hand_frames/coords) on the exact
160 stage-4 smoke queries.  Writes nothing outside /tmp.  No GPU: reuses the
frozen WAN feature cache."""
import json, h5py, numpy as np, os
from scipy.spatial.transform import Rotation
from scipy.stats import kendalltau

ROOT="/home/wxs/ReCAP-Cosmos-Policy"
DER=f"{ROOT}/data/Human2Robot/derived/v04"
FEAT="/DATA1/wxs/ReCAP_M5B_V04_RUNS/features"
SRC="/DATA1/wxs/DATASETS/Human2Robot/data/v1"
H,K,TOPK,I,J=8,8,3,16,20

def host(p): return p.replace("/workspace", ROOT)
split=json.load(open(f"{DER}/source_split_manifest.json"))["records"]
plan=json.load(open(f"{DER}/stage4_smoke_plan.json"))
gstat=json.load(open(f"{FEAT}/geometry_statistics.json"))
MEAN_A=np.asarray(gstat["mean_10d"]); STD_A=np.asarray(gstat["std_10d"])
widx={}
for sh in json.load(open(f"{FEAT}/wan_cache_index.json"))["shards"]:
    widx.setdefault((sh["partition"],sh["role"],sh["task"]),[]).append(sh["path"])

def wan(part,role,task):
    out={}
    for p in sorted(widx[(part,role,task)]):
        z=np.load(p); m=json.loads(str(z["manifest_json"].item()))
        if m.get("source_sha256"): out.setdefault(m["source_sha256"],{}).update(dict(zip(z["starts"].tolist(),z["features"])))
        else: out.setdefault(m.get("episode_id"),{}).update(dict(zip(z["starts"].tolist(),z["features"])))
    return out

def to10d(p6,g):
    m=Rotation.from_euler("XYZ",p6[:,3:6],degrees=True).as_matrix()
    return np.concatenate((p6[:,:3]*1e-3,m[:,:,0],m[:,:,1],np.asarray(g).reshape(-1,1)),axis=1)

CAL=np.load("/tmp/h2r_delta_calib.npy",allow_pickle=True).item(); SC,RC=CAL["s"],CAL["R"]
def humanB_from(hf,hc,ap_lo,ap_hi):
    """true-human 10D in robot-comparable frame: calibrated position, mapped rotation, aperture->gripper"""
    pos=(SC*(RC@(hf[:,0]+0.5*(hc[:,I]+hc[:,J])).T).T)
    rot=np.einsum('ij,tjk->tik',RC,np.transpose(hf[:,1:4],(0,2,1)))     # hand axes -> robot frame
    ap=np.linalg.norm(hc[:,I]-hc[:,J],axis=1)
    g=np.clip((ap-ap_lo)/(ap_hi-ap_lo+1e-9),0,1)
    return np.concatenate((pos,rot[:,:,0],rot[:,:,1],g[:,None]),axis=1)

def geom(hist,mean,std):
    rel=(hist-hist[-1]-mean)/std
    v=rel.reshape(-1); return v/ (np.linalg.norm(v)+1e-12)
def unit(v): 
    n=np.linalg.norm(v); return v/(n+1e-12)
def comb(g,v): return np.concatenate((unit(g),unit(v)))/np.sqrt(2.0)
def align(pool,cur):
    a=pool.copy(); a[:,:3]=cur[:3]+pool[:,:3]-pool[0,:3]
    def r6m(x):
        a1=x[:,0:3]; a2=x[:,3:6]
        b1=a1/np.linalg.norm(a1,axis=1,keepdims=True)
        b2=a2-(b1*a2).sum(1,keepdims=True)*b1; b2/=np.linalg.norm(b2,axis=1,keepdims=True)
        return np.stack((b1,b2,np.cross(b1,b2)),axis=2)
    pr=r6m(pool[:,3:9]); cr=r6m(cur[None,3:9])[0]
    ar=cr@pr[0].T@pr
    a[:,3:6]=ar[:,:,0]; a[:,6:9]=ar[:,:,1]
    return a

# ---------- seen-train statistics for variant B -------------------------------
seen=[r for r in split if r["source_partition"]=="seen_train"]
by={}
for r in sorted(seen,key=lambda x:x["source_sha256"]): by.setdefault(r["task"],[]).append(r)
fit=[r for t in sorted(by) for r in by[t][:4]]
aps=[]
for r in fit:
    with h5py.File(os.path.join(SRC,r["source_relative_path"]),"r") as f:
        hc=np.asarray(f["transformed_hand_coords"][:],dtype=np.float64)
    aps.append(np.linalg.norm(hc[:,I]-hc[:,J],axis=1))
AP=np.concatenate(aps); AP_LO,AP_HI=np.percentile(AP,2),np.percentile(AP,98)
rows=[]
for r in fit:
    with h5py.File(os.path.join(SRC,r["source_relative_path"]),"r") as f:
        hf=np.asarray(f["transformed_hand_frames"][:],dtype=np.float64)
        hc=np.asarray(f["transformed_hand_coords"][:],dtype=np.float64)
        ep=np.asarray(f["end_position"][:],dtype=np.float64); gs=np.asarray(f["gripper_state"][:],dtype=np.float64)
    B=humanB_from(hf,hc,AP_LO,AP_HI); Rb=to10d(ep,gs)
    for X in (B,Rb):
        n=len(X)-H+1
        if n<1: continue
        idx=np.arange(0,n,7)
        for s in idx: rows.append(X[s:s+H]-X[s+H-1])
S=np.concatenate(rows); MEAN_B=S.mean(0); STD_B=np.maximum(S.std(0),1e-8)
print("variant-B geometry stats from %d seen-train episodes, %d relative rows"%(len(fit),len(S)))
print("  aperture range used for gripper proxy: %.4f .. %.4f m"%(AP_LO,AP_HI))

# ---------- pool candidates ---------------------------------------------------
pool=[r for r in split if r["source_partition"]=="v04_human_pool"]
pool_by={}
for r in pool: pool_by.setdefault(r["task"],[]).append(r)
cand={}
for task,recs in pool_by.items():
    wp=wan("v04_human_pool","human",task)
    for r in recs:
        p=host(r["projection"]["path"])
        with h5py.File(p,"r") as f:
            d=f["data/demo_0"]
            a7=np.asarray(d["human/hand_action_7d"][:],dtype=np.float64)
            hf=np.asarray(d["human/hand_frames"][:],dtype=np.float64)
            hc=np.asarray(d["human/hand_coords"][:],dtype=np.float64)
            starts=np.asarray(d["time/legal_window_start"][:],dtype=np.int64)
        cand[r["source_sha256"]]=dict(rec=r,A=to10d(a7[:,:6],a7[:,6]),
                                      B=humanB_from(hf,hc,AP_LO,AP_HI),starts=starts,
                                      wan=wp.get(r["source_sha256"]) or wp.get(r["episode_id"]),
                                      frames=int(r["frame_count"]))
print("pool episodes loaded:", len(cand))

# ---------- queries -----------------------------------------------------------
qwan={t:wan("v04_robot_dev","robot",t) for t in pool_by}
res=[]
for q in plan["queries"]:
    qr=q["query_record"]; qs=int(q["query_start"]); task=qr["task"]
    p=host(qr["projection"]["path"])
    with h5py.File(p,"r") as f:
        d=f["data/demo_0"]
        pose=np.asarray(d["robot/observed_eef_pose_6d"][:],dtype=np.float64)
        grip=np.asarray(d["robot/gripper_state"][:],dtype=np.float64)
    Rq=to10d(pose,grip)
    qv=(qwan[task].get(qr["source_sha256"]) or qwan[task].get(qr["episode_id"]))[qs]
    hist=Rq[qs:qs+H]; cur=Rq[qs+H-1]; tgt=Rq[qs+H:qs+H+K]
    gq_A=geom(hist,MEAN_A,STD_A); gq_B=geom(hist,MEAN_B,STD_B)
    qphase=(qs+H)/float(qr["frame_count"])
    rowsA,rowsB=[],[]
    for sha,c in cand.items():
        if c["rec"]["task"]!=task: continue
        ph=(c["starts"]+H)/float(c["frames"])
        cs=int(c["starts"][int(np.argmin(np.abs(ph-qphase)))])            # frozen phase rule, identical for A/B
        cv=c["wan"][cs]
        dA=np.linalg.norm(comb(gq_A,qv)-comb(geom(c["A"][cs:cs+H],MEAN_A,STD_A),cv))
        dB=np.linalg.norm(comb(gq_B,qv)-comb(geom(c["B"][cs:cs+H],MEAN_B,STD_B),cv))
        gA=np.linalg.norm(gq_A-unit(geom(c["A"][cs:cs+H],MEAN_A,STD_A)))
        gB=np.linalg.norm(gq_B-unit(geom(c["B"][cs:cs+H],MEAN_B,STD_B)))
        rowsA.append((dA,gA,sha,cs)); rowsB.append((dB,gB,sha,cs))
    res.append(dict(task=task,cur=cur,tgt=tgt,A=rowsA,B=rowsB))
    if len(res)%40==0: print("  queries done:",len(res))
np.save("/tmp/h2r_cmp.npy",{"res":res,"MEAN_B":MEAN_B,"STD_B":STD_B,"AP":(AP_LO,AP_HI)},allow_pickle=True)
print("queries evaluated:",len(res))
