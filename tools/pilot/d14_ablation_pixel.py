"""NONFORMAL_DIAGNOSTIC (read-only, CPU): add a dense pixel descriptor to test whether the
16-d spatial-mean WAN feature is what breaks the visual branch."""
import json, h5py, numpy as np, pickle
ROOT="/home/wxs/ReCAP-Cosmos-Policy"; DER=f"{ROOT}/data/Human2Robot/derived/v04"
H=8; G=24                                  # 24x24 grayscale descriptor
def host(p): return p.replace("/workspace",ROOT)
split=json.load(open(f"{DER}/source_split_manifest.json"))["records"]
plan=json.load(open(f"{DER}/stage4_smoke_plan.json"))
D=pickle.load(open("/tmp/h2r_abl.pkl","rb")); pool=D["pool"]; queries=D["queries"]

def desc(imgs):
    """imgs (n,h,w,3) uint8 -> (n, G*G) zero-mean unit-norm grayscale descriptor"""
    x=imgs.astype(np.float32).mean(-1)
    h,w=x.shape[1:]
    ys=np.linspace(0,h,G+1).astype(int); xs=np.linspace(0,w,G+1).astype(int)
    out=np.empty((len(x),G,G),dtype=np.float32)
    for i in range(G):
        for j in range(G):
            out[:,i,j]=x[:,ys[i]:ys[i+1],xs[j]:xs[j+1]].mean((1,2))
    v=out.reshape(len(x),-1)
    v-=v.mean(1,keepdims=True)
    return v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-12)

byska={}
for r in split:
    if r["source_partition"] in ("v04_human_pool","v04_robot_dev"): byska[r["source_sha256"]]=r
done=0
for task,eps in pool.items():
    for e in eps:
        r=byska[e["sha"]]
        with h5py.File(host(r["projection"]["path"]),"r") as f:
            rows=(e["starts"]+H-1).astype(int)
            imgs=np.asarray(f["data/demo_0/human/images"][:],dtype=np.uint8)[rows]
        e["Pix"]=desc(imgs); done+=1
        print("  pool %d/40 %s"%(done,r["episode_id"]),flush=True)
qcache={}
for q,rec in zip(queries,plan["queries"]):
    qr=rec["query_record"]; sha=qr["source_sha256"]
    if sha not in qcache:
        with h5py.File(host(qr["projection"]["path"]),"r") as f:
            qcache[sha]=np.asarray(f["data/demo_0/robot/images"][:],dtype=np.uint8)
    q["Pix"]=desc(qcache[sha][None,q["start"]+H-1])[0]
print("query descriptors:",len(queries))
pickle.dump(dict(pool=pool,queries=queries),open("/tmp/h2r_abl2.pkl","wb"))
print("cached -> /tmp/h2r_abl2.pkl")
