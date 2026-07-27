"""NONFORMAL_DIAGNOSTIC: analyse the A/B retrieval comparison + aligned-plan residuals."""
import json, h5py, numpy as np, os
from scipy.spatial.transform import Rotation
from scipy.stats import kendalltau
ROOT="/home/wxs/ReCAP-Cosmos-Policy"; DER=f"{ROOT}/data/Human2Robot/derived/v04"
H,K,TOPK,I,J=8,8,3,16,20
d=np.load("/tmp/h2r_cmp.npy",allow_pickle=True).item(); res=d["res"]; AP_LO,AP_HI=d["AP"]
CAL=np.load("/tmp/h2r_delta_calib.npy",allow_pickle=True).item(); SC,RC=CAL["s"],CAL["R"]
split=json.load(open(f"{DER}/source_split_manifest.json"))["records"]
def host(p): return p.replace("/workspace",ROOT)
def to10d(p6,g):
    m=Rotation.from_euler("XYZ",p6[:,3:6],degrees=True).as_matrix()
    return np.concatenate((p6[:,:3]*1e-3,m[:,:,0],m[:,:,1],np.asarray(g).reshape(-1,1)),axis=1)
def humanB(hf,hc):
    pos=(SC*(RC@(hf[:,0]+0.5*(hc[:,I]+hc[:,J])).T).T)
    rot=np.einsum('ij,tjk->tik',RC,np.transpose(hf[:,1:4],(0,2,1)))
    ap=np.linalg.norm(hc[:,I]-hc[:,J],axis=1)
    g=np.clip((ap-AP_LO)/(AP_HI-AP_LO+1e-9),0,1)
    return np.concatenate((pos,rot[:,:,0],rot[:,:,1],g[:,None]),axis=1)
pool={}
for r in split:
    if r["source_partition"]!="v04_human_pool": continue
    with h5py.File(host(r["projection"]["path"]),"r") as f:
        dd=f["data/demo_0"]
        a7=np.asarray(dd["human/hand_action_7d"][:],dtype=np.float64)
        hf=np.asarray(dd["human/hand_frames"][:],dtype=np.float64)
        hc=np.asarray(dd["human/hand_coords"][:],dtype=np.float64)
    pool[r["source_sha256"]]=(to10d(a7[:,:6],a7[:,6]),humanB(hf,hc))
def align(pool_chunk,cur):
    a=pool_chunk.copy(); a[:,:3]=cur[:3]+pool_chunk[:,:3]-pool_chunk[0,:3]; return a

# ---------------- 1. ranking agreement ----------------
ovl3,ovl3g,r1,r1g,taus,taug=[],[],[],[],[],[]
dA_all,dB_all=[],[]
for q in res:
    A=sorted(q["A"]); B=sorted(q["B"])
    Ag=sorted(q["A"],key=lambda x:x[1]); Bg=sorted(q["B"],key=lambda x:x[1])
    sA=[x[2] for x in A]; sB=[x[2] for x in B]
    ovl3.append(len(set(sA[:3])&set(sB[:3]))/3.0); r1.append(sA[0]==sB[0])
    sAg=[x[2] for x in Ag]; sBg=[x[2] for x in Bg]
    ovl3g.append(len(set(sAg[:3])&set(sBg[:3]))/3.0); r1g.append(sAg[0]==sBg[0])
    order={s:i for i,s in enumerate(sA)}
    taus.append(kendalltau([order[s] for s in sA],[order[s] for s in sB]).statistic)
    og={s:i for i,s in enumerate(sAg)}
    taug.append(kendalltau([og[s] for s in sAg],[og[s] for s in sBg]).statistic)
    dA_all+= [x[0] for x in q["A"]]; dB_all+=[x[0] for x in q["B"]]
print("="*74)
print("1. 检索排序一致性  (160 queries x 10 pool candidates, 相同 phase 窗口规则)")
print("="*74)
print("  %-28s %14s %14s"%("","geometry+visual","geometry only"))
print("  %-28s %14.3f %14.3f"%("top-3 集合重合率",np.mean(ovl3),np.mean(ovl3g)))
print("  %-28s %14.3f %14.3f"%("rank-1 相同比例",np.mean(r1),np.mean(r1g)))
print("  %-28s %14.3f %14.3f"%("Kendall tau (完整10名)",np.nanmean(taus),np.nanmean(taug)))
print("\n  距离分布 (geometry+visual, 全部 1600 个 query-candidate 对):")
print("    A(action) : mean %.4f  std %.4f  spread(p90-p10) %.4f"%(np.mean(dA_all),np.std(dA_all),np.percentile(dA_all,90)-np.percentile(dA_all,10)))
print("    B(真人手) : mean %.4f  std %.4f  spread(p90-p10) %.4f"%(np.mean(dB_all),np.std(dB_all),np.percentile(dB_all,90)-np.percentile(dB_all,10)))
# discrimination: within-query spread of distances
sprA=[np.std([x[0] for x in q["A"]]) for q in res]; sprB=[np.std([x[0] for x in q["B"]]) for q in res]
sgA=[np.std([x[1] for x in q["A"]]) for q in res]; sgB=[np.std([x[1] for x in q["B"]]) for q in res]
print("    query 内候选距离 std: A %.4f  B %.4f   (geometry-only: A %.4f  B %.4f)"%(np.mean(sprA),np.mean(sprB),np.mean(sgA),np.mean(sgB)))

# ---------------- 2. aligned-plan residual ----------------
def residual(q,rows,key,rank=0,rand=None):
    r=sorted(rows)[rank] if rand is None else rows[rand]
    st=pool[r[2]][key]; cs=r[3]
    plan=st[cs+H:cs+H+K]
    if len(plan)<K: return None
    a=align(plan,q["cur"])
    return (np.linalg.norm(a[:,:3]-q["tgt"][:,:3],axis=1).mean()*1e3,
            np.abs(a[:,9]-q["tgt"][:,9]).mean())
rng=np.random.default_rng(0)
acc={k:[[],[]] for k in ("A_top1","B_top1","A_rand","B_rand","hold")}
for q in res:
    for tag,rows,key in (("A_top1",q["A"],0),("B_top1",q["B"],1)):
        v=residual(q,rows,key)
        if v: acc[tag][0].append(v[0]); acc[tag][1].append(v[1])
    ri=int(rng.integers(len(q["A"])))
    for tag,rows,key in (("A_rand",q["A"],0),("B_rand",q["B"],1)):
        v=residual(q,rows,key,rand=ri)
        if v: acc[tag][0].append(v[0]); acc[tag][1].append(v[1])
    acc["hold"][0].append(np.linalg.norm(q["tgt"][:,:3]-q["cur"][:3],axis=1).mean()*1e3)
    acc["hold"][1].append(np.abs(q["tgt"][:,9]-q["cur"][9]).mean())
print("\n"+"="*74)
print("2. 对齐后残差 |robot_future - aligned_plan|  (这就是 RECAP 要学的残差目标)")
print("="*74)
print("  %-34s %12s %12s %12s"%("plan 来源","位置 mean","位置 median","夹爪 mean"))
lab={"A_top1":"A: action, 检索 top-1","B_top1":"B: 真人手, 检索 top-1",
     "A_rand":"A: action, 随机候选","B_rand":"B: 真人手, 随机候选","hold":"无检索基线 (保持当前位姿)"}
for k in ("A_top1","B_top1","A_rand","B_rand","hold"):
    p,g=acc[k]
    print("  %-34s %9.1f mm %9.1f mm %12.4f"%(lab[k],np.mean(p),np.median(p),np.mean(g)))
print("\n  参考: 该 query 集上机器人 K=8 实际位移 mean %.1f mm"%np.mean(acc["hold"][0]))
