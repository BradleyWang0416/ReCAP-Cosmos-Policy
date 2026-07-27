"""NONFORMAL_DIAGNOSTIC: oracle ceiling of the retrieval-plan idea on this data/horizon."""
import json, h5py, numpy as np, os
from scipy.spatial.transform import Rotation
ROOT="/home/wxs/ReCAP-Cosmos-Policy"; DER=f"{ROOT}/data/Human2Robot/derived/v04"
H,K,I,J=8,8,16,20
d=np.load("/tmp/h2r_cmp.npy",allow_pickle=True).item(); res=d["res"]; AP_LO,AP_HI=d["AP"]
CAL=np.load("/tmp/h2r_delta_calib.npy",allow_pickle=True).item(); SC,RC=CAL["s"],CAL["R"]
split=json.load(open(f"{DER}/source_split_manifest.json"))["records"]
plan=json.load(open(f"{DER}/stage4_smoke_plan.json"))
def host(p): return p.replace("/workspace",ROOT)
def to10d(p6,g):
    m=Rotation.from_euler("XYZ",p6[:,3:6],degrees=True).as_matrix()
    return np.concatenate((p6[:,:3]*1e-3,m[:,:,0],m[:,:,1],np.asarray(g).reshape(-1,1)),axis=1)
def humanB(hf,hc):
    pos=(SC*(RC@(hf[:,0]+0.5*(hc[:,I]+hc[:,J])).T).T)
    rot=np.einsum('ij,tjk->tik',RC,np.transpose(hf[:,1:4],(0,2,1)))
    ap=np.linalg.norm(hc[:,I]-hc[:,J],axis=1)
    return np.concatenate((pos,rot[:,:,0],rot[:,:,1],np.clip((ap-AP_LO)/(AP_HI-AP_LO+1e-9),0,1)[:,None]),axis=1)
pool={}
for r in split:
    if r["source_partition"]!="v04_human_pool": continue
    with h5py.File(host(r["projection"]["path"]),"r") as f:
        dd=f["data/demo_0"]
        a7=np.asarray(dd["human/hand_action_7d"][:],dtype=np.float64)
        hf=np.asarray(dd["human/hand_frames"][:],dtype=np.float64)
        hc=np.asarray(dd["human/hand_coords"][:],dtype=np.float64)
        starts=np.asarray(dd["time/legal_window_start"][:],dtype=np.int64)
    pool.setdefault(r["task"],[]).append((to10d(a7[:,:6],a7[:,6]),humanB(hf,hc),starts))
def oracle(cur,tgt,key):
    best=np.inf
    for A,B,starts in pool[key[1]]:
        X=A if key[0]=="A" else B
        s=starts[starts+H+K<=len(X)]
        if len(s)==0: continue
        idx=s[:,None]+np.arange(H,H+K)[None,:]
        chunk=X[idx][:,:,:3]                       # (n,K,3)
        aligned=cur[:3][None,None,:]+chunk-chunk[:,0:1,:]
        err=np.linalg.norm(aligned-tgt[None,:,:3],axis=2).mean(1)
        best=min(best,float(err.min()))
    return best*1e3
oa,ob,hold=[],[],[]
for q in res:
    oa.append(oracle(q["cur"],q["tgt"],("A",q["task"])))
    ob.append(oracle(q["cur"],q["tgt"],("B",q["task"])))
    hold.append(np.linalg.norm(q["tgt"][:,:3]-q["cur"][:3],axis=1).mean()*1e3)
print("="*70)
print("3. 检索计划的 oracle 上界 (池内全部合法窗口中取最优, 位置 mm)")
print("="*70)
print("  %-42s %10s %10s"%("","mean","median"))
print("  %-42s %10.1f %10.1f"%("A: action 通道, oracle 最优窗口",np.mean(oa),np.median(oa)))
print("  %-42s %10.1f %10.1f"%("B: 真人手通道, oracle 最优窗口",np.mean(ob),np.median(ob)))
print("  %-42s %10.1f %10.1f"%("无检索基线 (保持当前位姿)",np.mean(hold),np.median(hold)))
print("\n  说明: oracle 使用真实未来来挑窗口, 不可实现; 它是'检索一个人类计划片段'")
print("        在本数据+本时间尺度下能达到的绝对上限。")
