import json, h5py, numpy as np, os
DER="/home/wxs/ReCAP-Cosmos-Policy/data/Human2Robot/derived/v04"
SRC="/DATA1/wxs/DATASETS/Human2Robot/data/v1"
split=json.load(open(f"{DER}/source_split_manifest.json"))["records"]
dts=[]; lens=[]
for r in [x for x in split if x["source_partition"]=="v04_robot_dev"]:
    with h5py.File(os.path.join(SRC,r["source_relative_path"]),"r") as f:
        ts=np.asarray(f["timestamp"][:],dtype=np.float64)
    d=np.diff(ts); dts.append(np.median(d)); lens.append(len(ts))
dt=np.median(dts)
print("timestamp median dt = %.4g (raw units)"%dt)
for unit,scale in (("ns",1e-9),("us",1e-6),("ms",1e-3),("s",1.0)):
    print("   if %-2s -> dt=%.4f s, fps=%.1f, H=8 span=%.2f s, K=8 horizon=%.2f s"%(unit,dt*scale,1/(dt*scale),8*dt*scale,8*dt*scale))
print("episode length: median %d frames"%np.median(lens))
