import h5py, numpy as np, glob
H,K=8,8
rng=np.random.default_rng(1)
rows=[]
for task in ["grab_cube2_v1","push_plate_v1","grab_to_plate1_v1","push_box_random_v1"]:
    files=sorted(glob.glob("/DATA1/wxs/DATASETS/Human2Robot/data/v1/%s/episode_*.hdf5"%task))[:8]
    eps=[]
    for p in files:
        with h5py.File(p,"r") as f:
            eps.append((np.asarray(f["action"][:,6],dtype=np.float64), np.asarray(f["gripper_state"][:],dtype=np.float64)))
    same=[];cross=[];const=[];chg=[]
    for i,(ha,rg) in enumerate(eps):
        T=len(rg)
        if T<H+K+2: continue
        for s in rng.integers(0,T-H-K,size=100):
            s=int(s); cur=rg[s+H-1]; tgt=rg[s+H:s+H+K]
            same.append(np.abs(ha[s+H:s+H+K]-tgt).mean())
            const.append(np.abs(cur-tgt).mean())
            chg.append(float(np.any(tgt!=cur)))
            j=(i+1)%len(eps); hj=eps[j][0]; Tj=len(hj)
            sj=max(0,min(Tj-H-K-1,int(round((s+H)/T*Tj))-H))
            cross.append(np.abs(hj[sj+H:sj+H+K]-tgt).mean())
    rows.append((task,len(same),np.mean(same),np.mean(cross),np.mean(const),np.mean(chg)))
print("%-22s %5s %12s %13s %12s %15s"%("task","N","same-ep plan","cross-ep plan","hold-current","P(grip change)"))
for t,n,a,b,c,d in rows:
    print("%-22s %5d %12.4f %13.4f %12.4f %15.3f"%(t,n,a,b,c,d))
