import h5py, numpy as np, glob
np.set_printoptions(precision=4, suppress=True)
p=sorted(glob.glob("/home/wxs/ReCAP-Cosmos-Policy/data/Human2Robot/derived/v04/episodes/human_pool/grab_cube2_v1/*.hdf5"))[0]
with h5py.File(p,"r") as f:
    hf=np.asarray(f["data/demo_0/human/hand_frames"][:],dtype=np.float64)
    hc=np.asarray(f["data/demo_0/human/hand_coords"][:],dtype=np.float64)
T=len(hf)
R=hf[:,1:4,:]                      # candidate rotation rows
print("== hand_frames row0 (candidate position, m) ==")
print(" min",hf[:,0].min(0)," max",hf[:,0].max(0)," norm range %.3f..%.3f"%(np.linalg.norm(hf[:,0],axis=1).min(),np.linalg.norm(hf[:,0],axis=1).max()))
print("== rows1-3 orthonormality ==")
gram=np.einsum('tij,tkj->tik',R,R)
I=np.eye(3)[None]
print(" max |R R^T - I| =", np.abs(gram-I).max())
print(" det range: %.6f .. %.6f"%(np.linalg.det(R).min(),np.linalg.det(R).max()))
print("== hand_coords structure ==")
d=np.linalg.norm(hc,axis=2)        # distance of each of 24 pts from local origin
print(" per-point mean dist from origin (m):")
print(" ", np.round(d.mean(0),4))
print(" points that are always exactly 0:", np.where(d.max(0)<1e-9)[0].tolist())
# pairwise distances among the far points, look for bimodal grasp signal
cand=[i for i in range(24) if d.mean(0)[i]>0.05]
print(" candidate fingertip-ish indices (mean dist>5cm):", cand)
