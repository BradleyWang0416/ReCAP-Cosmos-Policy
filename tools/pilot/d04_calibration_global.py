"""NONFORMAL_DIAGNOSTIC (read-only): quantify the human-wrist -> robot-EEF gap.

Reads raw Human2Robot seen-train episodes only. Writes nothing outside /tmp.
"""
import json, h5py, numpy as np, os
np.set_printoptions(precision=4, suppress=True)

SPLIT = "/home/wxs/ReCAP-Cosmos-Policy/data/Human2Robot/derived/v04/source_split_manifest.json"
SRC   = "/DATA1/wxs/DATASETS/Human2Robot/data/v1"
rec   = json.load(open(SPLIT))["records"]
seen  = [r for r in rec if r["source_partition"] == "seen_train"]

# deterministic subsample: 3 episodes per seen task (48 episodes)
by_task = {}
for r in sorted(seen, key=lambda x: x["source_sha256"]):
    by_task.setdefault(r["task"], []).append(r)
sample = [r for t in sorted(by_task) for r in by_task[t][:3]]
print("calibration sample: %d episodes / %d tasks" % (len(sample), len(by_task)))

def load(r):
    with h5py.File(os.path.join(SRC, r["source_relative_path"]), "r") as f:
        hf = np.asarray(f["transformed_hand_frames"][:], dtype=np.float64)
        hc = np.asarray(f["transformed_hand_coords"][:], dtype=np.float64)
        ep = np.asarray(f["end_position"][:], dtype=np.float64)
        g  = np.asarray(f["gripper_state"][:], dtype=np.float64)
        ac = np.asarray(f["action"][:], dtype=np.float64)
    return hf, hc, ep[:, :3] * 0.001, g, ac       # robot pos -> meters

def umeyama(P, Q):
    """similarity transform sR P + t ~= Q"""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    S = X.T @ Y / len(P)
    U, D, Vt = np.linalg.svd(S)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = (Vt.T @ W @ U.T)
    s = float(np.trace(np.diag(D) @ W) / X.var(0).sum())
    t = mq - s * R @ mp
    return s, R, t

# ---- temporal pairing check + global calibration -------------------------
Ps, Qs, per_ep = [], [], []
for r in sample:
    hf, hc, ep, g, ac = load(r)
    Ps.append(hf[:, 0]); Qs.append(ep)
P = np.concatenate(Ps); Q = np.concatenate(Qs)
s, R, t = umeyama(P, Q)
res = np.linalg.norm((s * (R @ P.T).T + t) - Q, axis=1)
ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
print("\n== global human-wrist -> robot-EEF similarity fit (all sampled frames) ==")
print("  frames=%d  scale=%.4f  rotation=%.1f deg  translation=%s" % (len(P), s, ang, np.round(t, 3)))
print("  residual mm: mean %.1f  median %.1f  p90 %.1f" % (res.mean()*1e3, np.median(res)*1e3, np.percentile(res,90)*1e3))
print("  robot EEF motion scale for reference: std %s mm" % np.round(Q.std(0)*1e3, 1))
# variance explained
ss_res = ((s*(R@P.T).T + t - Q)**2).sum(); ss_tot = ((Q-Q.mean(0))**2).sum()
print("  R^2 = %.4f" % (1 - ss_res/ss_tot))

# ---- per-episode fit (upper bound if calibration were per-episode) -------
pe = []
for r, Pi, Qi in zip(sample, Ps, Qs):
    si, Ri, ti = umeyama(Pi, Qi)
    ri = np.linalg.norm((si*(Ri@Pi.T).T + ti) - Qi, axis=1)
    pe.append((r["task"], si, np.degrees(np.arccos(np.clip((np.trace(Ri)-1)/2,-1,1))), ri.mean()*1e3))
print("\n== per-episode fits (scale / rot-deg / mean residual mm) ==")
print("  scale   : mean %.3f  std %.3f" % (np.mean([x[1] for x in pe]), np.std([x[1] for x in pe])))
print("  rotation: mean %.1f  std %.1f deg" % (np.mean([x[2] for x in pe]), np.std([x[2] for x in pe])))
print("  residual: mean %.1f  median %.1f mm" % (np.mean([x[3] for x in pe]), np.median([x[3] for x in pe])))

# ---- for contrast: action -> end_position (the current 'human' channel) --
Ra=[]
for r in sample:
    hf, hc, ep, g, ac = load(r)
    Ra.append(np.linalg.norm(ac[:, :3]*0.001 - ep, axis=1))
Ra = np.concatenate(Ra)
print("\n== current pipeline's 'human' channel (raw `action`) vs robot end_position ==")
print("  residual mm: mean %.1f  median %.1f  p90 %.1f   (no fitting needed - same frame)"
      % (Ra.mean()*1e3, np.median(Ra)*1e3, np.percentile(Ra,90)*1e3))

# ---- fingertip pair identification (seen-train only) ---------------------
best = None
for r in sample[:16]:
    hf, hc, ep, g, ac = load(r)
    if g.std() < 1e-6:  continue
    D = np.linalg.norm(hc[:, :, None, :] - hc[:, None, :, :], axis=-1)   # (T,24,24)
    C = np.zeros((24, 24))
    for i in range(24):
        for j in range(i+1, 24):
            d = D[:, i, j]
            C[i, j] = 0.0 if d.std() < 1e-9 else abs(np.corrcoef(d, g)[0, 1])
    if best is None: best = C.copy(); n = 1
    else: best += C; n += 1
best /= n
i, j = np.unravel_index(np.nanargmax(best), best.shape)
print("\n== gripper proxy: hand_coords pair most correlated with robot gripper_state ==")
print("  best pair = (%d, %d)  mean |corr| = %.3f" % (i, j, best[i, j]))
order = np.dstack(np.unravel_index(np.argsort(-best, axis=None)[:5], best.shape))[0]
print("  top-5 pairs:", [(int(a), int(b), round(float(best[a, b]), 3)) for a, b in order])
np.save("/tmp/h2r_calib.npy", {"s": s, "R": R, "t": t, "pair": (int(i), int(j))}, allow_pickle=True)
print("\n(calibration cached to /tmp/h2r_calib.npy)")
