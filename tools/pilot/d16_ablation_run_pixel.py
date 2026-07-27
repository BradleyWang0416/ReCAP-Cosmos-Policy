"""NONFORMAL_DIAGNOSTIC: ablation incl. the dense pixel descriptor.

Same optimisation as d15: candidate tables cached per (task, K, channel), distances
computed as one GEMM per feature.  The 576-d pixel feature is where the GEMM matters most.
"""
import os, pickle, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pilot_util import horizon_mask, pairwise_dist

H = 8
KS = (8, 32, 64)
FEATS = ("geom", "vis", "geom+vis", "abs-pos", "pix", "geom+pix")
CELLS = [(c, w, f) for c in ("A", "B") for w in ("phase", "search") for f in FEATS]

D = pickle.load(open("/tmp/h2r_abl2.pkl", "rb"))
pool, queries = D["pool"], D["queries"]


def build_table(task, K, ch):
    G, V, P, X, PX, ep_phase = [], [], [], [], [], []
    off = 0
    for e in pool[task]:
        S = e["A"] if ch == "A" else e["B"]
        gf = e["gA"] if ch == "A" else e["gB"]
        pf = e["posA"] if ch == "A" else e["posB"]
        m = horizon_mask(e["starts"], e["seg"], len(S), H, K)
        nsel = int(m.sum())
        if nsel == 0:
            continue
        starts = e["starts"][m]
        idx = starts[:, None] + np.arange(H, H + K)[None, :]
        X.append(S[idx][:, :, :3])
        G.append(gf[m]); V.append(e["V"][m]); P.append(pf[m][:, :3]); PX.append(e["Pix"][m])
        ep_phase.append((off, (starts + H) / e["frames"]))
        off += nsel
    if not G:
        return None
    return dict(G=np.concatenate(G), V=np.concatenate(V), P=np.concatenate(P),
                X=np.concatenate(X), PX=np.concatenate(PX), ep_phase=ep_phase, n=off)


def phase_index(table, q):
    qph = (q["start"] + H) / q["frames"]
    return np.array([off + int(np.argmin(np.abs(ph - qph))) for off, ph in table["ep_phase"]],
                    dtype=int)


def resid(chunks, cur, tgt):
    al = cur[None, None, :3] + chunks - chunks[:, 0:1, :]
    return np.linalg.norm(al - tgt[None, :, :3], axis=2).mean(1) * 1e3


queries_by_task = {}
for q in queries:
    queries_by_task.setdefault(q["task"], []).append(q)

for K in KS:
    t0 = time.time()
    acc = {c: [] for c in CELLS}
    hold, orc, n = [], [], 0
    tables = {(task, ch): build_table(task, K, ch) for task in pool for ch in ("A", "B")}

    for task, qlist in queries_by_task.items():
        valid = [q for q in qlist
                 if q["start"] + H + K <= len(q["R"])
                 and horizon_mask([q["start"]], q["seg"], len(q["R"]), H, K)[0]]
        if not valid:
            continue
        n += len(valid)
        curs = [q["R"][q["start"] + H - 1] for q in valid]
        tgts = [q["R"][q["start"] + H:q["start"] + H + K] for q in valid]
        for cur, tgt in zip(curs, tgts):
            hold.append(np.linalg.norm(tgt[:, :3] - cur[:3], axis=1).mean() * 1e3)
        best = [np.inf] * len(valid)

        for ch in ("A", "B"):
            T = tables[(task, ch)]
            if T is None:
                continue
            QG = np.stack([q["gA"] if ch == "A" else q["gB"] for q in valid])
            QV = np.stack([q["V"] for q in valid])
            QPX = np.stack([q["Pix"] for q in valid])
            QP = np.stack([c[:3] for c in curs])
            dist = {
                "geom": pairwise_dist(QG, T["G"]),
                "vis": pairwise_dist(QV, T["V"]),
                "geom+vis": pairwise_dist(np.concatenate((QG, QV), axis=1),
                                          np.concatenate((T["G"], T["V"]), axis=1)),
                "abs-pos": pairwise_dist(QP, T["P"]),
                "pix": pairwise_dist(QPX, T["PX"]),
                "geom+pix": pairwise_dist(np.concatenate((QG, QPX), axis=1),
                                          np.concatenate((T["G"], T["PX"]), axis=1)),
            }
            allidx = np.arange(T["n"])
            for i, q in enumerate(valid):
                r = resid(T["X"], curs[i], tgts[i])
                if ch == "A":
                    best[i] = min(best[i], float(r.min()))
                pm = phase_index(T, q)
                for f in FEATS:
                    di = dist[f][i]
                    for w in ("phase", "search"):
                        sub = pm if w == "phase" else allidx
                        if len(sub) == 0:
                            continue
                        o = sub[np.argsort(di[sub], kind="stable")]
                        acc[(ch, w, f)].append(r[o[0]])
        orc.extend(best)

    hb = np.mean(hold)
    print("\n" + "=" * 80)
    print("K=%-3d (≈%.2fs)  n=%d  无检索基线 %.1f mm  oracle %.1f mm   [%.1f s]"
          % (K, K / 30.0, n, hb, np.mean(orc), time.time() - t0))
    print("=" * 80)
    for m, c in sorted((np.mean(v), c) for c, v in acc.items() if v):
        print("  %-2s %-7s %-9s %8.1f mm  %6.2fx%s"
              % (c[0], c[1], c[2], m, m / hb, " ✅" if m < hb else ""))
