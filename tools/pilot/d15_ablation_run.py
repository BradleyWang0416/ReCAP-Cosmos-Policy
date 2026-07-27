"""NONFORMAL_DIAGNOSTIC: retrieval ablation on the frozen 160 dev queries.
Metric = |robot_future - aligned_plan| position error, vs the no-retrieval baseline.

Optimised over the serial original in two ways, without changing any number:
  * candidate tables are built once per (task, K, channel) instead of once per query;
  * candidate distances are one GEMM per (task, K, channel, feature) instead of a
    python loop of np.linalg.norm per query.
"""
import os, pickle, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pilot_util import horizon_mask, pairwise_dist

H = 8
KS = (8, 16, 32, 64)
FEATS = ("geom", "vis", "geom+vis", "abs-pos")
CELLS = [(c, w, f) for c in ("A", "B") for w in ("phase", "search") for f in FEATS]

D = pickle.load(open("/tmp/h2r_abl.pkl", "rb"))
pool, queries = D["pool"], D["queries"]


def build_table(task, K, ch):
    """All legal candidate windows of the task's pool, valid for horizon K."""
    G, V, P, CH, ep_phase = [], [], [], [], []
    off = 0
    for e in pool[task]:
        X = e["A"] if ch == "A" else e["B"]
        gf = e["gA"] if ch == "A" else e["gB"]
        pf = e["posA"] if ch == "A" else e["posB"]
        m = horizon_mask(e["starts"], e["seg"], len(X), H, K)
        nsel = int(m.sum())
        if nsel == 0:
            continue
        starts = e["starts"][m]
        idx = starts[:, None] + np.arange(H, H + K)[None, :]
        CH.append(X[idx][:, :, :3])
        G.append(gf[m]); V.append(e["V"][m]); P.append(pf[m][:, :3])
        ep_phase.append((off, (starts + H) / e["frames"]))
        off += nsel
    if not G:
        return None
    return dict(G=np.concatenate(G), V=np.concatenate(V), P=np.concatenate(P),
                CH=np.concatenate(CH), ep_phase=ep_phase, n=off)


def phase_index(table, q):
    """The frozen rule: one window per episode, closest normalised phase."""
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
    acc3 = {c: [] for c in CELLS}
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
            QP = np.stack([c[:3] for c in curs])
            dist = {
                "geom": pairwise_dist(QG, T["G"]),
                "vis": pairwise_dist(QV, T["V"]),
                "geom+vis": pairwise_dist(np.concatenate((QG, QV), axis=1) / np.sqrt(2.0),
                                          np.concatenate((T["G"], T["V"]), axis=1) / np.sqrt(2.0)),
                "abs-pos": pairwise_dist(QP, T["P"]),
            }
            allidx = np.arange(T["n"])
            for i, q in enumerate(valid):
                r = resid(T["CH"], curs[i], tgts[i])
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
                        acc3[(ch, w, f)].append(r[o[:3]].mean())
        orc.extend(best)

    hb = np.mean(hold)
    print("\n" + "=" * 86)
    print("K = %-3d (≈%.2f s @30fps)   有效 query %d   无检索基线 %.1f mm   oracle 上界 %.1f mm   [%.1f s]"
          % (K, K / 30.0, n, hb, np.mean(orc), time.time() - t0))
    print("=" * 86)
    print("  %-9s %-7s %-9s %10s %10s %9s" % ("通道", "窗口", "特征", "top1 mm", "top3均 mm", "vs基线"))
    rows = sorted((np.mean(acc[c]), c, np.mean(acc3[c])) for c in CELLS if acc[c])
    for m1, c, m3 in rows:
        rr = m1 / hb
        print("  %-9s %-7s %-9s %10.1f %10.1f %8.2fx%s"
              % (c[0], c[1], c[2], m1, m3, rr, " ✅" if rr < 1.0 else ""))
