"""NONFORMAL_DIAGNOSTIC (read-only, CPU): add a dense pixel descriptor to test whether the
16-d spatial-mean WAN feature is what breaks the visual branch.

Parallelised over episodes: the cost is per-frame gzip decompression, which is single-core
in HDF5.  Set PILOT_WORKERS to override the worker count.
"""
import json, os, pickle, sys, time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pilot_util import parallel_map, default_workers

ROOT = "/home/wxs/ReCAP-Cosmos-Policy"
DER = f"{ROOT}/data/Human2Robot/derived/v04"
H = 8
G = 24                                     # 24x24 grayscale descriptor


def host(p):
    return p.replace("/workspace", ROOT)


def desc(imgs):
    """imgs (n,h,w,3) uint8 -> (n, G*G) zero-mean unit-norm grayscale descriptor"""
    # mean(dtype=) avoids materialising a ~330 MB float32 copy per episode; the
    # arithmetic is identical (three exactly-representable uint8 values in float32).
    x = imgs.mean(-1, dtype=np.float32)
    h, w = x.shape[1:]
    ys = np.linspace(0, h, G + 1).astype(int)
    xs = np.linspace(0, w, G + 1).astype(int)
    out = np.empty((len(x), G, G), dtype=np.float32)
    for i in range(G):
        for j in range(G):
            out[:, i, j] = x[:, ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean((1, 2))
    v = out.reshape(len(x), -1)
    v -= v.mean(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def _episode_desc(args):
    """Worker: read one episode's frames and return their descriptors."""
    path, dataset, rows = args
    rows = np.asarray(rows, dtype=np.int64)
    with h5py.File(path, "r") as f:
        imgs = np.asarray(f[dataset][:], dtype=np.uint8)[rows]
    return desc(imgs)


def main():
    split = json.load(open(f"{DER}/source_split_manifest.json"))["records"]
    plan = json.load(open(f"{DER}/stage4_smoke_plan.json"))
    D = pickle.load(open("/tmp/h2r_abl.pkl", "rb"))
    pool, queries = D["pool"], D["queries"]
    workers = default_workers()
    print("workers: %d" % workers)

    by_sha = {r["source_sha256"]: r for r in split
              if r["source_partition"] in ("v04_human_pool", "v04_robot_dev")}

    # ---- pool episodes -------------------------------------------------
    episodes = [e for eps in pool.values() for e in eps]
    jobs = [(host(by_sha[e["sha"]]["projection"]["path"]),
             "data/demo_0/human/images",
             (e["starts"] + H - 1).astype(np.int64)) for e in episodes]
    t = time.time()
    for e, d in zip(episodes, parallel_map(_episode_desc, jobs, workers=workers)):
        e["Pix"] = d
    print("pool: %d episodes, %d windows, %.1f s" % (len(episodes), sum(len(j[2]) for j in jobs), time.time() - t))

    # ---- query frames, grouped by episode so each file is read once ----
    per_episode = {}
    for q, rec in zip(queries, plan["queries"]):
        qr = rec["query_record"]
        per_episode.setdefault(qr["source_sha256"], (host(qr["projection"]["path"]), []))[1].append(q)
    jobs = [(path, "data/demo_0/robot/images",
             np.asarray([q["start"] + H - 1 for q in qs], dtype=np.int64))
            for path, qs in per_episode.values()]
    t = time.time()
    for (path, qs), d in zip(per_episode.values(), parallel_map(_episode_desc, jobs, workers=workers)):
        for q, row in zip(qs, d):
            q["Pix"] = row
    print("queries: %d episodes, %d descriptors, %.1f s" % (len(jobs), len(queries), time.time() - t))

    assert all("Pix" in q for q in queries), "missing query descriptor"
    assert all("Pix" in e for eps in pool.values() for e in eps), "missing pool descriptor"
    pickle.dump(dict(pool=pool, queries=queries), open("/tmp/h2r_abl2.pkl", "wb"))
    print("cached -> /tmp/h2r_abl2.pkl")


if __name__ == "__main__":
    main()
