"""NONFORMAL_DIAGNOSTIC (read-only): is the `action` field a human hand signal or a robot command?

Primary evidence for pilot finding P0-1.  Reads raw Human2Robot seen-train episodes only,
writes nothing.  Reproduces the headline numbers in 方案/v05_pilot/PILOT_LOG.md entry 001.

Run:  python tools/pilot/d02_action_is_robot_command.py
"""
import json, os
import h5py
import numpy as np

SPLIT = "/home/wxs/ReCAP-Cosmos-Policy/data/Human2Robot/derived/v04/source_split_manifest.json"
SRC = "/DATA1/wxs/DATASETS/Human2Robot/data/v1"
K = 8
np.set_printoptions(precision=4, suppress=True)

records = json.load(open(SPLIT))["records"]
seen = [r for r in records if r["source_partition"] == "seen_train"]
by_task = {}
for r in sorted(seen, key=lambda x: x["source_sha256"]):
    by_task.setdefault(r["task"], []).append(r)
sample = [r for t in sorted(by_task) for r in by_task[t][:3]]
print("sample: %d seen-train episodes / %d tasks" % (len(sample), len(by_task)))

print("\n== 0. field inventory of one episode ==")
with h5py.File(os.path.join(SRC, sample[0]["source_relative_path"]), "r") as f:
    f.visititems(lambda n, o: print("   %-32s %s %s" % (n, o.shape, o.dtype))
                 if isinstance(o, h5py.Dataset) else None)

A, E, G, HF = [], [], [], []
for r in sample:
    with h5py.File(os.path.join(SRC, r["source_relative_path"]), "r") as f:
        A.append(np.asarray(f["action"][:], dtype=np.float64))
        E.append(np.asarray(f["end_position"][:], dtype=np.float64))
        G.append(np.asarray(f["gripper_state"][:], dtype=np.float64))
        HF.append(np.asarray(f["transformed_hand_frames"][:], dtype=np.float64))

print("\n== 1. `action` vs `end_position` (absolute) ==")
a = np.concatenate([x[:, :3] for x in A]); e = np.concatenate([x[:, :3] for x in E])
print("   per-axis corr        :", [round(float(np.corrcoef(a[:, i], e[:, i])[0, 1]), 3) for i in range(3)])
print("   per-axis mean |a-e|  : %s mm" % np.round(np.abs(a - e).mean(0), 2))
print("   per-axis std of e    : %s mm" % np.round(e.std(0), 1))
res = np.linalg.norm(a - e, axis=1)
print("   residual             : mean %.1f  median %.1f  p90 %.1f mm"
      % (res.mean(), np.median(res), np.percentile(res, 90)))
ss = ((a - e) ** 2).sum(); st = ((e - e.mean(0)) ** 2).sum()
print("   R^2 (identity map)   : %.3f" % (1 - ss / st))

print("\n== 2. rotation columns ==")
for name, X in (("action[:,3:6]", np.concatenate([x[:, 3:6] for x in A])),
                ("end_position[:,3:6]", np.concatenate([x[:, 3:6] for x in E]))):
    u = np.unique(np.round(X, 1), axis=0)
    print("   %-22s %d unique values, first: %s" % (name, len(u), u[0]))

print("\n== 3. gripper ==")
ag = np.concatenate([x[:, 6] for x in A]); gs = np.concatenate(G)
print("   mean |action[:,6] - gripper_state| = %.4f  (identical on %.1f%% of frames)"
      % (np.abs(ag - gs).mean(), 100.0 * (np.abs(ag - gs) < 1e-6).mean()))

print("\n== 4. increments (this is what the retrieval geometry feature is built on) ==")
for lag, label in ((1, "1-step"), (K, "K=8")):
    da = np.concatenate([x[lag:, :3] - x[:-lag, :3] for x in A])
    de = np.concatenate([x[lag:, :3] - x[:-lag, :3] for x in E])
    print("   %-7s magnitude: action %.2f mm, robot %.2f mm | per-axis corr %s"
          % (label, np.linalg.norm(da, axis=1).mean(), np.linalg.norm(de, axis=1).mean(),
             [round(float(np.corrcoef(da[:, i], de[:, i])[0, 1]), 3) for i in range(3)]))

print("\n== 5. is `action` derived from the human hand? ==")
hw = np.concatenate([x[:, 0] for x in HF])          # hand_frames row0 = wrist position (m)
C = np.array([[float(np.corrcoef(hw[:, i], a[:, j])[0, 1]) for j in range(3)] for i in range(3)])
print("   corr(hand_wrist_xyz, action_xyz) 3x3:")
print(np.round(C, 3))
print("   max |corr| = %.3f  (contrast with 0.94-0.96 against end_position above)" % np.abs(C).max())

print("\nCONCLUSION: `action` lives in the robot's coordinate frame and tracks `end_position`;")
print("it is the robot's commanded end-effector pose, not a human-hand signal.")
