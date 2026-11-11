"""E8: field-calibrated workload.  Hardware fault rates per node follow the Backblaze 2025 drive-model
statistics (backblaze.py); a fraction phi of the fleet is additionally unreliable (flaky / malicious
behaviour of the synthetic workload).  N = 1000 nodes, 365 daily epochs, held-out seeds 100-129.
RACE-Audit uses the hyper-parameters selected on the synthetic tuning seeds, unchanged."""
import json, sys, time
import numpy as np
from functools import partial
from multiprocessing import Pool
from scipy import stats
from sim import run, Uniform, RepGated, Race, ci95
import backblaze as bb

SEEDS = list(range(100, 130))
PHI = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
N, T = 1000, 365

def cell(args):
    g, Bn, phi = args
    kw = dict(N=N, B_per_node=Bn, epochs=T, pop_fn=partial(bb.population, gamma=g, phi=phi))
    out = {}
    runs = {"Uniform": [run(Uniform, seed=s, **kw) for s in SEEDS], "RACE-Audit": [run(Race, seed=s, **kw) for s in SEEDS]}
    best = None
    for cR in (128, 256, 460, 1024):
        if cR > 4 * Bn: continue
        r = [run(lambda **k: RepGated(cR=cR, **k), seed=s, **kw) for s in SEEDS]
        m = np.mean([x["exposure"] for x in r])
        if best is None or m < best[0]: best = (m, cR, r)
    runs["Rep-Gated"] = best[2]
    for k, r in runs.items():
        out[k] = dict(exposure=ci95([x["exposure"] for x in r]), mttd=ci95([x["mttd"] for x in r]),
                      episodes=float(np.mean([x["episodes"] for x in r])), raw=[x["exposure"] for x in r])
    out["Rep-Gated"]["cR"] = best[1]
    u, r = np.array(out["Uniform"]["raw"]), np.array(out["RACE-Audit"]["raw"])
    out["reduction_vs_uniform"] = float(1 - r.mean() / u.mean())
    out["reduction_vs_repgated"] = float(1 - r.mean() / np.mean(out["Rep-Gated"]["raw"]))
    d = u - r
    out["p_race_better"] = float(stats.wilcoxon(u, r, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    out["p_uniform_better"] = float(stats.wilcoxon(r, u, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    return (g, Bn, phi), out

if __name__ == "__main__":
    t0 = time.time()
    cells = [(g, Bn, phi) for g in (1, 10) for Bn in (64, 256) for phi in PHI]
    res = {}
    with Pool(2) as p:
        for key, out in p.imap_unordered(cell, cells):
            res["g%d_B%d_phi%g" % key] = out
            print(key, {k: round(out[k]["exposure"][0], 4) for k in ("Uniform", "Rep-Gated", "RACE-Audit")},
                  round(out["reduction_vs_uniform"], 3), round(time.time() - t0), flush=True)
            json.dump(dict(N=N, T=T, seeds=SEEDS, phi=PHI, cells=res), open("results/res_E8.json", "w"), indent=1)
