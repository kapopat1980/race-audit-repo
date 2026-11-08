"""Supplementary runs: (a) larger uniform / Rep-Gated budgets to bracket the
equal-assurance budget; (b) empirical check of Corollary 1 (AM/GM prediction)."""
import json, math
import numpy as np
from sim import *
from race_lib import beta_ucb, waterfill

SEEDS = list(range(100, 130))
out = {"uniform": {}, "rg": {}}
for B in [768, 1024, 1536]:
    for name, mk in [("uniform", lambda B: (lambda **k: Uniform(**k))), ("rg", None)]:
        if name == "uniform":
            runs = [run(lambda **k: Uniform(**k), B_per_node=B, seed=s, epochs=300) for s in SEEDS]
            out[name][B] = {k: ci95([r[k] for r in runs]) for k in ["exposure", "exposure_unw"]}
        else:
            best = None
            for cR in [460, 1024]:
                runs = [run(lambda **k: RepGated(cR=cR, **k), B_per_node=B, seed=s, epochs=300) for s in SEEDS]
                m = np.mean([r["exposure"] for r in runs])
                if best is None or m < best[0]:
                    best = (m, cR, runs)
            out[name][B] = {k: ci95([r[k] for r in best[2]]) for k in ["exposure", "exposure_unw"]} | {"cR": best[1]}
    print(B, out["uniform"][B]["exposure"], out["rg"][B]["exposure"], flush=True)

# Corollary 1: residual-risk ratio GM/AM and budget saving, from steady-state posteriors
ratios, dBs = [], []
for s in SEEDS[:10]:
    holder = []
    def fac(**k):
        r = Race(use_tiers=False, cmin=0, **k); holder.append(r); return r
    run(fac, B_per_node=256, seed=s, epochs=300)
    st = holder[0]
    rv = make_population(100, np.random.default_rng(s))["value"] * beta_ucb(st.f, st.h, 1.0, 9.0, st.kappa)
    L = st.L
    c = waterfill(rv, 256 * 100, L, 0, 1e9)
    Rstar = float((rv * np.exp(-L * c)).sum()); Ru = float((rv * np.exp(-L * 256)).sum())
    am, gm = rv.mean(), math.exp(np.log(rv).mean())
    ratios.append((Rstar / Ru, gm / am)); dBs.append(math.log(am / gm) / L)
out["corollary1"] = dict(ratio_numeric=float(np.mean([a for a, b in ratios])), ratio_gm_am=float(np.mean([b for a, b in ratios])),
                         dB_per_node=float(np.mean(dBs)))
print(out["corollary1"])
json.dump(out, open("results/supp.json", "w"), indent=1)
