"""All experiments.  RACE hyper-parameters (lambda=0.99, kappa=1, rho*=0.005) were selected on
tuning seeds 0-9 only; every number reported here uses held-out seeds 100-129."""
import json, time, os, math
import numpy as np
from scipy import stats
from sim import *
from race_lib import (MacKey, MacTagStore, Replica, race_challenge, group_test_localise,
                      waterfill, S_SECT, FE_BYTES)

SEEDS = list(range(100, 130))
EPOCHS = 300
R = {"seeds": SEEDS, "epochs": EPOCHS}
KEYS = ("exposure", "exposure_unw", "mttd", "p95_ttd", "det_frac", "samples_per_epoch", "exposure_attack")

def summarise(runs):
    out = {k: ci95([r[k] for r in runs]) for k in KEYS}
    out["raw_exposure"] = [r["exposure"] for r in runs]
    out["raw_exposure_unw"] = [r["exposure_unw"] for r in runs]
    out["raw_attack"] = [r["exposure_attack"] for r in runs]
    out["trace"] = np.mean([r["trace"] for r in runs], axis=0).tolist()
    return out

RG_CR = [128, 256, 460, 1024]

def rg_tuned(B, **runkw):
    """Rep-Gated with its sample size c_R tuned per scenario (best mean exposure) -
    deliberately favourable to the baseline."""
    best = None
    for cR in RG_CR:
        if cR > 4 * B:      # would audit < 1/4 of nodes - never better, skip
            continue
        runs = [run(lambda **k: RepGated(cR=cR, **k), B_per_node=B, seed=s, **runkw) for s in SEEDS]
        key = "exposure"   # tuned on TOTAL exposure: what the vendor actually incurs
        m = np.mean([r[key] for r in runs])
        if best is None or m < best[0]:
            best = (m, cR, runs)
    s = summarise(best[2]); s["cR"] = best[1]
    return s

ABL = {"RACE w/o UCB (kappa=0)": dict(kappa=0.0),
       "RACE w/o forgetting (lambda=1)": dict(lam=1.0),
       "RACE w/o floor (c_min=0)": dict(cmin=0),
       "RACE w/o value weights": dict(use_value=False),
       "RACE continuous (no tiers)": dict(use_tiers=False)}

import sys
SECS = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"E1","E2","E3","E4","E5"}
def default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    raise TypeError(type(o))
def dump(name, obj):
    json.dump(obj, open(f"results/{name}.json", "w"), indent=1, default=default)
okw = dict(epochs=400, mix=(0.8, 0.2, 0.0), onoff=10, onoff_start=150, onoff_rho=0.002)
t0 = time.time()
if "E1" in SECS:
    # ---------------- E1 budget sweep (static heterogeneous population)
    E1 = {}
    for B in [32, 64, 128, 256, 512]:
        E1[B] = {"Uniform": summarise([run(lambda **k: Uniform(**k), B_per_node=B, epochs=EPOCHS, seed=s) for s in SEEDS]),
                 "Rep-Gated (tuned)": rg_tuned(B, epochs=EPOCHS),
                 "RACE-Audit": summarise([run(lambda **k: Race(**k), B_per_node=B, epochs=EPOCHS, seed=s) for s in SEEDS])}
        if B in (64, 256):
            for name, kw in ABL.items():
                E1[B][name] = summarise([run(lambda **k: Race(**kw, **k), B_per_node=B, epochs=EPOCHS, seed=s) for s in SEEDS])
        print("E1", B, {k: round(v["exposure"][0], 3) for k, v in E1[B].items()}, round(time.time() - t0), flush=True)
    R["E1_budget"] = E1

    def wilcox(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if np.allclose(a, b):
            return float("nan")
        return float(stats.wilcoxon(a, b, alternative="less").pvalue)

    tests = {}
    for B in E1:
        r = np.array(E1[B]["RACE-Audit"]["raw_exposure"])
        ru = np.array(E1[B]["RACE-Audit"]["raw_exposure_unw"])
        for base in ["Uniform", "Rep-Gated (tuned)"]:
            b = np.array(E1[B][base]["raw_exposure"]); bu = np.array(E1[B][base]["raw_exposure_unw"])
            tests[f"{B}|{base}"] = dict(p=wilcox(r, b), mean_reduction=float(1 - r.mean() / b.mean()),
                                        p_unw=wilcox(ru, bu), mean_reduction_unw=float(1 - ru.mean() / bu.mean()))
    R["E1_tests"] = tests

    def budget_to_match(name, target, key="exposure"):
        Bs = sorted(E1)
        ex = [E1[b][name][key][0] for b in Bs]
        for i in range(len(Bs) - 1):
            if ex[i] >= target >= ex[i + 1]:
                x0, x1 = math.log(Bs[i]), math.log(Bs[i + 1])
                y0, y1 = math.log(ex[i]), math.log(ex[i + 1])
                return math.exp(x0 + (math.log(target) - y0) * (x1 - x0) / (y1 - y0))
        return float("nan")
    R["E1_equal_assurance_budget"] = {
        key: {n: budget_to_match(n, E1[256]["RACE-Audit"][key][0], key) for n in ["Uniform", "Rep-Gated (tuned)", "RACE-Audit"]}
        for key in ["exposure", "exposure_unw"]}
    print("equal assurance", R["E1_equal_assurance_budget"], flush=True)

    dump("res_E1", {k: R[k] for k in ['E1_budget', 'E1_tests', 'E1_equal_assurance_budget']})
    print("saved E1", flush=True)
if "E2" in SECS:
    # ---------------- E2 on-off (trust-building) attack: honest for 150 epochs, then stealthy corruption
    E2 = {}
    for B in [64, 128, 256]:
        E2[B] = {"Uniform": summarise([run(lambda **k: Uniform(**k), B_per_node=B, seed=s, **okw) for s in SEEDS]),
                 "Rep-Gated (tuned)": rg_tuned(B, **okw),
                 "RACE-Audit": summarise([run(lambda **k: Race(**k), B_per_node=B, seed=s, **okw) for s in SEEDS])}
        for name, kw in ABL.items():
            if "value" in name or "continuous" in name:
                continue
            E2[B][name] = summarise([run(lambda **k: Race(**kw, **k), B_per_node=B, seed=s, **okw) for s in SEEDS])
        print("E2", B, {k: round(v["exposure_attack"][0], 3) for k, v in E2[B].items()}, flush=True)
    R["E2_onoff"] = E2

    dump("res_E2", {k: R[k] for k in ['E2_onoff']})
    print("saved E2", flush=True)
if "E3" in SECS:
    # ---------------- E3 whitewashing (identity reset after each detection)
    E3 = {}
    for B in [64, 256]:
        E3[B] = {"Uniform": summarise([run(lambda **k: Uniform(**k), B_per_node=B, seed=s, epochs=EPOCHS, whitewash=True) for s in SEEDS]),
                 "Rep-Gated (tuned)": rg_tuned(B, epochs=EPOCHS, whitewash=True),
                 "RACE-Audit": summarise([run(lambda **k: Race(**k), B_per_node=B, seed=s, epochs=EPOCHS, whitewash=True) for s in SEEDS])}
        for name, kw in ABL.items():
            if "value" in name or "continuous" in name:
                continue
            E3[B][name] = summarise([run(lambda **k: Race(**kw, **k), B_per_node=B, seed=s, epochs=EPOCHS, whitewash=True) for s in SEEDS])
        print("E3", B, {k: round(v["exposure"][0], 3) for k, v in E3[B].items()}, flush=True)
    R["E3_whitewash"] = E3

    dump("res_E3", {k: R[k] for k in ['E3_whitewash']})
    print("saved E3", flush=True)
if "E4" in SECS:
    # ---------------- E4 sensitivity (B = 128)
    E4 = {"kappa": {}, "lambda": {}, "cmin": {}, "rho_star": {}}
    for k_ in [0.0, 0.5, 1.0, 2.0, 3.0]:
        E4["kappa"][k_] = summarise([run(lambda **kw: Race(kappa=k_, **kw), B_per_node=128, seed=s, epochs=EPOCHS, whitewash=True) for s in SEEDS])
    for lam in [0.9, 0.95, 0.97, 0.99, 1.0]:
        E4["lambda"][lam] = summarise([run(lambda **kw: Race(lam=lam, **kw), B_per_node=128, seed=s, **okw) for s in SEEDS])
    for cm in [0, 16, 32, 64, 96]:
        E4["cmin"][cm] = summarise([run(lambda **kw: Race(cmin=cm, **kw), B_per_node=128, seed=s, epochs=EPOCHS, whitewash=True) for s in SEEDS])
    for rs in [0.002, 0.005, 0.01, 0.02]:
        E4["rho_star"][rs] = summarise([run(lambda **kw: Race(rho_star=rs, **kw), B_per_node=128, seed=s, **okw) for s in SEEDS])
    print("E4 done", round(time.time() - t0), flush=True)
    R["E4_sensitivity"] = E4

    dump("res_E4", {k: R[k] for k in ['E4_sensitivity']})
    print("saved E4", flush=True)
if "E5" in SECS:
    # ---------------- E5 cost per epoch vs N (steady-state RACE allocation; uniform c=256 for baselines)
    E5 = {}
    for N in [10, 50, 100, 200, 500, 1000]:
        rr = run(lambda **k: Race(**k), N=N, B_per_node=256, epochs=150, seed=101, record_alloc=True)
        c_race = rr["allocs"][-1]
        exp_fail = rr["episodes"] / 150 * rr["det_frac"]
        nf = int(round(exp_fail))
        cu = np.full(N, 256.0)
        cr, cu_ = epoch_costs(c_race, n_fail_groups=nf), epoch_costs(cu, n_fail_groups=nf)
        row = {"RACE-Audit": cr["RACE-Audit"]}
        for s in ["U-HMAC", "U-BLS", "B-BLS", "U-MHT"]:
            row[s] = cu_[s]
        fails = np.random.default_rng(3).choice(N, nf, replace=False) if nf else []
        lc, chk = bbls_localisation_cost(cu, fails)
        row["B-BLS+loc"] = (row["B-BLS"][0] + (lc if nf else 0.0),) + tuple(row["B-BLS"][1:])
        row["meta"] = dict(race_samples=float(c_race.sum()), uniform_samples=float(cu.sum()),
                           race_tiers=sorted(set(map(int, c_race))), race_max_c=float(c_race.max()),
                           failures_per_epoch=exp_fail, bbls_loc_checks=chk if nf else 0)
        E5[N] = row
        print("E5", N, {k: (round(v[0], 4), round(v[3] / 1e3, 1)) for k, v in row.items() if k != "meta"}, flush=True)
    R["E5_cost"] = E5


    # ---------------- E6 localisation (real protocol, c = 512)
    key = MacKey(); hon = Replica(N_BLOCKS); tags = MacTagStore(key, hon)
    E6 = {}
    ek = os.urandom(32)
    c = 512
    for ds in [1, 2, 4, 8, 16, 32]:
        qd, qp, tim = [], [], []
        for rep in range(10):
            _, idx, nus = race_challenge(ek, 1000 * ds + rep, N_BLOCKS, c)
            rng = np.random.default_rng(rep)
            r = Replica(N_BLOCKS)
            bad = rng.choice(idx, ds, replace=False)
            r.corrupt(bad, rng)
            t1 = time.perf_counter()
            found, q, _ = group_test_localise(key, r, tags, idx, nus)
            tim.append(time.perf_counter() - t1)
            assert sorted(found) == sorted(int(b) for b in bad)
            _, q2, _ = group_test_localise(key, r, tags, idx, nus, differential=False)
            qd.append(q); qp.append(q2)
        resp = (S_SECT + 1) * FE_BYTES
        E6[ds] = dict(q_diff=ci95(qd), q_plain=ci95(qp), bound=ds * math.ceil(math.log2(c)),
                      bytes_diff=float(np.mean(qd)) * resp, bytes_plain=float(np.mean(qp)) * resp,
                      bytes_reveal=c * (4096 + 16), bytes_mht=c * (4096 + 14 * 32), time_diff=float(np.median(tim)))
        print("E6", ds, E6[ds]["q_diff"], E6[ds]["q_plain"], flush=True)
    R["E6_localisation"] = E6

    # ---------------- E7 allocator overhead
    E7 = {}
    for N in [100, 1000, 10000, 100000]:
        r = np.random.default_rng(0).random(N)
        t1 = time.perf_counter()
        for _ in range(5):
            waterfill(r, 256 * N, -math.log(0.99), 32, 1024)
        E7[N] = (time.perf_counter() - t1) / 5
    R["E7_allocator_s"] = E7
    print("E7", E7)

    # ---------------- allocation snapshot for figure + tier utilisation
    rr = run(lambda **k: Race(**k), N=100, B_per_node=256, epochs=300, seed=101, record_alloc=True)
    R["snapshot"] = dict(alloc=rr["allocs"][-1], types=rr["pop"]["types"], value=rr["pop"]["value"],
                         mean_alloc=rr["allocs"][150:].mean(axis=0))
    R["tier_budget_utilisation"] = {B: run(lambda **k: Race(**k), B_per_node=B, epochs=100, seed=102)["samples_per_epoch"] / (100 * B)
                                    for B in [64, 128, 256, 512]}

    dump("res_E5", {k: R[k] for k in ['E5_cost', 'E6_localisation', 'E7_allocator_s', 'snapshot', 'tier_budget_utilisation']})
    print("saved E5", flush=True)
