"""Collects the networked-deployment runs and compares their detection outcome with the simulator
run on the same node populations (same seed -> same behaviour classes and value weights; the
fault streams differ, so the simulator is averaged over 30 fault streams per population)."""
import json, glob
import numpy as np
from sim import run, Uniform, Race, ci95

out = {"detect": {}, "cost": {}}
for strat, fac, name in (("race", Race, "RACE-Audit"), ("uniform", Uniform, "Uniform")):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"results/deploy_detect_{strat}_*.json"))]
    rows = []
    for r in runs:
        sims = [run(fac, N=r["N"], B_per_node=r["Bn"], epochs=r["T"], seed=r["seed"], fault_seed=50_000 + 97 * r["seed"] + k) for k in range(30)]
        se = np.array([s["exposure"] for s in sims])
        rows.append(dict(seed=r["seed"], deploy=r["exposure"], sim_mean=float(se.mean()), sim_lo=float(np.percentile(se, 2.5)),
                         sim_hi=float(np.percentile(se, 97.5)), sim_mttd=float(np.mean([s["mttd"] for s in sims])), deploy_mttd=r["mttd"]))
    agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if isinstance(runs[0][k], (int, float)) and k not in ("seed",)}
    agg.update(seeds=[r["seed"] for r in runs], episodes_total=int(sum(r["episodes"] for r in runs)), detections_total=int(sum(r["detections"] for r in runs)),
               false_positives_total=int(sum(r["false_positives"] for r in runs)), repairs_clean_total=int(sum(r["repairs_clean"] for r in runs)),
               exposure_ci=ci95([r["exposure"] for r in runs]), sim_exposure_mean=float(np.mean([x["sim_mean"] for x in rows])),
               within_sim_band=int(sum(x["sim_lo"] <= x["deploy"] <= x["sim_hi"] for x in rows)), per_seed=rows)
    out["detect"][name] = agg
d = out["detect"]
out["detect"]["reduction_deploy"] = 1 - d["RACE-Audit"]["exposure"] / d["Uniform"]["exposure"]
out["detect"]["reduction_sim"] = 1 - d["RACE-Audit"]["sim_exposure_mean"] / d["Uniform"]["sim_exposure_mean"]
for fam in ("mac", "bls", "edic"):
    c = json.load(open(f"results/deploy_cost_{fam}.json"))
    meta = c.pop("meta"); out["cost"].update(c); out["cost"].setdefault("meta", {})[fam] = meta
json.dump(out, open("results/deploy_summary.json", "w"), indent=1)
for name in ("Uniform", "RACE-Audit"):
    a = d[name]
    print(name, "deploy exposure %.3f (sim %.3f)  mttd %.2f  episodes %d detections %d false alarms %d clean repairs %d  in-band %d/%d" %
          (a["exposure"], a["sim_exposure_mean"], a["mttd"], a["episodes_total"], a["detections_total"], a["false_positives_total"], a["repairs_clean_total"], a["within_sim_band"], len(a["per_seed"])))
print("reduction deploy %.3f sim %.3f" % (d["reduction_deploy"], d["reduction_sim"]))
for k, v in out["cost"].items():
    if k not in ("meta", "repair"): print(k, {m: round(v[m][0], 5) for m in ("verifier_s", "wall_s", "node_cpu_total", "node_cpu_max", "up", "down")})
print(out["cost"].get("repair"))
