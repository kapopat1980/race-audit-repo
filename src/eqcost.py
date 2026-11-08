"""Equal-assurance budgets (E1 + supplementary larger budgets) and the per-epoch
cost each scheme incurs to reach RACE-Audit's exposure at B/N = 256 (N = 100)."""
import json, math
import numpy as np
from sim import epoch_costs
E1 = json.load(open("results/res_E1.json"))["E1_budget"]; S = json.load(open("results/supp.json"))
E5 = json.load(open("results/res_E5.json"))["E5_cost"]
out = {}
for key in ["exposure", "exposure_unw"]:
    target = E1["256"]["RACE-Audit"][key][0]
    res = {}
    for name, sk in [("Uniform", "uniform"), ("Rep-Gated (tuned)", "rg")]:
        pts = sorted([(int(b), E1[b][name][key][0]) for b in E1] + [(int(b), v[key][0]) for b, v in S[sk].items()])
        Beq = float("nan")
        for (b0, y0), (b1, y1) in zip(pts, pts[1:]):
            if y0 >= target >= y1:
                Beq = math.exp(math.log(b0) + (math.log(target) - math.log(y0)) * (math.log(b1) - math.log(b0)) / (math.log(y1) - math.log(y0)))
                break
        res[name] = Beq
    out[key] = {"target": target, "budget": res}
Beq = out["exposure"]["budget"]["Uniform"]
c = epoch_costs(np.full(100, Beq), n_fail_groups=int(round(E5["100"]["meta"]["failures_per_epoch"])))
out["cost_at_equal_assurance"] = {"RACE-Audit@256": E5["100"]["RACE-Audit"]} | {f"{k}@{Beq:.0f}": v for k, v in c.items() if k != "RACE-Audit"}
json.dump(out, open("results/eqcost.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
