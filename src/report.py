import json, math
R = {}
for _s in ["E1", "E2", "E3", "E4", "E5"]:
    R.update(json.load(open(f"results/res_{_s}.json"))); S = json.load(open("results/supp.json"))
B = json.load(open("results/bench.json")); F = json.load(open("results/functional_validation.json"))
f = lambda x: f"{x[0]:.3f} ± {x[1]:.3f}"
print("== bench"); print({k: v for k, v in B.items() if k not in ("per_c", "multiexp_per_pt")})
for k, v in B["per_c"].items(): print(k, {c: round(t * 1e3, 3) for c, t in v.items()})
print("multiexp", B["multiexp_per_pt"]); print("functional", F)
print("== E1")
for b, d in R["E1_budget"].items():
    for n, s in d.items():
        print(b, n, "exp", f(s["exposure"]), "unw", f(s["exposure_unw"]), "mttd", f(s["mttd"]), "p95", f(s["p95_ttd"]),
              "det", f(s["det_frac"]), "samp", round(s["samples_per_epoch"][0]), "cR", s.get("cR"))
print(json.dumps(R["E1_tests"], indent=0))
print("eq", R["E1_equal_assurance_budget"])
print("supp", {k: v for k, v in S.items()})
for key in ["E2_onoff", "E3_whitewash"]:
    print("==", key)
    for b, d in R[key].items():
        for n, s in d.items():
            print(b, n, "total", f(s["exposure"]), "attack", f(s["exposure_attack"]), "mttd", f(s["mttd"]), "p95", f(s["p95_ttd"]), "cR", s.get("cR"))
print("== E4")
for k, d in R["E4_sensitivity"].items():
    for x, s in d.items(): print(k, x, "exp", f(s["exposure"]), "att", f(s["exposure_attack"]))
print("== E5")
for N, row in R["E5_cost"].items():
    print(N, row["meta"])
    for s, v in row.items():
        if s.startswith("RACE-Audit (measured"): print("   ", s, v)
        elif s != "meta": print("   ", s, "ver %.4f s | edge tot %.3f s | edge max %.2f ms | down %.1f kB | up %d B" % (v[0], v[1], v[2] * 1e3, v[3] / 1e3, v[4]))
print("== E6")
for d, v in R["E6_localisation"].items(): print(d, v)
print("E7", R["E7_allocator_s"]); print("tier util", R["tier_budget_utilisation"])
print("== E8 field-calibrated fleet")
E8 = json.load(open("results/res_E8.json"))["cells"]
for k, c in E8.items():
    print(k, "U", f(c["Uniform"]["exposure"]), "RG", f(c["Rep-Gated"]["exposure"]), "cR", c["Rep-Gated"]["cR"], "RACE", f(c["RACE-Audit"]["exposure"]),
          "red vs U %.3f" % c["reduction_vs_uniform"], "p(RACE better) %.2e" % c["p_race_better"], "p(U better) %.2e" % c["p_uniform_better"])
print("== networked deployment")
DP = json.load(open("results/deploy_summary.json"))
for n in ("Uniform", "RACE-Audit"):
    print(n, {k: v for k, v in DP["detect"][n].items() if k != "per_seed"})
    for r in DP["detect"][n]["per_seed"]: print("   ", r)
print("reduction deploy %.3f sim %.3f" % (DP["detect"]["reduction_deploy"], DP["detect"]["reduction_sim"]))
for k, v in DP["cost"].items(): print(k, v)
