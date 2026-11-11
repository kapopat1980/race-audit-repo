"""Element-wise median of the five benchmark runs -> results/bench.json"""
import json, statistics
runs = [json.load(open(f"results/bench_run{i}.json")) for i in (1, 2, 3, 4, 5)]
def med(xs):
    if isinstance(xs[0], dict): return {k: med([x[k] for x in xs]) for k in xs[0]}
    if isinstance(xs[0], (int, float)) and not isinstance(xs[0], bool): return statistics.median(xs)
    return xs[0]
out = med(runs)
def spread(xs):
    if isinstance(xs[0], dict): return max((spread([x[k] for x in xs]) for k in xs[0]), default=0)
    if isinstance(xs[0], float): return (max(xs) - min(xs)) / statistics.median(xs)
    return 0
out["runs"] = 5
out["max_relative_spread"] = spread(runs)
json.dump(out, open("results/bench.json", "w"), indent=1)
print("max relative spread across runs: %.2f" % out["max_relative_spread"])
