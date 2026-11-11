"""Direct wall-clock measurement of the RACE-Audit verifier (challenge expansion + Algorithm 2) on real
proofs at the steady-state allocations of experiment E5; five batches of 201 repetitions after a warm-up,
median of the batch medians.  Updates results/res_E5.json."""
import json, os, time
import numpy as np
from sim import run, Race, N_BLOCKS
from race_lib import MacKey, MacTagStore, Replica, race_challenge, expand_challenge, race_prove, race_verify_all

key = MacKey(); hon = Replica(N_BLOCKS); tags = MacTagStore(key, hon); ek = os.urandom(32)
R = json.load(open("results/res_E5.json"))
for N in sorted(map(int, R["E5_cost"])):
    cv = [int(x) for x in run(lambda **k: Race(**k), N=N, B_per_node=256, epochs=150, seed=101, record_alloc=True)["allocs"][-1] if x > 0]
    seed, idx, nus = race_challenge(ek, 7, N_BLOCKS, max(cv))
    proofs = {c: race_prove(hon, tags, idx, nus, c) for c in set(cv)}
    def once():
        resp = [(bytes(bytearray(proofs[c][0])), bytes(bytearray(proofs[c][1]))) for c in cv]     # distinct objects, as off the wire
        t = time.perf_counter()
        i2, n2 = expand_challenge(seed, N_BLOCKS, max(cv))
        ok, _, _ = race_verify_all(key, i2, n2, cv, resp)
        t = time.perf_counter() - t
        assert all(ok)
        return t
    for _ in range(50): once()
    batches = [float(np.median([once() for _ in range(201)])) for _ in range(5)]
    R["E5_cost"][str(N)]["RACE-Audit (measured verifier)"] = float(np.median(batches))
    R["E5_cost"][str(N)]["RACE-Audit (measured verifier, batches)"] = batches
    print(N, "model %.3f ms" % (1e3 * R["E5_cost"][str(N)]["RACE-Audit"][0]), "measured", [round(1e3 * b, 3) for b in batches], flush=True)
json.dump(R, open("results/res_E5.json", "w"), indent=1)
