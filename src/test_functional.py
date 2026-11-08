"""Functional correctness checks of all implemented schemes (real crypto)."""
import numpy as np, os, json, time
from race_lib import *

rng = np.random.default_rng(7)
n = 4096
honest = Replica(n)
key = MacKey()
tags = MacTagStore(key, honest)

# --- RACE: 12 nodes, two corrupted (1% and 3%), tiers mixed
N = 12
reps = [Replica(n) for _ in range(N)]
reps[3].corrupt(rng.choice(n, 41, replace=False), rng)
reps[8].corrupt(rng.choice(n, 123, replace=False), rng)
cs = np.array([64, 64, 128, 512, 64, 128, 256, 64, 512, 64, 128, 64])
ek = os.urandom(32)
seed, idx, nus = race_challenge(ek, 1, n, int(cs.max()))
resp = [race_prove(reps[i], tags, idx, nus, int(cs[i])) for i in range(N)]
ok, nver, nprf = race_verify_all(key, idx, nus, cs, resp)
print("RACE pass flags:", ok, "tag verifications:", nver, "shared PRF evals:", nprf)
truth = [not any(j in reps[i].corrupted for j in idx[:cs[i]]) for i in range(N)]
assert ok == truth, (ok, truth)

# --- localisation on node 8
bad = [i for i in range(N) if not ok[i]]
for b in bad:
    c = int(cs[b])
    found, q, by = group_test_localise(key, reps[b], tags, idx[:c], nus[:c])
    true_bad = sorted(j for j in idx[:c] if j in reps[b].corrupted)
    _, q2, by2 = group_test_localise(key, reps[b], tags, idx[:c], nus[:c], differential=False)
    print(f"node {b}: localised {found} (truth {true_bad}) diff-queries={q} plain-queries={q2} bytes={by}")
    assert found == true_bad

# --- tamper attempts: forge sigma / mu for a corrupted node
mu, sig = resp[8]
assert not mac_verify(key, mu, (sig + 1) % P, mac_tau(key, idx[:512], nus[:512]))
mu2 = list(mu); mu2[0] = (mu2[0] + 1) % P
assert not mac_verify(key, mu2, sig, mac_tau(key, idx[:512], nus[:512]))
# replay of last epoch's honest response fails on a fresh epoch
seed2, idx2, nus2 = race_challenge(ek, 2, n, 64)
old = resp[0]
assert not mac_verify(key, old[0], old[1], mac_tau(key, idx2[:64], nus2[:64]))
print("forgery / replay rejected")

# --- Empirical vs analytical detection probability (real protocol, 400 trials)
from math import lgamma, exp
def miss(n, d, c):
    if c > n - d: return 0.0
    return exp(lgamma(n - d + 1) - lgamma(n - d - c + 1) - lgamma(n + 1) + lgamma(n - c + 1))
d, c = 41, 64
trials, det = 2000, 0
rep = Replica(n); rep.corrupt(rng.choice(n, d, replace=False), rng)
for t in range(trials):
    s_, I, V = race_challenge(ek, 100 + t, n, c)
    mu, sig = race_prove(rep, tags, I, V, c)
    det += not mac_verify(key, mu, sig, mac_tau(key, I, V))
emp = det / trials
ana = 1 - miss(n, d, c)
print(f"detection d={d}/{n}, c={c}: empirical {emp:.3f} vs analytic {ana:.3f}")

# --- BLS baseline correctness
if HAVE_BLS:
    bk = BlsKey()
    I = idx[:16]; V = nus[:16]
    btags = {j: bls_tag(bk, honest.raw_block(j), j) for j in I}
    mu, sig = bls_prove(honest, btags, I, V)
    assert bls_verify(bk, I, V, mu, sig)
    badrep = Replica(n, corrupted={I[0]: os.urandom(4096)})
    mu, sig = bls_prove(badrep, btags, I, V)
    assert not bls_verify(bk, I, V, mu, sig)
    print("BLS baseline correct")

# --- Merkle baseline correctness
leaves = [sha(honest.raw_block(j)) for j in range(n)]
mt = MerkleTree(leaves)
j = 1234
assert merkle_verify(mt.root, honest.raw_block(j), j, mt.path(j))
assert not merkle_verify(mt.root, os.urandom(4096), j, mt.path(j))
print("Merkle baseline correct")
json.dump({"empirical_detection": emp, "analytic_detection": ana, "trials": trials,
           "d": d, "n": n, "c": c}, open("results/functional_validation.json", "w"), indent=1)
