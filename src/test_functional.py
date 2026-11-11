"""Functional correctness checks of all implemented schemes (real crypto, native kernels)."""
import numpy as np, os, json, time
from math import lgamma, exp
from race_lib import *

rng = np.random.default_rng(7)
n = 4096
honest = Replica(n)
key = MacKey()
tags = MacTagStore(key, honest)
ek = os.urandom(32)

# --- native kernels agree with the pure-Python reference implementation
for j in (0, 17, n - 1):
    assert tags[j] == ref_tag(key, honest.sectors(j), j)
_, I0, V0 = race_challenge(ek, 0, n, 48)
assert mac_prove(honest, tags, I0, V0) == ref_prove(honest, tags, I0, V0)
mu0, sig0 = mac_prove(honest, tags, I0, V0)
assert mac_verify(key, mu0, sig0, mac_tau(key, I0, V0)) and ref_verify(key, mu0, sig0, I0, V0)
ft = key.f_table(n)
assert mac_tau(key, I0, V0, ft) == mac_tau(key, I0, V0)
full = prp_prefix(epoch_seed(ek, 0), n, n)
assert sorted(full.tolist()) == list(range(n)) and (full[:48] == I0).all()      # permutation, nested prefixes
print("native kernels == Python reference")

# --- RACE: 12 nodes, two corrupted (1% and 3%), tiers mixed
N = 12
reps = [Replica(n) for _ in range(N)]
reps[3].corrupt(rng.choice(n, 41, replace=False), rng)
reps[8].corrupt(rng.choice(n, 123, replace=False), rng)
cs = np.array([64, 64, 128, 512, 64, 128, 256, 64, 512, 64, 128, 64])
seed, idx, nus = race_challenge(ek, 1, n, int(cs.max()))
resp = [race_prove(reps[i], tags, idx, nus, int(cs[i])) for i in range(N)]
ok, nver, nprf = race_verify_all(key, idx, nus, cs, resp)
print("RACE pass flags:", ok, "tag verifications:", nver, "shared PRF evals:", nprf)
truth = [not any(int(j) in reps[i].corrupted for j in idx[:cs[i]]) for i in range(N)]
assert ok == truth, (ok, truth)

# --- localisation on failing nodes, then repair from an intact replica and re-audit
bad = [i for i in range(N) if not ok[i]]
for b in bad:
    c = int(cs[b])
    found, q, by, rounds = group_test_localise(key, reps[b], tags, idx[:c], nus[:16 * c])
    true_bad = sorted(int(j) for j in idx[:c] if int(j) in reps[b].corrupted)
    _, q2, _, _ = group_test_localise(key, reps[b], tags, idx[:c], nus[:16 * c], differential=False)
    print(f"node {b}: localised {len(found)} blocks (exact={found == true_bad}) diff-queries={q} plain-queries={q2} round-trips={rounds} bytes={by}")
    assert verify_block(key, found[0], honest.raw_block(found[0]), tags.arr[16 * found[0]:16 * found[0] + 16])
    assert not verify_block(key, found[0], reps[b].raw_block(found[0]), tags.arr[16 * found[0]:16 * found[0] + 16])
    assert found == true_bad
    for j in found:
        reps[b].repair(j, honest.raw_block(j))
    mu, sig = race_prove(reps[b], tags, idx, nus, c)
    assert mac_verify(key, mu, sig, mac_tau(key, idx[:c], nus[:16 * c]))
print("localised blocks repaired; re-audit passes")

# --- tamper attempts: forge sigma / mu, replay an old response
rep8 = Replica(n); rep8.corrupt(rng.choice(n, 123, replace=False), rng)
mu, sig = race_prove(rep8, tags, idx, nus, 512)
tau = mac_tau(key, idx[:512], nus[:16 * 512])
assert not mac_verify(key, mu, sig, tau)
assert not mac_verify(key, mu, fe_bytes((fe_int(sig) + 1) % P), tau)
mu2 = fe_bytes((fe_int(mu[:16]) + 1) % P) + mu[16:]
assert not mac_verify(key, mu2, sig, tau)
seed2, idx2, nus2 = race_challenge(ek, 2, n, 64)
old = resp[0]
assert not mac_verify(key, old[0], old[1], mac_tau(key, idx2, nus2))
# replacement attack: answer with a different intact block in place of a requested one
swapped = idx[:64].copy(); swapped[0] = idx[100]
mu_s, sig_s = mac_prove(honest, tags, swapped, nus[:16 * 64])
assert not mac_verify(key, mu_s, sig_s, mac_tau(key, idx[:64], nus[:16 * 64]))
print("forgery / replay / replacement rejected")

# --- empirical vs analytical detection probability (real protocol)
def miss(n, d, c):
    if c > n - d: return 0.0
    return exp(lgamma(n - d + 1) - lgamma(n - d - c + 1) - lgamma(n + 1) + lgamma(n - c + 1))
d, c = 41, 64
trials, det = 20000, 0
rep = Replica(n); rep.corrupt(rng.choice(n, d, replace=False), rng)
for t in range(trials):
    s_, I, V = race_challenge(ek, 100 + t, n, c)
    mu, sig = mac_prove(rep, tags, I, V)
    det += not mac_verify(key, mu, sig, mac_tau(key, I, V))
emp = det / trials
ana = 1 - miss(n, d, c)
se = (ana * (1 - ana) / trials) ** 0.5
print(f"detection d={d}/{n}, c={c}: empirical {emp:.4f} vs analytic {ana:.4f} (binomial s.e. {se:.4f})")
assert abs(emp - ana) < 4 * se

# --- BLS baseline correctness
if HAVE_BLS:
    bk = BlsKey()
    I, V = idx[:16], nus[:16 * 16]
    btags = {int(j): bls_tag(bk, honest.raw_block(int(j)), int(j)) for j in I}
    mu, sig = bls_prove(honest, btags, I, V)
    assert bls_verify(bk, I, V, mu, sig)
    badrep = Replica(n); badrep.corrupt([int(I[0])], rng)
    mu, sig = bls_prove(badrep, btags, I, V)
    assert not bls_verify(bk, I, V, mu, sig)
    print("BLS baseline correct")

# --- EDI-C baseline correctness (small parameters for speed, then the benchmark parameters)
if HAVE_EDIC:
    for lp, lq in ((1024, 160), (2048, 256)):
        prm = EdicParams(lp, lq)
        assert (prm.p - 1) % prm.q == 0 and all(powmod(g, prm.q, prm.p) == 1 for g in prm.g[:3])
        I, V = idx[:8], nus[:16 * 8]
        hashes = {int(j): edic_hash_block(prm, honest.raw_block(int(j))) for j in I}
        assert edic_verify(prm, honest, I, V, edic_prove(prm, hashes, I, V))
        badrep = Replica(n); badrep.corrupt([int(I[0])], rng)
        bad_hashes = dict(hashes); bad_hashes[int(I[0])] = edic_hash_block(prm, badrep.raw_block(int(I[0])))
        assert not edic_verify(prm, honest, I, V, edic_prove(prm, bad_hashes, I, V))
    print("EDI-C baseline correct")

# --- Merkle baseline correctness
leaves = [sha(honest.raw_block(j)) for j in range(n)]
mt = MerkleTree(leaves)
j = 1234
assert merkle_verify(mt.root, honest.raw_block(j), j, mt.path(j))
assert not merkle_verify(mt.root, os.urandom(4096), j, mt.path(j))
print("Merkle baseline correct")
json.dump({"empirical_detection": emp, "analytic_detection": ana, "trials": trials, "binomial_se": se,
           "d": d, "n": n, "c": c}, open("results/functional_validation.json", "w"), indent=1)
