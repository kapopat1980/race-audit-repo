"""Measured (wall-clock) per-node costs of every scheme on this machine.
Produces results/bench.json used by the cost model.  All numbers are medians."""
import time, json, os, statistics, platform
import numpy as np
from race_lib import *

n = 16384                     # 64 MiB replica of 4 KiB blocks
CS = [64, 128, 256, 512, 1024]
REPS = 5
honest = Replica(n)
key = MacKey()
tags = MacTagStore(key, honest)
ek = os.urandom(32)

def med(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

out = {"machine": platform.processor() or platform.machine(), "python": platform.python_version(),
       "n_blocks": n, "block_bytes": BLOCK_BYTES, "s_mac": S_SECT}

# warm block cache and tags for the largest challenge (TagGen is offline)
_, idx, nus = race_challenge(ek, 0, n, max(CS))
for j in idx:
    honest.sectors(j); tags[j]

# primitives
out["t_prf"] = med(lambda: [key.f(j) for j in range(2000)]) / 2000
out["t_sha_4k"] = med(lambda: [sha(b"x" * 4096) for _ in range(2000)]) / 2000
out["t_sha_64"] = med(lambda: [sha(b"x" * 64) for _ in range(20000)]) / 20000
mu0, sig0 = mac_prove(honest, tags, idx[:64], nus[:64])
out["t_mac_check"] = med(lambda: [mac_verify(key, mu0, sig0, 0) for _ in range(200)]) / 200
out["t_eq_compare"] = med(lambda: [tuple(mu0) == tuple(mu0) for _ in range(2000)]) / 2000

res = {"mac_prove": {}, "mac_tau": {}, "bls_prove": {}, "bls_verify": {}, "bls_verify_batch_extra": {},
       "mht_prove": {}, "mht_verify": {}}
for c in CS:
    I, V = idx[:c], nus[:c]
    res["mac_prove"][c] = med(lambda: mac_prove(honest, tags, I, V))
    res["mac_tau"][c] = med(lambda: mac_tau(key, I, V))
    print("mac", c, res["mac_prove"][c], res["mac_tau"][c], flush=True)

if HAVE_BLS:
    bk = BlsKey()
    btags = {}
    for j in idx:
        btags[j] = bls_tag(bk, honest.raw_block(j), j)
    Hcache = {j: bls_H(j) for j in idx}
    def verify_cached(I, V, mu, sig):
        pts = [Hcache[j] for j in I] + bk.u
        sc = [scal(x) for x in V] + [scal(m) for m in mu]
        agg = bls.G1Point.multiexp_unchecked(pts, sc)
        return bls.GT.pairing(sig, bk.g2) == bls.GT.pairing(agg, bk.v)
    for c in CS:
        I, V = idx[:c], nus[:c]
        mu, sig = bls_prove(honest, btags, I, V)
        assert verify_cached(I, V, mu, sig)
        res["bls_prove"][c] = med(lambda: bls_prove(honest, btags, I, V), 3)
        res["bls_verify"][c] = med(lambda: verify_cached(I, V, mu, sig), 3)
        print("bls", c, res["bls_prove"][c], res["bls_verify"][c], flush=True)
    g1 = bls.G1Point(); g2 = bls.G2Point()
    out["t_pairing"] = med(lambda: [bls.GT.pairing(g1, g2) for _ in range(20)]) / 20
    # multiexp cost per point at large sizes (for batch verification of many nodes)
    pts = [Hcache[j] for j in idx]
    sc = [scal(x) for x in nus]
    out["t_multiexp_per_pt_1024"] = med(lambda: bls.G1Point.multiexp_unchecked(pts, sc), 3) / len(pts)
    out["t_scal_conv"] = med(lambda: [scal(x) for x in nus], 3) / len(nus)
    out["t_g1_add"] = med(lambda: [g1 + g1 for _ in range(2000)]) / 2000

# Merkle
leaves = [sha(honest.raw_block(j)) for j in range(n)]
mt = MerkleTree(leaves)
for c in CS:
    I = idx[:c]
    blocks = [honest.raw_block(j) for j in I]
    paths = [mt.path(j) for j in I]
    res["mht_prove"][c] = med(lambda: [(honest.raw_block(j), mt.path(j)) for j in I])
    res["mht_verify"][c] = med(lambda: all(merkle_verify(mt.root, b, j, p) for b, j, p in zip(blocks, I, paths)))
    print("mht", c, res["mht_prove"][c], res["mht_verify"][c], flush=True)

out["per_c"] = res
json.dump(out, open("results/bench.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_c"}, indent=1))
