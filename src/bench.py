"""Measured (wall-clock) per-node costs of every scheme on this machine.
Produces results/bench.json used by the cost model.  All numbers are medians.
All inner loops are native (race_native.c / OpenSSL, arkworks, GMP)."""
import time, json, os, statistics, platform
import numpy as np
from race_lib import *

n = 16384                     # 64 MiB replica of 4 KiB blocks
CS = [32, 64, 128, 256, 512, 1024]
honest = Replica(n)
key = MacKey()
tags = MacTagStore(key, honest)
ek = os.urandom(32)

def med(fn, reps=31):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

out = {"machine": platform.processor() or platform.machine(), "python": platform.python_version(),
       "n_blocks": n, "block_bytes": BLOCK_BYTES, "s_mac": S_SECT}
seed, idx, nus = race_challenge(ek, 0, n, max(CS))
ftab = key.f_table(n)

# ---- primitives
ar = np.arange(n, dtype=np.uint32)
out["t_prf"] = med(lambda: key.f_table(n)) / n
out["t_sha_4k"] = med(lambda: [sha(b"x" * 4096) for _ in range(2000)]) / 2000
mu0, sig0 = mac_prove(honest, tags, idx[:64], nus[:16 * 64])
tau0 = mac_tau(key, idx[:64], nus[:16 * 64])
out["t_mac_check"] = med(lambda: [mac_verify(key, mu0, sig0, tau0) for _ in range(500)]) / 500
def group_cost():
    rs = [(bytes(bytearray(mu0)), bytes(bytearray(sig0))) for _ in range(1000)]     # distinct objects, as off the wire
    t0 = time.perf_counter()
    g = {}
    for i, r in enumerate(rs):
        g.setdefault((64, r[0], r[1]), []).append(i)
    return (time.perf_counter() - t0) / 1000
out["t_group"] = statistics.median(group_cost() for _ in range(21))
t0 = time.perf_counter(); mac_taggen(key, honest); out["t_taggen_file"] = time.perf_counter() - t0

res = {k: {} for k in ["expand", "mac_prove", "mac_tau", "mac_tau_cached", "mac_tau_prefix", "bls_prove", "bls_verify",
                       "mht_prove", "mht_verify", "edic_prove", "edic_verify", "edic1024_prove", "edic1024_verify"]}
for c in CS:
    I, V = idx[:c], nus[:16 * c]
    res["expand"][c] = med(lambda: expand_challenge(seed, n, c))
    res["mac_prove"][c] = med(lambda: mac_prove(honest, tags, I, V))
    res["mac_tau"][c] = med(lambda: mac_tau(key, I, V))
    res["mac_tau_cached"][c] = med(lambda: mac_tau(key, I, V, ftab))
    print("mac", c, {k: round(res[k][c] * 1e6, 1) for k in ["expand", "mac_prove", "mac_tau", "mac_tau_cached"]}, flush=True)

if HAVE_BLS:
    bk = BlsKey()
    btags = {int(j): bls_tag(bk, honest.raw_block(int(j)), int(j)) for j in idx}
    Hcache = {int(j): bls_H(int(j)) for j in idx}
    def verify_cached(I, V, mu, sig):
        pts = [Hcache[int(j)] for j in I] + bk.u
        sc = [scal(x) for x in nu_ints(V)] + [scal(m) for m in mu]
        agg = bls.G1Point.multiexp_unchecked(pts, sc)
        return bls.GT.pairing(sig, bk.g2) == bls.GT.pairing(agg, bk.v)
    for c in CS:
        I, V = idx[:c], nus[:16 * c]
        mu, sig = bls_prove(honest, btags, I, V)
        assert verify_cached(I, V, mu, sig)
        res["bls_prove"][c] = med(lambda: bls_prove(honest, btags, I, V), 7)
        res["bls_verify"][c] = med(lambda: verify_cached(I, V, mu, sig), 7)
        print("bls", c, round(res["bls_prove"][c] * 1e3, 2), round(res["bls_verify"][c] * 1e3, 2), flush=True)
    g1 = bls.G1Point(); g2 = bls.G2Point()
    out["t_pairing"] = med(lambda: [bls.GT.pairing(g1, g2) for _ in range(20)], 7) / 20
    pts = [Hcache[int(j)] for j in idx] * 64
    scs = [scal(x) for x in nu_ints(nus)] * 64
    out["multiexp_per_pt"] = {}
    for m in [1024, 4096, 16384, 65536]:
        out["multiexp_per_pt"][m] = med(lambda: bls.G1Point.multiexp_unchecked(pts[:m], scs[:m]), 3) / m
    vv = nu_ints(nus)
    out["t_scal_conv"] = med(lambda: [scal(x) for x in vv], 7) / len(vv)
    out["t_g1_add"] = med(lambda: [g1 + g1 for _ in range(2000)]) / 2000

if HAVE_EDIC:
    for tag, (lp, lq) in (("edic", (2048, 256)), ("edic1024", (1024, 160))):
        prm = EdicParams(lp, lq)
        t0 = time.perf_counter()
        hashes = {int(j): edic_hash_block(prm, honest.raw_block(int(j))) for j in idx}
        out[f"t_{tag}_hash_block"] = (time.perf_counter() - t0) / len(idx)
        out[f"{tag}_proof_bytes"] = prm.proof_bytes
        out[f"{tag}_m"] = prm.m
        for c in CS:
            I, V = idx[:c], nus[:16 * c]
            pr = edic_prove(prm, hashes, I, V)
            assert edic_verify(prm, honest, I, V, pr)
            res[f"{tag}_prove"][c] = med(lambda: edic_prove(prm, hashes, I, V), 5)
            res[f"{tag}_verify"][c] = med(lambda: edic_verify(prm, honest, I, V, pr), 5)
            print(tag, c, round(res[f"{tag}_prove"][c] * 1e3, 2), round(res[f"{tag}_verify"][c] * 1e3, 2), flush=True)

# ---- Merkle
leaves = [sha(honest.raw_block(j)) for j in range(n)]
mt = MerkleTree(leaves)
for c in CS:
    I = [int(j) for j in idx[:c]]
    blocks = [honest.raw_block(j) for j in I]
    paths = [mt.path(j) for j in I]
    res["mht_prove"][c] = med(lambda: [(honest.raw_block(j), mt.path(j)) for j in I])
    res["mht_verify"][c] = med(lambda: all(merkle_verify(mt.root, b, j, p) for b, j, p in zip(blocks, I, paths)))
    print("mht", c, round(res["mht_prove"][c] * 1e3, 3), round(res["mht_verify"][c] * 1e3, 3), flush=True)

out["per_c"] = res
json.dump(out, open("results/bench.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_c"}, indent=1))
