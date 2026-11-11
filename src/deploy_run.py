"""Networked deployment: one verifier process and N edge-node processes over TCP.

  python3 deploy_run.py detect <race|uniform> <seed> [N] [T] [B/N]   -> results/deploy_detect_<strategy>_<seed>.json
  python3 deploy_run.py cost <mac|bls|edic> [N] [epochs]              -> results/deploy_cost_<family>.json

The verifier keeps only its keys.  Every epoch it (1) ticks the fleet clock (nodes inject their own
faults), (2) allocates sample sizes, (3) sends challenges and collects proofs concurrently,
(4) verifies, (5) for every failing node localises ALL corrupted blocks by level-synchronous
differential group testing over the whole replica, fetches verified blocks from a passing node,
writes them to the failing node and confirms with a full-replica proof, (6) updates its trust state.
Ground truth (which nodes really hold corrupted blocks) is read through a separate status call
that is used for evaluation only.
"""
import hmac, json, os, socket, struct, subprocess, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from race_lib import *
from sim import make_population, Race, Uniform

D = "deploy_data"
NB = 16384
PROOF = 16 * (S_SECT + 1)


def recv_exact(s, k):
    buf = bytearray()
    while len(buf) < k:
        chunk = s.recv(k - len(buf))
        if not chunk:
            raise ConnectionError
        buf += chunk
    return bytes(buf)


class Fleet:
    def __init__(self, N, seed, schemes="mac", faults=True):
        self.N = N
        self.pop = make_population(N, np.random.default_rng(seed))
        base = 12000 + (os.getpid() % 250) * 80                      # below the ephemeral port range
        self.procs = []
        for i in range(N):
            cfg = dict(id=i, port=base + i, data_dir=D, n_blocks=NB, schemes=schemes, seed=seed, rtt=0.0,
                       p_fault=float(self.pop["p_fault"][i]) if faults else 0.0,
                       rho_lo=float(self.pop["rho_lo"][i]), rho_hi=float(self.pop["rho_hi"][i]))
            self.procs.append(subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy_node.py"), json.dumps(cfg)]))
        self.socks = []
        for i in range(N):
            s = None
            for attempt in range(1200):
                if self.procs[i].poll() is not None:
                    break
                try:
                    s = socket.create_connection(("127.0.0.1", base + i)); break
                except OSError:
                    time.sleep(0.1)
            if s is None:
                for p in self.procs: p.kill()
                raise RuntimeError(f"node {i} did not start")
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socks.append(s)
        self.pool = ThreadPoolExecutor(N)
        self.up = self.down = 0

    def rpc(self, i, payload, count=True):
        s = self.socks[i]
        s.sendall(struct.pack(">I", len(payload)) + payload)
        out = recv_exact(s, struct.unpack(">I", recv_exact(s, 4))[0])
        if count:
            self.up += len(payload) + 4; self.down += len(out) + 4
        return out

    def fan(self, items, count=True):
        return list(self.pool.map(lambda it: self.rpc(it[0], it[1], count), items))

    def tick(self, t):
        self.fan([(i, b"T" + struct.pack(">I", t)) for i in range(self.N)], count=False)

    def status(self):
        r = [struct.unpack(">Ii", x) for x in self.fan([(i, b"Q") for i in range(self.N)], count=False)]
        return np.array([x[0] for x in r]), np.array([x[1] for x in r])

    def set_rtt(self, rtts):
        for i, s in enumerate(self.socks):
            s.sendall(struct.pack(">I", 9) + b"R" + struct.pack(">d", float(rtts[i])))
            recv_exact(s, struct.unpack(">I", recv_exact(s, 4))[0])

    def close(self):
        for s in self.socks:
            try:
                s.sendall(struct.pack(">I", 1) + b"X"); s.close()
            except OSError:
                pass
        for p in self.procs:
            p.wait(timeout=30)
        self.pool.shutdown()


class Verifier:
    def __init__(self, fleet):
        k = json.load(open(f"{D}/keys.json"))
        key = MacKey.__new__(MacKey)
        key.k, key.alpha, key.s = bytes.fromhex(k["k"]), bytes.fromhex(k["alpha"]), S_SECT
        key.alpha_int = [fe_int(key.alpha[16 * t:16 * t + 16]) for t in range(S_SECT)]
        self.key, self.ek, self.fleet, self.keys = key, bytes.fromhex(k["epoch_key"]), fleet, k
        self.ALL = np.arange(NB, dtype=np.uint32)

    def seed(self, *parts):
        return hmac.digest(self.ek, b"|".join(str(p).encode() for p in parts), "sha256")

    # ---- RACE-Audit: one nested challenge, consensus-grouped verification
    def audit_race(self, t, c):
        f, seed = self.fleet, self.seed("epoch", t)
        nodes = [i for i in range(f.N) if c[i] > 0]
        u0, d0, w0 = f.up, f.down, time.perf_counter()
        resp = f.fan([(i, b"P" + seed + struct.pack(">I", int(c[i]))) for i in nodes])
        v0 = time.perf_counter()
        idx, nus = expand_challenge(seed, NB, int(max(c)))
        ok, groups, _ = race_verify_all(self.key, idx, nus, [int(c[i]) for i in nodes],
                                        [(r[:16 * S_SECT], r[16 * S_SECT:PROOF]) for r in resp])
        v1 = time.perf_counter()
        return self._result(nodes, ok, resp, v1 - v0, v1 - w0, f.up - u0, f.down - d0, groups)

    # ---- U-HMAC: same tags, independent challenge and verification per node
    def audit_uniform(self, t, c):
        f = self.fleet
        nodes = [i for i in range(f.N) if c[i] > 0]
        seeds = {i: self.seed("epoch", t, i) for i in nodes}
        u0, d0, w0 = f.up, f.down, time.perf_counter()
        resp = f.fan([(i, b"P" + seeds[i] + struct.pack(">I", int(c[i]))) for i in nodes])
        v0 = time.perf_counter()
        ok = []
        for i, r in zip(nodes, resp):
            idx, nus = expand_challenge(seeds[i], NB, int(c[i]))
            ok.append(mac_verify(self.key, r[:16 * S_SECT], r[16 * S_SECT:PROOF], mac_tau(self.key, idx, nus)))
        v1 = time.perf_counter()
        return self._result(nodes, ok, resp, v1 - v0, v1 - w0, f.up - u0, f.down - d0, len(nodes))

    def _result(self, nodes, ok, resp, tv, wall, up, down, groups):
        flags = np.ones(self.fleet.N, bool)
        flags[nodes] = ok
        cpu = np.array([struct.unpack(">d", r[-8:])[0] for r in resp])
        return flags, dict(verifier_s=tv, wall_s=wall, up=up, down=down - 8 * len(resp), node_cpu_total=float(cpu.sum()),
                           node_cpu_max=float(cpu.max()), groups=groups, audited=len(nodes))

    # ---- baselines (cost runs)
    def setup_baseline(self, scheme):
        self.hon = Replica(NB)
        if scheme == "bls":
            self.bk = BlsKey(); self.bk.x = int(self.keys["bls_x"], 16); self.bk.v = bls.G2Point() * scal(self.bk.x)
            self.H = [bls_H(j) for j in range(NB)]                    # cached hash-to-curve points
        if scheme == "mht":
            self.root = MerkleTree([sha(self.hon.raw_block(j)) for j in range(NB)]).root
        if scheme == "edic":
            self.eprm = EdicParams(2048, 256)

    def audit_baseline(self, scheme, t, c):
        f = self.fleet
        seeds = [self.seed(scheme, t, i) for i in range(f.N)]
        op = {"bls": b"B", "mht": b"M", "edic": b"E"}[scheme]
        u0, d0, w0 = f.up, f.down, time.perf_counter()
        resp = f.fan([(i, op + seeds[i] + struct.pack(">I", c)) for i in range(f.N)])
        v0 = time.perf_counter()
        ok = []
        for i, r in enumerate(resp):
            if scheme == "bls":
                idx, nus = expand_challenge(seeds[i], NB, c)
                mu = [int.from_bytes(r[32 * t_:32 * t_ + 32], "big") for t_ in range(S_BLS)]
                sig = bls.G1Point.from_compressed_bytes(r[32 * S_BLS:32 * S_BLS + 48])
                agg = bls.G1Point.multiexp_unchecked([self.H[int(j)] for j in idx] + self.bk.u,
                                                     [scal(x) for x in nu_ints(nus)] + [scal(m) for m in mu])
                ok.append(bls.GT.pairing(sig, self.bk.g2) == bls.GT.pairing(agg, self.bk.v))
            elif scheme == "mht":
                idx = prp_prefix(seeds[i], NB, c)
                rec = BLOCK_BYTES + 14 * 32
                good = True
                for q, j in enumerate(idx):
                    blk = r[q * rec:q * rec + BLOCK_BYTES]
                    path = [r[q * rec + BLOCK_BYTES + 32 * z:q * rec + BLOCK_BYTES + 32 * z + 32] for z in range(14)]
                    good &= merkle_verify(self.root, blk, int(j), path)
                ok.append(good)
            else:
                idx, nus = expand_challenge(seeds[i], NB, c)
                ok.append(edic_verify(self.eprm, self.hon, idx, nus, mpz(int.from_bytes(r[:256], "big"))))
        v1 = time.perf_counter()
        return self._result(list(range(f.N)), ok, resp, v1 - v0, v1 - w0, f.up - u0, f.down - d0, f.N)

    # ---- localisation of all corrupted blocks of a failing node + verified repair
    def repair(self, i, sources, t):
        f = self.fleet
        u0, d0, w0 = f.up, f.down, time.perf_counter()
        stats = dict(blocks=0, queries=0, rounds=0, clean=False)
        for attempt in range(3):
            seed = self.seed("scan", t, i, attempt)
            nus = coeffs(seed, NB)
            def prove_ranges(reqs):
                r = f.rpc(i, b"S" + seed + struct.pack(">I", len(reqs)) + struct.pack(f">{2 * len(reqs)}I", *[x for q in reqs for x in q]))
                return [(r[PROOF * z:PROOF * z + 16 * S_SECT], r[PROOF * z + 16 * S_SECT:PROOF * (z + 1)]) for z in range(len(reqs))]
            parent = prove_ranges([(0, NB)])[0]
            found, q, rounds = localise(self.key, self.ALL, nus, parent, prove_ranges)
            stats["queries"] += q + 1; stats["rounds"] += rounds + 1
            if not found:
                stats["clean"] = True
                break
            stats["blocks"] += len(found)
            good = {}
            for src in sources:                                       # fetch from a passing node, verify every block
                need = [j for j in found if j not in good]
                if not need:
                    break
                r = f.rpc(src, b"G" + struct.pack(">I", len(need)) + struct.pack(f">{len(need)}I", *need))
                rec = BLOCK_BYTES + 16
                for z, j in enumerate(need):
                    blk, tg = r[rec * z:rec * z + BLOCK_BYTES], r[rec * z + BLOCK_BYTES:rec * (z + 1)]
                    if verify_block(self.key, j, blk, tg):
                        good[j] = blk
            f.rpc(i, b"W" + struct.pack(">I", len(good)) + b"".join(struct.pack(">I", j) + good[j] for j in good))
        stats.update(wall_s=time.perf_counter() - w0, up=f.up - u0, down=f.down - d0)
        return stats


def run_detect(strategy, seed, N=64, T=200, Bn=256):
    fleet = Fleet(N, seed)
    V = Verifier(fleet)
    value = fleet.pop["value"]
    strat = Race(N=N, B=N * Bn) if strategy == "race" else Uniform(N=N, B=N * Bn)
    exposure = exposure_unw = 0.0
    ttd, episodes, false_pos, epochs, repairs = [], 0, 0, [], []
    prev_onset = np.full(N, -1)
    for t in range(T):
        fleet.tick(t)
        dirty0, onset0 = fleet.status()
        episodes += int(((onset0 == t)).sum())
        c = strat.alloc(value).astype(int)
        ok, m = (V.audit_race if strategy == "race" else V.audit_uniform)(t, c)
        det = ~ok
        false_pos += int((det & (dirty0 == 0)).sum())
        passing = [int(i) for i in np.argsort(-c) if ok[i]]
        for i in np.where(det)[0]:
            ttd.append(t - int(onset0[i]))
            repairs.append(V.repair(int(i), passing[:3], t))
        dirty1, onset1 = fleet.status()
        exposure += float((value * (dirty1 > 0)).sum()); exposure_unw += float((dirty1 > 0).sum())
        strat.update(c > 0, det)
        m["samples"] = int(c.sum()); m["failures"] = int(det.sum())
        epochs.append(m)
    dirty1, onset1 = fleet.status()
    for i in np.where(dirty1 > 0)[0]:
        ttd.append(T - int(onset1[i]))
    fleet.close()
    out = dict(strategy=strategy, seed=seed, N=N, T=T, Bn=Bn, exposure=exposure / T, exposure_unw=exposure_unw / T,
               mttd=float(np.mean(ttd)) if ttd else 0.0, episodes=episodes, detections=len(repairs), false_positives=false_pos,
               repairs_clean=int(sum(r["clean"] for r in repairs)),
               verifier_ms=1e3 * float(np.mean([e["verifier_s"] for e in epochs])),
               node_cpu_total_ms=1e3 * float(np.mean([e["node_cpu_total"] for e in epochs])),
               proof_kb=float(np.mean([e["down"] for e in epochs])) / 1e3, challenge_b=float(np.mean([e["up"] for e in epochs])),
               samples=float(np.mean([e["samples"] for e in epochs])),
               repair_blocks=float(np.mean([r["blocks"] for r in repairs])) if repairs else 0.0,
               repair_queries=float(np.mean([r["queries"] for r in repairs])) if repairs else 0.0,
               repair_rounds=float(np.mean([r["rounds"] for r in repairs])) if repairs else 0.0,
               repair_kb=float(np.mean([r["up"] + r["down"] for r in repairs])) / 1e3 if repairs else 0.0)
    json.dump(out, open(f"results/deploy_detect_{strategy}_{seed}.json", "w"), indent=1)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)


def summarise(ms):
    keys = ["verifier_s", "wall_s", "up", "down", "node_cpu_total", "node_cpu_max"]
    return {k: [float(np.median([m[k] for m in ms])), float(np.percentile([m[k] for m in ms], 25)),
                float(np.percentile([m[k] for m in ms], 75))] for k in keys} | {"epochs": len(ms)}


def run_cost(family, N=64, epochs=30, seed=7, Bn=256, warm=150, rtt=(0.010, 0.050)):
    schemes = {"mac": "mac,mht", "bls": "mac,bls", "edic": "mac,edic"}[family]
    fleet = Fleet(N, seed, schemes=schemes)
    V = Verifier(fleet)
    value = fleet.pop["value"]
    rtts = np.random.default_rng(seed + 1).uniform(*rtt, N)
    res, sane = {}, True
    strat = Race(N=N, B=N * Bn)
    def housekeeping(t, det):
        ok_nodes = [int(i) for i in range(N) if not det[i]]
        for i in np.where(det)[0]:
            V.repair(int(i), ok_nodes[:3], t)
    if family == "mac":
        V.setup_baseline("mht")
        for t in range(warm):                                         # reach the steady-state allocation (no delay)
            fleet.tick(t)
            c = strat.alloc(value).astype(int)
            ok, _ = V.audit_race(t, c)
            housekeeping(t, ~ok); strat.update(c > 0, ~ok)
        fleet.set_rtt(rtts)
        acc = {"RACE-Audit": [], "U-HMAC": [], "U-MHT": []}
        rep = []
        for t in range(warm, warm + epochs):
            fleet.tick(t)
            dirty0, _ = fleet.status()
            cu = np.full(N, Bn)
            ok_u, m = V.audit_uniform(t, cu); acc["U-HMAC"].append(m); sane &= not (~ok_u & (dirty0 == 0)).any()
            ok_m, m = V.audit_baseline("mht", t, Bn); acc["U-MHT"].append(m); sane &= not (~ok_m & (dirty0 == 0)).any()
            c = strat.alloc(value).astype(int)
            ok, m = V.audit_race(t, c); m["samples"] = int(c.sum()); acc["RACE-Audit"].append(m)
            sane &= not (~ok & (dirty0 == 0)).any()
            det = ~ok | ~ok_u | ~ok_m
            for i in np.where(det)[0]:
                rep.append(V.repair(int(i), [int(j) for j in range(N) if not det[j]][:3], t))
            strat.update(c > 0, ~ok)
        res = {k: summarise(v) for k, v in acc.items()}
        res["RACE-Audit"]["samples"] = float(np.mean([m["samples"] for m in acc["RACE-Audit"]]))
        res["RACE-Audit"]["tiers"] = sorted(set(int(x) for x in c))
        res["RACE-Audit"]["groups"] = float(np.mean([m["groups"] for m in acc["RACE-Audit"]]))
        if rep:
            res["repair"] = {k: float(np.mean([r[k] for r in rep])) for k in ["blocks", "queries", "rounds", "wall_s", "up", "down"]} | {"n": len(rep)}
    else:
        V.setup_baseline(family)
        fleet.set_rtt(rtts)
        acc = []
        for t in range(epochs):
            fleet.tick(t)
            dirty0, _ = fleet.status()
            ok, m = V.audit_baseline(family, t, Bn); acc.append(m)
            sane &= not (~ok & (dirty0 == 0)).any()
            fleet.set_rtt(np.zeros(N)); housekeeping(t, dirty0 > 0); fleet.set_rtt(rtts)
        res = {{"bls": "U-BLS", "edic": "EDI-C"}[family]: summarise(acc)}
    fleet.close()
    res["meta"] = dict(N=N, Bn=Bn, epochs=epochs, rtt_ms=[1e3 * rtt[0], 1e3 * rtt[1]], rtt_mean_ms=1e3 * float(rtts.mean()),
                       rtt_max_ms=1e3 * float(rtts.max()), no_false_alarms=bool(sane))
    json.dump(res, open(f"results/deploy_cost_{family}.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "detect":
        run_detect(a[1], int(a[2]), *(int(x) for x in a[3:]))
    else:
        run_cost(a[1], *(int(x) for x in a[2:]))
