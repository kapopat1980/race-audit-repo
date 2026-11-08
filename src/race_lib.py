"""
RACE-Audit: Risk-Adaptive, Consensus-Enabled probabilistic integrity auditing
for replicated data on untrusted / semi-trusted edge nodes.

This module contains functional (really executed) implementations of:
  * RACE-Audit (proposed): pairing-free homomorphic MAC tags, Bayesian-UCB risk
    estimation, convex (water-filling) sample allocation with a randomness floor,
    nested common-prefix challenges quantised to tiers, consensus-grouped
    verification and adaptive group-testing corruption localisation.
  * U-HMAC  : uniform-sampling, per-node independent challenges, same MAC
              (represents lightweight algebraic/MAC EDI schemes, e.g. EDI-DA/EDI-SA family).
  * U-BLS   : uniform sampling, Shacham-Waters style BLS homomorphic
              signatures, per-node pairing verification (ICE/EDI-S/MEC-P family).
  * B-BLS   : same tags, aggregate (batch) verification across all nodes with a
              single pairing check (EDI-S style aggregation).
  * U-MHT   : uniform sampling with Merkle-hash-tree proofs that return the
              sampled blocks (EDI-V / sparse-Merkle family).
All cryptographic operations are real (HMAC-SHA256, SHA-256, BLS12-381 via
arkworks).  Field arithmetic is in GF(p), p = 2^127 - 1.
"""
import hashlib, hmac, math, os, struct
import numpy as np

P = (1 << 127) - 1          # Mersenne prime field for MAC scheme
SECTOR_BYTES = 15           # < 127 bits
BLOCK_BYTES = 4096
S_SECT = math.ceil(BLOCK_BYTES / SECTOR_BYTES)   # 274 sectors per block
FE_BYTES = 16

# ----------------------------------------------------------------- data model
class Replica:
    """A replica of an n-block file.  Block contents are generated
    deterministically from a file seed (equivalent to reading from storage);
    corruption is modelled by overriding specific blocks."""
    def __init__(self, n, file_seed=b"file", corrupted=None):
        self.n = n
        self.file_seed = file_seed
        self.corrupted = dict(corrupted or {})
        self._cache = {}

    def raw_block(self, j):
        if j in self.corrupted:
            return self.corrupted[j]
        if not hasattr(self, "_raw"):
            self._raw = {}
        b = self._raw.get(j)
        if b is None:
            b = hashlib.shake_256(self.file_seed + struct.pack(">Q", j)).digest(BLOCK_BYTES)
            if len(self._raw) < 20000:
                self._raw[j] = b
        return b

    def sectors(self, j):
        if j in self.corrupted:
            return bytes_to_sectors(self.corrupted[j])
        c = self._cache.get(j)
        if c is None:
            c = bytes_to_sectors(self.raw_block(j))
            if len(self._cache) < 50000:
                self._cache[j] = c
        return c

    def corrupt(self, idx, rng):
        for j in idx:
            self.corrupted[int(j)] = bytes(rng.integers(0, 256, BLOCK_BYTES, dtype=np.uint8))


def bytes_to_sectors(b):
    return [int.from_bytes(b[t:t + SECTOR_BYTES], "big") for t in range(0, len(b), SECTOR_BYTES)]


# ----------------------------------------------------------------- PRF / PRP
def prf(key, msg):
    return int.from_bytes(hmac.new(key, msg, hashlib.sha256).digest()[:16], "big") % P


def prp_prefix(seed, n, length):
    """First `length` elements of a keyed pseudo-random permutation of [0,n)
    (Fisher-Yates driven by an HMAC-DRBG; only `length` swaps are performed)."""
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(seed).digest()[:16], "big")))
    perm = {}
    out = []
    for i in range(length):
        r = int(rng.integers(i, n))
        vi, vr = perm.get(i, i), perm.get(r, r)
        perm[i], perm[r] = vr, vi
        out.append(vr)
    return out


def coeffs(seed, idx):
    return [prf(seed, b"nu" + struct.pack(">Q", j)) or 1 for j in idx]


# ----------------------------------------------------------------- MAC scheme
class MacKey:
    def __init__(self, s=S_SECT, rng=None):
        self.k = os.urandom(32)
        self.alpha = [int.from_bytes(os.urandom(16), "big") % P for _ in range(s)]
        self.s = s

    def f(self, j):
        return prf(self.k, b"tag" + struct.pack(">Q", j))


def mac_tag(key, sectors, j):
    acc = key.f(j)
    for a, m in zip(key.alpha, sectors):
        acc += a * m
    return acc % P


class MacTagStore:
    """Tags are produced once by the data owner and cached at the nodes.
    Here they are computed lazily and memoised (TagGen is offline)."""
    def __init__(self, key, replica_honest):
        self.key, self.rep = key, replica_honest
        self.t = {}

    def __getitem__(self, j):
        v = self.t.get(j)
        if v is None:
            v = mac_tag(self.key, self.rep.sectors(j), j)
            self.t[j] = v
        return v


def mac_prove(replica, tags, idx, nus):
    s = len(replica.sectors(idx[0]))
    mu = [0] * s
    sig = 0
    for j, nu in zip(idx, nus):
        sec = replica.sectors(j)
        for t in range(s):
            mu[t] += nu * sec[t]
        sig += nu * tags[j]
    return [x % P for x in mu], sig % P


def mac_tau(key, idx, nus):
    return sum(nu * key.f(j) for j, nu in zip(idx, nus)) % P


def mac_verify(key, mu, sig, tau):
    acc = tau
    for a, m in zip(key.alpha, mu):
        acc += a * m
    return acc % P == sig


# ----------------------------------------------------------------- RACE allocation
def beta_ucb(f, h, a0=1.0, b0=9.0, kappa=1.0):
    a, b = f + a0, h + b0
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return np.minimum(mean + kappa * np.sqrt(var), 1.0)


def waterfill(r, B, L, cmin, cmax):
    """argmin_c sum r_i exp(-L c_i)  s.t. sum c_i = B, cmin<=c_i<=cmax.
    KKT: c_i = clip((ln r_i - ln eta)/L).  eta found by bisection."""
    r = np.maximum(np.asarray(r, float), 1e-300)
    lr = np.log(r)
    N = len(r)
    if B <= N * cmin:
        return np.full(N, float(cmin))
    if B >= N * cmax:
        return np.full(N, float(cmax))
    lo, hi = lr.min() - L * cmax - 1, lr.max() + 1
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        c = np.clip((lr - mid) / L, cmin, cmax)
        if c.sum() > B:
            lo = mid
        else:
            hi = mid
    return np.clip((lr - hi) / L, cmin, cmax)


def quantise_to_tiers(c, tiers, B):
    """Round each c_i to a tier; greedy repair so that sum <= B (+ one tier step)."""
    tiers = np.asarray(sorted(tiers))
    k = np.searchsorted(tiers, c, side="left")
    k = np.clip(k, 0, len(tiers) - 1)
    # choose nearer of tiers[k-1], tiers[k]
    lower = np.clip(k - 1, 0, len(tiers) - 1)
    pick = np.where(np.abs(tiers[lower] - c) < np.abs(tiers[k] - c), lower, k)
    q = tiers[pick].astype(float)
    # budget repair: demote nodes with smallest marginal loss while over budget
    order = np.argsort(c - q)  # most over-allocated first
    i = 0
    while q.sum() > B and i < 10 * len(q):
        n_ = order[i % len(q)]
        pk = np.searchsorted(tiers, q[n_])
        if pk > 0:
            q[n_] = tiers[pk - 1]
        i += 1
    return q


class RaceAllocator:
    def __init__(self, N, B, rho_star=0.01, cmin=64, cmax=1024, tiers=(64, 128, 256, 512, 1024),
                 kappa=1.0, lam=0.97, w_fail=5.0, a0=1.0, b0=9.0, use_tiers=True):
        self.N, self.B = N, B
        self.L = -math.log(1 - rho_star)
        self.cmin, self.cmax = cmin, cmax
        self.tiers = tiers
        self.kappa, self.lam, self.w = kappa, lam, w_fail
        self.a0, self.b0 = a0, b0
        self.f = np.zeros(N)
        self.h = np.zeros(N)
        self.use_tiers = use_tiers

    def risk(self, value):
        return value * beta_ucb(self.f, self.h, self.a0, self.b0, self.kappa)

    def allocate(self, value):
        c = waterfill(self.risk(value), self.B, self.L, self.cmin, self.cmax)
        if self.use_tiers:
            return quantise_to_tiers(c, self.tiers, self.B)
        return np.floor(c)

    def update(self, audited, failed):
        self.f *= self.lam
        self.h *= self.lam
        self.h += audited & ~failed
        self.f += self.w * failed


# ----------------------------------------------------------------- RACE protocol (functional)
def race_challenge(epoch_key, epoch, n, cmax):
    seed = hmac.new(epoch_key, b"epoch" + struct.pack(">Q", epoch), hashlib.sha256).digest()
    idx = prp_prefix(seed, n, cmax)
    nus = coeffs(seed, idx)
    return seed, idx, nus


def race_prove(replica, tags, idx, nus, c):
    return mac_prove(replica, tags, idx[:c], nus[:c])


def race_verify_all(key, idx, nus, cs, responses):
    """Consensus-grouped verification.
    responses: list of (mu, sigma).  Returns (pass_flags, n_tag_verifications, n_prf)."""
    cmax = int(max(cs))
    # tau prefix sums: one PRF per challenged index, shared by all nodes
    tau_pref = [0] * (cmax + 1)
    acc = 0
    for i in range(cmax):
        acc = (acc + nus[i] * key.f(idx[i])) % P
        tau_pref[i + 1] = acc
    groups = {}
    for i, (mu, sig) in enumerate(responses):
        groups.setdefault((int(cs[i]), tuple(mu), sig), []).append(i)
    ok = [False] * len(responses)
    nver = 0
    for (c, mu, sig), members in groups.items():
        nver += 1
        res = mac_verify(key, mu, sig, tau_pref[c])
        for m in members:
            ok[m] = res
    return ok, nver, cmax


def group_test_localise(key, replica, tags, idx, nus, parent=None, differential=True):
    """Adaptive binary splitting over a failing sampled set.
    With differential=True the verifier requests only the LEFT half of a failing
    set and derives the right half by linearity (mu_R = mu_I - mu_L,
    sigma_R = sigma_I - sigma_L), halving the number of round-trips.
    Returns (corrupted_indices, n_queries, bytes_down)."""
    found, queries = [], 0
    if parent is None:
        parent = mac_prove(replica, tags, idx, nus)
    stack = [(list(idx), list(nus), parent)]

    def ok(I, V, resp):
        return mac_verify(key, resp[0], resp[1], mac_tau(key, I, V))

    while stack:
        I, V, resp = stack.pop()
        if ok(I, V, resp):
            continue
        if len(I) == 1:
            found.append(I[0])
            continue
        h = len(I) // 2
        left = mac_prove(replica, tags, I[:h], V[:h])
        queries += 1
        if differential:
            right = ([(a - b) % P for a, b in zip(resp[0], left[0])], (resp[1] - left[1]) % P)
        else:
            right = mac_prove(replica, tags, I[h:], V[h:])
            queries += 1
        stack.append((I[:h], V[:h], left))
        stack.append((I[h:], V[h:], right))
    return sorted(found), queries, queries * (S_SECT + 1) * FE_BYTES


# ----------------------------------------------------------------- BLS (Shacham-Waters style) baseline
try:
    import py_arkworks_bls12381 as bls
    R_ORDER = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
    BLS_SECTOR = 31
    S_BLS = math.ceil(BLOCK_BYTES / BLS_SECTOR)   # 133 sectors
    DST = b"RACE-AUDIT-H2C"

    def scal(x):
        return bls.Scalar.from_be_bytes_mod_order(int(x % R_ORDER).to_bytes(32, "big"))

    class BlsKey:
        def __init__(self, s=S_BLS):
            self.x = int.from_bytes(os.urandom(32), "big") % R_ORDER
            self.u = [bls.G1Point.hash_to_curve(DST, b"u" + struct.pack(">I", t)) for t in range(s)]
            self.v = bls.G2Point() * scal(self.x)
            self.g2 = bls.G2Point()
            self.s = s

    def bls_sectors(raw):
        return [int.from_bytes(raw[t:t + BLS_SECTOR], "big") for t in range(0, len(raw), BLS_SECTOR)]

    def bls_H(j):
        return bls.G1Point.hash_to_curve(DST, b"H" + struct.pack(">Q", j))

    def bls_tag(key, raw, j):
        sec = bls_sectors(raw)
        base = bls_H(j) + bls.G1Point.multiexp_unchecked(key.u, [scal(m) for m in sec])
        return base * scal(key.x)

    def bls_prove(replica, tags, idx, nus):
        if not hasattr(replica, "_bsec"):
            replica._bsec = {}
        secs = []
        for j in idx:
            if j in replica.corrupted:
                secs.append(bls_sectors(replica.corrupted[j]))
            else:
                v = replica._bsec.get(j)
                if v is None:
                    v = replica._bsec[j] = bls_sectors(replica.raw_block(j))
                secs.append(v)
        s = len(secs[0])
        mu = [sum(nu * sc[t] for nu, sc in zip(nus, secs)) % R_ORDER for t in range(s)]
        sig = bls.G1Point.multiexp_unchecked([tags[j] for j in idx], [scal(nu) for nu in nus])
        return mu, sig

    def bls_lhs_rhs(key, idx, nus, mu):
        pts = [bls_H(j) for j in idx] + key.u
        sc = [scal(nu) for nu in nus] + [scal(m) for m in mu]
        return bls.G1Point.multiexp_unchecked(pts, sc)

    def bls_verify(key, idx, nus, mu, sig):
        agg = bls_lhs_rhs(key, idx, nus, mu)
        return bls.GT.pairing(sig, key.g2) == bls.GT.pairing(agg, key.v)
    HAVE_BLS = True
except Exception:  # pragma: no cover
    HAVE_BLS = False


# ----------------------------------------------------------------- Merkle baseline
def sha(b):
    return hashlib.sha256(b).digest()


class MerkleTree:
    def __init__(self, leaves):
        self.levels = [leaves]
        while len(self.levels[-1]) > 1:
            L = self.levels[-1]
            if len(L) % 2:
                L = L + [L[-1]]
            self.levels.append([sha(L[i] + L[i + 1]) for i in range(0, len(L), 2)])
        self.root = self.levels[-1][0]

    def path(self, j):
        out = []
        for L in self.levels[:-1]:
            sib = j ^ 1
            out.append(L[sib] if sib < len(L) else L[j])
            j >>= 1
        return out


def merkle_verify(root, block, j, path):
    h = sha(block)
    for sib in path:
        h = sha(h + sib) if j % 2 == 0 else sha(sib + h)
        j >>= 1
    return h == root
