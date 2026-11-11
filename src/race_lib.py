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
  * EDI-C   : homomorphic-hash collaborative audit of Yang et al. (2024), re-implemented
              from the published description (GMP modular exponentiation).
All cryptographic operations are real (HMAC-SHA256, SHA-256, BLS12-381 via
arkworks, GMP).  Field arithmetic is in GF(p), p = 2^127 - 1.  The inner loops of
every scheme run in native code (race_native.c, arkworks, GMP, OpenSSL).
"""
import ctypes, hashlib, hmac, math, os, struct, subprocess
import numpy as np

P = (1 << 127) - 1          # Mersenne prime field for MAC scheme
SECTOR_BYTES = 15           # < 127 bits
BLOCK_BYTES = 4096
S_SECT = math.ceil(BLOCK_BYTES / SECTOR_BYTES)   # 274 sectors per block
FE_BYTES = 16

# ----------------------------------------------------------------- native kernels
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC, _SO = os.path.join(_HERE, "race_native.c"), os.path.join(_HERE, "libracenative.so")
if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(_SRC):
    subprocess.check_call(["gcc", "-O3", "-fPIC", "-shared", "-Wno-deprecated-declarations",
                           _SRC, "-o", _SO, "-lcrypto"])
_lib = ctypes.CDLL(_SO)
_vp, _u32, _u64, _int = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int
_lib.race_prf_batch.argtypes = [_vp, _vp, _u32, _vp]
_lib.race_taggen.argtypes = [_vp, _u64, _vp, _vp, _vp]
_lib.race_prove.argtypes = [_vp, _vp, _vp, _vp, _u32, _vp, _vp]
_lib.race_verify.argtypes = [_vp, _vp, _vp, _vp]; _lib.race_verify.restype = _int
_lib.race_tau_prefix.argtypes = [_vp, _vp, _vp, _u32, _vp]
_lib.race_tau.argtypes = [_vp, _vp, _vp, _u32, _vp]
_lib.race_tau_cached.argtypes = [_vp, _vp, _vp, _u32, _vp]
_lib.race_vec_sub.argtypes = [_vp, _vp, _u32, _vp]
_lib.acc_unreduced.argtypes = [_vp, _vp, _vp, _u32, _int, _int, _vp]
_lib.race_prp_prefix.argtypes = [_vp, _u32, _u32, _vp]


def _a(x):
    """address of a bytes / numpy buffer"""
    if isinstance(x, np.ndarray):
        return x.ctypes.data
    return np.frombuffer(x, np.uint8).ctypes.data


def fe_bytes(x):
    return int(x).to_bytes(16, "little")


def fe_int(b):
    return int.from_bytes(bytes(b), "little")


# ----------------------------------------------------------------- data model
_HONEST = {}


def honest_data(n, file_seed=b"file"):
    """Deterministic file contents (n blocks of 4 KiB), shared read-only by all intact replicas."""
    k = (n, file_seed)
    if k not in _HONEST:
        d = np.frombuffer(hashlib.shake_256(file_seed).digest(n * BLOCK_BYTES), np.uint8)
        _HONEST[k] = d                      # read-only view of immutable bytes
    return _HONEST[k]


class Replica:
    """A replica of an n-block file held in memory.  Intact replicas share one
    buffer; corrupting a replica gives it a private copy (copy-on-write)."""
    def __init__(self, n, file_seed=b"file", data=None):
        self.n = n
        self.data = honest_data(n, file_seed) if data is None else data
        self.corrupted = set()

    def raw_block(self, j):
        return self.data[j * BLOCK_BYTES:(j + 1) * BLOCK_BYTES].tobytes()

    def sectors(self, j):                   # pure-Python view, used only by the reference implementation
        return bytes_to_sectors(self.raw_block(j))

    def _own(self):
        if not self.data.flags.writeable:
            self.data = self.data.copy()

    def corrupt(self, idx, rng):
        self._own()
        for j in idx:
            j = int(j)
            self.data[j * BLOCK_BYTES:(j + 1) * BLOCK_BYTES] = rng.integers(0, 256, BLOCK_BYTES, dtype=np.uint8)
            self.corrupted.add(j)

    def repair(self, j, block):
        self._own()
        self.data[j * BLOCK_BYTES:(j + 1) * BLOCK_BYTES] = np.frombuffer(block, np.uint8)
        self.corrupted.discard(int(j))


def bytes_to_sectors(b):
    return [int.from_bytes(b[t:t + SECTOR_BYTES], "big") for t in range(0, len(b), SECTOR_BYTES)]


# ----------------------------------------------------------------- PRF / PRP
def prf(key, msg):
    return int.from_bytes(hmac.digest(key, msg, "sha256")[:16], "big") % P


def prp_prefix(seed, n, length):
    """First `length` elements of a seeded pseudo-random permutation of [0,n) (native partial
    Fisher-Yates).  Prefixes are nested: prp_prefix(seed, n, a) == prp_prefix(seed, n, b)[:a]."""
    out = np.empty(length, np.uint32)
    _lib.race_prp_prefix(_a(seed), n, length, _a(out))
    return out


def coeffs(seed, length):
    """`length` challenge coefficients in [0, 2^127), expanded from the seed with SHAKE-256 (nested)."""
    a = np.frombuffer(hashlib.shake_256(seed + b"nu").digest(16 * length), np.uint8).copy()
    a[15::16] &= 0x7F
    return a.tobytes()


def nu_ints(nus):
    return [int.from_bytes(nus[i:i + 16], "little") for i in range(0, len(nus), 16)]


# ----------------------------------------------------------------- MAC scheme
class MacKey:
    def __init__(self, s=S_SECT):
        self.k = os.urandom(32)
        self.alpha_int = [int.from_bytes(os.urandom(16), "big") % P for _ in range(s)]
        self.alpha = b"".join(fe_bytes(a) for a in self.alpha_int)
        self.s = s

    def f(self, j):                          # reference (Python) PRF
        return prf(self.k, b"tag" + struct.pack(">Q", j))

    def f_table(self, n):                    # all masks f_k(0..n-1), for the cached-mask verifier variant
        out = np.empty(16 * n, np.uint8)
        ar = np.arange(n, dtype=np.uint32)   # keep a reference alive during the native call
        _lib.race_prf_batch(_a(self.k), _a(ar), n, _a(out))
        return out


def mac_taggen(key, replica_honest):
    """TagGen (offline, by the data owner): one 16-byte tag per block."""
    tags = np.empty(16 * replica_honest.n, np.uint8)
    _lib.race_taggen(_a(replica_honest.data), replica_honest.n, _a(key.k), _a(key.alpha), _a(tags))
    return tags


class MacTagStore:
    """Tags of all blocks (computed natively once); indexable as integers for the reference code."""
    def __init__(self, key, replica_honest):
        self.arr = mac_taggen(key, replica_honest)

    def __getitem__(self, j):
        return fe_int(self.arr[16 * j:16 * j + 16])


def _tags(tags):
    return tags.arr if isinstance(tags, MacTagStore) else tags


def mac_prove(replica, tags, idx, nus):
    """Proof (mu, sigma) over blocks idx with coefficients nus; both returned as bytes."""
    c = len(idx)
    mu = np.empty(16 * S_SECT, np.uint8); sig = np.empty(16, np.uint8)
    _lib.race_prove(_a(replica.data), _a(_tags(tags)), _a(idx), _a(nus), c, _a(mu), _a(sig))
    return mu.tobytes(), sig.tobytes()


def mac_tau(key, idx, nus, ftab=None):
    out = np.empty(16, np.uint8)
    if ftab is None:
        _lib.race_tau(_a(key.k), _a(idx), _a(nus), len(idx), _a(out))
    else:
        _lib.race_tau_cached(_a(ftab), _a(idx), _a(nus), len(idx), _a(out))
    return out.tobytes()


def mac_verify(key, mu, sig, tau):
    return bool(_lib.race_verify(_a(key.alpha), _a(mu), _a(sig), _a(tau)))


def vec_sub(a, b):
    n = len(a) // 16
    out = np.empty(len(a), np.uint8)
    _lib.race_vec_sub(_a(a), _a(b), n, _a(out))
    return out.tobytes()


# pure-Python reference implementation (used by the tests to cross-check the native kernels)
def ref_tag(key, sectors, j):
    acc = key.f(j)
    for a, m in zip(key.alpha_int, sectors):
        acc += a * m
    return acc % P


def ref_prove(replica, tags, idx, nus):
    mu = [0] * S_SECT
    sig = 0
    for j, nu in zip([int(x) for x in idx], nu_ints(nus)):
        sec = replica.sectors(j)
        for t in range(S_SECT):
            mu[t] += nu * sec[t]
        sig += nu * tags[j]
    return b"".join(fe_bytes(x % P) for x in mu), fe_bytes(sig % P)


def ref_verify(key, mu, sig, idx, nus):
    acc = sum(nu * key.f(int(j)) for j, nu in zip(idx, nu_ints(nus)))
    for t, a in enumerate(key.alpha_int):
        acc += a * fe_int(mu[16 * t:16 * t + 16])
    return acc % P == fe_int(sig)


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


# ----------------------------------------------------------------- RACE protocol
def epoch_seed(epoch_key, epoch):
    return hmac.digest(epoch_key, b"epoch" + struct.pack(">Q", epoch), "sha256")


def expand_challenge(seed, n, c):
    """Run by verifier and by every node: nested index prefix and coefficients from the broadcast seed."""
    return prp_prefix(seed, n, c), coeffs(seed, c)


def race_challenge(epoch_key, epoch, n, cmax):
    seed = epoch_seed(epoch_key, epoch)
    idx, nus = expand_challenge(seed, n, cmax)
    return seed, idx, nus


def race_prove(replica, tags, idx, nus, c):
    return mac_prove(replica, tags, idx[:c], nus[:16 * c])


def race_verify_all(key, idx, nus, cs, responses):
    """Consensus-grouped verification.
    responses: list of (mu, sigma) bytes.  Returns (pass_flags, n_tag_verifications, n_prf)."""
    cmax = int(max(cs))
    tau_pref = np.empty(16 * (cmax + 1), np.uint8)      # one PRF per challenged index, shared by all nodes
    _lib.race_tau_prefix(_a(key.k), _a(idx), _a(nus), cmax, _a(tau_pref))
    groups = {}
    for i, r in enumerate(responses):
        groups.setdefault((int(cs[i]), r[0], r[1]), []).append(i)
    ok = [False] * len(responses)
    for (c, mu, sig), members in groups.items():
        res = mac_verify(key, mu, sig, tau_pref[16 * c:16 * c + 16])
        for m in members:
            ok[m] = res
    return ok, len(groups), cmax


def localise(key, idx, nus, parent, prove_ranges, differential=True):
    """Level-synchronous adaptive binary splitting over a failing sampled set idx (with proof `parent`).
    prove_ranges([(lo, hi), ...]) must return the node's proofs over idx[lo:hi] / nus[lo:hi]; all
    requests of one tree level travel in one round trip.  With differential=True only the LEFT half
    of every failing set is requested; the right half follows by linearity
    (mu_R = mu_I - mu_L, sigma_R = sigma_I - sigma_L).
    Returns (corrupted_indices, n_queries, n_round_trips)."""
    found, queries, rounds = [], 0, 0
    pending = [(0, len(idx), parent)]
    while pending:
        failing = []
        for lo, hi, resp in pending:
            if mac_verify(key, resp[0], resp[1], mac_tau(key, idx[lo:hi], nus[16 * lo:16 * hi])):
                continue
            if hi - lo == 1:
                found.append(int(idx[lo]))
                continue
            failing.append((lo, hi, resp))
        if not failing:
            break
        reqs = []
        for lo, hi, _ in failing:
            mid = lo + (hi - lo) // 2
            reqs.append((lo, mid))
            if not differential:
                reqs.append((mid, hi))
        proofs = prove_ranges(reqs)
        rounds += 1
        queries += len(reqs)
        pending, k = [], 0
        for lo, hi, resp in failing:
            mid = lo + (hi - lo) // 2
            left = proofs[k]; k += 1
            if differential:
                right = (vec_sub(resp[0], left[0]), vec_sub(resp[1], left[1]))
            else:
                right = proofs[k]; k += 1
            pending += [(lo, mid, left), (mid, hi, right)]
    return sorted(found), queries, rounds


def group_test_localise(key, replica, tags, idx, nus, parent=None, differential=True):
    """In-process localisation.  Returns (corrupted_indices, n_queries, bytes_down, n_round_trips)."""
    if parent is None:
        parent = mac_prove(replica, tags, idx, nus)
    found, q, rounds = localise(key, idx, nus, parent,
                                lambda reqs: [mac_prove(replica, tags, idx[lo:hi], nus[16 * lo:16 * hi]) for lo, hi in reqs],
                                differential)
    return found, q, q * (S_SECT + 1) * FE_BYTES, rounds


def verify_block(key, j, block, tag):
    """Check one block against its tag (used when fetching repair blocks from another node)."""
    blk = np.frombuffer(block, np.uint8)
    mu = np.empty(16 * S_SECT, np.uint8); sig = np.empty(16, np.uint8)
    one = fe_bytes(1); z = np.zeros(1, np.uint32)
    _lib.race_prove(_a(blk), _a(tag), _a(z), _a(one), 1, _a(mu), _a(sig))
    return mac_verify(key, mu, sig, mac_tau(key, np.array([j], np.uint32), one))


def acc_sectors(replica, idx, nus, sect_bytes, modulus):
    """sum_i nu_i * sector_t(block idx_i) mod modulus for every sector t (native accumulation)."""
    nsect = math.ceil(BLOCK_BYTES / sect_bytes)
    out = np.empty(56 * nsect, np.uint8)
    _lib.acc_unreduced(_a(replica.data), _a(idx), _a(nus), len(idx), sect_bytes, nsect, _a(out))
    raw = out.tobytes()
    return [int.from_bytes(raw[56 * t:56 * t + 56], "little") % modulus for t in range(nsect)]


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
        mu = acc_sectors(replica, idx, nus, BLS_SECTOR, R_ORDER)
        sig = bls.G1Point.multiexp_unchecked([tags[int(j)] for j in idx], [scal(nu) for nu in nu_ints(nus)])
        return mu, sig

    def bls_lhs_rhs(key, idx, nus, mu):
        pts = [bls_H(int(j)) for j in idx] + key.u
        sc = [scal(nu) for nu in nu_ints(nus)] + [scal(m) for m in mu]
        return bls.G1Point.multiexp_unchecked(pts, sc)

    def bls_verify(key, idx, nus, mu, sig):
        agg = bls_lhs_rhs(key, idx, nus, mu)
        return bls.GT.pairing(sig, key.g2) == bls.GT.pairing(agg, key.v)
    HAVE_BLS = True
except Exception:  # pragma: no cover
    HAVE_BLS = False


# ----------------------------------------------------------------- EDI-C baseline (Yang et al. 2024)
# Homomorphic hash H_G(b) = prod_t g_t^{b_t} mod p over a prime-order-q subgroup; no tags.  A verifier
# that holds its own replica checks  H_G(sum_i s_i b_{c_i} mod q) == prod_i H_G(b_{c_i})^{s_i} mod p.
try:
    import gmpy2
    from gmpy2 import mpz, powmod

    class EdicParams:
        def __init__(self, lp=2048, lq=256, seed=2024):
            rs = gmpy2.random_state(seed)
            q = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, lq) | (mpz(1) << (lq - 1)))
            while True:
                k = gmpy2.mpz_urandomb(rs, lp - lq) | (mpz(1) << (lp - lq - 1))
                p = k * q + 1
                if p.bit_length() == lp and gmpy2.is_prime(p):
                    break
            self.p, self.q, self.lp, self.lq = p, q, lp, lq
            self.sect_bytes = (lq - 1) // 8                      # every block element < q
            self.m = math.ceil(BLOCK_BYTES / self.sect_bytes)
            self.g = []
            while len(self.g) < self.m:
                h = powmod(gmpy2.mpz_urandomb(rs, lp - 1), (p - 1) // q, p)
                if h != 1:
                    self.g.append(h)
            self.proof_bytes = lp // 8

    def edic_hash_block(prm, raw):
        sb, acc = prm.sect_bytes, mpz(1)
        for t in range(prm.m):
            acc = acc * powmod(prm.g[t], int.from_bytes(raw[t * sb:(t + 1) * sb], "big"), prm.p) % prm.p
        return acc

    def edic_prove(prm, hashes, idx, nus):
        """Prover: stored block hashes raised to the challenge coefficients."""
        acc = mpz(1)
        for j, s_ in zip(idx, nu_ints(nus)):
            acc = acc * powmod(hashes[int(j)], s_, prm.p) % prm.p
        return acc

    def edic_verify(prm, replica, idx, nus, proof):
        """Verifier (holds an intact replica): recompute the hash of the linear combination."""
        v = acc_sectors(replica, idx, nus, prm.sect_bytes, int(prm.q))
        acc = mpz(1)
        for t in range(prm.m):
            acc = acc * powmod(prm.g[t], v[t], prm.p) % prm.p
        return acc == proof
    HAVE_EDIC = True
except Exception:  # pragma: no cover
    HAVE_EDIC = False


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
