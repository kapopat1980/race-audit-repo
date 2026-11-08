"""Multi-epoch Monte-Carlo evaluation of sampling/allocation strategies and
cost evaluation of all schemes using the measured unit costs in results/bench.json.

Detection outcomes use the exact hypergeometric miss probability of sampling c
distinct blocks out of n when d are corrupted (validated against the real
protocol in test_functional.py); the MAC forgery probability (<= 2^-126) is
neglected.
"""
import json, math, sys
import numpy as np
from scipy.special import gammaln
from scipy import stats
from race_lib import waterfill, quantise_to_tiers, beta_ucb, S_SECT, FE_BYTES

BENCH = json.load(open("results/bench.json"))
N_BLOCKS = 16384
TIERS = (32, 64, 128, 256, 512, 1024)
S_BLS = 133


def miss_prob(n, d, c):
    c = np.asarray(c, float)
    d = np.asarray(d, float)
    ok = c <= n - d
    with np.errstate(invalid="ignore"):
        lp = gammaln(n - d + 1) - gammaln(np.maximum(n - d - c, 0) + 1) - gammaln(n + 1) + gammaln(n - c + 1)
    return np.where(ok, np.exp(lp), 0.0)


# ------------------------------------------------------------------ population
def make_population(N, rng, mix=(0.7, 0.2, 0.1), onoff=0):
    types = rng.choice(3, N, p=mix)
    p_fault = np.array([0.002, 0.03, 0.2])[types]
    rho_lo = np.array([0.005, 0.005, 0.002])[types]
    rho_hi = np.array([0.05, 0.03, 0.01])[types]
    value = rng.lognormal(0, 0.75, N)
    value /= value.mean()
    pop = dict(types=types, p_fault=p_fault, rho_lo=rho_lo, rho_hi=rho_hi, value=value,
               onoff=np.zeros(N, bool), whitewash=np.zeros(N, bool))
    if onoff:
        sel = rng.choice(N, onoff, replace=False)
        pop["onoff"][sel] = True
        pop["types"][sel] = 3
    return pop


# ------------------------------------------------------------------ strategies
class Uniform:
    name = "Uniform"
    def __init__(s, N, B, **kw): s.c = np.full(N, float(B // N))
    def alloc(s, value): return s.c
    def update(s, aud, fail): pass
    def reset(s, i): pass


class RepGated:
    """Reputation-gated auditing (EDI-C / dynamic-assignment style): Beta mean
    reputation without forgetting; audits the M least-reputable nodes (plus
    eps-random exploration) at a fixed sample size giving 99% detection at 1%."""
    name = "Rep-Gated"
    def __init__(s, N, B, rng, cR=460, eps=0.2, **kw):
        s.N, s.B, s.cR, s.eps, s.rng = N, B, cR, eps, rng
        s.f, s.h = np.zeros(N), np.zeros(N)
    def alloc(s, value):
        M = int(s.B // s.cR)
        if M >= s.N:
            return np.full(s.N, float(s.B // s.N))
        risk = (s.f + 1) / (s.f + s.h + 10)
        k_exp = int(round(s.eps * M))
        top = np.argsort(-risk + 1e-9 * s.rng.random(s.N))[: M - k_exp]
        rest = np.setdiff1d(np.arange(s.N), top)
        expl = s.rng.choice(rest, k_exp, replace=False) if k_exp else np.array([], int)
        c = np.zeros(s.N); c[top] = s.cR; c[expl] = s.cR
        return c
    def update(s, aud, fail):
        s.h += aud & ~fail; s.f += fail
    def reset(s, i): s.f[i] = 0; s.h[i] = 0


class Race:
    name = "RACE-Audit"
    def __init__(s, N, B, kappa=1.0, lam=0.99, cmin=32, w=5.0, tiers=TIERS, use_tiers=True,
                 rho_star=0.005, use_value=True, name=None, **kw):
        s.N, s.B = N, B
        s.L = -math.log(1 - rho_star)
        s.kappa, s.lam, s.cmin, s.w = kappa, lam, cmin, w
        s.tiers = tuple(sorted(set(tiers) | ({cmin} if cmin > 0 else {0})))
        s.use_tiers, s.use_value = use_tiers, use_value
        s.f, s.h = np.zeros(N), np.zeros(N)
        if name: s.name = name
    def alloc(s, value):
        v = value if s.use_value else np.ones(s.N)
        r = v * beta_ucb(s.f, s.h, 1.0, 9.0, s.kappa)
        c = waterfill(r, s.B, s.L, s.cmin, max(s.tiers))
        return quantise_to_tiers(c, s.tiers, s.B) if s.use_tiers else np.floor(c)
    def update(s, aud, fail):
        s.f *= s.lam; s.h *= s.lam
        s.h += aud & ~fail; s.f += s.w * fail
    def reset(s, i): s.f[i] = 0; s.h[i] = 0


# ------------------------------------------------------------------ epoch simulation
def run(strategy_factory, N=100, B_per_node=256, epochs=300, seed=0, onoff=0, onoff_start=100,
        onoff_rho=0.003, onoff_p=0.5, whitewash=False, mix=(0.7, 0.2, 0.1), record_alloc=False):
    rng = np.random.default_rng(seed)
    pop = make_population(N, rng, mix=mix, onoff=onoff)
    srng = np.random.default_rng(seed + 10_000)        # identical corruption stream across strategies
    strat = strategy_factory(N=N, B=N * B_per_node, rng=np.random.default_rng(seed + 20_000))
    corrupt = np.zeros(N, bool)
    d = np.zeros(N)
    onset = np.full(N, -1)
    exposure = 0.0
    exposure_unw = 0.0
    trace = np.zeros(epochs)
    exp_attack = 0.0
    episodes, detected_eps, ttd = 0, 0, []
    total_samples = 0.0
    allocs = []
    for t in range(epochs):
        # --- corruption arrivals (pre-drawn random numbers keep paths paired)
        u = srng.random(N); rr = srng.random(N); uo = srng.random(N)
        p = pop["p_fault"].copy()
        active_onoff = pop["onoff"] & (t >= onoff_start)
        p[pop["onoff"]] = np.where(active_onoff[pop["onoff"]], onoff_p, 0.0)
        new = (~corrupt) & (u < p)
        rho = pop["rho_lo"] + rr * (pop["rho_hi"] - pop["rho_lo"])
        rho[pop["onoff"]] = onoff_rho
        d = np.where(new, np.maximum(1, np.round(rho * N_BLOCKS)), d)
        onset = np.where(new, t, onset)
        episodes += int(new.sum())
        corrupt |= new
        # --- audit
        c = strat.alloc(pop["value"])
        total_samples += c.sum()
        if record_alloc:
            allocs.append(c.copy())
        aud = c > 0
        pm = miss_prob(N_BLOCKS, np.where(corrupt, d, 0), c)
        det = corrupt & aud & (srng.random(N) >= pm)
        # --- outcomes
        for i in np.where(det)[0]:
            ttd.append(t - onset[i]); detected_eps += 1
        undetected = corrupt & ~det
        exposure += float((pop["value"] * undetected).sum())
        exposure_unw += float(undetected.sum())
        if onoff:
            ea = float((pop["value"] * (undetected & pop["onoff"])).sum())
            exp_attack += ea
            trace[t] = ea
        strat.update(aud, det)
        corrupt &= ~det                     # repair from a verified replica
        if whitewash:                       # caught malicious node rejoins under a fresh identity
            ww = det & (pop["types"] >= 2)
            for i in np.where(ww)[0]:
                strat.reset(i)
    # censored episodes: still undetected at the end
    for i in np.where(corrupt)[0]:
        ttd.append(epochs - onset[i])
    out = dict(exposure=exposure / epochs, exposure_unw=exposure_unw / epochs, exposure_attack=exp_attack / epochs, trace=trace,
               mttd=float(np.mean(ttd)) if ttd else 0.0,
               p95_ttd=float(np.percentile(ttd, 95)) if ttd else 0.0,
               det_frac=detected_eps / max(episodes, 1), episodes=episodes,
               samples_per_epoch=total_samples / epochs)
    if record_alloc:
        out["allocs"] = np.array(allocs)
        out["pop"] = pop
    return out


# ------------------------------------------------------------------ cost model
def interp(table, c):
    xs = np.array(sorted(int(k) for k in table))
    ys = np.array([table[str(k)] if str(k) in table else table[k] for k in xs])
    c = np.asarray(c, float)
    return np.interp(c, xs, ys, left=None, right=None) * np.where(c > xs.max(), c / xs.max(), 1.0)


def multiexp_pt(m):
    tbl = BENCH["multiexp_per_pt"]
    xs = np.array(sorted(int(k) for k in tbl)); ys = np.array([tbl[str(k)] for k in xs])
    return float(np.interp(np.log2(max(m, 1)), np.log2(xs), ys))


def epoch_costs(c, n_fail_groups=0, n_tiers_used=None):
    """Per-epoch costs for every scheme given a sample-size vector c (zeros = not audited).
    Returns dict scheme -> (verifier_s, edge_total_s, edge_max_s, bytes_down, bytes_up)."""
    pc = BENCH["per_c"]
    c = np.asarray(c, float)
    a = c[c > 0]
    Na = len(a)
    res = {}
    # RACE (MAC tags, shared tau prefix, consensus grouping)
    groups = (len(np.unique(a)) if n_tiers_used is None else n_tiers_used) + n_fail_groups
    v = float(interp(pc["mac_tau"], a.max())) + groups * BENCH["t_mac_check"] + Na * BENCH["t_eq_compare"]
    e = interp(pc["mac_prove"], a)
    resp = (S_SECT + 1) * FE_BYTES
    res["RACE-Audit"] = (v, e.sum(), e.max(), Na * resp, 32 + len(c))
    # U-HMAC (same tags, independent per-node challenges)
    v = float((interp(pc["mac_tau"], a) + BENCH["t_mac_check"]).sum())
    res["U-HMAC"] = (v, e.sum(), e.max(), Na * resp, Na * 36)
    # U-BLS (per-node pairing verification)
    eb = interp(pc["bls_prove"], a)
    v = float(interp(pc["bls_verify"], a).sum())
    rb = 48 + S_BLS * 32
    res["U-BLS"] = (v, eb.sum(), eb.max(), Na * rb, Na * 36)
    # B-BLS (aggregate verification, single pairing product)
    m = a.sum() + S_BLS
    v = m * multiexp_pt(m) + m * BENCH["t_scal_conv"] + Na * BENCH["t_g1_add"] + 2 * BENCH["t_pairing"]
    res["B-BLS"] = (v, eb.sum(), eb.max(), Na * rb, Na * 36)
    # U-MHT (Merkle proofs returning sampled blocks)
    depth = math.ceil(math.log2(N_BLOCKS))
    em = interp(pc["mht_prove"], a)
    v = float(interp(pc["mht_verify"], a).sum())
    res["U-MHT"] = (v, em.sum(), em.max(), float(a.sum() * (4096 + depth * 32)), Na * 36)
    return res


def bbls_localisation_cost(c, failing):
    """Extra verifier cost for B-BLS when the aggregate fails: recursive halving
    over the node set; each subset check = multiexp over its points + 2 pairings."""
    c = np.asarray(c, float)
    nodes = list(np.where(c > 0)[0])
    fail = set(failing)
    cost, checks = 0.0, 0
    stack = [nodes]
    while stack:
        S = stack.pop()
        m = c[S].sum() + S_BLS
        cost += m * multiexp_pt(m) + 2 * BENCH["t_pairing"]; checks += 1
        if not (set(S) & fail):
            continue
        if len(S) == 1:
            continue
        h = len(S) // 2
        stack += [S[:h], S[h:]]
    return cost, checks


def ci95(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
