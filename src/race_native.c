/* Native kernels for RACE-Audit and the baselines.
 *   - GF(p) arithmetic, p = 2^127 - 1 (homomorphic MAC: tag generation, proof, verification)
 *   - HMAC-SHA-256 PRF with pre-computed pad states (OpenSSL SHA-256)
 *   - unreduced multi-precision accumulation used by the BLS and EDI-C provers/verifiers
 * Field elements are 16-byte little-endian buffers.  Build:
 *   gcc -O3 -fPIC -shared -Wno-deprecated-declarations race_native.c -o libracenative.so -lcrypto
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <openssl/sha.h>

typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef uint8_t u8;

#define S_SECT 274
#define SECT 15
#define BLOCK 4096

static const u128 P = ((((u128)1) << 127) - 1);

static inline u128 red(u128 x) { x = (x & P) + (x >> 127); if (x >= P) x -= P; return x; }
static inline u128 addm(u128 a, u128 b) { return red(a + b); }           /* a, b < 2^127 */
static inline u128 mulm(u128 a, u128 b) {                                /* a, b < 2^127 */
    u64 a0 = (u64)a, a1 = (u64)(a >> 64), b0 = (u64)b, b1 = (u64)(b >> 64);
    u128 p00 = (u128)a0 * b0, p01 = (u128)a0 * b1, p10 = (u128)a1 * b0, p11 = (u128)a1 * b1;
    u128 mid = p01 + p10;                                                /* < 2^128 */
    u128 lo = p00 + (mid << 64);
    u128 hi = p11 + (mid >> 64) + (lo < p00);                            /* a*b = hi*2^128 + lo, hi < 2^126 */
    return red(red(lo) + (hi << 1));                                     /* 2^128 = 2 (mod p) */
}
static inline u128 load(const u8 *b) { u128 x; memcpy(&x, b, 16); return x; }
static inline void store(u8 *b, u128 x) { memcpy(b, &x, 16); }
static inline u128 be_sector(const u8 *b, int len) {
    if (len == SECT) {                                                   /* 15 big-endian bytes */
        u64 hi, lo; memcpy(&hi, b, 8); memcpy(&lo, b + 7, 8);
        hi = __builtin_bswap64(hi); lo = __builtin_bswap64(lo) & 0x00FFFFFFFFFFFFFFULL;
        return (((u128)hi) << 56) | lo;
    }
    u128 x = 0; for (int i = 0; i < len; i++) x = (x << 8) | b[i]; return x;
}
static inline int sect_len(int t) { return t == S_SECT - 1 ? BLOCK - SECT * (S_SECT - 1) : SECT; }

/* ------------------------------------------------------------------ PRF */
typedef struct { SHA256_CTX in, out; } prf_ctx;
static void prf_init(prf_ctx *c, const u8 *key) {
    u8 k[64] = {0}, p[64];
    memcpy(k, key, 32);
    for (int i = 0; i < 64; i++) p[i] = k[i] ^ 0x36;
    SHA256_Init(&c->in); SHA256_Update(&c->in, p, 64);
    for (int i = 0; i < 64; i++) p[i] = k[i] ^ 0x5c;
    SHA256_Init(&c->out); SHA256_Update(&c->out, p, 64);
}
static inline u128 prf_eval(const prf_ctx *c, u64 j) {     /* HMAC(k, "tag" || BE64(j))[:16] mod p */
    u8 msg[11] = {'t', 'a', 'g'}, d[32];
    for (int i = 0; i < 8; i++) msg[3 + i] = (u8)(j >> (56 - 8 * i));
    SHA256_CTX t = c->in; SHA256_Update(&t, msg, 11); SHA256_Final(d, &t);
    t = c->out; SHA256_Update(&t, d, 32); SHA256_Final(d, &t);
    u128 x = 0; for (int i = 0; i < 16; i++) x = (x << 8) | d[i];
    return red(x);
}
void race_prf_batch(const u8 *key, const uint32_t *idx, uint32_t c, u8 *out) {
    prf_ctx ctx; prf_init(&ctx, key);
    for (uint32_t i = 0; i < c; i++) store(out + 16 * i, prf_eval(&ctx, idx[i]));
}

/* ------------------------------------------------------------------ challenge expansion
 * First `len` elements of a seeded pseudo-random permutation of [0, n) (partial Fisher-Yates driven
 * by SHA-256 in counter mode).  Prefixes are nested.  For very large n the O(n) identity array
 * would be replaced by a sparse map; n is small here. */
void race_prp_prefix(const u8 *seed, uint32_t n, uint32_t len, uint32_t *out) {
    uint32_t *perm = (uint32_t *)malloc((size_t)n * 4);
    for (uint32_t i = 0; i < n; i++) perm[i] = i;
    u8 in[32 + 3 + 8], buf[32];
    memcpy(in, seed, 32); in[32] = 'p'; in[33] = 'r'; in[34] = 'p';
    int have = 0; u64 ctr = 0;
    for (uint32_t i = 0; i < len && i < n; i++) {
        if (have < 8) { memcpy(in + 35, &ctr, 8); ctr++; SHA256(in, sizeof in, buf); have = 32; }
        u64 draw; memcpy(&draw, buf + 32 - have, 8); have -= 8;
        uint32_t r = i + (uint32_t)(((u128)draw * (n - i)) >> 64);
        uint32_t tmp = perm[i]; perm[i] = perm[r]; perm[r] = tmp;
        out[i] = perm[i];
    }
    free(perm);
}

/* ------------------------------------------------------------------ MAC */
void race_taggen(const u8 *data, u64 n, const u8 *key, const u8 *alpha, u8 *tags) {
    prf_ctx ctx; prf_init(&ctx, key);
    u128 al[S_SECT];
    for (int t = 0; t < S_SECT; t++) al[t] = load(alpha + 16 * t);
    for (u64 j = 0; j < n; j++) {
        const u8 *b = data + j * BLOCK;
        u128 acc = prf_eval(&ctx, j);
        for (int t = 0; t < S_SECT; t++) acc = addm(acc, mulm(al[t], be_sector(b + SECT * t, sect_len(t))));
        store(tags + 16 * j, acc);
    }
}
void race_prove(const u8 *data, const u8 *tags, const uint32_t *idx, const u8 *nus, uint32_t c, u8 *mu_out, u8 *sig_out) {
    u128 mu[S_SECT]; memset(mu, 0, sizeof mu);
    u128 sig = 0;
    for (uint32_t i = 0; i < c; i++) {
        u128 nu = red(load(nus + 16 * i));
        const u8 *b = data + (size_t)idx[i] * BLOCK;
        for (int t = 0; t < S_SECT; t++) mu[t] = addm(mu[t], mulm(nu, be_sector(b + SECT * t, sect_len(t))));
        sig = addm(sig, mulm(nu, load(tags + 16 * (size_t)idx[i])));
    }
    for (int t = 0; t < S_SECT; t++) store(mu_out + 16 * t, mu[t]);
    store(sig_out, sig);
}
int race_verify(const u8 *alpha, const u8 *mu, const u8 *sig, const u8 *tau) {
    u128 acc = load(tau);
    for (int t = 0; t < S_SECT; t++) acc = addm(acc, mulm(load(alpha + 16 * t), load(mu + 16 * t)));
    return acc == load(sig);
}
/* prefix sums tau(0..c) of nu_i * f_k(idx_i); out has (c+1) elements */
void race_tau_prefix(const u8 *key, const uint32_t *idx, const u8 *nus, uint32_t c, u8 *out) {
    prf_ctx ctx; prf_init(&ctx, key);
    u128 acc = 0; store(out, acc);
    for (uint32_t i = 0; i < c; i++) {
        acc = addm(acc, mulm(red(load(nus + 16 * i)), prf_eval(&ctx, idx[i])));
        store(out + 16 * (i + 1), acc);
    }
}
void race_tau(const u8 *key, const uint32_t *idx, const u8 *nus, uint32_t c, u8 *out) {
    prf_ctx ctx; prf_init(&ctx, key);
    u128 acc = 0;
    for (uint32_t i = 0; i < c; i++) acc = addm(acc, mulm(red(load(nus + 16 * i)), prf_eval(&ctx, idx[i])));
    store(out, acc);
}
/* tau from a cached table of f_k(j) values (n * 16 bytes) */
void race_tau_cached(const u8 *ftab, const uint32_t *idx, const u8 *nus, uint32_t c, u8 *out) {
    u128 acc = 0;
    for (uint32_t i = 0; i < c; i++) acc = addm(acc, mulm(red(load(nus + 16 * i)), load(ftab + 16 * (size_t)idx[i])));
    store(out, acc);
}
/* out = a - b (element-wise, n field elements) */
void race_vec_sub(const u8 *a, const u8 *b, uint32_t n, u8 *out) {
    for (uint32_t i = 0; i < n; i++) store(out + 16 * i, addm(load(a + 16 * i), P - load(b + 16 * i)));
}

/* ------------------------------------------------------------------ unreduced accumulation
 * out[t] = sum_i nu_i * sector_t(block idx_i) as a 7-limb little-endian integer (56 bytes),
 * sectors are sect_bytes (<= 32) big-endian bytes; the caller reduces modulo its own modulus. */
void acc_unreduced(const u8 *data, const uint32_t *idx, const u8 *nus, uint32_t c, int sect_bytes, int nsect, u8 *out) {
    u64 acc[nsect][7];
    memset(acc, 0, sizeof acc);
    for (uint32_t i = 0; i < c; i++) {
        u64 a[2]; memcpy(a, nus + 16 * i, 16);
        const u8 *blk = data + (size_t)idx[i] * BLOCK;
        for (int t = 0; t < nsect; t++) {
            int off = t * sect_bytes, len = sect_bytes;
            if (off + len > BLOCK) len = BLOCK - off;
            u64 b[4] = {0, 0, 0, 0};
            for (int k = 0; k < len; k++) {                 /* big-endian bytes -> little-endian limbs */
                int pos = len - 1 - k;                      /* byte significance */
                b[pos / 8] |= ((u64)blk[off + k]) << (8 * (pos % 8));
            }
            u64 prod[6] = {0, 0, 0, 0, 0, 0};
            for (int x = 0; x < 2; x++) {
                u64 carry = 0;
                for (int y = 0; y < 4; y++) {
                    u128 tt = (u128)a[x] * b[y] + prod[x + y] + carry;
                    prod[x + y] = (u64)tt; carry = (u64)(tt >> 64);
                }
                prod[x + 4] = carry;
            }
            u64 carry = 0;
            for (int k = 0; k < 7; k++) {
                u128 tt = (u128)acc[t][k] + (k < 6 ? prod[k] : 0) + carry;
                acc[t][k] = (u64)tt; carry = (u64)(tt >> 64);
            }
        }
    }
    memcpy(out, acc, (size_t)nsect * 56);
}
