"""Edge-node process of the networked deployment.

One process = one edge node.  It maps the replica file copy-on-write, loads its tags, listens
on a TCP port and answers audit requests.  Faults are injected locally: at every epoch tick the
node, according to its (hidden) behaviour class, may silently corrupt a fraction of its blocks.
A configurable delay before every reply emulates the round-trip time to the verifier.

Frame format: 4-byte big-endian length, then payload; payload[0] is the opcode.
"""
import json, socket, struct, sys, time
import numpy as np
from race_lib import *

cfg = json.loads(sys.argv[1])
D = cfg["data_dir"]
n = cfg["n_blocks"]
schemes = set(cfg["schemes"].split(","))
rep = Replica(n, data=np.memmap(f"{D}/replica.bin", dtype=np.uint8, mode="c"))      # private copy-on-write
pristine = np.memmap(f"{D}/replica.bin", dtype=np.uint8, mode="r")                  # ground truth, evaluation only
tags = np.fromfile(f"{D}/tags.bin", dtype=np.uint8)
rng = np.random.default_rng([cfg["seed"], cfg["id"]])
dirty, onset = set(), -1
scan_cache = (None, None)
ALL = np.arange(n, dtype=np.uint32)

if "bls" in schemes:
    raw = open(f"{D}/bls_tags.bin", "rb").read()
    btags = [bls.G1Point.from_compressed_bytes_unchecked(raw[48 * j:48 * j + 48]) for j in range(n)]
if "mht" in schemes:
    mt = MerkleTree([sha(rep.raw_block(j)) for j in range(n)])
if "edic" in schemes:
    raw = open(f"{D}/edic_hashes.bin", "rb").read()
    ehash = [mpz(int.from_bytes(raw[256 * j:256 * j + 256], "big")) for j in range(n)]
    eprm = EdicParams(2048, 256)


def recv_exact(s, k):
    buf = bytearray()
    while len(buf) < k:
        chunk = s.recv(k - len(buf))
        if not chunk:
            raise ConnectionError
        buf += chunk
    return bytes(buf)


def tick(t):
    """Local fault injection (the node's hidden behaviour)."""
    global onset
    u, r = rng.random(), rng.random()
    if not dirty and u < cfg["p_fault"]:
        rho = cfg["rho_lo"] + r * (cfg["rho_hi"] - cfg["rho_lo"])
        bad = rng.choice(n, max(1, round(rho * n)), replace=False)
        rep.corrupt(bad, rng)
        dirty.update(int(j) for j in bad)
        onset = t


def handle(p):
    global scan_cache, onset
    op = p[:1]
    if op == b"T":
        tick(struct.unpack(">I", p[1:5])[0]); return b"k", False
    if op == b"R":
        cfg["rtt"] = struct.unpack(">d", p[1:9])[0]; return b"k", False
    if op == b"Q":
        return struct.pack(">Ii", len(dirty), onset), False
    t0 = time.process_time()
    if op == b"P":                                    # MAC proof over the nested prefix of length c
        seed, c = p[1:33], struct.unpack(">I", p[33:37])[0]
        idx, nus = expand_challenge(seed, n, c)
        mu, sig = mac_prove(rep, tags, idx, nus)
        out = mu + sig
    elif op == b"S":                                  # range proofs for localisation over the whole replica
        seed, k = p[1:33], struct.unpack(">I", p[33:37])[0]
        if scan_cache[0] != seed:
            scan_cache = (seed, coeffs(seed, n))
        nus = scan_cache[1]
        rr = struct.unpack(f">{2 * k}I", p[37:37 + 8 * k])
        out = b"".join(b"".join(mac_prove(rep, tags, ALL[rr[2 * i]:rr[2 * i + 1]], nus[16 * rr[2 * i]:16 * rr[2 * i + 1]]))
                       for i in range(k))
    elif op == b"G":                                  # serve blocks (+ tags) as a repair source
        k = struct.unpack(">I", p[1:5])[0]
        js = struct.unpack(f">{k}I", p[5:5 + 4 * k])
        out = b"".join(rep.raw_block(j) + tags[16 * j:16 * j + 16].tobytes() for j in js)
    elif op == b"W":                                  # accept repaired blocks
        k = struct.unpack(">I", p[1:5])[0]
        off = 5
        for _ in range(k):
            j = struct.unpack(">I", p[off:off + 4])[0]
            blk = p[off + 4:off + 4 + BLOCK_BYTES]
            rep.repair(j, blk)
            if blk == pristine[j * BLOCK_BYTES:(j + 1) * BLOCK_BYTES].tobytes():
                dirty.discard(j)
            off += 4 + BLOCK_BYTES
        if not dirty:
            onset = -1
        out = b"k"
    elif op == b"B":                                  # BLS homomorphic-signature proof
        seed, c = p[1:33], struct.unpack(">I", p[33:37])[0]
        idx, nus = expand_challenge(seed, n, c)
        mu, sig = bls_prove(rep, btags, idx, nus)
        out = b"".join(int(m).to_bytes(32, "big") for m in mu) + bytes(sig.to_compressed_bytes())
    elif op == b"M":                                  # Merkle proof: sampled blocks + authentication paths
        seed, c = p[1:33], struct.unpack(">I", p[33:37])[0]
        idx = prp_prefix(seed, n, c)
        out = b"".join(rep.raw_block(int(j)) + b"".join(mt.path(int(j))) for j in idx)
    elif op == b"E":                                  # EDI-C proof
        seed, c = p[1:33], struct.unpack(">I", p[33:37])[0]
        idx, nus = expand_challenge(seed, n, c)
        if dirty:                                     # a corrupted node's stored hash list reflects its data
            hs = {int(j): (edic_hash_block(eprm, rep.raw_block(int(j))) if int(j) in dirty else ehash[int(j)]) for j in idx}
        else:
            hs = ehash
        out = int(edic_prove(eprm, hs, idx, nus)).to_bytes(256, "big")
    else:
        raise ValueError(op)
    return out + struct.pack(">d", time.process_time() - t0), True


srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", cfg["port"])); srv.listen(1)
conn, _ = srv.accept()
conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
try:
    while True:
        ln = struct.unpack(">I", recv_exact(conn, 4))[0]
        p = recv_exact(conn, ln)
        if p[:1] == b"X":
            break
        out, delayed = handle(p)
        if delayed and cfg["rtt"] > 0:
            time.sleep(cfg["rtt"])
        conn.sendall(struct.pack(">I", len(out)) + out)
except ConnectionError:
    pass
