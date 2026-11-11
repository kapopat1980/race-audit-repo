"""One-time set-up for the networked deployment (what the data owner does offline):
writes the replica file, the verifier's keys, the MAC tags, and the per-block
authenticators needed by the baselines (BLS signatures, EDI-C homomorphic hashes)."""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from race_lib import *

D = "deploy_data"
N_BLOCKS = 16384
os.makedirs(D, exist_ok=True)

def edic_chunk(args):
    lo, hi = args
    prm = EdicParams(2048, 256)
    data = honest_data(N_BLOCKS)
    return b"".join(int(edic_hash_block(prm, data[j * BLOCK_BYTES:(j + 1) * BLOCK_BYTES].tobytes())).to_bytes(256, "big")
                    for j in range(lo, hi))

if __name__ == "__main__":
    what = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"mac", "bls", "edic"}
    hon = Replica(N_BLOCKS)
    if "mac" in what:
        hon.data.tofile(f"{D}/replica.bin")
        key = MacKey()
        json.dump({"k": key.k.hex(), "alpha": key.alpha.hex(), "epoch_key": os.urandom(32).hex(),
                   "bls_x": hex(int.from_bytes(os.urandom(32), "big") % R_ORDER)}, open(f"{D}/keys.json", "w"))
        mac_taggen(key, hon).tofile(f"{D}/tags.bin")
        print("mac done", flush=True)
    if "bls" in what:
        t0 = time.time()
        bk = BlsKey(); bk.x = int(json.load(open(f"{D}/keys.json"))["bls_x"], 16); bk.v = bls.G2Point() * scal(bk.x)
        with open(f"{D}/bls_tags.bin", "wb") as f:
            for j in range(N_BLOCKS):
                f.write(bytes(bls_tag(bk, hon.raw_block(j), j).to_compressed_bytes()))
        print("bls done", round(time.time() - t0), flush=True)
    if "edic" in what:
        t0 = time.time()
        step = 64
        with Pool(2) as pool, open(f"{D}/edic_hashes.bin", "wb") as f:
            for i, chunk in enumerate(pool.imap(edic_chunk, [(lo, min(lo + step, N_BLOCKS)) for lo in range(0, N_BLOCKS, step)])):
                f.write(chunk)
                if i % 32 == 0:
                    print("edic", i * step, round(time.time() - t0), flush=True)
        print("edic done", round(time.time() - t0), flush=True)
