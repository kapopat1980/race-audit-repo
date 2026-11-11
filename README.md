# RACE-Audit

**Risk-Adaptive, Consensus-Enabled integrity auditing for data replicas on untrusted edge nodes**

This repository contains the complete implementation, experiment scripts and raw results for the article:

> D. Thakar and K. Popat, "Risk-adaptive probabilistic sampling and consensus-enabled lightweight cryptographic auditing for data integrity on untrusted edge nodes", 2026.

Version 1.1 adds native-code (C) kernels, a networked deployment (one process per edge node, TCP), a re-implementation of the EDI-C homomorphic-hash proof as a baseline, and a workload calibrated with the public Backblaze drive-reliability statistics.

## What RACE-Audit does

A vendor caches replicas of a file on many edge nodes it does not fully trust and must check, without downloading them, that the replicas are intact. In every audit epoch RACE-Audit:

1. **Estimates risk.** Each node's propensity to hold corrupted data is tracked with a Beta posterior that forgets old evidence. Its upper confidence bound, multiplied by the node's user-value weight, gives a risk score.
2. **Allocates samples optimally.** The sampling budget is split across nodes by a closed-form water-filling rule that minimises the value-weighted miss probability. A minimum sample size (floor) per node bounds the detection time even for fully trusted nodes.
3. **Issues one nested challenge.** All nodes prove a prefix of the same keyed pseudo-random permutation. Honest replicas in the same tier return identical responses, so the verifier checks one representative per distinct response.
4. **Verifies with a pairing-free homomorphic MAC** over GF(2^127 − 1).
5. **Localises and repairs corruption.** Corrupted blocks are found by level-synchronous differential group testing (half of every sub-proof is derived by linearity); replacement blocks are fetched from a passing node and checked against their tags before they are written back.

## Repository layout

| Path | Contents |
|:--|:--|
| `src/race_native.c` | Native kernels: GF(2^127 − 1) arithmetic, HMAC-SHA-256 PRF, challenge expansion, tag generation, proof, verification |
| `src/race_lib.py` | RACE-Audit protocol and the baselines (U-HMAC, U-BLS, B-BLS, U-MHT, EDI-C); builds and loads the native library |
| `src/race_lib_v1_python.py` | Earlier pure-Python implementation of the library (reference only) |
| `src/test_functional.py` | Functional tests: native kernels vs. Python reference, forgery/replay/replacement rejection, localisation and repair, detection vs. analytical model |
| `src/bench.py`, `src/bench3.sh`, `src/bench_median.py` | Measured per-operation and per-node costs (five runs, element-wise median) → `results/bench.json` |
| `src/sim.py` | Multi-epoch simulator, allocation strategies and cost model |
| `src/experiments.py`, `src/verifier_timing.py` | Experiments E1–E7 (30 held-out seeds, 100–129); direct timing of the verifier |
| `src/backblaze.py`, `src/exp_field.py` | Drive-reliability table (Backblaze Drive Stats 2025) and the field-calibrated experiment E8 |
| `src/deploy_prep.py`, `src/deploy_node.py`, `src/deploy_run.py`, `src/deploy_summary.py` | Networked deployment: offline set-up, edge-node process, vendor/verifier and fleet driver, result summary |
| `src/supp.py`, `src/eqcost.py` | Larger uniform budgets, empirical check of Corollary 1, equal-assurance budgets and costs |
| `src/figs.py`, `src/report.py` | Figures 1–9 of the article; readable summary of the results |
| `results/` | Raw JSON results used in the article |
| `figures/` | Rendered figures |

## Requirements

Python 3.11 or later, gcc, and the OpenSSL development headers (`libssl-dev`); the native library is compiled automatically on first import.

```bash
pip install -r requirements.txt
```

`py_arkworks_bls12381` provides the BLS12-381 operations of the pairing-based baselines and `gmpy2` the modular exponentiations of EDI-C.

## Reproducing the results

Run all commands from the repository root.

```bash
python3 src/test_functional.py            # correctness checks and detection validation
bash src/bench3.sh                        # cost measurements on your machine (overwrites results/bench.json)
python3 src/experiments.py E1,E3,E5       # synthetic-fleet experiments (can run in parallel with the next line)
python3 src/experiments.py E2,E4
python3 src/verifier_timing.py           # direct timing of the RACE-Audit verifier on real proofs
python3 src/supp.py
python3 src/eqcost.py
python3 src/exp_field.py                  # field-calibrated fleet (E8)

python3 src/deploy_prep.py                # replica, keys, tags, BLS tags, EDI-C hashes (about 15 minutes)
python3 src/deploy_run.py cost mac 64 30  # deployment: cost of RACE-Audit, U-HMAC, U-MHT with emulated latency
python3 src/deploy_run.py cost bls 64 20
python3 src/deploy_run.py cost edic 64 20
for s in 100 101 102 103 104; do          # deployment: detection, localisation and repair
  python3 src/deploy_run.py detect race $s 64 200 256
  python3 src/deploy_run.py detect uniform $s 64 200 256
done
python3 src/deploy_summary.py

python3 src/figs.py
python3 src/report.py
```

Detection results of the simulator are deterministic for a given seed. Timing results depend on the machine; the article's numbers were measured on two vCPUs of an Intel Xeon at 2.1 GHz. The deployment starts one process per edge node on the local machine (64 nodes need about 2 GB of memory).

## Data source

`src/backblaze.py` reproduces the annual 2025 table of the Backblaze Drive Stats data set (drive count, drive days and failures per drive model), which is publicly available at <https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data>.

## Citation

If you use this code, please cite the article above (see also `CITATION.cff`).

## Licence

MIT — see `LICENSE`.

## Contact

Dr. Kalpesh Popat, Marwadi University, Rajkot, Gujarat, India — kapopat@gmail.com
