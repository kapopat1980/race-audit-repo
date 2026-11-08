# RACE-Audit

**Risk-Adaptive, Consensus-Enabled integrity auditing for data replicas on untrusted edge nodes**

This repository contains the complete implementation, experiment scripts and raw results for the article:

> D. Thakar and K. Popat, "RACE-Audit: Risk-adaptive probabilistic sampling and consensus-enabled lightweight cryptographic auditing for data integrity on untrusted edge nodes", submitted to *Cluster Computing* (Springer), 2026.

## What RACE-Audit does

A vendor caches replicas of a file on many edge nodes it does not fully trust and must check, without downloading them, that the replicas are intact. In every audit epoch RACE-Audit:

1. **Estimates risk.** Each node's propensity to hold corrupted data is tracked with a Beta posterior that forgets old evidence. Its upper confidence bound, multiplied by the node's user-value weight, gives a risk score.
2. **Allocates samples optimally.** The sampling budget is split across nodes by a closed-form water-filling rule that minimises the value-weighted miss probability. A minimum sample size (floor) per node bounds the detection time even for fully trusted nodes.
3. **Issues one nested challenge.** All nodes prove a prefix of the same keyed pseudo-random permutation. Honest replicas in the same tier return identical responses, so the verifier checks one representative per distinct response.
4. **Verifies with a pairing-free homomorphic MAC** over GF(2^127 − 1).
5. **Localises corruption.** Corrupted blocks are found by differential group testing, which derives half of every sub-proof by linearity.

## Repository layout

| Path | Contents |
|:--|:--|
| `src/race_lib.py` | RACE-Audit protocol and the baselines (U-HMAC, U-BLS, B-BLS, U-MHT) with real cryptography |
| `src/test_functional.py` | Functional tests: correctness, forgery/replay rejection, localisation, detection vs. analytical model |
| `src/bench.py` | Measured per-operation and per-node costs → `results/bench.json` |
| `src/sim.py` | Multi-epoch simulator, allocation strategies and cost model |
| `src/experiments.py` | All experiments E1–E7 (30 held-out seeds, 100–129) |
| `src/supp.py` | Larger uniform budgets and the empirical check of Corollary 1 |
| `src/eqcost.py` | Equal-assurance budgets and costs |
| `src/figs.py` | Figures 1–7 of the article |
| `src/report.py` | Prints every result in readable form |
| `results/` | Raw JSON results used in the article |
| `figures/` | Rendered figures |

## Requirements

Python 3.11 or later.

```bash
pip install -r requirements.txt
```

`py_arkworks_bls12381` provides the native BLS12-381 operations used by the pairing-based baselines.

## Reproducing the results

Run all commands from the repository root.

```bash
python3 src/test_functional.py          # correctness checks and detection validation
python3 src/bench.py                    # cost measurements on your machine (overwrites results/bench.json)
python3 src/experiments.py E1,E3,E5     # experiment sections (can run in parallel with the next line)
python3 src/experiments.py E2,E4
python3 src/supp.py
python3 src/eqcost.py
python3 src/figs.py
python3 src/report.py                   # summary of all numbers
```

Detection results are deterministic for a given seed. Timing results depend on the machine; the article's numbers were measured on two vCPUs of an Intel Xeon at 2.1 GHz with Python 3.13. Running `bench.py` again replaces the measured costs with those of your machine.

## Citation

If you use this code, please cite the article above (see also `CITATION.cff`).

## Licence

MIT — see `LICENSE`.

## Contact

Dr. Kalpesh Popat, Marwadi University, Rajkot, Gujarat, India — kapopat@gmail.com
