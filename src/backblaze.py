"""Drive-model reliability statistics used to calibrate the field workload.

Source: Backblaze Drive Stats, annual table for 1 January - 31 December 2025 (public data set,
https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data ; table as published in
"Backblaze Drive Stats for 2025").  Columns: manufacturer, model, size (TB), drive count,
average age (months), drive days, failures, published AFR (%).
"""
import numpy as np

TABLE = [
 ("HGST", "HMS5C4040BLE640", 4, 4121, 103.3, 375277, 5, 0.49),
 ("HGST", "HUH728080ALE600", 8, 1084, 90.4, 389340, 38, 3.56),
 ("HGST", "HUH721212ALE600", 12, 2623, 72.5, 951081, 17, 0.65),
 ("HGST", "HUH721212ALE604", 12, 13656, 53.5, 4853717, 358, 2.69),
 ("HGST", "HUH721212ALN604", 12, 10195, 78.0, 3628418, 393, 3.95),
 ("Seagate", "ST8000DM002", 8, 9078, 109.9, 3276650, 156, 1.74),
 ("Seagate", "ST8000NM0055", 8, 13533, 98.6, 4870236, 291, 2.18),
 ("Seagate", "ST10000NM0086", 10, 1044, 95.2, 367460, 57, 5.66),
 ("Seagate", "ST12000NM0007", 12, 1083, 72.9, 366658, 55, 5.48),
 ("Seagate", "ST12000NM0008", 12, 19145, 67.7, 6877112, 486, 2.58),
 ("Seagate", "ST12000NM000J", 12, 1075, 16.4, 347582, 4, 0.42),
 ("Seagate", "ST12000NM001G", 12, 13347, 57.8, 4817914, 127, 0.96),
 ("Seagate", "ST14000NM000J", 14, 421, 14.2, 117113, 7, 2.18),
 ("Seagate", "ST14000NM001G", 14, 10668, 57.3, 3849616, 142, 1.35),
 ("Seagate", "ST14000NM0138", 14, 1322, 59.6, 467111, 68, 5.31),
 ("Seagate", "ST16000NM001G", 16, 34563, 38.6, 12381689, 182, 0.54),
 ("Seagate", "ST16000NM002J", 16, 466, 35.9, 168586, 1, 0.22),
 ("Seagate", "ST24000NM002H", 24, 8549, 7.6, 1941010, 146, 2.75),
 ("Toshiba", "MG07ACA14TA", 14, 37705, 61.2, 13654728, 432, 1.15),
 ("Toshiba", "MG07ACA14TEY", 14, 1015, 37.1, 324488, 20, 2.25),
 ("Toshiba", "MG08ACA16TA", 16, 40411, 27.8, 14623715, 362, 0.90),
 ("Toshiba", "MG08ACA16TE", 16, 6324, 46.3, 2188375, 79, 1.32),
 ("Toshiba", "MG08ACA16TEY", 16, 5197, 47.5, 1843841, 318, 6.30),
 ("Toshiba", "MG09ACA16TE", 16, 357, 6.0, 63327, 3, 1.73),
 ("Toshiba", "MG10ACA20TE", 20, 18083, 10.1, 4909379, 80, 0.59),
 ("Toshiba", "MG11ACA24TE", 24, 4806, 2.6, 374087, 7, 0.68),
 ("WDC", "WUH721414ALE6L4", 14, 8707, 58.3, 3141290, 70, 0.81),
 ("WDC", "WUH721816ALE6L0", 16, 3029, 47.5, 1092886, 59, 1.97),
 ("WDC", "WUH721816ALE6L4", 16, 26856, 31.9, 9671421, 176, 0.66),
 ("WDC", "WUH722222ALE6L4", 22, 44577, 13.6, 13612879, 177, 0.47),
 ("WDC", "WUH722626ALE6L4", 26, 1201, 2.5, 91690, 1, 0.40),
]
PUBLISHED_TOTAL = dict(drive_days=115638676, failures=4317, afr=1.36)

COUNT = np.array([r[3] for r in TABLE], float)
DAYS = np.array([r[5] for r in TABLE], float)
FAIL = np.array([r[6] for r in TABLE], float)
DAILY = FAIL / DAYS                       # failures per drive-day, per model
AFR = 100 * 365 * DAILY


def population(N, rng, gamma=1.0, phi=0.0, **kw):
    """Fleet of N edge nodes; node i runs on a drive model drawn in proportion to the published drive
    counts.  A silent-corruption event starts on a clean node with daily probability
    gamma x (model failures per drive-day): gamma is the assumed number of silent-corruption events
    per recorded drive failure.  Corrupted fraction and data value as in the synthetic workload."""
    cls = rng.choice(len(TABLE), N, p=COUNT / COUNT.sum())
    value = rng.lognormal(0, 0.75, N); value /= value.mean()
    pop = dict(types=np.zeros(N, int), classes=cls, p_fault=np.minimum(1.0, gamma * DAILY[cls]),
               rho_lo=np.full(N, 0.005), rho_hi=np.full(N, 0.05), value=value,
               onoff=np.zeros(N, bool), whitewash=np.zeros(N, bool))
    # a fraction phi of the fleet is unreliable on top of the hardware background: flaky and malicious
    # nodes in the 2:1 ratio and with the behaviour of the synthetic workload (Table 5)
    k = int(round(phi * N))
    if k:
        sel = rng.choice(N, k, replace=False)
        typ = np.where(rng.random(k) < 2 / 3, 1, 2)
        pop["types"][sel] = typ
        pop["p_fault"][sel] = np.array([0, 0.03, 0.2])[typ]
        pop["rho_lo"][sel] = np.array([0, 0.005, 0.002])[typ]
        pop["rho_hi"][sel] = np.array([0, 0.03, 0.01])[typ]
    return pop


if __name__ == "__main__":
    worst = max(abs(a - r[7]) for a, r in zip(AFR, TABLE))
    print("models", len(TABLE), "drives", int(COUNT.sum()), "drive days", int(DAYS.sum()), "failures", int(FAIL.sum()))
    print("recomputed fleet AFR %.3f%% (published %.2f%%); max |recomputed - published| per model = %.3f pp" %
          (100 * 365 * FAIL.sum() / DAYS.sum(), PUBLISHED_TOTAL["afr"], worst))
    print("AFR range %.2f%% - %.2f%%; drive-weighted CV %.2f" % (AFR.min(), AFR.max(),
          np.sqrt(np.average((AFR - np.average(AFR, weights=COUNT)) ** 2, weights=COUNT)) / np.average(AFR, weights=COUNT)))
    for r, a in zip(TABLE, AFR):
        if abs(a - r[7]) > 0.011: print("  mismatch", r[1], round(a, 3), r[7])
