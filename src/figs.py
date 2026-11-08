import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = {}
for _s in ["E1", "E2", "E3", "E4", "E5"]:
    R.update(json.load(open(f"results/res_{_s}.json")))
C = {"RACE-Audit": "#2a78d6", "Uniform": "#eb6834", "Rep-Gated (tuned)": "#1baf7a",
     "U-HMAC": "#eb6834", "U-BLS": "#e87ba4", "B-BLS": "#6250d6", "B-BLS+loc": "#6250d6",
     "U-MHT": "#008300"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "Liberation Sans", "ps.fonttype": 42, "pdf.fonttype": 42, "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 300, "lines.linewidth": 2})
TYPE_NAMES = ["reliable", "flaky", "malicious", "on-off"]

import os
os.makedirs("figures/submission", exist_ok=True)
def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", bbox_inches="tight", facecolor="white")
    num = name.split("_")[0].replace("fig", "")
    fig.savefig(f"figures/submission/Fig{num}.eps", bbox_inches="tight", facecolor="white")
    fig.savefig(f"figures/submission/Fig{num}.tif", bbox_inches="tight", facecolor="white", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

# ---------------------------------------------------------------- Fig 1 architecture
fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 56)
def box(x, y, w, h, title, lines, fc="#eef4fc", ec="#2a78d6"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=8.6, weight="bold", color=INK)
    for k, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.4 - 3.1 * k, l, ha="center", va="top", fontsize=7.2, color=INK2)
def arrow(x0, y0, x1, y1, label="", dy=1.4, col=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=col))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", fontsize=6.8, color=INK)
box(1, 30, 30, 24, "Vendor / verifier V", ["(1) Bayesian-UCB risk r_i", "(2) water-filling c_i", "(3) tier quantisation",
                                             "(5) consensus-grouped check", "(6) differential localisation", "(7) repair + trust update"])
box(1, 2, 30, 20, "Secret state (never leaves V)", ["MAC key  k, alpha_1..alpha_s", "epoch key  for challenge seeds",
                                                     "Beta posteriors (f_i, h_i)", "value weights v_i"], fc="#f7f6f3", ec=INK2)
for k, (y, lab) in enumerate([(42.5, "edge node e_1  (tier 64)"), (28.5, "edge node e_2  (tier 512)"), (8, "edge node e_N  (tier 32)")]):
    box(64, y, 33, 11.5, lab, ["replica blocks m_j + tags sigma_j", "prefix proof (mu_i, sigma_i)"],
        fc="#fdf0ea" if k == 1 else "#f2faf6", ec="#eb6834" if k == 1 else "#1baf7a")
ax.text(80.5, 22, "...", ha="center", fontsize=12, color=INK2)
arrow(31.5, 50, 63.5, 50, "(4) broadcast seed_t + tier map  (32 B + N B)")
arrow(63.5, 46, 31.5, 44, "(mu, sigma) = 4.4 kB / node", dy=-3.6)
arrow(63.5, 34, 31.5, 40)
arrow(63.5, 13, 31.5, 36)
arrow(16, 22.5, 16, 29.5, col=INK2)
ax.text(47.5, 6, "nested challenge: e_1 checks first 64 indices\nof one keyed permutation, e_2 the first 512,\nso all nodes share the tau prefix sums",
        ha="center", fontsize=6.6, color=INK2, style="italic")
save(fig, "fig1_architecture")

# ---------------------------------------------------------------- Fig 2 exposure vs budget
E1 = R["E1_budget"]; Bs = sorted(map(int, E1))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
for name in ["Uniform", "Rep-Gated (tuned)", "RACE-Audit"]:
    for ax, key, lab in [(axs[0], "exposure", "Value-weighted exposure\n(undetected node-epochs per epoch)"),
                         (axs[1], "mttd", "Mean time to detection (epochs)")]:
        m = np.array([E1[str(b)][name][key][0] for b in Bs]); h = np.array([E1[str(b)][name][key][1] for b in Bs])
        ax.plot(Bs, m, marker="o", ms=4, color=C[name], label=name)
        import matplotlib.colors as mc
        rgb = np.array(mc.to_rgb(C[name])); blend = 0.15 * rgb + 0.85 * np.ones(3)
        ax.fill_between(Bs, m - h, m + h, color=blend, lw=0, zorder=0)
        ax.set_xscale("log", base=2); ax.set_xticks(Bs); ax.set_xticklabels(Bs)
        ax.set_xlabel("Budget B/N (blocks / node / epoch)"); ax.set_ylabel(lab)
axs[0].set_yscale("log")
axs[0].legend(fontsize=7.5)
save(fig, "fig2_budget")

# ---------------------------------------------------------------- Fig 3 allocation snapshot
S = R["snapshot"]; types = np.array(S["types"]); val = np.array(S["value"]); ma = np.array(S["mean_alloc"])
fig, ax = plt.subplots(figsize=(7.2, 2.8))
tc = ["#2a78d6", "#eda100", "#e34948"]
for t in range(3):
    sel = types == t
    ax.scatter(val[sel], ma[sel], s=22, color=tc[t], label=f"{TYPE_NAMES[t]} (n={sel.sum()})", edgecolor="white", lw=0.6)
ax.axhline(256, color=INK2, ls="--", lw=1); ax.text(val.min(), 272, "uniform allocation (256)", ha="left", fontsize=7, color=INK2)
ax.set_xscale("log"); ax.set_yscale("log", base=2)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 4])); ax.xaxis.set_major_formatter(FixedFormatter(["0.2", "0.5", "1", "2", "4"])); ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_locator(FixedLocator([32, 64, 128, 256, 512, 1024])); ax.yaxis.set_major_formatter(FixedFormatter(["32", "64", "128", "256", "512", "1024"])); ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel("Node value weight v_i (users served, normalised)"); ax.set_ylabel("Mean sampled blocks c_i\n(epochs 150-300)")
ax.legend(fontsize=7.5, loc="lower right")
save(fig, "fig3_allocation")

# ---------------------------------------------------------------- Fig 4 adversarial
E2, E3 = R["E2_onoff"], R["E3_whitewash"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
for name in ["Uniform", "Rep-Gated (tuned)", "RACE-Audit"]:
    tr = np.array(E2["128"][name]["trace"])
    k = 10; sm = np.convolve(tr, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(len(sm)) + k - 1, sm, color=C[name], label=name, lw=1.6)
ax.axvline(150, color=INK2, ls=":", lw=1); ax.text(148, 0.3, "attack starts ", fontsize=7, color=INK2, ha="right")
ax.set_xlabel("Epoch"); ax.set_ylabel("Exposure from on-off nodes\n(10-epoch moving mean)"); ax.legend(fontsize=7.2, loc="center left")
ax = axs[1]
names = ["Uniform", "Rep-Gated (tuned)", "RACE-Audit", "RACE w/o UCB (kappa=0)", "RACE w/o floor (c_min=0)", "RACE w/o forgetting (lambda=1)"]
short = ["Uniform", "Rep-Gated", "RACE", "w/o UCB", "w/o floor", "w/o forget"]
cols = [C["Uniform"], C["Rep-Gated (tuned)"], C["RACE-Audit"], "#9fc3ee", "#9fc3ee", "#9fc3ee"]
x = np.arange(len(names)); w = 0.38
for j, (B, hatch) in enumerate([("64", ""), ("256", "///")]):
    m = [E3[B][n]["exposure"][0] for n in names]; h = [E3[B][n]["exposure"][1] for n in names]
    ax.bar(x + (j - 0.5) * w, m, w * 0.92, yerr=h, color=cols, hatch=hatch, edgecolor="white", lw=0.5,
           error_kw=dict(lw=0.8, capsize=1.5, ecolor=INK2), label=f"B/N={B}")
ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha="right", fontsize=7.5)
ax.set_ylabel("Exposure under whitewashing")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#bbbbbb", ec="white", label="B/N=64"), Patch(fc="#bbbbbb", ec="white", hatch="///", label="B/N=256")], fontsize=7.2)
save(fig, "fig4_adversarial")

# ---------------------------------------------------------------- Fig 5 cost vs N
E5 = R["E5_cost"]; Ns = sorted(map(int, E5))
schemes = ["RACE-Audit", "U-HMAC", "U-MHT", "B-BLS+loc", "U-BLS"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
for s in schemes:
    for ax, k, lab, sc in [(axs[0], 0, "Verifier time per epoch (s)", 1), (axs[1], 2, "Slowest edge-node proof (ms)", 1e3),
                           (axs[2], 3, "Proof traffic per epoch (MB)", 1e-6)]:
        y = [E5[str(n)][s][k] * sc for n in Ns]
        ax.plot(Ns, y, marker="o", ms=3.5, color=C[s], label=s.replace("+loc", " (+loc.)"), lw=1.6,
                ls="-" if s == "RACE-Audit" else "--")
for ax in axs:
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Edge nodes N")
axs[0].set_ylabel("Verifier time per epoch (s)"); axs[1].set_ylabel("Slowest edge proof (ms)"); axs[2].set_ylabel("Proof traffic per epoch (MB)")
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7.2, bbox_to_anchor=(0.5, 1.07))
save(fig, "fig5_cost")

# ---------------------------------------------------------------- Fig 6 localisation
E6 = R["E6_localisation"]; ds = sorted(map(int, E6))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axs[0]
for key, lab, col in [("q_diff", "differential splitting (RACE)", C["RACE-Audit"]), ("q_plain", "plain binary splitting", C["Uniform"])]:
    m = [E6[str(d)][key][0] for d in ds]; h = [E6[str(d)][key][1] for d in ds]
    ax.errorbar(ds, m, yerr=h, marker="o", ms=4, color=col, label=lab, capsize=2, lw=1.6)
ax.plot(ds, [E6[str(d)]["bound"] for d in ds], color=INK2, ls=":", lw=1.2, label="bound d*ceil(log2 c)")
ax.set_xscale("log", base=2); ax.set_xticks(ds); ax.set_xticklabels(ds)
ax.set_xlabel("Corrupted blocks inside the sampled set (c = 512)"); ax.set_ylabel("Extra localisation queries"); ax.legend(fontsize=7.2)
ax = axs[1]
for key, lab, col in [("bytes_diff", "RACE differential", C["RACE-Audit"]), ("bytes_plain", "plain splitting", C["Uniform"]),
                      ("bytes_reveal", "reveal all sampled blocks", "#e87ba4"), ("bytes_mht", "Merkle proofs (U-MHT)", C["U-MHT"])]:
    ax.plot(ds, [E6[str(d)][key] / 1e6 for d in ds], marker="o", ms=3.5, color=col, label=lab, lw=1.6)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(ds); ax.set_xticklabels(ds)
ax.set_xlabel("Corrupted blocks inside the sampled set"); ax.set_ylabel("Localisation traffic (MB)"); ax.legend(fontsize=7)
save(fig, "fig6_localisation")

# ---------------------------------------------------------------- Fig 7 sensitivity
E4 = R["E4_sensitivity"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, k, lab, met, xl in [(axs[0], "kappa", "UCB width kappa", "exposure", "whitewashing, B/N=128"),
                            (axs[1], "lambda", "forgetting factor lambda", "exposure_attack", "on-off attack, B/N=128"),
                            (axs[2], "cmin", "randomness floor c_min", "exposure", "whitewashing, B/N=128")]:
    xs = sorted(E4[k], key=float)
    m = [E4[k][x][met][0] for x in xs]; h = [E4[k][x][met][1] for x in xs]
    ax.errorbar(range(len(xs)), m, yerr=h, marker="o", ms=4, color=C["RACE-Audit"], capsize=2, lw=1.6)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([x.rstrip("0").rstrip(".") if "." in x else x for x in xs])
    ax.set_xlabel(lab); ax.set_title(xl, fontsize=8, color=INK2)
axs[0].set_ylabel("Exposure")
save(fig, "fig7_sensitivity")
print("figures done")
