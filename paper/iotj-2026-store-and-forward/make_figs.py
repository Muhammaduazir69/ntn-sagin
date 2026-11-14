#!/usr/bin/env python3
"""Result figures of the IoT-J manuscript (Figs. 2-5), drawn at final size."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as FS  # noqa: E402
from figstyle import BLUE, ORANGE, GREEN, GOLD, PURPLE, TEAL, RED, GREY, INK, INK2, COL, PAGE  # noqa: E402

FS.setup()
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "..", "figures")
FLEET = {"orbcomm": ("Orbcomm", BLUE, "o"), "kineis": ("Kinéis", ORANGE, "s"), "sateliot": ("Sateliot", GREEN, "^")}
GWLAB = {"k5": "5 sites", "aws11": "11 sites"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def fig_contact():
    R = json.load(open(os.path.join(OUT, "contact_check.json")))
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for f, (lab, c, m) in FLEET.items():
        rows = R[f]["rows"]
        lat = [r["lat"] for r in rows]
        ax.plot(lat, [100 * r["p_an"] for r in rows], color=c, ls="-", lw=1.0)
        ax.plot(lat, [100 * r["p_sim"] for r in rows], ls="none", marker=m, color=c, mfc="white", ms=4,
                label=lab)
    ax.set_xlabel("Site latitude (deg)")
    ax.set_ylabel("Contact probability per satellite (%)")
    ax.legend(loc="upper left", fontsize=6.5, title="lines: Lemma 1, markers: SGP4", title_fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig2_contact.pdf")


WEEKS = (0, 1, 2)


def an(f, gw, w):
    return json.load(open(os.path.join(OUT, f"an_{f}_{gw}_w{w}.json")))


def rv(f, gw, w):
    return json.load(open(os.path.join(OUT, "rev4", f"replay_{f}_{gw}_w{w}.json")))


def fw(f, gw, w):
    return json.load(open(os.path.join(OUT, "rev4", f"forward_{f}_{gw}_w{w}.json")))


def fig_delay():
    fig, axs = plt.subplots(1, 2, figsize=(PAGE, 2.5))
    ax = axs[0]
    for f, (lab, c, m) in FLEET.items():
        em, er, ef = [], [], []
        for gw in ("k5", "aws11"):
            for w in WEEKS:
                d = rv(f, gw, w)["devices"]
                sim = np.array([o["sim_D"] for o in d])
                em += list(100 * np.abs((np.array([o["model_A"] + o["model_H"] for o in d])) / sim - 1))
                er += list(100 * np.abs(np.array([o["replay_D"] for o in d]) / sim - 1))
                ef += list(100 * np.abs(np.array([o["fwd_D"] for o in fw(f, gw, w)["devices"]]) / sim - 1))
        for e, ls, tag in ((ef, ":", "forward"), (er, "-", "replay"), (em, "--", "model")):
            x = np.sort(e)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, ls=ls, lw=1.1, label=f"{lab}, {tag}")
    ax.set_xscale("log"); ax.set_xlim(0.004, 200); ax.set_ylim(0, 1)
    ax.set_xlabel("Absolute error of a buoy's mean delay (%)")
    ax.set_ylabel("Share of buoys and weeks")
    ax.legend(loc="upper left", fontsize=5.5, ncol=1)
    ax.set_title("(a) Per buoy, same element sets as the truth", fontsize=8)
    ax = axs[1]
    names = [("forward", "Forward, same sets", GOLD), ("replay", "Plan replay", GREEN), ("semi", "Renewal model", BLUE), ("half", "Half gap", ORANGE), ("cf", "Closed form", PURPLE)]
    x = np.arange(6)
    wd = 0.17
    labels = []
    for j, (k, lab, c) in enumerate(names):
        mean, lo, hi = [], [], []
        for f in FLEET:
            for gw in ("k5", "aws11"):
                e = []
                for w in WEEKS:
                    if k == "forward":
                        e.append(100 * (np.mean([o["fwd_D"] for o in fw(f, gw, w)["devices"]]) / np.mean([o["sim_D"] for o in rv(f, gw, w)["devices"]]) - 1))
                    elif k == "replay":
                        d = rv(f, gw, w)["devices"]
                        e.append(100 * (np.mean([o["replay_D"] for o in d]) / np.mean([o["sim_D"] for o in d]) - 1))
                    else:
                        a = an(f, gw, w)["devices"]
                        sim = np.mean([o["sim"] for o in a])
                        pred = np.mean([o[f"EA_{k}"] + o[f"EH_{k}"] for o in a])
                        e.append(100 * (pred / sim - 1))
                mean.append(np.mean(e)); lo.append(np.mean(e) - min(e)); hi.append(max(e) - np.mean(e))
                if j == 0:
                    labels.append(f"{FLEET[f][0][:4]}.\n{GWLAB[gw].split()[0]}")
        ax.bar(x + (j - 2) * wd, mean, wd, color=c, label=lab, edgecolor="white", lw=0.3,
               yerr=[lo, hi], error_kw=dict(lw=0.6, capsize=1.2, ecolor=INK))
    held = [json.load(open(os.path.join(OUT, "rev4", f"skill_held_{f}_{gw}.json")))["E"]["fwd"]["err"] for f in FLEET for gw in ("k5", "aws11")]
    ax.plot(x - 2 * wd, held, ls="none", marker="D", ms=2.8, color=RED, label="Forward, later sets", zorder=5)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Error of fleet-mean delay (%)")
    ax.set_ylim(-105, 50)
    ax.legend(loc="lower left", fontsize=6, ncol=3)
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) Fleet mean, three weeks (whiskers: range)", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_delay.pdf")


def fig_placebo():
    """Control: what each metric does when satellites of the later sets are displaced along track."""
    lv = [100, 500, 2000, 5000, 20000]
    fig, axs = plt.subplots(2, 1, figsize=(COL, 3.0), sharex=True)
    for f, (lab, c, m) in FLEET.items():
        for gw, ls, mfc in (("k5", "-", c), ("aws11", "--", "white")):
            d = json.load(open(os.path.join(OUT, "rev4", f"skill_placebo_{f}_{gw}.json")))["rows"]
            e = [np.mean([abs(r["err"]) for r in d if r["level"] == x]) for x in lv]
            h = [np.mean([r["H"]["median"] for r in d if r["level"] == x]) for x in lv]
            lab_ = f"{lab}, {GWLAB[gw]}"
            axs[0].plot(lv, e, ls=ls, marker=m, color=c, mfc=mfc, ms=3.2, lw=0.9, label=lab_)
            axs[1].plot(lv, np.maximum(h, 5), ls=ls, marker=m, color=c, mfc=mfc, ms=3.2, lw=0.9)
    H = [json.load(open(os.path.join(OUT, "rev4", f"skill_held_{f}_{gw}.json"))) for f in FLEET for gw in ("k5", "aws11")]
    r = [abs(h["E"]["replay"]["err"]) for h in H]
    axs[0].axhspan(min(r), max(r), color=GREY, alpha=0.25, lw=0)
    axs[0].text(120, np.sqrt(min(r) * max(r)) * 1.05, "replay, later sets", fontsize=6, color=INK2, va="center")
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[1].set_yscale("log")
    axs[0].set_ylim(0.03, 900)
    axs[0].set_ylabel("|Fleet-mean error| (%)")
    axs[0].legend(loc="upper left", fontsize=5.5, ncol=2)
    axs[0].set_title("(a) Fleet-mean delay", fontsize=8)
    lw = [h["timing"]["H"]["lastweek"]["median"] for h in H]
    axs[1].axhspan(min(lw), max(lw), color=GREY, alpha=0.25, lw=0)
    axs[1].text(120, 1200, "last week's plan", fontsize=6, color=INK2, va="center")
    axs[1].set_xlabel("Along-track displacement of each satellite (km)")
    axs[1].set_ylabel("Median delivery-time error (s)")
    axs[1].set_title("(b) Per-upload timing", fontsize=8)
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig7_placebo.pdf")


def fig_uplink():
    """Device backlog under FCFS over 28 days on three start dates, near the break: the quantity the text judges by."""
    R = [json.loads(l) for l in open(os.path.join(OUT, "rev4", "finite_rr.jsonl"))]
    R = [r for r in R if r["part"] == "window" and r["fl"] == "sateliot" and "seed" not in r]
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for start, c, lab in ((-14.0, BLUE, "16 Oct."), (0.0, ORANGE, "30 Oct."), (14.0, GREEN, "13 Nov.")):
        for bps, ls in ((2000, "-"), (2200, "--")):
            r = [x for x in R if x["start"] == start and x["up_bps"] == bps][0]
            b = [x for x in r["backlog_t"] if x[0] <= r["days"] - 3.0]
            ax.plot([x[0] for x in b], [x[1] / 1000 for x in b], ls=ls, color=c, lw=1.0, label=f"{bps} b/s, {lab}")
    ax.set_xlabel("Day of the run")
    ax.set_ylabel("Device backlog (thousand reports)")
    ax.legend(loc="upper left", fontsize=5.5, ncol=2)
    ax.set_ylim(0, 52)
    fig.tight_layout()
    save(fig, "fig3_uplink.pdf")


def fig_period():
    """Regime with contexts on board: rise of the age under the lookup timer against the ratio of mean return leg to update period."""
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    for f, (lab, c, m) in FLEET.items():
        for gw, ls, mfc in (("k5", "-", c), ("aws11", "--", "white")):
            W = np.mean([json.load(open(os.path.join(OUT, "rev4", f"timers_{f}_{gw}_w{w}.json")))["W_mean"] for w in WEEKS])
            D = [json.load(open(os.path.join(OUT, "rev4", f"tgate2_{f}_{gw}_w{w}.json")))["rows"] for w in WEEKS]
            x, y = [], []
            for tu in (15, 30, 60, 120, 240):
                g = lambda ru: np.mean([[r for r in d if r["rule"] == ru and r["ctx"] == "any" and r["tu"] == tu and r["disc"] == "fcfs"][0]["aoi_all_h"] for d in D])
                x.append(W / (tu * 60.0)); y.append(100 * (g("lookup") / g("none") - 1))
            ax.plot(x, y, ls=ls, marker=m, ms=3.2, lw=0.9, color=c, mfc=mfc, label=f"{lab}, {GWLAB[gw]}")
    ax.set_xscale("log")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel(r"Mean return leg over update period, $\mathbb{E}[W]/T_{\mathrm{u}}$")
    ax.set_ylabel("Rise of the age (%)")
    ax.legend(loc="upper left", fontsize=5.5, ncol=1)
    fig.tight_layout()
    save(fig, "fig4_period.pdf")


def load_ns3():
    return [r for r in (json.loads(l) for l in open(os.path.join(OUT, "ns3_sf4.jsonl"))) if not r.get("error")]


def _ci(x):
    x = np.asarray(x, float)
    return (1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def fig_ns3():
    R = [r for r in load_ns3() if r["part"] == "final"]
    R3 = [r for r in R if r["disc"] == "fcfs" and r["policy"] in ("bp", "sf1")]
    pols = [("bp", "fcfs", "Bent pipe", GREY, "s"), ("sf1", "fcfs", "S&F, first available", ORANGE, "o"),
            ("sfe", "fcfs", "S&F, earliest delivery", BLUE, "^")]
    scen = [(f, g) for f in FLEET for g in ("k5", "aws11")]
    fig, axs = plt.subplots(1, 2, figsize=(PAGE, 2.3))
    for j, (p, disc, lab, c, mk) in enumerate(pols):
        dsh, dci = [], []
        for f, g in scen:
            rr = [r for r in R if r["fleet"] == f and r["gw"] == g and r["policy"] == p and r["disc"] == disc]
            s_ = [100 * r["served_share"] for r in rr]
            dsh.append(np.mean(s_) if s_ else np.nan); dci.append(_ci(s_) if s_ else 0)
        axs[0].errorbar(np.arange(len(scen)) + (j - 1) * 0.16, dsh, yerr=dci, ls="none", marker=mk, ms=4, color=c,
                        capsize=1.5, lw=0.8, label=lab)
    # paired age on the buoys bent pipe serves, per replication
    for j, (p, lab, c, mk) in enumerate((("bp", "Bent pipe", GREY, "s"), ("sf1", "S&F, first available", ORANGE, "o"))):
        ym, ye = [], []
        for f, g in scen:
            vals = []
            for rep in range(5):
                bp = [r for r in R3 if r["fleet"] == f and r["gw"] == g and r["policy"] == "bp" and r["rep"] == rep]
                me = [r for r in R3 if r["fleet"] == f and r["gw"] == g and r["policy"] == p and r["rep"] == rep]
                if bp and me:
                    v = [a for a, sv in zip(me[0]["aoi_per_buoy"], bp[0]["served_per_buoy"]) if sv]
                    if v:
                        vals.append(np.mean(v))
            ym.append(np.mean(vals)); ye.append(_ci(vals))
        axs[1].errorbar(np.arange(len(scen)) + (j - 0.5) * 0.2, ym, yerr=ye, ls="none", marker=mk, ms=4, color=c, capsize=1.5, lw=0.8)
    labels = [f"{FLEET[f][0][:4]}.\n{GWLAB[g].split()[0]}" for f, g in scen]
    for ax in axs:
        ax.set_xticks(range(len(scen))); ax.set_xticklabels(labels, fontsize=6.5); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Buoys ever served (%)")
    axs[1].set_ylabel("Mean AoI, buoys bent pipe serves (h)")
    axs[1].set_yscale("log")
    axs[1].set_yticks([0.2, 0.5, 1, 2, 5, 10, 20]); axs[1].set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
    axs[1].minorticks_off()
    axs[0].set_title("(a) Coverage", fontsize=8)
    axs[1].set_title("(b) Freshness, paired by buoy", fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_ns3.pdf")


def fig_storage():
    scen = [(f, g) for f in FLEET for g in ("k5", "aws11")]
    FR = [json.loads(l) for l in open(os.path.join(OUT, "rev4", "finite.jsonl"))]
    fig, axs = plt.subplots(1, 2, figsize=(PAGE, 2.3))
    ax = axs[0]
    x = np.arange(len(scen))
    fc = [np.mean([an(f, g, w)["store"]["sim_fcfs_p99"] for w in WEEKS]) for f, g in scen]
    nw = [np.mean([an(f, g, w)["store"]["sim_new_p99"] for w in WEEKS]) for f, g in scen]
    bd = [np.mean([an(f, g, w)["store"]["bound_p99"] for w in WEEKS]) for f, g in scen]
    dv = [np.mean([r["occ_p99"] for r in FR if r["part"] == "tu" and r["fl"] == f and r["gw"] == g and r["tu"] == 15 and r["disc"] == "devonly"]) for f, g in scen]
    ax.plot(x - 0.15, fc, ls="none", marker="s", ms=4, color=ORANGE, label="FCFS")
    ax.plot(x, dv, ls="none", marker="^", ms=4, color=GREEN, label="Keep-newest on device only")
    ax.plot(x + 0.15, nw, ls="none", marker="o", ms=4, color=BLUE, label="Keep-newest, device and board")
    ax.plot(x + 0.15, bd, ls="none", marker="_", ms=9, mew=1.4, color=INK, label="Bound, previous week")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{FLEET[f][0][:4]}.\n{GWLAB[g].split()[0]}" for f, g in scen], fontsize=6.5)
    ax.set_yscale("log")
    ax.set_yticks([10, 30, 100, 300, 1000]); ax.set_yticklabels(["10", "30", "100", "300", "1000"])
    ax.set_ylabel("Updates held, 99th pct.")
    ax.set_ylim(10, 6000)
    ax.legend(loc="upper center", fontsize=6, ncol=2)
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) 15-minute updates", fontsize=8)
    ax = axs[1]
    for f, (lab, c, m) in FLEET.items():
        for g, hollow in (("k5", False), ("aws11", True)):
            eg = []
            for w in WEEKS:
                eg += [2 * o["EA_half"] / (1 - o["pU"]) for o in an(f, g, w)["devices"] if o["EA_half"] > 0]
            eg = np.mean(eg)
            xs, ys = [], []
            for tu in (5, 15, 30, 60, 120, 240):
                a = np.mean([r["occ_p99"] for r in FR if r["part"] == "tu" and r["fl"] == f and r["gw"] == g and r["tu"] == tu and r["disc"] == "fcfs"])
                b = np.mean([r["occ_p99"] for r in FR if r["part"] == "tu" and r["fl"] == f and r["gw"] == g and r["tu"] == tu and r["disc"] == "both"])
                xs.append(eg / (tu * 60.0)); ys.append(a / b)
            ax.plot(xs, ys, color=c, marker=m, ms=3.5, lw=0.9, mfc="white" if hollow else c,
                    label=f"{lab}, {GWLAB[g]}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_xlabel(r"$\mathbb{E}[G_U]/T_{\mathrm{u}}$ (mean gap over update period)")
    ax.set_ylabel("Storage cut (FCFS / keep-newest)")
    ax.legend(loc="upper left", fontsize=5.8)
    ax.set_title(r"(b) $T_{\mathrm{u}}$ from 5 min to 4 h", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig5_storage.pdf")


def fig_finite():
    """AoI of served buoys against the delivery budget per gateway contact (ns-3, five gateways), with the limit of eq. (stable)."""
    from run_ns3_sf3 import FEEDS
    R = [r for r in load_ns3() if r["part"] == "feed"]
    fig, axs = plt.subplots(1, 2, figsize=(COL, 2.3), sharey=True)
    M_ = dict(l[len("\\newcommand{\\"):].rstrip().rstrip("}").split("}{") for l in open(os.path.join(HERE, "..", "macros_p6.tex")) if l.startswith("\\newcommand"))
    RHO = {"orbcomm": float(M_["RhoNsOrb"]), "sateliot": float(M_["RhoNsSat"])}
    discs = [("fcfs", "FCFS", ORANGE, "s"), ("lcfs", "LCFS", GREEN, "^"), ("newest", "Keep-newest", BLUE, "o")]
    for ax, (fl, feeds) in zip(axs, FEEDS.items()):
        for d, lab, col, mk in discs:
            xs, ys, es, unstable = [], [], [], []
            for fb in feeds:
                rr = [r for r in R if r["fleet"] == fl and r["disc"] == d and r.get("x_feedBudget") == fb]
                if not rr:
                    continue
                a = [r["aoi_served_h"] for r in rr]
                xs.append(fb); ys.append(np.mean(a)); es.append(_ci(a))
                unstable.append(np.mean([r["aoi_second_h"] / r["aoi_first_h"] for r in rr]) > 1.2)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ms=3.5, lw=1.0, capsize=1.5, label=lab)
            for x, y, u in zip(xs, ys, unstable):
                if u:
                    ax.plot(x, y, ls="none", marker=mk, ms=6, mfc="white", mec=col, mew=1.0)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(feeds); ax.set_xticklabels([str(x) for x in feeds], fontsize=6.5)
        ax.set_title(FLEET[fl][0], fontsize=8)
        ax.axvline(RHO[fl], color=INK2, lw=0.7, ls=":")
        ax.set_xlabel("Deliveries per gateway contact")
    axs[0].set_ylabel("Mean AoI of served buoys (h)")
    axs[0].legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    save(fig, "fig6_finite.pdf")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["storage", "uplink", "period"]):
        globals()[f"fig_{w}"]()
