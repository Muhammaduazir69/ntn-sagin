#!/usr/bin/env python3
"""Write ../macros_p6.tex: every number the IoT-J manuscript cites, from out/an_*_w*.json and
out/sf_*_w*.npz (three Monte Carlo weeks, each predicted from the week before), out/contact_check.json
(Lemma 1), out/rev4/ (replay, forward propagation, what-if, slot simulator), out/ns3_sf4.jsonl (ns-3, first binary) and
out/ns3_sf5.jsonl (ns-3 rows with a shared uplink, rebuilt binary with per-report round robin; these replace the rows
with upBps > 0 of ns3_sf4, which stay on file as the record of the whole-backlog rule)."""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FLEETS = ["orbcomm", "kineis", "sateliot"]
GWS = ["k5", "aws11"]
TAG = {"orbcomm": "Orb", "kineis": "Kin", "sateliot": "Sat", "k5": "Five", "aws11": "Eleven"}
M = {}
RAWCUT, WKCUT = {}, {}


def f(x, n=1):
    return f"{x:.{n}f}"


# ---- delay model per scenario: three disjoint weeks, each predicted from the week before
WEEKS = (0, 1, 2)
errs = {"semi": [], "half": [], "cf": []}
q_errs, bound_ratio, pergap_v, pergap_n, p99err = [], [], 0, 0, []
for fl in FLEETS:
    for gw in GWS:
        t = TAG[fl] + TAG[gw]
        sims, preds, per_all, ea, eh, half_f = [], [], [], [], [], []
        qs, qsim = {"p95": [], "p99": [], "mean": []}, {"p95": [], "p99": [], "mean": []}
        st = {"bound": [], "new": [], "fcfs": []}
        h0_sim, h0_pred = [], []
        for w in WEEKS:
            A = json.load(open(os.path.join(OUT, f"an_{fl}_{gw}_w{w}.json")))
            a = A["devices"]
            sim = np.array([o["sim"] for o in a])
            sims.append(sim.mean())
            for k in errs:
                pred = np.array([o[f"EA_{k}"] + o[f"EH_{k}"] for o in a])
                errs[k].append(100 * (pred.mean() / sim.mean() - 1))
                if k == "semi":
                    preds.append(pred.mean())
                    per_all += list(np.abs(pred / sim - 1))
            ea.append(np.mean([o["EA_semi"] for o in a]))
            half_f += [o["EA_semi"] / o["EA_half"] for o in a if o["EA_half"] > 0]
            for q in ("mean", "p95", "p99"):
                qs[q].append(A["quant"][f"pred_{q}"] / 3600)
                qsim[q].append(A["quant"][f"sim_{q}"] / 3600)
            q_errs += [abs(A["quant"]["pred_p95"] / A["quant"]["sim_p95"] - 1),
                       abs(A["quant"]["pred_p99"] / A["quant"]["sim_p99"] - 1)]
            for k, kk in (("bound", "bound_p99"), ("new", "sim_new_p99"), ("fcfs", "sim_fcfs_p99")):
                st[k].append(A["store"][kk])
            bound_ratio.append(A["store"]["sim_new_p99"] / A["store"]["bound_p99"])
            pergap_v += A["store"]["pergap_violations"]
            pergap_n += A["store"]["pergap_n"]
            p99err += [abs(o["p99_pred"] / o["p99_sim"] - 1) for o in a if o.get("p99_sim", 0) > 0]
            z = np.load(os.path.join(OUT, f"sf_{fl}_{gw}_w{w}.npz"))
            okd = z["sf1_del"] < 10 ** 12
            h0_sim.append(np.mean(z["sf1_del"][okd] == z["sf1_up"][okd]))
            h0_pred.append(np.mean([o["q"] for o in a]))
        M[f"DSim{t}"] = f(np.mean(sims) / 3600, 2)
        M[f"DPred{t}"] = f(np.mean(preds) / 3600, 2)
        e = [100 * (p / s - 1) for p, s in zip(preds, sims)]
        M[f"DErrSemi{t}"] = f(np.mean(e), 0)
        M[f"DMedAbs{t}"] = f(100 * np.median(per_all), 0)
        M[f"DNinety{t}"] = f(100 * np.percentile(per_all, 90), 0)
        M[f"AccessShare{t}"] = f(100 * np.mean(ea) / np.mean(sims), 0)
        M[f"HalfFactor{t}"] = f(np.mean(half_f), 1)
        M[f"CvSq{t}"] = f(np.mean(half_f) - 1, 1)
        for q, nm in (("mean", "QMean"), ("p95", "QNine"), ("p99", "QNn")):
            M[f"{nm}{t}"] = f(np.mean(qsim[q]), 2)
            M[f"{nm}Pred{t}"] = f(np.mean(qs[q]), 2)
        M[f"NnOverMean{t}"] = f(np.mean(qsim["p99"]) / np.mean(qsim["mean"]), 1)
        M[f"StoreFcfs{t}"] = f(np.mean(st["fcfs"]), 0)
        M[f"StoreNew{t}"] = f(np.mean(st["new"]), 0)
        M[f"StoreBound{t}"] = f(np.mean(st["bound"]), 0)
        M[f"StoreCut{t}"] = f(np.mean(st["fcfs"]) / np.mean(st["new"]), 1)
        RAWCUT[t] = np.mean(st["fcfs"]) / np.mean(st["new"])
        WKCUT[t] = [a_ / b_ for a_, b_ in zip(st["fcfs"], st["new"])]
        M[f"HzeroSim{t}"] = f(100 * np.mean(h0_sim), 1)
        M[f"HzeroPred{t}"] = f(100 * np.mean(h0_pred), 1)
for fl in FLEETS:
    near, far = [], []
    for w in WEEKS:
        for o in json.load(open(os.path.join(OUT, f"an_{fl}_k5_w{w}.json")))["devices"]:
            e = abs((o["EA_semi"] + o["EH_semi"]) / o["sim"] - 1)
            (near if o["q"] > 0 else far).append(e)
    M[f"ErrNear{TAG[fl]}"], M[f"ErrFar{TAG[fl]}"] = f(100 * np.median(near), 0), f(100 * np.median(far), 0)
SW = json.load(open(os.path.join(OUT, "storage_w0.json")))
under = [100 * (1 - v["compound_p99"] / v["sim_p99"]) for v in SW.values()]
M["CompoundUnderLo"], M["CompoundUnderHi"] = f(min(under), 0), f(max(under), 0)
M["HalfFactorSat"] = M["HalfFactorSatFive"]
_HE = []
for fl in FLEETS:
    for gw in GWS:
        e_ = []
        for w in WEEKS:
            a = json.load(open(os.path.join(OUT, f"an_{fl}_{gw}_w{w}.json")))["devices"]
            e_.append(100 * (np.mean([o["EA_half"] + o["EH_half"] for o in a]) / np.mean([o["sim"] for o in a]) - 1))
        M[f"HalfErr{TAG[fl]}{TAG[gw]}"] = f(np.mean(e_), 0).replace("-", "$-$")
        _HE.append(abs(np.mean(e_)))
M["DelayErrMax"] = f(max(abs(e) for e in errs["semi"]), 1)
M["HalfTabLo"], M["HalfTabHi"] = f(min(_HE), 0), f(max(_HE), 0)
M["DelayErrLitMax"] = f(max(abs(e) for e in errs["half"]), 0)
M["DelayErrLitMin"] = f(min(abs(e) for e in errs["half"]), 0)
M["DelayErrCfMax"] = f(max(abs(e) for e in errs["cf"]), 0)
M["QuantErrMax"] = f(100 * max(q_errs), 0)
M["BoundRatioMax"] = f(max(bound_ratio), 2)
M["BoundRatioMin"] = f(min(bound_ratio), 2)
M["BoundShort"] = f(max(0.0, 100 * (max(bound_ratio) - 1)), 0)
M["PergapViol"], M["PergapN"] = str(pergap_v), f"{pergap_n:,}"
M["PNnBuoyMed"], M["PNnBuoyNinety"] = f(100 * np.median(p99err), 0), f(100 * np.percentile(p99err, 90), 0)
# from the unrounded ratios (formatting an already rounded value twice gave 22 for 22.55)
cuts = [RAWCUT[f"{TAG[fl]}{TAG[gw]}"] for fl in FLEETS for gw in GWS]
M["StorageCut"], M["StorageCutLo"] = f(max(cuts), 1), f(min(cuts), 1)
M["StoreCutWkLoSatFive"], M["StoreCutWkHiSatFive"] = f(min(WKCUT["SatFive"]), 1), f(max(WKCUT["SatFive"]), 1)
M["StoreCutWkLoSatEleven"], M["StoreCutWkHiSatEleven"] = f(min(WKCUT["SatEleven"]), 1), f(max(WKCUT["SatEleven"]), 1)
z = np.load(os.path.join(OUT, "sf_orbcomm_k5_w0.npz"))["d_SF1"]
z = z[np.isfinite(z)]
M["AboveMeanOrbFive"] = f(100 * np.mean(z > z.mean()), 0)

# ---- Lemma 1
C = json.load(open(os.path.join(OUT, "contact_check.json")))
for fl in FLEETS:
    rows = C[fl]["rows"]
    inc = min(min(i, 180 - i) for i in C[fl]["inc"])
    inner = [r for r in rows if r["lat"] <= inc - 15]
    M[f"PErr{TAG[fl]}"] = f(100 * max(abs(r["p_an"] / r["p_sim"] - 1) for r in inner), 0)
    M[f"PErrAll{TAG[fl]}"] = f(100 * max(abs(r["p_an"] / r["p_sim"] - 1) for r in rows if r["p_sim"] > 0), 0)
    M[f"EcErr{TAG[fl]}"] = f(100 * max(abs(r["ec_an"] / r["ec_sim"] - 1) for r in inner if r["ec_sim"] > 0), 0)

# ---- ns-3: two binaries that differ only in the branch taken when upBps > 0; the earlier ns3_sf*.jsonl are not read
R4_ = [r for r in (json.loads(l) for l in open(os.path.join(OUT, "ns3_sf4.jsonl"))) if not r.get("error")]
R5_ = [r for r in (json.loads(l) for l in open(os.path.join(OUT, "ns3_sf5.jsonl"))) if not r.get("error")]
for r in R4_ + R5_:
    r["part"] = {"final": "main"}.get(r["part"], r["part"])
REX = [r for r in R4_ if r.get("x_upBps")]   # whole-backlog rule, kept for comparison only
R = [r for r in R4_ if not r.get("x_upBps")] + R5_
_sha4, _sha5 = sorted({r["sha"] for r in R4_}), sorted({r["sha"] for r in R5_})
assert len(_sha4) == 1 and len(_sha5) == 1, "each ns-3 file must come from one binary"
assert {r["key"] for r in REX} == {r["key"] for r in R5_}, "round-robin rows must replace exactly the shared-uplink rows"
assert len({r["key"] for r in R}) == len(R) == 825, "ns-3 rows"
M["NsHash"], M["NsHashRr"] = _sha4[0][:12], _sha5[0][:12]
M["NsRunsRr"], M["NsRunsOld"] = str(len(R5_)), str(len(R) - len(R5_))
# the rebuilt binary must reproduce rows of the first one: unlimited rows, and the old rule behind --upRoundRobin=0
_k4 = {r["key"]: r for r in R4_}
_ck = [json.loads(l) for l in open(os.path.join(OUT, "ns3_sf5_check.jsonl"))]
assert all(r["sha"] == _sha5[0] and not r.get("error") for r in _ck), "check rows"
for r in _ck:
    a_ = json.loads(r["key"])
    a_[-1] = [x for x in a_[-1] if x[0] != "upRoundRobin"]
    o_ = _k4[json.dumps(a_)]
    for c_ in ("generated", "delivered", "aoi_h", "d_mean", "d_p99", "replaced_dev", "replaced_sat", "left_dev"):
        assert o_[c_] == r[c_], ("binary check", r["key"], c_)
M["NsCheckN"] = str(len(_ck))


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), (stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)


def sel(part="main", fl=None, gw=None, pol=None, disc=None, **x):
    out = []
    for r in R:
        if r["part"] != part or (fl and r["fleet"] != fl) or (gw and r["gw"] != gw) or (pol and r["policy"] != pol) \
                or (disc and r["disc"] != disc):
            continue
        if any(f"x_{k}" not in r for k in x):
            continue
        out.append(r)
    return out


POLS = (("bp", "fcfs", "Bp"), ("sf1", "fcfs", "Sfone"), ("sfe", "fcfs", "Sfe"), ("sf1", "newest", "SfoneNew"),
        ("sfe", "newest", "SfeNew"), ("sf1", "lcfs", "SfoneLcfs"))
for fl in FLEETS:
    for gw in GWS:
        t = TAG[fl] + TAG[gw]
        for pol, disc, pt in POLS:
            rr = sel("main", fl, gw, pol, disc)
            if not rr:
                continue
            M[f"Del{pt}{t}"] = f(100 * np.mean([r["delivered_share"] for r in rr]), 1)
            M[f"Served{pt}{t}"] = f(100 * np.mean([r["served_share"] for r in rr]), 0)
            dm = [r["d_mean"] for r in rr if r["d_mean"] == r["d_mean"]]
            if dm:
                m, h = ci(dm)
                M[f"D{pt}{t}"], M[f"D{pt}{t}Ci"] = f(m, 2), f(h, 2)
            m, h = ci([r["aoi_served_h"] for r in rr])
            M[f"Aoi{pt}{t}"], M[f"Aoi{pt}{t}Ci"] = f(m, 2), f(h, 2)
            M[f"ReplSat{pt}{t}"] = str(int(np.mean([r["replaced_sat"] for r in rr])))
            M[f"ReplDev{pt}{t}"] = str(int(np.mean([r["replaced_dev"] for r in rr])))
bps = [np.mean([r["served_share"] for r in sel("main", fl, gw, "bp", "fcfs")]) for fl in FLEETS for gw in GWS
       if sel("main", fl, gw, "bp", "fcfs")]
if bps:
    M["ServedBpMax"] = f(100 * max(bps), 0)
    M["ServedBpMin"] = f(100 * min(bps), 0)
sf = [r["delivered_share"] for r in sel("main", pol="sf1", disc="fcfs")]
M["SFDeliveredPct"] = f(100 * np.mean(sf), 0) if sf else "--"
# earliest delivery against first available: paired by replication, per scenario
gain, pvals = [], []
for fl in FLEETS:
    for gw in GWS:
        a = {r["rep"]: r["d_mean"] for r in sel("main", fl, gw, "sf1", "fcfs")}
        b = {r["rep"]: r["d_mean"] for r in sel("main", fl, gw, "sfe", "fcfs")}
        ks = sorted(k for k in a if k in b)
        if len(ks) > 1:
            x, y = np.array([a[k] for k in ks]), np.array([b[k] for k in ks])
            gain.append(100 * np.mean((x - y) / x))
            pvals.append(stats.ttest_rel(x, y).pvalue)
rel = []
for fl in FLEETS:
    for gw in GWS:
        a = {r["rep"]: r["d_mean"] for r in sel("main", fl, gw, "sf1", "fcfs")}
        b = {r["rep"]: r["d_mean"] for r in sel("main", fl, gw, "sfe", "fcfs")}
        rel += [100 * (a[k] - b[k]) / a[k] for k in a if k in b]
if rel:
    m, h = ci(rel)
    M["SFEDiff"], M["SFEDiffCi"] = f(m, 1), f(h, 1)
if gain:
    M["SFEGainPct"] = f(np.mean(gain), 0)
    M["SFEGainLo"], M["SFEGainHi"] = f(min(gain), 0), f(max(gain), 0)
    M["SFEPMin"] = f(min(pvals), 3)
    M["SFESig"] = str(sum(p < 0.05 for p in pvals))
# keep-newest against FCFS: AoI over served buoys, paired
aoi_diff = []
for fl in FLEETS:
    for gw in GWS:
        for pol in ("sf1", "sfe"):
            a = {r["rep"]: r["aoi_served_h"] for r in sel("main", fl, gw, pol, "fcfs")}
            b = {r["rep"]: r["aoi_served_h"] for r in sel("main", fl, gw, pol, "newest")}
            aoi_diff += [abs(a[k] - b[k]) / a[k] for k in a if k in b]
if aoi_diff:
    M["AoiDiffMaxPct"] = f(100 * max(aoi_diff), 2)
# on-board buffer (M8: the per-contact budgets of the earlier campaign are retired)
for fl in ("orbcomm", "sateliot"):
    for disc, dt in (("fcfs", "Fcfs"), ("lcfs", "Lcfs"), ("newest", "New")):
        rr = sel("buffer", fl, "k5", "sf1", disc)
        if rr:
            m, h = ci([r["aoi_served_h"] for r in rr])
            t = f"{TAG[fl]}Buffer{dt}"
            M[f"Aoi{t}"], M[f"Aoi{t}Ci"] = f(m, 2), f(h, 2)
            M[f"Del{t}"] = f(100 * np.mean([r["delivered_share"] for r in rr]), 1)
            M[f"Drop{t}"] = str(int(np.mean([r["dropped_sat"] for r in rr])))
M["NsRuns"] = str(len(R))
R3 = R
# paired bent pipe vs S&F on the buoys bent pipe serves
for fl in FLEETS:
    for gw in GWS:
        bp = {r["rep"]: r for r in R3 if r["part"] == "main" and r["disc"] == "fcfs" and r["fleet"] == fl and r["gw"] == gw and r["policy"] == "bp"}
        sf = {r["rep"]: r for r in R3 if r["part"] == "main" and r["disc"] == "fcfs" and r["fleet"] == fl and r["gw"] == gw and r["policy"] == "sf1"}
        x, y = [], []
        for k in bp:
            if k in sf:
                for ab, asf, s in zip(bp[k]["aoi_per_buoy"], sf[k]["aoi_per_buoy"], bp[k]["served_per_buoy"]):
                    if s:
                        x.append(ab); y.append(asf)
        if x:
            t = TAG[fl] + TAG[gw]
            M[f"PairBp{t}"], M[f"PairSf{t}"] = f(np.mean(x), 2), f(np.mean(y), 2)
            M[f"PairN{t}"] = str(len(x))
# load the satellites carry: updates delivered per non-empty gateway contact in the traffic window
# (Monte Carlo week 0, 150 buoys), scaled to the 60 buoys of the ns-3 runs
for fl in ("orbcomm", "sateliot"):
    z = np.load(os.path.join(OUT, f"sf_{fl}_k5_w0.npz"))
    held = (z["sf1_up"] < z["sf1_del"]) & (z["sf1_del"] < 10 ** 12)
    tt = z["sf1_del"][held] * 20.0
    win = (tt >= 86400) & (tt <= 4 * 86400)
    _, cnt = np.unique(z["sf1_sat"][held][win].astype(np.int64) * 10 ** 9 + z["sf1_del"][held][win], return_counts=True)
    M[f"FeedLoad{TAG[fl]}"] = f(cnt.mean() * 60 / 150, 0)
# delivery-budget sweep
from run_ns3_sf3 import FEEDS  # noqa: E402
for fl, feeds in FEEDS.items():
    for lv, fb in zip("ABCD", feeds):
        M[f"Feed{TAG[fl]}{lv}"] = str(fb)
        for disc, dt in (("fcfs", "Fcfs"), ("lcfs", "Lcfs"), ("newest", "New")):
            rr = [r for r in R3 if r["part"] == "feed" and r["fleet"] == fl and r["disc"] == disc and r.get("x_feedBudget") == fb]
            if not rr:
                continue
            t = f"{TAG[fl]}Feed{lv}{dt}"
            m, h = ci([r["aoi_served_h"] for r in rr])
            M[f"Aoi{t}"], M[f"Aoi{t}Ci"] = f(m, 2), f(h, 2)
            M[f"Grow{t}"] = f(np.mean([r["aoi_second_h"] / r["aoi_first_h"] for r in rr]), 2)
            M[f"Del{t}"] = f(100 * np.mean([r["delivered_share"] for r in rr]), 1)
# end-to-end lifetime
exp3 = {}
for fl in FLEETS:
    for gw in GWS:
        rr = [r for r in R3 if r["part"] == "lifetime" and r["fleet"] == fl and r["gw"] == gw]
        if rr:
            sh = [100 * (r["expired_sat"] + r["expired_dev"]) / r["generated"] for r in rr]
            m, h = ci(sh)
            exp3[(fl, gw)] = m
            M[f"ExpE{TAG[fl]}{TAG[gw]}"], M[f"ExpE{TAG[fl]}{TAG[gw]}Ci"] = f(m, 2), f(h, 2)
            M[f"ExpSatShare{TAG[fl]}{TAG[gw]}"] = f(100 * np.sum([r["expired_sat"] for r in rr]) /
                                                     max(1, np.sum([r["expired_sat"] + r["expired_dev"] for r in rr])), 0)
if exp3:
    M["ExpEMin"], M["ExpEMax"] = f(min(exp3.values()), 2), f(max(exp3.values()), 2)
    M["ExpEMean"] = f(np.mean(list(exp3.values())), 2)
M["NDevMC"] = "150"
from run_ns3_sf4 import BUF  # noqa: E402
M["BufUpdates"] = str(BUF)
M["NDevNs"] = "60"
M["NBuoys"] = str(sum(1 for l in open(os.path.join(HERE, "..", "..", "..", "ns-3-dev", "contrib", "ntn-sagin", "data",
                                                    "buoys.csv"))) - 1)

# paired tests with Holm's correction (m2)
def holm(pv):
    pv = np.asarray(pv)
    o = np.argsort(pv)
    adj = np.zeros(len(pv))
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(pv) - i) * pv[j])
        adj[j] = min(1.0, run)
    return adj


M["SFEHolmMax"] = f(max(holm(pvals)), 3)
M["SFEHolmSig"] = str(int(np.sum(holm(pvals) < 0.05)))
fp = []
for fl, feeds in FEEDS.items():
    for fb in feeds:
        for a_, b_ in (("fcfs", "lcfs"), ("lcfs", "newest")):
            ra = {r["rep"]: r["aoi_served_h"] for r in R3 if r["part"] == "feed" and r["fleet"] == fl and r["disc"] == a_ and r.get("x_feedBudget") == fb}
            rb = {r["rep"]: r["aoi_served_h"] for r in R3 if r["part"] == "feed" and r["fleet"] == fl and r["disc"] == b_ and r.get("x_feedBudget") == fb}
            ks = sorted(set(ra) & set(rb))
            fp.append(stats.ttest_rel([ra[k] for k in ks], [rb[k] for k in ks]).pvalue)
M["FeedHolmN"], M["FeedHolmSig"], M["FeedHolmMax"] = str(len(fp)), str(int(np.sum(holm(fp) < 0.05))), f(max(holm(fp)), 3)
# development week (m8): the headline on the two weeks first run after the model was fixed
e12 = []
for fl in FLEETS:
    for gw in GWS:
        for w in (1, 2):
            a = json.load(open(os.path.join(OUT, f"an_{fl}_{gw}_w{w}.json")))["devices"]
            e12.append(abs(100 * (np.mean([o["EA_semi"] + o["EH_semi"] for o in a]) / np.mean([o["sim"] for o in a]) - 1)))
M["DelayErrLaterWeeks"] = f(max(e12), 1)
# expiry per scenario range, AoI difference
M["AoiDiffLess"] = f(np.ceil(1000 * 100 * max(aoi_diff)) / 1000, 3)  # upper bound at three decimals

# ---- Nov 2026 revision: plan replay, Release 19 lookups, what-if, held-out sets, margin, sweeps (out/rev4)
RV = os.path.join(OUT, "rev4")
M["FieldMaxH"] = f(65535 / 3600, 1)
rep = {(fl, gw, w): json.load(open(os.path.join(RV, f"replay_{fl}_{gw}_w{w}.json"))) for fl in FLEETS for gw in GWS for w in WEEKS}
ferr = {k: [] for k in ("replay", "A_model", "A_replay", "H_model", "H_replay", "H_track", "p99_model", "p99_replay")}
wins, wins99 = 0, 0
med_m, med_r, n90_m, n90_r, p99m, p99r = [], [], [], [], [], []
for (fl, gw, w), r in rep.items():
    d = r["devices"]
    g = lambda k: np.array([o[k] for o in d])
    sD, sA, sH = g("sim_D"), g("sim_A"), g("sim_H")
    mD, rD = g("model_A") + g("model_H"), g("replay_D")
    e = lambda a, b: 100 * (a.mean() / b.mean() - 1)
    ferr["replay"].append(e(rD, sD))
    ferr["A_model"].append(e(g("model_A"), sA)); ferr["A_replay"].append(e(g("replay_A"), sA))
    ferr["H_model"].append(e(g("model_H"), sH)); ferr["H_replay"].append(e(g("replay_H"), sH))
    ferr["H_track"].append(e(g("track_H"), sH))
    q = r["quant"]
    ferr["p99_model"].append(100 * (q["pred_p99"] / q["sim_p99"] - 1))
    ferr["p99_replay"].append(100 * (r["pool"]["replay_p99"] / q["sim_p99"] - 1))
    a, b = np.median(np.abs(mD / sD - 1)), np.median(np.abs(rD / sD - 1))
    wins += int(b < a)
    a9, b9 = np.median(np.abs(g("model_p99") / g("sim_p99") - 1)), np.median(np.abs(g("replay_p99") / g("sim_p99") - 1))
    wins99 += int(b9 < a9)
    med_m.append(100 * a); med_r.append(100 * b); p99m.append(100 * a9); p99r.append(100 * b9)
    n90_m.append(100 * np.percentile(np.abs(mD / sD - 1), 90)); n90_r.append(100 * np.percentile(np.abs(rD / sD - 1), 90))
amax = lambda x: max(abs(v) for v in x)
M["ReplayErrMax"] = f(amax(ferr["replay"]), 1)
M["RivalWins"], M["RivalWinsPnn"], M["RivalN"] = str(wins), str(wins99), str(len(rep))
M["ModelBuoyMedLo"], M["ModelBuoyMedHi"] = f(min(med_m), 0), f(max(med_m), 0)
M["ReplayBuoyMedLo"], M["ReplayBuoyMedHi"] = f(min(med_r), 0), f(max(med_r), 0)
M["ModelBuoyNinetyHi"], M["ReplayBuoyNinetyHi"] = f(max(n90_m), 0), f(max(n90_r), 0)
M["ModelBuoyPnnHi"], M["ReplayBuoyPnnHi"] = f(max(p99m), 0), f(max(p99r), 0)
M["AccessErrModel"], M["AccessErrReplay"] = f(amax(ferr["A_model"]), 1), f(amax(ferr["A_replay"]), 1)
M["HoldErrModel"], M["HoldErrReplay"], M["HoldErrTrack"] = f(amax(ferr["H_model"]), 0), f(amax(ferr["H_replay"]), 1), f(amax(ferr["H_track"]), 0)
M["PnnErrModel"], M["PnnErrReplay"] = f(amax(ferr["p99_model"]), 0), f(amax(ferr["p99_replay"]), 0)
for fl in FLEETS:
    for gw in GWS:
        t = TAG[fl] + TAG[gw]
        rr = [rep[(fl, gw, w)] for w in WEEKS]
        M[f"QMeanRep{t}"] = f(np.mean([x["pool"]["replay_mean"] for x in rr]) / 3600, 2)
        M[f"QNnRep{t}"] = f(np.mean([x["pool"]["replay_p99"] for x in rr]) / 3600, 2)
        M[f"HoldSim{t}"] = f(np.mean([np.mean([o["sim_H"] for o in x["devices"]]) for x in rr]) / 3600, 2)
M["TimeMoments"] = f(1e3 * np.mean([r["time"]["moments_s_per_dev"] for r in rep.values()]), 1)
M["TimeReplay"] = f(1e3 * np.mean([r["time"]["replay_s_per_dev"] for r in rep.values()]), 1)
# forward propagation on the test week's plan: same element sets, so its error is phase noise (sf_forward.py)
FW = {(fl, gw, w): json.load(open(os.path.join(RV, f"forward_{fl}_{gw}_w{w}.json"))) for fl in FLEETS for gw in GWS for w in WEEKS}
fe, fm, f99, fA = [], [], [], []
fwins_r, fwins_m = 0, 0
for k_, x_ in FW.items():
    # list order, not coordinates: co-located stations must stay apart
    assert all((a["lat"], a["lon"]) == (b["lat"], b["lon"]) for a, b in zip(rep[k_]["devices"], x_["devices"])), k_
    d_ = list(zip(rep[k_]["devices"], x_["devices"]))
    assert len(d_) == len(rep[k_]["devices"]) == len(x_["devices"]), k_
    s_ = np.array([a["sim_D"] for a, _ in d_]); v_ = np.array([b["fwd_D"] for _, b in d_])
    rd_ = np.array([a["replay_D"] for a, _ in d_]); md_ = np.array([a["model_A"] + a["model_H"] for a, _ in d_])
    fe.append(100 * (v_.mean() / s_.mean() - 1))
    fm.append(100 * np.median(np.abs(v_ / s_ - 1)))
    fwins_r += int(np.median(np.abs(v_ / s_ - 1)) < np.median(np.abs(rd_ / s_ - 1)))
    fwins_m += int(np.median(np.abs(v_ / s_ - 1)) < np.median(np.abs(md_ / s_ - 1)))
    f99.append(100 * (x_["pool"]["p99"] / rep[k_]["quant"]["sim_p99"] - 1))
M["FwdErrMax"] = f(amax(fe), 2)
M["FwdBuoyMedLo"], M["FwdBuoyMedHi"] = f(min(fm), 1), f(max(fm), 1)
M["FwdWinsReplay"], M["FwdWinsModel"] = str(fwins_r), str(fwins_m)
M["FwdPnnMax"] = f(amax(f99), 1)
M["FwdTimeMs"] = f(1e3 * np.mean([x_["s_per_dev"] for x_ in FW.values()]), 1)
for fl in FLEETS:
    for gw in GWS:
        t = TAG[fl] + TAG[gw]
        rr = [FW[(fl, gw, w)] for w in WEEKS]
        M[f"QMeanFwd{t}"] = f(np.mean([x["pool"]["mean"] for x in rr]) / 3600, 2)
        M[f"QNnFwd{t}"] = f(np.mean([x["pool"]["p99"] for x in rr]) / 3600, 2)
# replay against the model on the fleet mean: two worst cases hide a closer race (R2-m3)
fm_err, fr_err, scen_ = [], [], []
for (fl, gw, w), r in rep.items():
    d = r["devices"]
    s_ = np.mean([o["sim_D"] for o in d])
    fr_err.append(abs(100 * (np.mean([o["replay_D"] for o in d]) / s_ - 1)))
    fm_err.append(abs(100 * (np.mean([o["model_A"] + o["model_H"] for o in d]) / s_ - 1)))
    scen_.append(fl)
fm_err, fr_err, scen_ = np.array(fm_err), np.array(fr_err), np.array(scen_)
M["FleetWinsReplay"] = str(int(np.sum(fr_err < fm_err)))
M["FleetMedReplay"], M["FleetMedModel"] = f(np.median(fr_err), 1), f(np.median(fm_err), 1)
M["FleetWinsReplaySat"] = str(int(np.sum((fr_err < fm_err)[scen_ == "sateliot"])))
M["FleetNSat"] = str(int(np.sum(scen_ == "sateliot")))
# held-out element sets
H = [json.load(open(os.path.join(RV, f"heldout_{fl}_{gw}.json"))) for fl in FLEETS for gw in GWS]
M["HeldHalfLo"], M["HeldHalfHi"] = f(min(abs(h["err"]["half"]) for h in H), 0), f(max(abs(h["err"]["half"]) for h in H), 0)
M["HeldLookupPnn"] = f(max(h["lookup_err_s"]["p99"] for h in H), 0)
for h in H[::2]:
    M[f"Drift{TAG[h['fleet']]}"] = f(h["drift_km_mean"], 0)
    M[f"DriftMax{TAG[h['fleet']]}"] = f(h["drift_km_max"], 0)
# later-issued sets: ten phase draws per buoy, 95 % intervals over buoys, floor alongside (sf_skill.py, part held)
SK = {(fl, gw): json.load(open(os.path.join(RV, f"skill_held_{fl}_{gw}.json"))) for fl in FLEETS for gw in GWS}
_sk = list(SK.values())
_ciabs = lambda nm: max(max(abs(c) for c in h["E"][nm]["ci"]) for h in _sk)
M["HeldDraws"] = str(_sk[0]["draws"])
M["HeldFwdMax"] = f(amax([h["E"]["fwd"]["err"] for h in _sk]), 2)
M["HeldFwdCiMax"], M["HeldFloorCiMax"] = f(_ciabs("fwd"), 2), f(_ciabs("floor"), 2)
M["HeldFloorMax"] = f(amax([h["E"]["floor"]["err"] for h in _sk]), 2)
M["HeldFwdInside"] = str(sum(h["E"]["fwd"]["ci"][0] <= 0 <= h["E"]["fwd"]["ci"][1] for h in _sk))
M["HeldFwdN"] = str(len(_sk))
M["HeldReplayLo"], M["HeldReplayMax"] = f(min(abs(h["E"]["replay"]["err"]) for h in _sk), 2), f(amax([h["E"]["replay"]["err"] for h in _sk]), 2)
M["HeldModelMax"] = f(amax([h["E"]["model"]["err"] for h in _sk]), 2)
M["HeldGapLoCi"] = f(min(h["gap_fleet"]["ci"][0] for h in _sk), 2)
M["HeldGapResolved"] = str(sum(h["gap_fleet"]["ci"][0] > 0 for h in _sk))
M["HeldGapMedLoCi"] = f(min(h["gap_med"]["ci"][0] for h in _sk), 2)
M["HeldFwdMedLo"], M["HeldFwdMedHi"] = f(min(h["E"]["fwd"]["med"] for h in _sk), 2), f(max(h["E"]["fwd"]["med"] for h in _sk), 2)
M["HeldFloorMedLo"], M["HeldFloorMedHi"] = f(min(h["E"]["floor"]["med"] for h in _sk), 2), f(max(h["E"]["floor"]["med"] for h in _sk), 2)
M["HeldReplayMedLo"], M["HeldReplayMedHi"] = f(min(h["E"]["replay"]["med"] for h in _sk), 1), f(max(h["E"]["replay"]["med"] for h in _sk), 1)
M["HeldModelMedLo"], M["HeldModelMedHi"] = f(min(h["E"]["model"]["med"] for h in _sk), 1), f(max(h["E"]["model"]["med"] for h in _sk), 1)
M["HeldFwdPnn"] = f(amax([100 * (h["p99"]["fwd"] / h["p99"]["ref"] - 1) for h in _sk]), 1)
M["HeldPnnReplay"] = f(amax([100 * (h["p99"]["replay"] / h["p99"]["ref"] - 1) for h in _sk]), 1)
M["HeldPnnModel"] = f(amax([100 * (h["model_p99"] / h["p99"]["ref"] - 1) for h in _sk]), 0)
for (fl, gw), h in SK.items():
    t = TAG[fl] + TAG[gw]
    for nm, tn in (("floor", "Floor"), ("fwd", "Fwd"), ("replay", "Rep"), ("model", "Mod")):
        dec = 2 if nm in ("floor", "fwd") else 1
        lo_, hi_ = h["E"][nm]["ci"]
        v_ = f(h["E"][nm]["err"], dec)
        M[f"Sk{tn}{t}"] = (v_.lstrip("-") if float(v_) == 0 else v_).replace("-", "$-$")
        M[f"Sk{tn}Hw{t}"] = f((hi_ - lo_) / 2, dec)
    M[f"SkHfwd{t}"] = f(h["timing"]["H"]["fwd"]["p99"], 0)
    M[f"SkHlw{t}"] = f(h["timing"]["H"]["lastweek"]["median"] / 60, 1)
    M[f"SkHlwP{t}"] = f(h["timing"]["H"]["lastweek"]["p99"] / 60, 0)
    M[f"SkHres{t}"] = f(h["timing"]["H"]["residual"]["median"] / 60, 0)
_tm = lambda leg, pr, st: [h["timing"][leg][pr][st] for h in _sk]
M["TimeHFwdMedMax"] = f(max(_tm("H", "fwd", "median")), 0)
M["TimeHFwdPnnMax"], M["TimeWFwdPnnMax"] = f(max(_tm("H", "fwd", "p99")), 0), f(max(_tm("W", "fwd", "p99")), 0)
M["TimeWFwdMedMax"] = f(max(_tm("W", "fwd", "median")), 0)
M["TimeHLwMedLo"], M["TimeHLwMedHi"] = f(min(_tm("H", "lastweek", "median")) / 60, 1), f(max(_tm("H", "lastweek", "median")) / 60, 0)
M["TimeHLwPnnLo"], M["TimeHLwPnnHi"] = f(min(_tm("H", "lastweek", "p99")) / 60, 0), f(max(_tm("H", "lastweek", "p99")) / 60, 0)
M["TimeWLwMedLo"], M["TimeWLwMedHi"] = f(min(_tm("W", "lastweek", "median")) / 60, 1), f(max(_tm("W", "lastweek", "median")) / 60, 0)
M["TimeWLwPnnHi"] = f(max(_tm("W", "lastweek", "p99")) / 3600, 0)
M["TimeHResMedLo"], M["TimeHResMedHi"] = f(min(_tm("H", "residual", "median")) / 60, 0), f(max(_tm("H", "residual", "median")) / 60, 0)
M["HeldWShareLo"] = f(100 * min(h["w_within_60"] for h in _sk), 0)
M["HeldWShareHi"] = f(100 * max(h["w_within_60"] for h in _sk), 0)
# sets aged further: forward from 30 Oct against later sets over weeks 1 and 2, both extrapolated
_ag = [json.load(open(os.path.join(RV, f"forward_held_{fl}_{gw}_w{w}.json"))) for fl in FLEETS for gw in GWS for w in (1, 2)]
M["AgreeFwdMax"] = f(amax([h["err"]["fwd"] for h in _ag]), 2)
M["AgreeMedHi"] = f(max(h["med"]["fwd"] for h in _ag), 1)
M["AgreeWErrPnnMax"] = f(max(h["w_err_s"]["p99"] for h in _ag) / 60, 0)
M["AgreeHErrPnnMax"] = f(max(h["h_err_s"]["p99"] for h in _ag) / 60, 0)
# placebo control: same later sets, each satellite displaced along track (sf_skill.py, part placebo)
PL = {(fl, gw): json.load(open(os.path.join(RV, f"skill_placebo_{fl}_{gw}.json"))) for fl in FLEETS for gw in GWS}
_LV = ((100, "Hund"), (500, "FiveH"), (2000, "TwoK"), (5000, "FiveK"), (20000, "TwentyK"), ("phase", "Phase"), ("node", "Node"))
for lv, nm in _LV:
    rr = {k: [r for r in v["rows"] if r["level"] == lv] for k, v in PL.items()}
    assert all(len(x) == 3 for x in rr.values()), lv
    sp = [abs(r["err"]) for k, x in rr.items() if k[0] != "kineis" for r in x]
    ki = [abs(r["err"]) for k, x in rr.items() if k[0] == "kineis" for r in x]
    sa = [abs(r["err"]) for k, x in rr.items() if k[0] == "sateliot" for r in x]
    orb = [abs(r["err"]) for k, x in rr.items() if k[0] == "orbcomm" for r in x]
    M[f"PlSpMax{nm}"], M[f"PlKinMax{nm}"] = f(max(sp), 1), f(max(ki), 1)
    M[f"PlSpMed{nm}"], M[f"PlKinMed{nm}"] = f(np.median(sp), 1), f(np.median(ki), 1)
    M[f"PlSatMax{nm}"], M[f"PlOrbMax{nm}"] = f(max(sa), 1), f(max(orb), 1)
    h_ = [r["H"]["median"] for x in rr.values() for r in x]
    M[f"PlHMed{nm}"], M[f"PlHMedLo{nm}"], M[f"PlHMedHi{nm}"] = f(np.median(h_), 0), f(min(h_), 0), f(max(h_), 0)
    M[f"PlHPnn{nm}"] = f(np.median([r["H"]["p99"] for x in rr.values() for r in x]) / 60, 0)
_fl = [abs(v["floor"]["err"]) for v in PL.values()]
M["PlFloorMax"] = f(max(_fl), 2)
M["PlNode"] = str(sum(len(v["rows"]) for v in PL.values()))
_ns = [r["err"] for (fl_, g_), v in PL.items() if fl_ == "sateliot" for r in v["rows"] if r["level"] == "node"]
M["PlNodeSatLo"], M["PlNodeSatHi"] = f(min(abs(x) for x in _ns), 0), f(max(abs(x) for x in _ns), 0)
# wait-timer rules, per upload (sf_timer_cost.py)
TC = {(fl, gw): [json.load(open(os.path.join(RV, f"tcost_{fl}_{gw}_w{w}.json"))) for w in WEEKS] for fl in FLEETS for gw in GWS}
_tcm = lambda v, ru, k: float(np.mean([x[ru][k] for x in v]))
for (fl, gw), v in TC.items():
    t = TAG[fl] + TAG[gw]
    for ru, nm in (("lookup", "lookup"), ("none", "none"), ("fallback", "fallback"), ("lastp75", "lastq"), ("lastp95", "lastp"), ("lag", "lag")):
        M[f"TcFirst{nm}{t}"] = f(100 * _tcm(v, ru, "first"), 0)
        M[f"TcWait{nm}{t}"] = f(_tcm(v, ru, "wait_h"), 2)
_five = {k: v for k, v in TC.items() if k[1] == "k5"}
_rg = lambda D, ru, k, sc=1.0: [sc * _tcm(v, ru, k) for v in D.values()]
M["LastQExcessLo"], M["LastQExcessHi"] = f(min(_rg(TC, "lastp75", "excess_h")), 1), f(max(_rg(TC, "lastp75", "excess_h")), 1)
M["LastQFirstLo"], M["LastQFirstHi"] = f(min(_rg(TC, "lastp75", "first", 100)), 0), f(max(_rg(TC, "lastp75", "first", 100)), 0)
_wk = lambda ru, k, sc=1.0: max(sc * (max(x[ru][k] for x in v) - min(x[ru][k] for x in v)) for v in TC.values())
M["SpreadFirst"] = f(max(_wk(r_, "first", 100) for r_ in ("none", "fallback", "lastp75", "lastp95", "lag")), 0)
M["SpreadWait"] = f(max(_wk(r_, "wait_h") for r_ in ("lookup", "none", "fallback", "lastp75", "lastp95", "lag")), 1)
M["FallbackLo"], M["FallbackHi"] = f(min(_rg(_five, "fallback", "first", 100)), 0), f(max(_rg(_five, "fallback", "first", 100)), 0)
M["FallFailedHi"] = f(max(_rg(_five, "fallback", "failed")), 1)
M["FallExcessLo"], M["FallExcessHi"] = f(min(_rg(_five, "fallback", "excess_h")), 1), f(max(_rg(_five, "fallback", "excess_h")), 1)
M["LastExcessLo"], M["LastExcessHi"] = f(min(_rg(TC, "lastp95", "excess_h")), 1), f(max(_rg(TC, "lastp95", "excess_h")), 0)
M["PNineOkLo"], M["PNineOkHi"] = f(min(_rg(TC, "lastp95", "first", 100)), 0), f(max(_rg(TC, "lastp95", "first", 100)), 0)
M["LagFirstLo"], M["LagFirstHi"] = f(min(_rg(TC, "lag", "first", 100)), 0), f(max(_rg(TC, "lag", "first", 100)), 0)
M["LagExcessLo"], M["LagExcessHi"] = f(min(_rg(TC, "lag", "excess_h")), 1), f(max(_rg(TC, "lag", "excess_h")), 1)
TH = {(fl, gw): json.load(open(os.path.join(RV, f"tcost_held_{fl}_{gw}.json"))) for fl in FLEETS for gw in GWS}
M["OlderFirstLo"], M["OlderFirstHi"] = f(min(100 * h["older"]["first"] for h in TH.values()), 1), f(max(100 * h["older"]["first"] for h in TH.values()), 1)
M["OlderExcessMin"] = f(max(60 * h["older"]["excess_h"] for h in TH.values()), 1)
M["LagHMedHi"] = f(max(h["H"]["lag"]["median"] for h in TH.values()), 0)
M["LagHPnnLo"], M["LagHPnnHi"] = f(min(h["H"]["lag"]["p99"] for h in TH.values()) / 60, 0), f(max(h["H"]["lag"]["p99"] for h in TH.values()) / 60, 0)
_bey = set()
for k_, h in TH.items():
    _bey |= {(k_[0], i) for i, p_ in h["per_sat"].items() if p_["p99"] > 20.0}
_nm = {}
for fl in FLEETS:
    L_ = [x.rstrip() for x in open(os.path.join(HERE, "data_raw", f"{fl}.tle")) if x.strip()]
    for i in range(0, len(L_) - 2, 3):
        _nm[L_[i + 1][2:7]] = L_[i].strip()
M["NSatsAll"] = str(len(_nm))
M["NSatsBeyond"] = str(len(_bey))
_oh = max([h for k_, h in TH.items() if k_[0] == "orbcomm"], key=lambda h: h["H"]["older"]["p99"])
_oid = _oh["worst"]["id"]
M["SetOutlierName"] = _nm[_oid].replace("ORBCOMM ", "Orbcomm ")
M["SetOutlierShare"] = f(100 * _oh["worst"]["share"], 0)
# sf_timer_cost.py stores in "worst" the statistics OF the outlier; the rest come from per_sat
M["OutlierExclPnn"] = f(max(p_["p99"] for k_, h in TH.items() for i, p_ in h["per_sat"].items() if i != _oid), 0)
M["OlderHPnnMax"] = f(max(h["H"]["older"]["p99"] for h in TH.values()), 0)
M["OutlierHMed"] = f(_oh["worst"]["median"], 0)
_ow = [TH[("orbcomm", g_)]["per_sat"][_oid] for g_ in GWS]
M["OutlierOwnLo"], M["OutlierOwnHi"] = f(min(x["p99"] for x in _ow), 0), f(max(x["p99"] for x in _ow), 0)
M["OutlierOwnMax"] = f(max(TH[("orbcomm", g_)]["worst"]["max"] for g_ in GWS), 0)
# hypothetical rejects at the start of a contact: the denominator the 90-minute example governs (sf_reject.py)
RJ = {(fl, gw): [json.load(open(os.path.join(RV, f"reject_{fl}_{gw}_w{w}.json"))) for w in WEEKS] for fl in FLEETS for gw in GWS}
_rj = {k: 100 * np.mean([x["over90"] for x in v]) for k, v in RJ.items()}
M["RejLo"], M["RejHi"] = f(min(v for k, v in _rj.items() if k[1] == "k5"), 0), f(max(v for k, v in _rj.items() if k[1] == "k5"), 0)
M["RejElevenLo"], M["RejElevenHi"] = f(min(v for k, v in _rj.items() if k[1] != "k5"), 0), f(max(v for k, v in _rj.items() if k[1] != "k5"), 0)
M["RejSpread"] = f(max(100 * (max(x["over90"] for x in v) - min(x["over90"] for x in v)) for v in RJ.values()), 0)
# the timer and custody run together (sf_timer_gate.py), slot simulator only: two context regimes, periods from 15 to 240 min
TG = {(fl, gw): [json.load(open(os.path.join(RV, f"tgate2_{fl}_{gw}_w{w}.json")))["rows"] for w in WEEKS] for fl in FLEETS for gw in GWS}
def _tg(d_, ru, ctx="any", tu=15, di="fcfs"):
    return [[x for x in r_ if x["rule"] == ru and x["ctx"] == ctx and x["tu"] == tu and x["disc"] == di][0] for r_ in d_]
_gm = lambda d_, k, *a, **kw: float(np.mean([r[k] for r in _tg(d_, *a, **kw)]))
_br = np.random.default_rng(11)
_rise = {15: {}, 30: {}, 60: {}, 120: {}, 240: {}}
_riseq, _risep, _risef, _nas, _kdiff, _kdel, _ksp, _fail, _agegap, _unch = {}, {}, {}, {}, [], {}, [], {}, {}, []
for k_, d_ in TG.items():
    t = TAG[k_[0]] + TAG[k_[1]]
    for ru, nm in (("none", "None"), ("lookup", "Look"), ("fix30", "Fix"), ("p75", "Q"), ("p95", "P")):
        M[f"GateAge{nm}{t}"] = f(_gm(d_, "aoi_all_h", ru), 2)
        _ksp.append(max(r["aoi_all_h"] for r in _tg(d_, ru)) - min(r["aoi_all_h"] for r in _tg(d_, ru)))
        M[f"RetAge{nm}{t}"] = f(_gm(d_, "aoi_all_h", ru, "return"), 2)
        for ctx in ("any", "return"):
            _kdiff += [abs(a["aoi_all_h"] / b["aoi_all_h"] - 1) * 100 for a, b in zip(_tg(d_, ru, ctx, di="both"), _tg(d_, ru, ctx))]
        M[f"RetFail{nm}{t}"] = f(np.sum([r["n_failed"] for r in _tg(d_, ru, "return")]) / np.sum([r["n_nas"] for r in _tg(d_, ru, "return")]), 2)
    for tu in _rise:
        _rise[tu][k_] = 100 * (_gm(d_, "aoi_all_h", "lookup", tu=tu) / _gm(d_, "aoi_all_h", "none", tu=tu) - 1)
        M[f"GateRise{tu}{t}".replace("15", "Fifteen").replace("30", "Thirty").replace("60", "Sixty").replace("120", "Twoh").replace("240", "Fourh")] = f(_rise[tu][k_] + 0.0, 0 if tu < 240 else 1).replace("-0.0", "0.0")
    a0 = np.concatenate([r["aoi_per_dev"] for r in _tg(d_, "none")])
    al = np.concatenate([r["aoi_per_dev"] for r in _tg(d_, "lookup")])
    rs = lambda i: 100 * (al[i].mean() / a0[i].mean() - 1)
    v_ = np.array([rs(_br.integers(0, len(a0), len(a0))) for _ in range(1000)])
    M[f"GateRiseHw{t}"] = f((np.percentile(v_, 97.5) - np.percentile(v_, 2.5)) / 2, 0)
    M[f"GateRiseHwD{t}"] = f((np.percentile(v_, 97.5) - np.percentile(v_, 2.5)) / 2, 1)
    n0 = _gm(d_, "aoi_all_h", "none")
    _riseq[k_], _risep[k_], _risef[k_] = (100 * (_gm(d_, "aoi_all_h", ru) / n0 - 1) for ru in ("p75", "p95", "fix30"))
    M[f"GateRiseFix{t}"] = f(_risef[k_], 0)
    _nas[k_] = 100 * (1 - _gm(d_, "n_nas", "lookup") / _gm(d_, "n_nas", "none"))
    M[f"GateNas{t}"] = f(_nas[k_], 0)
    _unch += [abs(a["uplinked"] / b["uplinked"] - 1) * 100 for a, b in zip(_tg(d_, "lookup"), _tg(d_, "none"))]
    _kdel[k_] = 100 * _gm(d_, "share", "lookup", di="both")
    _fail[k_] = np.sum([r["n_failed"] for r in _tg(d_, "none", "return")]) / np.sum([r["n_nas"] for r in _tg(d_, "none", "return")])
    _agegap[k_] = 100 * abs(_gm(d_, "aoi_all_h", "none", "return") / _gm(d_, "aoi_all_h", "lookup", "return") - 1)
_5 = [k for k in TG if k[1] == "k5"]
_11 = [k for k in TG if k[1] != "k5"]
for tu, nm in ((15, "Fifteen"), (60, "Sixty"), (240, "Fourh")):
    M[f"Rise{nm}Lo"], M[f"Rise{nm}Hi"] = f(abs(min(_rise[tu][k] for k in _5)) if tu == 240 else min(_rise[tu][k] for k in _5), 0 if tu < 240 else 1), f(max(_rise[tu][k] for k in _5), 0 if tu < 240 else 1)
    M[f"Rise{nm}ElevenLo"], M[f"Rise{nm}ElevenHi"] = f(min(_rise[tu][k] for k in _11), 0 if tu < 240 else 1), f(max(_rise[tu][k] for k in _11), 0 if tu < 240 else 1)
M["GateRiseLo"], M["GateRiseHi"], M["GateRiseElevenLo"], M["GateRiseElevenHi"] = M["RiseFifteenLo"], M["RiseFifteenHi"], M["RiseFifteenElevenLo"], M["RiseFifteenElevenHi"]
M["GateRisePHi"], M["GateRiseQHi"] = f(max(_risep[k] for k in _5), 0), f(max(_riseq[k] for k in _5), 0)
M["GateRiseFixLo"], M["GateRiseFixHi"] = f(min(_risef[k] for k in _5), 0), f(max(_risef[k] for k in _5), 0)
M["GateNasLo"], M["GateNasHi"] = f(min(_nas[k] for k in _5), 0), f(max(_nas[k] for k in _5), 0)
M["GateUnchanged"] = f(np.ceil(100 * max(max(_unch), 1e-6)) / 100, 2)
M["GateKnDiffLess"] = f(np.ceil(100 * max(max(_kdiff), 1e-6)) / 100, 2)
M["GateKnDelLo"], M["GateKnDelHi"] = f(min(_kdel.values()), 1), f(max(_kdel.values()), 1)
M["GateSpread"] = f(max(_ksp), 2)
M["RetFailLo"], M["RetFailHi"] = f(min(_fail[k] for k in _5), 1), f(max(_fail[k] for k in _5), 1)
M["RetFailElevenLo"], M["RetFailElevenHi"] = f(min(_fail[k] for k in _11), 1), f(max(_fail[k] for k in _11), 1)
M["RetAgeGap"] = f(max(_agegap.values()), 1)
_fx5 = [100 * (1 - _gm(TG[k], "n_nas", "fix30") / _gm(TG[k], "n_nas", "none")) for k in _5]
for k_, d_ in TG.items():
    t = TAG[k_[0]] + TAG[k_[1]]
    M[f"GateNasFix{t}"] = f(100 * (1 - _gm(d_, "n_nas", "fix30") / _gm(d_, "n_nas", "none")), 0)
    M[f"GateRiseFixD{t}"] = f(_risef[k_], 1)
    M[f"GateRiseD{t}"] = f(_rise[15][k_], 1)
M["GateCutLo"], M["GateCutHi"] = f(min(min(_fx5), min(_nas[k] for k in _5)), 0), f(max(max(_fx5), max(_nas[k] for k in _5)), 0)
M["GateNasFixLo"], M["GateNasFixHi"] = f(min(_fx5), 0), f(max(_fx5), 0)
# the penalty of the must-return regime against contexts on board, no timer: by (9) the same uploads as the lookup on board
_mr = {k: 100 * (_gm(TG[k], "aoi_all_h", "none", "return") / _gm(TG[k], "aoi_all_h", "none") - 1) for k in TG}
M["MrRiseHi"], M["MrRiseLo"] = f(max(_mr[k] for k in _5), 1), f(min(_mr[k] for k in _5), 1)
M["MrGapMax"] = f(max(abs(_mr[k] - _rise[15][k]) for k in TG), 1)
M["MrOrbFive"], M["RiseDOrbFive"] = f(_mr[("orbcomm", "k5")], 1), f(_rise[15][("orbcomm", "k5")], 1)
_fxr = {k: 100 * (_gm(TG[k], "aoi_all_h", "fix30", "return") / _gm(TG[k], "aoi_all_h", "lookup", "return") - 1) for k in TG}
M["RetFixLo"], M["RetFixHi"] = f(min(_fxr.values()), 0), f(max(_fxr.values()), 0)
_fxf = [np.sum([r["n_failed"] for r in _tg(TG[k], "fix30", "return")]) / np.sum([r["n_nas"] for r in _tg(TG[k], "fix30", "return")]) for k in TG]
M["RetFixFailLo"], M["RetFixFailHi"] = f(min(_fxf), 2), f(max(_fxf), 2)
M["AbsKinEleven"] = f(_gm(TG[("kineis", "aws11")], "aoi_all_h", "lookup") - _gm(TG[("kineis", "aws11")], "aoi_all_h", "none"), 2)
assert max(_kdiff) < 1e-9, "keep-newest and FCFS age must be identical under gating"
# rejects the must-return regime actually produces, no timer (sf_timer_gate.py rejects)
_rx = {(fl, gw): [json.load(open(os.path.join(RV, f"rejx_{fl}_{gw}_w{w}.json"))) for w in WEEKS] for fl in FLEETS for gw in GWS}
_rf = {k: 100 * sum(x["rej_first_over"] for x in v) / max(1, sum(x["rej_first_n"] for x in v)) for k, v in _rx.items()}
_ra = {k: 100 * sum(x["rej_over"] for x in v) / max(1, sum(x["rej_n"] for x in v)) for k, v in _rx.items()}
M["RxFirstLo"], M["RxFirstHi"] = f(min(_rf[k] for k in _5), 0), f(max(_rf[k] for k in _5), 0)
M["RxAllLo"], M["RxAllHi"] = f(min(_ra[k] for k in _5), 0), f(max(_ra[k] for k in _5), 0)
M["RetAgeGapLess"] = f(np.ceil(10 * max(max(_agegap.values()), 1e-6)) / 10, 1)
M["GateRiseFixElevenHi"] = f(max(_risef[k] for k in _11), 0)
M["RetLookFailMax"] = f(max(float(M[f"RetFailLook{TAG[k[0]]}{TAG[k[1]]}"]) for k in TG), 2)
_chk = json.load(open(os.path.join(RV, "gating_check.json")))
assert all(abs(o["loop_age"] - o["sim_age"]) < 5e-4 and o["loop_up"] == o["sim_up"] for o in _chk), "gating check"
M["GateCheckN"] = str(len(_chk))
_gs = json.load(open(os.path.join(RV, "tgate_stress.json")))["rows"]
for di, nm in (("fcfs", "Fcfs"), ("lcfs", "Lcfs"), ("devonly", "Dev")):
    for ru, rn in (("none", "None"), ("lookup", "Look")):
        x_ = [r for r in _gs if r["disc"] == di and r["rule"] == ru][0]
        M[f"GStress{nm}{rn}"] = f(x_["aoi_all_h"], 2)
        M[f"GStress{nm}{rn}Del"] = f(100 * x_["share"], 1)
from sgp4.api import jday as _jd  # noqa: E402
_j0 = sum(_jd(2026, 10, 30, 0, 0, 0))
_age = {}
for fl in FLEETS:
    L_ = [x.rstrip() for x in open(os.path.join(HERE, "data_raw", f"{fl}.tle")) if x.strip()]
    for i in range(0, len(L_) - 2, 3):
        yr, doy = int(L_[i + 1][18:20]), float(L_[i + 1][20:32])
        _age[L_[i + 1][2:7]] = _j0 - (sum(_jd(2000 + yr, 1, 1, 0, 0, 0)) + doy - 1)
_a2 = [v for k_, v in _age.items() if k_ != _oid]
M["SetAgeLo"], M["SetAgeHi"], M["SetOutlierAge"] = f(min(_a2), 1), f(max(_a2), 1), f(_age[_oid], 1)
M["NP"] = "60"
M["PlArithTwoK"] = f(2000.0 / 7.53, 0)
_rows = [x for x in open(os.path.join(HERE, "..", "..", "..", "ns-3-dev", "contrib", "ntn-sagin", "data", "buoys.csv"))][1:]
M["NGreatLakes"] = str(sum(41 <= float(x.split(",")[1]) <= 49.5 and -93 <= float(x.split(",")[2]) <= -75.5 for x in _rows))
# Release 19 lookups
TM = {(fl, gw): [json.load(open(os.path.join(RV, f"timers_{fl}_{gw}_w{w}.json"))) for w in WEEKS] for fl in FLEETS for gw in GWS}
ov, fb, ov11, fb11, self_f, wm = [], [], [], [], [], []
for (fl, gw), R_ in TM.items():
    t = TAG[fl] + TAG[gw]
    m = lambda k: float(np.mean([r[k] for r in R_]))
    for k, nm in (("H_mean", "TmHMean"), ("H_p99", "TmHNn"), ("W_mean", "TmWMean"), ("W_p50", "TmWMed"), ("W_p95", "TmWNine"), ("W_p99", "TmWNn"), ("W_self_mean", "TmSelfMean"), ("W_self_p99", "TmSelfNn"), ("W_half_mean", "TmHalfMean"), ("W_half_p99", "TmHalfNn"), ("mae_res", "TmMaeRes"), ("mae_former", "TmMaeFormer")):
        M[f"{nm}{t}"] = f(m(k) / 3600, 2)
    M[f"TmOverNinety{t}"] = f(100 * m("over90"), 0)
    M[f"TmFallback{t}"] = f(100 * m("ok_fallback"), 0)
    M[f"TmPNine{t}"] = f(100 * m("ok_p95"), 0)
    M[f"TmSelfField{t}"] = f(100 * m("self_over_field"), 1)
    M[f"TmReplayW{t}"] = f(100 * m("wp99_replay_med"), 0)
    (ov if gw == "k5" else ov11).append(100 * m("over90"))
    (fb if gw == "k5" else fb11).append(100 * m("ok_fallback"))
    self_f.append(100 * m("self_over_field"))
    wm.append(max(r["W_max"] for r in R_))
M["OverNinetyLo"], M["OverNinetyHi"] = f(min(ov), 0), f(max(ov), 0)
M["OverNinetyElevenLo"], M["OverNinetyElevenHi"] = f(min(ov11), 0), f(max(ov11), 0)
M["FallbackElevenLo"], M["FallbackElevenHi"] = f(min(fb11), 0), f(max(fb11), 0)
M["SelfFieldMax"] = f(max(self_f), 1)
M["WMaxAll"] = f(max(wm) / 3600, 1)
M["RetLow"] = f(min(np.mean([r["R_over_A"] for r in v]) for v in TM.values()), 1)
M["RetHigh"] = f(max(np.mean([r["R_over_A"] for r in v]) for v in TM.values()), 1)
# per-device lifetime and margin
MG = {(fl, gw): [json.load(open(os.path.join(RV, f"margin_{fl}_{gw}_w{w}.json"))) for w in WEEKS] for fl in FLEETS for gw in GWS}
mg = lambda rule, k: [float(np.mean([r[rule][k] for r in v])) for v in MG.values()]
M["MgPooledLo"], M["MgPooledHi"] = f(min(mg("pooled", "fleet")), 2), f(max(mg("pooled", "fleet")), 2)
M["MgRepLo"], M["MgRepHi"] = f(min(mg("dev_replay", "fleet")), 2), f(max(mg("dev_replay", "fleet")), 2)
M["MgLo"], M["MgHi"] = f(min(mg("dev_margin", "fleet")), 2), f(max(mg("dev_margin", "fleet")), 2)
M["MgDevLo"], M["MgDevHi"] = f(min(mg("dev_margin", "dev_ok")), 0), f(max(mg("dev_margin", "dev_ok")), 0)
M["MgRepDevLo"], M["MgRepDevHi"] = f(min(mg("dev_replay", "dev_ok")), 0), f(max(mg("dev_replay", "dev_ok")), 0)
mh = [r["margin_h"] for v in MG.values() for r in v]
M["MgMinLo"], M["MgMinHi"] = f(60 * min(mh), 0), f(60 * max(mh), 0)
lb = MG[("orbcomm", "k5")][0]["latbin"]
M["OrbHighLat"] = f(np.mean([v[0]["latbin"]["60-90"]["p99_h"] for k, v in MG.items() if k == ("orbcomm", "k5")]), 0)
M["OrbMidLatLo"] = f(min(lb[k]["p99_h"] for k in ("0-30", "30-45", "45-60")), 1)
M["OrbMidLatHi"] = f(max(lb[k]["p99_h"] for k in ("0-30", "30-45", "45-60")), 1)
M["OrbHighLatN"] = str(lb["60-90"]["n"])
# what-if
for fl in FLEETS:
    W_ = json.load(open(os.path.join(RV, f"whatif_sat_{fl}_k5.json")))
    d = W_["devices"]
    g = lambda k: np.array([o[k] for o in d])
    t = TAG[fl]
    b, bp, p, tr = g("base"), g("base_pred"), g("pred"), g("truth")
    # week 0 is the development week; the reported test is weeks 1 and 2 (sf_whatif_sat_frozen.py)
    M[f"WhatIfDevTrue{t}"] = f(100 * (b.mean() - tr.mean()) / b.mean(), 0)
    M[f"WhatIfDevPred{t}"] = f(100 * (bp.mean() - p.mean()) / bp.mean(), 0)
    Z_ = [json.load(open(os.path.join(RV, f"whatif_sat_frozen_{fl}_w{w_}.json"))) for w_ in (1, 2)]
    zt, zc, zp, zb, zpl = [], [], [], [], []
    for z_ in Z_:
        bz = np.mean([o["base"] for o in z_]); bpz = np.mean([o["base_pred"] for o in z_]); pz = np.mean([o["pred"] for o in z_])
        plz = np.array([o["each"] for o in z_]).mean(axis=0)
        cz = 100 * (bz - plz) / bz
        zt.append(cz.mean()); zc.append(stats.t.ppf(0.975, len(cz) - 1) * cz.std(ddof=1) / np.sqrt(len(cz)))
        zp.append(100 * (bpz - pz) / bpz); zb.append(bz); zpl.append(plz)
    M[f"WhatIfCutTrue{t}"], M[f"WhatIfCutCi{t}"] = f(np.mean(zt), 0), f(max(zc), 0)
    M[f"WhatIfCutPred{t}"] = f(np.mean(zp), 0)
    S_ = {"orbcomm": 14, "kineis": 25, "sateliot": 4}[fl]
    M[f"WhatIfCutShare{t}"] = f(100 / (S_ + 1), 0)
    # exponential form on the buoy samples of weeks 1 and 2 (it needs no plan)
    from sf_whatif import alt as _alt
    import sf_core as _Fw
    _sw = _Fw.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    _c1, _c0 = [], []
    for w_ in (1, 2):
        for lat_ in np.load(os.path.join(OUT, f"sf_{fl}_k5_w{w_}.npz"))["buoys"][:, 0]:
            _st = [_Fw.contact_stats(np.radians(lat_), _alt(s_), s_.inclo, np.radians(20.0)) for s_ in list(_sw) + [_sw[0]]]
            pk_, eg_ = np.array([q_["p"] for q_ in _st]), np.array([q_["EG"] for q_ in _st])
            ok_ = np.isfinite(eg_) & (pk_ > 0)
            _c1.append(np.prod(1 - pk_) / np.sum(1 / eg_[ok_]) if ok_.any() else np.nan)
            _c0.append(np.prod(1 - pk_[:-1]) / np.sum(1 / eg_[:-1][ok_[:-1]]) if ok_[:-1].any() else np.nan)
    M[f"WhatIfCutCf{t}"] = f(100 * (np.nanmean(_c0) - np.nanmean(_c1)) / np.nanmean(_c0), 0)
    M[f"WhatIfLevelErr{t}"] = f(abs(100 * (p.mean() / tr.mean() - 1)), 1)
    M[f"WhatIfBuoy{t}"] = f(100 * np.median(np.abs(p / tr - 1)), 0)
    pl = np.concatenate(zpl)
    M[f"WhatIfPlaceLo{t}"], M[f"WhatIfPlaceHi{t}"] = f(pl.min() / 3600, 2), f(pl.max() / 3600, 2)
    M[f"WhatIfMean{t}"] = f(tr.mean() / 3600, 2)
    M[f"WhatIfBase{t}"] = f(np.mean(zb) / 3600, 2)
    M[f"WhatIfMs{t}"] = f(1e3 * W_["time"]["whatif_s_per_dev"], 0)
    M[f"WhatIfReplayMs{t}"] = f(1e3 * W_["time"]["replay_s_per_dev_30"], 0)
ge, gn = [], []
for fl in FLEETS:
    for gw in GWS:
        for o in json.load(open(os.path.join(RV, f"whatif_gw_{fl}_{gw}.json"))):
            ge.append(abs(100 * (o["pred"] / o["truth"] - 1)))
            gn.append(abs(100 * (o["base_prior"] / o["truth"] - 1)))
M["GwWhatN"] = str(len(ge))
M["GwWhatMed"], M["GwWhatMax"] = f(np.median(ge), 0), f(max(ge), 0)
M["GwNoneMed"], M["GwNoneMax"] = f(np.median(gn), 0), f(max(gn), 0)
M["GwWhatTwenty"] = str(int(np.sum(np.array(ge) <= 20)))
# Walker Star of Kellermann et al.
KL = json.load(open(os.path.join(RV, "kellermann.json")))
for mk, nm in (("10.0", "Ten"), ("20.0", "Twenty"), ("30.0", "Thirty"), ("40.0", "Forty")):
    M[f"KellMed{nm}"] = f(KL[mk]["med_e2e_h"], 2); M[f"KellUe{nm}"] = f(KL[mk]["med_ue_h"], 2); M[f"KellSat{nm}"] = f(KL[mk]["med_sat_h"], 2)
    M[f"KellModel{nm}"] = f(abs(KL[mk]["fleet_err_model"]), 1); M[f"KellReplay{nm}"] = f(abs(KL[mk]["fleet_err_replay"]), 1)
    M[f"KellModelMed{nm}"] = f(KL[mk]["per_med_model"], 0); M[f"KellReplayMed{nm}"] = f(KL[mk]["per_med_replay"], 0)
# Ari et al. schedulers
AR = {(fl, w): json.load(open(os.path.join(RV, f"ari_{fl}_k5_w{w}.json"))) for fl in FLEETS for w in WEEKS}
for fl in FLEETS:
    t = TAG[fl]
    a = [AR[(fl, w)] for w in WEEKS]
    M[f"AriContPeak{t}"] = f(np.mean([x["contact"][0] for x in a]), 1)
    M[f"AriContMean{t}"] = f(np.mean([x["contact"][1] for x in a]), 1)
    M[f"AriContUp{t}"] = f(np.mean([x["contact"][2] for x in a]), 0)
    for mb, nm in (("14", "Ft"), ("56", "Ff")):
        for p_ in ("periodic", "smdp", "plan"):
            M[f"Ari{p_.capitalize()}{nm}Peak{t}"] = f(np.mean([x["budgets"][mb][p_][0] for x in a]), 1)
            M[f"Ari{p_.capitalize()}{nm}Mean{t}"] = f(np.mean([x["budgets"][mb][p_][1] for x in a]), 1)
            M[f"Ari{p_.capitalize()}{nm}Up{t}"] = f(np.mean([x["budgets"][mb][p_][2] for x in a]), 0)
# slot simulator sweeps
FR = [r for r in (json.loads(l) for l in open(os.path.join(RV, "finite.jsonl"))) if r["part"] not in ("uplink", "uplink2")]
FR += [json.loads(l) for l in open(os.path.join(RV, "finite_rr.jsonl"))]   # shared uplink: round robin; part index: old rule
fr = lambda **k: [r for r in FR if all(r.get(a) == b for a, b in k.items())]
ckD, ckO = [], []
for r in fr(part="check", disc="fcfs"):
    z = np.load(os.path.join(OUT, f"sf_{r['fl']}_{r['gw']}_w{r['w']}.npz"))
    d = z["d_SF1"][np.isfinite(z["d_SF1"])]
    ckD.append(abs(100 * (r["d_mean_h"] / (d.mean() / 3600) - 1)))
    ckO.append(abs(100 * (r["occ_p99"] / np.percentile(z["occ_fcfs"], 99) - 1)))
M["CheckDelay"], M["CheckOcc"] = f(max(ckD), 1), f(max(ckO), 0)
pm = lambda **k: float(np.mean([r["occ_p99"] for r in fr(**k)]))
cuth, cut15, xs, ys = [], [], [], []
for fl in FLEETS:
    for gw in GWS:
        t = TAG[fl] + TAG[gw]
        for tu in (5, 15, 30, 60, 120, 240):
            a, b = pm(part="tu", fl=fl, gw=gw, tu=tu, disc="fcfs"), pm(part="tu", fl=fl, gw=gw, tu=tu, disc="both")
            c, e_ = pm(part="tu", fl=fl, gw=gw, tu=tu, disc="devonly"), pm(part="tu", fl=fl, gw=gw, tu=tu, disc="satonly")
            M[f"CutTu{tu}{t}".replace("5", "Five").replace("1", "One").replace("2", "Two").replace("3", "Three").replace("6", "Six").replace("4", "Four").replace("0", "Z")] = f(a / b, 1)
            if tu == 60:
                cuth.append(a / b)
            if tu == 15:
                cut15.append(a / b)
                M[f"CutDev{t}"] = f(a / c, 1)
                M[f"CutSat{t}"] = f(a / e_, 1)
                M[f"CutBoth{t}"] = f(a / b, 1)
                u = fr(part="tu", fl=fl, gw=gw, tu=tu, disc="devonly")
                uf = fr(part="tu", fl=fl, gw=gw, tu=tu, disc="fcfs")
                M[f"UpShare{t}"] = f(100 * np.mean([r["uplinked"] for r in u]) / np.mean([r["uplinked"] for r in uf]), 0)
                sat_ = fr(part="tu", fl=fl, gw=gw, tu=tu, disc="both")
                M[f"BothShare{t}"] = f(100 * np.mean([r["share"] for r in sat_]), 0)
M["CutHourLo"], M["CutHourHi"] = f(min(cuth), 1), f(max(cuth), 1)
sd = fr(part="tu", fl="sateliot", gw="k5", tu=15, disc="both")
M["DevShareSatFive"] = f(100 * np.mean([r["superseded_dev"] / (r["superseded_dev"] + r["superseded_sat"]) for r in sd]), 0)
ds = [100 * np.mean([r["superseded_dev"] / max(1, r["superseded_dev"] + r["superseded_sat"]) for r in fr(part="tu", fl=fl, gw=gw, tu=15, disc="both")]) for fl in FLEETS for gw in GWS]
M["DevShareLo"], M["DevShareHi"] = f(min(ds), 0), f(max(ds), 0)
oc = lambda fl, n, d: fr(part="pop", fl=fl, gw="k5", n=n, disc=d)[0]["occ_p99"]
for fl in FLEETS:
    t = TAG[fl]
    for n, nm in ((60, "Sixty"), (400, "Four"), (919, "All")):
        a, b = oc(fl, n, "fcfs"), oc(fl, n, "both")
        M[f"Pop{nm}Fcfs{t}"], M[f"Pop{nm}New{t}"], M[f"Pop{nm}Cut{t}"] = f(a, 0), f(b, 0), f(a / b, 1)
        M[f"Pop{nm}Kb{t}"] = f(a * 52 / 1000, 0)
M["PopAll"] = "919"
# stability limit of the delivery budget
from sf_replay import plan as _plan  # noqa: E402
import sf_core as _F  # noqa: E402
import sf_mc as _mc  # noqa: E402
for fl in ("orbcomm", "sateliot"):
    t = TAG[fl]
    sats = _F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    _, _, v = _plan(sats, _mc.load_gateways("k5"), 0.0)
    C, G = [], []
    for k in range(len(sats)):
        c_, g_ = _F.runs(v[k])
        C += list(c_ * 20.0); G += list(g_ * 20.0)
    EC, EG = np.mean(C), np.mean(G)
    rho = 60 * (EC + EG) / (len(sats) * 900.0)
    M[f"RhoNs{t}"] = f(rho, 0)
    M[f"EcGw{t}"] = f(EC / 60, 1)
    M[f"EgGw{t}"] = f(EG / 60, 0)
    cap = 0.5 * 1e6 * EC / (8 * 52)
    M[f"ShareStar{t}"] = f(100 * rho * 8 * 52 / (1e6 * EC), 3)
    M[f"NStar{t}"] = f(1e6 * EC / (8 * 52) * len(sats) * 900.0 / (EC + EG) / 1e6, 1)
    feeds = {"orbcomm": (10, 20, 40, 80), "sateliot": (20, 40, 80, 160)}[fl]
    for b_, nm in zip(feeds, "ABCD"):
        for dc, dn in (("fcfs", "Fcfs"), ("lcfs", "Lcfs"), ("both", "New")):
            for days, dd_ in ((4, "Short"), (14, "Long")):
                rr = fr(part="feed", fl=fl, feed=b_, disc=dc, days=days)
                M[f"Mc{dd_}{t}{nm}{dn}"] = f(np.mean([r["aoi_h"] for r in rr]), 2)
                M[f"McGrow{dd_}{t}{nm}{dn}"] = f(np.mean([r["aoi_second_h"] / r["aoi_first_h"] for r in rr]), 2)
                M[f"McDel{dd_}{t}{nm}{dn}"] = f(100 * np.mean([r["share"] for r in rr]), 0)
# shared uplink and Monitoring List
for fl in FLEETS:
    t = TAG[fl]
    base = fr(part="tu", fl=fl, gw="k5", tu=15, disc="fcfs", w=0)[0]
    bnew = fr(part="tu", fl=fl, gw="k5", tu=15, disc="both", w=0)[0]
    for n, nm in ((150, "One"), (919, "All")):
        bs = base if n == 150 else fr(part="pop", fl=fl, gw="k5", n=919, disc="fcfs")[0]
        bn = bnew if n == 150 else fr(part="pop", fl=fl, gw="k5", n=919, disc="both")[0]
        for bps, bn_ in ((400, "Slow"), (4000, "Fast")):
            a = fr(part="rr", fl=fl, gw="k5", n=n, up_bps=bps, disc="fcfs", w=0)[0]
            b = fr(part="rr", fl=fl, gw="k5", n=n, up_bps=bps, disc="both", w=0)[0]
            c_ = fr(part="rr", fl=fl, gw="k5", n=n, up_bps=bps, disc="lcfs", w=0)[0]
            e_ = fr(part="rr", fl=fl, gw="k5", n=n, up_bps=bps, disc="devonly", w=0)[0]
            M[f"Up{bn_}{nm}Fcfs{t}"] = f(a["aoi_all_h"], 2); M[f"Up{bn_}{nm}New{t}"] = f(b["aoi_all_h"], 2)
            M[f"Up{bn_}{nm}Lcfs{t}"] = f(c_["aoi_all_h"], 2); M[f"Up{bn_}{nm}Dev{t}"] = f(e_["aoi_all_h"], 2)
            M[f"Up{bn_}{nm}Del{t}"] = f(100 * a["share"], 0)
            M[f"Up{bn_}{nm}NewGrow{t}"] = f(b["aoi_second_h"] / b["aoi_first_h"], 2)
        M[f"UpNone{nm}Fcfs{t}"] = f(bs["aoi_h"], 2); M[f"UpNone{nm}New{t}"] = f(bn["aoi_h"], 2)
    S_ = {"orbcomm": 14, "kineis": 25, "sateliot": 4}[fl]
    bl = np.mean([r["d_mean_h"] for r in fr(part="tu", fl=fl, gw="k5", tu=15, disc="fcfs")])
    for m_, nm in ((1, "One"), (max(1, S_ // 2), "Half")):
        rr = fr(part="ml", fl=fl, gw="k5", ml=m_)
        M[f"Ml{nm}{t}"] = f(np.mean([r["d_mean_h"] for r in rr]), 2)
        M[f"Ml{nm}Del{t}"] = f(100 * np.mean([r["share"] for r in rr]), 0)
    M[f"MlAll{t}"] = f(bl, 2)
    M[f"MlHalfSize{t}"] = str(max(1, S_ // 2))

# ---- ns-3 parts added by the Nov 2026 campaign
def nsm(key, rr):
    return float(np.mean([r[key] for r in rr]))


def rows(part, fl=None, gw=None, disc=None, pol="sf1", **x):
    return [r for r in R if r["part"] == part and (fl is None or r["fleet"] == fl) and (gw is None or r["gw"] == gw)
            and (disc is None or r["disc"] == disc) and r["policy"] == pol
            and all(r.get(f"x_{k}") == v for k, v in x.items())
            and ("devKeepNewest" in x or "x_devKeepNewest" not in r)]


# record of the queue defect: the earlier campaign (default transmit queue), first-available FCFS, kept in out/ns3_sf2.jsonl
_old = [r for r in (json.loads(l) for l in open(os.path.join(OUT, "ns3_sf2.jsonl"))) if not r.get("error") and r["part"] == "main"
        and r["policy"] == "sf1" and r["disc"] == "fcfs"]
_w = max(_old, key=lambda r: r["generated"] - r["delivered"])
M["OldQueueLossMax"], M["OldQueueLossOf"] = str(_w["generated"] - _w["delivered"]), f"{_w['generated']:,}".replace(",", "\\,")
M["NsLostMax"] = str(max(r["unaccounted"] for r in R))
M["NsLostRuns"] = str(sum(r["unaccounted"] > 0 for r in R))
assert min(r["unaccounted"] for r in R) >= 0, "more updates accounted for than generated"
M["NsMacDrops"] = str(sum(r["mac_drops"] for r in R))
# occupancy log (M4): same quantity as eq. (occ), at 60 buoys, against the slot simulator at 60 buoys
ncut, ndev, nsat, nshare, nhour, occerr, delnew = [], [], [], [], [], [], []
for fl in FLEETS:
    for gw in GWS:
        t = TAG[fl] + TAG[gw]
        a = nsm("occ_p99", rows("main", fl, gw, "fcfs"))
        b = nsm("occ_p99", rows("main", fl, gw, "newest"))
        c = nsm("occ_p99", rows("ablate", fl, gw, "fcfs", devKeepNewest=1))
        e_ = nsm("occ_p99", rows("ablate", fl, gw, "fcfs", satKeepNewest=1))
        M[f"NsOccFcfs{t}"], M[f"NsOccNew{t}"] = f(a, 0), f(b, 0)
        M[f"NsCutBoth{t}"], M[f"NsCutDev{t}"], M[f"NsCutSat{t}"] = f(a / b, 1), f(a / c, 1), f(a / e_, 1)
        ncut.append(a / b); ndev.append(a / c); nsat.append(a / e_)
        rn = rows("main", fl, gw, "newest")
        sh = 100 * np.sum([r["replaced_dev"] for r in rn]) / max(1, np.sum([r["replaced_dev"] + r["replaced_sat"] for r in rn]))
        M[f"NsDevShare{t}"] = f(sh, 0)
        nshare.append(sh)
        rd = rows("ablate", fl, gw, "fcfs", devKeepNewest=1)
        M[f"NsUpShare{t}"] = f(100 * np.mean([1 - r["replaced_dev"] / r["generated"] for r in rd]), 0)
        pa = nsm("occ_p99", rows("period", fl, gw, "fcfs", periodMin=60))
        pb = nsm("occ_p99", rows("period", fl, gw, "newest", periodMin=60))
        M[f"NsCutHour{t}"] = f(pa / pb, 1)
        nhour.append(pa / pb)
        slot = fr(part="pop", fl=fl, gw=gw, n=60, disc="fcfs")
        if slot:
            occerr.append(abs(100 * (a / slot[0]["occ_p99"] - 1)))
        delnew.append(nsm("delivered_share", rn))
M["NsCutLo"], M["NsCutHi"] = f(min(ncut), 1), f(max(ncut), 0)
M["NsCutDevLo"], M["NsCutDevHi"] = f(min(ndev), 1), f(max(ndev), 0)
M["NsCutSatLo"], M["NsCutSatHi"] = f(min(nsat), 1), f(max(nsat), 0)
M["NsDevShareLo"], M["NsDevShareHi"] = f(min(nshare), 0), f(max(nshare), 0)
M["NsCutHourLo"], M["NsCutHourHi"] = f(min(nhour), 1), f(max(nhour), 1)
if occerr:
    M["NsOccSlotErrMax"], M["NsOccSlotErrMed"] = f(max(occerr), 0), f(np.median(occerr), 0)
M["NsDelFcfsMin"] = f(100 * min(r["delivered_share"] for r in rows("main", disc="fcfs")), 2)
# 14-day delivery budgets (M7)
for fl, feeds in FEEDS.items():
    for lv, fb in zip("ABCD", feeds):
        for disc, dt in (("fcfs", "Fcfs"), ("lcfs", "Lcfs"), ("newest", "New")):
            rr = rows("feedlong", fl, "k5", disc, feedBudget=fb)
            if rr:
                t = f"{TAG[fl]}{lv}{dt}"
                M[f"NsLong{t}"] = f(nsm("aoi_served_h", rr), 2)
                M[f"NsLongGrow{t}"] = f(np.mean([r["aoi_second_h"] / r["aoi_first_h"] for r in rr]), 2)
                M[f"NsLongDel{t}"] = f(100 * nsm("delivered_share", rr), 1)
# shared uplink (M10)
for fl in FLEETS:
    for bps, bn_ in ((0, "None"), (400, "Slow"), (4000, "Fast")):
        for disc, dt in (("fcfs", "Fcfs"), ("newest", "New")):
            rr = rows("uplink", fl, "k5", disc, upBps=bps)
            if rr:
                t = f"NsUp{bn_}{dt}{TAG[fl]}"
                m, h = ci([r["aoi_h"] for r in rr])
                M[t], M[t + "Ci"] = f(m, 2), f(h, 2)
for bps, bn_ in ((0, "None"), (400, "Slow")):
    for disc, dt in (("fcfs", "Fcfs"), ("newest", "New")):
        rr = rows("upall", "sateliot", "k5", disc, upBps=bps)
        if rr:
            t = f"NsAll{bn_}{dt}"
            m, h = ci([r["aoi_h"] for r in rr])
            M[t], M[t + "Ci"] = f(m, 2), f(h, 2)
            M[t + "Del"] = f(100 * nsm("delivered_share", rr), 1)
            M[t + "Grow"] = f(np.mean([r["aoi_second_h"] / r["aoi_first_h"] for r in rr]), 2)
_a = rows("upall", "sateliot", "k5", "fcfs", upBps=400)
M["NsAllSlowFcfsAll"] = f(nsm("aoi_h", _a), 2)
M["NsAllStarved"] = str(int(round(919 * (1 - nsm("served_share", _a)))))
_b = rows("upall", "sateliot", "k5", "lcfs", upBps=400)
M["NsAllSlowLcfs"], M["NsAllSlowLcfsDel"] = f(nsm("aoi_h", _b), 2), f(100 * nsm("delivered_share", _b), 1)
_c = rows("upall", "sateliot", "k5", "fcfs", upBps=400, devKeepNewest=1)
M["NsAllSlowDev"], M["NsAllSlowDevDel"] = f(nsm("aoi_h", _c), 2), f(100 * nsm("delivered_share", _c), 1)
for disc, dt in (("fcfs", "Fcfs"), ("lcfs", "Lcfs"), ("newest", "New")):
    _d = rows("uplong", "sateliot", "k5", disc, upBps=400)
    M[f"NsUpLong{dt}"], M[f"NsUpLong{dt}Del"] = f(nsm("aoi_h", _d), 2), f(100 * nsm("delivered_share", _d), 1)
    M[f"NsUpLong{dt}Grow"] = f(np.mean([r["aoi_second_h"] / r["aoi_first_h"] for r in _d]), 2)
# slot simulator under round robin at the overload point, three weeks, and its old fixed-priority rule
_rr = lambda d: fr(part="rr", fl="sateliot", n=919, up_bps=400, disc=d)
for d_, nm_ in (("fcfs", "Fcfs"), ("lcfs", "Lcfs"), ("devonly", "Dev"), ("both", "New")):
    x_ = _rr(d_)
    M[f"RrSlot{nm_}Lo"], M[f"RrSlot{nm_}Hi"] = f(min(r["aoi_all_h"] for r in x_), 2), f(max(r["aoi_all_h"] for r in x_), 2)
    M[f"RrSlot{nm_}Del"] = f(100 * np.mean([r["share"] for r in x_]), 1)
    M[f"RrSlot{nm_}Delay"] = f(np.mean([r["d_mean_h"] for r in x_]), 1)
    M[f"RrSlot{nm_}Served"] = str(min(r["n_served"] for r in x_))
    M[f"RrSlot{nm_}Grow"] = f(np.mean([r["aoi_all_second_h"] / r["aoi_all_first_h"] for r in x_]), 2)
M["UpSlowAllLcfsSat"], M["UpSlowAllDevSat"] = M["RrSlotLcfsLo"], M["RrSlotDevLo"]
# rate sweep at all stations, Corollary 2, ns-3 at the rates around the break
_rate = lambda fl_, d_, b_: (fr(part="rate", fl=fl_, disc=d_, up_bps=b_) or fr(part="rr", fl=fl_, n=919, disc=d_, up_bps=b_, w=0))[0]
_sw = (400, 500, 600, 700, 800, 1000, 1600, 2400, 4000)
_gr = lambda r: r["aoi_all_second_h"] / r["aoi_all_first_h"]
_stable = [b for b in _sw if _gr(_rate("sateliot", "fcfs", b)) < 1.2 and _rate("sateliot", "fcfs", b)["share"] > 0.995]
_unst = [b for b in _sw if b not in _stable]
M["BreakBelow"], M["BreakAbove"] = str(max(_unst)), str(min(_stable))
for b_ in _sw:
    x_ = _rate("sateliot", "fcfs", b_)
    M[f"SwF{b_}".replace("0", "Z").replace("1", "O").replace("2", "T").replace("4", "F").replace("5", "V").replace("6", "S").replace("7", "N").replace("8", "E")] = f(x_["aoi_all_h"], 2)
_l_all = [_rate("sateliot", "lcfs", b)["aoi_all_h"] for b in _sw]
_d_all = [_rate("sateliot", "devonly", b)["aoi_all_h"] for b in _sw if fr(part="rate", fl="sateliot", disc="devonly", up_bps=b) or fr(part="rr", fl="sateliot", n=919, disc="devonly", up_bps=b, w=0)]
M["SwLcfsLo"], M["SwLcfsHi"] = f(min(_l_all), 2), f(max(_l_all), 2)
M["SwDevLo"], M["SwDevHi"] = f(min(_d_all), 2), f(max(_d_all), 2)
UL = json.load(open(os.path.join(RV, "uplink_limit.json")))
M["RStarAllSat"] = f(UL["sateliot"]["all"]["rstar"], 0)
M["RStarSlide"], M["RStarSlideN"] = f(UL["sateliot"]["sliding"]["rstar"], 0), str(UL["sateliot"]["sliding"]["n"])
M["PktBits"] = str(8 * 52)
# the break of FCFS at long horizons, from the device backlog per day (sf_finite_runs.py part long)
_lg = [r for r in FR if r["part"] == "long" and r["disc"] == "fcfs"]
_days = sorted({r["days"] for r in _lg})
M["LongDaysA"], M["LongDaysB"] = str(_days[0]), str(_days[-1])


def _slope(r):
    """Reports per day added to the device backlog over the second half of the traffic."""
    b = [x for x in r["backlog_t"] if x[0] <= r["days"] - 3.0]
    b = b[len(b) // 2:]
    return float(np.polyfit([x[0] for x in b], [x[1] for x in b], 1)[0])


_lb = {(r["days"], r["up_bps"]): r for r in _lg}
_st = [b_ for b_ in sorted({k[1] for k in _lb}) if all(_slope(_lb[(d_, b_)]) < 919 * 0.2 for d_ in _days)]
M["LongStable"] = str(min(_st))
_l16 = _lb[(_days[-1], 1600)]
M["LongSixteenAge"], M["LongSixteenGrow"] = f(_l16["aoi_all_h"], 2), f(_gr(_l16), 2)
M["SwFSixteenGrow"] = f(_gr(_rate("sateliot", "fcfs", 1600)), 2)
_un = max(b_ for b_ in {k[1] for k in _lb} if b_ < min(_st))
M["LongUnstable"] = str(_un)
M["LongUnstableSlope"] = f"{_slope(_lb[(_days[-1], _un)]):,.0f}".replace(",", "\\,")
M["LongStableSlope"] = f(_slope(_lb[(_days[-1], min(_st))]), 0)
for fl, t_ in (("orbcomm", "Orb"), ("kineis", "Kin")):
    for b_, nm_ in ((50, "Fifty"), (100, "Hund"), (200, "Twoh")):
        x_ = fr(part="rate", fl=fl, disc="fcfs", up_bps=b_)[0]
        M[f"Sw{t_}{nm_}"], M[f"Sw{t_}{nm_}Del"] = f(x_["aoi_all_h"], 1), f(100 * x_["share"], 0)
_Sh = 3750 * np.log2(1 + 10 ** (15.0 / 10))
M["ShannonKbps"] = f(_Sh / 1000, 0)
M["ShannonRatio"] = f(_Sh / 400, 0)
M["ShannonDevices"] = f(919 * _Sh / min(_stable) / 1000, 0)
_nsw = [r for r in (json.loads(l) for l in open(os.path.join(OUT, "ns3_sf5_sweep.jsonl"))) if not r.get("error")]
assert len(_nsw) == 15 and {r["sha"] for r in _nsw} == set(_sha5), "ns-3 sweep"
for b_, nm_ in ((1000, "Oneth"), (1600, "Sixteen"), (2400, "TwoFour")):
    x_ = [r for r in _nsw if r["x_upBps"] == b_]
    M[f"NsSw{nm_}"], M[f"NsSw{nm_}Grow"] = f(nsm("aoi_h", x_), 2), f(np.mean([r["aoi_second_h"] / r["aoi_first_h"] for r in x_]), 2)
    M[f"NsSw{nm_}Del"] = f(100 * nsm("delivered_share", x_), 1)
M["NsSwRuns"] = str(len(_nsw))
_used = [r for r in R if r["part"] not in ("feed", "feedlong", "buffer")]
_rex = [r for r in REX if r["part"] == "upall" and r.get("x_upBps") == 400 and r["disc"] != "newest"]  # FCFS, LCFS, device only
M["NsRunsAll"] = str(len(_used) + len(_nsw) + len(_rex))
M["NsRunsNew"] = str(len(R5_) + len(_nsw))
M["NsRunsOld"] = str(len(_used) - len(R5_))
M["NsRunsWhole"] = str(len(_rex))
# the break over three start dates at 28 days (slot simulator, part window) and the same horizon as ns-3 (part same)
_win = [r for r in FR if r["part"] == "window"]
_w1 = lambda st, b_: [r for r in _win if r.get("start") == st and r["up_bps"] == b_ and "seed" not in r and r["fl"] == "sateliot"][0]
_rates = (1800, 2000, 2200, 2400)
_starts = (-14.0, 0.0, 14.0)
_end = lambda r: [x for x in r["backlog_t"] if x[0] <= r["days"] - 3.0][-1][1]
_floor = np.mean([_end(_w1(st, 2400)) for st in (-14.0, 0.0)])   # the two level runs; 13 Nov at 2400 b/s is still lifting
M["WinFloor"] = f(_floor / 1000, 0)
M["WinThreshold"] = f(919 * 0.2, 0)
_s = {b_: [_slope(_w1(st, b_)) for st in _starts] for b_ in _rates}
_e = {b_: [_end(_w1(st, b_)) / _floor for st in _starts] for b_ in _rates}
for b_, nm in ((1800, "Eighteen"), (2000, "Twenty"), (2200, "TwentyTwo"), (2400, "TwentyFour")):
    M[f"WinSlope{nm}Lo"], M[f"WinSlope{nm}Hi"] = f(min(_s[b_]), 0).replace("-", "$-$"), f(max(_s[b_]), 0).replace("-", "$-$")
    M[f"WinEnd{nm}Lo"], M[f"WinEnd{nm}Hi"] = f(min(_e[b_]), 1), f(max(_e[b_]), 1)
_sd = {b_: [r for r in _win if r.get("seed") == 41 and r["up_bps"] == b_][0] for b_ in (2000, 2200)}
_w3000 = _w1(14.0, 3000)
M["WinThreeKSlope"], M["WinThreeKEnd"] = f(_slope(_w3000), 0).replace("-", "$-$"), f(_end(_w3000) / _floor, 1)
M["WinSeedDiff"] = f(max(abs(_sd[b_]["aoi_all_h"] - _w1(0.0, b_)["aoi_all_h"]) for b_ in (2000, 2200)), 2)
for fl, t_ in (("orbcomm", "Orb"), ("kineis", "Kin")):
    x_ = [r for r in _win if r["fl"] == fl][0]
    M[f"WinOther{t_}Age"], M[f"WinOther{t_}Slope"] = f(x_["aoi_all_h"], 2), f(_slope(x_), 0).replace("-", "$-$")
_sm = {r["up_bps"]: r for r in FR if r["part"] == "same"}
# ns-3 means with 95 % half-widths, five gateways, Table of ns-3 results
def _tci(vals, n=2):
    m_, h_ = ci(vals)
    return f(m_, n), f(h_, n)
for fl in FLEETS:
    for gw in GWS:
        t = TAG[fl] + TAG[gw]
        bp_, kn_, fc_ = rows("main", fl, gw, "fcfs", pol="bp"), rows("main", fl, gw, "newest"), rows("main", fl, gw, "fcfs")
        M[f"NtServ{t}"], M[f"NtServ{t}Hw"] = _tci([100 * r["served_share"] for r in bp_], 0)
        M[f"NtDel{t}"], M[f"NtDel{t}Hw"] = _tci([100 * r["delivered_share"] for r in kn_], 1)
        M[f"NtOcc{t}"], M[f"NtOcc{t}Hw"] = _tci([r["occ_p99"] for r in fc_], 0)
        M[f"NtOccN{t}"], M[f"NtOccN{t}Hw"] = _tci([r["occ_p99"] for r in kn_], 0)
        _p = {r["rep"]: r["occ_p99"] for r in fc_}
        M[f"NtCut{t}"], M[f"NtCut{t}Hw"] = _tci([_p[r["rep"]] / max(r["occ_p99"], 1e-9) for r in kn_], 1)
        _x, _y = {r["rep"]: r for r in bp_}, {r["rep"]: r for r in fc_}
        _pb, _ps = [], []
        for k_ in _x:
            s_ = [(a, b) for a, b, c in zip(_x[k_]["aoi_per_buoy"], _y[k_]["aoi_per_buoy"], _x[k_]["served_per_buoy"]) if c]
            _pb.append(np.mean([a for a, _ in s_])); _ps.append(np.mean([b for _, b in s_]))
        M[f"NtPb{t}"], M[f"NtPb{t}Hw"] = _tci(_pb); M[f"NtPs{t}"], M[f"NtPs{t}Hw"] = _tci(_ps)
for fl in FLEETS:
    t = TAG[fl] + "Five"
    S_ = {"orbcomm": 14, "kineis": 25, "sateliot": 4}[fl]
    M[f"NtMl{t}"], M[f"NtMl{t}Hw"] = _tci([r["d_mean"] for r in rows("main", fl, "k5", "fcfs")])
    for m_, nm in ((1, "One"), (max(1, S_ // 2), "Half")):
        M[f"NtMl{nm}{t}"], M[f"NtMl{nm}{t}Hw"] = _tci([r["d_mean"] for r in rows("ml", fl, "k5", "fcfs", monitorList=m_)])
for d_, nm_ in (("fcfs", "Fcfs"), ("lcfs", "Lcfs"), ("devonly", "Dev")):
    x_ = fr(part="index", fl="sateliot", n=919, up_bps=400, disc=d_)[0]
    M[f"IdxSlot{nm_}"], M[f"IdxSlot{nm_}Served"] = f(x_["aoi_all_h"], 2), str(x_["n_served"])
# ns-3 at the overload point: all buoys counted, what each rule sends, paired LCFS against device keep-newest
_upall = {"Fcfs": rows("upall", "sateliot", "k5", "fcfs", upBps=400), "Lcfs": rows("upall", "sateliot", "k5", "lcfs", upBps=400),
          "Dev": rows("upall", "sateliot", "k5", "fcfs", upBps=400, devKeepNewest=1), "New": rows("upall", "sateliot", "k5", "newest", upBps=400)}
for nm_, x_ in _upall.items():
    m_, h_ = ci([r["aoi_h"] for r in x_])
    M[f"NsRr{nm_}"], M[f"NsRr{nm_}Ci"] = f(m_, 2), f(h_, 2)
    M[f"NsRr{nm_}Del"] = f(100 * nsm("delivered_share", x_), 1)
    M[f"NsRr{nm_}Delay"] = f(nsm("d_mean", x_), 1)
    M[f"NsRr{nm_}Up"] = f(100 * nsm("uplinked", x_) / nsm("generated", x_), 1)
    M[f"NsRr{nm_}Served"] = str(int(round(919 * nsm("served_share", x_))))
    M[f"NsRr{nm_}Grow"] = f(np.mean([r["aoi_second_h"] / r["aoi_first_h"] for r in x_]), 2)
    M[f"NsRr{nm_}Left"] = f(nsm("left_dev", x_) / 919, 0)
    M[f"NsRr{nm_}BuoyMax"] = f(np.max(np.mean([r["aoi_per_buoy"] for r in x_], axis=0)), 1)
_l = {r["rep"]: r["aoi_h"] for r in _upall["Lcfs"]}
_d = {r["rep"]: r["aoi_h"] for r in _upall["Dev"]}
_lk = sorted(set(_l) & set(_d))
_dif = np.array([_l[k] - _d[k] for k in _lk])
_m, _h = ci(_dif)
M["NsRrLcfsMinusDev"], M["NsRrLcfsMinusDevCi"] = f(_m, 3), f(_h, 3)
M["NsRrLcfsDevPct"] = f(100 * abs(_m) / np.mean(list(_d.values())), 2)
M["NsRrUpRatio"] = f(nsm("uplinked", _upall["Lcfs"]) / nsm("uplinked", _upall["Dev"]), 1)
M["NsRrUpRatioR"] = f(nsm("uplinked", _upall["Lcfs"]) / nsm("uplinked", _upall["Dev"]), 0)
_cmp = [(_sm[400]["aoi_all_h"], nsm("aoi_h", _upall["Fcfs"]))] + [(_sm[b_]["aoi_all_h"], nsm("aoi_h", [r for r in _nsw if r["x_upBps"] == b_])) for b_ in (1000, 1600, 2400)]
M["SameErrMax"] = f(max(100 * abs(a / b - 1) for a, b in _cmp), 1)
# first binary's whole-backlog rule at the same point, all buoys counted
_ex = lambda *a, **k: [r for r in REX if r["part"] == "upall" and r["disc"] == a[0] and r.get("x_upBps") == 400
                       and (bool(k.get("dev")) == ("x_devKeepNewest" in r))]
M["ExNsFcfs"], M["ExNsLcfs"], M["ExNsDev"] = f(nsm("aoi_h", _ex("fcfs")), 2), f(nsm("aoi_h", _ex("lcfs")), 2), f(nsm("aoi_h", _ex("fcfs", dev=1)), 2)
M["ExNsStarved"] = str(int(round(919 * (1 - nsm("served_share", _ex("fcfs"))))))
_f = rows("uplink", "sateliot", "k5", "fcfs", upBps=400)
M["NsUpSlowWorstSat"] = f(max(r["aoi_h"] for r in _f), 2)
M["NsUpSlowWorstPnnSat"] = f(max(r["d_p99"] for r in _f), 1)
M["NsUpNonePnnSat"] = f(nsm("d_p99", rows("uplink", "sateliot", "k5", "fcfs", upBps=0)), 1)
_o = rows("lifeper", "orbcomm", "k5", "fcfs", life="device0")
M["NsLifeSatShareOrbFive"] = f(100 * sum(r["expired_sat"] for r in _o) / sum(r["expired_sat"] + r["expired_dev"] for r in _o), 0)
# Monitoring List (M10), paired with the main runs of the same buoys
for fl in FLEETS:
    t = TAG[fl]
    S_ = {"orbcomm": 14, "kineis": 25, "sateliot": 4}[fl]
    M[f"NsMlAll{t}"] = f(nsm("d_mean", rows("main", fl, "k5", "fcfs")), 2)
    for m_, nm in ((1, "One"), (max(1, S_ // 2), "Half")):
        rr = rows("ml", fl, "k5", "fcfs", monitorList=m_)
        M[f"NsMl{nm}{t}"] = f(nsm("d_mean", rr), 2)
        M[f"NsMlServed{nm}{t}"] = f(100 * nsm("served_share", rr), 1)
# per-device lifetime in ns-3 (M11)
pv, pm_, dv_, dm_ = [], [], [], []
for fl in FLEETS:
    for gw in GWS:
        for mg, store, stored, nm in ((0, pv, dv_, "Replay"), (1, pm_, dm_, "Margin")):
            rr = rows("lifeper", fl, gw, "fcfs", life=f"device{mg}")
            sh = [100 * (r["expired_sat"] + r["expired_dev"]) / r["generated"] for r in rr]
            m, h = ci(sh)
            M[f"NsLife{nm}{TAG[fl]}{TAG[gw]}"], M[f"NsLife{nm}{TAG[fl]}{TAG[gw]}Ci"] = f(m, 2), f(h, 2)
            store.append(m)
            ok = []
            for r in rr:
                per = r["generated"] / r["n"]
                # buoys that received nothing at all never expire by this count unless they generated, so all count
                ok += [100 * (1 - x / per) <= 1.0 for x in r["delivered_per_buoy"]]
            stored.append(100 * np.mean(ok))
M["NsLifeMarginUpperMax"] = f(max(float(M[f"NsLifeMargin{TAG[fl]}{TAG[gw]}"]) + float(M[f"NsLifeMargin{TAG[fl]}{TAG[gw]}Ci"]) for fl in FLEETS for gw in GWS), 2)
M["NsLifeReplayLo"], M["NsLifeReplayHi"] = f(min(pv), 2), f(max(pv), 2)
M["NsLifeMarginLo"], M["NsLifeMarginHi"] = f(min(pm_), 2), f(max(pm_), 2)
M["NsLifeReplayMean"], M["NsLifeMarginMean"] = f(np.mean(pv), 2), f(np.mean(pm_), 2)
M["NsLifeDevReplayLo"], M["NsLifeDevReplayHi"] = f(min(dv_), 0), f(max(dv_), 0)
M["NsLifeDevMarginLo"], M["NsLifeDevMarginHi"] = f(min(dm_), 0), f(max(dm_), 0)
# frozen out-of-sample test of the gateway what-if
GF = []
for fl in FLEETS:
    pth = os.path.join(RV, f"whatif_gw_frozen_{fl}.json")
    if os.path.exists(pth):
        GF += json.load(open(pth))
if GF:
    def _err(rr, k):
        return np.array([abs(100 * (o[k] / o["truth"] - 1)) for o in rr])
    lo, ad = [o for o in GF if o["kind"] == "loo"], [o for o in GF if o["kind"] == "add"]
    for rr, nm in ((lo, "Loo"), (ad, "Add")):
        M[f"GwFz{nm}N"] = str(len(rr))
        for k, kn in (("pred", ""), ("pred_fleetavg", "Avg"), ("none", "None")):
            e_ = _err(rr, k)
            M[f"GwFz{nm}{kn}Med"], M[f"GwFz{nm}{kn}Max"] = f(np.median(e_), 0), f(e_.max(), 0)
            M[f"GwFz{nm}{kn}Ninety"] = f(np.percentile(e_, 90), 0)
        M[f"GwFz{nm}OverTwentyFive"] = f(100 * np.mean(_err(rr, "pred") > 25), 0)
        M[f"GwFz{nm}OverFifty"] = f(100 * np.mean(_err(rr, "pred") > 50), 0)
        M[f"GwFz{nm}Beats"] = str(int(np.sum(_err(rr, "pred") < _err(rr, "none"))))
    M["GwFzFleets"] = str(len({o["fleet"] for o in GF}))

assert not [k for k, v in M.items() if "?" in str(v)], "placeholder in a macro"
assert not os.path.exists(os.path.join(HERE, "..", "macros_pending.tex")), "macros_pending.tex must go before macros are final"
with open(os.path.join(HERE, "..", "macros_p6.tex"), "w") as fo:
    fo.write("% written by sim/make_macros.py; do not edit\n")
    for k in sorted(M):
        fo.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(len(M), "macros")
