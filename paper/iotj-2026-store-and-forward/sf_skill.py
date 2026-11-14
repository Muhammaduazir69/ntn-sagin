#!/usr/bin/env python3
"""What can rank element sets: placebo control and intervals (Nov 2026, round 3).

The reference in every test here is a propagated plan from later-issued element sets (data_raw/issued_2026-11-03), not
a measured delay. Predictor and reference share the first-available code, so a small error can mean insensitivity.

Part held     week 0, all buoys, DRAWS phase draws per buoy per predictor: reference (later sets), floor (later sets,
              independent draws), fwd (sets of 30 Oct), replay (sets of 30 Oct, previous week), model (out/rev4/replay_*).
              Per-buoy rows saved; paired bootstrap over buoys. Timing per upload of the reference, predicted three ways:
              fwd (older sets), lastweek (same instant of the previous week's plan), residual (the satellite's mean
              gateway residual from last week, delivery time only).
Part placebo  reference as above; forecast from the SAME later sets with each satellite moved along track by a stated
              distance with random sign, or its mean anomaly drawn at random, or mean anomaly and node drawn at random.
              Same metrics, first NP buoys, NREP draws of the displacement.

    python3 sf_skill.py orbcomm k5 held|placebo
Output out/rev4/skill_held_<fleet>_<gw>.json, out/rev4/skill_placebo_<fleet>_<gw>.json (resumable per row).
"""
import json
import math
import os
import sys

import numpy as np
from sgp4.api import Satrec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402
from sf_heldout import by_id  # noqa: E402
from sf_replay import plan, first_available, DT  # noqa: E402
from sf_timers import return_leg  # noqa: E402

BIG = 10 ** 12
RV = os.path.join(HERE, "out", "rev4")
PER, HORIZON = 900.0, 4 * 86400.0
DRAWS, NP, NREP, STRIDE = 10, 60, 3, 8
LEVELS = (100, 500, 2000, 5000, 20000, "phase", "node")


def gens(rng, n):
    return np.concatenate([np.arange(rng.random() * PER, HORIZON, PER) for _ in range(n)])


def mean_delay(P, gr, vis, xyz, rng, nt, n):
    gen = gens(rng, n)
    _, k1, up, dl = first_available(P, vis, gr, xyz, gen, nt)
    g = dl < BIG
    return (up[g] * DT - gen[g]) + (dl[g] - up[g]) * DT, k1[g], up[g], dl[g]


def hplan(P, v):
    return [sf_mc.intervals(v[k]) for k in range(P.shape[0])]


def vis_of(P, xyz):
    return F.visibility(P, xyz[None, :], 20.0)[:, :, 0]


def lookups(k1, up, dl, gr_t, dev_t, gr_f, dev_f):
    """Absolute error (s) of delivery time and return leg on the reference's uploads, using another plan's contacts."""
    k8, u8, d8 = k1[::STRIDE], up[::STRIDE], dl[::STRIDE]
    d_f = np.array([sf_mc.next_in(*gr_f[k], np.array([u]))[0] for k, u in zip(k8, u8)])
    ok = d_f < BIG
    he = np.abs((d_f[ok] - d8[ok]) * DT)
    c_t = return_leg(dev_t, gr_t, d8, range(len(gr_t)))
    c_f = np.full(len(d8), BIG, dtype=np.int64)
    c_f[ok] = return_leg(dev_f, gr_f, d_f[ok], range(len(gr_t)))
    b = (c_t < BIG) & (c_f < BIG)
    return he, np.abs((c_f[b] - c_t[b]) * DT)


def q(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)), p99=float(np.percentile(x, 99)),
                mean=float(x.mean()))


def err(a, b):
    return 100 * (a.mean() / b.mean() - 1)


def boot(fn, n, rng, B=2000):
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def held(fl, gw):
    path = os.path.join(RV, f"skill_held_{fl}_{gw}.json")
    if os.path.exists(path):
        return
    old = by_id(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    new = by_id(os.path.join(HERE, "data_raw", "issued_2026-11-03", f"{fl}.tle"))
    ids = [i for i in old if i in new]
    gws = sf_mc.load_gateways(gw)
    pred = json.load(open(os.path.join(RV, f"replay_{fl}_{gw}_w0.json")))
    t, Pn, vn = plan([new[i] for i in ids], gws, 0.0)
    _, Po, vo = plan([old[i] for i in ids], gws, 0.0)
    _, Pp, vp = plan([old[i] for i in ids], gws, -7.0)
    nt, S = len(t), len(ids)
    rn, ro, rp = hplan(Pn, vn), hplan(Po, vo), hplan(Pp, vp)
    res = []
    for k in range(S):
        _, g = F.runs(vp[k])
        g = g * DT
        res.append(np.sum(g ** 2) / (2 * np.sum(g)) if len(g) else np.nan)
    res = np.array(res)
    rng = np.random.default_rng(2026)
    rows, pools = [], {"ref": [], "fwd": [], "replay": []}
    he_f, we_f, he_l, we_l, he_r = [], [], [], [], []
    for o in pred["devices"]:     # list order, so co-located stations stay apart
        xyz = F.geodetic_to_ecef(np.array([o["lat"]]), np.array([o["lon"]]))[0]
        vn_, vo_, vp_ = vis_of(Pn, xyz), vis_of(Po, xyz), vis_of(Pp, xyz)
        D, k1, up, dl = mean_delay(Pn, rn, vn_, xyz, rng, nt, DRAWS)
        Dfl, *_ = mean_delay(Pn, rn, vn_, xyz, rng, nt, DRAWS)
        Df, *_ = mean_delay(Po, ro, vo_, xyz, rng, nt, DRAWS)
        Dr, *_ = mean_delay(Pp, rp, vp_, xyz, rng, nt, DRAWS)
        rows.append([float(x.mean()) for x in (D, Dfl, Df, Dr)] + [o["model_A"] + o["model_H"]])
        pools["ref"].append(D[::DRAWS]); pools["fwd"].append(Df[::DRAWS]); pools["replay"].append(Dr[::DRAWS])
        dn = [sf_mc.intervals(vn_[k]) for k in range(S)]
        a, b = lookups(k1, up, dl, rn, dn, ro, [sf_mc.intervals(vo_[k]) for k in range(S)])
        he_f.append(a); we_f.append(b)
        a, b = lookups(k1, up, dl, rn, dn, rp, [sf_mc.intervals(vp_[k]) for k in range(S)])
        he_l.append(a); we_l.append(b)
        k8, u8, d8 = k1[::STRIDE], up[::STRIDE], dl[::STRIDE]
        he_r.append(np.abs(res[k8] - (d8 - u8) * DT))
    a = np.array(rows)
    n = len(a)
    br = np.random.default_rng(7)
    E = {}
    for c, nm in ((1, "floor"), (2, "fwd"), (3, "replay"), (4, "model")):
        E[nm] = dict(err=float(err(a[:, c], a[:, 0])), ci=boot(lambda i, c=c: err(a[i, c], a[i, 0]), n, br),
                     med=float(100 * np.median(np.abs(a[:, c] / a[:, 0] - 1))))
    gap = lambda i: abs(err(a[i, 3], a[i, 0])) - abs(err(a[i, 2], a[i, 0]))
    gapm = lambda i: np.median(np.abs(a[i, 3] / a[i, 0] - 1)) * 100 - np.median(np.abs(a[i, 2] / a[i, 0] - 1)) * 100
    out = dict(fleet=fl, gw=gw, n=n, draws=DRAWS, E=E, gap_fleet=dict(v=float(gap(np.arange(n))), ci=boot(gap, n, br)),
               gap_med=dict(v=float(gapm(np.arange(n))), ci=boot(gapm, n, br)),
               p99={k: float(np.percentile(np.concatenate(v), 99)) for k, v in pools.items()}, model_p99=pred["quant"]["pred_p99"],
               timing=dict(H=dict(fwd=q(np.concatenate(he_f)), lastweek=q(np.concatenate(he_l)), residual=q(np.concatenate(he_r))),
                           W=dict(fwd=q(np.concatenate(we_f)), lastweek=q(np.concatenate(we_l)))),
               w_within_60=float(np.mean(np.concatenate(we_f) <= 60.0)),
               rows=[[round(x, 1) for x in r] for r in rows])
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, path)
    print(fl, gw, "HELD", {k: (round(v["err"], 3), [round(c, 3) for c in v["ci"]], round(v["med"], 2)) for k, v in E.items()},
          "gap", round(out["gap_fleet"]["v"], 2), out["gap_fleet"]["ci"], {k: {a_: round(b_["median"]) for a_, b_ in v.items()} for k, v in out["timing"].items()}, flush=True)


def tle_lines(path):
    L = [x.rstrip() for x in open(path) if x.strip()]
    return {L[i + 1][2:7]: (L[i + 1], L[i + 2]) for i in range(0, len(L) - 2, 3)}


def moved(l1, l2, sat, level, rng):
    if level in ("phase", "node"):
        M = rng.random() * 360.0
    else:
        a = float(sat.a) * 6378.135
        M = (float(l2[43:51]) + math.degrees(level / a) * rng.choice([-1, 1])) % 360.0
    l2b = l2[:43] + f"{M:8.4f}" + l2[51:]
    if level == "node":
        l2b = l2b[:17] + f"{rng.random() * 360.0:8.4f}" + l2b[25:]
    return Satrec.twoline2rv(l1, l2b)


def placebo(fl, gw):
    path = os.path.join(RV, f"skill_placebo_{fl}_{gw}.json")
    done = json.load(open(path)) if os.path.exists(path) else dict(rows=[])
    tle = os.path.join(HERE, "data_raw", "issued_2026-11-03", f"{fl}.tle")
    new, lines = by_id(tle), tle_lines(tle)
    ids = list(new)
    gws = sf_mc.load_gateways(gw)
    dev = json.load(open(os.path.join(RV, f"replay_{fl}_{gw}_w0.json")))["devices"][:NP]
    XY = [F.geodetic_to_ecef(np.array([o["lat"]]), np.array([o["lon"]]))[0] for o in dev]
    t, Pn, vn = plan([new[i] for i in ids], gws, 0.0)
    nt, S = len(t), len(ids)
    rn = hplan(Pn, vn)
    rng0 = np.random.default_rng(11)
    ref, ev = [], []
    for xyz in XY:
        v_ = vis_of(Pn, xyz)
        D, k1, up, dl = mean_delay(Pn, rn, v_, xyz, rng0, nt, DRAWS // 2)
        ref.append(D.mean())
        ev.append((k1, up, dl, [sf_mc.intervals(v_[k]) for k in range(S)]))
    ref = np.array(ref)
    if not done["rows"]:
        fl_ = np.array([mean_delay(Pn, rn, vis_of(Pn, xyz), xyz, rng0, nt, DRAWS // 2)[0].mean() for xyz in XY])
        done["floor"] = dict(err=float(err(fl_, ref)), med=float(100 * np.median(np.abs(fl_ / ref - 1))))
    have = {(str(r["level"]), r["rep"]) for r in done["rows"]}
    for level in LEVELS:
        for rep in range(NREP):
            if (str(level), rep) in have:
                continue
            seed = sum(ord(c) for c in f"{fl}{gw}{level}") * 100 + rep
            rng = np.random.default_rng(seed)
            _, Pf, vf = plan([moved(*lines[i], new[i], level, rng) for i in ids], gws, 0.0)
            rf = hplan(Pf, vf)
            f, he, we = [], [], []
            for xyz, (k1, up, dl, dn) in zip(XY, ev):
                v_ = vis_of(Pf, xyz)
                f.append(mean_delay(Pf, rf, v_, xyz, rng, nt, DRAWS // 2)[0].mean())
                a, b = lookups(k1, up, dl, rn, dn, rf, [sf_mc.intervals(v_[k]) for k in range(S)])
                he.append(a); we.append(b)
            f = np.array(f)
            done["rows"].append(dict(level=level, rep=rep, err=float(err(f, ref)), med=float(100 * np.median(np.abs(f / ref - 1))),
                                     H=q(np.concatenate(he)), W=q(np.concatenate(we))))
            tmp = path + ".tmp"
            json.dump(done, open(tmp, "w"))
            os.replace(tmp, path)
            r = done["rows"][-1]
            print(fl, gw, level, rep, f"err {r['err']:.2f} med {r['med']:.2f} H med {r['H']['median']:.0f} p99 {r['H']['p99']:.0f} W med {r['W']['median']:.0f} p99 {r['W']['p99']:.0f}", flush=True)


if __name__ == "__main__":
    os.nice(10)
    {"held": held, "placebo": placebo}[sys.argv[3]](sys.argv[1], sys.argv[2])
