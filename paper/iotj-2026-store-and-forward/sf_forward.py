#!/usr/bin/env python3
"""Forward propagation of the element sets as a predictor of the delay (Nov 2026, round 2).

Under (A1) the operator can propagate the coming week at the cost of the plan the model and replay already need.

Part week  on the same buoys as the Monte Carlo week and with the same first-available code as sf_replay.py, a fresh
           draw of update phases on the plan of the TEST week, from the same element sets as the truth. Its error can
           only be phase noise, so it measures the sampling floor of the comparison, not forecasting skill.
Part held  the real test. Prediction: propagate the sets of 30 Oct over a target week. Target: the same week from the
           sets issued later (data_raw/issued_2026-11-03). w = 0 is the held-out test of sf_heldout.py. For w = 1, 2
           the later sets are themselves extrapolated, so that is a disagreement between two forecasts, not an error.
           Also, per upload of the target week, the error of the two lookups made with the older sets: delivery time
           H_k and return leg W_k (all satellites listed, as sf_timers.py).

    python3 sf_forward.py orbcomm k5 [week|held]
Output out/rev4/forward_<fleet>_<gw>_w<w>.json, out/rev4/forward_held_<fleet>_<gw>_w<w>.json (small, resumable).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402
from sf_heldout import by_id  # noqa: E402
from sf_replay import plan, first_available, DT  # noqa: E402
from sf_timers import return_leg  # noqa: E402

BIG = 10 ** 12
RV = os.path.join(HERE, "out", "rev4")
PER = 900.0
HORIZON = 4 * 86400.0


def delays(P, gr, xyz, rng, nt):
    """One device's updates on a plan: delay legs and the upload events, as sf_replay.first_available."""
    vis = F.visibility(P, xyz[None, :], 20.0)[:, :, 0]
    gen = np.arange(rng.random() * PER, HORIZON, PER)
    _, k1, up, dl = first_available(P, vis, gr, xyz, gen, nt)
    g = dl < BIG
    A = up[g] * DT - gen[g]
    H = (dl[g] - up[g]) * DT
    return vis, A, H, k1[g], up[g], dl[g]


def summary(a):
    e = lambda c: float(100 * (a[:, c].mean() / a[:, 0].mean() - 1))
    m = lambda c: float(100 * np.median(np.abs(a[:, c] / a[:, 0] - 1)))
    return e, m


def week(fl, gw, w):
    path = os.path.join(RV, f"forward_{fl}_{gw}_w{w}.json")
    if os.path.exists(path):
        return
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    gws = sf_mc.load_gateways(gw)
    z = np.load(os.path.join(HERE, "out", f"sf_{fl}_{gw}_w{w}.npz"))
    devs = json.load(open(os.path.join(RV, f"replay_{fl}_{gw}_w{w}.json")))["devices"]
    t, P, v = plan(sats, gws, 7.0 * w)
    nt = len(t)
    gr = [sf_mc.intervals(v[k]) for k in range(len(sats))]
    rng = np.random.default_rng(77 + w)
    out, pool, tc = [], [], 0.0
    for o in devs:     # list order, so co-located stations stay apart
        lat, lon = o["lat"], o["lon"]
        xyz = F.geodetic_to_ecef(np.array([lat]), np.array([lon]))[0]
        t0 = time.perf_counter()
        _, A, H, *_ = delays(P, gr, xyz, rng, nt)
        tc += time.perf_counter() - t0
        D = A + H
        pool.append(D)
        out.append(dict(lat=float(lat), lon=float(lon), fwd_D=float(D.mean()), fwd_A=float(A.mean()), fwd_H=float(H.mean()),
                        fwd_p99=float(np.percentile(D, 99))))
    pool = np.concatenate(pool)
    res = dict(devices=out, pool=dict(mean=float(pool.mean()), p99=float(np.percentile(pool, 99))), s_per_dev=tc / len(out))
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, path)
    a = np.array([(o["sim_D"], d["fwd_D"], o["replay_D"], o["model_A"] + o["model_H"]) for o, d in zip(devs, out)])
    e, m = summary(a)
    print(fl, gw, w, f"fleet fwd {e(1):.2f} rep {e(2):.2f} mod {e(3):.2f} | med fwd {m(1):.2f} rep {m(2):.2f} mod {m(3):.2f}", flush=True)


def held(fl, gw, w):
    path = os.path.join(RV, f"forward_held_{fl}_{gw}_w{w}.json")
    if os.path.exists(path):
        return
    old = by_id(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    new = by_id(os.path.join(HERE, "data_raw", "issued_2026-11-03", f"{fl}.tle"))
    ids = [i for i in old if i in new]
    S = len(ids)
    gws = sf_mc.load_gateways(gw)
    z = np.load(os.path.join(HERE, "out", f"sf_{fl}_{gw}_w{w}.npz"))
    devs = json.load(open(os.path.join(RV, f"replay_{fl}_{gw}_w{w}.json")))["devices"]
    t, Pn, vn = plan([new[i] for i in ids], gws, 7.0 * w)
    _, Po, vo = plan([old[i] for i in ids], gws, 7.0 * w)
    nt = len(t)
    rn = [sf_mc.intervals(vn[k]) for k in range(S)]
    ro = [sf_mc.intervals(vo[k]) for k in range(S)]
    rng_t, rng_f = np.random.default_rng(5 + w), np.random.default_rng(6 + w)
    rows, pt, pf, pr, herr, werr, wrel = [], [], [], [], [], [], []
    for o in devs:     # list order, so co-located stations stay apart
        lat, lon = o["lat"], o["lon"]
        xyz = F.geodetic_to_ecef(np.array([lat]), np.array([lon]))[0]
        visn, An, Hn, k1, up, dl = delays(Pn, rn, xyz, rng_t, nt)
        viso, Af, Hf, *_ = delays(Po, ro, xyz, rng_f, nt)
        Dn, Df = An + Hn, Af + Hf
        pt.append(Dn)
        pf.append(Df)
        rows.append((Dn.mean(), Df.mean(), o["replay_D"], o["model_A"] + o["model_H"]))
        # the two lookups on the target's uploads, every eighth upload, with the older sets in place of the later ones
        k8, u8, d8 = k1[::8], up[::8], dl[::8]
        dev_n = [sf_mc.intervals(visn[k]) for k in range(S)]
        dev_o = [sf_mc.intervals(viso[k]) for k in range(S)]
        d_old = np.array([sf_mc.next_in(*ro[k], np.array([u]))[0] for k, u in zip(k8, u8)])
        c_new = return_leg(dev_n, rn, d8, range(S))
        ok = d_old < BIG
        c_old = np.full(len(d8), BIG, dtype=np.int64)
        c_old[ok] = return_leg(dev_o, ro, d_old[ok], range(S))
        herr += np.abs((d_old[ok] - d8[ok]) * DT).tolist()
        both = (c_new < BIG) & (c_old < BIG)
        werr += np.abs((c_old[both] - c_new[both]) * DT).tolist()
        wrel += ((c_new[both] - u8[both]) * DT).tolist()
    a = np.array(rows)
    e, m = summary(a)
    pt, pf = np.concatenate(pt), np.concatenate(pf)
    wa = np.array(werr)
    res = dict(fleet=fl, gw=gw, w=w, n=len(rows),
               err=dict(fwd=e(1), replay=e(2), model=e(3)), med=dict(fwd=m(1), replay=m(2), model=m(3)),
               p99=dict(truth=float(np.percentile(pt, 99)), fwd=float(np.percentile(pf, 99))),
               h_err_s=dict(median=float(np.median(herr)), p99=float(np.percentile(herr, 99))),
               w_err_s=dict(median=float(np.median(wa)), p90=float(np.percentile(wa, 90)), p99=float(np.percentile(wa, 99)),
                            mean=float(wa.mean()), share_le_60s=float(np.mean(wa <= 60.0)), w_mean=float(np.mean(wrel))))
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    print(fl, gw, "HELD", w, res["err"], res["med"], res["h_err_s"], res["w_err_s"], flush=True)


if __name__ == "__main__":
    os.nice(10)
    fl, gw = sys.argv[1:3]
    what = sys.argv[3] if len(sys.argv) > 3 else "both"
    for w in (0, 1, 2):
        if what in ("week", "both"):
            week(fl, gw, w)
    for w in (0, 1, 2):
        if what in ("held", "both"):
            held(fl, gw, w)
