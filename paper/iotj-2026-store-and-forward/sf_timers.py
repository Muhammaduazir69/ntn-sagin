#!/usr/bin/env python3
"""Release 19 quantities with the semantics of TS 23.401 cl. 4.13.9.1 and TS 24.301 cl. 4.11.5 (Nov 2026 revision).

Event: a device uploads to satellite k at step t (first-available S&F, events of out/sf_<tag>.npz).
  Estimated S&F UL delivery time: hold of the serving satellite for that upload. Under (A1) a lookup, H_k(t).
    Compared with: the mean residual of satellite k's gateway gaps from last week (no position needed), and the
    former mapping E[A] + E[H] of the device.
  S&F wait time: time from t until the device meets a satellite on its Monitoring List that has seen a gateway
    since the data reached the ground, W = min_m { c_m } - t, f_m first gateway contact of m at or after t + H,
    c_m first contact of device with m at or after f_m. The return leg is R = W - H; R >= forward wait of U_d.
  Limits of the signaling: 2-octet field in seconds (TS 24.301), UE maximum for a first reject, e.g. 90 min.
  First re-attempt: the UE waits tau, then meets the first satellite in view; success if it has been fed.
    tau policies: fallback uniform 30 to 60 min (TS 24.301), per-device 95th percentile from last week's replay,
    the lookup itself.

    python3 sf_timers.py orbcomm k5 0        Output out/rev4/timers_<tag>.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402
from sf_replay import plan, DT  # noqa: E402

FIELD_MAX = 65535.0
UE_MAX = 90 * 60.0
BIG = 10 ** 12


def return_leg(dev_runs, gw_runs, t_del, members):
    """Sample index of the first contact with a member satellite that has been fed since t_del."""
    best = np.full(len(t_del), BIG, dtype=np.int64)
    for m in members:
        f = sf_mc.next_in(*gw_runs[m], t_del)
        c = np.full(len(t_del), BIG, dtype=np.int64)
        ok = f < BIG
        c[ok] = sf_mc.next_in(*dev_runs[m], f[ok])
        best = np.minimum(best, c)
    return best


def events_on(P, vis_gw, gw_runs, xyz, nt, per, rng):
    """Upload events of one device on a plan: first-available rule, generation every per seconds."""
    from sf_replay import first_available
    vis_d = F.visibility(P, xyz[None, :], 20.0)[:, :, 0]
    gen = np.arange(rng.random() * per, (nt - 1) * DT - 2 * 86400.0, per)
    gi, k1, up, dl = first_available(P, vis_d, gw_runs, xyz, gen, nt)
    good = dl < BIG
    u = np.unique(np.c_[k1[good], up[good], dl[good]], axis=0)
    return vis_d, u


def main(fl, gw, w):
    tag = f"{fl}_{gw}_w{w}"
    z = np.load(os.path.join(HERE, "out", f"sf_{tag}.npz"))
    an = json.load(open(os.path.join(HERE, "out", f"an_{tag}.json")))
    amap = {(o["lat"], o["lon"]): o for o in an["devices"]}
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    gws = sf_mc.load_gateways(gw)
    S = len(sats)
    t_t, P_t, vg_t = plan(sats, gws, 7.0 * w)
    t_p, P_p, vg_p = plan(sats, gws, 7.0 * w - 7.0)
    nt = len(t_t)
    gr_t = [sf_mc.intervals(vg_t[k]) for k in range(S)]
    gr_p = [sf_mc.intervals(vg_p[k]) for k in range(S)]
    res_sat = []
    for k in range(S):
        _, g = F.runs(vg_p[k])
        g = g * DT
        res_sat.append((1 - vg_p[k].mean()) * np.sum(g ** 2) / (2 * np.sum(g)) / max(1 - vg_p[k].mean(), 1e-9) if len(g) else np.nan)
    res_sat = np.array(res_sat)
    buoys = z["buoys"]
    rng = np.random.default_rng(77 + w)
    rows = dict(H=[], W=[], R=[], W_half=[], W_self=[], res=[], former=[], ok_fallback=[], ok_p95=[], blocked=[], tau95=[], Wprior_p99=[], Wtest_p99=[], lat=[], A=[], Rmean_dev=[])
    half = rng.choice(S, size=max(1, S // 2), replace=False).tolist()
    for j, (lat, lon) in enumerate(buoys):
        xyz = F.geodetic_to_ecef(np.array([lat]), np.array([lon]))[0]
        o = amap.get((float(lat), float(lon)))
        if o is None:
            continue
        vis_d, u = events_on(P_t, vg_t, gr_t, xyz, nt, float(z["period"]), rng)
        if len(u) == 0:
            continue
        dev_runs = [sf_mc.intervals(vis_d[k]) for k in range(S)]
        k_e, t_e, d_e = u.T
        W_c = return_leg(dev_runs, gr_t, d_e, range(S))
        ok = W_c < BIG
        k_e, t_e, d_e, W_c = k_e[ok], t_e[ok], d_e[ok], W_c[ok]
        if len(t_e) == 0:
            continue
        W = (W_c - t_e) * DT
        H = (d_e - t_e) * DT
        Wh = return_leg(dev_runs, gr_t, d_e, half)
        Wh = np.where(Wh < BIG, (Wh - t_e) * DT, np.nan)
        Ws = np.array([sf_mc.next_in(*dev_runs[k], np.array([sf_mc.next_in(*gr_t[k], np.array([d]))[0]]))[0] for k, d in zip(k_e, d_e)])
        Ws = np.where(Ws < BIG, (Ws - t_e) * DT, np.nan)
        # last week's replay at the same device, per-device 95th percentile as a timer
        vis_p, up = events_on(P_p, vg_p, gr_p, xyz, nt, float(z["period"]), rng)
        tau95 = np.nan
        Wp = np.array([])
        if len(up):
            dev_p = [sf_mc.intervals(vis_p[k]) for k in range(S)]
            cp = return_leg(dev_p, gr_p, up[:, 2], range(S))
            Wp = (cp[cp < BIG] - up[:, 1][cp < BIG]) * DT
            if len(Wp):
                tau95 = float(np.percentile(Wp, 95))
        # first re-attempt after tau: the first satellite in view, success if it has been fed since delivery
        def attempt(tau):
            ts = t_e + np.ceil(tau / DT).astype(np.int64)
            nx = np.stack([sf_mc.next_in(*dev_runs[k], ts) for k in range(S)])
            a = nx.argmin(axis=0)
            ta = nx[a, np.arange(len(ts))]
            f = np.stack([sf_mc.next_in(*gr_t[k], d_e) for k in range(S)])
            fa = f[a, np.arange(len(ts))]
            return (ta < BIG) & (fa <= ta)
        rows["ok_fallback"] += attempt(rng.uniform(1800.0, 3600.0, size=len(t_e))).tolist()
        if np.isfinite(tau95):
            rows["ok_p95"] += attempt(np.full(len(t_e), tau95)).tolist()
        # contacts the UE sits out while the timer runs at the lookup value
        U = vis_d.any(axis=0)
        st, _ = sf_mc.intervals(U)
        rows["blocked"] += np.maximum(np.searchsorted(st, W_c, side="left") - np.searchsorted(st, t_e, side="right"), 0).tolist()  # not reported; sf_timer_gate.py measures the cost
        rows["H"] += H.tolist(); rows["W"] += W.tolist(); rows["R"] += (W - H).tolist()
        rows["W_half"] += Wh.tolist(); rows["W_self"] += Ws.tolist()
        rows["res"] += res_sat[k_e].tolist()
        rows["former"] += [o["EA_semi"] + o["EH_semi"]] * len(H)
        rows["tau95"].append(tau95)
        rows["A"].append(o["EA_semi"])
        rows["Rmean_dev"].append(float(np.mean(W - H)))
        rows["lat"].append(float(lat))
        rows["Wtest_p99"].append(float(np.percentile(W, 99)))
        rows["Wprior_p99"].append(float(np.percentile(Wp, 99)) if len(Wp) else np.nan)
    H, W, R = (np.array(rows[k]) for k in ("H", "W", "R"))
    res, former = np.array(rows["res"]), np.array(rows["former"])
    A, Rd = np.array(rows["A"]), np.array(rows["Rmean_dev"])
    wt, wp = np.array(rows["Wtest_p99"]), np.array(rows["Wprior_p99"])
    wf = np.isfinite(wp) & (wt > 0)
    out = dict(n_events=int(len(W)), H_mean=float(H.mean()), H_p99=float(np.percentile(H, 99)), H_zero=float(np.mean(H == 0)),
               W_mean=float(W.mean()), W_p50=float(np.median(W)), W_p95=float(np.percentile(W, 95)), W_p99=float(np.percentile(W, 99)),
               W_max=float(W.max()), R_mean=float(R.mean()), A_mean=float(A.mean()), R_over_A=float(Rd.mean() / A.mean()),
               R_lowerbound_ratio_med=float(np.median(Rd / A)),
               W_half_mean=float(np.nanmean(rows["W_half"])), W_half_miss=float(np.mean(~np.isfinite(rows["W_half"]))),
               W_self_mean=float(np.nanmean(rows["W_self"])), W_self_miss=float(np.mean(~np.isfinite(rows["W_self"]))),
               W_half_p99=float(np.nanpercentile(rows["W_half"], 99)), W_self_p99=float(np.nanpercentile(rows["W_self"], 99)),
               over90=float(np.mean(W > UE_MAX)), over_field=float(np.mean(W > FIELD_MAX)),
               self_over_field=float(np.nanmean(np.array(rows["W_self"]) > FIELD_MAX)),
               mae_res=float(np.mean(np.abs(res - H))), mae_former=float(np.mean(np.abs(former - H))), h_lookup_mean=float(H.mean()),
               ok_fallback=float(np.mean(rows["ok_fallback"])), ok_p95=float(np.mean(rows["ok_p95"])),
               blocked_mean=float(np.mean(rows["blocked"])), wp99_replay_med=float(np.median(np.abs(wp[wf] / wt[wf] - 1))))
    json.dump(out, open(os.path.join(HERE, "out", "rev4", f"timers_{tag}.json"), "w"), indent=1)
    print(tag, {k: (round(v / 3600, 2) if k[:1] in "HWR" and "miss" not in k and "ratio" not in k and "over" not in k else round(v, 3)) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]))
