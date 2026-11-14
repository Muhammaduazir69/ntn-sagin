#!/usr/bin/env python3
"""Plan replay as a rival predictor (Nov 2026 revision), and the measured legs of the delay.

For each scenario and test week, on the same buoys as the Monte Carlo out/sf_<tag>.npz:
  replay  next-contact lookup per update on the PREVIOUS week's contact plan (first-available rule),
          the same input the renewal model takes from that plan
  model   Propositions 1 and 2 (read from out/an_<tag>.json)
  track   Proposition 2 with the residual conditioned on where the satellite is on its track when it
          crosses the device's latitude (hold table per satellite: latitude band, direction)
Also the compute time of the model's moments and of the replay on the same plan, and split of A and H.

    python3 sf_replay.py orbcomm k5 0
Output out/rev4/replay_<tag>.json
"""
import json
import os
import sys
import time

import numpy as np
from sgp4.api import jday

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402

DT = 20.0
LATBIN = 5.0


def plan(sats, gws, start_day, days=7.0, gw_mask=10.0):
    jd0, fr0 = jday(2026, 10, 30, 0, 0, 0)
    fr0 += start_day
    t = np.arange(0, days * 86400.0, DT)
    P = F.ecef_positions(sats, jd0, fr0, t)
    gw_xyz = F.geodetic_to_ecef(np.array([g[1] for g in gws]), np.array([g[2] for g in gws]))
    return t, P, F.visibility(P, gw_xyz, gw_mask).any(axis=2)


def first_available(P, vis_d, gw_runs, dev_xyz, gen, nt):
    """Upload and delivery sample of each update under the first-available rule (as sf_mc.run)."""
    S = P.shape[0]
    gi = np.ceil(gen / DT).astype(np.int64)
    dev_runs = [sf_mc.intervals(vis_d[k]) for k in range(S)]
    up = np.stack([sf_mc.next_in(*dev_runs[k], gi) for k in range(S)])
    dl = np.stack([sf_mc.next_in(*gw_runs[k], np.minimum(up[k], nt)) for k in range(S)])
    dl = np.where(up >= 10 ** 12, 10 ** 12, dl)
    umin = up.min(axis=0)
    tie = up == umin[None, :]
    idx = np.where(umin < nt, umin, 0)
    los = P[:, idx, :] - dev_xyz[None, None, :]
    sin_e = (los @ (dev_xyz / np.linalg.norm(dev_xyz))) / np.linalg.norm(los, axis=2)
    k1 = np.argmax(np.where(tie, sin_e, -2.0), axis=0)
    m = np.arange(len(gi))
    return gi, k1, up[k1, m], dl[k1, m]


def hold_table(P, vis_gw, gw_runs, nt):
    """Mean time to the next gateway contact per satellite, by latitude band and direction of flight,
    over the samples outside a gateway contact: the residual conditioned on the point of the track."""
    S = P.shape[0]
    R = np.linalg.norm(P, axis=2)
    lat = np.degrees(np.arcsin(P[:, :, 2] / R))
    asc = np.diff(P[:, :, 2], axis=1, append=P[:, -1:, 2]) >= 0
    nb = int(180 / LATBIN)
    tab = np.full((S, 2, nb), np.nan)
    idx = np.arange(nt)
    for k in range(S):
        nx = sf_mc.next_in(*gw_runs[k], idx)
        ok = (nx < nt) & ~vis_gw[k]
        res = (nx - idx) * DT
        b = np.clip(((lat[k] + 90) / LATBIN).astype(int), 0, nb - 1)
        for d in (0, 1):
            sel = ok & (asc[k] == bool(d))
            s = np.bincount(b[sel], weights=res[sel], minlength=nb)
            c = np.bincount(b[sel], minlength=nb)
            with np.errstate(invalid="ignore", divide="ignore"):
                tab[k, d] = s / c
    return tab


def main(fl, gw, w):
    tag = f"{fl}_{gw}_w{w}"
    z = np.load(os.path.join(HERE, "out", f"sf_{tag}.npz"))
    an = json.load(open(os.path.join(HERE, "out", f"an_{tag}.json")))
    a_dev = an["devices"]
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    gws = sf_mc.load_gateways(gw)
    t, P, vis_gw = plan(sats, gws, 7.0 * w - 7.0)
    nt, S = len(t), len(sats)
    gw_runs = [sf_mc.intervals(vis_gw[k]) for k in range(S)]
    tab = hold_table(P, vis_gw, gw_runs, nt)
    buoys = z["buoys"]
    dev, gen_all = z["dev"], z["gen"]
    up_all, del_all = z["sf1_up"], z["sf1_del"]
    rng = np.random.default_rng(1000 + w)
    per = float(z["period"])
    out, pool, tA, tR = [], [], 0.0, 0.0
    h = np.mean([(F.MU / (s.no_kozai / 60) ** 2) ** (1 / 3) - F.RE for s in sats])
    psi = np.degrees(F.cap_angle(h, np.radians(20.0)))
    # every buoy of a_dev was kept by analyse(); match by coordinates
    for j, (lat, lon) in enumerate(buoys):
        sel = dev == j
        ok = sel & (del_all < 10 ** 12)
        if not ok.any():
            continue
        match = [o for o in a_dev if o["lat"] == float(lat) and o["lon"] == float(lon)]
        if not match:
            continue
        o = match[0]
        dev_xyz = F.geodetic_to_ecef(np.array([lat]), np.array([lon]))[0]
        vis_d = F.visibility(P, dev_xyz[None, :], 20.0)[:, :, 0]
        # the moments the model takes from the plan
        t0 = time.perf_counter()
        U = vis_d.any(axis=0)
        cu, gu = F.runs(U)
        gu = gu * DT
        _ = (1 - U.mean()) * np.mean(gu ** 2) / (2 * np.mean(gu)) if len(gu) else np.nan
        tA += time.perf_counter() - t0
        # replay: the device's own periodic updates over the plan, the last day kept as delivery tail
        t0 = time.perf_counter()
        gen = np.arange(rng.random() * per, t[-1] - 86400.0, per)
        gi, k1, up, dl = first_available(P, vis_d, gw_runs, dev_xyz, gen, nt)
        tR += time.perf_counter() - t0
        good = dl < 10 ** 12
        dA = (up[good] * DT - gen[good])
        dH = (dl[good] - up[good]) * DT
        dD = dA + dH
        pool.append(dD)
        # track-conditioned hold: entry of the device's cap, weighted as in Proposition 2
        pk = np.array([F.contact_stats(np.radians(lat), (F.MU / (s.no_kozai / 60) ** 2) ** (1 / 3) - F.RE, s.inclo,
                                       np.radians(20.0))["EG"] for s in sats])
        w_k = np.where(np.isfinite(pk) & (pk > 0), 1 / pk, 0.0)
        w_k = w_k / w_k.sum() if w_k.sum() > 0 else w_k
        nb = tab.shape[2]
        val = np.zeros(S)
        for k in range(S):
            vals = []
            for d, la in ((1, lat - psi), (0, lat + psi), (1, lat), (0, lat)):
                b = int(np.clip((la + 90) / LATBIN, 0, nb - 1))
                if np.isfinite(tab[k, d, b]):
                    vals.append(tab[k, d, b])
            val[k] = np.mean(vals) if vals else np.nan
        eh_track = (1 - o["q"]) * np.nansum(w_k * val)
        # the test week, split by leg
        sA = up_all[ok] * DT - gen_all[ok]
        sH = (del_all[ok] - up_all[ok]) * DT
        sD = sA + sH
        out.append(dict(lat=float(lat), lon=float(lon), q=o["q"],
                        sim_A=float(sA.mean()), sim_H=float(sH.mean()), sim_D=float(sD.mean()),
                        sim_p95=float(np.percentile(sD, 95)), sim_p99=float(np.percentile(sD, 99)),
                        h0_sim=float(np.mean(sH == 0)),
                        model_A=o["EA_semi"], model_H=o["EH_semi"], half_A=o["EA_half"], half_H=o["EH_half"],
                        cf_A=o["EA_cf"], cf_H=o["EH_cf"], track_H=float(eh_track),
                        replay_A=float(dA.mean()), replay_H=float(dH.mean()), replay_D=float(dD.mean()),
                        replay_p95=float(np.percentile(dD, 95)), replay_p99=float(np.percentile(dD, 99)),
                        model_p99=o["p99_pred"]))
    pool = np.concatenate(pool)
    res = dict(devices=out, pool=dict(replay_mean=float(pool.mean()), replay_p95=float(np.percentile(pool, 95)),
                                      replay_p99=float(np.percentile(pool, 99))),
               time=dict(moments_s_per_dev=tA / len(out), replay_s_per_dev=tR / len(out)), quant=an["quant"])
    json.dump(res, open(os.path.join(HERE, "out", "rev4", f"replay_{tag}.json"), "w"), indent=1)
    print(tag, "done", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]))
