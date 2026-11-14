#!/usr/bin/env python3
"""Closed-form store-and-forward delay against the SGP4 Monte Carlo (Section VI-A of the IoT-J study).

For each device:
  access wait A (time to the first visible satellite):  E[A] = prod_k (1 - p_dk) / sum_k 1/E[G_dk]
  custody hold H (residual gateway gap of the satellite, zero with probability q_d):
    E[H] = (1 - q_d) sum_k w_k E[G_gk^2] / (2 E[G_gk])
  The gap moments come from the contact plan of the previous week; see analyse().
p, E[C], E[G] come from the sub-satellite latitude density (sf_core.contact_stats).
"""
import json
import os
import sys

import numpy as np
from sgp4.api import jday

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402


def fleet(scenario):
    return F.load_tles(os.path.join(HERE, "data_raw", f"{scenario}.tle"))


def elements(s):
    a = (F.MU / (s.no_kozai / 60) ** 2) ** (1 / 3)
    return a - F.RE, s.inclo


def cap_overlap(lat, lon, gws, h, dev_mask, gw_mask, n=4000, seed=0):
    """Fraction of the device's visibility cap (sub-satellite points that see the device) that lies in
    the union of the gateway caps, by uniform sampling of the cap on the sphere."""
    rng = np.random.default_rng(seed)
    psi_d = F.cap_angle(h, np.radians(dev_mask))
    psi_g = F.cap_angle(h, np.radians(gw_mask))
    u = np.array([np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                  np.cos(np.radians(lat)) * np.sin(np.radians(lon)), np.sin(np.radians(lat))])
    cz = 1 - rng.random(n) * (1 - np.cos(psi_d))
    az = rng.random(n) * 2 * np.pi
    e = np.cross([0, 0, 1.0], u)
    e = e / np.linalg.norm(e) if np.linalg.norm(e) > 1e-9 else np.array([1.0, 0, 0])
    nn = np.cross(u, e)
    sz = np.sqrt(1 - cz ** 2)
    pts = cz[:, None] * u + sz[:, None] * (np.cos(az)[:, None] * e + np.sin(az)[:, None] * nn)
    g = np.array([[np.cos(np.radians(a)) * np.cos(np.radians(o)), np.cos(np.radians(a)) * np.sin(np.radians(o)),
                   np.sin(np.radians(a))] for _, a, o in gws])
    inside = (pts @ g.T) >= np.cos(psi_g)
    return float(inside.any(axis=1).mean())


def residual_sampler(gaps, rng, n):
    """n draws of the forward recurrence time of a renewal process with these gap lengths:
    a gap chosen with probability proportional to its length, then a uniform point inside it."""
    g = np.asarray(gaps, float)
    if len(g) == 0:
        return np.full(n, np.nan)
    pick = rng.choice(len(g), size=n, p=g / g.sum())
    return rng.random(n) * g[pick]


def analyse(scenario, gwset, tag, dev_mask=20.0, gw_mask=10.0, days=7, start_day=0.0, n_draw=4000):
    """Predict from the contact plan of the week BEFORE the simulated window and compare.

    Mean delay of device d (Propositions 1 and 2):
      E[A] = (1 - p_U) E[G_U^2] / (2 E[G_U])                    union contact process of the device
      E[H] = (1 - q_d) sum_k w_k E[G_gk^2] / (2 E[G_gk])          gateway gaps of satellite k
    with q_d the share of the device's cap inside a gateway cap and w_k the share of first contacts
    made with satellite k, taken proportional to 1/E[G_dk] from Lemma 1. Estimates compared:
      semi  gap moments from the previous week's plan (the model)
      cf    exponential gaps with Lemma 1 moments, no plan
      half  half the mean gap on each leg, with the same (1 - p_U) and (1 - q_d) factors
    Quantiles of D: A and H drawn from their residual laws (prob. p_U and q_d of zero) and summed.
    Keep-newest storage bound: devices that see satellite k during one of its gateway gaps."""
    d = np.load(os.path.join(HERE, "out", f"sf_{tag}.npz"))
    sats = fleet(scenario)
    gws = sf_mc.load_gateways(gwset)
    buoys = d["buoys"]
    rng = np.random.default_rng(7)
    jd0, fr0 = jday(2026, 10, 30, 0, 0, 0)
    fr0 += start_day - days                       # previous week
    t = np.arange(0, days * 86400.0, 20.0)
    P = F.ecef_positions(sats, jd0, fr0, t)
    gw_xyz = F.geodetic_to_ecef(np.array([g[1] for g in gws]), np.array([g[2] for g in gws]))
    vis_gw = F.visibility(P, gw_xyz, gw_mask).any(axis=2)
    S = len(sats)
    gw_gaps, gw_res, H_cf = [], [], []
    for k, s in enumerate(sats):
        h, inc = elements(s)
        p = min(sum(F.contact_prob(np.radians(g[1]), h, inc, np.radians(gw_mask)) for g in gws), 0.999)
        st = F.contact_stats(np.radians(np.mean([abs(g[1]) for g in gws])), h, inc, np.radians(gw_mask))
        eg = st["EC"] * (1 - p) / p
        c_, g = F.runs(vis_gw[k])
        g = g * 20.0
        gw_gaps.append(g)
        gw_res.append(np.mean(g ** 2) / (2 * np.mean(g)) if len(g) else np.nan)
        H_cf.append(eg)                              # exponential gaps: E[G^2]/(2E[G]) = E[G]
    gw_res, H_cf = np.array(gw_res), np.array(H_cf)
    gw_half = np.array([np.mean(g) / 2 if len(g) else np.nan for g in gw_gaps])
    # gateway gaps of each satellite as sample intervals, for the storage bound
    gap_iv = []
    win_lo, win_hi = int(86400.0 / 20.0), int(4 * 86400.0 / 20.0)    # same window as the Monte Carlo
    for k in range(S):
        st_, en_ = sf_mc.intervals(~vis_gw[k])
        keep = (st_ > 0) & (en_ < len(t)) & (en_ >= win_lo) & (en_ <= win_hi)
        gap_iv.append((st_[keep], en_[keep]))
    seen = [np.zeros(len(iv[0]), int) for iv in gap_iv]
    out, D_draws = [], []
    dev = d["dev"]
    h_mean = np.mean([elements(s)[0] for s in sats])
    for j, (lat, lon) in enumerate(buoys):
        pk, egk = [], []
        for s in sats:
            h, inc = elements(s)
            st = F.contact_stats(np.radians(lat), h, inc, np.radians(dev_mask))
            pk.append(st["p"])
            egk.append(st["EG"])
        pk, egk = np.array(pk), np.array(egk)
        ok = np.isfinite(egk) & (pk > 0)
        if not ok.any():
            continue
        ea_cf = np.prod(1 - pk) / np.sum(1 / egk[ok])
        vis_dk = F.visibility(P, F.geodetic_to_ecef(np.array([lat]), np.array([lon])), dev_mask)[:, :, 0]
        vis_d = vis_dk.any(axis=0)
        cu, gu = F.runs(vis_d)
        gu = gu * 20.0
        pU = vis_d.mean()
        ea_semi = (1 - pU) * np.mean(gu ** 2) / (2 * np.mean(gu)) if len(gu) else np.nan
        ea_half = (1 - pU) * np.mean(gu) / 2 if len(gu) else np.nan
        q = cap_overlap(lat, lon, gws, h_mean, dev_mask, gw_mask)
        w = np.where(ok, 1 / egk, 0)
        w = w / w.sum()
        eh_semi = (1 - q) * (w @ np.nan_to_num(gw_res))
        eh_cf = (1 - q) * (w @ H_cf)
        eh_half = (1 - q) * (w @ np.nan_to_num(gw_half))
        # quantile draws of D for this device
        a_dr = np.where(rng.random(n_draw) < pU, 0.0, residual_sampler(gu, rng, n_draw))
        ks = rng.choice(S, size=n_draw, p=w)
        h_dr = np.zeros(n_draw)
        for k in np.unique(ks):
            sel = ks == k
            h_dr[sel] = residual_sampler(gw_gaps[k], rng, sel.sum())
        h_dr = np.where(rng.random(n_draw) < q, 0.0, h_dr)
        D_draws.append(a_dr + h_dr)
        p99_pred_dev = float(np.percentile(a_dr + h_dr, 99))
        # storage bound: does this device see satellite k during each gateway gap of k?
        for k in range(S):
            cs = np.r_[0, np.cumsum(vis_dk[k])]
            st_, en_ = gap_iv[k]
            seen[k] += (cs[en_] - cs[st_]) > 0
        sel = dev == j
        ms = d["d_SF1"][sel]
        out.append(dict(lat=float(lat), lon=float(lon), q=float(q), pU=float(pU),
                        EA_semi=float(ea_semi), EA_cf=float(ea_cf), EA_half=float(ea_half),
                        EH_semi=float(eh_semi), EH_cf=float(eh_cf), EH_half=float(eh_half),
                        sim=float(np.nanmean(ms)), sim_fe=float(np.nanmean(d["d_SFE"][sel])), sim_bp=float(np.nanmean(d["d_BP"][sel])),
                        bp_lost=float(np.mean(~np.isfinite(d["d_BP"][sel]))),
                        p99_pred=p99_pred_dev,
                        p99_sim=float(np.percentile(ms[np.isfinite(ms)], 99)) if np.isfinite(ms).any() else float("nan")))
    D_draws = np.concatenate(D_draws)
    sim_all = d["d_SF1"][np.isfinite(d["d_SF1"])]
    store_bound = np.concatenate([s for s in seen if len(s)])
    res = {"devices": out,
           "quant": {"pred_mean": float(np.mean(D_draws)), "pred_p95": float(np.percentile(D_draws, 95)),
                     "pred_p99": float(np.percentile(D_draws, 99)), "sim_mean": float(sim_all.mean()),
                     "sim_p95": float(np.percentile(sim_all, 95)), "sim_p99": float(np.percentile(sim_all, 99))},
             "store": {"bound_p99": float(np.percentile(store_bound, 99)),
                       "pergap_violations": int(np.sum(d["occ_new"] > d["seen_win"])) if len(d["seen_win"]) == len(d["occ_new"]) else -1,
                       "pergap_n": int(len(d["occ_new"])),
                       "sim_new_p99": float(np.percentile(d["occ_new"], 99)),
                       "sim_fcfs_p99": float(np.percentile(d["occ_fcfs"], 99))}}
    json.dump(res, open(os.path.join(HERE, "out", f"an_{tag}.json"), "w"), indent=1)
    sim = np.array([o["sim"] for o in out])
    for k in ("cf", "half", "semi"):
        pred = np.array([o[f"EA_{k}"] + o[f"EH_{k}"] for o in out])
        err = (pred - sim) / sim
        print(f"{tag} {k:5s}: fleet-mean error {100 * (pred.mean() / sim.mean() - 1):+.1f} %, "
              f"median |err| {100 * np.median(np.abs(err)):.1f} %, p90 |err| {100 * np.percentile(np.abs(err), 90):.1f} %")
    print(tag, res["quant"], res["store"])


if __name__ == "__main__":
    a = sys.argv[1:]
    analyse(a[0], a[1], a[2], start_day=float(a[3]) if len(a) > 3 else 0.0)
