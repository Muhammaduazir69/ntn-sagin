#!/usr/bin/env python3
r"""What-if tests of the renewal laws (Nov 2026 revision), scored against the plan of the changed system.

add-satellite  A new satellite of an existing class with an undecided orbit phase (random node and mean
               anomaly). Proposition 4: for a phase independent of the fleet,
                   Pr(A' > x) = Pr(A > x) Pr(R_new > x),
               with both tails the residual laws of last week's plan; the law of R_new is the mean over the
               satellites of the class, so no orbit of the new satellite is propagated.
               Truth: test week, mean over N_PLACE random placements, each propagated.
               Rivals: scaling E[A] by K/(K+1); the exponential form of Proposition 1.
add-gateway    One site added to a gateway set (leave one out of the set). Corollary 2:
                   E[H'] = (1 - q_new) \int Pr(H > x) exp(-x / E[G_new]) dx,
               Pr(H > x) from replay of the reduced set last week; E[G_new] from Lemma 1, q_new the share of
               the device's cap inside the new gateway's cap. Truth: replay of the full set on the test week.
    python3 sf_whatif.py sat orbcomm k5   |   python3 sf_whatif.py gw orbcomm k5
Output out/rev4/whatif_<kind>_<fleet>_<gw>.json
"""
import json
import os
import sys
import time

import numpy as np
from sgp4.api import Satrec, WGS72, jday

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402
from sf_replay import plan, first_available, DT  # noqa: E402
from sf_analysis import cap_overlap  # noqa: E402

N_PLACE = 30
N_DEV = 150


def alt(s):
    return (F.MU / (s.no_kozai / 60) ** 2) ** (1 / 3) - F.RE


def clone(s, rng):
    """SGP4 element set of a new satellite of the class of s: same altitude, inclination, drag term; random phase."""
    jd0, fr0 = jday(2026, 10, 30, 0, 0, 0)
    n = Satrec()
    n.sgp4init(WGS72, "i", 99000 + rng.integers(900), jd0 + fr0 - 2433281.5, s.bstar, 0.0, 0.0, 1e-4, 0.0, s.inclo,
               rng.random() * 2 * np.pi, s.no_kozai, rng.random() * 2 * np.pi)
    return n


def tail(gaps, T, x):
    """Pr(R > x), R the forward wait of a stationary contact process with these gaps over time T."""
    g = np.sort(np.asarray(gaps, float))
    if len(g) == 0:
        return np.ones_like(x) * 0.0
    cs = np.r_[np.cumsum(g[::-1])[::-1], 0.0]
    idx = np.searchsorted(g, x, side="right")
    return (cs[idx] - (len(g) - idx) * x) / T


def circular_gaps(mask):
    """Gap lengths (s) of a boolean series closed on itself, so the two censored ends form one gap and the
    gaps sum to the time out of contact; without this the tail at zero falls short of 1 - p."""
    if not mask.any():
        return np.array([len(mask) * DT])
    m = np.roll(mask, -int(np.argmax(mask)))
    d = np.diff(np.r_[1, m.astype(np.int8), 1])
    st, en = np.where(d == -1)[0], np.where(d == 1)[0]
    return (en - st) * DT


def forward_mean(mask, horizon_days=5.0):
    """Mean forward wait (s) from every sample of the first horizon_days to the next True sample; the rest of the
    series closes the waits, so no gap is censored."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), n + 10 ** 7)
    nx = np.minimum.accumulate(idx[::-1])[::-1]
    k = int(horizon_days * 86400.0 / DT)
    return float(np.mean((nx[:k] - np.arange(k))) * DT)


def run_sat(fl, gw):
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    gws = sf_mc.load_gateways(gw)
    rng = np.random.default_rng(11)
    buoys = np.load(os.path.join(HERE, "out", f"sf_{fl}_{gw}_w0.npz"))["buoys"]
    t_p, P_p, _ = plan(sats, gws, -7.0)
    t_t, P_t, _ = plan(sats, gws, 0.0)
    T = len(t_p) * DT
    x = np.arange(0, 7 * 86400.0, DT)
    S = len(sats)
    place = [clone(sats[rng.integers(S)], rng) for _ in range(N_PLACE)]
    jd0, fr0 = jday(2026, 10, 30, 0, 0, 0)
    P_new = F.ecef_positions(place, jd0, fr0, t_t)
    xyz = F.geodetic_to_ecef(buoys[:, 0], buoys[:, 1])
    res = []
    tm, tr = 0.0, 0.0
    for j in range(len(buoys)):
        vp = F.visibility(P_p, xyz[j:j + 1], 20.0)[:, :, 0]
        vt = F.visibility(P_t, xyz[j:j + 1], 20.0)[:, :, 0]
        t0 = time.perf_counter()
        Sa = tail(circular_gaps(vp.any(axis=0)), T, x)
        Sr = np.mean([tail(circular_gaps(vp[k]), T, x) for k in range(S)], axis=0)
        pred = float(np.trapezoid(Sa * Sr, x))
        base_pred = float(np.trapezoid(Sa, x))
        tm += time.perf_counter() - t0
        U = vt.any(axis=0)
        base = forward_mean(U)
        t0 = time.perf_counter()
        vn = F.visibility(P_new, xyz[j:j + 1], 20.0)[:, :, 0]
        each = [forward_mean(U | vn[m]) for m in range(N_PLACE)]
        tr += time.perf_counter() - t0
        # exponential form of Proposition 1 with the new satellite's rate from Lemma 1
        pk, eg = [], []
        for s in list(sats) + [sats[0]]:
            st = F.contact_stats(np.radians(buoys[j, 0]), alt(s), s.inclo, np.radians(20.0))
            pk.append(st["p"]); eg.append(st["EG"])
        pk, eg = np.array(pk), np.array(eg)
        ok = np.isfinite(eg) & (pk > 0)
        cf = float(np.prod(1 - pk) / np.sum(1 / eg[ok])) if ok.any() else np.nan
        cf0 = float(np.prod(1 - pk[:-1]) / np.sum(1 / eg[:-1][ok[:-1]])) if ok[:-1].any() else np.nan
        res.append(dict(lat=float(buoys[j, 0]), base=base, base_pred=base_pred, pred=pred, truth=float(np.mean(each)),
                        truth_sd=float(np.std(each)), scale=base * S / (S + 1), cf=cf, cf0=cf0, each=[float(e) for e in each]))
    json.dump(dict(devices=res, time=dict(whatif_s_per_dev=tm / len(buoys), replay_s_per_dev_30=tr / len(buoys))),
              open(os.path.join(HERE, "out", "rev4", f"whatif_sat_{fl}_{gw}.json"), "w"), indent=1)
    print("sat", fl, gw, "done", flush=True)


def holds(P, gwset_vis, buoys, xyz, rng):
    """Replay of the first-available rule; per-device hold samples, and access times."""
    S, nt = P.shape[0], P.shape[1]
    gw_runs = [sf_mc.intervals(gwset_vis[k]) for k in range(S)]
    out = []
    for j in range(len(buoys)):
        vis_d = F.visibility(P, xyz[j:j + 1], 20.0)[:, :, 0]
        gen = np.arange(rng.random() * 900.0, (nt - 1) * DT - 86400.0, 900.0)
        gi, k1, up, dl = first_available(P, vis_d, gw_runs, xyz[j], gen, nt)
        good = dl < 10 ** 12
        out.append(((dl[good] - up[good]) * DT, k1[good]))
    return out


def run_gw(fl, gw):
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    gws = sf_mc.load_gateways(gw)
    buoys = np.load(os.path.join(HERE, "out", f"sf_{fl}_{gw}_w0.npz"))["buoys"]
    xyz = F.geodetic_to_ecef(buoys[:, 0], buoys[:, 1])
    t_p, P_p, _ = plan(sats, gws, -7.0)
    t_t, P_t, _ = plan(sats, gws, 0.0)
    gxyz = F.geodetic_to_ecef(np.array([g[1] for g in gws]), np.array([g[2] for g in gws]))
    vgp = F.visibility(P_p, gxyz, 10.0)
    vgt = F.visibility(P_t, gxyz, 10.0)
    h_mean = float(np.mean([alt(s) for s in sats]))
    x = np.arange(0, 7 * 86400.0, DT)
    full = holds(P_t, vgt.any(axis=2), buoys, xyz, np.random.default_rng(3))
    full_mean = float(np.mean([np.mean(h) for h, _ in full]))
    rows = []
    for g in range(len(gws)):
        keep = [i for i in range(len(gws)) if i != g]
        rest_p = holds(P_p, vgp[:, :, keep].any(axis=2), buoys, xyz, np.random.default_rng(3))
        rest_t = holds(P_t, vgt[:, :, keep].any(axis=2), buoys, xyz, np.random.default_rng(3))
        st = [F.contact_stats(np.radians(gws[g][1]), alt(s), s.inclo, np.radians(10.0)) for s in sats]
        eg = np.array([s["EG"] if np.isfinite(s["EG"]) and s["p"] > 0 else np.inf for s in st])
        pred = []
        for j in range(len(buoys)):
            q = cap_overlap(buoys[j, 0], buoys[j, 1], [gws[g]], h_mean, 20.0, 10.0, n=1000)
            h, kk = rest_p[j]
            e = eg[kk]
            # int_0^inf Pr(H > x) exp(-x/e) dx = E[e (1 - exp(-H/e))], which tends to E[H] when e is infinite
            v = np.where(np.isfinite(e), e * (1 - np.exp(-h / np.where(np.isfinite(e), e, 1.0))), h)
            pred.append((1 - q) * float(np.mean(v)))
        rows.append(dict(site=gws[g][0], base_prior=float(np.mean([np.mean(h) for h, _ in rest_p])),
                         base=float(np.mean([np.mean(h) for h, _ in rest_t])), pred=float(np.mean(pred)), truth=full_mean))
    json.dump(rows, open(os.path.join(HERE, "out", "rev4", f"whatif_gw_{fl}_{gw}.json"), "w"), indent=1)
    print("gw", fl, gw, "done", flush=True)


if __name__ == "__main__":
    k, fl, gw = sys.argv[1:4]
    (run_sat if k == "sat" else run_gw)(fl, gw)
