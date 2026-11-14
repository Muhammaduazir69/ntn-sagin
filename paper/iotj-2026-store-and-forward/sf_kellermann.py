#!/usr/bin/env python3
"""Reproduce Kellermann et al., OJ-COMS 2025: 16-satellite Walker Star, 550 km, polar, 4 planes of 4, 45 deg node
spacing, 90 deg in-plane spacing, 22.5 deg phasing, five gateways, 10 deg feeder mask, UEs on a global grid.
They report median UE delay 1.18 h, satellite delay 0.36 h and end to end 1.63 h. Our run covers 30 days and also
scores Propositions 1 and 2 and the replay on this fleet, from the plan of the 30 days before.
    python3 sf_kellermann.py [ue_mask ...]    Output out/rev4/kellermann.json
"""
import json
import os
import sys

import numpy as np
from sgp4.api import Satrec, WGS72, jday

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402
from sf_replay import first_available, DT  # noqa: E402

BIG = 10 ** 12


def star():
    jd, fr = jday(2026, 10, 30, 0, 0, 0)
    n = np.sqrt(F.MU / (F.RE + 550e3) ** 3) * 60.0
    out = []
    for p in range(4):
        for s in range(4):
            sat = Satrec()
            sat.sgp4init(WGS72, "i", p * 10 + s, jd + fr - 2433281.5, 0.0, 0.0, 0.0, 1e-7, 0.0, np.radians(90.0),
                         np.radians((90.0 * s + 22.5 * p) % 360), n, np.radians(45.0 * p))
            out.append(sat)
    return out


def grid(n=612):
    """About n points evenly on the sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1 - 2 * i / n))
    lon = (np.degrees(np.pi * (1 + 5 ** 0.5) * i) + 180) % 360 - 180
    return np.c_[lat, lon]


def main(masks):
    sats = star()
    gws = sf_mc.load_gateways("k5")
    days = 30.0
    jd0, fr0 = jday(2026, 10, 30, 0, 0, 0)
    t = np.arange(0, days * 86400.0, DT)
    P = F.ecef_positions(sats, jd0, fr0, t)
    Pp = F.ecef_positions(sats, jd0, fr0 - days, t)
    g = F.geodetic_to_ecef(np.array([x[1] for x in gws]), np.array([x[2] for x in gws]))
    vg, vgp = F.visibility(P, g, 10.0).any(axis=2), F.visibility(Pp, g, 10.0).any(axis=2)
    S, nt = len(sats), len(t)
    gr, grp = [sf_mc.intervals(vg[k]) for k in range(S)], [sf_mc.intervals(vgp[k]) for k in range(S)]
    B = grid()
    xyz = F.geodetic_to_ecef(B[:, 0], B[:, 1])
    gaps_g = [F.runs(vgp[k])[1] * DT for k in range(S)]
    resg = np.array([np.mean(x ** 2) / (2 * np.mean(x)) for x in gaps_g])
    rng = np.random.default_rng(21)
    res = {}
    gen_stop = (nt - 1) * DT - 3 * 86400.0
    for mask in masks:
        As, Hs, Ds, mod, rep, sim = [], [], [], [], [], []
        for j in range(len(B)):
            vis = F.visibility(P, xyz[j:j + 1], mask)[:, :, 0]
            visp = F.visibility(Pp, xyz[j:j + 1], mask)[:, :, 0]
            gen = np.arange(rng.random() * 900.0, gen_stop, 3600.0)
            _, k1, up, dl = first_available(P, vis, gr, xyz[j], gen, nt)
            ok = dl < BIG
            A, H = up[ok] * DT - gen[ok], (dl[ok] - up[ok]) * DT
            As.append(A); Hs.append(H); Ds.append(A + H)
            _, gu = F.runs(visp.any(axis=0))
            gu = gu * DT
            ea = (1 - visp.any(axis=0).mean()) * np.mean(gu ** 2) / (2 * np.mean(gu)) if len(gu) else np.nan
            mod.append(ea + np.mean(resg))
            _, k2, u2, d2 = first_available(Pp, visp, grp, xyz[j], gen, nt)
            ok2 = d2 < BIG
            rep.append(np.mean(d2[ok2] * DT - gen[ok2]))
            sim.append(np.mean(A + H))
        A, H, D = map(np.concatenate, (As, Hs, Ds))
        sim, mod, rep = map(np.array, (sim, mod, rep))
        res[str(mask)] = dict(med_ue_h=float(np.median(A) / 3600), med_sat_h=float(np.median(H) / 3600), med_e2e_h=float(np.median(D) / 3600),
                              mean_e2e_h=float(D.mean() / 3600), fleet_err_model=float(100 * (np.nanmean(mod) / sim.mean() - 1)),
                              fleet_err_replay=float(100 * (rep.mean() / sim.mean() - 1)),
                              per_med_model=float(100 * np.nanmedian(np.abs(mod / sim - 1))), per_med_replay=float(100 * np.median(np.abs(rep / sim - 1))))
        print(mask, res[str(mask)], flush=True)
    json.dump(res, open(os.path.join(HERE, "out", "rev4", "kellermann.json"), "w"), indent=1)


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or [10.0, 20.0, 30.0, 40.0])
