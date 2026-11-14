#!/usr/bin/env python3
"""Held-out test with element sets issued after the prediction (Nov 2026 revision).

Prediction (made from what was available on 30 Oct): model and replay from the plan of 23 to 30 Oct, element
sets downloaded 30 Oct (epochs 23 to 29 Oct), out/rev4/replay_<fleet>_<gw>_w0.json.
Truth: first-available S&F from 30 Oct to 6 Nov propagated from the sets issued later (data_raw/issued_2026-11-03,
downloaded 3 Nov, epochs about 2 Nov), so the truth does not share element sets with the prediction.
Also the error of the lookup of the delivery time when it uses the older sets.
    python3 sf_heldout.py orbcomm k5      Output out/rev4/heldout_<fleet>_<gw>.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402
from sf_replay import plan, first_available, DT  # noqa: E402

BIG = 10 ** 12


def by_id(path):
    from sgp4.api import Satrec
    L = [x.rstrip() for x in open(path) if x.strip()]
    return {L[i + 1][2:7]: Satrec.twoline2rv(L[i + 1], L[i + 2]) for i in range(0, len(L) - 2, 3)}


def main(fl, gw):
    old = by_id(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    new = by_id(os.path.join(HERE, "data_raw", "issued_2026-11-03", f"{fl}.tle"))
    ids = [i for i in old if i in new]
    gws = sf_mc.load_gateways(gw)
    z = np.load(os.path.join(HERE, "out", f"sf_{fl}_{gw}_w0.npz"))
    pred = json.load(open(os.path.join(HERE, "out", "rev4", f"replay_{fl}_{gw}_w0.json")))
    t, Pn, vn = plan([new[i] for i in ids], gws, 0.0)
    _, Po, vo = plan([old[i] for i in ids], gws, 0.0)
    nt, S = len(t), len(ids)
    rn = [sf_mc.intervals(vn[k]) for k in range(S)]
    ro = [sf_mc.intervals(vo[k]) for k in range(S)]
    drift = np.linalg.norm(Pn - Po, axis=2)
    rng = np.random.default_rng(5)
    rows, pool, herr, A = [], [], [], []
    for o in pred["devices"]:     # list order, so co-located stations stay apart
        lat, lon = o["lat"], o["lon"]
        xyz = F.geodetic_to_ecef(np.array([lat]), np.array([lon]))[0]
        vis = F.visibility(Pn, xyz[None, :], 20.0)[:, :, 0]
        gen = np.arange(rng.random() * 900.0, 4 * 86400.0, 900.0)
        gi, k1, up, dl = first_available(Pn, vis, rn, xyz, gen, nt)
        good = dl < BIG
        dA = up[good] * DT - gen[good]
        dH = (dl[good] - up[good]) * DT
        D = dA + dH
        pool.append(D)
        # delivery-time lookup with the older sets: same satellite and upload instant
        dl_old = np.array([sf_mc.next_in(*ro[k], np.array([u]))[0] for k, u in zip(k1[good][::8], up[good][::8])])
        ok = dl_old < BIG
        herr += np.abs((dl_old[ok] - dl[good][::8][ok]) * DT).tolist()
        rows.append(dict(sim=float(D.mean()), model=o["model_A"] + o["model_H"], replay=o["replay_D"],
                         half=o["half_A"] + o["half_H"], sim_p99=float(np.percentile(D, 99)), replay_p99=o["replay_p99"],
                         model_p99=o["model_p99"], sim_A=float(dA.mean()), model_A=o["model_A"], replay_A=o["replay_A"],
                         sim_H=float(dH.mean()), model_H=o["model_H"], replay_H=o["replay_H"]))
    pool = np.concatenate(pool)
    g = lambda k: np.array([r[k] for r in rows])
    s = g("sim")
    out = dict(fleet=fl, gw=gw, n_common=len(ids), n_old=len(old), n_new=len(new),
               drift_km_mean=float(drift.mean() / 1e3), drift_km_max=float(drift.max() / 1e3),
               err=dict(model=float(100 * (g("model").mean() / s.mean() - 1)), replay=float(100 * (g("replay").mean() / s.mean() - 1)),
                        half=float(100 * (g("half").mean() / s.mean() - 1))),
               med=dict(model=float(100 * np.median(np.abs(g("model") / s - 1))), replay=float(100 * np.median(np.abs(g("replay") / s - 1)))),
               p99=dict(sim=float(np.percentile(pool, 99)), model=pred["quant"]["pred_p99"], replay=pred["pool"]["replay_p99"]),
               lookup_err_s=dict(median=float(np.median(herr)), p99=float(np.percentile(herr, 99))), sim_mean_h=float(s.mean() / 3600))
    json.dump(out, open(os.path.join(HERE, "out", "rev4", f"heldout_{fl}_{gw}.json"), "w"), indent=1)
    print(out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
