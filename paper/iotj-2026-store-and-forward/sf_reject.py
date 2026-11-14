#!/usr/bin/env python3
"""The denominator the 90-minute example governs (Nov 2026, round 5). TS 24.301 cl. 4.11.5.2 lets a device replace a value above
its maximum at the first attempt after a reject without integrity protection with cause #83. We take such a reject as possible
at the start of every contact a device has with the fleet, through the highest-elevation satellite, and ask how long the return
leg (9) from that instant would be. Output out/rev4/reject_<fl>_<gw>_w<w>.json.
    python3 sf_reject.py orbcomm k5 0
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
from sf_timers import return_leg  # noqa: E402

BIG = 10 ** 12
RV = os.path.join(HERE, "out", "rev4")


def main(fl, gw, w):
    path = os.path.join(RV, f"reject_{fl}_{gw}_w{w}.json")
    if os.path.exists(path):
        return
    os.nice(19)
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    t, P, v = plan(sats, sf_mc.load_gateways(gw), 7.0 * w)
    S, nt = len(sats), len(t)
    gr = [sf_mc.intervals(v[k]) for k in range(S)]
    devs = json.load(open(os.path.join(RV, f"replay_{fl}_{gw}_w{w}.json")))["devices"]
    out = []
    for o in devs:
        xyz = F.geodetic_to_ecef(np.array([o["lat"]]), np.array([o["lon"]]))[0]
        vd = F.visibility(P, xyz[None, :], 20.0)[:, :, 0]
        dev = [sf_mc.intervals(vd[k]) for k in range(S)]
        st = np.flatnonzero(np.diff(np.r_[0, vd.any(axis=0).astype(np.int8)]) == 1)
        st = st[st < nt - int(2 * 86400 / DT)]
        if not len(st):
            continue
        los = P[:, st, :] - xyz[None, None, :]
        se = (los @ (xyz / np.linalg.norm(xyz))) / np.linalg.norm(los, axis=2)
        k = np.argmax(np.where(vd[:, st], se, -2.0), axis=0)
        dl = np.array([sf_mc.next_in(*gr[a], np.array([b]))[0] for a, b in zip(k, st)])
        g = dl < BIG
        c = return_leg(dev, gr, dl[g], range(S))
        ok = c < BIG
        out += ((c[ok] - st[g][ok]) * DT).tolist()
    a = np.array(out)
    res = dict(fleet=fl, gw=gw, w=w, n=len(a), over90=float(np.mean(a > 5400.0)), median=float(np.median(a)), p99=float(np.percentile(a, 99)))
    json.dump(res, open(path, "w"))
    print(fl, gw, w, res, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
