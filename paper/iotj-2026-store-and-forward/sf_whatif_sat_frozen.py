#!/usr/bin/env python3
r"""Out-of-sample test of Proposition (adding a satellite), Nov 2026. sf_whatif.run_sat scored week 0 with its buoy
sample, the development week, and was corrected once on that week (censored gaps). Here the same code path, unchanged,
is scored on weeks 1 and 2 with their own buoy samples and fresh random placements, and run once.
    python3 sf_whatif_sat_frozen.py orbcomm 1     Output out/rev4/whatif_sat_frozen_<fleet>_w<week>.json
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
from sf_replay import plan, DT  # noqa: E402
from sf_whatif import N_PLACE, circular_gaps, clone, forward_mean, tail  # noqa: E402


def run(fl, w):
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    gws = sf_mc.load_gateways("k5")
    rng = np.random.default_rng(1000 + w)
    buoys = np.load(os.path.join(HERE, "out", f"sf_{fl}_k5_w{w}.npz"))["buoys"]
    d_p, d_t = 7.0 * (w - 1), 7.0 * w
    t_p, P_p, _ = plan(sats, gws, d_p)
    t_t, P_t, _ = plan(sats, gws, d_t)
    T = len(t_p) * DT
    x = np.arange(0, 7 * 86400.0, DT)
    S = len(sats)
    place = [clone(sats[rng.integers(S)], rng) for _ in range(N_PLACE)]
    jd0, fr0 = jday(2026, 10, 30, 0, 0, 0)
    P_new = F.ecef_positions(place, jd0, fr0 + d_t, t_t)
    xyz = F.geodetic_to_ecef(buoys[:, 0], buoys[:, 1])
    res = []
    for j in range(len(buoys)):
        vp = F.visibility(P_p, xyz[j:j + 1], 20.0)[:, :, 0]
        vt = F.visibility(P_t, xyz[j:j + 1], 20.0)[:, :, 0]
        Sa = tail(circular_gaps(vp.any(axis=0)), T, x)
        Sr = np.mean([tail(circular_gaps(vp[k]), T, x) for k in range(S)], axis=0)
        U = vt.any(axis=0)
        vn = F.visibility(P_new, xyz[j:j + 1], 20.0)[:, :, 0]
        res.append(dict(base_pred=float(np.trapezoid(Sa, x)), pred=float(np.trapezoid(Sa * Sr, x)), base=forward_mean(U),
                        each=[forward_mean(U | vn[m]) for m in range(N_PLACE)]))
    json.dump(res, open(os.path.join(HERE, "out", "rev4", f"whatif_sat_frozen_{fl}_w{w}.json"), "w"))
    print("done", fl, w, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
