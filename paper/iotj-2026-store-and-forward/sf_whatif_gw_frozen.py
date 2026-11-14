#!/usr/bin/env python3
r"""Out-of-sample test of the gateway what-if (Nov 2026). The form in eq. (whatifgw), per-satellite gap E[G_new,k], was
changed from a fleet-average gap after seeing errors on week 0 of the leave-one-out test (out/rev4/whatif_gw_*). Week 0
and its buoy sample are therefore the development split. Here the form is frozen as it stands and scored on data it
has never seen; nothing in this file was tuned on its output and it was run once:

  loo   leave-one-out of every site of both gateway sets, weeks 1 and 2, each week's own buoy sample,
        predicted from the week before, truth from replay of the full set on the test week.
  add   a site that was in no development case as an added site for this base: each eleven-set site not in the five-set
        added to the five gateways, week 1 and 2, predicted from the week before on the five-gateway plan.

The superseded form (fleet-average gap) is scored too, on the same cases, so the text can say whether the change mattered.
    python3 sf_whatif_gw_frozen.py orbcomm     Output out/rev4/whatif_gw_frozen_<fleet>.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402
from sf_analysis import cap_overlap  # noqa: E402
from sf_replay import plan  # noqa: E402
from sf_whatif import alt, holds  # noqa: E402

START = {1: (0.0, 7.0), 2: (7.0, 14.0)}


def predict(rest, buoys, new_site, sats, h_mean):
    st = [F.contact_stats(np.radians(new_site[1]), alt(s), s.inclo, np.radians(10.0)) for s in sats]
    eg = np.array([s["EG"] if np.isfinite(s["EG"]) and s["p"] > 0 else np.inf for s in st])
    finite = eg[np.isfinite(eg)]
    # superseded form: one gap for the whole fleet, the mean rate of the satellites that can see the site
    eg_fleet = len(eg) / np.sum(1.0 / finite) if len(finite) else np.inf
    per, avg = [], []
    for j in range(len(buoys)):
        q = cap_overlap(buoys[j, 0], buoys[j, 1], [new_site], h_mean, 20.0, 10.0, n=1000)
        h, kk = rest[j]
        e = eg[kk]
        v = np.where(np.isfinite(e), e * (1 - np.exp(-h / np.where(np.isfinite(e), e, 1.0))), h)
        per.append((1 - q) * float(np.mean(v)))
        w = eg_fleet * (1 - np.exp(-h / eg_fleet)) if np.isfinite(eg_fleet) else h
        avg.append((1 - q) * float(np.mean(w)))
    return float(np.mean(per)), float(np.mean(avg))


def mean_hold(hh):
    return float(np.mean([np.mean(h) for h, _ in hh if len(h)]))


def run(fl):
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    h_mean = float(np.mean([alt(s) for s in sats]))
    rows = []
    g5, g11 = sf_mc.load_gateways("k5"), sf_mc.load_gateways("aws11")
    for w, (d_p, d_t) in START.items():
        for gw, gws in (("k5", g5), ("aws11", g11)):
            buoys = np.load(os.path.join(HERE, "out", f"sf_{fl}_{gw}_w{w}.npz"))["buoys"]
            xyz = F.geodetic_to_ecef(buoys[:, 0], buoys[:, 1])
            _, P_p, _ = plan(sats, gws, d_p)
            _, P_t, _ = plan(sats, gws, d_t)
            gx = F.geodetic_to_ecef(np.array([g[1] for g in gws]), np.array([g[2] for g in gws]))
            vp, vt = F.visibility(P_p, gx, 10.0), F.visibility(P_t, gx, 10.0)
            truth = mean_hold(holds(P_t, vt.any(axis=2), buoys, xyz, np.random.default_rng(100 + w)))
            for g in range(len(gws)):
                keep = [i for i in range(len(gws)) if i != g]
                rest = holds(P_p, vp[:, :, keep].any(axis=2), buoys, xyz, np.random.default_rng(100 + w))
                per, avg = predict(rest, buoys, gws[g], sats, h_mean)
                rows.append(dict(kind="loo", fleet=fl, gw=gw, week=w, site=gws[g][0], pred=per, pred_fleetavg=avg,
                                 none=mean_hold(rest), truth=truth))
                print("loo", fl, gw, w, gws[g][0], flush=True)
            if gw == "k5":
                names5 = {g[0] for g in g5}
                for a in g11:
                    if a[0] in names5:
                        continue
                    rest = holds(P_p, vp.any(axis=2), buoys, xyz, np.random.default_rng(100 + w))
                    per, avg = predict(rest, buoys, a, sats, h_mean)
                    ax = F.geodetic_to_ecef(np.array([a[1]]), np.array([a[2]]))
                    v_all = np.concatenate([vt, F.visibility(P_t, ax, 10.0)], axis=2)
                    tr = mean_hold(holds(P_t, v_all.any(axis=2), buoys, xyz, np.random.default_rng(100 + w)))
                    rows.append(dict(kind="add", fleet=fl, gw="k5", week=w, site=a[0], pred=per, pred_fleetavg=avg,
                                     none=mean_hold(rest), truth=tr))
                    print("add", fl, w, a[0], flush=True)
    json.dump(rows, open(os.path.join(HERE, "out", "rev4", f"whatif_gw_frozen_{fl}.json"), "w"), indent=1)
    print("done", fl, flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
