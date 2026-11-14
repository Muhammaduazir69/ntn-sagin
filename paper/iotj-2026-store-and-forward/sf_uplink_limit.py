#!/usr/bin/env python3
"""Necessary rate of the shared uplink, Corollary 2 (Nov 2026, round 3). For a set C of devices, reports of C can only
leave while some satellite hears a member of C as that member's best, so FCFS and LCFS backlogs grow unless
R >= |C| L / (T_u sum_k eta_{C,k}). C is every box of a latitude-longitude grid, at several sizes, and the whole fleet of
devices. All stations, five gateways, week 0, the same draw of stations as sf_finite_runs.py.
    python3 sf_uplink_limit.py        Output out/rev4/uplink_limit.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_finite as S  # noqa: E402
import sf_mc  # noqa: E402

L_BITS, TU = 8 * 52, 900.0
BOXES = (5, 10, 20, 30, 45)


def limit(pairs, buoys, nsat):
    out, nt = {}, len(pairs)
    n = len(buoys)
    for size in BOXES:
        box = [(int((la + 90) // size), int((lo + 180) // size)) for la, lo in buoys]
        count = {}
        for b in box:
            count[b] = count.get(b, 0) + 1
        hits = {}
        for row in pairs:
            best = {}
            for d, k, s in row:
                if d not in best or s > best[d][1]:
                    best[d] = (k, s)
            for key in {(box[d], v[0]) for d, v in best.items()}:
                hits[key] = hits.get(key, 0) + 1
        sum_eta = {}
        for (b, k), h in hits.items():
            sum_eta[b] = sum_eta.get(b, 0.0) + h / nt
        r = {b: count[b] * L_BITS / TU / e for b, e in sum_eta.items()}
        w = max(r, key=r.get)
        out[str(size)] = dict(rstar=float(r[w]), n=count[w], lat=w[0] * size - 90, lon=w[1] * size - 180, sum_eta=float(sum_eta[w]))
    # sliding boxes on the Sateliot population, to see how far the bound can be raised by choosing the set
    best = dict(rstar=0.0)
    rows = []
    for row in pairs:
        d_ = {}
        for d, k, s in row:
            if d not in d_ or s > d_[d][1]:
                d_[d] = (k, s)
        rows.append([(d, v[0]) for d, v in d_.items()])
    ar = np.array(buoys)
    for size in (15, 20, 25, 30, 35, 40):
        for la0 in range(-60, 60, 5):
            for lo0 in range(-180, 180, 10):
                m = (ar[:, 0] >= la0) & (ar[:, 0] < la0 + size) & (ar[:, 1] >= lo0) & (ar[:, 1] < lo0 + size)
                c = int(m.sum())
                if c < 100:
                    continue
                hit = set()
                for i, row in enumerate(rows):
                    for d, k in row:
                        if m[d]:
                            hit.add((i, k))
                r_ = c * L_BITS / TU / (len(hit) / nt)
                if r_ > best["rstar"]:
                    best = dict(rstar=float(r_), n=c, lat=la0, lon=lo0, size=size)
    out["sliding"] = best
    busy = S.heard_duty(pairs, nsat)
    out["all"] = dict(rstar=float(n * L_BITS / TU / sum(busy)), n=n, sum_eta=float(sum(busy)))
    return out


if __name__ == "__main__":
    os.nice(10)
    res = {}
    rng = np.random.default_rng(500)
    buoys = np.array([[b[1], b[2]] for b in sf_mc.load_buoys(919, rng)])
    for fl in ("sateliot",):
        rng = np.random.default_rng(500)
        buoys = np.array([[b[1], b[2]] for b in sf_mc.load_buoys(919, rng)])
        t, vg, pairs = S.contact_plan(fl, "k5", 0.0, 7, buoys)
        res[fl] = limit(pairs, buoys.tolist(), vg.shape[0])
        print(fl, res[fl], flush=True)
        json.dump(res, open(os.path.join(HERE, "out", "rev4", "uplink_limit.json"), "w"), indent=1)  # noqa
