#!/usr/bin/env python3
"""The wait timer and custody run together (Nov 2026, round 5). T3451 gates uploads over other satellites while it runs
(TS 24.301 cl. 4.11.5.2; all satellites taken to be in one TAI list; the satellite that gave the value stays usable).
Slot simulator, unlimited links, the Monte Carlo week's own 150 buoys. Rules: none, lookup (return leg of each upload),
per-device 75th and 95th percentile of last week's return leg, a fixed 30 minutes. Custody: FCFS, keep-newest on both hops.
Two context regimes: any satellite can accept (contexts pushed to the fleet), or the context must return (TS 23.401 cl. 4.13.9.1).
Periods: 15 to 240 min. The T3451 gating exists only in this slot simulator; the ns-3 example has no timer and no NAS signaling.

    python3 sf_timer_gate.py orbcomm k5 0      Output out/rev4/tgate2_<fl>_<gw>_w<w>.json
    python3 sf_timer_gate.py stress            Sateliot, all stations, five gateways, 400 b/s: none and lookup, FCFS, LCFS, device rule
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_finite as S  # noqa: E402
import sf_mc  # noqa: E402

RV = os.path.join(HERE, "out", "rev4")
DISC = {"fcfs": ("fcfs", "fcfs"), "both": ("newest", "newest"), "lcfs": ("lcfs", "lcfs"), "devonly": ("newest", "fcfs")}


def row(r, rule, disc, ctx, tu, per=False):
    keep = ("aoi_all_h", "share", "uplinked", "n_nas", "n_failed", "d_mean_h", "d_p99_h", "occ_p99", "n_served", "generated")
    o = dict(rule=rule, disc=disc, ctx=ctx, tu=tu, **{k: r[k] for k in keep})
    if per:
        o["aoi_per_dev"] = r["aoi_per_dev"]
    return o


def week(fl, gw, w):
    """Grid: contexts must return, or are on board everywhere ("any"); periods; custody. See GRID."""
    path = os.path.join(RV, f"tgate2_{fl}_{gw}_w{w}.json")
    if os.path.exists(path):
        return
    rng = np.random.default_rng(w + 1)
    buoys = np.array([[b[1], b[2]] for b in sf_mc.load_buoys(150, rng)])
    n = len(buoys)
    t, vg, pairs = S.contact_plan(fl, gw, 7.0 * w - 7.0, 7, buoys)
    prev = S.simulate(t, vg, pairs, n, 900.0, seed=40 + w, dev="fcfs", sat="fcfs", tail_days=3.0, occ_tail_days=3.0, record_w=True)
    table = {q: {d: int(np.percentile(prev["w_rec"][d], q)) if prev["w_rec"].get(d) else 270 for d in range(n)} for q in (75, 95)}
    fix30 = {d: int(1800 / S.DT) for d in range(n)}
    rules = {"none": None, "lookup": "lookup", "p75": table[75], "p95": table[95], "fix30": fix30}
    t, vg, pairs = S.contact_plan(fl, gw, 7.0 * w, 7, buoys)
    grid = []
    for ctx in ("any", "return"):
        for disc in ("fcfs", "both"):
            grid += [(15, ctx, disc, ru) for ru in rules]
        for tu in (60, 240):
            grid += [(tu, ctx, "fcfs", ru) for ru in ("none", "lookup", "fix30")]
    for tu in (30, 120):
        grid += [(tu, "any", "fcfs", ru) for ru in ("none", "lookup")]
    rows = []
    for tu, ctx, disc, ru in grid:
        dv, st = DISC[disc]
        r = S.simulate(t, vg, pairs, n, tu * 60.0, seed=40 + w, dev=dv, sat=st, tail_days=3.0, occ_tail_days=3.0, timer=rules[ru], per_dev=True, context=ctx)
        rows.append(row(r, ru, disc, ctx, tu, per=(disc == "fcfs" and ru in ("none", "lookup"))))
    tmp = path + ".tmp"
    json.dump(dict(fleet=fl, gw=gw, w=w, rows=rows), open(tmp, "w"))
    os.replace(tmp, path)
    print(fl, gw, w, "done", len(rows), flush=True)


def rejects(fl, gw, w):
    """Rejects the must-return regime actually produces with no timer: wait to the next satellite holding the context."""
    path = os.path.join(RV, f"rejx_{fl}_{gw}_w{w}.json")
    if os.path.exists(path):
        return
    buoys = np.array([[b[1], b[2]] for b in sf_mc.load_buoys(150, np.random.default_rng(w + 1))])
    t, vg, pairs = S.contact_plan(fl, gw, 7.0 * w, 7, buoys)
    r = S.simulate(t, vg, pairs, len(buoys), 900.0, seed=40 + w, tail_days=3.0, occ_tail_days=3.0, context="return")
    o = dict(fleet=fl, gw=gw, w=w, **{k: r[k] for k in ("rej_n", "rej_over", "rej_first_n", "rej_first_over", "n_failed", "n_nas")})
    json.dump(o, open(path, "w"))
    print(o, flush=True)


def stress():
    path = os.path.join(RV, "tgate_stress.json")
    rng = np.random.default_rng(500)
    buoys = np.array([[b[1], b[2]] for b in sf_mc.load_buoys(919, rng)])
    t, vg, pairs = S.contact_plan("sateliot", "k5", 0.0, 7, buoys)
    rows = []
    for disc in ("fcfs", "lcfs", "devonly"):
        for rule, tm in (("none", None), ("lookup", "lookup")):
            dv, st = DISC[disc]
            r = S.simulate(t, vg, pairs, 919, 900.0, seed=40, dev=dv, sat=st, up_bps=400.0, tail_days=3.0, occ_tail_days=3.0, timer=tm, per_dev=True)
            rows.append(row(r, rule, disc, "any", 15, per=True))
            print("stress", disc, rule, round(r["aoi_all_h"], 2), round(r["share"], 3), flush=True)
            tmp = path + ".tmp"
            json.dump(dict(rows=rows), open(tmp, "w"))
            os.replace(tmp, path)


if __name__ == "__main__":
    os.nice(19)
    if sys.argv[1] == "stress":
        stress()
    elif len(sys.argv) > 4 and sys.argv[4] == "rejects":
        rejects(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        week(sys.argv[1], sys.argv[2], int(sys.argv[3]))
