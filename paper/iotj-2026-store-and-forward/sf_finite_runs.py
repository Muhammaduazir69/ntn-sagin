#!/usr/bin/env python3
"""Campaign of sf_finite.py (Nov 2026 revision). Parts:
 check    no limits, same buoys as Monte Carlo week: must match sf_mc.py
 tu       update period sweep, FCFS / device-only / satellite-only / both keep-newest (M4, M5)
 pop      population sweep (M4, M10)
 feed     delivery budget against the stability limit, 4 and 14 days (M7)
 uplink   shared uplink at stated rates, population sweep (M10)
 ml       Monitoring List of a subset of the fleet (M10)
 rr       shared uplink under per-report round robin, all fleets, both populations, both rates, with LCFS and device-only
          keep-newest, and on Sateliot with all stations two more weeks. Output out/rev4/finite_rr.jsonl.
 index    the first campaign's fixed-priority rule at the overload point, in finite_rr.jsonl, for the comparison.
 window   the break over three start dates at 28 days, a seed check, Orbcomm and Kineis at 400 b/s; same: four-day rows for ns-3.
 long     FCFS at all stations over 14 and 28 days, with the device backlog per day, in finite_rr.jsonl.
 rate     break point of the shared uplink, all stations, rates swept, with eta_k of Corollary 2, in finite_rr.jsonl.
Output: out/rev4/finite.jsonl (appended, resumable by key); the parts uplink and uplink2 there are the old rule."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_finite as S  # noqa: E402
import sf_mc  # noqa: E402

OUT = os.path.join(HERE, "out", "rev4", "finite.jsonl")
FL, GW = ("orbcomm", "kineis", "sateliot"), ("k5", "aws11")
DISC = {"fcfs": ("fcfs", "fcfs"), "lcfs": ("lcfs", "lcfs"), "both": ("newest", "newest"),
        "devonly": ("newest", "fcfs"), "satonly": ("fcfs", "newest")}


def jobs(parts):
    J = []
    for w in (0, 1, 2):
        for f in FL:
            for g in GW:
                if "check" in parts:
                    for d in ("fcfs", "both"):
                        J.append(dict(part="check", fl=f, gw=g, w=w, disc=d, tu=15, n=150))
                if "tu" in parts:
                    for tu in (5, 15, 30, 60, 120, 240):
                        for d in ("fcfs", "both", "devonly", "satonly"):
                            J.append(dict(part="tu", fl=f, gw=g, w=w, disc=d, tu=tu, n=150))
                if "pop" in parts and w == 0:
                    for n in (60, 400, 919):
                        for d in ("fcfs", "both", "devonly"):
                            J.append(dict(part="pop", fl=f, gw=g, w=w, disc=d, tu=15, n=n))
                if "uplink" in parts and w == 0:
                    for n in (150, 919):
                        for bps in (400, 4000):
                            for d in ("fcfs", "both"):
                                J.append(dict(part="uplink", fl=f, gw=g, w=w, disc=d, tu=15, n=n, up_bps=bps))
                if "ml" in parts:
                    S_ = {"orbcomm": 14, "kineis": 25, "sateliot": 4}[f]
                    for m in (1, max(1, S_ // 2)):
                        J.append(dict(part="ml", fl=f, gw=g, w=w, disc="fcfs", tu=15, n=150, ml=m))
    if "uplink2" in parts:   # closest baselines at the overload point of the shared uplink
        for d in ("lcfs", "devonly"):
            J.append(dict(part="uplink2", fl="sateliot", gw="k5", w=0, disc=d, tu=15, n=919, up_bps=400))
    if "rr" in parts:
        for f in FL:
            for n in (150, 919):
                for bps in (400, 4000):
                    for d in ("fcfs", "lcfs", "devonly", "both"):
                        J.append(dict(part="rr", fl=f, gw="k5", w=0, disc=d, tu=15, n=n, up_bps=bps))
        for w in (1, 2):
            for d in ("fcfs", "lcfs", "devonly", "both"):
                J.append(dict(part="rr", fl="sateliot", gw="k5", w=w, disc=d, tu=15, n=919, up_bps=400))
    if "rate" in parts:   # break point of the shared uplink against Corollary 2, all stations
        for bps in (500, 600, 700, 800, 1000, 1600, 2400):
            for d in ("fcfs", "lcfs", "devonly"):
                J.append(dict(part="rate", fl="sateliot", gw="k5", w=0, disc=d, tu=15, n=919, up_bps=bps))
        for bps in (50, 100, 200):
            for f in ("orbcomm", "kineis"):
                J.append(dict(part="rate", fl=f, gw="k5", w=0, disc="fcfs", tu=15, n=919, up_bps=bps))
    if "long" in parts:   # the break of FCFS at horizons of 14 and 28 days; backlog per day settles it, not a ratio of ages
        for days in (14, 28):
            for bps in (1000, 1300, 1600, 1800, 2000, 2200, 2400, 3000):
                J.append(dict(part="long", fl="sateliot", gw="k5", w=0, disc="fcfs", tu=15, n=919, up_bps=bps, days=days))
        J.append(dict(part="long", fl="sateliot", gw="k5", w=0, disc="lcfs", tu=15, n=919, up_bps=400, days=14))
    if "window" in parts:  # the break over three start dates, 28 days, and a seed check; backlog per day settles it
        for start in (-14.0, 0.0, 14.0):
            for bps in (1800, 2000, 2200, 2400):
                J.append(dict(part="window", fl="sateliot", gw="k5", w=0, disc="fcfs", tu=15, n=919, up_bps=bps, days=28, start=start))
        for bps in (2000, 2200):
            J.append(dict(part="window", fl="sateliot", gw="k5", w=0, disc="fcfs", tu=15, n=919, up_bps=bps, days=28, start=0.0, seed=41))
        J.append(dict(part="window", fl="sateliot", gw="k5", w=0, disc="fcfs", tu=15, n=919, up_bps=3000, days=28, start=14.0))
        for f in ("orbcomm", "kineis"):
            J.append(dict(part="window", fl=f, gw="k5", w=0, disc="fcfs", tu=15, n=919, up_bps=400, days=28, start=0.0))
    if "same" in parts:    # same horizon as the ns-3 runs of all stations: four days, one of them free of traffic
        for bps in (400, 1000, 1600, 2400):
            J.append(dict(part="same", fl="sateliot", gw="k5", w=0, disc="fcfs", tu=15, n=919, up_bps=bps, days=4, start=0.0, tail=1.0))
    if "index" in parts:
        for d in ("fcfs", "lcfs", "devonly"):
            J.append(dict(part="index", fl="sateliot", gw="k5", w=0, disc=d, tu=15, n=919, up_bps=400))
    if "feed" in parts:
        for rep in range(5):
            for f, feeds in (("orbcomm", (10, 20, 40, 80)), ("sateliot", (20, 40, 80, 160))):
                for fb in feeds:
                    for d in ("fcfs", "lcfs", "both"):
                        for days in (4, 14):
                            J.append(dict(part="feed", fl=f, gw="k5", w=rep, disc=d, tu=15, n=60, feed=fb, days=days))
    return J


def key(j):
    return json.dumps({k: j[k] for k in sorted(j)}, sort_keys=True)


_cache = {}


def run(j):
    days = j.get("days", 7)
    if j["part"] == "feed":
        rows = open(os.path.join(HERE, "out", f"buoys_rep{j['w']}.csv")).read().strip().splitlines()[1:]
        buoys = np.array([[float(x) for x in r.split(",")[1:3]] for r in rows])
        start, tail = 0.0, 1.0
        seed = j["w"] + 1
    else:
        rng = np.random.default_rng(j["w"] + 1 if j["n"] == 150 else 500 + j["w"])
        buoys = np.array([[b[1], b[2]] for b in sf_mc.load_buoys(j["n"], rng)])
        start, tail = j.get("start", 7.0 * j["w"]), j.get("tail", 3.0)
        seed = j.get("seed", 40 + j["w"])
    ck = (j["fl"], j["gw"], j["w"], j["n"], days, start)
    if ck not in _cache:
        _cache.clear()
        _cache[ck] = S.contact_plan(j["fl"], j["gw"], start, days, buoys)
    t, vg, pairs = _cache[ck]
    dv, st = DISC[j["disc"]]
    r = S.simulate(t, vg, pairs, len(buoys), j["tu"] * 60.0, seed=seed, dev=dv, sat=st, feed_budget=j.get("feed", 0),
                   up_bps=j.get("up_bps", 0.0), ml=j.get("ml", 0), tail_days=tail, occ_tail_days=tail,
                   up_share="index" if j["part"] == "index" else "rr")
    r.update(j)
    if j["part"] == "rate":
        r["eta"] = S.heard_duty(pairs, vg.shape[0])
    return r


if __name__ == "__main__":
    parts = sys.argv[1].split(",")
    if parts[0] in ("rr", "index", "rate", "long", "window", "same"):
        OUT = os.path.join(HERE, "out", "rev4", "finite_rr.jsonl")
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            done.add(r["_k"])
    J = [j for j in jobs(parts) if key(j) not in done]
    J.sort(key=lambda j: (j["fl"], j["gw"], j["w"], j["n"], j.get("days", 7), j.get("start", 0.0)))
    print(len(J), "runs", flush=True)
    with ProcessPoolExecutor(workers) as ex, open(OUT, "a") as fo:
        for j, r in zip(J, ex.map(run, J, chunksize=8 if workers > 1 else 1)):
            r["_k"] = key(j)
            fo.write(json.dumps(r) + "\n")
            fo.flush()
