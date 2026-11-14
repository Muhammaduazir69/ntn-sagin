#!/usr/bin/env python3
"""ns-3 rows of the shared uplink under per-report round robin (Nov 2026, round 2), out/ns3_sf5.jsonl.

Rebuilt binary sim/bin/p6-sf-iot (ns3_patch/build.sh); only the branch taken when upBps > 0 changed, so only those
95 rows are redone: uplink at 400 and 4000 b/s (60), upall at 400 b/s (20), uplong (15). out/ns3_sf4.jsonl stays as
the record of the first binary, whose whole-backlog rule make_macros.py quotes for comparison.

  python3 run_ns3_sf5.py [PARALLEL=3]          the 95 rows, resumable, nice 10
  python3 run_ns3_sf5.py sweep                 upall FCFS at rates around the break point, out/ns3_sf5_sweep.jsonl
  python3 run_ns3_sf5.py check                 rows that must reproduce ns3_sf4 exactly, written to out/ns3_sf5_check.jsonl:
                                               upBps = 0, and upBps = 400 with --upRoundRobin=0
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_ns3_sf4 as R4  # noqa: E402

OUT = os.path.join(HERE, "out", "ns3_sf5.jsonl")
CHECK = os.path.join(HERE, "out", "ns3_sf5_check.jsonl")
SWEEP = os.path.join(HERE, "out", "ns3_sf5_sweep.jsonl")
RATES = (1000, 1600, 2400)   # bracket the slot simulator's break point
R4.LOG = os.path.join(HERE, "out", "ns3_sf5.log")
R4.TMP = os.environ.get("P6_TMP", R4.TMP)


def rr_jobs():
    return [j for j in R4.jobs() if j["x"].get("upBps")]


def check_jobs():
    J = R4.jobs()
    pick = lambda **k: [j for j in J if all(j.get(a) == b for a, b in k.items() if a != "x") and
                        all(j["x"].get(a) == b for a, b in k.get("x", {}).items())]
    out = []
    # unlimited rows of three parts: nothing in the patch may move them
    out += pick(part="uplink", fleet="sateliot", disc="fcfs", rep=0, x=dict(upBps=0))
    out += pick(part="final", fleet="sateliot", gw="k5", policy="sf1", disc="lcfs", rep=0)
    out += pick(part="feed", fleet="sateliot", disc="lcfs", rep=0, x=dict(feedBudget=40))
    # stratified over fleets, policies, disciplines and options, added after the third review
    out += pick(part="final", fleet="orbcomm", gw="k5", policy="sf1", disc="newest", rep=1)
    out += pick(part="final", fleet="kineis", gw="aws11", policy="sfe", disc="fcfs", rep=2)
    out += pick(part="final", fleet="orbcomm", gw="aws11", policy="bp", disc="fcfs", rep=3)
    out += pick(part="ml", fleet="orbcomm", rep=3, x=dict(monitorList=7))
    out += pick(part="lifeper", fleet="sateliot", gw="k5", rep=4, x=dict(life="device1"))
    out += pick(part="buffer", fleet="orbcomm", disc="newest", rep=0)
    out += pick(part="ablate", fleet="kineis", gw="k5", rep=1, x=dict(devKeepNewest=1))
    out += pick(part="period", fleet="sateliot", gw="aws11", disc="newest", rep=2)
    # the old rule through the new binary
    for j in pick(part="uplink", fleet="sateliot", disc="fcfs", rep=0, x=dict(upBps=400)):
        j = json.loads(json.dumps(j))
        j["x"]["upRoundRobin"] = 0
        out.append(j)
    return out


def sweep_jobs():
    return [R4.job("upsweep", "sateliot", "k5", "sf1", "fcfs", rep, n=919, upBps=b) for b in RATES for rep in range(R4.REPS)]


def main():
    os.nice(10)
    chk = len(sys.argv) > 1 and sys.argv[1] == "check"
    swp = len(sys.argv) > 1 and sys.argv[1] == "sweep"
    path = CHECK if chk else SWEEP if swp else OUT
    par = 3 if chk or swp or len(sys.argv) < 2 else min(int(sys.argv[1]), 3)
    binsha = R4.sha()
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if not r.get("error") and r.get("sha") == binsha:
                done.add(r["key"])
    J = [j for j in (check_jobs() if chk else sweep_jobs() if swp else rr_jobs()) if R4.key(j) not in done]
    cost = lambda j: j["n"] * R4.K[j["fleet"]] * j["days"]
    J.sort(key=lambda j: (cost(j), j["rep"]))
    print(len(J), "runs to do; binary", binsha, flush=True)
    lock, stop, n = threading.Lock(), threading.Event(), [0]

    def work(j):
        if stop.is_set():
            return
        if R4.disk_gb() < R4.MIN_DISK_GB:
            stop.set()
            print(f"STOP: {R4.disk_gb():.1f} GB free on /", flush=True)
            return
        r = R4.run(j, binsha)
        with lock:
            open(path, "a").write(json.dumps(r) + "\n")
            n[0] += 1
            print(time.strftime("%H:%M:%S"), f"{n[0]}/{len(J)}", r["part"], r["fleet"], r["n"], r["disc"], r["rep"],
                  ("ERR " + r["error"]) if r.get("error") else f"aoi {r['aoi_h']:.2f} served {r['served_share']:.3f} "
                  f"del {r['delivered_share']:.3f} {r['wall_s']}s", f"free {R4.disk_gb():.1f}G", flush=True)

    with ThreadPoolExecutor(par) as ex:
        list(ex.map(work, J))
    print("END", "stopped" if stop.is_set() else "complete", flush=True)


if __name__ == "__main__":
    main()
