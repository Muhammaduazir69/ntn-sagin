#!/usr/bin/env python3
"""ns-3 campaign of the IoT-J study (Section VI-B): sagin-store-forward-iot on real fleets.

Fleets ORBCOMM, Kineis and Sateliot (CelesTrak element sets of 30 Oct 2026), gateway sets K5 and
AWS11, policies bp / sf1 / sfe, FCFS or keep-newest custody, 60 NOAA buoys, 4 days (3 days of updates
every 15 min, one day of drain), five replications with different buoy subsets and update phases.
Per run: delivered share, delay mean / median / p95 / p99, age of information, updates replaced.
Writes out/ns3_sf.csv.
"""
import csv
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NS3 = "/home/uzair/6g_ntn_ns3/ns-3-dev"
EXE = os.path.join(NS3, "build/contrib/ntn-sagin/examples/ns3.43-sagin-store-forward-iot-optimized")
FLEETS = ["orbcomm", "kineis", "sateliot"]
GWS = ["k5", "aws11"]
POLICIES = [("bp", 0), ("sf1", 0), ("sfe", 0), ("sf1", 1), ("sfe", 1)]
DAYS, GEN_DAYS, N_DEV, REPS = 4.0, 3.0, 60, 5


def buoy_subset(rep):
    rows = open(os.path.join(NS3, "contrib/ntn-sagin/data/buoys.csv")).read().strip().splitlines()
    head, body = rows[0], rows[1:]
    rng = np.random.default_rng(100 + rep)
    pick = rng.choice(len(body), size=N_DEV, replace=False)
    path = os.path.join(HERE, "out", f"buoys_rep{rep}.csv")
    open(path, "w").write(head + "\n" + "\n".join(body[i] for i in sorted(pick)) + "\n")
    return path


def aoi(gen, rx, t0, t1):
    order = np.argsort(rx)
    cur, last, area = t0, t0, 0.0
    for g, r in zip(gen[order], rx[order]):
        if r > t1:
            break
        if g <= cur:
            continue
        area += ((r - cur) ** 2 - (last - cur) ** 2) / 2
        last, cur = r, g
    area += ((t1 - cur) ** 2 - (last - cur) ** 2) / 2
    return area / (t1 - t0)


def run(job):
    fleet, gw, pol, keep, rep, buoys = job
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        out = tf.name
    cmd = [EXE, f"--tle=contrib/ntn-sagin/data/{fleet}.tle", f"--devices={buoys}",
           f"--gateways=contrib/ntn-sagin/data/gateways_{gw}.csv", f"--nDev={N_DEV}", f"--days={DAYS}",
           f"--policy={pol}", f"--keepNewest={keep}", f"--out={out}"]
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(NS3, "build/lib"))
    r = subprocess.run(cmd, cwd=NS3, env=env, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        return dict(fleet=fleet, gw=gw, policy=pol, keep=keep, rep=rep, error=r.stderr[-300:])
    summ = dict(kv.split("=", 1) for kv in r.stdout.split() if "=" in kv)
    rows = list(csv.DictReader(open(out)))
    os.unlink(out)
    period = 15 * 60.0
    generated = N_DEV * GEN_DAYS * 86400.0 / period
    dev = np.array([int(x["device"]) for x in rows])
    gen = np.array([float(x["gen_s"]) for x in rows])
    rx = np.array([float(x["rx_s"]) for x in rows])
    d = (rx - gen) / 3600.0
    ages = [aoi(gen[dev == k], rx[dev == k], 0.0, GEN_DAYS * 86400.0) / 3600.0 for k in range(N_DEV)]
    return dict(fleet=fleet, gw=gw, policy=pol, keep=keep, rep=rep, delivered=len(rows),
                delivered_share=len(rows) / generated,
                d_mean=float(d.mean()) if len(d) else np.nan, d_median=float(np.median(d)) if len(d) else np.nan,
                d_p95=float(np.percentile(d, 95)) if len(d) else np.nan,
                d_p99=float(np.percentile(d, 99)) if len(d) else np.nan,
                aoi_h=float(np.mean(ages)), replaced=int(summ.get("replaced", 0)), error="")


def main():
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    jobs = []
    for rep in range(REPS):
        b = buoy_subset(rep)
        for fleet in FLEETS:
            for gw in GWS:
                for pol, keep in POLICIES:
                    jobs.append((fleet, gw, pol, keep, rep, b))
    print(len(jobs), "runs", flush=True)
    import json
    import threading
    done_path = os.path.join(HERE, "out", "ns3_sf.jsonl")
    done = set()
    if os.path.exists(done_path):
        for line in open(done_path):
            r = json.loads(line)
            done.add((r["fleet"], r["gw"], r["policy"], int(r["keep"]), int(r["rep"])))
    todo = [j for j in jobs if (j[0], j[1], j[2], j[3], j[4]) not in done]
    lock = threading.Lock()

    def work(j):
        r = run(j)
        with lock:
            with open(done_path, "a") as fo:
                fo.write(json.dumps(r) + "\n")
        return r

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(work, todo))
    res = [json.loads(line) for line in open(done_path)]
    keys = ["fleet", "gw", "policy", "keep", "rep", "delivered", "delivered_share", "d_mean", "d_median",
            "d_p95", "d_p99", "aoi_h", "replaced", "error"]
    with open(os.path.join(HERE, "out", "ns3_sf.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in res:
            w.writerow({k: r.get(k, "") for k in keys})
    print("done", flush=True)


if __name__ == "__main__":
    main()
