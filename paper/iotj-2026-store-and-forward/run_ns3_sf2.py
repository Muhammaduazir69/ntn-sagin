#!/usr/bin/env python3
"""ns-3 campaign of the revised IoT-J study: sagin-store-forward-iot on real fleets (resumable jsonl).

Part main     fleets x gateway sets x {bp, sf1, sfe} x {FCFS, keep-newest} plus sf1 with LCFS,
              five replications (buoy subset and RngRun differ per replication).
Part finite   Orbcomm and Sateliot with five gateways, sf1, FCFS / LCFS / keep-newest, under
              (a) per-contact budgets of UP updates per device contact and FEED per gateway contact,
              (b) an on-board buffer of BUF updates.
Part lifetime FCFS sf1 with the custody lifetime set to the predicted 99th-percentile delay
              (Corollary 1, out/an_<fleet>_<gw>_w0.json), all six scenarios.
Per run: generated and delivered counts, served-buoy share, delay mean/median/p95/p99, AoI per buoy
after a 12 h warm-up (all buoys and served buoys), and the per-hop counters of the custody queues.
"""
import csv, json, os, subprocess, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NS3 = "/home/uzair/6g_ntn_ns3/ns-3-dev"
EXE = os.path.join(NS3, "build/contrib/ntn-sagin/examples/ns3.43-sagin-store-forward-iot-optimized")
OUT = os.path.join(HERE, "out", "ns3_sf2.jsonl")
FLEETS, GWS = ["orbcomm", "kineis", "sateliot"], ["k5", "aws11"]
DAYS, GEN_DAYS, N_DEV, REPS, WARM = 4.0, 3.0, 60, 5, 12 * 3600.0
UP, FEED, BUF, PKT_BYTES = 4, 20, 20, 52      # budgets in updates; custody counts 40 B payload + 12 B header


def buoy_subset(rep):
    rows = open(os.path.join(NS3, "contrib/ntn-sagin/data/buoys.csv")).read().strip().splitlines()
    rng = np.random.default_rng(100 + rep)
    pick = rng.choice(len(rows) - 1, size=N_DEV, replace=False)
    path = os.path.join(HERE, "out", f"buoys_rep{rep}.csv")
    open(path, "w").write(rows[0] + "\n" + "\n".join(rows[1:][i] for i in sorted(pick)) + "\n")
    return path


def aoi(gen, rx, t0, t1):
    """Time-average age over [t0, t1]; before the first reception after t0 the age runs from t0 - 0
    only if an update generated before t0 had arrived, otherwise from the start of the run."""
    order = np.argsort(rx)
    cur, last, area = 0.0, t0, 0.0
    for g, r in zip(gen[order], rx[order]):
        if r <= t0:
            cur = max(cur, g)
            continue
        if r > t1:
            break
        if g <= cur:
            continue
        area += ((r - cur) ** 2 - (last - cur) ** 2) / 2
        last, cur = r, g
    area += ((t1 - cur) ** 2 - (last - cur) ** 2) / 2
    return area / (t1 - t0)


def lifetime_h(fleet, gw):
    a = json.load(open(os.path.join(HERE, "out", f"an_{fleet}_{gw}_w0.json")))
    return a["quant"]["pred_p99"] / 3600.0


def run(job):
    part, fleet, gw, pol, disc, extra, rep, buoys = job
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        out = tf.name
    cmd = [EXE, f"--tle=contrib/ntn-sagin/data/{fleet}.tle", f"--devices={buoys}",
           f"--gateways=contrib/ntn-sagin/data/gateways_{gw}.csv", f"--nDev={N_DEV}", f"--days={DAYS}",
           f"--policy={pol}", f"--keepNewest={int(disc == 'newest')}", f"--lcfs={int(disc == 'lcfs')}",
           f"--out={out}", f"--RngRun={rep + 1}"] + [f"--{k}={v}" for k, v in extra.items()]
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(NS3, "build/lib"))
    base = dict(part=part, fleet=fleet, gw=gw, policy=pol, disc=disc, rep=rep, **{f"x_{k}": v for k, v in extra.items()})
    r = subprocess.run(["nice", "-n", "5"] + cmd, cwd=NS3, env=env, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        return dict(base, error=r.stderr[-300:])
    summ = dict(kv.split("=", 1) for kv in r.stdout.split() if "=" in kv)
    rows = list(csv.DictReader(open(out)))
    os.unlink(out)
    gen_n = int(summ["generated"])
    dev = np.array([int(x["device"]) for x in rows], int)
    gen = np.array([float(x["gen_s"]) for x in rows])
    rx = np.array([float(x["rx_s"]) for x in rows])
    d = (rx - gen) / 3600.0
    t1 = GEN_DAYS * 86400.0
    ages = np.array([aoi(gen[dev == k], rx[dev == k], WARM, t1) / 3600.0 for k in range(N_DEV)])
    served = np.array([np.any(dev == k) for k in range(N_DEV)])
    pct = lambda q: float(np.percentile(d, q)) if len(d) else float("nan")
    return dict(base, generated=gen_n, delivered=len(rows), delivered_share=len(rows) / gen_n,
                served_share=float(served.mean()), d_mean=float(d.mean()) if len(d) else float("nan"),
                d_median=pct(50), d_p95=pct(95), d_p99=pct(99), aoi_h=float(ages.mean()),
                aoi_served_h=float(ages[served].mean()) if served.any() else float("nan"),
                **{k: int(summ[k]) for k in ("replaced_sat", "replaced_dev", "expired_sat", "expired_dev",
                                            "dropped_sat", "dropped_dev")}, error="")


def jobs():
    J = []
    for rep in range(REPS):
        b = buoy_subset(rep)
        for fl in FLEETS:
            for gw in GWS:
                for pol, disc in (("bp", "fcfs"), ("sf1", "fcfs"), ("sfe", "fcfs"), ("sf1", "newest"),
                                  ("sfe", "newest"), ("sf1", "lcfs")):
                    J.append(("main", fl, gw, pol, disc, {}, rep, b))
        for fl in ("orbcomm", "sateliot"):
            for disc in ("fcfs", "lcfs", "newest"):
                J.append(("finite", fl, "k5", "sf1", disc, {"upBudget": UP, "feedBudget": FEED}, rep, b))
                J.append(("finite", fl, "k5", "sf1", disc, {"satBufferBytes": BUF * PKT_BYTES}, rep, b))
        for fl in FLEETS:
            for gw in GWS:
                J.append(("lifetime", fl, gw, "sf1", "fcfs", {"lifetimeH": "LIFE"}, rep, b))
    return J


def key(j):
    return json.dumps([j[0], j[1], j[2], j[3], j[4], sorted(j[5]), j[6]])


def main(par):
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            if not r.get("error"):
                ex = sorted(k[2:] for k in r if k.startswith("x_"))
                done.add(json.dumps([r["part"], r["fleet"], r["gw"], r["policy"], r["disc"], ex, r["rep"]]))
    todo = [j for j in jobs() if key(j) not in done]
    print(len(todo), "runs to do", flush=True)
    lock = threading.Lock()

    def work(j):
        if j[5].get("lifetimeH") == "LIFE":
            j = j[:5] + ({"lifetimeH": round(lifetime_h(j[1], j[2]), 4)},) + j[6:]
        r = run(j)
        with lock:
            open(OUT, "a").write(json.dumps(r) + "\n")
        print(r["part"], r["fleet"], r["gw"], r["policy"], r["disc"], r["rep"], "ERR" if r.get("error") else "ok", flush=True)

    # lifetime runs last: they need the Monte Carlo predictions
    todo.sort(key=lambda j: j[0] == "lifetime")
    with ThreadPoolExecutor(par) as ex:
        list(ex.map(work, todo))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
