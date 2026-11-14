#!/usr/bin/env python3
"""The paper's ns-3 campaign on the first binary (Nov 2026), out/ns3_sf4.jsonl. Rows with upBps > 0 are superseded by
run_ns3_sf5.py (per-report round robin); make_macros.py keeps them only as the record of the old rule.

Binary: sim/bin/p6-sf-iot, built by ns3_patch/build.sh outside the shared build. Its SHA-256 is in every row, and a
row with another hash does not count as done. Resumable: run it again and it skips finished rows.

  python3 run_ns3_sf4.py [PARALLEL=3] [--parts a,b] [--reps N]
This refers to the first binary, 824142ca..., rebuilt from ns3_patch/sagin-store-forward-iot.first_binary.cc. Do not run it
with the current binary: every row would count as not done and a second hash would trip make_macros.py.

Part final    (m10) fleets x gateway sets x {bp, sf1, sfe} x {fcfs, newest} plus sf1 lcfs: 3x2x7x5 = 210, occupancy log
Part ablate   (M4, M5) sf1 fcfs, devKeepNewest only, satKeepNewest only: 3x2x2x5 = 60, occupancy log
Part period   (M4) sf1, periodMin 60, fcfs and keep-newest: 3x2x2x5 = 60, occupancy log
Part feed     (Fig. 6) Orbcomm and Sateliot, five gateways, four budgets, fcfs/lcfs/newest: 2x4x3x5 = 120
Part feedlong (M7) as feed, 14 days, three budgets (tightest, just under the limit, well above): 2x3x3x5 = 90
Part buffer   Orbcomm and Sateliot, five gateways, on-board buffer of 20 updates, fcfs/lcfs/newest: 2x3x5 = 30
Part uplink   (M10) three fleets, five gateways, 150 devices, 0, 400 and 4000 b/s, fcfs/newest: 3x3x2x5 = 90
Part upall    (M10) Sateliot, five gateways, all 919 stations, 0 and 400 b/s, fcfs/newest: 2x2x5 = 20
Part ml       (M10) three fleets, five gateways, monitorList 1 and K/2, fcfs: 3x2x5 = 30
Part lifetime fleet-pooled lifetime at the predicted p99, counted end to end, fcfs: 3x2x5 = 30
Part lifeper  (M11) per-device lifetime at the replay p99 of the week before, without and with the margin: 60
Part upall also has LCFS and device-only keep-newest at 400 b/s: 2x5 = 10
Part uplong (M10) Sateliot, five gateways, all 919 stations, 400 b/s, 14 days, fcfs/lcfs/newest: 3x5 = 15
Total 825 runs.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_ns3_sf3 as R3  # noqa: E402  (aoi, lifetime_h and the fixed design constants only)

NS3 = R3.NS3
EXE = os.path.join(HERE, "bin", "p6-sf-iot")
OUT = os.path.join(HERE, "out", "ns3_sf4.jsonl")
LOG = os.path.join(HERE, "out", "ns3_sf4.log")
LIFE = os.path.join(HERE, "out", "lifeper")
TMP = os.environ.get("P6_TMP", "/tmp/p6sf4")
FLEETS, GWS = R3.FLEETS, R3.GWS
REPS, N_DEV, DAYS, BUF, PKT = 5, 60, 4.0, 20, 52
K = {"orbcomm": 14, "kineis": 25, "sateliot": 4}
FEEDS = R3.FEEDS
MIN_DISK_GB = 3.0
MIN_RAM_GB = 3.0


def sha():
    return hashlib.sha256(open(EXE, "rb").read()).hexdigest()


def buoys(rep, n=N_DEV):
    """Same draw as run_ns3_sf3 for n = 60 (seed 100 + rep), so replications keep their buoy subsets."""
    path = os.path.join(HERE, "out", f"buoys_rep{rep}.csv" if n == N_DEV else f"buoys_n{n}_rep{rep}.csv")
    rows = open(os.path.join(NS3, "contrib/ntn-sagin/data/buoys.csv")).read().strip().splitlines()
    pick = np.random.default_rng(100 + rep).choice(len(rows) - 1, size=n, replace=False)
    tmp = path + f".{os.getpid()}.{threading.get_ident()}"
    open(tmp, "w").write(rows[0] + "\n" + "\n".join(rows[1:][i] for i in sorted(pick)) + "\n")
    os.replace(tmp, path)
    return path


def lifetimes(fl, gw, rep, margin):
    """Per-device replay p99 on the week before the run, plus the margin of out/rev4/margin_*. Kept in
    out/lifeper, one small file per case, so the inputs of the run can be read back."""
    path = os.path.join(LIFE, f"{fl}_{gw}_rep{rep}_m{margin}.csv")
    if os.path.exists(path):
        return path
    import sf_core as F
    import sf_mc
    from sf_margin import replay_d
    from sf_replay import plan
    rows = open(buoys(rep)).read().strip().splitlines()[1:]
    sites = [tuple(float(x) for x in r.split(",")[1:3]) for r in rows[:N_DEV]]
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    t, P, v = plan(sats, sf_mc.load_gateways(gw), -7.0)
    gr = [sf_mc.intervals(v[k]) for k in range(len(sats))]
    rng = np.random.default_rng(rep)
    m = json.load(open(os.path.join(HERE, "out", "rev4", f"margin_{fl}_{gw}_w0.json")))["margin_h"] if margin else 0.0
    life = []
    for lat, lon in sites:
        d = replay_d(P, v, gr, F.geodetic_to_ecef(np.array([lat]), np.array([lon]))[0], rng, len(t))
        life.append(np.percentile(d, 99) / 3600.0 + m if len(d) else 48.0)
    os.makedirs(LIFE, exist_ok=True)
    tmp = path + f".{os.getpid()}"
    open(tmp, "w").write("\n".join(f"{x:.4f}" for x in life) + "\n")
    os.replace(tmp, path)
    return path


def job(part, fl, gw, pol, disc, rep, n=N_DEV, days=DAYS, **x):
    return dict(part=part, fleet=fl, gw=gw, policy=pol, disc=disc, rep=rep, n=n, days=days, x=x)


def key(j):
    return json.dumps([j["part"], j["fleet"], j["gw"], j["policy"], j["disc"], j["rep"], j["n"], j["days"],
                       sorted(j["x"].items())])


def jobs():
    J = []
    for rep in range(REPS):
        for fl in FLEETS:
            for gw in GWS:
                for pol in ("bp", "sf1", "sfe"):
                    for d in ("fcfs", "newest"):
                        J.append(job("final", fl, gw, pol, d, rep, occ=1))
                J.append(job("final", fl, gw, "sf1", "lcfs", rep, occ=1))
                J.append(job("ablate", fl, gw, "sf1", "fcfs", rep, devKeepNewest=1, occ=1))
                J.append(job("ablate", fl, gw, "sf1", "fcfs", rep, satKeepNewest=1, occ=1))
                for d in ("fcfs", "newest"):
                    J.append(job("period", fl, gw, "sf1", d, rep, periodMin=60, occ=1))
                J.append(job("lifetime", fl, gw, "sf1", "fcfs", rep, life="pooled"))
                for mg in (0, 1):
                    J.append(job("lifeper", fl, gw, "sf1", "fcfs", rep, life=f"device{mg}"))
        for fl, feeds in FEEDS.items():
            for lv, fb in enumerate(feeds):
                for d in ("fcfs", "lcfs", "newest"):
                    J.append(job("feed", fl, "k5", "sf1", d, rep, feedBudget=fb))
                    if lv in (0, 1, 3):
                        J.append(job("feedlong", fl, "k5", "sf1", d, rep, days=14.0, feedBudget=fb))
            for d in ("fcfs", "lcfs", "newest"):
                J.append(job("buffer", fl, "k5", "sf1", d, rep, satBufferBytes=BUF * PKT))
        for fl in FLEETS:
            for bps in (0, 400, 4000):
                for d in ("fcfs", "newest"):
                    J.append(job("uplink", fl, "k5", "sf1", d, rep, n=150, upBps=bps))
            for m in (1, max(1, K[fl] // 2)):
                J.append(job("ml", fl, "k5", "sf1", "fcfs", rep, monitorList=m, monitorSeed=rep + 1))
        for bps in (0, 400):
            for d in ("fcfs", "newest"):
                J.append(job("upall", "sateliot", "k5", "sf1", d, rep, n=919, upBps=bps))
        # added after the hostile review: the closest baselines and the run-length check at the overload point
        J.append(job("upall", "sateliot", "k5", "sf1", "lcfs", rep, n=919, upBps=400))
        J.append(job("upall", "sateliot", "k5", "sf1", "fcfs", rep, n=919, upBps=400, devKeepNewest=1))
        for d in ("fcfs", "lcfs", "newest"):
            J.append(job("uplong", "sateliot", "k5", "sf1", d, rep, n=919, days=14.0, upBps=400))
    return J


def run(j, binsha):
    fl, gw, pol, disc, rep, n, days = j["fleet"], j["gw"], j["policy"], j["disc"], j["rep"], j["n"], j["days"]
    x = dict(j["x"])
    os.makedirs(TMP, exist_ok=True)
    tag = hashlib.md5(key(j).encode()).hexdigest()[:12]
    out, occ = os.path.join(TMP, tag + ".csv"), None
    if x.pop("occ", None):
        occ = os.path.join(TMP, tag + ".occ")
        x["occLog"] = occ
    life = x.pop("life", None)
    if life == "pooled":
        x.update(lifetimeH=round(R3.lifetime_h(fl, gw), 4), lifetimeE2E=1)
    elif life:
        x["lifetimeCsv"] = lifetimes(fl, gw, rep, int(life[-1]))
    cmd = [EXE, f"--tle=contrib/ntn-sagin/data/{fl}.tle", f"--devices={buoys(rep, n)}",
           f"--gateways=contrib/ntn-sagin/data/gateways_{gw}.csv", f"--nDev={n}", f"--days={days}",
           f"--policy={pol}", f"--keepNewest={int(disc == 'newest')}", f"--lcfs={int(disc == 'lcfs')}",
           f"--out={out}", f"--RngRun={rep + 1}"] + [f"--{k}={v}" for k, v in x.items()]
    base = dict(key=key(j), part=j["part"], fleet=fl, gw=gw, policy=pol, disc=disc, rep=rep, n=n, days=days,
                sha=binsha, **{f"x_{k}": v for k, v in j["x"].items()})
    if "lifetimeH" in x:
        base["lifetime_h"] = x["lifetimeH"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=NS3, capture_output=True, text=True, timeout=6 * 3600)
        if r.returncode != 0:
            return dict(base, error=r.stderr[-300:])
        summ = dict(kv.split("=", 1) for kv in r.stdout.split() if "=" in kv)
        rows = list(csv.DictReader(open(out)))
        gen_n = int(summ["generated"])
        dev = np.array([int(a["device"]) for a in rows], int)
        gen = np.array([float(a["gen_s"]) for a in rows])
        rx = np.array([float(a["rx_s"]) for a in rows])
        d = (rx - gen) / 3600.0
        t1 = (days - 1) * 86400.0
        ages = np.array([R3.aoi(gen[dev == k], rx[dev == k], R3.WARM, t1) / 3600.0 for k in range(n)])
        mid = (R3.WARM + t1) / 2
        a1 = np.array([R3.aoi(gen[dev == k], rx[dev == k], R3.WARM, mid) / 3600.0 for k in range(n)])
        a2 = np.array([R3.aoi(gen[dev == k], rx[dev == k], mid, t1) / 3600.0 for k in range(n)])
        served = np.array([np.any(dev == k) for k in range(n)])
        pct = lambda q: float(np.percentile(d, q)) if len(d) else float("nan")
        rr = dict(base, generated=gen_n, delivered=len(rows), delivered_share=len(rows) / gen_n,
                  served_share=float(served.mean()), d_mean=float(d.mean()) if len(d) else float("nan"),
                  d_median=pct(50), d_p95=pct(95), d_p99=pct(99), aoi_h=float(ages.mean()),
                  aoi_served_h=float(ages[served].mean()) if served.any() else float("nan"),
                  aoi_first_h=float(a1[served].mean()) if served.any() else float("nan"),
                  aoi_second_h=float(a2[served].mean()) if served.any() else float("nan"),
                  wall_s=round(time.time() - t0, 1), error="",
                  **{k: int(summ[k]) for k in ("replaced_sat", "replaced_dev", "expired_sat", "expired_dev", "dropped_sat",
                                               "dropped_dev", "mac_drops", "left_sat", "left_dev")})
        # generated = delivered + superseded + expired + dropped + still held + lost in flight
        rr["unaccounted"] = gen_n - len(rows) - sum(rr[k] for k in (
            "replaced_sat", "replaced_dev", "expired_sat", "expired_dev", "dropped_sat", "dropped_dev", "left_sat", "left_dev"))
        rr["uplinked"] = gen_n - sum(rr[k] for k in ("replaced_dev", "expired_dev", "dropped_dev", "left_dev"))
        if n <= N_DEV or j["x"].get("upBps"):
            rr["aoi_per_buoy"] = [round(float(a), 4) for a in ages]
            rr["served_per_buoy"] = [bool(a) for a in served]
        if j["part"] == "lifeper":
            rr["delivered_per_buoy"] = [int(a) for a in np.bincount(dev, minlength=n)]
        if occ:
            a = np.loadtxt(occ, delimiter=",", skiprows=1, ndmin=2)
            v = a[(a[:, 0] >= 86400.0) & (a[:, 0] <= t1), 2] if a.size else np.zeros(0)
            rr["occ_p99"] = float(np.percentile(v, 99)) if len(v) else 0.0
            rr["occ_mean"] = float(v.mean()) if len(v) else 0.0
            rr["occ_max"] = float(v.max()) if len(v) else 0.0
            rr["occ_n"] = int(len(v))
        return rr
    except Exception as e:  # noqa: BLE001  a failed row is reported and retried on the next pass
        return dict(base, error=f"{type(e).__name__}: {e}"[:300])
    finally:
        for p in (out, occ):
            if p and os.path.exists(p):
                os.unlink(p)


def disk_gb():
    s = os.statvfs("/")
    return s.f_bavail * s.f_frsize / 2 ** 30


def ram_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 2 ** 20
    return 99.0


def main():
    os.nice(10)  # inherited by every ns-3 run
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    par = min(int(args[0]) if args else 3, 3)
    parts = reps = None
    for i, a in enumerate(sys.argv):
        if a == "--parts":
            parts = set(sys.argv[i + 1].split(","))
        if a == "--reps":
            reps = int(sys.argv[i + 1])
    binsha = sha()
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            if not r.get("error") and r.get("sha") == binsha:
                done.add(r["key"])
    J = [j for j in jobs() if key(j) not in done]
    if parts:
        J = [j for j in J if j["part"] in parts]
    if reps:
        J = [j for j in J if j["rep"] < reps]
    # cheap configurations first, so that a stop leaves the most finished rows
    cost = lambda j: j["n"] * K[j["fleet"]] * j["days"] * (2 if j["policy"] == "sfe" else 1)
    J.sort(key=lambda j: (j["rep"], cost(j)))
    lock = threading.Lock()
    stop = threading.Event()
    n_done = [0]
    print(len(J), "runs to do; binary", binsha, flush=True)

    def work(j):
        if stop.is_set():
            return
        if disk_gb() < MIN_DISK_GB:
            stop.set()
            print(f"STOP: {disk_gb():.1f} GB free on /", flush=True)
            return
        big = j["n"] * K[j["fleet"]] > 3000
        while big and ram_gb() < MIN_RAM_GB + 2 and not stop.is_set():
            time.sleep(30)
        r = run(j, binsha)
        with lock:
            with open(OUT, "a") as fo:
                fo.write(json.dumps(r) + "\n")
            n_done[0] += 1
            msg = (f"{time.strftime('%H:%M:%S')} {n_done[0]}/{len(J)} {r['part']} {r['fleet']} {r['gw']} {r['policy']} "
                   f"{r['disc']} rep{r['rep']} {('ERR ' + r['error']) if r.get('error') else 'ok ' + str(r['wall_s']) + 's'} "
                   f"free {disk_gb():.1f}G")
            print(msg, flush=True)
            open(LOG, "a").write(msg + "\n")

    with ThreadPoolExecutor(par) as ex:
        list(ex.map(work, J))
    print("END", "stopped" if stop.is_set() else "complete", flush=True)


if __name__ == "__main__":
    main()
