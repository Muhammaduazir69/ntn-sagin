#!/usr/bin/env python3
"""Per-device custody lifetime and a margin from a finite-sample quantile of past changes (Nov 2026 revision).

For test week w: tau_d = 99th percentile of device d's delay on the replay of week w-1. The offset m is the
finite-sample (1 - alpha) quantile over the devices of s_d = p99(D_d on replay of week w-1) - p99(D_d on replay of
week w-2), which uses no test data. Expiry on the test week is the share of updates of device d with D > tau.
Rules scored: fleet-pooled model p99 (the former rule), per-device model p99, per-device replay p99, replay p99 + m.
    python3 sf_margin.py orbcomm k5 0     Output out/rev4/margin_<tag>.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402
from sf_replay import plan, first_available, DT  # noqa: E402

ALPHA = 0.10
BIG = 10 ** 12


def replay_d(P, vg, gr, xyz, rng, nt):
    vis = F.visibility(P, xyz[None, :], 20.0)[:, :, 0]
    gen = np.arange(rng.random() * 900.0, (nt - 1) * DT - 86400.0, 900.0)
    gi, k1, up, dl = first_available(P, vis, gr, xyz, gen, nt)
    good = dl < BIG
    return (dl[good] * DT - gen[good])


def main(fl, gw, w):
    tag = f"{fl}_{gw}_w{w}"
    z = np.load(os.path.join(HERE, "out", f"sf_{tag}.npz"))
    an = json.load(open(os.path.join(HERE, "out", f"an_{tag}.json")))
    amap = {(o["lat"], o["lon"]): o for o in an["devices"]}
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    gws = sf_mc.load_gateways(gw)
    S = len(sats)
    t1, P1, v1 = plan(sats, gws, 7.0 * w - 7.0)
    t2, P2, v2 = plan(sats, gws, 7.0 * w - 14.0)
    nt = len(t1)
    r1 = [sf_mc.intervals(v1[k]) for k in range(S)]
    r2 = [sf_mc.intervals(v2[k]) for k in range(S)]
    rng = np.random.default_rng(9 + w)
    dev, dd = z["dev"], z["d_SF1"]
    tau_r, tau_m, score, lat_, D_test = [], [], [], [], []
    for j, (lat, lon) in enumerate(z["buoys"]):
        o = amap.get((float(lat), float(lon)))
        d = dd[dev == j]
        d = d[np.isfinite(d)]
        if o is None or len(d) == 0:
            continue
        xyz = F.geodetic_to_ecef(np.array([lat]), np.array([lon]))[0]
        a = replay_d(P1, v1, r1, xyz, rng, nt)
        b = replay_d(P2, v2, r2, xyz, rng, nt)
        if len(a) == 0 or len(b) == 0:
            continue
        tau_r.append(np.percentile(a, 99))
        score.append(np.percentile(a, 99) - np.percentile(b, 99))
        tau_m.append(o["p99_pred"])
        lat_.append(lat)
        D_test.append(d)
    tau_r, tau_m, score, lat_ = map(np.array, (tau_r, tau_m, score, lat_))
    n = len(score)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    margin = max(0.0, float(np.quantile(score, level)))
    pooled = an["quant"]["pred_p99"]

    def score_rule(tau):
        e = np.array([np.mean(d > t) for d, t in zip(D_test, tau)])
        tot = sum(np.sum(d > t) for d, t in zip(D_test, tau)) / sum(len(d) for d in D_test)
        return dict(fleet=float(100 * tot), dev_ok=float(100 * np.mean(e <= 0.01)), dev_max=float(100 * e.max()),
                    tau_h=float(np.mean(tau) / 3600))
    sim_p99 = np.array([np.percentile(d, 99) for d in D_test])
    bins = [(0, 30), (30, 45), (45, 60), (60, 90)]
    latbin = {f"{a}-{b}": dict(n=int(np.sum((abs(lat_) >= a) & (abs(lat_) < b))),
                               p99_h=float(np.mean(sim_p99[(abs(lat_) >= a) & (abs(lat_) < b)]) / 3600) if np.any((abs(lat_) >= a) & (abs(lat_) < b)) else None)
              for a, b in bins}
    out = dict(n=n, margin_h=margin / 3600, level=float(level),
               pooled=score_rule(np.full(n, pooled)), dev_model=score_rule(tau_m), dev_replay=score_rule(tau_r),
               dev_margin=score_rule(tau_r + margin), latbin=latbin)
    json.dump(out, open(os.path.join(HERE, "out", "rev4", f"margin_{tag}.json"), "w"), indent=1)
    print(tag, {k: out[k] for k in ("margin_h", "pooled", "dev_replay", "dev_margin")}, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]))
