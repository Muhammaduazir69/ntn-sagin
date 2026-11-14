#!/usr/bin/env python3
"""Message-level Monte Carlo on an SGP4 contact plan (Section V-A of the IoT-J manuscript).

Devices (ocean buoys from the NOAA NDBC station table) generate status updates; satellites take them
in custody when visible and deliver them at their next gateway contact. Policies:
  BP   bent pipe (transparent): delivered when one satellite sees the device and a gateway at once
  SF1  store and forward to the first visible satellite; when several come into view in the same
       sample, the one with the highest elevation (as in the ns-3 example)
  SFE  store and forward with earliest-delivery selection (contact-graph routing on the two-hop plan)
Outputs out/sf_<scenario>.npz: per-message delays per policy, per-satellite custody occupancy just
before each gateway contact (FCFS and keep-newest), and per-device age of information.

    python3 sf_mc.py --scenario orbcomm --days 7 --devices 150
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from sgp4.api import jday

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402

DT = 20.0


def load_buoys(n, rng, lat_max=None):
    rows = []
    for line in open(os.path.join(HERE, "data_raw", "ndbc_station_table.txt")):
        if line.startswith("#"):
            continue
        f = line.split("|")
        if len(f) < 7 or "buoy" not in f[2].lower():
            continue
        loc = f[6].split("(")[0].split()
        try:
            lat = float(loc[0]) * (1 if loc[1] == "N" else -1)
            lon = float(loc[2]) * (1 if loc[3] == "E" else -1)
        except (ValueError, IndexError):
            continue
        if lat_max is None or abs(lat) <= lat_max:
            rows.append((f[0], lat, lon))
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in sorted(idx)]


def load_gateways(gwset="aws11"):
    out = []
    with open(os.path.join(HERE, "data", f"gateways_{gwset}.csv")) as fh:
        for r in csv.DictReader(fh):
            out.append((r["name"], float(r["lat"]), float(r["lon"])))
    return out


def intervals(mask):
    """Start and end (exclusive) sample indices of the True runs of a boolean series."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


def next_in(starts, ends, t):
    """First sample index >= t inside one of the runs (vectorized over t); n_big if none."""
    j = np.searchsorted(ends, t, side="right")
    out = np.full(len(t), 10 ** 12, dtype=np.int64)
    ok = j < len(starts)
    out[ok] = np.maximum(t[ok], starts[j[ok]])
    return out


def run(scenario, days, n_dev, seed, dev_mask, gw_mask, period_min, out_tag=None, gwset="aws11", start_day=0.0):
    rng = np.random.default_rng(seed)
    if scenario in ("orbcomm", "kineis", "sateliot"):
        sats = F.load_tles(os.path.join(HERE, "data_raw", f"{scenario}.tle"))
    elif scenario.startswith("walker"):
        # walker-<total>-<planes>-<alt km>-<inc deg>
        _, n, p, h, i = scenario.split("-")
        sats = F.walker(int(n), int(p), 1, float(h) * 1e3, float(i))
    else:
        raise ValueError(scenario)
    # window start: 30 Oct 2026 plus start_day; the TLE epochs are within days of 30 Oct
    jd0, fr0 = jday(2026, 10, 30, 0, 0, 0)
    fr0 += start_day
    t = np.arange(0, days * 86400.0, DT)
    nt = len(t)
    P = F.ecef_positions(sats, jd0, fr0, t)
    buoys = load_buoys(n_dev, rng)
    gws = load_gateways(gwset)
    dev_xyz = F.geodetic_to_ecef(np.array([b[1] for b in buoys]), np.array([b[2] for b in buoys]))
    gw_xyz = F.geodetic_to_ecef(np.array([g[1] for g in gws]), np.array([g[2] for g in gws]))
    t0 = time.time()
    vis_gw = F.visibility(P, gw_xyz, gw_mask).any(axis=2)          # (S, T)
    S = len(sats)
    gw_runs = [intervals(vis_gw[k]) for k in range(S)]
    # status updates: periodic with a random phase per device
    per = period_min * 60.0
    res = {"BP": [], "SF1": [], "SFE": []}
    gen_all, dev_all = [], []
    sat_sf1, del_sf1, up_sf1 = [], [], []
    aoi = {"SF1": [], "SFE": []}
    # gateway gaps of each satellite that end inside the traffic window (day 1 to the end of traffic):
    # the occupancy at the end of such a gap is a steady-state sample, not warm-up or drain
    win_lo, win_hi = int(86400.0 / DT), int((t[-1] - 3 * 86400.0) / DT)
    gaps_win = []
    for k in range(S):
        st_k, en_k = gw_runs[k]
        sel = [(en_k[j - 1], st_k[j]) for j in range(1, len(st_k)) if win_lo <= st_k[j] <= win_hi]
        gaps_win.append(sel)
    seen_win = [np.zeros(len(g), int) for g in gaps_win]
    for d in range(len(buoys)):
        vis_d = F.visibility(P, dev_xyz[d:d + 1], dev_mask)[:, :, 0]   # (S, T)
        for k in range(S):
            if gaps_win[k]:
                cs = np.r_[0, np.cumsum(vis_d[k])]
                a_, b_ = np.array(gaps_win[k]).T
                seen_win[k] += (cs[b_] - cs[a_]) > 0
        dev_runs = [intervals(vis_d[k]) for k in range(S)]
        both_runs = intervals((vis_d & vis_gw).any(axis=0))
        phase = rng.random() * per
        gen = np.arange(phase, t[-1] - 3 * 86400.0, per)     # keep 3 days of tail for delivery
        gi = np.ceil(gen / DT).astype(np.int64)
        # bent pipe
        bp = next_in(*both_runs, gi)
        # per-satellite upload time and delivery time
        up = np.stack([next_in(*dev_runs[k], gi) for k in range(S)])            # (S, M)
        dl = np.stack([next_in(*gw_runs[k], np.minimum(up[k], nt)) for k in range(S)])
        dl = np.where(up >= 10 ** 12, 10 ** 12, dl)
        # first visible satellite; ties (same sample) go to the highest elevation
        umin = up.min(axis=0)
        tie = up == umin[None, :]
        ok_t = umin < nt
        idx = np.where(ok_t, umin, 0)
        los = P[:, idx, :] - dev_xyz[d][None, None, :]
        sin_e = (los @ (dev_xyz[d] / np.linalg.norm(dev_xyz[d]))) / np.linalg.norm(los, axis=2)
        k1 = np.argmax(np.where(tie, sin_e, -2.0), axis=0)
        m = np.arange(len(gi))
        sf1 = dl[k1, m]
        sfe = dl.min(axis=0)
        ke = np.argmin(dl, axis=0)
        for name, x in (("BP", bp), ("SF1", sf1), ("SFE", sfe)):
            res[name].append(np.where(x >= 10 ** 12, np.nan, (x - gi) * DT + (gi * DT - gen)))
        gen_all.append(gen)
        dev_all.append(np.full(len(gen), d))
        sat_sf1.append(k1)
        up_sf1.append(up[k1, m])
        del_sf1.append(sf1)
        # age of information at the collector: deliveries of this device's updates
        for name, x in (("SF1", sf1), ("SFE", sfe)):
            ok = x < 10 ** 12
            aoi[name].append(age_average(gen[ok], x[ok] * DT, gen[0], gen[-1]))
        if (d + 1) % 25 == 0:
            print(f"{scenario}: device {d + 1}/{len(buoys)} {time.time() - t0:.0f} s", flush=True)
    # custody occupancy of each satellite just before each gateway contact (SF1), FCFS vs newest
    k_all = np.concatenate(sat_sf1)
    u_all = np.concatenate(up_sf1)
    dl_all = np.concatenate(del_sf1)
    dv_all = np.concatenate(dev_all)
    occ_fcfs, occ_new, gaps_all, contacts_all = [], [], [], []
    for k in range(S):
        st, en = gw_runs[k]
        c, g = F.runs(vis_gw[k])
        gaps_all += list(g * DT)
        contacts_all += list(c * DT)
        sel = (k_all == k) & (dl_all < 10 ** 12) & (u_all < dl_all)    # held in custody at least one sample
        for s0 in st[1:]:
            if not (win_lo <= s0 <= win_hi):
                continue
            held = sel & (dl_all == s0)
            occ_fcfs.append(int(held.sum()))
            occ_new.append(int(len(np.unique(dv_all[held]))))
    tag = out_tag or scenario
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "out", f"sf_{tag}.npz"),
                        **{f"d_{k}": np.concatenate(v) for k, v in res.items()},
                        gen=np.concatenate(gen_all), dev=np.concatenate(dev_all),
                        occ_fcfs=np.array(occ_fcfs), occ_new=np.array(occ_new),
                        seen_win=np.concatenate([s for s in seen_win if len(s)]) if any(len(s) for s in seen_win) else np.array([]),
                        gw_gaps=np.array(gaps_all), gw_contacts=np.array(contacts_all),
                        aoi_sf1=np.array(aoi["SF1"]), aoi_sfe=np.array(aoi["SFE"]),
                        buoys=np.array([[b[1], b[2]] for b in buoys]), n_sats=S, period=per,
                        sf1_sat=k_all, sf1_up=u_all, sf1_del=dl_all)
    print(f"{scenario}: done {time.time() - t0:.0f} s", flush=True)


def age_average(gen, rec, t_start, t_end):
    """Time-average age of information at the collector over [t_start, t_end] from (generation,
    reception) pairs; updates that arrive out of order or older than the current one are ignored."""
    order = np.argsort(rec)
    gen, rec = gen[order], rec[order]
    cur_gen = t_start
    last = t_start
    area = 0.0
    for g, r in zip(gen, rec):
        if r > t_end:
            break
        if g <= cur_gen:
            continue
        # age grows linearly from (last - cur_gen) to (r - cur_gen)
        area += ((r - cur_gen) ** 2 - (last - cur_gen) ** 2) / 2
        last, cur_gen = r, g
    area += ((t_end - cur_gen) ** 2 - (last - cur_gen) ** 2) / 2
    return area / (t_end - t_start)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="orbcomm")
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--devices", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dev-mask", type=float, default=20.0)
    ap.add_argument("--gw-mask", type=float, default=10.0)
    ap.add_argument("--period-min", type=float, default=15.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--gwset", default="aws11")
    ap.add_argument("--start-day", type=float, default=0.0)
    a = ap.parse_args()
    run(a.scenario, a.days, a.devices, a.seed, a.dev_mask, a.gw_mask, a.period_min, a.tag, a.gwset, a.start_day)
