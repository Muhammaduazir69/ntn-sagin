#!/usr/bin/env python3
"""Slot-level custody simulator on the SGP4 contact plan (Nov 2026 revision).

Queues on devices and on satellites, one discipline per hop, with optional limits the Monte Carlo sf_mc.py
does not have: a delivery budget per gateway contact, a shared uplink per satellite, an on-board Monitoring
List per device, any update period and population, any horizon. With no limits it must reproduce sf_mc.py
(checked by sf_finite_runs.py part 'check').

  dev/sat discipline: fcfs | lcfs | newest   (newest = keep only the newest update of each device)
  feed_budget        updates a satellite may deliver per gateway contact, 0 = unlimited
  up_bps             uplink rate a satellite shares among the devices it hears, 0 = unlimited
  up_share           'rr': one report per device per pass in device order, with a pointer kept per satellite across
                     slots; 'index': each slot restarts at the lowest index (the first campaign's rule, fixed priority)
  ml                 satellites per device in its Monitoring List, 0 = all
  context            "any": every satellite can accept a NAS procedure (contexts pushed to the fleet, TS 23.401 Annex O.2 NOTE 1).
                     "return": TS 23.401 cl. 4.13.9.1, an MME without the context shall reject, and after an upload only the
                     serving satellite or one that has seen a gateway since that upload's delivery holds it. Each pass of a
                     satellite over a device with something to send counts once, as a failed attempt or an accepted upload.
  timer              T3451 gating, TS 24.301 cl. 4.11.5.2: after an upload (an accept) the device starts no NAS procedure over
                     another satellite until the timer expires; the satellite that gave the value stays usable and restarts
                     it. None = no timer; "lookup" = return leg (9) of that upload; a dict device -> steps = a fixed value
"""
import os
import sys
from collections import deque

import numpy as np
from sgp4.api import jday

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402

DT = 20.0
PKT = 52


def contact_plan(fleet, gwset, start_day, days, buoys, sats=None, keep_gw=None):
    sats = sats if sats is not None else F.load_tles(os.path.join(HERE, "data_raw", f"{fleet}.tle"))
    gws = sf_mc.load_gateways(gwset)
    if keep_gw is not None:
        gws = [gws[i] for i in keep_gw]
    jd0, fr0 = jday(2026, 10, 30, 0, 0, 0)
    fr0 += start_day
    t = np.arange(0, days * 86400.0, DT)
    P = F.ecef_positions(sats, jd0, fr0, t)
    g = F.geodetic_to_ecef(np.array([x[1] for x in gws]), np.array([x[2] for x in gws]))
    vis_gw = F.visibility(P, g, 10.0).any(axis=2)
    xyz = F.geodetic_to_ecef(buoys[:, 0], buoys[:, 1])
    # sparse device-satellite pairs per step, with the sine of the elevation
    pairs = [[] for _ in range(len(t))]
    for d in range(len(buoys)):
        u = xyz[d] / np.linalg.norm(xyz[d])
        los = P - xyz[d]
        sin_e = (los @ u) / np.linalg.norm(los, axis=2)
        ks, ts = np.nonzero(sin_e >= np.sin(np.radians(20.0)))
        for k, ti, s in zip(ks.tolist(), ts.tolist(), sin_e[ks, ts].tolist()):
            pairs[ti].append((d, k, s))
    return t, vis_gw, pairs


def simulate(t, vis_gw, pairs, n_dev, period_s, seed=1, dev="fcfs", sat="fcfs", feed_budget=0, up_bps=0.0, ml=0,
             tail_days=3.0, warm_h=12.0, occ_tail_days=3.0, up_share="rr", timer=None, record_w=False, per_dev=False, context="any"):
    rng = np.random.default_rng(seed)
    nt, S = len(t), vis_gw.shape[0]
    gen_end = t[-1] - tail_days * 86400.0
    gen_at = {}
    first = {}
    for d in range(n_dev):
        ph = rng.random() * period_s
        first[d] = ph
        for g in np.arange(ph, gen_end, period_s):
            gen_at.setdefault(int(np.ceil(g / DT)), []).append((d, float(g)))
    allow = None
    if ml > 0:
        allow = [set(rng.choice(S, size=min(ml, S), replace=False).tolist()) for _ in range(n_dev)]
    dq = [deque() for _ in range(n_dev)]
    sq = [deque() for _ in range(S)]                 # fcfs/lcfs: (d, g)
    sn = [dict() for _ in range(S)]                  # newest: d -> g, insertion order kept by re-insert
    used = [0] * S
    was_up = [False] * S
    delivered = []
    sup_dev = sup_sat = uplinked = 0
    occ, occ_dist, occ_t = [], [], []
    up_cap = up_bps * DT / (8 * PKT) if up_bps > 0 else 0
    credit = [0.0] * S
    win_lo, win_hi = int(86400.0 / DT), int((t[-1] - occ_tail_days * 86400.0) / DT)
    gw_steps = [np.flatnonzero(vis_gw[k]).tolist() for k in range(S)]
    gw_at = {}
    for k in range(S):
        for i in gw_steps[k]:
            gw_at.setdefault(i, []).append(k)
    backlog_t = []
    n_nas = 0
    w_rec = {}
    gating = timer is not None or record_w or context == "return"
    n_failed = 0
    rej_all, rej_first = [], []                      # steps from a reject to the next satellite holding the context
    rej_since = {}
    last_seen = {}                                   # (d, k) -> last step a failed attempt's pass was in view
    if gating:
        from bisect import bisect_left
        dev_steps = {}
        for ti, row in enumerate(pairs):
            for d_, k_, _ in row:
                dev_steps.setdefault((d_, k_), []).append(ti)
        timer_until = [-1] * n_dev
        timer_sat = [-1] * n_dev
        last_sat = [-1] * n_dev
        last_del = [None] * n_dev                    # first gateway contact of the serving satellite after the last upload

        def has_context(d_, k_, i_):
            if last_sat[d_] < 0 or k_ == last_sat[d_]:
                return True
            if last_del[d_] is None:
                return False
            g_ = gw_steps[k_]
            j_ = bisect_left(g_, last_del[d_])
            return j_ < len(g_) and g_[j_] <= i_

        def next_with_context(d_, i_):
            """First step at or after i_ when the device sees a satellite holding its context; None if none on the plan."""
            if last_del[d_] is None:
                return None
            best_ = None
            for m_ in range(S):
                gm = gw_steps[m_]
                a_ = bisect_left(gm, last_del[d_])
                if a_ >= len(gm):
                    continue
                ds_ = dev_steps.get((d_, m_))
                if not ds_:
                    continue
                b_ = bisect_left(ds_, max(gm[a_], i_))
                if b_ < len(ds_) and (best_ is None or ds_[b_] < best_):
                    best_ = ds_[b_]
            return best_

        def return_steps(d_, k_, i_):
            """Steps from an upload at i_ to the first contact with a satellite fed after delivery, eq. (9)."""
            g_ = gw_steps[k_]
            j_ = bisect_left(g_, i_)
            if j_ >= len(g_):
                return None
            f0 = g_[j_]
            best_ = None
            for m_ in range(S):
                gm = gw_steps[m_]
                a_ = bisect_left(gm, f0)
                if a_ >= len(gm):
                    continue
                ds_ = dev_steps.get((d_, m_))
                if not ds_:
                    continue
                b_ = bisect_left(ds_, gm[a_])
                if b_ < len(ds_) and (best_ is None or ds_[b_] < best_):
                    best_ = ds_[b_]
            return None if best_ is None else best_ - i_
    day_marks = set(range(int(86400 / DT), nt, int(86400 / DT)))
    last_served = {}                                 # satellite -> device it served last (round robin)
    for i in range(nt):
        for d, g in gen_at.get(i, ()):
            if dev == "newest" and dq[d]:
                sup_dev += len(dq[d])
                dq[d].clear()
            dq[d].append(g)
        here = gw_at.get(i, ())
        here_set = set(here)
        # a contact starts: storage just before it, as in eq. (occ)
        for k in range(S):
            if (k in here_set) and not was_up[k]:
                used[k] = 0
                if win_lo <= i <= win_hi:
                    if sat == "newest":
                        occ.append(len(sn[k])); occ_dist.append(len(sn[k]))
                    else:
                        occ.append(len(sq[k])); occ_dist.append(len({x[0] for x in sq[k]}))
                    occ_t.append(i)
            was_up[k] = k in here_set
        # uplink: each device with a backlog offers it to the visible satellite with the highest elevation
        best = {}
        refused = {}
        for d, k, s in pairs[i]:
            if not dq[d] or (allow is not None and k not in allow[d]):
                continue
            if timer is not None and i < timer_until[d] and k != timer_sat[d]:
                continue
            if context == "return" and not has_context(d, k, i):
                if d not in refused or s > refused[d][1]:
                    refused[d] = (k, s)
                continue
            if d not in best or s > best[d][1]:
                best[d] = (k, s)
        for d, (k, _) in refused.items():
            if d in best:
                continue
            if last_seen.get((d, k)) != i - 1:       # a new pass of this satellite: one failed attempt, rejected
                n_failed += 1
                w_rej = next_with_context(d, i)
                if w_rej is not None:
                    rej_all.append(w_rej - i)
                    if not rej_since.get(d):
                        rej_first.append(w_rej - i)
                rej_since[d] = True
                if timer == "lookup":
                    n_ = next_with_context(d, i)
                    timer_until[d], timer_sat[d] = (n_ if n_ is not None else nt), -1
                elif isinstance(timer, dict):
                    timer_until[d], timer_sat[d] = i + timer[d], -1
            last_seen[(d, k)] = i
        if best:
            by = {}
            for d, (k, _) in best.items():
                by.setdefault(k, []).append(d)
            for k, ds in by.items():
                if up_cap:
                    credit[k] = min(credit[k] + up_cap, up_cap)   # the slot's airtime; no carry over
                    budget = int(credit[k] + rng.random())
                else:
                    budget = None
                ds = sorted(ds)
                if up_cap and up_share == "rr" and k in last_served:
                    p = last_served[k]
                    ds = [x for x in ds if x > p] + [x for x in ds if x <= p]
                active = list(ds)
                while active and (budget is None or budget > 0):
                    nxt = []
                    for d in active:
                        if budget is not None and budget <= 0:
                            break
                        if not dq[d]:
                            continue
                        if budget is None:
                            items = list(dq[d])
                            dq[d].clear()
                        else:
                            items = [dq[d].pop() if dev == "lcfs" else dq[d].popleft()]
                            budget -= 1
                            last_served[k] = d
                            if dq[d]:
                                nxt.append(d)
                        uplinked += len(items)
                        n_nas += 1
                        if gating:
                            w_ = return_steps(d, k, i)
                            last_sat[d] = k
                            rej_since[d] = False
                            j_ = bisect_left(gw_steps[k], i)
                            last_del[d] = gw_steps[k][j_] if j_ < len(gw_steps[k]) else None
                            if record_w and w_ is not None:
                                w_rec.setdefault(d, []).append(w_)
                            if timer == "lookup":
                                timer_until[d] = i + w_ if w_ is not None else nt
                                timer_sat[d] = k
                            elif isinstance(timer, dict):
                                timer_until[d] = i + timer[d]
                                timer_sat[d] = k
                        for g in items:
                            if sat == "newest":
                                if d in sn[k]:
                                    sup_sat += 1
                                    if g < sn[k][d]:
                                        continue
                                    del sn[k][d]
                                sn[k][d] = g
                            else:
                                sq[k].append((d, g))
                    active = nxt
        if i in day_marks:
            backlog_t.append((round(i * DT / 86400.0, 2), sum(len(q) for q in dq)))
        # delivery
        for k in here:
            if sat == "newest":
                while sn[k] and (feed_budget == 0 or used[k] < feed_budget):
                    d = next(iter(sn[k]))
                    g = sn[k].pop(d)
                    delivered.append((d, g, i * DT))
                    used[k] += 1
            else:
                q = sq[k]
                while q and (feed_budget == 0 or used[k] < feed_budget):
                    d, g = q.pop() if sat == "lcfs" else q.popleft()
                    delivered.append((d, g, i * DT))
                    used[k] += 1
    D = np.array(delivered) if delivered else np.zeros((0, 3))
    n_gen = sum(len(v) for v in gen_at.values())
    # age of information per device, from warm_h to the end of traffic
    t0, t1 = warm_h * 3600.0, gen_end
    mid = 0.5 * (t0 + t1)
    aoi, a1, a2, served = [], [], [], []
    for d in range(n_dev):
        s = D[:, 0] == d
        served.append(bool(s.any()))
        a = _aoi(D[s, 1], D[s, 2], t0, t1, first[d])
        aoi.append(a)
        a1.append(_aoi(D[s, 1], D[s, 2], t0, mid, first[d]))
        a2.append(_aoi(D[s, 1], D[s, 2], mid, t1, first[d]))
    delay = D[:, 2] - D[:, 1] if len(D) else np.zeros(0)
    sv = np.array(served)
    aoi_h_all = np.array(aoi) / 3600
    res = dict(generated=int(n_gen), delivered=int(len(D)), share=float(len(D) / max(n_gen, 1)),
               superseded_dev=int(sup_dev), superseded_sat=int(sup_sat), uplinked=int(uplinked),
               aoi_h=float(np.mean(np.array(aoi)[sv]) / 3600) if sv.any() else float("nan"),
               n_nas=int(n_nas), n_failed=int(n_failed), backlog_t=backlog_t, aoi_all_h=float(aoi_h_all.mean()), n_served=int(sv.sum()),
               aoi_all_p95_h=float(np.percentile(aoi_h_all, 95)), aoi_all_max_h=float(aoi_h_all.max()),
               aoi_all_first_h=float(np.mean(a1) / 3600), aoi_all_second_h=float(np.mean(a2) / 3600),
               aoi_first_h=float(np.mean(np.array(a1)[sv]) / 3600) if sv.any() else float("nan"),
               aoi_second_h=float(np.mean(np.array(a2)[sv]) / 3600) if sv.any() else float("nan"),
               d_mean_h=float(delay.mean() / 3600) if len(delay) else float("nan"),
               d_p99_h=float(np.percentile(delay, 99) / 3600) if len(delay) else float("nan"),
               occ_p99=float(np.percentile(occ, 99)) if occ else 0.0, occ_max=int(max(occ)) if occ else 0,
               occ_mean=float(np.mean(occ)) if occ else 0.0,
               occ_dist_p99=float(np.percentile(occ_dist, 99)) if occ_dist else 0.0, n_contacts=len(occ))
    if context == "return":
        over = int(5400 / DT)
        res.update(rej_n=len(rej_all), rej_over=int(sum(x > over for x in rej_all)), rej_first_n=len(rej_first),
                   rej_first_over=int(sum(x > over for x in rej_first)))
    if record_w:
        res["w_rec"] = w_rec
    if per_dev:
        res["aoi_per_dev"] = [round(float(x), 3) for x in aoi_h_all]
    return res


def heard_duty(pairs, S, allow=None):
    """Share of plan steps in which each satellite hears at least one device as that device's best: eta_k of Corollary 2."""
    busy = [0] * S
    for row in pairs:
        best = {}
        for d, k, s in row:
            if d not in best or s > best[d][1]:
                best[d] = (k, s)
        for k in {v[0] for v in best.values()}:
            busy[k] += 1
    return [b / len(pairs) for b in busy]


def _aoi(gen, rx, t0, t1, g0):
    o = np.argsort(rx)
    gen, rx = gen[o], rx[o]
    cur, last, area = min(g0, t0), t0, 0.0
    for g, r in zip(gen.tolist(), rx.tolist()):
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
