#!/usr/bin/env python3
"""What a wait-timer rule costs, per upload (Nov 2026, round 4). Same event model as sf_timers.py.

After an upload at t the device waits tau, then meets the first satellite in view. The attempt succeeds if a satellite in
view at that step has seen a gateway since the data reached the ground. If not, it waits a fresh 30 to 60 min draw
(TS 24.301 cl. 4.11.5.2, item 2) and tries again. Scored: first-attempt success, failed attempts per upload, time to the
first success, excess over the earliest possible return W, contacts sat out while the first timer runs.
Rules: lookup (tau = W from the plan used), fallback (30 to 60 min), lastp95 (per-device 95th percentile of last week's
return leg), lag (lookup on last week's plan, one best lag per satellite fitted on the two weeks before, no test data).

  python3 sf_timer_cost.py orbcomm k5 0        truth: 30 Oct sets, week w. Output out/rev4/tcost_<tag>.json
  python3 sf_timer_cost.py orbcomm k5 held     truth: later-issued sets, week 0; also the lookup from the 30 Oct sets and
                                               delivery-time error per satellite for three plans. tcost_held_<fl>_<gw>.json
References are propagated in both modes; no delay is measured.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402
from sf_heldout import by_id  # noqa: E402
from sf_replay import plan, first_available, DT  # noqa: E402
from sf_timers import return_leg  # noqa: E402

BIG = 10 ** 12
RV = os.path.join(HERE, "out", "rev4")
PER = 900.0
STRIDE = int(os.environ.get("P6_STRIDE", "1"))


def runs(v):
    return [sf_mc.intervals(v[k]) for k in range(v.shape[0])]


def fit_lags(vq, vp):
    """Per satellite, the lag in 6 to 8 days that best maps the older fortnight onto its last five days (gateway masks)."""
    cand = np.arange(int(6 * 86400 / DT), int(8 * 86400 / DT))
    lags = []
    for k in range(vq.shape[0]):
        s = np.r_[vq[k], vp[k]].astype(np.int8)
        ref = s[-int(5 * 86400 / DT):]
        end = len(s)
        c = np.fft.irfft(np.fft.rfft(s - s.mean(), 2 * end) * np.fft.rfft((ref - ref.mean())[::-1], 2 * end), 2 * end)
        lags.append(int(cand[int(np.argmax(c[end - 1 - cand]))]))
    return np.array(lags)


def lagged(mask2, lags, nt):
    """Predicted masks of the test week from two earlier weeks, per satellite."""
    out = np.zeros((mask2.shape[0], nt), bool)
    n2 = mask2.shape[1]
    for k in range(mask2.shape[0]):
        idx = n2 + np.arange(nt) - lags[k]
        ok = (idx >= 0) & (idx < n2)
        out[k, ok] = mask2[k, idx[ok]]
    return out


def score(tau_steps, te, de, dev, gw, rng, W, ustarts=None):
    """Follow each upload to its first successful attempt. tau_steps: first timer, in plan steps. With ustarts (no timer) the
    device tries at the start of every later contact with the fleet instead of drawing a back-off."""
    S = len(gw)
    fed = np.stack([sf_mc.next_in(*gw[k], de) for k in range(S)])
    if ustarts is not None:
        at = np.searchsorted(ustarts, te, side="right")
        ts = np.where(at < len(ustarts), ustarts[np.minimum(at, len(ustarts) - 1)], BIG)
    else:
        ts = te + tau_steps
    n = len(te)
    done = np.zeros(n, bool)
    att = np.zeros(n, int)
    tsucc = np.full(n, BIG, dtype=np.int64)
    first = np.zeros(n, bool)
    for it in range(400 if ustarts is not None else 60):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        nx = np.stack([sf_mc.next_in(*dev[k], ts[live]) for k in range(S)])
        ta = nx.min(axis=0)
        none = ta >= BIG
        done[live[none]] = True           # no further contact on the plan: censored
        ok = ((nx == ta) & (fed[:, live] <= ta)).any(axis=0) & ~none
        att[live[~none]] += 1
        tsucc[live[ok]] = ta[ok]
        if it == 0:
            first[live[ok]] = True
        done[live[ok]] = True
        again = live[~ok & ~none]
        if ustarts is not None:
            at = np.searchsorted(ustarts, ta[~ok & ~none], side="right")
            ts[again] = np.where(at < len(ustarts), ustarts[np.minimum(at, len(ustarts) - 1)], BIG)
        else:
            ts[again] = ta[~ok & ~none] + (rng.uniform(1800.0, 3600.0, len(again)) / DT).astype(np.int64)
    g = tsucc < BIG
    return dict(n=int(n), n_ok=int(g.sum()), first=float(first.mean()), failed=float((att[g] - 1).mean()),
                wait_h=float((tsucc[g] - te[g]).mean() * DT / 3600), excess_h=float(((tsucc[g] - te[g]) * DT - W[g]).mean() / 3600))


def events(P, vis, gr, xyz, nt, rng):
    gen = np.arange(rng.random() * PER, (nt - 1) * DT - 2 * 86400.0, PER)
    _, k1, up, dl = first_available(P, vis, gr, xyz, gen, nt)
    g = dl < BIG
    return np.unique(np.c_[k1[g], up[g], dl[g]], axis=0)[::STRIDE]


def merge(acc, name, chunks):
    tot = sum(c["n_ok"] for c in chunks)
    acc[name] = dict(n=sum(c["n"] for c in chunks), censored=1 - tot / max(1, sum(c["n"] for c in chunks)),
                     first=float(np.average([c["first"] for c in chunks], weights=[c["n"] for c in chunks])),
                     **{k: float(np.average([c[k] for c in chunks], weights=[max(c["n_ok"], 1) for c in chunks])) for k in ("failed", "wait_h", "excess_h")})


def tau_lookup(te, de, dev_p, gr_p):
    S = len(gr_p)
    c = return_leg(dev_p, gr_p, de, range(S))
    return c


def main(fl, gw, mode):
    held = mode == "held"
    w = 0 if held else int(mode)
    path = os.path.join(RV, f"tcost_held_{fl}_{gw}.json" if held else f"tcost_{fl}_{gw}_w{w}.json")
    if os.path.exists(path):
        return
    old = by_id(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    ids = list(old)
    sats_old = [old[i] for i in ids]
    gws = sf_mc.load_gateways(gw)
    if held:
        new = by_id(os.path.join(HERE, "data_raw", "issued_2026-11-03", f"{fl}.tle"))
        sats_t = [new[i] for i in ids]
    else:
        sats_t = sats_old
    t, Pt, vt = plan(sats_t, gws, 7.0 * w)
    _, Pp, vp = plan(sats_old, gws, 7.0 * w - 7.0)
    _, Pq, vq = plan(sats_old, gws, 7.0 * w - 14.0)
    nt, S = len(t), len(ids)
    grt = runs(vt)
    grp = runs(vp)
    lags = fit_lags(vq, vp)
    grl = runs(lagged(np.c_[vq, vp], lags, nt))
    if held:
        _, Po, vo = plan(sats_old, gws, 0.0)
        gro = runs(vo)
    devs = json.load(open(os.path.join(RV, f"replay_{fl}_{gw}_w{w}.json")))["devices"]
    rng = np.random.default_rng(4040 + w)
    C = {k: [] for k in ("lookup", "none", "fallback", "lastp75", "lastp95", "lag", "older")}
    Her = {k: [] for k in ("older", "same", "lag")}
    Hk = []
    for o in devs:
        xyz = F.geodetic_to_ecef(np.array([o["lat"]]), np.array([o["lon"]]))[0]
        vd = F.visibility(Pt, xyz[None, :], 20.0)[:, :, 0]
        ev = events(Pt, vd, grt, xyz, nt, rng)
        if not len(ev):
            continue
        k_e, te, de = ev.T
        dev = runs(vd)
        Wc = return_leg(dev, grt, de, range(S))
        g = Wc < BIG
        k_e, te, de, Wc = k_e[g], te[g], de[g], Wc[g]
        if not len(te):
            continue
        W = (Wc - te) * DT
        C["lookup"].append(score(Wc - te, te, de, dev, grt, rng, W))
        ust = np.flatnonzero(np.diff(np.r_[0, vd.any(axis=0).astype(np.int8)]) == 1)
        C["none"].append(score(None, te, de, dev, grt, rng, W, ustarts=ust))
        C["fallback"].append(score((rng.uniform(1800.0, 3600.0, len(te)) / DT).astype(np.int64), te, de, dev, grt, rng, W))
        vdp = F.visibility(Pp, xyz[None, :], 20.0)[:, :, 0]
        devp = runs(vdp)
        evp = events(Pp, vdp, grp, xyz, nt, rng)
        if len(evp):
            cp = return_leg(devp, grp, evp[:, 2], range(S))
            wp = (cp[cp < BIG] - evp[:, 1][cp < BIG]) * DT
            for q_, nm_ in ((75, "lastp75"), (95, "lastp95")):
                tau = int(np.percentile(wp, q_) / DT) if len(wp) else 270
                C[nm_].append(score(np.full(len(te), tau), te, de, dev, grt, rng, W))
        # lookup on last week's plan at the fitted lag: its own delivery and return leg for this upload
        vdq = F.visibility(Pq, xyz[None, :], 20.0)[:, :, 0]
        devl = runs(lagged(np.c_[vdq, vdp], lags, nt))
        dl_l = np.array([sf_mc.next_in(*grl[k], np.array([u]))[0] for k, u in zip(k_e, te)])
        okl = dl_l < BIG
        cl = np.full(len(te), BIG, dtype=np.int64)
        if okl.any():
            cl[okl] = return_leg(devl, grl, dl_l[okl], range(S))
        tau_l = np.where(cl < BIG, cl - te, (rng.uniform(1800.0, 3600.0, len(te)) / DT).astype(np.int64))
        C["lag"].append(score(tau_l, te, de, dev, grt, rng, W))
        if held:
            vdo = F.visibility(Po, xyz[None, :], 20.0)[:, :, 0]
            devo = runs(vdo)
            dl_o = np.array([sf_mc.next_in(*gro[k], np.array([u]))[0] for k, u in zip(k_e, te)])
            oko = dl_o < BIG
            co = np.full(len(te), BIG, dtype=np.int64)
            co[oko] = return_leg(devo, gro, dl_o[oko], range(S))
            tau_o = np.where(co < BIG, np.maximum(co - te, 0), Wc - te)
            C["older"].append(score(tau_o, te, de, dev, grt, rng, W))
            dl_s = np.array([sf_mc.next_in(*grp[k], np.array([u]))[0] for k, u in zip(k_e, te)])
            Hk.append(k_e)
            for nm, dd in (("older", dl_o), ("same", dl_s), ("lag", dl_l)):
                e = np.full(len(te), np.nan)
                ok = dd < BIG
                e[ok] = np.abs(dd[ok] - de[ok]) * DT
                Her[nm].append(e)
    out = dict(fleet=fl, gw=gw, w=w)
    for k, v in C.items():
        if v:
            merge(out, k, v)
    if held:
        Hk = np.concatenate(Hk)
        perc = lambda x: dict(median=float(np.nanmedian(x)), p90=float(np.nanpercentile(x, 90)), p99=float(np.nanpercentile(x, 99)),
                              max=float(np.nanmax(x)), within60=float(np.nanmean(x[np.isfinite(x)] <= 60)))
        out["H"] = {k: perc(np.concatenate(v)) for k, v in Her.items()}
        eo = np.concatenate(Her["older"])
        per = {ids[k]: dict(n=int((Hk == k).sum()), p99=float(np.nanpercentile(eo[Hk == k], 99)), median=float(np.nanmedian(eo[Hk == k])))
               for k in range(S) if (Hk == k).any()}
        out["per_sat"] = per
        worst = max(per, key=lambda i: per[i]["p99"])
        keep = Hk != ids.index(worst)
        out["worst"] = dict(id=worst, share=float(np.mean(~keep)), **perc(eo[~keep]))  # statistics OF the outlier
        out["lags_days"] = [round(x * DT / 86400, 3) for x in lags]
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"), indent=0)
    os.replace(tmp, path)
    print(fl, gw, mode, {k: (round(v["first"], 3), round(v["failed"], 2), round(v["wait_h"], 2), round(v["excess_h"], 2)) for k, v in out.items() if isinstance(v, dict) and "first" in v}, out.get("H", ""), flush=True)


if __name__ == "__main__":
    os.nice(10)
    main(*sys.argv[1:4])
