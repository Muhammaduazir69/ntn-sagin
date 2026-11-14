#!/usr/bin/env python3
"""Independent check of the T3451 gating in sf_finite.simulate: a per-device event loop that shares only the contact plan.
Modes: none; lookup; return (context must return, no timer). Must match simulate() in age to three decimals and in upload count.
    python3 check_gating_indep.py        Output out/rev4/gating_check.json
"""
import bisect
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_finite as SF  # noqa: E402
import sf_mc  # noqa: E402

PER, DT, N = 900.0, 20.0, 40


def loop(t, vg, pairs, n, seed, mode):
    nt, S = len(t), vg.shape[0]
    rng = np.random.default_rng(seed)
    ph = [rng.random() * PER for _ in range(n)]
    end = t[-1] - 3 * 86400.0
    gw = [np.flatnonzero(vg[k]) for k in range(S)]
    per = [[] for _ in range(n)]
    for i, row in enumerate(pairs):
        for d, k, s in row:
            per[d].append((i, k, s))

    def nextg(k, i):
        j = np.searchsorted(gw[k], i)
        return gw[k][j] if j < len(gw[k]) else None

    ages, ups = [], 0
    t0, t1 = 12 * 3600.0, end
    for d in range(n):
        gens = np.arange(ph[d], end, PER)
        gi = 0
        vis = {}
        for i, k, s in per[d]:
            vis.setdefault(i, []).append((k, s))
        byk = {}
        for i in sorted(vis):
            for k, _ in vis[i]:
                byk.setdefault(k, []).append(i)
        rec, pend, until, sk, f0 = [], [], -1, -1, None
        for i in sorted(vis):
            while gi < len(gens) and np.ceil(gens[gi] / DT) <= i:
                pend.append(gens[gi])
                gi += 1
            if not pend:
                continue
            c = vis[i]
            if mode == "lookup" and i < until:
                c = [x for x in c if x[0] == sk]
            if mode == "return" and sk >= 0:
                c = [x for x in c if x[0] == sk or (f0 is not None and nextg(x[0], f0) is not None and nextg(x[0], f0) <= i)]
            if not c:
                continue
            k = max(c, key=lambda x: x[1])[0]
            dl = nextg(k, i)
            ups += 1
            if dl is not None:
                rec += [(dl * DT, g) for g in pend]
            pend, sk, f0 = [], k, dl
            if mode == "lookup":
                best = None
                if dl is not None:
                    for m, st in byk.items():
                        a = nextg(m, dl)
                        if a is None:
                            continue
                        j = bisect.bisect_left(st, a)
                        if j < len(st) and (best is None or st[j] < best):
                            best = st[j]
                until = best if best is not None else nt
        rec.sort()
        cur, last, area = min(ph[d], t0), t0, 0.0
        for r, g in rec:
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
        ages.append(area / (t1 - t0) / 3600)
    return float(np.mean(ages)), ups


if __name__ == "__main__":
    os.nice(19)
    out = []
    for fl, g in (("orbcomm", "k5"), ("kineis", "k5"), ("sateliot", "k5"), ("orbcomm", "aws11")):
        bu = np.array([[b[1], b[2]] for b in sf_mc.load_buoys(150, np.random.default_rng(1))])[:N]
        t, vg, pairs = SF.contact_plan(fl, g, 0.0, 7, bu)
        for mode, kw in (("none", {}), ("lookup", dict(timer="lookup")), ("return", dict(context="return"))):
            a, u = loop(t, vg, pairs, N, 40, mode)
            r = SF.simulate(t, vg, pairs, N, PER, seed=40, tail_days=3.0, occ_tail_days=3.0, **kw)
            o = dict(fleet=fl, gw=g, mode=mode, loop_age=a, loop_up=u, sim_age=r["aoi_all_h"], sim_up=r["n_nas"], failed=r["n_failed"])
            out.append(o)
            print(o, flush=True)
    json.dump(out, open(os.path.join(HERE, "out", "rev4", "gating_check.json"), "w"), indent=0)
