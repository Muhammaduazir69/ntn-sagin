#!/usr/bin/env python3
"""Peak-age schedulers of Ari et al. (Asilomar 2025) re-implemented on contact models fitted to an SGP4 plan.

Their setting: a device with a budget of m uploads over T slots, generate-at-will, uplink availability a 2-state
Markov chain, S&F downlink delay an N-state Markov chain, objective the peak age. Their parameters are set by hand;
here they are fitted to last week's plan (assumption of this re-implementation): slot 20 min, T one week, N = 3 hold
states from the terciles of the positive holds, and evaluation on the test week's plan.
  periodic   uploads aimed at k_i = round(i T / (m + 1)), sent at the first contact at or after k_i (their TSw/CC,
             with the favorable reading that an aim without contact waits for the next one)
  smdp       backward induction on (slot, U, D, budget left, age) with the fitted chains (their TSwCCC)
  plan       min-max peak age with the test week's plan known (their TSwFCC, a lower bound), bisection and greedy
  contact    this article's keep-newest: one fresh update at every contact, no budget (uploads counted)
Age is in hours; a delivery at slot s of an update generated at u leaves age s - u.
    python3 sf_ari.py orbcomm k5 0     Output out/rev4/ari_<tag>.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402
import sf_mc  # noqa: E402
from sf_replay import plan, DT  # noqa: E402

SLOT = 1200.0
BUDGETS = (7, 14, 28, 56)
N_DEV = 40
BIG = 10 ** 12


def slot_plan(P, vg_runs, xyz, nt):
    """For each slot: is a satellite in view, and the hold in slots of the earliest-reached satellite in that slot."""
    vis = F.visibility(P, xyz[None, :], 20.0)[:, :, 0]
    n = int(SLOT / DT)
    T = nt // n
    U = np.zeros(T, bool)
    H = np.zeros(T, int)
    for s in range(T):
        lo, hi = s * n, (s + 1) * n
        sub = vis[:, lo:hi]
        if not sub.any():
            continue
        k_t = np.argmax(sub, axis=1)
        have = sub.any(axis=1)
        tt = np.where(have, k_t, 10 ** 9)
        k = int(np.argmin(tt))
        t0 = lo + int(tt[k])
        d = sf_mc.next_in(*vg_runs[k], np.array([t0]))[0]
        if d >= BIG:
            continue
        U[s] = True
        H[s] = int(np.ceil((d - t0) * DT / SLOT))
    return U, H


def fit(U, H, states=3):
    u = U.astype(int)
    pu = np.ones((2, 2)) * 1.0
    for a, b in zip(u[:-1], u[1:]):
        pu[a, b] += 1
    pu = pu / pu.sum(axis=1, keepdims=True)
    hv = H[U]
    if len(hv) == 0 or hv.max() == 0:
        edges, d = np.array([0.0]), np.array([0, 0, 0])
    else:
        edges = np.quantile(hv, [1 / 3, 2 / 3])
    cls = np.digitize(hv, edges) if len(hv) else np.zeros(0, int)
    d = np.array([int(np.round(hv[cls == c].mean())) if np.any(cls == c) else 0 for c in range(states)])
    pd = np.ones((states, states))
    for a, b in zip(cls[:-1], cls[1:]):
        pd[a, b] += 1
    pd = pd / pd.sum(axis=1, keepdims=True)
    return pu, pd, d, edges


def smdp(T, m, pu, pd, d, amax):
    """Backward induction for min E[max age]. V[t][u, c, r, a]: expected peak from slot t with server age a; policy
    True means upload now. Transmission is treated as an immediate jump of the age to d[c] with peak a + d[c]."""
    C = len(d)
    ages = np.arange(amax + 1)
    # terminal value: the age itself at the horizon
    V = np.zeros((2, C, m + 1, amax + 1))
    V[:] = ages
    pol = np.zeros((T, 2, C, m + 1, amax + 1), bool)
    for t in range(T - 1, -1, -1):
        # expectation over the next (u', c')
        EV = np.zeros_like(V)
        for u in (0, 1):
            for c in range(C):
                EV[u, c] = sum(pu[u, a] * pd[c, b] * V[a, b] for a in (0, 1) for b in range(C))
        nxt = np.minimum(ages + 1, amax)
        wait = np.maximum(ages[None, None, None, :], EV[:, :, :, nxt])
        new = wait.copy()
        for c in range(C):
            ja = np.minimum(d[c] + 1, amax)
            for r in range(1, m + 1):
                send = np.maximum(ages + d[c], EV[1, c, r - 1, ja])
                better = send < wait[1, c, r]
                pol[t, 1, c, r] = better
                new[1, c, r] = np.minimum(send, wait[1, c, r])
        V = new
    return pol


def evaluate(sends, H, T):
    """Peak and mean age (slots) given the upload slots and the hold of each slot, on the real plan."""
    rec = sorted((s + H[s], s) for s in sends)
    cur = 0
    t = 0
    peak = 0
    area = 0.0
    for r, g in rec:
        if r >= T:
            break
        if g < cur:
            continue
        peak = max(peak, r - cur)
        area += ((r - cur) ** 2 - (t - cur) ** 2) / 2
        t, cur = r, g
    peak = max(peak, T - cur)
    area += ((T - cur) ** 2 - (t - cur) ** 2) / 2
    return peak, area / T


def run_policy_periodic(U, T, m):
    aims = [int(round(i * T / (m + 1))) for i in range(1, m + 1)]
    out, last = [], -1
    for a in aims:
        s = np.flatnonzero(U[max(a, last + 1):])
        if len(s):
            last = max(a, last + 1) + int(s[0])
            out.append(last)
    return out


def run_policy_smdp(U, H, T, m, pol, edges, amax):
    r, a, out = m, 0, []
    for s in range(T):
        if U[s]:
            c = int(np.digitize(H[s], edges))
            if r > 0 and pol[s, 1, c, r, min(a, amax)]:
                out.append(s)
                r -= 1
                a = H[s] + 1
                continue
        a = min(a + 1, amax)
    return out


def plan_optimal(U, H, T, m):
    """Min-max peak age with the plan known: for a target P choose, greedily, the latest upload whose delivery keeps
    the age below P. Bisection over P."""
    slots = np.flatnonzero(U)

    def feasible(P):
        cur, used, out = 0, 0, []
        while True:
            if T - cur <= P:
                return out
            cand = [s for s in slots if s + H[s] - cur <= P and s >= cur]
            cand = [s for s in cand if s > cur or not out]
            if not cand or used >= m:
                return None
            best = max(cand)
            if best <= cur and out:
                return None
            out.append(int(best))
            cur = int(best)
            used += 1
    lo, hi = 1, T + 1
    best = feasible(hi) or []
    while lo < hi:
        mid = (lo + hi) // 2
        f = feasible(mid)
        if f is not None:
            best, hi = f, mid
        else:
            lo = mid + 1
    return best


def main(fl, gw, w):
    tag = f"{fl}_{gw}_w{w}"
    z = np.load(os.path.join(HERE, "out", f"sf_{tag}.npz"))
    sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fl}.tle"))
    gws = sf_mc.load_gateways(gw)
    S = len(sats)
    t1, P1, v1 = plan(sats, gws, 7.0 * w - 7.0)
    t2, P2, v2 = plan(sats, gws, 7.0 * w)
    nt = len(t1)
    r1 = [sf_mc.intervals(v1[k]) for k in range(S)]
    r2 = [sf_mc.intervals(v2[k]) for k in range(S)]
    rng = np.random.default_rng(31 + w)
    pick = rng.choice(len(z["buoys"]), size=min(N_DEV, len(z["buoys"])), replace=False)
    T = nt // int(SLOT / DT)
    amax = 200
    rows = {m: {"periodic": [], "smdp": [], "plan": []} for m in BUDGETS}
    cont = []
    for j in pick:
        lat, lon = z["buoys"][j]
        xyz = F.geodetic_to_ecef(np.array([lat]), np.array([lon]))[0]
        U1, H1 = slot_plan(P1, r1, xyz, nt)
        U2, H2 = slot_plan(P2, r2, xyz, nt)
        if U1.sum() < 3 or U2.sum() < 3:
            continue
        pu, pd, d, edges = fit(U1, H1)
        for m in BUDGETS:
            pol = smdp(T, m, pu, pd, d, amax)
            for name, sends in (("periodic", run_policy_periodic(U2, T, m)),
                                ("smdp", run_policy_smdp(U2, H2, T, m, pol, edges, amax)),
                                ("plan", plan_optimal(U2, H2, T, m))):
                pk, av = evaluate(sends, H2, T)
                rows[m][name].append((pk * SLOT / 3600, av * SLOT / 3600, len(sends)))
        sends = np.flatnonzero(U2).tolist()
        pk, av = evaluate(sends, H2, T)
        cont.append((pk * SLOT / 3600, av * SLOT / 3600, len(sends)))
    out = dict(contact=np.mean(cont, axis=0).tolist(), n=len(cont),
               budgets={str(m): {k: np.mean(v, axis=0).tolist() for k, v in rows[m].items()} for m in BUDGETS})
    json.dump(out, open(os.path.join(HERE, "out", "rev4", f"ari_{tag}.json"), "w"), indent=1)
    print(tag, out, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]))
