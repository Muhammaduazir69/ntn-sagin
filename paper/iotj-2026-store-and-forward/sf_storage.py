#!/usr/bin/env python3
"""On-board custody occupancy against the mixed-Poisson law (Proposition 2 of the IoT-J study).

Just before a gateway contact, satellite k holds the updates uploaded to it during the preceding gap.
With uploads at the long-run rate L_k and gap G, the count is Poisson(L_k G) given G, so over the gaps
of the orbit it is mixed Poisson: P(N > B) = E_G[ Poisson tail(B; L_k G) ]. We compare the 99th
percentile of the simulated occupancy with the mixed-Poisson quantile (gaps from the ephemeris) and
with the plain Poisson quantile at the mean gap, for FCFS custody; keep-newest custody is reported
next to it.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def quantile_mixed(lams, q=0.99):
    """Smallest B with mean_i P(Poisson(lams_i) > B) <= 1 - q."""
    lams = np.asarray(lams, float)
    B = 0
    while np.mean(stats.poisson.sf(B, lams)) > 1 - q:
        B += 1
    return B


def panjer_sf(lam, f, nmax):
    """Survival P(S > n), n = 0..nmax, of a compound Poisson sum with rate lam and severity pmf f
    (f[0] = 0), by Panjer's recursion."""
    p = np.zeros(nmax + 1)
    p[0] = np.exp(-lam)
    j = np.arange(len(f))
    for n in range(1, nmax + 1):
        m = min(n, len(f) - 1)
        p[n] = lam / n * np.sum(j[1:m + 1] * f[1:m + 1] * p[n - j[1:m + 1]])
    return 1 - np.cumsum(p)


def quantile_compound(nus, f, q=0.99, nmax=4000):
    sf = np.mean([panjer_sf(nu, f, nmax) for nu in nus], axis=0)
    return int(np.argmax(sf <= 1 - q))


def main(tags):
    res = {}
    for tag in tags:
        d = np.load(os.path.join(HERE, "out", f"sf_{tag}.npz"))
        occ, occn = d["occ_fcfs"], d["occ_new"]
        gaps = d["gw_gaps"]
        days = 7.0
        # long-run upload rate into custody per satellite: total held updates / (satellites * horizon)
        S = int(d["n_sats"])
        lam = occ.sum() / (S * (days - 3) * 86400.0)
        mixed = quantile_mixed(lam * gaps)
        plain = int(stats.poisson.ppf(0.99, lam * gaps.mean()))
        # upload batches into custody: updates sharing satellite and upload instant
        held = (d["sf1_up"] < d["sf1_del"]) & (d["sf1_del"] < 10 ** 12)
        keys = d["sf1_sat"][held].astype(np.int64) * 10 ** 9 + d["sf1_up"][held]
        _, sizes = np.unique(keys, return_counts=True)
        f = np.bincount(sizes).astype(float)
        f /= f.sum()
        nu = len(sizes) / (S * (days - 3) * 86400.0)
        comp = quantile_compound(nu * gaps, f)
        res[tag] = dict(lam_per_h=float(lam * 3600), batch_mean=float(sizes.mean()), sim_p99=float(np.percentile(occ, 99)),
                        compound_p99=int(comp),
                        sim_max=int(occ.max()), mixed_p99=int(mixed), plain_p99=plain,
                        newest_p99=float(np.percentile(occn, 99)), newest_max=int(occn.max()),
                        gap_mean_h=float(gaps.mean() / 3600), gap_cv=float(gaps.std() / gaps.mean()))
        print(tag, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[tag].items()})
    json.dump(res, open(os.path.join(HERE, "out", "storage_w0.json" if tags[0].endswith("_w0") else "storage.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or [f"{s}_{g}" for s in ("orbcomm", "kineis", "sateliot") for g in ("k5", "aws11")])
