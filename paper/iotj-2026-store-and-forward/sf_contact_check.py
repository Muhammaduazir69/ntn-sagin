#!/usr/bin/env python3
"""Lemma 1 against SGP4: contact probability and mean contact length per fleet and site latitude.

For each fleet, sites at latitudes 0..75 deg (longitude 0) with the device mask; 14 days of SGP4 at
20 s. The closed form is evaluated per satellite with its own altitude and inclination and averaged
over the fleet (the fleet-average contact probability). Writes out/contact_check.json.
"""
import json
import os
import sys

import numpy as np
from sgp4.api import jday

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf_core as F  # noqa: E402

LATS = np.arange(0, 76, 5)
MASK = 20.0


def main():
    jd0, fr0 = jday(2026, 10, 30, 0, 0, 0)
    dt = 20.0
    t = np.arange(0, 14 * 86400.0, dt)
    res = {}
    for fleet in ("orbcomm", "kineis", "sateliot"):
        sats = F.load_tles(os.path.join(HERE, "data_raw", f"{fleet}.tle"))
        P = F.ecef_positions(sats, jd0, fr0, t)
        rows = []
        for lat in LATS:
            site = F.geodetic_to_ecef(np.array([float(lat)]), np.array([0.0]))
            vis = F.visibility(P, site, MASK)[:, :, 0]
            p_sim, c_sim, p_an, c_an = [], [], [], []
            for k, s in enumerate(sats):
                h = (F.MU / (s.no_kozai / 60) ** 2) ** (1 / 3) - F.RE
                st = F.contact_stats(np.radians(lat), h, s.inclo, np.radians(MASK))
                c, _ = F.runs(vis[k])
                p_sim.append(vis[k].mean())
                p_an.append(st["p"])
                if len(c):
                    c_sim.extend(list(c * dt))
                c_an.append(st["EC"])
            rows.append(dict(lat=int(lat), p_sim=float(np.mean(p_sim)), p_an=float(np.mean(p_an)),
                             ec_sim=float(np.mean(c_sim)) if c_sim else float("nan"),
                             ec_an=float(np.mean(c_an))))
        res[fleet] = dict(rows=rows, inc=[float(np.degrees(s.inclo)) for s in sats])
        print(fleet, [(r["lat"], round(r["p_sim"], 4), round(r["p_an"], 4)) for r in rows], flush=True)
    json.dump(res, open(os.path.join(HERE, "out", "contact_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
