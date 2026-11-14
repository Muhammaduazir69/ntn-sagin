#!/usr/bin/env python3
"""Monte Carlo replications: three disjoint weeks (from 30 Oct, 6 Nov, 13 Nov 2026), each with its own
buoy sample; for each, the analysis predicts from the week before. Resumable (skips finished outputs)."""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FLEETS, GWS, REPS = ("orbcomm", "kineis", "sateliot"), ("k5", "aws11"), (0, 1, 2)


def job(a):
    fl, gw, r = a
    tag = f"{fl}_{gw}_w{r}"
    log = open(os.path.join(HERE, "out", f"log_{tag}.txt"), "a")
    if not os.path.exists(os.path.join(HERE, "out", f"sf_{tag}.npz")):
        subprocess.run([sys.executable, "sf_mc.py", "--scenario", fl, "--gwset", gw, "--days", "7", "--devices", "150",
                        "--seed", str(r + 1), "--start-day", str(7 * r), "--tag", tag], cwd=HERE, stdout=log, stderr=log)
    if not os.path.exists(os.path.join(HERE, "out", f"an_{tag}.json")):
        subprocess.run([sys.executable, "sf_analysis.py", fl, gw, tag, str(7 * r)], cwd=HERE, stdout=log, stderr=log)
    return tag


if __name__ == "__main__":
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for t in ex.map(job, [(f, g, r) for r in REPS for f in FLEETS for g in GWS]):
            print("done", t, flush=True)
