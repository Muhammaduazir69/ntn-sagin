# Paper 6 patch to the store-and-forward example (Nov 2026)

`ns-3-dev` is shared with running campaigns, so this revision neither edited nor rebuilt it. The paper's ns-3 data come from standalone binaries of `sim/bin/p6-sf-iot`, built by `build.sh` against the existing optimized libraries in `ns-3-dev/build/lib`:

- First binary, SHA-256 824142ca91deb280a18eb994032e83f6c784db4be12de6159ea016d8ad932d63, every row of `../out/ns3_sf4.jsonl`. Its shared uplink served each device's whole backlog in a slowly rotating order. `make_macros.py` still reads its 730 rows without a shared uplink; its 95 rows with one are kept only as the record of that rule.
- Current binary, SHA-256 0d876d7e5460ba46e30a9acc56ed04bd6cb18708804f17006f3fdd0a75b8d7a8, every row of `../out/ns3_sf5.jsonl` (the 95 rows, `../run_ns3_sf5.py`) and of the further rate sweep `../out/ns3_sf5_sweep.jsonl` (15 rows). It differs only in the branch taken when `--upBps` is above zero: per-report round robin (one report per device per pass, increasing device index, pointer kept per satellite across ticks, same credit). `--upRoundRobin=0` restores the old rule. `../out/ns3_sf5_check.jsonl` holds rows of this binary, stratified over fleets, policies, disciplines and options, that `make_macros.py` asserts equal to the first binary's.
- Rebuild proof: `build.sh` overwrote the first binary, so its source is kept as `sagin-store-forward-iot.first_binary.cc`, which is the current source with only the round-robin hunks removed (`strip_to_first_binary.py`). Compiled with `build.sh`'s flags plus `-ffile-prefix-map=<scratch>=<this folder>` (the source path enters through `__FILE__`), it gives SHA-256 824142ca91deb280a18eb994032e83f6c784db4be12de6159ea016d8ad932d63, the first binary's hash. All removed hunks sit behind `upBps > 0`. The libraries in `ns-3-dev/build/lib` are dated 2 Nov, before both campaigns.

- `sagin-store-forward-iot.orig.cc`: the example as in the tree on 4 Nov 2026.
- `sagin-store-forward-iot.cc`: the patched copy. `sagin-store-forward-iot_rev4.patch` is the diff (apply with `patch -p1` from `ns-3-dev`).

Only the example changes, so no library symbol is overridden. What the patch adds: occupancy log, device-only and satellite-only keep-newest, shared uplink rate, per-report round robin on that uplink, Monitoring List, per-device lifetime, per-run accounting in the summary line, a larger link transmit queue, and no attribute writes on links that stay out of view (speed only).

Defects found in the first, never-compiled draft of the patch and fixed before any data were taken:
- Per-device lifetime offset had the wrong sign: it gave a lifetime of 2L - L_d, not L_d.
- Uplink credit was reset each tick and lost the fraction of a packet: 400 b/s ran at 374 b/s.
- Runner left DAYS = 14 set for the uplink stage, had no resume, and used nice 5.

Defect found by review in the first binary (fixed in the current one): whole-backlog service with a global order moving one place per 10 s tick reaches at most about 48 of 919 positions in a pass, so the uplink was not shared fairly and some buoys were starved.

Defect found in the existing example: with the default 100-packet device queue, a custody drain overflows into fq_codel, which drops updates already counted as forwarded (up to 16 of 17280 per run in `ns3_sf2`). `--macQueue=100` restores the old behavior and reproduced an `ns3_sf2` run exactly (Sateliot, five gateways, replication 0: 17268 delivered, mean delay 5.094196372640144 h).

**To merge at the shared build (author's step, once no campaign is using the tree):** from `ns-3-dev`, `patch -p1 < ../papers/paper6_iotj_sf_iot/sim/ns3_patch/sagin-store-forward-iot_rev4.patch`, then rebuild the example. Until then the shared build's example does not have these options. The patch and the example's data files are not committed.
