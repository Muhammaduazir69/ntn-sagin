# Code and data: timers and custody costs for store-and-forward IoT over sparse LEO constellations

Companion material for M. Uzair, "Timer Values and Custody Costs for Release 19 Store-and-Forward IoT Over Sparse LEO Constellations," 2026.

The study sizes the Release 19 store-and-forward timers and measures the custody, age and upload-session costs on sparse constellations (Orbcomm, Kineis and others) from contact plans and packet-level ns-3 runs of the ntn-sagin module.

| Path | Contents |
|---|---|
| `ns3_patch/` | Campaign example source of record and build script |
| `run_ns3_sf*.py` | Packet-level ns-3 campaign drivers |
| `sf_core.py`, `sf_*.py` | Contact plans, timer rules, custody and age models, what-if studies, held-out checks |
| `data/`, `data_raw/` | Element sets and station tables (CelesTrak, NOAA NDBC) |
| `out/` | Every result file |
| `make_macros.py`, `make_figs.py`, `check_ship.py` | Numbers, figures and consistency checks |

Contact: Muhammad Uzair, muhammaduzairr69@gmail.com, ORCID 0009-0002-4104-2680. License: as the repository (GPL-2.0-only for code).
