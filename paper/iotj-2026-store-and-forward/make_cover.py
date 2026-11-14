#!/usr/bin/env python3
"""Writes ../cover_letter.md from macros_p6.tex, so its numbers are those of the PDF."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
M = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(os.path.join(HERE, "..", "macros_p6.tex")).read()))
T = """Muhammad Uzair
Department of Computer Science, COMSATS University Islamabad
Islamabad 45550, Pakistan
muhammaduzairr69@gmail.com | ORCID 0009-0002-4104-2680

{date}

The Editor-in-Chief
IEEE Internet of Things Journal

Dear Editor,

I submit the manuscript "Timer Values and Custody Costs for Release 19 Store-and-Forward IoT Over Sparse LEO Constellations" for consideration as a regular article in the IEEE Internet of Things Journal.

Release 19 lets an IoT satellite hold uplink data until its next gateway contact, and allows the on-board MME to give the device an estimated delivery time and a wait timer, leaving how to compute them to the implementation. On the orbits of three IoT fleets, with an assumed payload, the article sets those values against the signaling limits. The ephemeris can supply the delivery time and, if the timer is set as a return leg through the Monitoring List, which is our reading and not a definition in TS 23.401, the timer. In a reject whose request must reach the ground, the 90-minute example limit of TS 24.301 would be exceeded at {RejLo} to {RejHi} % of contacts with five gateways. After an accept that limit does not apply, and the cost depends on where the context lives, which we score in a slot simulator. If it must return, a device with no timer wastes {RetFailLo} to {RetFailHi} passes per upload for the same age, so a lookup wins. If contexts are pushed to the fleet, no timer is needed for age or admission, and a timer only trades age for fewer upload sessions: at 15-minute updates with five gateways the lookup raises the age by {RiseFifteenLo} to {RiseFifteenHi} %, {RiseSixtyLo} to {RiseSixtyHi} % at hourly ones and at most {RiseFourhHi} % at four-hour ones, timers cut sessions by {GateCutLo} to {GateCutHi} %, and a fixed 30 minutes on Sateliot costs {GateRiseFixDSatFive} % of age for {GateNasFixSatFive} % fewer sessions.

The article also accounts for custody hop by hop at its cost. Keep-newest custody leaves the age of information unchanged when contacts drain the queue, under every timer rule, takes most of its on-board saving from discards on the device, and delivers {DelSfoneNewSatFive} to {DelSfoneNewKinEleven} % of updates. On a per-satellite uplink at an assumed 400 b/s, shared by per-report round robin among {NBuoys} buoys on Sateliot, first-come first-served diverges ({NsRrFcfs} h at four days and {NsUpLongFcfs} h at 14) while last-come first-served holds about {NsRrLcfs} h; the device rule uploads {NsRrUpRatio} times fewer reports at equal age.

For timers an operator should propagate its element sets, and fleet-mean delay cannot judge them. References are propagated, not measured, and span days; the radio access itself is not modeled at packet level beyond a shared rate.

I believe the work suits the journal's readership in massive IoT connectivity and non-terrestrial networks: it gives operators of store-and-forward IoT services the timer values a deployment would meet and what a custody rule really buys.

The manuscript is original, has not been published, and is not under consideration elsewhere. Earlier work on the simulation toolkit used here appeared in the IEEE Open Journal of the Communications Society (DOI 10.1109/OJCOMS.2026.3745010); the present article does not overlap with it in results. The simulation code and scripts will be released upon acceptance. The use of AI is disclosed in the Acknowledgment.

Suggested topics for reviewers: satellite IoT, store-and-forward and delay-tolerant networking, age of information, LEO constellation coverage.

Sincerely,
Muhammad Uzair
"""
open(os.path.join(HERE, "..", "cover_letter.md"), "w").write(T.format(date="November 2026", **M))
