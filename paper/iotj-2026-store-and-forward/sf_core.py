#!/usr/bin/env python3
"""Store-and-forward IoT over sparse LEO: geometry, contact plans and closed forms (IoT-J study).

Closed forms (Section IV of the manuscript):
  sub-satellite latitude density   f(phi) = cos(phi) / (pi sqrt(sin^2 i - sin^2 phi))
  contact probability of a site     p = integral over the visibility cap of f(phi)/(2 pi R^2 cos phi) dA
  mean contact length               E[C] = (pi/2) psi / omega_g   (mean chord of the cap)
  mean gap                          E[G] = E[C] (1 - p) / p
  custody delay (residual gap)      P(H > x) = (1/(E[C]+E[G])) int_x^inf P(G > u) du
Truth: SGP4 propagation of real element sets or Walker shells, rotated into an Earth-fixed frame.
"""
import os

import numpy as np
from scipy import integrate
from sgp4.api import Satrec, WGS72, jday

MU = 3.986004418e14
RE = 6378137.0
OMEGA_E = 7.2921150e-5
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- geometry
def cap_angle(h, elev_min):
    """Central angle of the visibility cap of a site for altitude h and elevation mask."""
    r = RE + h
    return np.arccos(RE / r * np.cos(elev_min)) - elev_min


def lat_density(phi, inc):
    """Density of the sub-satellite latitude of a circular orbit of inclination inc (rad)."""
    s = np.sin(inc) ** 2 - np.sin(phi) ** 2
    return np.where(s > 0, np.cos(phi) / (np.pi * np.sqrt(np.maximum(s, 1e-300))), 0.0)


def contact_prob(site_lat, h, inc, elev_min, n=400):
    """Probability that one satellite is inside the cap of a site, exact integral over the cap of the
    sub-point density f(phi) / (2 pi cos phi) per unit solid angle (longitude uniform). For a
    retrograde orbit use pi - inc."""
    inc = min(inc, np.pi - inc)
    psi = cap_angle(h, elev_min)
    # integrate over the cap in latitude strips: width of the cap at latitude phi in longitude
    lo, hi = site_lat - psi, site_lat + psi
    ph = np.linspace(lo, hi, n)
    # half-width in longitude of the cap at latitude ph (spherical cap around (site_lat, 0))
    c = (np.cos(psi) - np.sin(site_lat) * np.sin(ph)) / np.maximum(np.cos(site_lat) * np.cos(ph), 1e-12)
    dlon = 2 * np.arccos(np.clip(c, -1, 1))
    # probability mass: f(phi) dphi * (dlon / 2 pi)
    g = lat_density(ph, inc) * dlon / (2 * np.pi)
    # integrable singularity at |phi| = inc: use a substitution-free trapezoid on a fine grid
    return float(np.trapezoid(g, ph))


def ground_rate(h, inc, lat=0.0):
    """Angular rate of the sub-satellite point relative to the rotating Earth at latitude lat (rad/s):
    sqrt(w^2 - 2 w w_E cos i + w_E^2 cos^2 lat)."""
    r = RE + h
    w = np.sqrt(MU / r ** 3)
    return np.sqrt(w ** 2 - 2 * w * OMEGA_E * np.cos(inc) + OMEGA_E ** 2 * np.cos(lat) ** 2)


def contact_stats(site_lat, h, inc, elev_min):
    p = contact_prob(site_lat, h, inc, elev_min)
    psi = cap_angle(h, elev_min)
    ec = np.pi / 2 * psi / ground_rate(h, inc, site_lat)
    eg = ec * (1 - p) / p if p > 0 else np.inf
    return dict(p=p, EC=ec, EG=eg)


def residual_sf(x, gaps, contacts):
    """P(H > x) for the equilibrium residual of an alternating contact/gap process (empirical gaps)."""
    gaps = np.asarray(gaps, float)
    tot = gaps.sum() + np.sum(contacts)
    x = np.atleast_1d(x)
    return np.array([np.sum(np.maximum(gaps - xx, 0)) / tot for xx in x])


# --------------------------------------------------------------------------- truth
def gmst(jd, fr=0.0):
    # jd and its fraction kept apart (a single double near 2.46e6 days resolves only ~40 us)
    t = ((jd - 2451545.0) + fr) / 36525.0
    g = 67310.54841 + (876600 * 3600 + 8640184.812866) * t + 0.093104 * t * t - 6.2e-6 * t ** 3
    return np.radians((g % 86400) / 240.0)


def load_tles(path):
    L = [x.rstrip() for x in open(path) if x.strip()]
    out = []
    for i in range(0, len(L) - 2, 3):
        if L[i + 1].startswith("1 ") and L[i + 2].startswith("2 "):
            out.append(Satrec.twoline2rv(L[i + 1], L[i + 2]))
    return out


def walker(n_total, n_planes, f_phase, h, inc_deg, epoch=(2026, 10, 30, 0, 0, 0)):
    """Walker-delta shell as SGP4 element sets (circular, no drag)."""
    jd, fr = jday(*epoch)
    sats = []
    s_per = n_total // n_planes
    n = np.sqrt(MU / (RE + h) ** 3) * 60.0  # rad/min
    for p in range(n_planes):
        for s in range(s_per):
            raan = 2 * np.pi * p / n_planes
            ma = 2 * np.pi * s / s_per + 2 * np.pi * f_phase * p / n_total
            sat = Satrec()
            sat.sgp4init(WGS72, "i", p * 100 + s, jd + fr - 2433281.5, 0.0, 0.0, 0.0, 1e-7,
                         0.0, np.radians(inc_deg), ma % (2 * np.pi), n, raan % (2 * np.pi))
            sats.append(sat)
    return sats


def ecef_positions(sats, jd0, fr0, t):
    """Satellite ECEF positions (m), shape (n_sats, n_t, 3), at times t (s) after (jd0, fr0)."""
    t = np.asarray(t, float)
    fr = fr0 + t / 86400.0
    jd = np.full_like(fr, jd0)
    th = gmst(jd, fr)
    c, s = np.cos(th), np.sin(th)
    out = np.empty((len(sats), len(t), 3))
    for k, sat in enumerate(sats):
        e, r, v = sat.sgp4_array(jd, fr)
        r = np.asarray(r) * 1e3
        out[k, :, 0] = c * r[:, 0] + s * r[:, 1]
        out[k, :, 1] = -s * r[:, 0] + c * r[:, 1]
        out[k, :, 2] = r[:, 2]
    return out


def geodetic_to_ecef(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    N = RE / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.stack([N * np.cos(lat) * np.cos(lon), N * np.cos(lat) * np.sin(lon), N * (1 - e2) * np.sin(lat)], -1)


def visibility(P, sites, elev_min_deg):
    """Boolean (n_sats, n_t, n_sites): satellite above the mask at each site."""
    up = sites / np.linalg.norm(sites, axis=1, keepdims=True)
    sin_min = np.sin(np.radians(elev_min_deg))
    out = np.empty((P.shape[0], P.shape[1], len(sites)), bool)
    for j, (g, u) in enumerate(zip(sites, up)):
        los = P - g
        out[:, :, j] = (los @ u) >= sin_min * np.linalg.norm(los, axis=2)
    return out


def runs(mask):
    """Contact and gap lengths (in samples) of a boolean series, excluding the censored ends."""
    m = np.asarray(mask, int)
    d = np.diff(np.r_[0, m, 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    contacts = ends - starts
    gaps = starts[1:] - ends[:-1]
    if len(contacts) and starts[0] == 0:
        contacts = contacts[1:]
    if len(contacts) and ends[-1] == len(m):
        contacts = contacts[:-1]
    return contacts, gaps
