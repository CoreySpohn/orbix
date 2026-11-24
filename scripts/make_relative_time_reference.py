r"""Regenerate relative_time_base_x64.json from the absolute-MJD implementation.

The reference holds float64 positions and keepout decisions computed by orbix
commit f88a329, the last one whose observatory methods took an MJD. Extract that
commit's sources and run this script against them::

    git archive f88a329 src | tar -x -C /tmp/orbix_f88a329
    cp src/orbix/_version.py /tmp/orbix_f88a329/src/orbix/
    PYTHONPATH=/tmp/orbix_f88a329/src JAX_ENABLE_X64=1 python \
        scripts/make_relative_time_reference.py \
        tests/reference/relative_time_base_x64.json
"""

import inspect
import json
import pathlib
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.epoch import Epoch

from orbix.observatory import Observatory, ObservatoryL2Halo
from orbix.observatory.orbit import epoch_mjd
from orbix.observatory.solar_system import earth_position_ecliptic

assert "mjd" in inspect.signature(ObservatoryL2Halo.position_ecliptic).parameters, (
    "run against the f88a329 sources, whose methods take an MJD"
)
assert jax.config.jax_enable_x64, "set JAX_ENABLE_X64=1"
EP = {
    "equinox": (2400000.5, 60575.25),
    "j2000": (2400000.5, 51544.5),
    "late": (2400000.5, 80000.0),
}
T = [0.0, 100.7, 999.9]
out = {"x32_points": [], "epochs": EP, "times": T, "positions": [], "mask": []}
for h in (0.0, 30.0):
    obs = ObservatoryL2Halo.from_default(halo_start_day=h)
    for k, (j1, j2) in EP.items():
        e = Epoch(j1, j2, scale="tt")
        for t in T:
            m = epoch_mjd(e, jnp.asarray(t))
            out["positions"].append(
                {
                    "halo_start_d": h,
                    "epoch": k,
                    "time_d": t,
                    "halo": np.asarray(obs.position_ecliptic(m), dtype=float).tolist(),
                    "earth": np.asarray(
                        earth_position_ecliptic(m), dtype=float
                    ).tolist(),
                }
            )
ob = Observatory(ObservatoryL2Halo.from_default(), keepout_min_deg=45.0)
ras = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
decs = [10.0, -20.0, 0.0, 35.0, -60.0, 5.0, 50.0, -5.0]
for k, (j1, j2) in EP.items():
    e = Epoch(j1, j2, scale="tt")
    for t in T:
        for ra, dec in zip(ras, decs, strict=True):
            ang = float(ob.orbit.solar_elongation_deg(ra, dec, e, t))
            ok = bool(ob.is_observable_at(ra, dec, e, t))
            out["mask"].append(
                {
                    "epoch": k,
                    "time_d": t,
                    "ra": ra,
                    "dec": dec,
                    "sun_angle_deg": ang,
                    "observable": ok,
                }
            )

EP4 = dict(EP, late=(2400000.5, 80000.0), late_odd=(2400000.5, 80000.3137))
T5 = [0.5, 17.3, 100.7, 250.0, 999.9]
period = ObservatoryL2Halo.from_default().period_yr
tt = np.linspace(0.0, period, 1001)
ang = 2.0 * np.pi * tt / period
fast = pathlib.Path(tempfile.mkdtemp()) / "fast_halo.npz"
np.savez(
    fast,
    t_yr=tt,
    r_earth_AU=0.49 * np.stack([np.cos(ang), np.sin(ang), 0.1 * np.cos(ang)], axis=1),
    period_yr=period,
    L2_dist_AU=0.01,
    mu=3e-6,
)
for name, o in (
    ("bundled", ObservatoryL2Halo.from_default()),
    ("fast", ObservatoryL2Halo.from_npz(str(fast))),
):
    for k, (j1, j2) in EP4.items():
        e = Epoch(j1, j2, scale="tt")
        for t in T5:
            m = epoch_mjd(e, jnp.asarray(t))
            out["x32_points"].append(
                {
                    "halo": name,
                    "epoch": k,
                    "time_d": t,
                    "position": np.asarray(
                        o.position_ecliptic(m), dtype=float
                    ).tolist(),
                    "earth": np.asarray(
                        earth_position_ecliptic(m), dtype=float
                    ).tolist(),
                }
            )
json.dump(out, open(sys.argv[1], "w"), indent=0)
