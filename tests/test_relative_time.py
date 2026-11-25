"""Observatory geometry takes ``(epoch, time_d)``, never an absolute date under a trace.

The Earth ephemeris and the halo phase are reduced about the static epoch on the
host in float64, so float32 keeps the resolution of a time of the order of the
observation span instead of that of a date (a quarter of a day for an MJD).
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.epoch import Epoch
from hwoutils.leaves import check_tree
from hwoutils.numerics import roundoff
from hwoutils.testing import precision_pair

from orbix.observatory import Observatory, ObservatoryL2Halo, sun_ecliptic_longitude_deg
from orbix.observatory.orbit import epoch_mjd
from orbix.observatory.solar_system import earth_position_ecliptic

# Regenerate the reference with scripts/make_relative_time_reference.py (see its
# docstring); it runs the absolute-MJD sources of orbix commit f88a329.
_REFERENCE = json.loads(
    (Path(__file__).parent / "reference" / "relative_time_base_x64.json").read_text()
)
# Epochs: the bundled equinox, J2000, and a late epoch 19425 d (53 yr) past the equinox
# whose offset is not a float32 number (spacing there is 1.95e-3 d).
_EPOCHS = {
    "equinox": Epoch(2400000.5, 60575.25, scale="tt"),
    "j2000": Epoch(2400000.5, 51544.5, scale="tt"),
    "late": Epoch(2400000.5, 80000.0, scale="tt"),
}
# MJD 80000.3137: the offset from the equinox, 19425.0637 d, rounds by up to 1e-3 d
# in float32.
_LATE_ODD = Epoch(2400000.5, 80000.3137, scale="tt")
# The stored float32 points keep half a day or more from the halo wrap (multiples of
# the 182.6 d period): the bundled table is not exactly periodic, its two ends differ
# by about 6e-6 AU, which float32 phase rounding would flip across.


def test_positions_match_base_code():
    """Positions agree with the float64 absolute-date implementation in both lanes.

    Truth: positions stored from the absolute-MJD implementation this change
    replaced, in float64, for three epochs (one 53 yr from the halo equinox),
    times to 1000 d and two halo start offsets. Tolerance basis: floating-point.
    Two roundings of the reference MJD at 0.018 AU/day bound its time error.
    The working path rounds its phase on conversion to centuries, scaling
    by the mean motion, adding the intercept, and converting to radians.
    Twelve unit-radius operations cover the trig, rotation and halo sum.
    """
    for rec in _REFERENCE["positions"]:
        epoch = _EPOCHS[rec["epoch"]]
        obs = ObservatoryL2Halo.from_default(halo_start_day=rec["halo_start_d"])
        for got, want in (
            (obs.position_ecliptic(epoch, rec["time_d"]), rec["halo"]),
            (earth_position_ecliptic(epoch, rec["time_d"]), rec["earth"]),
        ):
            want = np.asarray(want)
            radius = np.linalg.norm(want)
            t = rec["time_d"]
            reference_error = 2 * np.spacing(float(epoch_mjd(epoch, t))) * 0.018
            phase = 2 * np.pi + 0.018 * abs(t)
            atol = reference_error + radius * float(
                roundoff(phase, n_ops=4, dtype=got.dtype)
                + roundoff(1.0, n_ops=12, dtype=got.dtype)
            )
            np.testing.assert_allclose(got, want, atol=atol, rtol=0)
    assert len(_REFERENCE["positions"]) == 18


def test_observability_mask_matches_base_code():
    """Fails if the keepout decision changes away from the keepout edge.

    Truth: the mask and solar elongation stored from the absolute-MJD
    implementation, 8 targets by 3 epochs by 3 times, 45 deg solar keepout.
    Exact equality for every point farther than 1e-6 deg (the float64 position
    difference of 1e-12 AU is about 1e-10 deg, four decades inside the margin)
    from the 45 deg edge; the stored set has no point nearer than 0.66 deg, so the
    margin excludes nothing, and both outcomes occur.
    """
    obs = Observatory(ObservatoryL2Halo.from_default(), keepout_min_deg=45.0)
    kept = 0
    seen = set()
    for name, epoch in _EPOCHS.items():
        recs = [r for r in _REFERENCE["mask"] if r["epoch"] == name]
        got = obs.is_observable_at(
            jnp.array([r["ra"] for r in recs]),
            jnp.array([r["dec"] for r in recs]),
            epoch,
            jnp.array([r["time_d"] for r in recs]),
        )
        for rec, value in zip(recs, np.asarray(got), strict=True):
            if abs(rec["sun_angle_deg"] - 45.0) < 1e-6:
                continue
            assert bool(value) == rec["observable"]
            kept += 1
            seen.add(rec["observable"])
    assert kept == len(_REFERENCE["mask"]) == 72
    assert seen == {True, False}


def _fast_halo(tmp_path):
    """Halo whose telescope moves at 0.0169 AU/d, close to the Earth's orbital speed.

    A circle of radius 0.49 AU over the default period has speed 2 pi A / P =
    0.0169 AU/d, so a phase error of dt days is a position error of 0.0169 dt AU,
    visible against the 5e-6 AU bound where the bundled halo (at most 2.2e-4 AU/d)
    would hide a phase error of a day. The stored reference was made from this
    same table.
    """
    period_yr = ObservatoryL2Halo.from_default().period_yr
    t = np.linspace(0.0, period_yr, 1001)
    ang = 2.0 * np.pi * t / period_yr
    r = 0.49 * np.stack([np.cos(ang), np.sin(ang), 0.1 * np.cos(ang)], axis=1)
    path = tmp_path / "fast_halo.npz"
    np.savez(
        path,
        t_yr=t,
        r_earth_AU=r,
        period_yr=period_yr,
        L2_dist_AU=0.01,
        mu=3e-6,
    )
    return str(path)


@pytest.mark.float32_only(reason="float32 epoch-conditioning floor")
def test_x32_positions_agree_with_x64_truth(tmp_path):
    """Fails if an absolute date, or the halo phase, is formed in float32.

    Truth: float64 positions stored from the absolute-MJD implementation this
    change replaced, so the truth does not share the code under test. Tolerance
    basis: spike plus spacing. The float32 step of a relative time of 1000 d is
    6.1e-5 d; at the Earth's orbital speed of 0.0172 AU/d that is 1.0e-6 AU, and
    the bound of 5e-6 AU allows five steps. Points: four epochs (the last 53 yr
    past the halo equinox, offset not a float32 number), times 0.5 to 999.9 d,
    the bundled halo and a fast synthetic halo (0.0169 AU/d) that makes a
    halo-phase error of 3e-4 d visible. The base code measured 1.1e-4 AU (bundled)
    and 1.3e-4 AU (fast) at these points.
    """
    epochs = {**_EPOCHS, "late_odd": _LATE_ODD}
    fast_path = _fast_halo(tmp_path)
    points = _REFERENCE["x32_points"]
    assert len(points) == 40
    worst = 0.0
    observatories = {
        "bundled": ObservatoryL2Halo.from_default(),
        "fast": ObservatoryL2Halo.from_npz(fast_path),
    }
    assert observatories["fast"].position_ecliptic(_LATE_ODD, 1.0).dtype == (
        jnp.float32
    )
    for rec in points:
        epoch, t = epochs[rec["epoch"]], rec["time_d"]
        for got, want in (
            (observatories[rec["halo"]].position_ecliptic(epoch, t), "position"),
            (earth_position_ecliptic(epoch, t), "earth"),
        ):
            err = np.abs(np.asarray(got, dtype=np.float64) - rec[want]).max()
            worst = max(worst, float(err))
    assert worst < 5e-6, worst


@pytest.mark.float64_only(
    reason="float64 JAX reference for the float32 precision comparison"
)
def test_x32_sun_longitude_agrees_with_x64():
    """Fails if the Sun's longitude forms an absolute date in float32.

    Truth: the same function in float64, whose own error is below 1e-6 deg.
    Tolerance basis: spike plus spacing. The Sun moves 0.986 deg/d and the
    float32 step of a relative time of 1000 d is 6.1e-5 d (6e-5 deg); the
    measured worst case over 401 times to 1000 d and three epochs is 2.0e-4 deg
    (the angle and Kepler arithmetic add a few steps), and the bound of 3e-4 deg
    is five steps. The absolute-date implementation errs by about 1e-3 deg.
    """
    times = jnp.linspace(0.0, 1000.0, 401)

    def direction(epoch, t):
        angle = sun_ecliptic_longitude_deg(epoch, t)
        assert angle.dtype == t.dtype
        # Unit phasors compare across the 360-degree seam; the chord is at
        # most the angular separation in radians.
        return jnp.exp(1j * jnp.radians(angle))

    for epoch in (*_EPOCHS.values(), _LATE_ODD):
        precision_pair(
            lambda t, epoch=epoch: direction(epoch, t),
            (times,),
            bound=2 * np.sin(np.radians(3e-4) / 2),
            scale=1.0,
        )


def test_epoch_mjd_is_host_only():
    """Fails if the host helper accepts a traced time or leaves float64 NumPy.

    Truth: the closed form MJD = epoch MJD + time_d, exact in float64 for these
    binary fractions (tolerance exact-algebra).
    """
    epoch = Epoch(2400000.5, 60000.0, scale="tt")
    got = epoch_mjd(epoch, [0.0, 0.5, 365.25])
    assert isinstance(got, np.ndarray) and got.dtype == np.float64
    np.testing.assert_array_equal(got, [60000.0, 60000.5, 60365.25])
    with pytest.raises(TypeError, match="host-only"):
        jax.jit(lambda t: epoch_mjd(epoch, t))(1.0)
    with pytest.raises(TypeError, match="host-only"):
        jax.grad(lambda t: epoch_mjd(epoch, t))(1.0)


def test_halo_has_no_python_number_or_wrong_dtype_leaf(precision):
    """Fails if a time or length scalar of the halo is stored as a dynamic leaf.

    Truth: the leaf rule of the hwoutils leaf check (a Python number, NumPy
    array, weak array or non-active float is a violation), run in float64 and in
    float32. The Python numbers are static fields; the interpolators are the
    only leaves.
    """
    obs = ObservatoryL2Halo.from_default()
    assert check_tree(obs) == ()
    leaves = jax.tree_util.tree_leaves(eqx.filter(obs, eqx.is_array))
    assert leaves and all(leaf.dtype == precision for leaf in leaves)
