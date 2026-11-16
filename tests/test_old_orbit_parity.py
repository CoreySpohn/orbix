"""Compatibility regressions to delete with the angle-field orbit and solver."""

import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg

from orbix.kepler.markley import solve_trig
from orbix.orbit import AbstractOrbit, KeplerianOrbit
from orbix.system import Body, FrameOrbit, SkyFrame, System


@pytest.mark.parametrize("e,inc", [(1e-6, 0.7), (0.6, 1.1), (0.2, 2.6)])
def test_existing_orbit_agreement(e, inc):
    """Fails if sky convention, epoch, mass, reflex or velocity units change.

    Truth: the existing KeplerianOrbit with explicit Markley (not its grid
    default), equal star/planet masses and a shared day origin. Its t0_d
    and new epoch_d both equal 23 days, with M0=.2 at that instant.
    Allow 3e-13 arcsec, 3e-14 AU/day and 3e-10 m/s: several hundred x64
    ULPs for independently arranged phase, basis and mass arithmetic.
    """
    elements = dict(a_AU=1.7, e=e, i_rad=inc, W_rad=0.4, w_rad=0.7, M0_rad=0.2)
    sys = System.from_classical(
        Body(1.2), Body([0.01]), SkyFrame(12.0), **elements, epoch_d=23.0
    )
    old = KeplerianOrbit(**elements, t0_d=23.0)
    times = jnp.array([-100.0, 23.0, 200.0, 700.0])
    context = dict(Ms_kg=1.2 * Msun2kg, Mp_kg=0.01 * Msun2kg)
    eph = sys.solve(times)
    offsets = np.asarray(eph.offset_arcsec(1))
    old_offsets = np.asarray(
        old.position_arcsec(solve_trig, times, **context, dist_pc=12.0)
    )[:, 0]
    reflex = np.asarray(eph.offset_arcsec(0, "barycenter"))
    old_reflex = np.asarray(
        old.reflex_position_arcsec(solve_trig, times, **context, dist_pc=12.0)
    )[:, 0]
    rv = eph.radial_velocity_m_per_s(0)
    old_rv = old.radial_velocity_m_s(solve_trig, times, **context)[0]
    velocity = eph.velocity_AU_per_d[1] - eph.velocity_AU_per_d[0]
    old_velocity = old.velocity_AU_d(solve_trig, times, **context)[0].T
    np.testing.assert_allclose(offsets, old_offsets, atol=3e-13, rtol=0)
    np.testing.assert_allclose(reflex, old_reflex, atol=3e-13, rtol=0)
    np.testing.assert_allclose(rv, old_rv, atol=3e-10, rtol=0)
    np.testing.assert_allclose(velocity, old_velocity, atol=3e-14, rtol=0)
    print(
        "parity max (offset, reflex, RV, velocity):",
        *(
            float(np.max(np.abs(a - b)))
            for a, b in [
                (offsets, old_offsets),
                (reflex, old_reflex),
                (rv, old_rv),
                (velocity, old_velocity),
            ]
        ),
    )


def test_lazy_root_and_engine_solver_exports():
    """Fails if importing the engine loads the legacy orbit or aliases diverge."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import orbix.system as system
import orbix
from importlib import import_module
solvers = import_module('orbix.kepler')
assert 'orbix.orbit' not in sys.modules
assert orbix.__all__ == ['AbstractOrbit', 'KeplerianOrbit', '__version__']
assert set(orbix.__all__) <= set(dir(orbix))
for name in ('kepler', 'AbstractKeplerSolver', 'Markley', 'Halley', 'Grid'):
    assert getattr(system, name) is getattr(solvers, name)
# Loading the preparation and rendering entry points must stay independent too.
from orbix.viz import prepare_sky_track, plot_orbit, animate_orbit
import numpy as np
frame = system.FrameOrbit.from_classical(
    1.0, 0.2, 0.4, 0.5, 0.6, 0.7, mu_AU3_per_d2=0.0003
)
prepare_sky_track(frame, time_d=np.array([0., 1.]), dist_pc=10.)
assert 'orbix.orbit' not in sys.modules
from orbix import AbstractOrbit, KeplerianOrbit
assert orbix.KeplerianOrbit is KeplerianOrbit
assert orbix.AbstractOrbit is AbstractOrbit
assert issubclass(KeplerianOrbit, AbstractOrbit)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_viz_engine_and_legacy_prepared_arrays():
    """Fails if sky or 3D preparation changes frame, phase, distance or batching.

    Delete with the legacy adapter. Two paths of at most 2 AU span up to
    seven periods. 128 float64 eps times their size bounds independent
    phase/basis arithmetic with a shared table-free Markley strategy:
    6e-14 AU and 6e-15 arcsec at 10 pc, including coordinate zeros.
    """
    from eyepiece.prepared import find_element

    from orbix.viz import prepare_sky_track, prepare_sky_view
    from orbix.viz.orbit import _positions, plot_orbit

    class PositionOnly(AbstractOrbit):
        wrapped: KeplerianOrbit

        def propagate(self, *args, **kwargs):
            return self.wrapped.propagate(*args, **kwargs)

    times = np.linspace(-300.0, 3300.0, 31)
    for e, inc in ((0.43, 1.1), (0.0, 0.0)):
        elements = dict(
            a_AU=np.array([2.0, 1.3]),
            e=np.full(2, e),
            i_rad=np.full(2, inc),
            W_rad=np.full(2, 0.4),
            w_rad=np.full(2, 0.7),
            M0_rad=np.full(2, 0.3),
        )
        old = KeplerianOrbit(**elements, t0_d=np.full(2, 23.0))
        frame = FrameOrbit.from_classical(
            **elements,
            mu_AU3_per_d2=G * Msun2kg,
            epoch_d=23.0,
        )
        # Zero planet masses make the two independent tracks a System too.
        system = System(Body(1.0), Body([0.0, 0.0]), frame, SkyFrame(10.0))
        reference = prepare_sky_track(
            old,
            time_jd=times,
            Ms_kg=Msun2kg,
            dist_pc=10.0,
            trig_solver=solve_trig,
        )
        xyz, _ = _positions(old, times, Msun2kg, solve_trig)
        for value in (frame, system):
            prepared = prepare_sky_track(value, time_d=times, dist_pc=10.0)
            static = prepare_sky_view(value, time_d=times, dist_pc=10.0)
            actual, _ = _positions(value, times, None, None)
            np.testing.assert_allclose(actual, xyz, atol=6e-14, rtol=0)
            for k in range(2):
                expected = find_element(reference.template, f"path/{k}").xy
                np.testing.assert_allclose(
                    find_element(prepared.template, f"path/{k}").xy,
                    expected,
                    atol=6e-15,
                    rtol=0,
                )
                np.testing.assert_array_equal(
                    find_element(static, f"path/{k}").xy,
                    find_element(prepared.template, f"path/{k}").xy,
                )

    # A custom legacy subclass only promises propagation, not ellipse elements.
    import matplotlib.pyplot as plt

    result = plot_orbit(PositionOnly(old), times, Ms_kg=Msun2kg, trig_solver=solve_trig)
    assert result.ax.get_xlabel() == "East [AU]"
    plt.close(result.fig)
