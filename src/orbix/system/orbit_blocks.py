"""Blocks that free rows of a stacked orbit through one of the orbit charts.

The model holds a :class:`System` at the path ``system`` (for example ``"dynamics"`` on
a skyscapes ``Scene``) whose ``orbits`` are stacked over the K Jacobi bodies, inner to
outer. A block frees a static tuple of those rows and reads the held rows from the
model. Coordinates are real numbers, a dict keyed by the block's ``coordinate_names``,
each of shape ``(len(rows),)`` per draw.

A block composes three peers: a chart (the coordinate map and its volume), an orbit law
(the density, independent of the chart) and the placement (``rows`` and ``system``).
``log_prob`` is the normalized density of the coordinates against Lebesgue ``dx``,
``law.log_prob`` of the orbits plus the chart volume.

Coordinates have no boundary, as the block contract requires: a block maps every
element its law bounds to a finite interval (see :meth:`AbstractOrbitLaw.support`) and
for which its chart has a coordinate onto the whole real line, through a logit on that
interval, and the density gains the log derivative of that map. A mapped size
coordinate takes the prefix ``logit_`` in ``coordinate_names`` (``logit_ln_a_AU``); the
classical eccentricity coordinate is a logit already and keeps its name, and the disk
charts keep ``ux`` and ``uy``, the plane coordinates of the disk scaled to the law's
upper bound. Where no coordinate carries a bound, it stays in the law and ``log_prob``
returns ``-inf`` there, with no error. Two cases exist. A disk eccentricity chart with a
lower eccentricity bound above zero maps the upper bound and leaves a zero-density disk
around the origin (an annulus has no smooth map from the plane; the classical chart is
the choice that maps both bounds). A Thiele-Innes chart with a bounded ``a_AU`` law
leaves the walls of that law, since its size coordinate is ``ln n``. ``sample`` and
``from_unit`` never land in a zero-density region.

``forward`` builds a freed row at the model's stored epoch and, for the mass-reading
charts, at the Jacobi mu of the model's current star and planet masses. A held row keeps
its geometry and takes its mean motion from the masses the block read, so the written
node never mixes two masses.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import priorities
from hwoutils.blocks import AbstractBlock
from hwoutils.constants import two_pi
from hwoutils.model_paths import join, resolve
from jaxtyping import Array

from orbix.equations.orbit import mean_motion
from orbix.system.charts import AbstractChart, ThieleInnesChart
from orbix.system.core import GM_SUN_AU3_PER_D2, enclosed_mu
from orbix.system.frame_orbit import FrameOrbit
from orbix.system.laws import AbstractOrbitLaw, size_term


def _take(orbit: FrameOrbit, rows) -> FrameOrbit:
    """The listed rows of a stacked orbit, on the last batch axis."""
    idx = jnp.asarray(rows)
    return FrameOrbit(
        *(
            getattr(orbit, name)[..., idx]
            for name in ("a_AU", "n_rad_per_d", "ex", "ey")
        ),
        orbit.quat[..., idx, :],
        orbit.epoch_d[..., idx],
    )


def _splice(held: FrameOrbit, rows, new: FrameOrbit) -> FrameOrbit:
    """Replace the listed rows of a stacked orbit, broadcasting held rows to a batch."""
    idx = jnp.asarray(rows)
    batch = new.a_AU.shape[:-1]

    def put(name):
        old = getattr(held, name)
        old = jnp.broadcast_to(old, (*batch, *old.shape))
        return old.at[..., idx].set(getattr(new, name))

    quat = jnp.broadcast_to(held.quat, (*batch, *held.quat.shape))
    return FrameOrbit(
        *(put(name) for name in ("a_AU", "n_rad_per_d", "ex", "ey")),
        quat.at[..., idx, :].set(new.quat),
        put("epoch_d"),
    )


def _rated(orbit: FrameOrbit, mu) -> FrameOrbit:
    """The orbit with its mean motion set from ``mu``, geometry kept."""
    return eqx.tree_at(
        lambda o: o.n_rad_per_d, orbit, mean_motion(orbit.a_AU, jnp.asarray(mu))
    )


class OrbitBlock(AbstractBlock):
    """Free some rows of a stacked orbit through a chart that reads the masses.

    Works for every chart except the Thiele-Innes chart, whose coordinates imply the
    mass (use :class:`ThieleInnesBlock`). The block reads the star and planet masses at
    ``system``; a mass that another entry frees is applied first.

    Attributes:
        chart: The coordinate map and its exact volume.
        law: The density of the freed orbits against the orbit reference measure.
            Parameters broadcast over the freed rows.
        rows: The freed Jacobi indices.
        system: Path of the :class:`System` in the model.
    """

    chart: AbstractChart = eqx.field(static=True)
    law: AbstractOrbitLaw
    rows: tuple[int, ...] = eqx.field(static=True)
    system: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate the static choice of chart and rows."""
        if isinstance(self.chart, ThieleInnesChart):
            raise ValueError(
                "ThieleInnesChart implies the star mass; use ThieleInnesBlock"
            )
        _check_rows(self.rows)

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """The chart's names, with ``logit_`` before a size coordinate the law maps."""
        return _coordinate_names(self.chart, self.law.support())

    @property
    def periods(self) -> tuple[float | None, ...]:
        """The chart's periods: ``2 pi`` on each plain angle, ``None`` elsewhere."""
        return self.chart.periods

    @property
    def owns(self) -> str:
        """Path of the stacked orbits."""
        return join(self.system, "orbits")

    @property
    def _star_mass(self) -> str:
        return join(self.system, "star.mass_Msun")

    @property
    def _planet_masses(self) -> str:
        return join(self.system, "planets.mass_Msun")

    @property
    def reads(self) -> tuple[str, ...]:
        """The star and planet masses that set the Jacobi mu of a row."""
        return (self._star_mass, self._planet_masses)

    @property
    def writes(self) -> tuple[str, ...]:
        """Only the owned orbits."""
        return (self.owns,)

    def _mu(self, model) -> Array:
        return enclosed_mu(
            resolve(model, self._star_mass), resolve(model, self._planet_masses)
        )

    def chart_support(self, model) -> dict[str, tuple[Array, Array]]:
        """Bounds of the law in chart coordinates, at the freed rows' rate in ``model``.

        Keys are ``"ln_size"`` (the size coordinate, ``ln a`` or ``ln n``) and ``"e"``
        (the eccentricity), present for the coordinates the law bounds and the chart
        maps; each value is ``(low, high)``. The dict is what the chart's
        ``constrain`` and ``unconstrain`` take.
        """
        mu = self._mu(model)[..., jnp.asarray(self.rows)]
        return _chart_support(self.chart, self.law.support(), mu)

    def _freed(self, x, model):
        """Held orbits, all rows' mu, chart coordinates, freed epochs, log Jacobian."""
        held = resolve(model, self.owns)
        rows = jnp.asarray(self.rows)
        mu = self._mu(model)
        coords, log_jac = self.chart.constrain(
            _stack(self.coordinate_names, self.rows, x), self.chart_support(model)
        )
        return held, mu, coords, held.epoch_d[..., rows], log_jac

    def forward(self, x, model) -> dict[str, FrameOrbit]:
        """Build the freed rows; held rows keep geometry and take the model's rate."""
        held, mu, coords, epoch, _ = self._freed(x, model)
        new = self.chart.to_orbit(
            coords, mu_AU3_per_d2=mu[..., jnp.asarray(self.rows)], epoch_d=epoch
        )
        return {self.owns: _splice(_rated(held, mu), self.rows, new)}

    def log_prob(self, x, model) -> Array:
        """Sum over the freed rows of the law and the chart volume.

        Coordinates the chart excludes (:meth:`AbstractChart.exclude`) read ``-inf``.
        The exactly circular orbit of a disk chart also reads ``-inf`` (see
        :meth:`OrbitLaw.log_prob`): a set of measure zero that no draw lands on.
        """
        _, mu, coords, epoch, log_jac = self._freed(x, model)
        coords, excluded = self.chart.exclude(coords)
        orbit = self.chart.to_orbit(
            coords, mu_AU3_per_d2=mu[..., jnp.asarray(self.rows)], epoch_d=epoch
        )
        total = jnp.sum(
            self.law.log_prob(orbit) + self.chart.log_volume(coords) + log_jac, -1
        )
        return jnp.where(jnp.any(excluded, -1), -jnp.inf, total)

    def sample(self, key, model) -> dict[str, Array]:
        """Draw the freed orbits from the law at the model's mu, as coordinates."""
        held = resolve(model, self.owns)
        rows = jnp.asarray(self.rows)
        k_law, k_chart = jax.random.split(key)
        mu = self._mu(model)[..., rows]
        orbit = self.law.sample(
            k_law, mu_AU3_per_d2=mu, epoch_d=held.epoch_d[..., rows]
        )
        coords = self.chart.sample_coordinates(k_chart, orbit)
        return _unstack(
            self.coordinate_names,
            self.chart.unconstrain(coords, self.chart_support(model)),
        )

    def inverse(self, model) -> dict[str, Array]:
        """Coordinates of the freed rows of the model's orbits."""
        coords = self.chart.from_orbit(_take(resolve(model, self.owns), self.rows))
        return _unstack(
            self.coordinate_names,
            self.chart.unconstrain(coords, self.chart_support(model)),
        )

    @property
    def unit_size(self) -> int | None:
        """Unit-cube length: per row the law's coordinates and the chart's gauge."""
        per_row = self.law.unit_size
        if per_row is None:
            return None
        return len(self.rows) * (per_row + self.chart.gauge_unit_size)

    def from_unit(self, c, model) -> dict[str, Array]:
        """Map a unit-cube point to coordinates with the law of ``sample``.

        ``c`` has shape ``(unit_size,)``, laid out row by row: the law's coordinates
        (:meth:`AbstractOrbitLaw.from_unit`) then the chart's gauge
        (:meth:`AbstractChart.unit_coordinates`).
        """
        c = _unit_rows(self, c)
        held = resolve(model, self.owns)
        rows = jnp.asarray(self.rows)
        n_law = self.law.unit_size
        mu = self._mu(model)[..., rows]
        orbit = self.law.from_unit(
            c[:, :n_law], mu_AU3_per_d2=mu, epoch_d=held.epoch_d[..., rows]
        )
        coords = self.chart.unit_coordinates(c[:, n_law:], orbit)
        return _unstack(
            self.coordinate_names,
            self.chart.unconstrain(coords, self.chart_support(model)),
        )

    def same_layout(self, other) -> bool:
        """Whether ``other`` matches in type, chart, rows and system; not the law."""
        return (
            type(other) is type(self)
            and other.chart == self.chart
            and other.rows == self.rows
            and other.system == self.system
        )

    def same_map(self, other) -> bool:
        """Whether ``other`` has the same layout and the bounds the chart maps.

        The map from coordinates to orbits depends on the law's bounds that the chart
        has a coordinate for, so a law with other such bounds is another map (a traced
        bound counts as different). The shape of the law inside the bounds, and a bound
        no coordinate carries, are not part of the map.
        """
        return self.same_layout(other) and _same_bounds(
            _map_bounds(self.chart, self.law.support()),
            _map_bounds(other.chart, other.law.support()),
        )


class ThieleInnesBlock(AbstractBlock):
    """Free one row of a stacked orbit through the Thiele-Innes chart.

    The coordinates imply the total enclosed mass, so the block also writes the star
    mass, ``Ms = n^2 a^3 / GM - sum of the planet masses up to the row``, and holds a
    law for it. It reads the planet masses and nothing else. Only one row can be
    freed: one star mass cannot absorb two implied masses. The other rows take their
    mean motion from the star mass the block writes.

    The density against Lebesgue ``dx`` is the orbit law summed over the two
    line-of-sight sheets (the chart's inverse sends an orbit and its reflection to the
    same coordinates), plus the star mass law, the chart volume and ``log(M_enclosed)``
    (the Jacobian from the chart's ``d ln mu`` to the star mass). The sum is normalized
    for every law and equals ``log 2`` plus the law for one that is symmetric under the
    reflection, as an isotropic orientation is.

    ``forward`` builds the orbit on the chart's static branch. With a symmetric law the
    fit is the posterior given that branch and ``log_evidence`` is the evidence given
    that branch; a term that tells the sheets apart (radial velocity, phase-dependent
    photometry) needs both branches run and combined as
    ``logaddexp(Z_plus, Z_minus) - log 2``. With an asymmetric law the sum over sheets
    is exact only when no term tells the sheets apart; otherwise use a chart that covers
    both sheets.

    Attributes:
        law: The density of the orbit given the enclosed mass.
        star_mass_Msun: Scalar law on the star mass in solar masses.
        rows: The freed row; exactly one.
        system: Path of the :class:`System` in the model.
        chart: The Thiele-Innes chart; its branch sets the sheet ``forward`` builds.
    """

    law: AbstractOrbitLaw
    star_mass_Msun: priorities.AbstractDistribution
    rows: tuple[int, ...] = eqx.field(static=True)
    system: str = eqx.field(static=True)
    chart: ThieleInnesChart = eqx.field(static=True, default=ThieleInnesChart())

    def __check_init__(self):
        """Validate the static rows; one row only."""
        _check_rows(self.rows)
        if len(self.rows) != 1:
            raise ValueError("ThieleInnesBlock frees exactly one row (one star mass)")

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """The chart's names, with ``logit_`` before a size coordinate the law maps."""
        return _coordinate_names(self.chart, self.law.support())

    @property
    def periods(self) -> tuple[float | None, ...]:
        """The chart's periods: ``2 pi`` on each plain angle, ``None`` elsewhere."""
        return self.chart.periods

    @property
    def owns(self) -> str:
        """Path of the stacked orbits."""
        return join(self.system, "orbits")

    @property
    def _star_mass(self) -> str:
        return join(self.system, "star.mass_Msun")

    @property
    def _planet_masses(self) -> str:
        return join(self.system, "planets.mass_Msun")

    @property
    def reads(self) -> tuple[str, ...]:
        """The planet masses summed into the enclosed mass."""
        return (self._planet_masses,)

    @property
    def writes(self) -> tuple[str, ...]:
        """The owned orbits and the star mass the coordinates imply."""
        return (self.owns, self._star_mass)

    def _inner_mass(self, model) -> Array:
        planets = resolve(model, self._planet_masses)
        return jnp.sum(planets[..., : self.rows[0] + 1], axis=-1)

    def _enclosed_mass(self, coords) -> Array:
        return self.chart.implied_mu(coords)[..., 0] / GM_SUN_AU3_PER_D2

    def chart_support(self, model) -> dict[str, tuple[Array, Array]]:
        """Bounds of the law in chart coordinates, as :meth:`OrbitBlock.chart_support`.

        The Thiele-Innes size coordinate is ``ln n``, so the bounds do not depend on
        ``model``; the argument keeps one signature across blocks.
        """
        return _chart_support(self.chart, self.law.support(), None)

    def _coords(self, x, model):
        """Chart coordinates of block coordinates ``x`` and the map's log Jacobian."""
        return self.chart.constrain(
            _stack(self.coordinate_names, self.rows, x), self.chart_support(model)
        )

    def forward(self, x, model) -> dict[str, object]:
        """Build the row and return it with the star mass it implies."""
        held = resolve(model, self.owns)
        coords, _ = self._coords(x, model)
        new = self.chart.to_orbit(
            coords,
            mu_AU3_per_d2=1.0,
            epoch_d=held.epoch_d[..., jnp.asarray(self.rows)],
        )
        star_mass = self._enclosed_mass(coords) - self._inner_mass(model)
        mu = enclosed_mu(star_mass, resolve(model, self._planet_masses))
        return {
            self.owns: _splice(_rated(held, mu), self.rows, new),
            self._star_mass: star_mass,
        }

    def log_prob(self, x, model) -> Array:
        """Sheet-summed law, star mass law, chart volume and ``log(M_enclosed)``.

        A zero sky block has no orbit (the chart excludes zero size) and reads ``-inf``.
        Exactly face-on, the chart volume is ``+inf``: an integrable singularity that no
        draw lands on.
        """
        held = resolve(model, self.owns)
        coords, log_jac = self._coords(x, model)
        coords, excluded = self.chart.exclude(coords)
        epoch = held.epoch_d[..., jnp.asarray(self.rows)]
        sheets = [
            self.law.log_prob(
                ThieleInnesChart(branch=branch).to_orbit(
                    coords, mu_AU3_per_d2=1.0, epoch_d=epoch
                )
            )
            for branch in (1, -1)
        ]
        enclosed = self._enclosed_mass(coords)
        star_mass = enclosed - self._inner_mass(model)
        total = jnp.sum(
            jnp.logaddexp(*sheets) + self.chart.log_volume(coords) + log_jac, -1
        ) + size_term(self.star_mass_Msun.log_prob(star_mass), jnp.log(enclosed))
        return jnp.where(jnp.any(excluded, -1), -jnp.inf, total)

    def sample(self, key, model) -> dict[str, Array]:
        """Draw the star mass, then the orbit at the implied mu, as coordinates."""
        held = resolve(model, self.owns)
        k_mass, k_law, k_chart = jax.random.split(key, 3)
        enclosed = self.star_mass_Msun.sample(k_mass) + self._inner_mass(model)
        orbit = self.law.sample(
            k_law,
            mu_AU3_per_d2=jnp.reshape(GM_SUN_AU3_PER_D2 * enclosed, (1,)),
            epoch_d=held.epoch_d[..., jnp.asarray(self.rows)],
        )
        coords = self.chart.sample_coordinates(k_chart, orbit)
        return _unstack(
            self.coordinate_names,
            self.chart.unconstrain(coords, self.chart_support(model)),
        )

    def inverse(self, model) -> dict[str, Array]:
        """Coordinates of the freed row of the model's orbits."""
        coords = self.chart.from_orbit(_take(resolve(model, self.owns), self.rows))
        return _unstack(
            self.coordinate_names,
            self.chart.unconstrain(coords, self.chart_support(model)),
        )

    @property
    def unit_size(self) -> int | None:
        """Unit-cube length: the star mass, then the law's coordinates of the row."""
        per_row = self.law.unit_size
        if per_row is None or not self.star_mass_Msun.has_icdf:
            return None
        return 1 + per_row

    def from_unit(self, c, model) -> dict[str, Array]:
        """Map a unit-cube point to coordinates with the law of ``sample``.

        ``c[0]`` gives the star mass through its inverse CDF and ``c[1:]`` the orbit
        through :meth:`AbstractOrbitLaw.from_unit` at the implied rate.
        """
        c = _unit_rows(self, c, rows=1)
        held = resolve(model, self.owns)
        enclosed = self.star_mass_Msun.icdf(c[0, 0]) + self._inner_mass(model)
        orbit = self.law.from_unit(
            c[:, 1:],
            mu_AU3_per_d2=jnp.reshape(GM_SUN_AU3_PER_D2 * enclosed, (1,)),
            epoch_d=held.epoch_d[..., jnp.asarray(self.rows)],
        )
        coords = self.chart.unit_coordinates(c[:, :0], orbit)
        return _unstack(
            self.coordinate_names,
            self.chart.unconstrain(coords, self.chart_support(model)),
        )

    def same_layout(self, other) -> bool:
        """Whether ``other`` matches in type, chart, rows and system; not the law."""
        return (
            type(other) is type(self)
            and other.chart == self.chart
            and other.rows == self.rows
            and other.system == self.system
        )

    def same_map(self, other) -> bool:
        """Whether ``other`` has the same layout and the bounds the chart maps.

        The map from coordinates to orbits depends on the law's bounds that the chart
        has a coordinate for, so a law with other such bounds is another map (a traced
        bound counts as different). The shape of the law inside the bounds, and a bound
        no coordinate carries, are not part of the map.
        """
        return self.same_layout(other) and _same_bounds(
            _map_bounds(self.chart, self.law.support()),
            _map_bounds(other.chart, other.law.support()),
        )


def _mapped_keys(chart, bounds) -> frozenset[str]:
    """The support keys (``"ln_size"``, ``"e"``) a chart has a coordinate for."""
    keys = set()
    if "e" in bounds and chart.eccentricity_coordinate is not None:
        keys.add("e")
    if (
        chart.size_coordinate == "ln_a" and ("a_AU" in bounds or "period_d" in bounds)
    ) or (chart.size_coordinate == "ln_n" and "period_d" in bounds):
        keys.add("ln_size")
    return frozenset(keys)


def _coordinate_names(chart, bounds) -> tuple[str, ...]:
    """Chart names; a mapped size coordinate (the first) is a logit of its interval."""
    names = chart.coordinate_names
    if "ln_size" in _mapped_keys(chart, bounds):
        return (f"logit_{names[0]}", *names[1:])
    return names


def _chart_support(chart, bounds, mu) -> dict:
    """Chart-coordinate bounds from a law's bounds; ``mu`` is the freed rows' rate.

    The eccentricity bound passes through (a disk chart uses only its upper end). The
    size bound becomes a bound on ``ln a`` (from an ``a_AU`` law, or from a period law
    at the given rate) or on ``ln n`` (from a period law only: ``n = 2 pi / P``,
    independent of the masses).
    """
    keys = _mapped_keys(chart, bounds)
    out = {}
    if "e" in keys:
        out["e"] = bounds["e"]
    if "ln_size" in keys:
        if chart.size_coordinate == "ln_n":
            low, high = bounds["period_d"]
            out["ln_size"] = (jnp.log(two_pi / high), jnp.log(two_pi / low))
        elif "a_AU" in bounds:
            out["ln_size"] = tuple(jnp.log(b) for b in bounds["a_AU"])
        else:
            out["ln_size"] = tuple(
                (jnp.log(mu) + 2 * (jnp.log(b) - jnp.log(two_pi))) / 3
                for b in bounds["period_d"]
            )
    return out


def _map_bounds(chart, bounds) -> dict:
    """The part of ``law.support()`` the chart's coordinates are mapped by."""
    keys = _mapped_keys(chart, bounds)
    out = {}
    if "e" in keys:
        out["e"] = bounds["e"]
    if "ln_size" in keys:
        size = (
            "a_AU"
            if chart.size_coordinate == "ln_a" and "a_AU" in bounds
            else "period_d"
        )
        out[size] = bounds[size]
    return out


def _same_bounds(a, b) -> bool:
    """Whether two ``law.support()`` dicts hold the same concrete bounds."""
    if a.keys() != b.keys():
        return False
    for key in a:
        for u, v in zip(a[key], b[key], strict=True):
            if isinstance(u, jax.core.Tracer) or isinstance(v, jax.core.Tracer):
                return False
            if not np.array_equal(np.asarray(u), np.asarray(v)):
                return False
    return True


def _check_rows(rows) -> None:
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(not isinstance(r, int) or isinstance(r, bool) or r < 0 for r in rows)
    ):
        raise ValueError("rows must be a nonempty tuple of distinct nonnegative ints")


def _unit_rows(block, c, rows=None) -> Array:
    """Check a unit-cube vector and shape it ``(rows, per_row)``.

    For the Thiele-Innes block (``rows=1``) the single row is the whole vector, whose
    first entry is the star mass.
    """
    size = block.unit_size
    if size is None:
        raise NotImplementedError(f"{type(block).__name__} has no unit-cube map")
    c = jnp.asarray(c)
    if c.shape != (size,):
        raise ValueError(f"the unit-cube point must have shape ({size},)")
    return c.reshape(rows or len(block.rows), -1)


def _stack(names, rows, x) -> Array:
    """Stack named coordinates to shape ``(..., len(rows), n_coords)``."""
    if set(x) != set(names):
        raise ValueError(f"coordinates must be exactly {names}, got {tuple(x)}")
    coords = jnp.stack([jnp.asarray(x[name]) for name in names], axis=-1)
    if coords.shape[-2:-1] != (len(rows),):
        raise ValueError(f"each coordinate needs a trailing axis of {len(rows)} rows")
    return coords


def _unstack(names, coords) -> dict[str, Array]:
    return {name: coords[..., k] for k, name in enumerate(names)}
