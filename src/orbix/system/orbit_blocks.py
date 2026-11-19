"""Blocks that free rows of a stacked orbit through one of the orbit charts.

The model holds a :class:`System` at the path ``system`` (for example ``"dynamics"`` on
a skyscapes ``Scene``) whose ``orbits`` are stacked over the K Jacobi bodies, inner to
outer. A block frees a static tuple of those rows and reads the held rows from the
model. Coordinates are the chart's own, a dict keyed by ``chart.coordinate_names``, each
of shape ``(len(rows),)`` per draw.

A block composes three peers: a chart (the coordinate map and its volume), an orbit law
(the density, independent of the chart) and the placement (``rows`` and ``system``).
``log_prob`` is the normalized density of the coordinates against Lebesgue ``dx``,
``law.log_prob`` of the orbits plus the chart volume.

``forward`` builds a freed row at the model's stored epoch and, for the mass-reading
charts, at the Jacobi mu of the model's current star and planet masses. A held row keeps
its geometry and takes its mean motion from the masses the block read, so the written
node never mixes two masses.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import priorities
from hwoutils.blocks import AbstractBlock
from hwoutils.constants import two_pi
from hwoutils.model_paths import resolve
from jaxtyping import Array

from orbix.equations.orbit import mean_motion
from orbix.system.charts import AbstractChart, ThieleInnesChart
from orbix.system.core import GM_SUN_AU3_PER_D2, enclosed_mu
from orbix.system.frame_orbit import FrameOrbit
from orbix.system.laws import AbstractOrbitLaw


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
        wall_free: When true, an element the law bounds to a finite interval (see
            :meth:`AbstractOrbitLaw.support`) gets a coordinate on the whole real line
            through a logit on that interval, so a gradient sampler meets no wall; the
            density gains the log derivative of that map. Default false: coordinates
            are the chart's own, with the law's walls. With a disk-valued
            eccentricity chart the eccentricity law must start at zero.
    """

    chart: AbstractChart = eqx.field(static=True)
    law: AbstractOrbitLaw
    rows: tuple[int, ...] = eqx.field(static=True)
    system: str = eqx.field(static=True)
    wall_free: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        """Validate the static choice of chart and rows."""
        if isinstance(self.chart, ThieleInnesChart):
            raise ValueError(
                "ThieleInnesChart implies the star mass; use ThieleInnesBlock"
            )
        _check_rows(self.rows)
        _check_wall_free(self)

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """The chart's coordinate names."""
        return self.chart.coordinate_names

    @property
    def periods(self) -> tuple[float | None, ...]:
        """The chart's periods: ``2 pi`` on each plain angle, ``None`` elsewhere."""
        return self.chart.periods

    @property
    def owns(self) -> str:
        """Path of the stacked orbits."""
        return f"{self.system}.orbits"

    @property
    def _star_mass(self) -> str:
        return f"{self.system}.star.mass_Msun"

    @property
    def _planet_masses(self) -> str:
        return f"{self.system}.planets.mass_Msun"

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

    def _support(self, mu) -> dict:
        """Chart-coordinate bounds of the law at the freed rows' mu; empty if walled."""
        if not self.wall_free:
            return {}
        return _chart_support(self.chart, self.law.support(), mu)

    def _freed(self, x, model):
        """Held orbits, all rows' mu, chart coordinates, freed epochs, log Jacobian."""
        held = resolve(model, self.owns)
        rows = jnp.asarray(self.rows)
        mu = self._mu(model)
        coords, log_jac = self.chart.constrain(
            _stack(self.chart, self.rows, x), self._support(mu[..., rows])
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
        """Sum over the freed rows of the law and the chart volume."""
        _, mu, coords, epoch, log_jac = self._freed(x, model)
        orbit = self.chart.to_orbit(
            coords, mu_AU3_per_d2=mu[..., jnp.asarray(self.rows)], epoch_d=epoch
        )
        return _no_density_if_overflow(
            jnp.sum(
                self.law.log_prob(orbit) + self.chart.log_volume(coords) + log_jac, -1
            )
        )

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
        return _unstack(self.chart, self.chart.unconstrain(coords, self._support(mu)))

    def inverse(self, model) -> dict[str, Array]:
        """Coordinates of the freed rows of the model's orbits."""
        coords = self.chart.from_orbit(_take(resolve(model, self.owns), self.rows))
        mu = self._mu(model)[..., jnp.asarray(self.rows)]
        return _unstack(self.chart, self.chart.unconstrain(coords, self._support(mu)))

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
        return _unstack(self.chart, self.chart.unconstrain(coords, self._support(mu)))

    def same_map(self, other) -> bool:
        """Whether ``other`` matches in chart, rows, system and coordinates."""
        return (
            type(other) is type(self)
            and other.chart == self.chart
            and other.rows == self.rows
            and other.system == self.system
            and other.wall_free == self.wall_free
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
        wall_free: As for :class:`OrbitBlock`; the size coordinate ``ln n`` is bounded
            by a period law (an ``a_AU`` law bounds no coordinate here), and the
            eccentricity law must start at zero.
    """

    law: AbstractOrbitLaw
    star_mass_Msun: priorities.AbstractDistribution
    rows: tuple[int, ...] = eqx.field(static=True)
    system: str = eqx.field(static=True)
    chart: ThieleInnesChart = eqx.field(static=True, default=ThieleInnesChart())
    wall_free: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        """Validate the static rows; one row only."""
        _check_rows(self.rows)
        _check_wall_free(self)
        if len(self.rows) != 1:
            raise ValueError("ThieleInnesBlock frees exactly one row (one star mass)")

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """The chart's coordinate names."""
        return self.chart.coordinate_names

    @property
    def periods(self) -> tuple[float | None, ...]:
        """The chart's periods: ``2 pi`` on each plain angle, ``None`` elsewhere."""
        return self.chart.periods

    @property
    def owns(self) -> str:
        """Path of the stacked orbits."""
        return f"{self.system}.orbits"

    @property
    def _star_mass(self) -> str:
        return f"{self.system}.star.mass_Msun"

    @property
    def _planet_masses(self) -> str:
        return f"{self.system}.planets.mass_Msun"

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

    def _support(self) -> dict:
        """Chart-coordinate bounds of the law; empty unless ``wall_free``."""
        if not self.wall_free:
            return {}
        return _chart_support(self.chart, self.law.support(), None)

    def _coords(self, x):
        """Chart coordinates of wall-free ``x`` and the log Jacobian of the map."""
        return self.chart.constrain(_stack(self.chart, self.rows, x), self._support())

    def forward(self, x, model) -> dict[str, object]:
        """Build the row and return it with the star mass it implies."""
        held = resolve(model, self.owns)
        coords, _ = self._coords(x)
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
        """Sheet-summed law, star mass law, chart volume and ``log(M_enclosed)``."""
        held = resolve(model, self.owns)
        coords, log_jac = self._coords(x)
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
        return _no_density_if_overflow(
            jnp.sum(
                jnp.logaddexp(*sheets) + self.chart.log_volume(coords) + log_jac, -1
            )
            + self.star_mass_Msun.log_prob(star_mass)
            + jnp.log(enclosed)
        )

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
        return _unstack(self.chart, self.chart.unconstrain(coords, self._support()))

    def inverse(self, model) -> dict[str, Array]:
        """Coordinates of the freed row of the model's orbits."""
        coords = self.chart.from_orbit(_take(resolve(model, self.owns), self.rows))
        return _unstack(self.chart, self.chart.unconstrain(coords, self._support()))

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
        return _unstack(self.chart, self.chart.unconstrain(coords, self._support()))

    def same_map(self, other) -> bool:
        """Whether ``other`` matches in chart, rows, system and coordinates."""
        return (
            type(other) is type(self)
            and other.chart == self.chart
            and other.rows == self.rows
            and other.system == self.system
            and other.wall_free == self.wall_free
        )


def _no_density_if_overflow(log_prob) -> Array:
    """Replace a NaN or ``+inf`` log density by ``-inf``.

    Coordinates far enough out overflow the chart (an infinite size, a saturated
    eccentricity) and the law turns ``inf - inf`` into NaN or a volume term into
    ``+inf``; such a point has no density. The gradient at a replaced point is not
    meaningful.
    """
    return jnp.where(jnp.isnan(log_prob) | (log_prob == jnp.inf), -jnp.inf, log_prob)


def _chart_support(chart, bounds, mu) -> dict:
    """Chart-coordinate bounds from a law's bounds; ``mu`` is the freed rows' rate.

    The eccentricity bound passes through. The size bound becomes a bound on ``ln a``
    (from an ``a_AU`` law, or from a period law at the given rate) or on ``ln n`` (from
    a period law only: ``n = 2 pi / P``, independent of the masses).
    """
    out = {}
    if "e" in bounds and chart.eccentricity_coordinate is not None:
        out["e"] = bounds["e"]
    if chart.size_coordinate == "ln_a":
        if "a_AU" in bounds:
            out["ln_size"] = tuple(jnp.log(b) for b in bounds["a_AU"])
        elif "period_d" in bounds:
            out["ln_size"] = tuple(
                (jnp.log(mu) + 2 * (jnp.log(b) - jnp.log(two_pi))) / 3
                for b in bounds["period_d"]
            )
    elif chart.size_coordinate == "ln_n" and "period_d" in bounds:
        low, high = bounds["period_d"]
        out["ln_size"] = (jnp.log(two_pi / high), jnp.log(two_pi / low))
    return out


def _check_wall_free(block) -> None:
    """A disk eccentricity chart maps only an interval that starts at zero."""
    if not block.wall_free or block.chart.eccentricity_coordinate != "disk":
        return
    low = block.law.support().get("e", (None,))[0]
    if (
        low is not None
        and not isinstance(low, jax.core.Tracer)
        and bool(jnp.any(low != 0))
    ):
        raise ValueError(
            "a wall-free block on a disk eccentricity chart needs an eccentricity "
            "law that starts at zero; use ClassicalChart for a lower bound"
        )


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


def _stack(chart, rows, x) -> Array:
    """Stack named coordinates to shape ``(..., len(rows), n_coords)``."""
    names = chart.coordinate_names
    if set(x) != set(names):
        raise ValueError(f"coordinates must be exactly {names}, got {tuple(x)}")
    coords = jnp.stack([jnp.asarray(x[name]) for name in names], axis=-1)
    if coords.shape[-2:-1] != (len(rows),):
        raise ValueError(f"each coordinate needs a trailing axis of {len(rows)} rows")
    return coords


def _unstack(chart, coords) -> dict[str, Array]:
    return {name: coords[..., k] for k, name in enumerate(chart.coordinate_names)}
