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

    def _freed(self, x, model):
        """The held orbits, all rows' mu, the stacked coordinates and freed epochs."""
        held = resolve(model, self.owns)
        rows = jnp.asarray(self.rows)
        return (
            held,
            self._mu(model),
            _stack(self.chart, self.rows, x),
            held.epoch_d[..., rows],
        )

    def forward(self, x, model) -> dict[str, FrameOrbit]:
        """Build the freed rows; held rows keep geometry and take the model's rate."""
        held, mu, coords, epoch = self._freed(x, model)
        new = self.chart.to_orbit(
            coords, mu_AU3_per_d2=mu[..., jnp.asarray(self.rows)], epoch_d=epoch
        )
        return {self.owns: _splice(_rated(held, mu), self.rows, new)}

    def log_prob(self, x, model) -> Array:
        """Sum over the freed rows of the law and the chart volume."""
        _, mu, coords, epoch = self._freed(x, model)
        orbit = self.chart.to_orbit(
            coords, mu_AU3_per_d2=mu[..., jnp.asarray(self.rows)], epoch_d=epoch
        )
        return jnp.sum(self.law.log_prob(orbit) + self.chart.log_volume(coords), -1)

    def sample(self, key, model) -> dict[str, Array]:
        """Draw the freed orbits from the law at the model's mu, as coordinates."""
        held = resolve(model, self.owns)
        rows = jnp.asarray(self.rows)
        k_law, k_chart = jax.random.split(key)
        orbit = self.law.sample(
            k_law,
            mu_AU3_per_d2=self._mu(model)[..., rows],
            epoch_d=held.epoch_d[..., rows],
        )
        return _unstack(self.chart, self.chart.sample_coordinates(k_chart, orbit))

    def inverse(self, model) -> dict[str, Array]:
        """Coordinates of the freed rows of the model's orbits."""
        coords = self.chart.from_orbit(_take(resolve(model, self.owns), self.rows))
        return _unstack(self.chart, coords)

    def same_map(self, other) -> bool:
        """Whether ``other`` has this chart, rows and system; only the law differs."""
        return (
            type(other) is type(self)
            and other.chart == self.chart
            and other.rows == self.rows
            and other.system == self.system
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

    def forward(self, x, model) -> dict[str, object]:
        """Build the row and return it with the star mass it implies."""
        held = resolve(model, self.owns)
        coords = _stack(self.chart, self.rows, x)
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
        coords = _stack(self.chart, self.rows, x)
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
        return (
            jnp.sum(jnp.logaddexp(*sheets) + self.chart.log_volume(coords), -1)
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
        return _unstack(self.chart, self.chart.sample_coordinates(k_chart, orbit))

    def inverse(self, model) -> dict[str, Array]:
        """Coordinates of the freed row of the model's orbits."""
        coords = self.chart.from_orbit(_take(resolve(model, self.owns), self.rows))
        return _unstack(self.chart, coords)

    def same_map(self, other) -> bool:
        """Whether ``other`` has this chart, rows and system; only laws differ."""
        return (
            type(other) is type(self)
            and other.chart == self.chart
            and other.rows == self.rows
            and other.system == self.system
        )


def _check_rows(rows) -> None:
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(not isinstance(r, int) or isinstance(r, bool) or r < 0 for r in rows)
    ):
        raise ValueError("rows must be a nonempty tuple of distinct nonnegative ints")


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
