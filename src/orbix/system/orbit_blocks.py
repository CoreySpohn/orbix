"""Blocks that free rows of a stacked orbit through one of the orbit charts.

The model holds a :class:`System` at ``system`` (``"dynamics"`` on a skyscapes
``Scene``) whose ``orbits`` are stacked over the K Jacobi bodies, inner to outer. A
block frees a static tuple of those rows and reads the held rows from the model.
Coordinates are the chart's own, a dict keyed by ``chart.coordinate_names``, each of
shape ``(len(rows),)`` per draw.

``forward`` builds a freed row at the model's stored epoch and, for the mass-reading
charts, at the Jacobi mu of the model's current star and planet masses. Held rows are
returned untouched, including their stored mean motion.
"""

import re

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from orbix.system.block import AbstractBlock
from orbix.system.charts import AbstractChart, ThieleInnesChart
from orbix.system.core import GM_SUN_AU3_PER_D2, enclosed_mu
from orbix.system.frame_orbit import FrameOrbit

_STEP = re.compile(r"([A-Za-z_]\w*)((?:\[\d+\])*)")


def _get(model, path: str):
    """Read a dotted path, where ``[k]`` indexes a tuple or list."""
    value = model
    for part in path.split("."):
        found = _STEP.fullmatch(part)
        if found is None:
            raise ValueError(f"malformed model path {path!r}")
        value = getattr(value, found.group(1))
        for index in re.findall(r"\[(\d+)\]", found.group(2)):
            value = value[int(index)]
    return value


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


class OrbitBlock(AbstractBlock):
    """Free some rows of a stacked orbit through a chart that reads the masses.

    Works for every chart except the Thiele-Innes chart, whose coordinates imply the
    mass (use :class:`ThieleInnesBlock`). ``rows`` are the freed Jacobi indices,
    ``system`` the path of the :class:`System` in the model. The block reads the star
    and planet masses at ``system``; a mass that another entry frees is applied first.
    """

    chart: AbstractChart = eqx.field(static=True)
    rows: tuple[int, ...] = eqx.field(static=True)
    system: str = eqx.field(static=True, default="dynamics")

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
    def reads(self) -> tuple[str, ...]:
        """The star and planet masses that set the Jacobi mu of a freed row."""
        return (f"{self.system}.star.mass_Msun", f"{self.system}.planets.mass_Msun")

    @property
    def writes(self) -> tuple[str, ...]:
        """Only the owned orbits."""
        return (self.owns,)

    def _coordinates(self, x) -> Array:
        return _stack(self.chart, self.rows, x)

    def forward(self, x, model) -> dict[str, FrameOrbit]:
        """Build the freed rows; the held rows of the stacked orbits stay intact."""
        held = _get(model, self.owns)
        mu = enclosed_mu(_get(model, self.reads[0]), _get(model, self.reads[1]))[
            ..., jnp.asarray(self.rows)
        ]
        new = self.chart.to_orbit(
            self._coordinates(x),
            mu_AU3_per_d2=mu,
            epoch_d=held.epoch_d[..., jnp.asarray(self.rows)],
        )
        return {self.owns: _splice(held, self.rows, new)}

    def log_volume(self, x, model) -> Array:
        """Sum of the chart's log volume over the freed rows."""
        return jnp.sum(self.chart.log_volume(self._coordinates(x)), axis=-1)

    def inverse(self, model) -> dict[str, Array]:
        """Coordinates of the freed rows of the model's orbits."""
        coords = self.chart.from_orbit(_take(_get(model, self.owns), self.rows))
        return _unstack(self.chart, coords)


class ThieleInnesBlock(AbstractBlock):
    """Free one row of a stacked orbit through the Thiele-Innes chart.

    The coordinates imply the total enclosed mass, so the block also writes the star
    mass, ``Ms = n^2 a^3 / GM - sum of the planet masses up to the row``. It reads the
    planet masses and nothing else. Only one row can be freed: one star mass cannot
    absorb two implied masses.

    The log volume is the chart's, plus ``log(M_enclosed)`` (the Jacobian from the
    chart's ``d ln mu`` to the star mass), plus ``log 2``. The chart covers one
    line-of-sight branch, half of the orientation measure, so against a prior that is
    symmetric under the line-of-sight reflection (isotropic orientation is) the density
    on the branch is twice the prior's, and integrates to one. A prior that is not
    symmetric needs its own correction.
    """

    chart: ThieleInnesChart = eqx.field(static=True, default=ThieleInnesChart())
    rows: tuple[int, ...] = eqx.field(static=True, default=(0,))
    system: str = eqx.field(static=True, default="dynamics")

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
    def reads(self) -> tuple[str, ...]:
        """The planet masses summed into the enclosed mass."""
        return (f"{self.system}.planets.mass_Msun",)

    @property
    def writes(self) -> tuple[str, ...]:
        """The owned orbits and the star mass the coordinates imply."""
        return (self.owns, f"{self.system}.star.mass_Msun")

    def _enclosed_mass(self, coords) -> Array:
        return self.chart.implied_mu(coords)[..., 0] / GM_SUN_AU3_PER_D2

    def forward(self, x, model) -> dict[str, object]:
        """Build the row and return it with the star mass it implies."""
        held = _get(model, self.owns)
        coords = _stack(self.chart, self.rows, x)
        new = self.chart.to_orbit(
            coords,
            mu_AU3_per_d2=1.0,
            epoch_d=held.epoch_d[..., jnp.asarray(self.rows)],
        )
        planets = _get(model, self.reads[0])
        inner = jnp.sum(planets[..., : self.rows[0] + 1], axis=-1)
        return {
            self.owns: _splice(held, self.rows, new),
            self.writes[1]: self._enclosed_mass(coords) - inner,
        }

    def log_volume(self, x, model) -> Array:
        """The chart's log volume to ``(reference measure, star mass)``, one branch."""
        coords = _stack(self.chart, self.rows, x)
        volume = self.chart.log_volume(coords)[..., 0]
        return volume + jnp.log(2.0 * self._enclosed_mass(coords))

    def inverse(self, model) -> dict[str, Array]:
        """Coordinates of the freed row of the model's orbits."""
        coords = self.chart.from_orbit(_take(_get(model, self.owns), self.rows))
        return _unstack(self.chart, coords)


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
