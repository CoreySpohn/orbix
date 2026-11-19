"""The block contract: a free entry that owns a node of a model and writes it.

A block turns coordinates ``x`` (a dict of named arrays) into values for one or more
model paths, carries the log volume that makes its coordinate density normalized against
a reference measure, and declares statically which paths it reads and writes. A host
that frees several blocks uses :func:`check_writers` to refuse two writers of one path
and :func:`block_order` to apply the blocks in an order that respects their reads.

This module imports nothing from the rest of orbix, so a domain-free library can
recognize a block by this base class alone.
"""

from abc import abstractmethod
from collections.abc import Iterable, Mapping

import equinox as eqx
from jaxtyping import Array


def paths_overlap(a: str, b: str) -> bool:
    """Whether two model paths are equal or one names a node containing the other.

    ``"dynamics.orbits"`` overlaps ``"dynamics.orbits"`` and ``"dynamics.orbits[0].ex"``
    but not ``"dynamics.orbits_extra"``.
    """
    short, long = sorted((a, b), key=len)
    return long == short or (long.startswith(short) and long[len(short)] in (".", "["))


class AbstractBlock(eqx.Module):
    """Coordinates that write a node of a model, with a normalized-density volume.

    Coordinates ``x`` are a dict of named arrays with one trailing axis per freed row.
    The methods are per draw; a host batches with ``jax.vmap``. ``reads`` and ``writes``
    are static tuples of model paths. ``writes`` contains ``owns`` and any further path
    the block sets, such as a mass the coordinates imply. A path in ``reads`` is read
    from the model the block is given; the block never reads a path outside ``reads``,
    ``owns`` and its own fields.
    """

    @property
    @abstractmethod
    def owns(self) -> str:
        """The path of the node this block owns; the key of its entry in a fit."""

    @property
    @abstractmethod
    def reads(self) -> tuple[str, ...]:
        """Model paths read besides the owned node."""

    @property
    @abstractmethod
    def writes(self) -> tuple[str, ...]:
        """Model paths written, ``owns`` first."""

    @abstractmethod
    def forward(self, x: dict[str, Array], model) -> dict[str, object]:
        """Return ``{path: value}`` for every path in ``writes``."""

    @abstractmethod
    def log_volume(self, x: dict[str, Array], model) -> Array:
        """Return the log volume of the coordinates against the reference measure."""

    @abstractmethod
    def inverse(self, model) -> dict[str, Array]:
        """Return coordinates whose ``forward`` rebuilds the owned node of ``model``."""


def check_writers(
    blocks: Mapping[str, AbstractBlock], leaf_paths: Iterable[str] = ()
) -> None:
    """Refuse two writers of one path.

    ``blocks`` maps each block's key to the block and ``leaf_paths`` are the paths of
    free leaf laws. Raises ``ValueError`` naming the path and both writers when a
    block's key is not its owned path, two blocks write overlapping paths, or a leaf
    law overlaps a path a block writes.
    """
    writers: list[tuple[str, str]] = []
    for key, block in blocks.items():
        if key != block.owns:
            raise ValueError(
                f"block entry {key!r} must be keyed by the path it owns, {block.owns!r}"
            )
        writers.extend((path, f"block {key!r}") for path in block.writes)
    writers.extend((path, f"free leaf {path!r}") for path in leaf_paths)
    for i, (path_a, who_a) in enumerate(writers):
        for path_b, who_b in writers[i + 1 :]:
            if who_a != who_b and paths_overlap(path_a, path_b):
                raise ValueError(
                    f"path {path_a!r} has two writers: {who_a} and {who_b}"
                )


def block_order(blocks: Mapping[str, AbstractBlock]) -> tuple[str, ...]:
    """Keys ordered so each block runs after the blocks that write what it reads.

    The order is stable (ties keep the mapping's order). Raises ``ValueError`` on a
    cycle.
    """
    keys = list(blocks)
    after = {
        k: {
            j
            for j in keys
            if j != k
            and any(
                paths_overlap(w, r) for w in blocks[j].writes for r in blocks[k].reads
            )
        }
        for k in keys
    }
    order: list[str] = []
    while len(order) < len(keys):
        ready = next(
            (k for k in keys if k not in order and after[k] <= set(order)), None
        )
        if ready is None:
            raise ValueError("blocks read each other's writes in a cycle")
        order.append(ready)
    return tuple(order)
