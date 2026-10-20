"""orbix 3.0.0 is the root of the (east, north, away) sky chart release train.

Every sibling that reads orbix rows pins ``orbix>=3.0.0``; orbix itself must stay
the root, so none of the downstream libraries may appear in any of its dependency
groups (that would let an old sibling be installed beside orbix 3). The metadata is
read from ``pyproject.toml`` without installing anything.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

DOWNSTREAM = {"skyscapes", "photomancy", "coronagraphoto", "coronalyze"}


def _requirements():
    """Yield (group, Requirement) for every declared dependency."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    for text in project["dependencies"]:
        yield "dependencies", Requirement(text)
    for extra, texts in project.get("optional-dependencies", {}).items():
        for text in texts:
            yield f"extra:{extra}", Requirement(text)


def test_no_downstream_library_in_any_dependency_group():
    """No sky-chart sibling is a dependency of orbix, in any install."""
    found = [
        (group, req.name) for group, req in _requirements() if req.name in DOWNSTREAM
    ]
    assert not found, f"downstream libraries declared by orbix: {found}"
