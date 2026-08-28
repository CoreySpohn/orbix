"""Sphinx configuration file."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

project = "orbix"
copyright = "2026, Corey Spohn"
author = "Corey Spohn"

try:
    release = _get_version("orbix")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoapi_dirs = ["../src"]
autoapi_ignore = ["**/_version.py"]
autodoc_typehints = "description"

myst_enable_extensions = ["amsmath", "dollarmath"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
master_doc = "index"
html_title = "orbix"
html_theme_options = {
    "repository_url": "https://github.com/CoreySpohn/orbix",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
html_context = {"default_mode": "dark"}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
# Pages are authored output-free as MyST markdown and execute at build.
nb_execution_mode = "auto"
nb_execution_timeout = 300
# Every page executes at build time; a cell that stops working fails the build
# here rather than rendering a broken page on the documentation host.
nb_execution_raise_on_error = True
# ... and prints the failing cell's traceback. Without this the build fails
# naming only the page, which is not enough to fix it from a CI log.
nb_execution_show_tb = True

# Warnings belong in the build log, not stamped across the page. hwostyle asks
# for Inter/Helvetica/Arial and a docs builder has none of them, which emitted a
# findfont warning per figure and pushed ~50 stderr blocks onto the rendered
# pages. "remove-warn" drops them from the page and still reports them here.
nb_output_stderr = "remove-warn"
