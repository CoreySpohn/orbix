"""Orbit fitting module — pure JAX likelihoods and forward models.

This module provides differentiable forward models, data containers, and
likelihood functions for joint RV + astrometry + imaging orbit fitting.
The core components are pure JAX (no sampler dependency). The optional
NumPyro model builder requires ``pip install orbix[fitting]``.
"""

from orbix.fitting.autodiff import diff_solve_trig
from orbix.fitting.data import AstromData, ImagingData, NullData, RVData
from orbix.fitting.eig import (
    alias_breaking_eig,
    evaluate_candidates,
    geometric_eig,
)
from orbix.fitting.forward import predict_astrometry, predict_photometry, predict_rv
from orbix.fitting.grid_search import (
    AbstractGridStrategy,
    AbstractShapeParam,
    AdaptiveImportanceSampler,
    EccVectorShape,
    ParamBounds,
    ParticlePosterior,
    grid_search,
)
from orbix.fitting.init import (
    find_init,
    find_init_top_k,
    ti_to_init,
)
from orbix.fitting.laplace import (
    LaplaceMixtureResult,
    LaplaceResult,
    map_laplace_fit,
    map_laplace_mixture_fit,
)
from orbix.fitting.likelihoods import (
    loglike_astrom,
    loglike_imaging,
    loglike_null,
    loglike_rv_marginalized,
)
from orbix.fitting.priors import (
    ECC_PRIOR_NAMES,
    eccentricity_disk_transform,
    period_to_sma,
    sample_ecc_prior,
)
from orbix.fitting.thiele_innes import (
    TIFitResult,
    thiele_innes_fit,
    thiele_innes_grid_search,
)

__all__ = [
    # Differentiable Kepler solver
    "diff_solve_trig",
    # Data containers
    "RVData",
    "AstromData",
    "NullData",
    "ImagingData",
    # Forward models
    "predict_rv",
    "predict_astrometry",
    "predict_photometry",
    # Likelihoods
    "loglike_rv_marginalized",
    "loglike_astrom",
    "loglike_null",
    "loglike_imaging",
    # Priors
    "eccentricity_disk_transform",
    "period_to_sma",
    "sample_ecc_prior",
    "ECC_PRIOR_NAMES",
    # Thiele-Innes fitter
    "TIFitResult",
    "thiele_innes_fit",
    "thiele_innes_grid_search",
    # Initialization
    "ti_to_init",
    "find_init",
    "find_init_top_k",
    # MAP + Laplace
    "LaplaceResult",
    "map_laplace_fit",
    # Multi-start MAP + Laplace mixture
    "LaplaceMixtureResult",
    "map_laplace_mixture_fit",
    # Bayesian Experimental Design
    "geometric_eig",
    "alias_breaking_eig",
    "evaluate_candidates",
    # Grid-search (adaptive importance sampling)
    "AbstractGridStrategy",
    "AbstractShapeParam",
    "AdaptiveImportanceSampler",
    "EccVectorShape",
    "ParamBounds",
    "ParticlePosterior",
    "grid_search",
]
