"""MAP initialization for MCMC — converts TI results to NumPyro init dicts.

Provides utilities to seed NUTS/HMC warmup from Thiele-Innes grid search
results, avoiding random initialization in complex posteriors.

Usage::

    from orbix.fitting.init import find_init
    init_vals = find_init(astrom_data, Ms, dist_pc)
    kernel = NUTS(model, init_strategy=init_to_value(values=init_vals))
"""

import jax.numpy as jnp

from orbix.fitting.thiele_innes import (
    thiele_innes_fit,
    thiele_innes_grid_search,
)


def ti_to_init(ti_result, Ms, n_planets=1):
    """Convert a :class:`TIFitResult` to a NumPyro ``init_to_value`` dict.

    Maps the TI-recovered orbital elements back to the raw NumPyro
    parameter names used by :func:`~orbix.fitting.numpyro_model.build_model`.

    Values are clamped to lie strictly within the support of each
    NumPyro distribution (e.g. ``e_raw`` away from 0 for Beta priors).
    Any NaN values (common with very sparse data) are replaced with
    prior-center defaults before clamping.

    Args:
        ti_result: A :class:`TIFitResult` from the TI fitter or grid search.
        Ms: Stellar mass (kg). Needed only for logging/validation.
        n_planets: Number of planets in the model. Default 1.

    Returns:
        Dict mapping NumPyro sample site names to initial values, suitable
        for ``numpyro.infer.init_to_value(values=...)``.
    """
    eps = 1e-6  # small offset to keep values inside open supports

    # Period → log10(T)
    log_P = jnp.log10(ti_result.T)
    log_P = jnp.where(jnp.isnan(log_P), 2.5, log_P)

    # Eccentricity → e_raw.  Clamp to (eps, 1-eps) so it lies strictly
    # inside Beta(a,b) support (0,1).  e=0 from a circular-orbit grid
    # point would give -inf log-prob under Beta.
    e_raw = jnp.where(jnp.isnan(ti_result.e), 0.2, ti_result.e)
    e_raw = jnp.clip(e_raw, eps, 1.0 - eps)

    # Argument of periapsis → w_raw in (eps, 2π-eps)
    sin_w = jnp.where(jnp.isnan(ti_result.sin_w), 0.0, ti_result.sin_w)
    cos_w = jnp.where(jnp.isnan(ti_result.cos_w), 1.0, ti_result.cos_w)
    w = jnp.arctan2(sin_w, cos_w) % (2.0 * jnp.pi)
    w = jnp.clip(w, eps, 2.0 * jnp.pi - eps)

    # cos_i — clamp to (-1+eps, 1-eps) for Uniform(-1,1)
    cos_i = jnp.where(jnp.isnan(ti_result.cos_i), 0.0, ti_result.cos_i)
    cos_i = jnp.clip(cos_i, -1.0 + eps, 1.0 - eps)

    # Longitude of ascending node — clamp to (eps, 2π-eps)
    W_val = jnp.where(jnp.isnan(ti_result.W), jnp.pi, ti_result.W)
    W = W_val % (2.0 * jnp.pi)
    W = jnp.clip(W, eps, 2.0 * jnp.pi - eps)

    # Convert tp → M0: M0 = n * (0 - tp) = -2π·tp / T, then mod 2π
    tp_val = jnp.where(jnp.isnan(ti_result.tp), 0.0, ti_result.tp)
    T_safe = jnp.where(jnp.isnan(ti_result.T), 1.0, ti_result.T)
    M0 = (-tp_val * 2.0 * jnp.pi / T_safe) % (2.0 * jnp.pi)
    M0 = jnp.clip(M0, eps, 2.0 * jnp.pi - eps)

    # Pack into plate-shaped arrays (shape = (n_planets,))
    return {
        "log_P": jnp.full(n_planets, log_P),
        "e_raw": jnp.full(n_planets, e_raw),
        "w_raw": jnp.full(n_planets, w),
        "cos_i": jnp.full(n_planets, cos_i),
        "W": jnp.full(n_planets, W),
        "M0": jnp.full(n_planets, M0),
    }


def find_init(
    astrom_data,
    Ms,
    dist_pc,
    log_T_range=(1.0, 4.0),
    n_log_T=100,
    e_grid=None,
    n_tp=30,
    n_planets=1,
):
    """Find good MCMC initialization via Thiele-Innes grid search.

    Performs a 3D grid search over ``(T, e, tp)`` using the linear
    Thiele-Innes fitter, then converts the best-fit result into a
    NumPyro ``init_to_value`` dict.

    This is the recommended way to initialize NUTS for astrometry-based
    orbit fitting when the true parameters are unknown.

    Args:
        astrom_data: An :class:`~orbix.fitting.data.AstromData` instance.
        Ms: Stellar mass (kg). Scalar.
        dist_pc: Distance to system (parsec). Scalar.
        log_T_range: (min, max) log10(T/days) for the period grid.
            Default ``(1.0, 4.0)`` covers 10–10,000 days.
        n_log_T: Number of period grid points. Default 100.
        e_grid: Array of eccentricity values to search. Default is
            ``[0.0, 0.1, 0.2, ..., 0.8]``.
        n_tp: Number of tp grid points per period. Default 30.
        n_planets: Number of planets in the model. Default 1.

    Returns:
        A dict mapping NumPyro sample site names to initial values,
        suitable for ``numpyro.infer.init_to_value(values=...)``.

    Example::

        from numpyro.infer import MCMC, NUTS, init_to_value
        from orbix.fitting.init import find_init
        from orbix.fitting.numpyro_model import build_model

        init_vals = find_init(astrom_data, Ms, dist_pc)
        model = build_model(Ms, dist_pc, astrom_data=astrom_data)
        kernel = NUTS(model, init_strategy=init_to_value(values=init_vals))
        mcmc = MCMC(kernel, num_warmup=500, num_samples=2000)
        mcmc.run(jax.random.PRNGKey(0))
    """
    if e_grid is None:
        e_grid = jnp.linspace(0.0, 0.8, 9)

    log_T_grid = jnp.linspace(log_T_range[0], log_T_range[1], n_log_T)

    best = thiele_innes_grid_search(
        astrom_data,
        Ms,
        dist_pc,
        log_T_grid,
        e_grid,
        n_tp=n_tp,
    )

    return ti_to_init(best, Ms, n_planets=n_planets)


def find_init_top_k(
    astrom_data,
    Ms,
    dist_pc,
    k=5,
    log_T_range=(1.0, 4.0),
    n_log_T=100,
    e_grid=None,
    n_tp=30,
    n_planets=1,
):
    """Find the top-k MCMC initializations via Thiele-Innes grid search.

    Returns multiple initialization dicts for multi-chain MCMC, seeding
    each chain from a different local maximum in the likelihood surface.

    Args:
        astrom_data: An :class:`~orbix.fitting.data.AstromData` instance.
        Ms: Stellar mass (kg). Scalar.
        dist_pc: Distance to system (parsec). Scalar.
        k: Number of top initializations to return. Default 5.
        log_T_range: (min, max) log10(T/days) for the period grid.
        n_log_T: Number of period grid points. Default 100.
        e_grid: Array of eccentricity values to search. Default is
            ``[0.0, 0.1, 0.2, ..., 0.8]``.
        n_tp: Number of tp grid points per period. Default 30.
        n_planets: Number of planets in the model. Default 1.

    Returns:
        A list of ``k`` dicts, each mapping NumPyro sample site names
        to initial values.
    """
    import jax

    if e_grid is None:
        e_grid = jnp.linspace(0.0, 0.8, 9)

    log_T_grid = jnp.linspace(log_T_range[0], log_T_range[1], n_log_T)
    tp_fracs = jnp.linspace(0.0, 1.0, n_tp, endpoint=False)

    # Build the 3D grid and evaluate
    def _fit_single(log_T, e_val, tp_frac):
        T = 10.0**log_T
        tp = tp_frac * T
        return thiele_innes_fit(astrom_data, T, e_val, tp, Ms, dist_pc)

    log_T_flat = jnp.repeat(jnp.asarray(log_T_grid), len(e_grid) * n_tp)
    e_flat = jnp.tile(
        jnp.repeat(jnp.asarray(e_grid), n_tp),
        len(log_T_grid),
    )
    tp_flat = jnp.tile(tp_fracs, len(log_T_grid) * len(e_grid))

    results = jax.vmap(_fit_single)(log_T_flat, e_flat, tp_flat)

    # Get top-k indices
    top_k_indices = jnp.argsort(results.log_likelihood)[-k:][::-1]

    # Extract results and convert to init dicts
    init_dicts = []
    for idx in top_k_indices:
        single = jax.tree.map(lambda x: x[idx], results)
        init_dicts.append(ti_to_init(single, Ms, n_planets=n_planets))

    return init_dicts
