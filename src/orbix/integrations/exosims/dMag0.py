"""Calculate the dimmest detectable dMag values for an EXOSIMS input."""

import hashlib
import pickle
import tempfile
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap

from orbix.integrations.exosims._availability import is_available

if is_available():
    import astropy.units as u
    import EXOSIMS
    from EXOSIMS.util.utils import dictToSortedStr, genHexStr
    from tqdm import tqdm

# Define a machine-local cache directory under the user's home directory
CACHE_DIR = Path(Path.home(), ".orbix", "cache", "dMag0_grid")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def safe_cache_load(cache_path):
    """Safely load a pickle file, handling corruption gracefully."""
    try:
        if cache_path.exists() and cache_path.stat().st_size > 0:
            with cache_path.open("rb") as f:
                return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, OSError):
        # File is corrupted, remove it
        try:
            cache_path.unlink()
        except OSError:
            pass
    return None


def safe_cache_save(cache_path, data):
    """Safely save data to a pickle file using atomic write."""
    # Write to a temporary file first
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=cache_path.parent, suffix=".tmp", delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            pickle.dump(data, tmp_file)
            tmp_file.flush()
            # Atomic rename (on most filesystems)
            tmp_path.replace(cache_path)
        except Exception:
            # Clean up on failure
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise


class dMag0Grid(eqx.Module):
    """A simple Equinox module packing a 4D dMag grid and its coordinate axes."""

    # Regular array fields
    kEZs: jnp.ndarray  # shape (N_kEZ,)
    # days
    int_times: jnp.ndarray  # shape (N_t,)
    # shape (N_fZ, N_kEZ, N_alpha, N_t)
    # NOTE: transposed from (N_fZ, N_kEZ, N_t, N_alpha) in the init
    #  which it gets calculated as
    grid: jnp.ndarray
    log_fZs: jnp.ndarray
    alphas: jnp.ndarray  # shape (N_alpha,)
    log_alphas: jnp.ndarray

    # Static scalar fields
    inv_log_fZ_step: float
    n_fZ: int
    inv_log_alpha_step: float
    n_alpha: int
    n_kEZ: int

    def __init__(self, fZs, kEZs, alphas, int_times, grid):
        """Create necessary parameters from the grid."""
        # Store original arrays
        self.kEZs = kEZs
        self.alphas = alphas
        self.int_times = int_times
        self.grid = grid.transpose(0, 1, 3, 2)

        # Store dimensions for bounds checking
        self.n_fZ = len(fZs)
        self.n_alpha = len(alphas)
        self.n_kEZ = len(kEZs)

        # fZ is log-spaced - store log values and step
        self.log_fZs = jnp.log10(fZs)
        log_fZ_step = self.log_fZs[1] - self.log_fZs[0]
        self.inv_log_fZ_step = 1.0 / log_fZ_step

        # Alpha is log spaced
        self.log_alphas = jnp.log10(alphas)
        log_alpha_step = self.log_alphas[1] - self.log_alphas[0]
        self.inv_log_alpha_step = 1.0 / log_alpha_step

    def _alpha_dMag_mask_sing(self, trig_solver, alpha, dMag, fZ, kEZ):
        """Calculate detection probability directly from this grid.

        Returns:
        -------
        pdet : (n_tint)
        """
        # First check the geometric constraint set by the IWA and OWA
        alpha_min = self.alphas[0]
        alpha_max = self.alphas[-1]
        geom_mask = (alpha >= alpha_min) & (alpha <= alpha_max)

        # zodi index
        log_fZ = jnp.log10(fZ)
        fZ_ind = (log_fZ - self.log_fZs[0]) * self.inv_log_fZ_step
        fZ0 = jnp.clip(jnp.floor(fZ_ind).astype(jnp.int32) + 1, 0, self.n_fZ - 2)

        # alpha indices in grid
        log_alpha = jnp.log10(alpha)
        a_ind = (log_alpha - self.log_alphas[0]) * self.inv_log_alpha_step
        a0 = jnp.clip(a_ind.astype(jnp.int32), 0, self.n_alpha - 1)

        # kEZ indices in grid
        kEZ_ind = jnp.searchsorted(self.kEZs, kEZ, side="right") - 1

        # Calculate interpolation weights once
        dalpha = (a_ind - a0).astype(self.grid.dtype)

        # Gather & interpolate
        # Function for a single (orbit, time) pair
        def get_slice_single(a0_val, dalpha_val, fz, kEZ_ind):
            # TODO: Implement kEZ handling (2nd dimension)
            patch = lax.dynamic_slice(
                self.grid, (fz, kEZ_ind, 0, a0_val), (1, 1, self.grid.shape[-2], 2)
            )[0, 0]
            # Linear interpolation
            return patch[:, 0] + dalpha_val * (patch[:, 1] - patch[:, 0])

        # Vectorize over all orbits (for a given time point and zodiacal index)
        get_slice_orbits = vmap(get_slice_single, in_axes=(0, 0, None, None))

        # Apply the fully vectorized function to all data at once
        result = get_slice_orbits(a0, dalpha, fZ0, kEZ_ind)

        # Calculate the detection values for each orbit
        # NOTE: signbit returns True if the value is negative
        det_mask = geom_mask[..., None] & jnp.signbit(dMag[..., None] - result)

        return det_mask

    def _alpha_dMag_mask(self, alpha, dMag, fZ_vals, kEZ_val):
        """Calculate detection probability directly from this grid.

        Returns:
        -------
        pdet : (ntimes, n_tint)
        """
        # First check the geometric constraint set by the IWA and OWA
        alpha_min = self.alphas[0]
        alpha_max = self.alphas[-1]
        geom_mask = (alpha >= alpha_min) & (alpha <= alpha_max)

        # zodi index
        log_fZ = jnp.log10(fZ_vals)
        fZ_ind = (log_fZ - self.log_fZs[0]) * self.inv_log_fZ_step
        fZ0 = jnp.clip(jnp.floor(fZ_ind).astype(jnp.int32) + 1, 0, self.n_fZ - 2)

        # alpha indices in grid
        log_alpha = jnp.log10(alpha)
        a_ind = (log_alpha - self.log_alphas[0]) * self.inv_log_alpha_step
        a0 = jnp.clip(a_ind.astype(jnp.int32), 0, self.n_alpha - 1)

        # kEZ index in grid
        kEZ_ind = jnp.searchsorted(self.kEZs, kEZ_val, side="right") - 1

        # Calculate interpolation weights once
        dalpha = (a_ind - a0).astype(self.grid.dtype)

        # Gather & interpolate
        # Function for a single (orbit, time) pair
        def get_slice_single(a0_val, dalpha_val, fz, kEZ_ind):
            # TODO: Implement kEZ handling (2nd dimension)
            patch = lax.dynamic_slice(
                self.grid, (fz, kEZ_ind, 0, a0_val), (1, 1, self.grid.shape[-2], 2)
            )[0, 0]
            # Linear interpolation
            return patch[:, 0] + dalpha_val * (patch[:, 1] - patch[:, 0])

        # Vectorize over all orbits (for a given time point and zodiacal index)
        get_slice_orbits = vmap(get_slice_single, in_axes=(0, 0, None, None))

        # Vectorize over all time points
        get_slice_all = vmap(get_slice_orbits, in_axes=(1, 1, 0, None))

        # Apply the fully vectorized function to all data at once
        result = get_slice_all(a0, dalpha, fZ0, kEZ_ind)

        # Transpose from (nt, norb, n_tint) to (norb, nt, n_tint)
        dim = jnp.transpose(result, (1, 0, 2))

        # Calculate the detection values for each orbit
        # NOTE: signbit returns True if the value is negative
        det_mask = geom_mask[..., None] & jnp.signbit(dMag[..., None] - dim)

        return det_mask

    def _pdet_alpha_dMag_sing(self, trig_solver, alpha, dMag, fZ, kEZ):
        """Calculate detection probability directly from this grid.

        Returns:
        -------
        pdet : (n_tint)
        """
        det_mask = self._alpha_dMag_mask_sing(trig_solver, alpha, dMag, fZ, kEZ)

        # Average over all orbits to get probability of detection
        pdet = det_mask.mean(axis=0)
        return pdet

    @eqx.filter_jit
    def pdet_alpha_dMag_sing(self, trig_solver, alpha, dMag, fZ, kEZ):
        """JIT-compiled version of _pdet_alpha_dMag_sing."""
        return self._pdet_alpha_dMag_sing(trig_solver, alpha, dMag, fZ, kEZ)

    def _pdet_alpha_dMag(self, trig_solver, alpha, dMag, fZ_vals, kEZ_vals):
        """Calculate detection probability directly from this grid.

        Returns:
        -------
        pdet : (ntimes, n_tint)
        """
        det_mask = self._alpha_dMag_mask(alpha, dMag, fZ_vals, kEZ_vals)

        # Average over all orbits to get probability of detection
        # (ntimes, n_tint)
        pdet = det_mask.mean(axis=0)
        return pdet

    def _pdet_planets(self, trig_solver, times, planets, fZ_vals, kEZ_vals):
        """Calculate detection probability directly from this grid.

        Returns:
        -------
        pdet : (ntimes, n_tint)
        """
        # Propagation
        alpha, dMag = planets.alpha_dMag(trig_solver, times)

        return self._pdet_alpha_dMag(trig_solver, alpha, dMag, fZ_vals, kEZ_vals)

    @eqx.filter_jit
    def pdet_planets(self, trig_solver, times, planets, fZ_vals, kEZ_vals):
        """JIT-compiled version of img_pdet_grid."""
        return self._pdet_planets(trig_solver, times, planets, fZ_vals, kEZ_vals)

    @eqx.filter_jit
    def pdet_alpha_dMag(self, trig_solver, alpha, dMag, fZ_vals, kEZ_vals):
        """JIT-compiled version of img_pdet_grid."""
        return self._pdet_alpha_dMag(trig_solver, alpha, dMag, fZ_vals, kEZ_vals)

    @eqx.filter_jit
    def alpha_dMag_mask(self, alpha, dMag, fZ_vals, kEZ_vals):
        """JIT-compiled version of img_pdet_grid."""
        return self._alpha_dMag_mask(alpha, dMag, fZ_vals, kEZ_vals)

    def _dyn_comp_single(
        self,
        trig_solver,
        alpha: jnp.ndarray,  # (N_orb,)
        dmag: jnp.ndarray,  # (N_orb,)
        fz: float,
        kEZ: float,
        valid_mask: jnp.ndarray,  # (N_orb,) bool
    ):
        """Dynamic completeness for **one** star at one epoch.

        All input vectors have the fixed length N_orb
        `valid_mask` is True for the orbits that are still plausible.
        """
        # Reduce peak memory by streaming across orbits instead of
        # materialising a (N_orb, N_tint) array. Accumulate in float32 to
        # avoid inadvertent float64 upcasts.
        acc_dtype = jnp.float32
        init = jnp.zeros(self.int_times.shape, dtype=acc_dtype)

        def body(carry, inputs):
            a_i, d_i, v_i = inputs
            # Call the vectorised helper with a length-1 batch to reuse logic
            a1 = jnp.reshape(a_i, (1,))
            d1 = jnp.reshape(d_i, (1,))
            det_i = self._alpha_dMag_mask_sing(trig_solver, a1, d1, fz, kEZ)[0]
            det_i = det_i.astype(acc_dtype) * v_i.astype(acc_dtype)
            return carry + det_i, None

        summed, _ = lax.scan(body, init, (alpha, dmag, valid_mask))
        dyn_comp = summed / alpha.shape[0]

        # integration-time-normalised figure of merit
        comp_div_t = dyn_comp / self.int_times.astype(acc_dtype)
        return comp_div_t

    @eqx.filter_jit
    def dyn_comp(self, trig_solver, alpha_all, dmag_all, fz, kEZ, valid_mask):
        """JIT-compiled version of _dyn_comp_single."""
        return self._dyn_comp_single(
            trig_solver, alpha_all, dmag_all, fz, kEZ, valid_mask
        )

    @eqx.filter_jit
    def dyn_comp_vec(self, trig_solver, alpha_all, dmag_all, fz, kEZ, valid_mask):
        """Vectorized version of dyn_comp over multiple times."""
        return vmap(self._dyn_comp_single, in_axes=(None, 1, 1, 0, None, None))(
            trig_solver, alpha_all, dmag_all, fz, kEZ, valid_mask
        )


def dMag0_grid(SS, mode, int_times, nEZ_range, n_fZs=10, n_alphas=50, n_kEZs=3):
    """Wraps the call to EXOSIMS.calc_dMag_per_intTime."""
    # Get the range of fZ and JEZ values to use in the grid
    TL = SS.TargetList
    OS = TL.OpticalSystem
    ZL = SS.ZodiacalLight

    int_times_d = int_times.to_value(u.d)
    # Generate a hex string that only includes parameters affecting dMag0
    dMag0_hex = gen_dMag0_hex(mode, SS)

    # Base cache string without per-star information
    cache_base = (
        EXOSIMS.__version__
        + TL.__class__.__name__
        + TL.StarCatalog.__class__.__name__
        + ZL.__class__.__name__
        + OS.__class__.__name__
        + SS.Observatory.__class__.__name__
        + dMag0_hex  # Use our new hex instead of mode["hex"]
        + f"n_fZs_{n_fZs}"
        + f"n_alphas_{n_alphas}"
        + f"n_EZ_range_{nEZ_range[0]}-{nEZ_range[-1]}"
        + f"int_times{int_times_d[0]:.2f}{int_times_d[-1]:.2f}{int_times_d.shape[0]}"
    )

    # Get the JEZ values as a copy to avoid modifying the original value
    JEZ0 = TL.JEZ0[mode["hex"]].copy()
    # Calculate the range of kEZ values
    inclinations = np.linspace(0, 180, 180) * u.deg
    fbeta_range = ZL.calc_fbeta(inclinations)
    kEZ_inc = fbeta_range * (1 - (np.sin(inclinations) ** 2) / 2)
    kEZs = np.linspace(
        np.min(kEZ_inc) * np.min(nEZ_range), np.max(kEZ_inc) * np.max(nEZ_range), n_kEZs
    )

    # Get the fZMap which has all stars
    fZMap = ZL.fZMap[mode["systName"]]

    # Create the grid of alpha values
    IWA, OWA = mode["IWA"].to_value(u.arcsec), mode["OWA"].to_value(u.arcsec)
    alphas = np.linspace(IWA, OWA, n_alphas) << u.arcsec

    # Create the jax arrays for the dMag0_grid module
    alphas_jnp = jnp.array(alphas.to_value(u.arcsec))
    kEZs_jnp = jnp.array(kEZs)
    int_times_jnp = jnp.array(int_times.to_value(u.d))

    # Dictionary to store all grid objects
    all_grids = {}

    # Loop through all stars and save the dMag0 grid for each star
    for sInd in tqdm(range(TL.nStars), desc="Creating dMag0 grid", position=0):
        # Create per-star cache key including the star name
        star_name = TL.Name[sInd]
        star_cache_key = cache_base + f"star_{star_name}"
        star_cache_hash = hashlib.sha256(star_cache_key.encode()).hexdigest()[:16]

        # Check for existing cache for this specific star
        cache_path = CACHE_DIR / f"{star_cache_hash}.pkl"

        # Try to load from cache first
        cached_grid = safe_cache_load(cache_path)
        if cached_grid is not None:
            all_grids[sInd] = cached_grid
            continue

        # If not in cache, compute the grid for this star
        _dMag0s = np.zeros(
            [n_fZs, len(kEZs), n_alphas, len(int_times)], dtype=np.float32
        )
        # Get the 0.01 and 0.99 quantiles of fZMap for the star
        fZ_range = np.quantile(fZMap[sInd], [0.01, 0.99])
        fZs = np.logspace(np.log10(fZ_range[0]), np.log10(fZ_range[1]), n_fZs)
        _sInd = np.repeat(sInd, len(int_times))
        fZs_jnp = jnp.array(fZs)
        approx_r2 = (alphas.to_value(u.rad) * TL.dist[sInd].to_value(u.AU)) ** 2
        for fZ_ind, fZ in enumerate(fZs):
            # repeat the fZ value for each integration time
            _fZ = np.repeat(fZ, len(int_times)) << SS.fZ_unit
            for kEZ_ind, _kEZ in enumerate(kEZs):
                for alpha_ind, _alpha in enumerate(alphas):
                    # Add a tiny amount to the first alpha to avoid nans
                    if alpha_ind == 0:
                        _alpha += 1e-6 * u.arcsec
                    # On last iteration, subtract a small amount to avoid
                    # singularities
                    elif alpha_ind == len(alphas) - 1:
                        _alpha -= 1e-6 * u.arcsec

                    # JEZ is a function of kEZ and alpha and is the same for all
                    # integration times but depends on the star and observing mode
                    _JEZ = _kEZ * JEZ0[sInd] / approx_r2[alpha_ind]
                    # All inputs should have the same shape as int_times
                    _dMag0s[fZ_ind, kEZ_ind, alpha_ind] = OS.calc_dMag_per_intTime(
                        int_times,
                        TL,
                        _sInd,
                        _fZ,
                        _JEZ,
                        _alpha,
                        mode,
                        analytic_only=True,
                    )

        _dMag0s_jnp = jnp.array(_dMag0s, dtype=jnp.float16)
        star_grid = dMag0Grid(
            fZs=fZs_jnp,
            kEZs=kEZs_jnp,
            alphas=alphas_jnp,
            int_times=int_times_jnp,
            grid=_dMag0s_jnp,
        )

        # Cache the individual star's grid safely
        safe_cache_save(cache_path, star_grid)
        all_grids[sInd] = star_grid

    return all_grids


def gen_dMag0_hex(mode, SS):
    """Create a hex string similar to mode['hex'] without unnecessary params.

    Args:
        mode (dict): EXOSIMS observing mode dictionary
        SS (SurveySimulation): EXOSIMS SurveySimulation object

    Returns:
        str: Hex string for caching
    """
    OS = SS.TargetList.OpticalSystem
    PP = SS.PostProcessing
    # Get the instrument and system for this mode
    inst = [
        inst
        for inst in OS._outspec["scienceInstruments"]
        if inst["name"] == mode["instName"]
    ][0]
    syst = [
        syst
        for syst in OS._outspec["starlightSuppressionSystems"]
        if syst["name"] == mode["systName"]
    ][0]

    # Create a minimal dictionary with only the parameters that affect dMag0
    dMag0_params = {
        # Mode parameters
        "losses": mode.get("losses", None),
        "lam": mode.get("lam", None),
        "SNR": mode.get("SNR", None),
        "deltaLam": mode.get("deltaLam", None),
        "IWA": mode.get("IWA", None),
        "OWA": mode.get("OWA", None),
        # Instrument parameters
        "inst_name": inst.get("name", None),
        "QE": inst.get("QE", None),
        "optics": inst.get("optics", None),
        "pixelScale": inst.get("pixelScale", None),
        "pixelSize": inst.get("pixelSize", None),
        "sread": inst.get("sread", None),
        "idark": inst.get("idark", None),
        "CIC": inst.get("CIC", None),
        "texp": inst.get("texp", None),
        "PCeff": inst.get("PCeff", None),
        "ENF": inst.get("ENF", None),
        # System parameters
        "syst_name": syst.get("name", None),
        "syst_optics": syst.get("optics", None),
        "core_contrast": syst.get("core_contrast", None),
        "occ_trans": syst.get("occ_trans", None),
        "core_thruput": syst.get("core_thruput", None),
        "BW": syst.get("BW", None),
        "core_area": syst.get("core_area", None),
        "syst_lam": syst.get("lam", None),
    }
    if "imag" in mode["instName"]:
        dMag0_params["ppFact"] = PP._outspec["ppFact"]
    else:
        dMag0_params["ppFact"] = PP._outspec["ppFact_char"]
        if "lenslSamp" in inst:
            dMag0_params["lenslSamp"] = inst["lenslSamp"]
            dMag0_params["Rs"] = inst["Rs"]

    # Convert any quantities to their base values
    for key, val in dMag0_params.items():
        if hasattr(val, "value"):
            dMag0_params[key] = val.value

    # Create string representation and hash
    param_str = dictToSortedStr(dMag0_params)
    return genHexStr(param_str)
