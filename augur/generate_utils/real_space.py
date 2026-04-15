"""Real-space (xi) two-point correlation functions for SACC generation.

This module provides infrastructure for adding real-space two-point
statistics (e.g. xi_+, xi_-, gamma_t, w(theta)) to a SACC data vector.

Scale cuts can be specified in terms of:
  - **theta** (angular separation in arcmin, the default unit), or
  - **r** (comoving separation in Mpc/h, converted via the fiducial cosmology).
"""

import numpy as np
import pyccl as ccl
import sacc
import sacc.windows
import warnings

from firecrown.generators.two_point import LogLinearElls
from firecrown.likelihood.two_point import TwoPoint
from augur.utils.firecrown_interface import create_twopoint_filter
from augur.generate_utils.scale_cuts import parse_combination_scale_cut
from augur.generate_utils.tracers import get_tracers
from augur.utils.config_io import parse_array

# Radians → arcmin conversion factor
_RAD_TO_ARCMIN = 180.0 * 60.0 / np.pi

# Plausible single-survey theta range (arcmin); values outside trigger a warning
_THETA_MIN_WARN = 0.1
_THETA_MAX_WARN = 1000.0

# --------------------------------------------------------------------------- #
#  Scale-cut helpers
# --------------------------------------------------------------------------- #


def _get_scale_cuts_real_space(stat_cfg, comb, cosmo=None, dndz=None,
                               tr1=None, tr2=None):
    """Return angular scale cuts (in arcmin) for a tracer combination.

    The config block may specify *either* ``theta_min`` / ``theta_max``
    (in arcmin) *or* ``r_min`` / ``r_max`` (in Mpc/h).  When ``r`` is used,
    the fiducial cosmology and tracer dndz objects (providing ``zav``) are
    required for the conversion.

    The conversion formula is::

        theta [arcmin] = (r [Mpc/h] / h) / chi(z_mean) [Mpc]  ×  _RAD_TO_ARCMIN

    where ``chi`` is the comoving radial distance to the mean redshift of each
    tracer, and the most conservative (smallest ``theta_min``, largest
    ``theta_max``) value across the two tracers is adopted.

    Parameters
    ----------
    stat_cfg : dict
        Per-statistic config block for the real-space data type.
    comb : sequence
        Current tracer-bin combination. Used to resolve list-valued cuts.
    cosmo : pyccl.Cosmology, optional
        Required only when scale cuts are given in ``r`` (Mpc/h).
    dndz : dict, optional
        Tracer-name → N(z) object mapping (each object must expose ``zav``).
        Required only when scale cuts are given in ``r``.
    tr1, tr2 : str, optional
        Tracer names for the bin pair.  Required when scale cuts are given
        in ``r``.

    Returns
    -------
    theta_min : float or None
        Minimum angular scale in **arcmin**.
    theta_max : float or None
        Maximum angular scale in **arcmin**.
    """
    has_theta = 'theta_min' in stat_cfg or 'theta_max' in stat_cfg
    has_r = 'r_min' in stat_cfg or 'r_max' in stat_cfg
    if has_theta and has_r:
        raise ValueError(
            'Cannot specify both angular (theta) and comoving (r) scale cuts.'
        )

    if has_theta:
        theta_min = parse_combination_scale_cut(stat_cfg, 'theta_min', comb)
        theta_max = parse_combination_scale_cut(stat_cfg, 'theta_max', comb)
        return theta_min, theta_max

    if not has_r:
        return None, None

    # --- comoving cuts (Mpc/h) → angular cuts (arcmin) --- #
    if cosmo is None or dndz is None or tr1 is None or tr2 is None:
        raise ValueError(
            'A fiducial cosmology and tracer dndz objects are required to '
            'convert r (Mpc/h) scale cuts to angular scale cuts.'
        )
    warnings.warn(
        'Real-space r-based scale cuts are experimental and assume '
        'the mean redshift of each tracer for the r→theta conversion.'
    )
    h = float(cosmo['h'])
    zmean1 = dndz[tr1].zav
    zmean2 = dndz[tr2].zav
    a12 = np.array([1.0 / (1.0 + zmean1), 1.0 / (1.0 + zmean2)])
    chi12 = ccl.comoving_radial_distance(cosmo, a12)  # Mpc (not Mpc/h)

    r_min = parse_combination_scale_cut(stat_cfg, 'r_min', comb)
    r_max = parse_combination_scale_cut(stat_cfg, 'r_max', comb)
    theta_min = None
    theta_max = None
    if r_min is not None:
        # Adopt the smallest resulting angle (most conservative lower cut)
        theta_min = float(np.min((r_min / h) / chi12)) * _RAD_TO_ARCMIN
    if r_max is not None:
        # Adopt the largest resulting angle (most permissive upper cut)
        theta_max = float(np.max((r_max / h) / chi12)) * _RAD_TO_ARCMIN
    return theta_min, theta_max


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #


def add_real_space_two_point(config, S, sources, dndz, cosmo):
    """Populate a SACC object with real-space (xi) two-point data points.

    Reads the ``statistics_real_space`` block of the config, sets up angular
    bins (in arcmin), applies scale cuts, and writes placeholder zero-valued
    data together with ``TopHatWindow`` bin-edge metadata to the SACC.

    The ``TopHatWindow`` objects store the lower and upper edges (in arcmin)
    of each theta bin so that TJPCov can read exact bin edges from the SACC
    without needing to reconstruct them from centres.

    Optional per-statistic ``ell_for_xi`` config key
    (``minimum``, ``midpoint``, ``maximum``, ``n_log``) is forwarded to the
    Firecrown ``TwoPoint`` constructor as the ``interp_ells_gen`` parameter,
    controlling the intermediate Cl grid used for the Hankel transform.

    Parameters
    ----------
    config : dict
        Full Augur config dict (should contain ``statistics_real_space``).
    S : sacc.Sacc
        SACC object to populate.
    sources : dict
        Tracer-name → firecrown source mapping.
    dndz : dict
        Tracer-name → N(z) object mapping.
    cosmo : pyccl.Cosmology
        Fiducial cosmology.

    Returns
    -------
    stats : list
        Firecrown TwoPoint (real-space) statistics.
    tp_filters : list
        TwoPoint filters encoding angular scale cuts.
    """
    stat_cfg = config.get('statistics_real_space', None)
    if stat_cfg is None:
        return [], []

    warnings.warn(
        "Real-space two-point generation is experimental. "
        "Firecrown real-space TwoPoint factories may not yet be available "
        "for all data types."
    )
    ignore_sc = config['general'].get('ignore_scale_cuts', False)
    ignore_sc_likelihood = config['general'].get('ignore_scale_cuts_likelihood', False)
    if not ignore_sc and ignore_sc_likelihood:
        raise ValueError(
            "Cannot ignore scale cuts in likelihood while "
            "applying them to the data vector."
        )

    stats = []
    tp_filters = []

    for key in stat_cfg:
        tracer_combs = stat_cfg[key]['tracer_combs']

        # Angular bins in arcmin
        theta_edges = parse_array(stat_cfg[key]['theta_edges'])  # arcmin
        theta_centers = np.sqrt(theta_edges[:-1] * theta_edges[1:])

        # Sanity-check theta range
        if np.any(theta_edges < _THETA_MIN_WARN) or np.any(theta_edges > _THETA_MAX_WARN):
            warnings.warn(
                f"Theta edges for '{key}' contain values outside the plausible "
                f"arcmin range [{_THETA_MIN_WARN}, {_THETA_MAX_WARN}]. "
                "Ensure theta_edges are specified in arcmin."
            )

        # ell grid for Cl→xi Hankel transform (configurable per statistic)
        ell_for_xi_cfg = stat_cfg[key].get('ell_for_xi', {})
        interp_ells_gen = (LogLinearElls(**ell_for_xi_cfg)
                           if ell_for_xi_cfg else LogLinearElls())

        for comb in tracer_combs:
            tr1, tr2 = get_tracers(key, comb)

            theta_min_cut, theta_max_cut = _get_scale_cuts_real_space(
                stat_cfg[key], comb, cosmo=cosmo, dndz=dndz, tr1=tr1, tr2=tr2
            )

            # Build boolean mask for scale cuts
            mask = np.ones(len(theta_centers), dtype=bool)
            if theta_min_cut is not None:
                mask &= theta_centers >= theta_min_cut
            if theta_max_cut is not None:
                mask &= theta_centers <= theta_max_cut

            theta_cut = theta_centers[mask]
            lo_cut = theta_edges[:-1][mask]
            hi_cut = theta_edges[1:][mask]

            if len(theta_cut) == 0:
                warnings.warn(
                    f"No theta bins survive scale cuts for '{key}' "
                    f"({tr1} x {tr2}); skipping this combination."
                )
                continue

            if not ignore_sc_likelihood:
                tp_filters.append(
                    create_twopoint_filter(
                        key, tr1, tr2,
                        cut_low=theta_cut[0],
                        cut_high=theta_cut[-1],
                    )
                )

            # When scale cuts are ignored in the data vector, include all bins
            if ignore_sc:
                theta_here = theta_centers
                lo_here = theta_edges[:-1]
                hi_here = theta_edges[1:]
            else:
                theta_here = theta_cut
                lo_here = lo_cut
                hi_here = hi_cut

            # Store per-bin TopHatWindow objects so TJPCov can read exact
            # bin edges from the SACC without reconstructing them from centres.
            windows = [sacc.windows.TopHatWindow(float(lo), float(hi))
                       for lo, hi in zip(lo_here, hi_here)]

            S.add_theta_xi(key, tr1, tr2,
                           theta_here, np.zeros(len(theta_here)),
                           window=windows)

            stats.append(
                TwoPoint(source0=sources[tr1], source1=sources[tr2],
                         sacc_data_type=key,
                         interp_ells_gen=interp_ells_gen)
            )

    return stats, tp_filters
