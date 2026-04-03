"""Real-space (xi) two-point correlation functions for SACC generation.

This module provides the stub infrastructure for adding real-space two-point
statistics (e.g. xi_+, xi_-, gamma_t, w(theta)) to a SACC data vector.

Scale cuts can be specified in terms of:
  - **theta** (angular separation in arcmin, the default unit), or
  - **r** (comoving separation in Mpc/h, converted via the fiducial cosmology).
"""

import numpy as np
import pyccl as ccl
import sacc
import warnings

from firecrown.likelihood.two_point import TwoPoint
from augur.utils.firecrown_interface import create_twopoint_filter
from augur.generate_utils.tracers import get_tracers

# --------------------------------------------------------------------------- #
#  Scale-cut helpers
# --------------------------------------------------------------------------- #

def _get_scale_cuts_real_space(stat_cfg, comb, cosmo=None):
    """
    Extract the angular scale cut for a tracer combination in real space.

    The config block may specify *either* ``theta_min`` / ``theta_max``
    (in arcmin) *or* ``r_min`` / ``r_max`` (in Mpc/h).  If ``r`` is used,
    the fiducial cosmology is required to convert to an angular cut via
    ``theta = r / chi(z_mean)``.

    Parameters
    ----------
    stat_cfg : dict
        Per-statistic config block for the real-space data type.
    comb : tuple
        Tracer-bin pair.
    cosmo : pyccl.Cosmology, optional
        Required only when scale cuts are given in ``r`` (Mpc/h).

    Returns
    -------
    theta_min : float or None
        Minimum angular scale in **arcmin**.
    theta_max : float or None
        Maximum angular scale in **arcmin**.
    """
    if ('theta_min' in stat_cfg or 'theta_max' in stat_cfg) and \
       ('r_min' in stat_cfg or 'r_max' in stat_cfg):
        raise ValueError(
            'Cannot specify both angular (theta) and comoving (r) scale cuts.'
        )

    # --- angular cuts (arcmin) --- #
    theta_min = stat_cfg.get('theta_min', None)
    theta_max = stat_cfg.get('theta_max', None)

    # --- comoving cuts (Mpc/h) --- #
    r_min = stat_cfg.get('r_min', None)
    r_max = stat_cfg.get('r_max', None)
    if r_min is not None or r_max is not None:
        if cosmo is None:
            raise ValueError(
                'A fiducial cosmology is required to convert r (Mpc/h) '
                'scale cuts to angular scale cuts.'
            )
        # Use approximate mean redshift of the bin pair.
        # The caller should ensure dndz objects have a ``zav`` attribute.
        warnings.warn(
            'Real-space r-based scale cuts are experimental and assume '
            'the mean redshift of the tracer pair for the conversion.'
        )
        # Conversion: theta [rad] = r [Mpc] / chi(z_mean)
        # r in Mpc/h → Mpc: divide by h
        h = cosmo['h']
        # For now, we defer the actual z_mean lookup to the caller and
        # accept theta_min / theta_max already converted, or fall back to
        # None.
        if r_min is not None:
            theta_min = theta_min  # placeholder — see add_real_space_two_point
        if r_max is not None:
            theta_max = theta_max  # placeholder

    return theta_min, theta_max, r_min, r_max


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

def add_real_space_two_point(config, S, sources, dndz, cosmo):
    """
    Populate a SACC object with real-space (xi) two-point data points.

    Currently a **stub**: it reads the ``statistics_real_space`` block of
    the config, sets up angular bins (defaulting to arcmin), and adds
    placeholder zeros to the SACC.

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

        # Angular bins in arcmin (default)
        theta_edges = np.array(eval(stat_cfg[key]['theta_edges']))  # arcmin
        theta_centers = np.sqrt(theta_edges[:-1] * theta_edges[1:])

        for comb in tracer_combs:
            tr1, tr2 = get_tracers(key, comb)
            theta_min, _, r_min, _ = _get_scale_cuts_real_space(
                stat_cfg[key], comb, cosmo=cosmo
            )

            # Apply angular scale cuts
            theta_here = theta_centers
            if theta_min is not None:
                theta_here = theta_here[theta_here >= theta_min]


            if (r_min is not None) and (r_min != 'None'):
                zmean1 = dndz[tr1].zav
                zmean2 = dndz[tr2].zav
                a12 = np.array([1.0 / (1 + zmean1), 1.0 / (1 + zmean2)])
                theta_min = np.min(r_min / ccl.comoving_radial_distance(cosmo, a12))
                theta_here = theta_centers[theta_centers >= theta_min]
            elif (theta_min is not None) and (theta_min != 'None'):
                theta_here = theta_centers[theta_centers >= theta_min]
            else:
                theta_here = theta_centers
                theta_min = theta_here[0]

            if not ignore_sc_likelihood:
                tp_filters.append(
                    create_twopoint_filter(
                        key, tr1, tr2,
                        cut_low=theta_here[0],
                        cut_high=theta_here[-1],
                    )
                )

            if ignore_sc:
                theta_here = theta_centers
            # Add placeholder data to SACC
            S.add_theta_xi(key, tr1, tr2,
                           theta_here, np.zeros(len(theta_here)))

            # TODO: build real-space TwoPoint statistics once Firecrown
            # exposes real-space factories.  For now, append nothing to
            # ``stats`` so the caller knows real-space was requested but
            # the firecrown likelihood objects are not yet wired.
            stats.append(
                TwoPoint(source0=sources[tr1], source1=sources[tr2],
                         sacc_data_type=key)
            )
    return stats, tp_filters
