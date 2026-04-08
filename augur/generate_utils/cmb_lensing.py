"""CMB lensing statistics for SACC generation.

This module adds harmonic-space CMB lensing auto- and cross-spectra to the
SACC data vector and builds matching firecrown ``TwoPoint`` statistics.
"""

import numpy as np
import sacc
import warnings

from firecrown.likelihood.cmb import CMBConvergence
from firecrown.likelihood.two_point import TwoPoint

from augur.utils.firecrown_interface import create_twopoint_filter


def add_cmb_lensing(config, S, sources, dndz, cosmo):
    """
    Populate a SACC object with harmonic CMB-lensing data points.

    Parameters
    ----------
    config : dict
        Full Augur config dict (should contain ``cmb_lensing``).
    S : sacc.Sacc
        SACC object to populate.
    sources : dict
        Tracer-name → firecrown source mapping (updated in-place if a
        CMB-lensing tracer is added).
    dndz : dict
        Tracer-name → N(z) object mapping (unused for CMB lensing, but
        kept for API consistency).
    cosmo : pyccl.Cosmology
        Fiducial cosmology.

    Returns
    -------
    stats : list
        Firecrown TwoPoint statistics for CMB auto/cross terms.
    tp_filters : list
        TwoPoint filters encoding ell scale cuts.
    """
    cmb_cfg = config.get('cmb_lensing', None)
    if cmb_cfg is None:
        return [], []

    ignore_sc = config['general'].get('ignore_scale_cuts', False)
    ignore_sc_likelihood = config['general'].get('ignore_scale_cuts_likelihood', False)
    if not ignore_sc and ignore_sc_likelihood:
        raise ValueError(
            "Cannot ignore scale cuts in likelihood while "
            "applying them to the data vector."
        )

    # NOTE: For current Firecrown factory metadata extraction, CMB convergence
    # is represented with an NZ-like tracer entry carrying a delta-function
    # source distribution at z_source.
    z_source = cmb_cfg.get('z_source', 1100.0)
    if 'cmb_convergence' not in S.tracers:
        S.add_tracer(
            'NZ', 'cmb_convergence', np.array([z_source]), np.array([1.0]),
            quantity='cmb_convergence',
        )
    if 'cmb_convergence' not in sources:
        sources['cmb_convergence'] = CMBConvergence(
            sacc_tracer='cmb_convergence', z_source=z_source,
        )

    stats = []
    tp_filters = []

    supported = {
        'cmb_convergence_cl',
        'cmbGalaxy_convergenceDensity_cl',
        'cmbGalaxy_convergenceShear_cl_e',
    }

    stat_keys = cmb_cfg.get('statistics', {})
    for key in stat_keys:
        if key not in supported:
            raise NotImplementedError(
                f"CMB lensing statistic '{key}' is not supported in Augur yet. "
                f"Supported keys are: {sorted(supported)}"
            )

        tracer_combs = stat_keys[key].get('tracer_combs', [])
        ell_edges = np.array(eval(stat_keys[key]['ell_edges']))
        ells = np.sqrt(ell_edges[:-1] * ell_edges[1:])
        lmax_cfg = stat_keys[key].get('lmax', None)

        for comb in tracer_combs:
            if key == 'cmb_convergence_cl':
                if len(comb) != 0:
                    raise ValueError(
                        "For cmb_convergence_cl use empty tracer combinations: "
                        "tracer_combs: [[]]"
                    )
                tr1, tr2 = 'cmb_convergence', 'cmb_convergence'
            elif key == 'cmbGalaxy_convergenceDensity_cl':
                if len(comb) != 1:
                    raise ValueError(
                        "For cmbGalaxy_convergenceDensity_cl use single-bin combos, "
                        "e.g. [[0], [1], ...]"
                    )
                tr1, tr2 = 'cmb_convergence', f'lens{comb[0]}'
            elif key == 'cmbGalaxy_convergenceShear_cl_e':
                if len(comb) != 1:
                    raise ValueError(
                        "For cmbGalaxy_convergenceShear_cl_e use single-bin combos, "
                        "e.g. [[0], [1], ...]"
                    )
                tr1, tr2 = 'cmb_convergence', f'src{comb[0]}'
            else:
                raise RuntimeError("Unreachable CMB statistic branch")

            if tr2 not in sources:
                raise ValueError(
                    f"Tracer '{tr2}' needed for '{key}' is missing. "
                    "Ensure the matching sources/lenses section is defined."
                )

            ells_here = ells
            if lmax_cfg is not None and lmax_cfg != 'None':
                lmax = float(lmax_cfg)
                ells_here = ells[ells <= lmax]
                if len(ells_here) == 0:
                    raise ValueError(
                        f"lmax={lmax} removes all ell bins for {key} ({tr1}, {tr2})."
                    )

            if not ignore_sc_likelihood:
                tp_filters.append(
                    create_twopoint_filter(
                        key, tr1, tr2,
                        cut_low=float(ells_here[0]),
                        cut_high=float(ells_here[-1]),
                    )
                )

            if ignore_sc:
                ells_here = ells

            S.add_ell_cl(key, tr1, tr2, ells_here, np.zeros(len(ells_here)))
            stats.append(
                TwoPoint(
                    source0=sources[tr1],
                    source1=sources[tr2],
                    sacc_data_type=key,
                )
            )

    if len(stats) == 0:
        warnings.warn(
            "cmb_lensing section is present but produced no statistics. "
            "Check cmb_lensing.statistics and tracer_combs entries."
        )

    return stats, tp_filters
