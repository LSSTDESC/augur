"""Harmonic-space (C_ell) two-point statistics for SACC generation."""

import numpy as np
import pyccl as ccl
import sacc

from firecrown.likelihood.two_point import TwoPoint
from augur.utils.firecrown_interface import create_twopoint_filter
from augur.generate_utils.tracers import get_tracers
from augur.generate_utils.scale_cuts import parse_combination_scale_cut


def _get_scale_cuts(stat_cfg, comb):
    """
    Extract the ell/k scale-cut for a tracer combination.

    ``lmax`` and ``kmax`` may be provided either as scalars (applied to all
    combinations) or as lists with one entry per ``tracer_combs`` element.

    Parameters
    ----------
    stat_cfg : dict
        Per-statistic config block.
    comb : tuple
        Tracer-bin pair.

    Returns
    -------
    lmax : float or None
    kmax : float or None
    """
    if 'kmax' in stat_cfg and 'lmax' in stat_cfg:
        raise ValueError('Cannot specify both lmax and kmax for scale cuts.')

    lmax = parse_combination_scale_cut(stat_cfg, 'lmax', comb)
    kmax = parse_combination_scale_cut(stat_cfg, 'kmax', comb)
    return lmax, kmax


def add_harmonic_two_point(config, S, sources, dndz, cosmo):
    """
    Populate a SACC object with harmonic-space (C_ell) two-point data points
    and build the corresponding firecrown ``TwoPoint`` statistics.

    Parameters
    ----------
    config : dict
        Full Augur config dict.
    S : sacc.Sacc
        SACC object to populate.
    sources : dict
        Tracer-name → firecrown source mapping.
    dndz : dict
        Tracer-name → N(z) object mapping.
    cosmo : pyccl.Cosmology
        Fiducial cosmology (needed for kmax → lmax conversion).

    Returns
    -------
    stats : list[TwoPoint]
        Firecrown TwoPoint statistics.
    tp_filters : list
        TwoPoint filters encoding scale cuts for the likelihood.
    """
    stat_cfg = config['statistics']
    stats = []
    tp_filters = []

    ignore_sc = config['general'].get('ignore_scale_cuts', False)
    ignore_sc_likelihood = config['general'].get('ignore_scale_cuts_likelihood', False)
    if not ignore_sc and ignore_sc_likelihood:
        raise ValueError(
            "Cannot ignore scale cuts in likelihood while "
            "applying them to the data vector."
        )

    bandpower = config['general'].get('bandpower_windows', 'None')

    for key in stat_cfg:
        tracer_combs = stat_cfg[key]['tracer_combs']
        ell_edges = eval(stat_cfg[key]['ell_edges'])
        ells = np.sqrt(ell_edges[:-1] * ell_edges[1:])  # geometric mean

        for comb in tracer_combs:
            lmax, kmax = _get_scale_cuts(stat_cfg[key], comb)
            tr1, tr2 = get_tracers(key, comb)

            if (kmax is not None) and (kmax != 'None'):
                zmean1 = dndz[tr1].zav
                zmean2 = dndz[tr2].zav
                a12 = np.array([1.0 / (1 + zmean1), 1.0 / (1 + zmean2)])
                lmax = np.min(kmax * ccl.comoving_radial_distance(cosmo, a12))
                ells_here = ells[ells <= lmax]
            elif (lmax is not None) and (lmax != 'None'):
                ells_here = ells[ells <= lmax]
            else:
                ells_here = ells
                lmax = ells_here[-1]

            if not ignore_sc_likelihood:
                tp_filters.append(
                    create_twopoint_filter(
                        key, tr1, tr2,
                        cut_low=ells_here[0],
                        cut_high=lmax,
                    )
                )

            if ignore_sc:
                ells_here = ells

            # Bandpower windows
            if bandpower == 'TopHat':
                ells_aux = np.arange(0, np.max(ells_here) + 1)
                wgt = np.zeros((len(ells_aux), len(ells_here)))
                for i in range(len(ells_here)):
                    in_win = (ells_aux > ell_edges[i]) & (ells_aux < ell_edges[i + 1])
                    wgt[in_win, i] = 1.0
                win = sacc.BandpowerWindow(ells_aux, wgt)
                S.add_ell_cl(key, tr1, tr2,
                             ells_here, np.zeros(len(ells_here)), window=win)
            elif bandpower == 'NaMaster':
                raise NotImplementedError(
                    "NaMaster bandpower windows not yet implemented in Augur generation."
                )
            else:
                S.add_ell_cl(key, tr1, tr2,
                             ells_here, np.zeros(len(ells_here)))

            stats.append(
                TwoPoint(source0=sources[tr1], source1=sources[tr2],
                         sacc_data_type=key)
            )

    return stats, tp_filters
