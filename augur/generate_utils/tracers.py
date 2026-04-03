"""Tracer / N(z) setup and naming helpers."""

import numpy as np
import sacc

from augur.tracers.two_point import ZDist, LensSRD2018, SourceSRD2018, ZDistFromFile

import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc

implemented_nzs = [ZDist, LensSRD2018, SourceSRD2018, ZDistFromFile]


def get_tracers(statistic, comb):
    """
    Map a statistic name and bin combination to SACC tracer names.

    Parameters
    ----------
    statistic : str
        SACC data-type string (e.g. ``'galaxy_density_cl'``).
    comb : tuple
        Pair of bin indices.

    Returns
    -------
    tr1, tr2 : str
        SACC tracer names (e.g. ``'lens0'``, ``'src1'``).
    """
    if 'galaxy_density' in statistic:
        tr1 = f'lens{comb[0]}'
        tr2 = f'lens{comb[1]}'
    elif 'galaxy_shear' in statistic and 'Density' not in statistic:
        tr1 = f'src{comb[0]}'
        tr2 = f'src{comb[1]}'
    elif 'galaxy_shearDensity' in statistic:
        tr1 = f'lens{comb[0]}'
        tr2 = f'src{comb[1]}'
    else:
        raise NotImplementedError(
            f'Tracer mapping not implemented for statistic: {statistic}'
        )
    return tr1, tr2


def add_nz(cfg, nbins, src_root, S, dndz):
    """
    Add N(z) distributions for a set of tomographic bins to a SACC object.

    Parameters
    ----------
    cfg : dict
        Tracer-level config (``sources`` or ``lenses``).
    nbins : int
        Number of tomographic bins.
    src_root : str
        Root name for the SACC tracer (``'src'`` or ``'lens'``).
    S : sacc.Sacc
        SACC object to populate.
    dndz : dict
        Dictionary collecting the N(z) objects (modified in-place).

    Returns
    -------
    dndz : dict
        Updated dictionary of N(z) objects.
    """
    z = np.linspace(0.004004004004004004,
                    4.004004004004004004, 1000)

    Nz_centers = None
    if 'Nz_center' in cfg['Nz_kwargs'].keys():
        Nz_centers = eval(cfg['Nz_kwargs']['Nz_center'])
        cfg['Nz_kwargs'].pop('Nz_center')

        if np.isscalar(Nz_centers):
            Nz_centers = [Nz_centers]
            if nbins != 1:
                raise ValueError('Nz_centers should have the same length as the number of bins')
        else:
            if len(Nz_centers) != nbins:
                raise ValueError('Nz_centers should have the same length as the number of bins')

    for i in range(nbins):
        sacc_tracer = f'{src_root}{i}'
        if isinstance(cfg['Nz_type'], list):
            if eval(cfg['Nz_type'][i]) in implemented_nzs:
                if 'ZDistFromFile' not in cfg['Nz_type'][i]:
                    if Nz_centers is not None:
                        dndz[sacc_tracer] = eval(cfg['Nz_type'][i])(
                            z, Nz_center=Nz_centers[i], Nz_nbins=nbins,
                            **cfg['Nz_kwargs'])
                    else:
                        dndz[sacc_tracer] = eval(cfg['Nz_type'][i])(
                            z, Nz_ibin=i, Nz_nbins=nbins,
                            **cfg['Nz_kwargs'])
                else:
                    dndz[sacc_tracer] = ZDistFromFile(**cfg['Nz_kwargs'], ibin=i)
            else:
                raise NotImplementedError('The selected N(z) is yet not implemented')
        else:
            if eval(cfg['Nz_type']) in implemented_nzs:
                if 'ZDistFromFile' not in cfg['Nz_type']:
                    if Nz_centers is not None:
                        dndz[sacc_tracer] = eval(cfg['Nz_type'])(
                            z, Nz_center=Nz_centers[i], Nz_nbins=nbins,
                            **cfg['Nz_kwargs'])
                    else:
                        dndz[sacc_tracer] = eval(cfg['Nz_type'])(
                            z, Nz_ibin=i, Nz_nbins=nbins,
                            **cfg['Nz_kwargs'])
                else:
                    dndz[sacc_tracer] = ZDistFromFile(**cfg['Nz_kwargs'], ibin=i)
            else:
                raise NotImplementedError('The selected N(z) is yet not implemented')
        S.add_tracer('NZ', sacc_tracer, dndz[sacc_tracer].z, dndz[sacc_tracer].Nz)
    return dndz


def setup_sources(config, S):
    """
    Read the ``sources`` section of the config and register weak-lensing
    tracers in the SACC object.

    Parameters
    ----------
    config : dict
        Full Augur config dictionary.
    S : sacc.Sacc
        SACC object to populate.

    Returns
    -------
    sources : dict
        Mapping of tracer names to firecrown WeakLensing objects.
    dndz : dict
        Mapping of tracer names to N(z) objects.
    """
    sources = {}
    dndz = {}
    if 'sources' in config:
        src_cfg = config['sources']
        nbins = src_cfg['nbins']
        src_root = 'src'
        dndz = add_nz(src_cfg, nbins, src_root, S, dndz)
        for i in range(nbins):
            sacc_tracer = f'{src_root}{i}'
            sources[sacc_tracer] = wl.WeakLensing(sacc_tracer=sacc_tracer)
    return sources, dndz


def setup_lenses(config, S, sources, dndz):
    """
    Read the ``lenses`` section of the config and register number-count
    tracers in the SACC object.

    Parameters
    ----------
    config : dict
        Full Augur config dictionary.
    S : sacc.Sacc
        SACC object to populate.
    sources : dict
        Existing source dict to update in-place.
    dndz : dict
        Existing N(z) dict to update in-place.

    Returns
    -------
    sources : dict
        Updated mapping including lens tracers.
    dndz : dict
        Updated N(z) mapping.
    """
    if 'lenses' in config:
        lns_cfg = config['lenses']
        nbins = lns_cfg['nbins']
        lns_root = 'lens'
        dndz = add_nz(lns_cfg, nbins, lns_root, S, dndz)
        for i in range(nbins):
            sacc_tracer = f'{lns_root}{i}'
            sources[sacc_tracer] = nc.NumberCounts(
                sacc_tracer=sacc_tracer, derived_scale=True
            )
    return sources, dndz
