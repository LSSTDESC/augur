"""CMB lensing statistics for SACC generation.

This module provides the stub infrastructure for adding CMB-lensing
auto- and cross-spectra to a SACC data vector.

.. warning::

    CMB lensing is **not yet implemented** in Augur or Firecrown.
    Calling :func:`add_cmb_lensing` will register placeholders in the
    SACC object but will issue a warning that firecrown likelihood
    objects are not available.
"""

import numpy as np
import sacc
import warnings


def add_cmb_lensing(config, S, sources, dndz, cosmo):
    """
    Populate a SACC object with CMB-lensing data points.

    Currently a **stub**: it reads the ``cmb_lensing`` block of the config,
    adds placeholder tracers / data-points to the SACC, and warns that no
    firecrown likelihood objects are returned.

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
        Empty — no firecrown statistics are built yet.
    tp_filters : list
        Empty — no scale-cut filters are built yet.
    """
    cmb_cfg = config.get('cmb_lensing', None)
    if cmb_cfg is None:
        return [], []

    warnings.warn(
        "CMB lensing generation is not yet implemented in Augur. "
        "A placeholder CMB-lensing tracer will be added to the SACC, "
        "but no firecrown likelihood objects are created. "
        "Firecrown factories for CMB lensing with CROW are not yet available."
    )

    # Register a placeholder CMB-lensing tracer
    # SACC expects a 'Map' tracer for CMB kappa.
    z_source = cmb_cfg.get('z_source', 1100.0)
    S.add_tracer('Map', 'cmb_convergence', quantity='cmb_convergence',
                 spin=0, ell=None, beam=None)

    # If cross-correlations with galaxy bins are specified, add ell-binning
    stat_keys = cmb_cfg.get('statistics', {})
    for key in stat_keys:
        tracer_combs = stat_keys[key].get('tracer_combs', [])
        ell_edges = np.array(eval(stat_keys[key]['ell_edges']))
        ells = np.sqrt(ell_edges[:-1] * ell_edges[1:])

        for comb in tracer_combs:
            # Convention: comb = (galaxy_bin_index,) for cross,
            #             comb = () for auto (kappa-kappa).
            if len(comb) == 0:
                tr1, tr2 = 'cmb_convergence', 'cmb_convergence'
            else:
                # Cross with a galaxy tracer
                tr1 = 'cmb_convergence'
                tr2 = f'src{comb[0]}' if 'shear' in key else f'lens{comb[0]}'
            S.add_ell_cl(key, tr1, tr2, ells, np.zeros(len(ells)))

    return [], []
