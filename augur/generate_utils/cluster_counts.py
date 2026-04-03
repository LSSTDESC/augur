"""Cluster-count statistics for SACC generation.

This module provides the stub infrastructure for adding cluster-count
observables (number counts in mass/redshift bins) to a SACC data vector.

.. warning::

    Cluster counts are **not yet implemented** in Augur's generation
    pipeline.  Calling :func:`add_cluster_counts` will register
    placeholders in the SACC object but will issue a warning that
    firecrown likelihood objects are not available.  Firecrown
    factories for cluster counts with CROW are not yet available.
"""

import numpy as np
import sacc
import warnings


def add_cluster_counts(config, S, sources, cosmo):
    """
    Populate a SACC object with cluster-count data points.

    Currently a **stub**: it reads the ``cluster_counts`` block of the
    config, adds placeholder data-points to the SACC, and warns that no
    firecrown likelihood objects are returned.

    Parameters
    ----------
    config : dict
        Full Augur config dict (should contain ``cluster_counts``).
    S : sacc.Sacc
        SACC object to populate.
    sources : dict
        Tracer-name → firecrown source mapping (updated in-place if
        cluster tracers are added).
    cosmo : pyccl.Cosmology
        Fiducial cosmology (needed for mass-function evaluations once
        implemented).

    Returns
    -------
    stats : list
        Empty — no firecrown statistics are built yet.
    tp_filters : list
        Empty — no filters are built yet.
    """
    cc_cfg = config.get('cluster_counts', None)
    if cc_cfg is None:
        return [], []

    warnings.warn(
        "Cluster count generation is not yet implemented in Augur. "
        "Placeholder tracers will be added to the SACC, but no firecrown "
        "likelihood objects are created. "
        "Firecrown factories for cluster counts with CROW are not yet available."
    )

    # Expected config structure (for future implementation):
    #   cluster_counts:
    #     z_edges: [0.2, 0.4, 0.6, 0.8, 1.0]
    #     mass_proxy_edges: [13.0, 13.5, 14.0, 14.5, 15.0]  # log10(M/Msun)
    #     survey_area: 18000  # deg^2

    z_edges = np.array(cc_cfg.get('z_edges', [0.2, 0.5, 0.8, 1.1]))
    mass_edges = np.array(cc_cfg.get('mass_proxy_edges', [14.0, 14.5, 15.0]))
    n_z_bins = len(z_edges) - 1
    n_m_bins = len(mass_edges) - 1

    # Register a cluster tracer per (z, M) bin
    for iz in range(n_z_bins):
        for im in range(n_m_bins):
            tracer_name = f'cluster_z{iz}_m{im}'
            S.add_tracer('Map', tracer_name,
                         quantity='generic', spin=0,
                         ell=None, beam=None)

    return [], []
