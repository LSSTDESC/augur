"""Cluster-count statistics for SACC generation.

This module provides the stub infrastructure for adding cluster-count
observables (number counts in mass/redshift bins) to a SACC data vector.

TJPCov Covariance Support:
==========================
TJPCov (via ClusterCountsGaussian, ClusterCountsSSC) supports cluster number
count covariance including shot noise, Gaussian contributions, and super-sample
covariance (SSC). To enable TJPCov cluster covariance, the SACC file must have:

1. **Redshift bin tracers**: bin_z_0, bin_z_1, ... with (z_min, z_max) edges
2. **Richness/Lambda bin tracers**: bin_richness_0, ... with (log10(λ_min), log10(λ_max))
3. **Survey metadata**: survey tracer with total area in steradians
4. **Data type**: sacc.standard_types.cluster_counts
5. **Data points**: cluster counts in each (z_bin, richness_bin) combination

See :func:`setup_cluster_sacc_tracers_for_tjpcov` for tracer setup.

.. warning::

    Cluster counts datavector generation is **not yet implemented** in Augur.
    Calling :func:`add_cluster_counts` will register placeholder tracers and
    issue a warning. Firecrown factories for cluster counts with CROW are
    not yet available. However, the covariance layer is prepared to handle
    cluster counts if the SACC structure is set up correctly (see above).
"""

import numpy as np
import sacc

from augur.utils.config_io import parse_array
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


def setup_cluster_sacc_tracers_for_tjpcov(S, config):
    """
    Set up cluster bin tracers in SACC for TJPCov covariance compatibility.

    This creates the tracer structure that TJPCov's ClusterCountsGaussian and
    ClusterCountsSSC classes expect when calling load_from_sacc():
      - bin_z tracers with redshift bin edges
      - bin_richness tracers with log10(richness) bin edges
      - survey tracer with total area in steradians

    When present, the covariance layer (:func:`augur.generate_utils.covariance.compute_covariance`)
    passes these tracers to TJPCov, which automatically computes shot noise and SSC
    from the Halo Mass Function and mass-richness relation.

    Parameters
    ----------
    S : sacc.Sacc
        SACC object to populate with cluster tracers (modified in-place).
    config : dict
        Augur configuration dict with ``cluster_counts`` section.

    Returns
    -------
    None
        Tracers are added directly to *S*.

    Config Structure (cluster_counts section):
    -------------------------------------------
    cluster_counts:
      z_edges: [0.3, 0.5, 0.7, 1.2]           # Cluster redshift bin edges
      richness_log_edges: [1.0, 1.48, 1.90, 2.30]  # log10(richness) edges
      survey_area: 4000                        # Survey area in deg^2

      tjpcov_options:                          # Optional TJPCov settings
        min_halo_mass: 1e13                    # Minimum halo mass (M_sun)
        has_mproxy: true                       # Include mass-richness scatter

    Notes
    -----
    This function assumes the config keys are present. If any are missing,
    defaults are used from TJPCov's internal logic.
    """
    cluster_cfg = config.get('cluster_counts', {})

    if not cluster_cfg:
        return

    # Parse redshift and richness bin edges
    z_edges = parse_array(cluster_cfg.get('z_edges', None))
    richness_log_edges = parse_array(
        cluster_cfg.get('richness_log_edges', None)
    )
    if z_edges is None or richness_log_edges is None:
        raise ValueError(
            "Cluster counts require 'z_edges' and 'richness_log_edges' in config."
        )

    # Add bin_z tracers (redshift bins)
    for i in range(len(z_edges) - 1):
        tracer_name = f"bin_z_{i}"
        z_min, z_max = float(z_edges[i]), float(z_edges[i + 1])
        # SACC tracer type 'bin_z' with (lower, upper) redshift bounds
        S.add_tracer("bin_z", tracer_name, z_min, z_max)

    # Add bin_richness tracers (richness/lambda bins in log10 scale)
    for i in range(len(richness_log_edges) - 1):
        tracer_name = f"bin_richness_{i}"
        log_lam_min = float(richness_log_edges[i])
        log_lam_max = float(richness_log_edges[i + 1])
        # SACC tracer type 'bin_richness' with (log10_lambda_min, log10_lambda_max)
        S.add_tracer("bin_richness", tracer_name, log_lam_min, log_lam_max)

    # Add survey metadata tracer
    survey_area_deg2 = float(cluster_cfg.get('survey_area', 4000.0))
    survey_area_sr = survey_area_deg2 * (np.pi / 180.0) ** 2  # Convert deg^2 to steradians
    S.add_tracer("survey", "mock_survey", survey_area_sr)
