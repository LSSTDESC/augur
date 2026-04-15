"""Covariance-matrix computation for Augur SACC data vectors."""

import numpy as np
import warnings

from augur.utils.cov_utils import get_gaus_cov, get_SRD_cov, get_noise_power
from augur.utils.cov_utils import TJPCovGaus, TJPCovClusterGaus, TJPCovClusterSSC
from augur.utils.config_io import parse_array


def compute_covariance(config, S, lk, cosmo, tools, probes=None):
    """
    Compute and attach a covariance matrix to a SACC object.

    Supported covariance types (via ``config['cov_options']['cov_type']``):
      - ``'gaus_internal'`` — analytic Gaussian covariance.
      - ``'SRD'`` — pre-computed SRD covariance.
      - ``'tjpcov'`` — TJPCov Fourier-space Gaussian fsky covariance.

    For probes that TJPCov does not support (real-space correlations),
    the covariance block is filled with an identity matrix and a warning
    is emitted.

    TJPCov Supported Probes:
      - ``'harmonic'`` — galaxy weak lensing (WL), clustering (GCl).
      - ``'cmb_lensing'`` — CMB convergence auto/cross-correlations.
      - ``'cluster_counts'`` — cluster number counts (Gaussian + SSC via ClusterCountsGaussian
        and ClusterCountsSSC; shot noise computed automatically from HMF).

    Parameters
    ----------
    config : dict
        Full Augur configuration dictionary.
    S : sacc.Sacc
        SACC object (data vector must already be filled).
    lk : firecrown likelihood
        The likelihood object (needed for internal-Gaussian and TJPCov paths).
    cosmo : pyccl.Cosmology
        Fiducial cosmology.
    tools : firecrown.modeling_tools.ModelingTools
        Modelling tools (needed for TJPCov path).
    probes : list[str], optional
        Which probes are present (e.g. ``['harmonic', 'real_space',
        'cmb_lensing', 'cluster_counts']``).  Used to decide whether
        TJPCov can handle the full data vector.  If *None*, all probes
        in the SACC are assumed to be harmonic-space.

    Returns
    -------
    None — the covariance is added to *S* in-place.
    """
    cov_type = config['cov_options']['cov_type']
    cov_type = cov_type.lower()

    # ---- check whether TJPCov can handle the requested probes ---------- #
    # unsupported_probes = []
    # if probes is not None:
    #     tjpcov_capable = {'harmonic', 'cmb_lensing', 'cluster_counts'}
    #     unsupported_probes = [p for p in probes if p not in tjpcov_capable]

    # ---- Probe support check for gaus_internal / SRD ---------- #
    _gaus_supported = {'harmonic', 'cmb_lensing'}
    _srd_supported = {'harmonic'}  # SRD matrices are pre-computed for WL+GCl only
    _active = set(probes) if probes else {'harmonic'}

    if cov_type == 'gaus_internal':
        unsupported = _active - _gaus_supported
        if unsupported:
            warnings.warn(
                f"gaus_internal does not support probes {unsupported}. "
                "Their covariance blocks will receive a placeholder "
                "(variance = 1e30) so the matrix stays invertible."
            )
        fsky = config['cov_options']['fsky']
        cov = get_gaus_cov(S, lk, cosmo, fsky, config)
        S.add_covariance(cov)

    elif cov_type == 'srd':
        unsupported = _active - _srd_supported
        if unsupported:
            warnings.warn(
                f"SRD covariance does not support probes {unsupported}. "
                "Only harmonic two-point blocks are covered; other blocks "
                "will receive a placeholder (variance = 1e30)."
            )
        cov = get_SRD_cov(config['cov_options'], S)
        S.add_covariance(cov)

    elif cov_type == 'tjpcov':
        # ---- Separate probes by type: Fourier (harmonic+CMB) vs. Cluster ---- #
        has_fourier = 'harmonic' in (probes or ['harmonic'])
        has_cmb = 'cmb_lensing' in (probes or [])
        has_clusters = 'cluster_counts' in (probes or [])

        fourier_probes = ['harmonic', 'cmb_lensing']
        cluster_probes = ['cluster_counts']
        unsupported = [p for p in (probes or []) if p not in fourier_probes + cluster_probes]

        if unsupported:
            warnings.warn(
                f"TJPCov does not support the following probes: {unsupported}. "
                "Their covariance blocks will be filled with identity matrices."
            )

        # Initialize full covariance matrix
        ndata = len(S.mean)
        cov_all = np.zeros((ndata, ndata))

        # ---- Compute Fourier (Harmonic + CMB Lensing) Covariance ---- #
        if has_fourier or has_cmb:
            tjpcov_ell_edges = parse_array(
                config['cov_options']['binning_info']['ell_edges']
            )

            # TJPCov FourierGaussianFsky uses a single global ell binning.
            # Enforce equality with every harmonic-space block in the data vector.
            all_harmonic_stats = {}
            all_harmonic_stats.update(config.get('statistics', {}))
            cmb_stats = config.get('cmb_lensing', {}).get('statistics', {})
            for stat_name, stat_info in cmb_stats.items():
                if '_cl' in stat_name:
                    all_harmonic_stats[stat_name] = stat_info

            for stat_name, stat_info in all_harmonic_stats.items():
                dv_ell_edges = parse_array(stat_info['ell_edges'])
                if (
                    tjpcov_ell_edges.shape != dv_ell_edges.shape
                    or not np.allclose(tjpcov_ell_edges, dv_ell_edges, rtol=0.0, atol=0.0)
                ):
                    raise ValueError(
                        "TJPCov ell_edges do not match the data-vector ell_edges "
                        f"for statistic `{stat_name}`. "
                        "Please set `cov_options.binning_info.ell_edges` to match "
                        "all harmonic-space blocks (including cmb_lensing statistics)."
                    )

            tjpcov_config = dict()
            tjpcov_config['tjpcov'] = dict()
            tjpcov_config['tjpcov']['cosmo'] = tools.ccl_cosmo
            ccl_tracers = dict()
            bias_all = dict()
            for i, myst1 in enumerate(lk.statistics):
                trname1 = myst1.statistic.source0.sacc_tracer
                trname2 = myst1.statistic.source1.sacc_tracer
                tr1 = myst1.statistic.source0.tracers[0].ccl_tracer
                tr2 = myst1.statistic.source1.tracers[0].ccl_tracer
                ccl_tracers[trname1] = tr1
                ccl_tracers[trname2] = tr2
                if 'lens' in trname1:
                    bias_all[trname1] = myst1.statistic.source0.bias
                if 'lens' in trname2:
                    bias_all[trname2] = myst1.statistic.source1.bias
            for key in bias_all:
                tjpcov_config['tjpcov'][f'bias_{key}'] = bias_all[key]
            tjpcov_config['tjpcov']['sacc_file'] = S
            tjpcov_config['tjpcov']['IA'] = config['cov_options'].get('IA', None)
            tjpcov_config['tjpcov']['cov_type'] = ['FourierGaussianFsky']
            tjpcov_config['tjpcov']['fsky'] = config['cov_options']['fsky']
            tjpcov_config['tjpcov']['binning_info'] = dict()
            tjpcov_config['tjpcov']['binning_info']['ell_edges'] = tjpcov_ell_edges
            for tr in S.tracers:
                if ('src' not in tr) and ('lens' not in tr):
                    continue
                _, ndens = get_noise_power(config, S, tr, return_ndens=True)
                tjpcov_config['tjpcov'][f'Ngal_{tr}'] = ndens
                if 'src' in tr:
                    tjpcov_config['tjpcov'][f'sigma_e_{tr}'] = \
                        config['sources']['ellipticity_error']

            # Optional CMB lensing noise (default 0.0) for Fourier covariance.
            cmb_noise = config.get('cmb_lensing', {}).get('noise_cl', 0.0)
            tjpcov_config['tjpcov']['cmb_noise'] = float(cmb_noise)

            cov_calc = TJPCovGaus(tjpcov_config)

            if config['general']['ignore_scale_cuts']:
                cov_fourier = cov_calc.get_covariance()
            else:
                cov_fourier = np.zeros((ndata, ndata))
                for i, trcombs1 in enumerate(S.get_tracer_combinations()):
                    ii = S.indices(tracers=trcombs1)
                    for trcombs2 in S.get_tracer_combinations()[i:]:
                        jj = S.indices(tracers=trcombs2)
                        ii_all, jj_all = np.meshgrid(ii, jj, indexing='ij')
                        cov_here = cov_calc.get_covariance_block(
                            trcombs1, trcombs2, include_b_modes=False
                        )
                        cov_fourier[ii_all, jj_all] = cov_here[:len(ii), :len(jj)]
                        cov_fourier[jj_all.T, ii_all.T] = cov_here[:len(ii), :len(jj)].T

            cov_all += cov_fourier

        # ---- Compute Cluster Number Counts Covariance ---- #
        if has_clusters:
            # Check that SACC has cluster count data
            if 'cluster_counts' not in S.get_data_types():
                raise ValueError(
                    "Cluster count covariance requested but no cluster_counts "
                    "data type found in SACC file."
                )

            cluster_tjpcov_config = dict()
            cluster_tjpcov_config['tjpcov'] = {
                'sacc_file': S,
                'cosmo': tools.ccl_cosmo,
                'fsky': config['cov_options']['fsky'],
            }

            # Optional cluster-specific TJPCov settings
            cluster_opts = config.get('cluster_counts', {}).get('tjpcov_options', {})
            cluster_tjpcov_config['tjpcov'].update(cluster_opts)

            # Compute Gaussian (shot noise) covariance for clusters.
            # TJPCov's get_covariance() returns the full (n_data x n_data) matrix
            # with zeros outside the cluster_counts data type blocks.
            cov_gauss = TJPCovClusterGaus(cluster_tjpcov_config).get_covariance()

            # Compute SSC (super-sample covariance) for clusters.
            # Also returns full matrix with zeros outside cluster blocks.
            cov_ssc = TJPCovClusterSSC(cluster_tjpcov_config).get_covariance()

            # Combine: TJPCov ensures that each builder only populates its own
            # data-type blocks, so addition is safe and correct for multi-probe.
            cov_cluster = cov_gauss + cov_ssc

            # Add cluster contribution to full covariance.
            # No indexing issues: each builder filters to its data types automatically.
            cov_all += cov_cluster

        S.add_covariance(cov_all)

    elif cov_type == 'none':
        ndata = len(S.mean)
        warnings.warn(
             "User specified no covariance to be attached to the SACC file. "
             "Using identity matrix as covariance. "
             "This is not recommended for realistic analyses, "
             "but may be useful for testing or debugging."
        )
        S.add_covariance(np.eye(ndata))

    else:
        warnings.warn(
            f"Covariance type '{cov_type}' is not recognized. "
            "Currently only 'gaus_internal', 'SRD', and 'tjpcov' are "
            "implemented. Using identity matrix as covariance."
        )
        ndata = len(S.mean)
        S.add_covariance(np.eye(ndata))
