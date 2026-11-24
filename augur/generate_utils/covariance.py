"""Covariance-matrix computation for Augur SACC data vectors."""

import numpy as np
import warnings

from firecrown.likelihood.two_point import TwoPoint
from augur.utils.cov_utils import get_gaus_cov, get_SRD_cov, get_noise_power
from augur.utils.cov_utils import TJPCovGaus, get_cluster_ssc
from augur.utils.config_io import parse_array
from augur.generate_utils.cluster_counts import (
    CLUSTER_DATA_TYPES, CLUSTER_TRACER_TYPES, cluster_statistics,
)

# TJPCov calculator shared with forked covariance workers. Set before the pool
# is created so children inherit it instead of pickling it per task.
_POOL_CALC = None


def _covariance_block(tracer_comb1, tracer_comb2):
    """Worker entry point for the covariance pool.

    Defined at module level so ``multiprocessing`` can pickle it by reference.
    """
    return _POOL_CALC.get_covariance_block(
        tracer_comb1, tracer_comb2, include_b_modes=False
    )


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

    Cluster counts and mean log mass (probe ``'cluster_counts'``) get their
    own block, whatever the two-point covariance type, with zero cross-terms
    to the two-point data (block-diagonal); see :func:`cluster_covariance`.

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
    _gaus_supported = {'harmonic', 'cmb_lensing', 'cluster_counts'}
    _srd_supported = {'harmonic', 'cluster_counts'}  # SRD: WL+GCl only
    _active = set(probes) if probes else {'harmonic'}
    has_clusters = 'cluster_counts' in _active

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

    elif cov_type == 'srd':
        unsupported = _active - _srd_supported
        if unsupported:
            warnings.warn(
                f"SRD covariance does not support probes {unsupported}. "
                "Only harmonic two-point blocks are covered; other blocks "
                "will receive a placeholder (variance = 1e30)."
            )
        cov = get_SRD_cov(config['cov_options'], S)

    elif cov_type == 'tjpcov':
        # ---- Separate probes by type: Fourier (harmonic+CMB) vs. Cluster ---- #
        has_fourier = 'harmonic' in (probes or ['harmonic'])
        has_cmb = 'cmb_lensing' in (probes or [])

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
            for myst1 in lk.statistics:
                myst1 = myst1.statistic
                if not isinstance(myst1, TwoPoint):
                    continue
                trname1 = myst1.source0.sacc_tracer
                trname2 = myst1.source1.sacc_tracer
                ccl_tracers[trname1] = myst1.source0.tracers[0].ccl_tracer
                ccl_tracers[trname2] = myst1.source1.tracers[0].ccl_tracer
                if 'lens' in trname1:
                    bias_all[trname1] = myst1.source0.bias
                if 'lens' in trname2:
                    bias_all[trname2] = myst1.source1.bias
            for key in bias_all:
                tjpcov_config['tjpcov'][f'bias_{key}'] = bias_all[key]
            # TJPCov's Fourier builder only understands two-point tracers.
            tjpcov_config['tjpcov']['sacc_file'] = _two_point_sacc(S) if has_clusters else S
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

            # if config['general']['ignore_scale_cuts']:
            #     cov_fourier = cov_calc.get_covariance()
            # else:
            #     cov_fourier = np.zeros((ndata, ndata))
            #     for i, trcombs1 in enumerate(S.get_tracer_combinations()):
            #         ii = S.indices(tracers=trcombs1)
            #         for trcombs2 in S.get_tracer_combinations()[i:]:
            #             jj = S.indices(tracers=trcombs2)
            #             ii_all, jj_all = np.meshgrid(ii, jj, indexing='ij')
            #             cov_here = cov_calc.get_covariance_block(
            #                 trcombs1, trcombs2, include_b_modes=False
            #             )
            #             cov_fourier[ii_all, jj_all] = cov_here[:len(ii), :len(jj)]
            #             cov_fourier[jj_all.T, ii_all.T] = cov_here[:len(ii), :len(jj)].T

            nproc = int(config['cov_options'].get('nproc', 1))
            combs = [c for c in S.get_tracer_combinations() if len(c) == 2]
            pairs = [(combs[i], combs[j])
                     for i in range(len(combs)) for j in range(i, len(combs))]
            global _POOL_CALC
            _POOL_CALC = cov_calc
            if nproc == 1:
                blocks = [_covariance_block(*p) for p in pairs]
            else:
                # limits=1 is required, not an optimisation: forking a process
                # with live BLAS threads deadlocks.
                import multiprocessing as mp
                from threadpoolctl import threadpool_limits
                with threadpool_limits(limits=1), \
                        mp.get_context('fork').Pool(nproc) as pool:
                    blocks = pool.starmap(_covariance_block, pairs)

            cov_fourier = np.zeros((ndata, ndata))
            # scale-cuts taken into account here
            for (t1, t2), blk in zip(pairs, blocks):
                ii, jj = S.indices(tracers=t1), S.indices(tracers=t2)
                ii_all, jj_all = np.meshgrid(ii, jj, indexing='ij')
                cov_fourier[ii_all, jj_all] = blk[:len(ii), :len(jj)]
                cov_fourier[jj_all.T, ii_all.T] = blk[:len(ii), :len(jj)].T

            cov_all += cov_fourier

        cov = cov_all

    elif cov_type == 'none':
        ndata = len(S.mean)
        warnings.warn(
             "User specified no covariance to be attached to the SACC file. "
             "Using identity matrix as covariance. "
             "This is not recommended for realistic analyses, "
             "but may be useful for testing or debugging."
        )
        S.add_covariance(np.eye(ndata))
        return

    else:
        warnings.warn(
            f"Covariance type '{cov_type}' is not recognized. "
            "Currently only 'gaus_internal', 'SRD', and 'tjpcov' are "
            "implemented. Using identity matrix as covariance."
        )
        ndata = len(S.mean)
        S.add_covariance(np.eye(ndata))
        return

    if has_clusters:
        idx, block = cluster_covariance(config, S, lk, tools.ccl_cosmo)
        cov[np.ix_(idx, idx)] = block
    S.add_covariance(cov)


def _two_point_sacc(S):
    """Return a copy of *S* without the cluster data points and tracers."""
    S2 = S.copy()
    for data_type in CLUSTER_DATA_TYPES.values():
        S2.remove_selection(data_type=data_type)
    S2.remove_tracers([n for n, t in S2.tracers.items()
                       if t.tracer_type in CLUSTER_TRACER_TYPES])
    return S2


def cluster_covariance(config, S, lk, cosmo):
    """Covariance block of the cluster counts and mean log10 mass in *S*.

    * counts: Poisson variance equal to crow's fiducial counts in *S*, plus the
      TJPCov super-sample term rescaled to those counts (``covariance.ssc``,
      default on). Both follow CLPipe (LSSTDESC/CLPipe,
      ``clpipe/clp_covariance.py``, ``CLPCovariance.replace_crow_counts``,
      commit 4a6128d); see :func:`augur.utils.cov_utils.get_cluster_ssc`;
    * mean log10 mass: ``mean_log_mass_sigma**2 / N`` per bin, with N the
      fiducial count of that bin; no cross-terms with the counts.

    Returns the SACC indices of the cluster points and their covariance.
    """
    cov_cfg = config['cluster_counts'].get('covariance', {})
    i_n = S.indices(data_type=CLUSTER_DATA_TYPES['counts'])
    i_m = S.indices(data_type=CLUSTER_DATA_TYPES['mass'])
    counts = S.mean[i_n]
    if np.any(counts <= 0):
        raise ValueError("Some cluster bins have no expected clusters; their "
                         "covariance is singular. Change the binning or the model.")
    # Poisson term from crow's counts rather than TJPCov's own (CLPipe convention):
    # crow includes the completeness and purity models, TJPCov does not.
    block_n = np.diag(counts)
    if cov_cfg.get('ssc', True):
        (st,) = cluster_statistics(lk)
        block_n = block_n + get_cluster_ssc(S, st.cluster_recipe, cosmo, counts,
                                            cov_cfg.get('halo_bias', 'Tinker10'))
    idx = np.concatenate([i_n, i_m])
    block = np.zeros((len(idx), len(idx)))
    block[:len(i_n), :len(i_n)] = block_n
    if len(i_m):
        if 'mean_log_mass_sigma' not in cov_cfg:
            raise ValueError("cluster_counts.covariance.mean_log_mass_sigma (dex) is "
                             "needed for the mean log mass covariance.")
        n_of_bin = {S.data[i].tracers: S.data[i].value for i in i_n}
        var_m = [cov_cfg['mean_log_mass_sigma']**2 / n_of_bin[S.data[i].tracers] for i in i_m]
        block[len(i_n):, len(i_n):] = np.diag(var_m)
    return idx, block
