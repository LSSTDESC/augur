"""Covariance-matrix computation for Augur SACC data vectors."""

import numpy as np
import warnings

from augur.utils.cov_utils import get_gaus_cov, get_SRD_cov, get_noise_power
from augur.utils.cov_utils import TJPCovGaus


def compute_covariance(config, S, lk, cosmo, tools, probes=None):
    """
    Compute and attach a covariance matrix to a SACC object.

    Supported covariance types (via ``config['cov_options']['cov_type']``):
      - ``'gaus_internal'`` — analytic Gaussian covariance.
      - ``'SRD'`` — pre-computed SRD covariance.
      - ``'tjpcov'`` — TJPCov Fourier-space Gaussian fsky covariance.

    For probes that TJPCov does not support (CMB lensing, cluster counts,
    real-space correlations), the covariance block is filled with an identity
    matrix and a warning is emitted.

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
    unsupported_probes = []
    if probes is not None:
        tjpcov_capable = {'harmonic'}
        unsupported_probes = [p for p in probes if p not in tjpcov_capable]

    if cov_type == 'gaus_internal':
        fsky = config['cov_options']['fsky']
        cov = get_gaus_cov(S, lk, cosmo, fsky, config)
        S.add_covariance(cov)

    elif cov_type == 'srd':
        cov = get_SRD_cov(config['cov_options'], S)
        S.add_covariance(cov)

    elif cov_type == 'tjpcov':
        if unsupported_probes:
            warnings.warn(
                f"TJPCov does not currently support the following probes: "
                f"{unsupported_probes}.  Their covariance blocks will be "
                f"filled with an identity matrix."
            )

        tjpcov_ell_edges = np.asarray(
            eval(config['cov_options']['binning_info']['ell_edges'])
        )
        for stat_name, stat_info in config['statistics'].items():
            dv_ell_edges = np.asarray(eval(stat_info['ell_edges']))
            if (
                tjpcov_ell_edges.shape != dv_ell_edges.shape
                or not np.allclose(tjpcov_ell_edges, dv_ell_edges, rtol=0.0, atol=0.0)
            ):
                raise ValueError(
                    "TJPCov ell_edges do not match the data-vector ell_edges "
                    f"for statistic `{stat_name}`. "
                    "Please set `cov_options.binning_info.ell_edges` to match "
                    "`statistics.<stat>.ell_edges`."
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
            _, ndens = get_noise_power(config, S, tr, return_ndens=True)
            tjpcov_config['tjpcov'][f'Ngal_{tr}'] = ndens
            if 'src' in tr:
                tjpcov_config['tjpcov'][f'sigma_e_{tr}'] = config['sources']['ellipticity_error']

        cov_calc = TJPCovGaus(tjpcov_config)
        if config['general']['ignore_scale_cuts']:
            cov_all = cov_calc.get_covariance()
        else:
            ndata = len(S.mean)
            cov_all = np.zeros((ndata, ndata))
            for i, trcombs1 in enumerate(S.get_tracer_combinations()):
                ii = S.indices(tracers=trcombs1)
                for trcombs2 in S.get_tracer_combinations()[i:]:
                    jj = S.indices(tracers=trcombs2)
                    ii_all, jj_all = np.meshgrid(ii, jj, indexing='ij')
                    cov_here = cov_calc.get_covariance_block(
                        trcombs1, trcombs2, include_b_modes=False
                    )
                    cov_all[ii_all, jj_all] = cov_here[:len(ii), :len(jj)]
                    cov_all[jj_all.T, ii_all.T] = cov_here[:len(ii), :len(jj)].T
        S.add_covariance(cov_all)

    elif cov_type == 'none':
        ndata = len(S.mean)
        S.add_covariance(np.eye(ndata))

    else:
        warnings.warn(
            f"Covariance type '{cov_type}' is not recognized.  "
            "Currently only 'gaus_internal', 'SRD', and 'tjpcov' are "
            "implemented.  Using identity matrix as covariance."
        )
        ndata = len(S.mean)
        S.add_covariance(np.eye(ndata))
