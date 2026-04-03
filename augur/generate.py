"""Data Generation module

This code generates synthetic data vectors by assembling a configuration
for firecrown and using it to produce likelihoods and SACC files.

The heavy lifting for each probe type lives in :mod:`augur.generate_utils`.
This module orchestrates the high-level flow:

1. Initialise cosmology and tracers.
2. Dispatch to probe-specific helpers (harmonic-space C_ell,
   real-space xi, CMB lensing, cluster counts).
3. Compute the covariance matrix.
4. Optionally write the result to a SACC file.
"""

import numpy as np
import pyccl as ccl
import sacc
import warnings

import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.parameters import ParamsMap
from augur.utils.config_io import parse_config
from augur.utils.theory_utils import compute_new_theory_vector
from augur.utils.firecrown_interface import create_modeling_tools

# ── generate_utils modules ------------------------------------------------ #
from augur.generate_utils.cosmology import initialize_cosmology
from augur.generate_utils.tracers import (
    get_tracers as _get_tracers_impl,
    add_nz as _add_nz_impl,
    setup_sources,
    setup_lenses,
)
from augur.generate_utils.harmonic import (
    add_harmonic_two_point,
    _get_scale_cuts as _get_scale_cuts_impl,
)
from augur.generate_utils.real_space import add_real_space_two_point
from augur.generate_utils.cmb_lensing import add_cmb_lensing
from augur.generate_utils.cluster_counts import add_cluster_counts
from augur.generate_utils.covariance import compute_covariance


# ── Backward-compatible aliases ------------------------------------------- #
# Tests and external code import these directly from augur.generate.

def _get_tracers(statistic, comb):
    """Backward-compatible wrapper — see :func:`augur.generate_utils.tracers.get_tracers`."""
    return _get_tracers_impl(statistic, comb)


def _get_scale_cuts(stat_cfg, comb):
    """Backward-compatible wrapper — see :func:`augur.generate_utils.harmonic._get_scale_cuts`."""
    return _get_scale_cuts_impl(stat_cfg, comb)


# -------------------------------------------------------------------------- #
#  generate_sacc_and_stats
# -------------------------------------------------------------------------- #

def generate_sacc_and_stats(config):
    """
    Generate a placeholder SACC file containing the data-vector
    combinations and tracers specified in the configuration, together with
    the firecrown ``TwoPoint`` statistics needed to build the likelihood.

    The function dispatches to probe-specific helpers in
    :mod:`augur.generate_utils` based on which config sections are present:

    * ``statistics``  →  harmonic-space C_ell (always required for now)
    * ``statistics_real_space``  →  real-space xi (experimental stub)
    * ``cmb_lensing``  →  CMB-lensing cross-/auto-spectra (stub)
    * ``cluster_counts``  →  cluster number counts (stub)

    Parameters
    ----------
    config : dict or path
        Analysis configuration dictionary (or path to YAML).

    Returns
    -------
    S : sacc.Sacc
        Placeholder SACC file.
    cosmo : pyccl.Cosmology
        Fiducial cosmology.
    stats : list[TwoPoint]
        Firecrown TwoPoint statistics for the likelihood.
    sys_params : ParamsMap
        Systematic-parameter map.
    tp_filters : list
        TwoPoint filters encoding scale cuts.
    """
    config = parse_config(config)

    # ── 1. Cosmology ──────────────────────────────────────────────────── #
    cosmo, _ = initialize_cosmology(config)

    # ── 2. Empty SACC ─────────────────────────────────────────────────── #
    S = sacc.Sacc()

    # ── 3. Tracers ────────────────────────────────────────────────────── #
    sources, dndz = setup_sources(config, S)
    sources, dndz = setup_lenses(config, S, sources, dndz)

    sys_params = config.get('systematics', {})

    # ── 4. Probe dispatch ─────────────────────────────────────────────── #
    all_stats = []
    all_tp_filters = []
    active_probes = []

    # 4a. Harmonic-space two-point (required for existing configs)
    if 'statistics' in config:
        stats_h, filters_h = add_harmonic_two_point(
            config, S, sources, dndz, cosmo
        )
        all_stats.extend(stats_h)
        all_tp_filters.extend(filters_h)
        active_probes.append('harmonic')

    # 4b. Real-space two-point (experimental)
    if 'statistics_real_space' in config:
        stats_r, filters_r = add_real_space_two_point(
            config, S, sources, dndz, cosmo
        )
        all_stats.extend(stats_r)
        all_tp_filters.extend(filters_r)
        active_probes.append('real_space')

    # 4c. CMB lensing (stub)
    if 'cmb_lensing' in config:
        stats_cmb, filters_cmb = add_cmb_lensing(
            config, S, sources, dndz, cosmo
        )
        all_stats.extend(stats_cmb)
        all_tp_filters.extend(filters_cmb)
        active_probes.append('cmb_lensing')

    # 4d. Cluster counts (stub)
    if 'cluster_counts' in config:
        stats_cc, filters_cc = add_cluster_counts(
            config, S, sources, cosmo
        )
        all_stats.extend(stats_cc)
        all_tp_filters.extend(filters_cc)
        active_probes.append('cluster_counts')

    if not active_probes:
        raise ValueError(
            'No probe sections found in config.  At least one of '
            '"statistics", "statistics_real_space", "cmb_lensing", '
            'or "cluster_counts" is required.'
        )

    # ── 5. Placeholder covariance ─────────────────────────────────────── #
    ndata = len(S.mean)
    S.add_covariance(np.eye(ndata))
    sys_params = ParamsMap(sys_params)
    return S, cosmo, all_stats, sys_params, all_tp_filters


# -------------------------------------------------------------------------- #
#  generate
# -------------------------------------------------------------------------- #

def generate(configs, return_all_outputs=False, write_sacc=True, use_sacc=None,
             sacc_path=None, lk=None, tools=None):
    """
    Generate a likelihood object and SACC file with the fiducial cosmology.

    Parameters
    ----------
    configs : dict or path
        Analysis configuration dictionary (or path to YAML file).
    return_all_outputs : bool
        If *True*, return ``(lk, S, tools, sys_params)``; otherwise just
        return the likelihood.
    write_sacc : bool
        If *True*, write the fiducial SACC to disk.
    use_sacc : sacc.Sacc, optional
        Bypass ``generate_sacc_and_stats`` and use this pre-existing SACC.
    sacc_path : str, optional
        File path for *use_sacc* (required by Firecrown factories).
    lk : firecrown likelihood, optional
        Inject a pre-built likelihood.
    tools : firecrown ModelingTools, optional
        Inject pre-built modelling tools.

    Returns
    -------
    lk : firecrown.likelihood.ConstGaussian
        Likelihood (always returned).
    S : sacc.Sacc
        SACC with fiducial data vector and covariance
        (only if *return_all_outputs*).
    tools : ModelingTools
        (only if *return_all_outputs*).
    sys_params : ParamsMap
        (only if *return_all_outputs*).
    """
    config = parse_config(configs)

    # ================================================================== #
    #  use_sacc path — bypass generation entirely
    # ================================================================== #
    if use_sacc is not None:
        S = use_sacc

        # Build firecrown source objects from existing SACC tracers
        sources = {}
        for tracer_name in S.tracers:
            tracer_obj = S.get_tracer(tracer_name)
            if tracer_obj.quantity == "galaxy_shear":
                sources[tracer_name] = wl.WeakLensing(sacc_tracer=tracer_obj)
            elif tracer_obj.quantity == "galaxy_density":
                sources[tracer_name] = nc.NumberCounts(
                    sacc_tracer=tracer_obj, derived_scale=True
                )

        if 'statistics' not in config:
            raise ValueError('statistics key is required in config file')
        stat_cfg = config['statistics']
        stats = []
        ignore_sc = config['general'].get('ignore_scale_cuts', False)
        ignore_sc_likelihood = config['general'].get('ignore_scale_cuts_likelihood', False)
        if not ignore_sc and ignore_sc_likelihood:
            raise ValueError(
                "Cannot ignore scale cuts in likelihood while "
                "applying them to the data vector."
            )

        # Detect flat vs nested structure
        if 'tracer_combs' in stat_cfg:
            if use_sacc is None:
                raise ValueError("Flat 'statistics' format requires use_sacc")
            data_types = S.get_data_types()
            if len(data_types) != 1:
                raise ValueError(
                    "Flat statistics format only supported when SACC "
                    "has a single data type"
                )
            key = data_types[0]
            for comb in stat_cfg['tracer_combs']:
                tr1, tr2 = _get_tracers(key, comb)
                stats.append(
                    TwoPoint(source0=sources[tr1], source1=sources[tr2],
                             sacc_data_type=key)
                )
        else:
            for key in stat_cfg:
                for comb in stat_cfg[key]['tracer_combs']:
                    tr1, tr2 = _get_tracers(key, comb)
                    stats.append(
                        TwoPoint(source0=sources[tr1], source1=sources[tr2],
                                 sacc_data_type=key)
                    )

        sys_params = config.get('systematics', {})
        if tools is None:
            tools, cosmo = create_modeling_tools(config)
        else:
            if tools.ccl_cosmo is None:
                cosmo = tools.ccl_factory.build()
                tools.set_ccl_cosmology(cosmo)
            else:
                cosmo = tools.ccl_cosmo

        # Build likelihood
        if lk is None:
            if "Firecrown_Factory" in config:
                from augur.utils.firecrown_interface import load_likelihood_from_yaml
                if sacc_path is None:
                    raise ValueError("Must include sacc_path when passing a sacc")
                lk = load_likelihood_from_yaml(config, tools.ccl_factory, sacc_path)
                _pars = config.get('cosmo', {}).copy()
                print(_pars)
                if (
                    use_sacc is not None
                    and hasattr(use_sacc, 'covariance')
                    and use_sacc.covariance is not None
                ):
                    lk.inv_cov = np.linalg.inv(S.covariance.covmat)
                    lk.cov = S.covariance.covmat
                    lk.data_vector = S.mean
                _, lk, tools = compute_new_theory_vector(
                    lk, tools, sys_params, _pars, return_all=True
                )
            else:
                lk = ConstGaussian(statistics=stats)
                lk.read(S)
        else:
            raise RuntimeError(
                "Non-YAML likelihood with use_sacc is not supported cleanly"
            )

        if return_all_outputs:
            return lk, tools, sys_params
        else:
            return lk

    # ================================================================== #
    #  Normal generation path
    # ================================================================== #

    # ── 1. Generate placeholder SACC and statistics ─────────────────── #
    S, cosmo, stats, sys_params, tp_filters = generate_sacc_and_stats(config)

    # ── 2. Modelling tools ──────────────────────────────────────────── #
    if tools is None:
        tools, cosmo = create_modeling_tools(config)
    cosmo.compute_nonlin_power()

    # ── 3. Likelihood ───────────────────────────────────────────────── #
    if lk is None:
        if "Firecrown_Factory" in config:
            import tempfile, os
            tmp_dir = tempfile.mkdtemp(prefix="augur_sacc_")
            tmp_sacc_path = os.path.join(tmp_dir, "template_placeholder_sacc.fits")
            S.save_fits(tmp_sacc_path, overwrite=True)
            from augur.utils.firecrown_interface import load_likelihood_from_yaml
            lk = load_likelihood_from_yaml(config, tools.ccl_factory, tmp_sacc_path)
        else:
            lk = ConstGaussian(statistics=stats)
            lk.read(S)

    _pars = cosmo.to_dict()
    _, lk, tools = compute_new_theory_vector(
        lk, tools, sys_params, _pars, return_all=True
    )

    # ── 4. Fill SACC with fiducial theory predictions ───────────────── #
    from augur.generate_utils.sacc_interface import (
        extract_x_and_windows, add_data_points,
    )

    x_dict = {}
    win_dict = {}
    for st in lk.statistics:
        st = st.statistic
        tr1 = st.source0.sacc_tracer
        tr2 = st.source1.sacc_tracer
        dtype = st.sacc_data_type
        x_dict[(tr1, tr2)], win_dict[(tr1, tr2)] = extract_x_and_windows(
            S, dtype, tr1, tr2,
        )

    S.covariance = None
    S.data = []
    for st in lk.statistics:
        st = st.statistic
        tr1 = st.source0.sacc_tracer
        tr2 = st.source1.sacc_tracer
        st.ready = False
        add_data_points(
            S, st.sacc_data_type, tr1, tr2,
            x_dict[(tr1, tr2)], st.get_theory_vector(),
            window=win_dict[(tr1, tr2)],
        )

    # ── 5. Covariance ───────────────────────────────────────────────── #
    compute_covariance(config, S, lk, cosmo, tools)

    # ── 6. Write SACC ───────────────────────────────────────────────── #
    if write_sacc:
        print(config['fiducial_sacc_path'])
        S.save_fits(config['fiducial_sacc_path'], overwrite=True)

    # ── 7. Re-build likelihood with scale-cut filters ───────────────── #
    if tp_filters:
        print('loading filters')
        config = parse_config(configs)
        from augur.utils.firecrown_interface import load_likelihood_from_yaml
        lk = load_likelihood_from_yaml(
            config, tools.ccl_factory,
            config['fiducial_sacc_path'],
            filters=tp_filters,
        )
    elif "Firecrown_Factory" in config:
        import tempfile, os
        tmp_dir = tempfile.mkdtemp(prefix="augur_sacc_final_")
        tmp_sacc_path = os.path.join(tmp_dir, "final_sacc.fits")
        S.save_fits(tmp_sacc_path, overwrite=True)
        from augur.utils.firecrown_interface import load_likelihood_from_yaml
        lk = load_likelihood_from_yaml(config, tools.ccl_factory, tmp_sacc_path)
    else:
        lk = ConstGaussian(statistics=stats)
        lk.read(S)

    if return_all_outputs:
        return lk, S, tools, sys_params
