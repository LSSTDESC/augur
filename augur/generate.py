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

import logging
import numpy as np
import sacc
import warnings
import tempfile
import os

import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
import firecrown.likelihood.cmb as cmb
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.parameters import ParamsMap
from augur.utils.config_io import parse_config
from augur.utils.theory_utils import compute_new_theory_vector
from augur.utils.firecrown_interface import create_modeling_tools
from augur.utils.firecrown_interface import load_likelihood_from_yaml
from augur.utils.firecrown_interface import create_twopoint_filter
import pyccl as ccl

# ── generate_utils modules ------------------------------------------------ #
from augur.generate_utils.cosmology import initialize_cosmology
from augur.generate_utils.tracers import (
    get_tracers as _get_tracers_impl,
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
from augur.generate_utils.sacc_interface import (
    extract_x_and_windows, add_data_points,
)


logger = logging.getLogger(__name__)


# ── Backward-compatible aliases ------------------------------------------- #
# Tests and external code import these directly from augur.generate.

def _get_tracers(statistic, comb):
    """Backward-compatible wrapper — see :func:`augur.generate_utils.tracers.get_tracers`."""
    return _get_tracers_impl(statistic, comb)


def _get_scale_cuts(stat_cfg, comb):
    """Backward-compatible wrapper — see :func:`augur.generate_utils.harmonic._get_scale_cuts`."""
    return _get_scale_cuts_impl(stat_cfg, comb)


def _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood):
    """
    Build ``TwoPointBinFilter`` objects for the ``Firecrown_Factory`` + *use_sacc*
    path so that scale cuts and tracer subsetting are honoured when the data
    vector comes from a pre-existing SACC (instead of being generated here).

    For every tracer combination listed under a harmonic ``statistics`` block:

    - combinations absent from the SACC are skipped (with a warning);
    - the upper cut is taken from ``kmax``/``lmax`` (or the SACC's own ell range
      when ``ignore_sc_likelihood`` is set);
    - the lower bound is floored at 0 (there is no lower scale cut here; see the
      note below).

    Combinations present in the SACC but *not* listed in the config are excluded
    from the likelihood via an out-of-range filter, so a pre-made SACC can carry
    more data than a given analysis uses.

    Parameters
    ----------
    stat_cfg : dict
        The ``statistics`` section of the config (nested format only).
    S : sacc.Sacc
        Pre-existing SACC supplying the data vector.
    cosmo : pyccl.Cosmology
        Fiducial cosmology, used only for the ``kmax`` -> ``lmax`` conversion.
    ignore_sc_likelihood : bool
        When *True*, ell cuts are suppressed and filters span the full SACC ell
        range (tracer subsetting is still applied).

    Returns
    -------
    tp_filters : list of firecrown ``TwoPointBinFilter``
    """
    tp_filters = []

    for key in stat_cfg.keys():
        # Restrict to this data type so cross-probe combinations don't collide.
        sacc_combs = set(S.get_tracer_combinations(data_type=key))
        config_stats = stat_cfg[key]['tracer_combs']
        config_pairs = set()

        for comb in config_stats:
            lmax, kmax = _get_scale_cuts(stat_cfg[key], comb)
            tr1, tr2 = _get_tracers(key, comb)
            config_pairs.add((tr1, tr2))

            if (tr1, tr2) not in sacc_combs and (tr2, tr1) not in sacc_combs:
                logger.warning(
                    'Tracer combination (%s, %s) for %s not found in SACC; skipping.',
                    tr1, tr2, key,
                )
                continue

            try:
                ells_in_sacc, _ = S.get_ell_cl(key, tr1, tr2)
            except Exception:
                logger.warning(
                    'Could not retrieve ells for (%s, %s) / %s from SACC; skipping.',
                    tr1, tr2, key,
                )
                continue

            # cut_low is the floor of the kept range, not a lower scale cut. With
            # bandpower windows and the default SUPPORT filter method a bin is
            # kept only if its whole window lies within [cut_low, cut_high]; using
            # the lowest bin CENTRE would straddle its window and silently drop
            # that bin, so we floor at 0.
            cut_low = 0.0

            if ignore_sc_likelihood:
                cut_high = float(ells_in_sacc[-1])
            elif (kmax is not None) and (kmax != 'None'):
                t1, t2 = S.get_tracer(tr1), S.get_tracer(tr2)
                zmean1 = np.average(t1.z, weights=t1.nz)
                zmean2 = np.average(t2.z, weights=t2.nz)
                a12 = np.array([1.0 / (1 + zmean1), 1.0 / (1 + zmean2)])
                cut_high = float(np.min(kmax * ccl.comoving_radial_distance(cosmo, a12)))
            elif (lmax is not None) and (lmax != 'None'):
                cut_high = float(lmax)
            else:
                cut_high = float(ells_in_sacc[-1])

            tp_filters.append(
                create_twopoint_filter(key, tr1, tr2, cut_low=cut_low, cut_high=cut_high)
            )

        # Exclude SACC combinations that the config does not ask for.
        for (tr1, tr2) in sacc_combs:
            if (tr1, tr2) in config_pairs or (tr2, tr1) in config_pairs:
                continue
            logger.warning(
                'Tracer combination (%s, %s) found in SACC but not in config; '
                'excluding it from the likelihood.', tr1, tr2,
            )
            try:
                ells_in_sacc, _ = S.get_ell_cl(key, tr1, tr2)
            except Exception:
                continue
            cut_high = float(ells_in_sacc[-1])
            tp_filters.append(
                create_twopoint_filter(key, tr1, tr2,
                                       cut_low=cut_high + 1, cut_high=cut_high + 2)
            )

    return tp_filters


# -------------------------------------------------------------------------- #
#  Private helpers for generate()
# -------------------------------------------------------------------------- #


def _resolve_tools(config, tools):
    """Create or validate modelling tools, returning (tools, cosmo).

    If *tools* is None, calls :func:`create_modeling_tools`.  Otherwise
    ensures the CCL cosmology is attached.
    """
    if tools is None:
        tools, cosmo = create_modeling_tools(config)
    else:
        if tools.ccl_cosmo is None:
            cosmo = tools.ccl_factory.build()
            tools.set_ccl_cosmology(cosmo)
        else:
            cosmo = tools.ccl_cosmo
    return tools, cosmo


def _sacc_to_disk(S, *, hint_path=None, prefix="augur_sacc_",
                  filename="sacc.sacc"):
    """Ensure a SACC object is available on disk and return the path.

    If *hint_path* is provided (and not None), returns it immediately
    without writing — the file is assumed to already exist.  Otherwise
    creates a temporary directory and writes *S* there.
    """
    if hint_path is not None:
        return hint_path
    tmp_dir = tempfile.mkdtemp(prefix=prefix)
    path = os.path.join(tmp_dir, filename)
    S.save_fits(path, overwrite=True)
    logger.debug("Wrote temporary SACC to %s", path)
    return path


def _build_sources_and_stats_from_sacc(config, S):
    """Build firecrown source objects and TwoPoint statistics from a SACC.

    Used by the *use_sacc* path in :func:`generate` to reconstruct the
    sources dict, statistics list, and systematic-parameter map from an
    existing SACC file rather than generating them from scratch.

    Returns
    -------
    sources : dict
        Map of tracer name to firecrown source.
    stats : list[TwoPoint]
        Firecrown TwoPoint statistics constructed from the config.
    sys_params : dict
        Raw systematic-parameter dictionary (not yet a ParamsMap).
    """
    sources = {}
    for tracer_name in S.tracers:
        tracer_obj = S.get_tracer(tracer_name)
        quantity = getattr(tracer_obj, "quantity", None)
        if quantity == "galaxy_shear" or tracer_name.startswith("src"):
            sources[tracer_name] = wl.WeakLensing(sacc_tracer=tracer_obj)
        elif quantity == "galaxy_density" or tracer_name.startswith("lens"):
            sources[tracer_name] = nc.NumberCounts(
                sacc_tracer=tracer_obj, derived_scale=True
            )
        elif quantity == "cmb_convergence" or tracer_name == "cmb_convergence":
            z_source = config.get('cmb_lensing', {}).get('z_source', 1100.0)
            sources[tracer_name] = cmb.CMBConvergence(
                sacc_tracer=tracer_name,
                z_source=z_source,
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
    return sources, stats, sys_params


def _refill_sacc_with_theory(S, lk):
    """Replace placeholder SACC data with the likelihood's theory vector.

    Preserves non-two-point data entries (e.g. cluster counts) by
    storing them before clearing and re-appending after the refill.
    Mutates *S* in place.
    """
    lk_tracer_keys = set()
    x_dict = {}
    win_dict = {}
    for st in lk.statistics:
        st = st.statistic
        tr1 = st.source0.sacc_tracer
        tr2 = st.source1.sacc_tracer
        dtype = st.sacc_data_type
        lk_tracer_keys.add((dtype, tr1, tr2))
        x_dict[(tr1, tr2)], win_dict[(tr1, tr2)] = extract_x_and_windows(
            S, dtype, tr1, tr2,
        )

    # Preserve non-two-point data entries (e.g. cluster counts)
    other_data = [
        dp for dp in S.data
        if (
            dp.data_type,
            dp.tracers[0] if dp.tracers else None,
            dp.tracers[1] if len(dp.tracers) > 1 else None,
        )
        not in lk_tracer_keys
    ]

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

    S.data.extend(other_data)


def _build_final_lk(config, S, tools, stats, tp_filters, *,
                    sacc_on_disk_path=None):
    """Re-build the likelihood after covariance has been applied.

    Handles three cases:
    1. *tp_filters* present: load with scale-cut filters.
    2. Firecrown_Factory config but no filters: load from SACC on disk.
    3. No factory: build a plain ConstGaussian.

    Parameters
    ----------
    sacc_on_disk_path : str or None
        If the SACC was already written to disk (e.g. via *write_sacc*),
        pass that path here to avoid creating another temp file.
    """
    if tp_filters:
        logger.debug("Rebuilding likelihood with scale-cut filters")
        sacc_for_rebuild = _sacc_to_disk(
            S, hint_path=sacc_on_disk_path, prefix="augur_sacc_filters_",
            filename="filtered_sacc.sacc",
        )
        lk = load_likelihood_from_yaml(
            config, tools.ccl_factory,
            sacc_for_rebuild,
            filters=tp_filters,
        )
    elif "Firecrown_Factory" in config:
        sacc_for_rebuild = _sacc_to_disk(
            S, prefix="augur_sacc_final_", filename="final_sacc.sacc",
        )
        lk = load_likelihood_from_yaml(config, tools.ccl_factory, sacc_for_rebuild)
    else:
        lk = ConstGaussian(statistics=stats)
        lk.read(S)
    return lk


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
    active_probes : list[str]
        Which probes are present (e.g. ``['harmonic', 'cmb_lensing']``).
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

    if 'statistics' in config and 'statistics_real_space' in config:
        raise ValueError(
            "Augur does not support specifying both \'statistics\'"
            " and \'statistics_real_space\' in config."
        )

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
        if 'statistics_real_space' in config:
            warnings.warn(
                "CMB lensing generation is currently only implemented for "
                "harmonic-space statistics. If 'statistics_real_space' is also "
                "present in the config, the CMB lensing section will be ignored."
            )
        else:
            stats_cmb, filters_cmb = add_cmb_lensing(
                config, S, sources, dndz, cosmo
            )
            all_stats.extend(stats_cmb)
            all_tp_filters.extend(filters_cmb)
            active_probes.append('cmb_lensing')

    # 4d. Cluster counts (stub — registers SACC tracers for TJPCov but
    #     no firecrown statistics are returned yet)
    if 'cluster_counts' in config:
        warnings.warn(
            "Cluster count datavector generation is not yet implemented in "
            "Augur. Placeholder tracers will be registered in the SACC for "
            "TJPCov covariance, but no firecrown likelihood statistics are "
            "created. Firecrown factories for cluster counts with CROW are "
            "not yet available."
        )
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
    return S, cosmo, all_stats, sys_params, all_tp_filters, active_probes


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
        logger.info("Using pre-existing SACC file for generation: %s, "
                    "setting write_sacc=False", sacc_path)
        write_sacc = False
        S = use_sacc

        sources, stats, sys_params = _build_sources_and_stats_from_sacc(
            config, S
        )
        tools, cosmo = _resolve_tools(config, tools)

        # Build likelihood
        if lk is None:
            if "Firecrown_Factory" in config:
                stat_cfg = config.get('statistics', {})
                ignore_sc_likelihood = config.get('general', {}).get(
                    'ignore_scale_cuts_likelihood', False
                )
                sacc_input = _sacc_to_disk(S, hint_path=sacc_path)
                if 'tracer_combs' not in stat_cfg:
                    # Nested statistics format: build per-combination ell filters
                    # so scale cuts and tracer subsetting from the config are
                    # applied to the pre-made SACC.  Combinations absent from the
                    # config are excluded; those absent from the SACC are skipped.
                    tp_filters = _build_tp_filters_from_sacc(
                        stat_cfg, S, cosmo, ignore_sc_likelihood
                    )
                    lk = load_likelihood_from_yaml(
                        config, tools.ccl_factory, sacc_input, filters=tp_filters
                    )
                else:
                    # Flat statistics format: the SACC drives everything unchanged.
                    lk = load_likelihood_from_yaml(
                        config, tools.ccl_factory, sacc_input
                    )
                # Use the built CCL cosmology dictionary so expected amplitude
                # keys (A_s/sigma8) and extra parameters are always present.
                if tools.ccl_cosmo is not None:
                    _pars = tools.ccl_cosmo.to_dict()
                else:
                    _pars = config.get('cosmo', {}).copy()
                logger.debug("use_sacc cosmo params: %s", _pars)
                if (
                    hasattr(S, 'covariance')
                    and S.covariance is not None
                ):
                    # Only copy the SACC covariance onto the likelihood when the
                    # shapes agree.  When scale-cut filters were applied the
                    # likelihood is shorter than the full SACC covariance, and
                    # Firecrown has already installed the correctly filtered
                    # covariance from the SACC — do not overwrite it.
                    n_lk = len(lk.get_data_vector())
                    n_cov = S.covariance.covmat.shape[0]
                    if n_lk == n_cov:
                        lk.inv_cov = np.linalg.inv(S.covariance.covmat)
                        lk.cov = S.covariance.covmat
                        lk.data_vector = S.mean
                    else:
                        logger.info(
                            "use_sacc: likelihood length %d != SACC covariance "
                            "%d (scale cuts applied); using Firecrown's filtered "
                            "covariance.", n_lk, n_cov,
                        )
                _, lk, tools = compute_new_theory_vector(
                    lk, tools, sys_params, _pars, return_all=True
                )
            else:
                lk = ConstGaussian(statistics=stats)
                logger.debug(
                    "use_sacc path: building ConstGaussian with %d statistics",
                    len(stats),
                )
                lk.read(S)
        else:
            raise RuntimeError(
                "Non-YAML likelihood with use_sacc is not supported cleanly"
            )

        if return_all_outputs:
            return lk, S, tools, sys_params
        return lk

    # ================================================================== #
    #  Normal generation path
    # ================================================================== #

    # ── 1. Generate placeholder SACC and statistics ─────────────────── #
    S, cosmo, stats, sys_params, tp_filters, active_probes = \
        generate_sacc_and_stats(config)

    # ── 2. Modelling tools ──────────────────────────────────────────── #
    if tools is None:
        tools, cosmo = _resolve_tools(config, tools)
    cosmo.compute_nonlin_power()

    # ── 3. Likelihood ───────────────────────────────────────────────── #
    if lk is None:
        if "Firecrown_Factory" in config:
            lk = load_likelihood_from_yaml(
                config, tools.ccl_factory,
                _sacc_to_disk(S, prefix="augur_sacc_"),
            )
        else:
            lk = ConstGaussian(statistics=stats)
            lk.read(S)

    _pars = cosmo.to_dict()
    _, lk, tools = compute_new_theory_vector(
        lk, tools, sys_params, _pars, return_all=True
    )

    # ── 4. Fill SACC with fiducial theory predictions ───────────────── #
    _refill_sacc_with_theory(S, lk)

    # ── 5. Covariance ───────────────────────────────────────────────── #
    compute_covariance(config, S, lk, cosmo, tools, probes=active_probes)

    # ── 6. Write SACC ───────────────────────────────────────────────── #
    if write_sacc:
        logger.info("Writing SACC to %s", config['fiducial_sacc_path'])
        S.save_fits(config['fiducial_sacc_path'], overwrite=True)

    # ── 7. Re-build likelihood with scale-cut filters ───────────────── #
    lk = _build_final_lk(
        config, S, tools, stats, tp_filters,
        sacc_on_disk_path=config['fiducial_sacc_path'] if write_sacc else None,
    )

    if return_all_outputs:
        return lk, S, tools, sys_params
    return lk
