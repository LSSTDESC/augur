"""Binned cluster number counts and mean log10 mass for SACC generation.

The ``cluster_counts`` config section describes one cluster sample: its
redshift-richness binning and its crow model, written as a firecrown
``ClusterModelFactory``::

    cluster_counts:
      survey_name: clusters            # SACC survey tracer (default 'clusters')
      survey_area: 18000               # deg^2
      z_edges: [0.2, 0.4, 0.6, 0.8]
      richness_log_edges: [1.3, 1.5, 1.8, 2.3]   # log10(richness)
      observables: [counts, mass]      # default [counts]
      model:
        mass_distribution: {type: MurataUnbinnedFactory,
                            pivot_log_mass: 14.3, pivot_redshift: 0.5}
        recipe: {type: GridBinnedClusterRecipeFactory,
                 mass_interval: [13.0, 16.0], true_z_interval: [0.1, 1.5]}
      covariance:
        ssc: true                      # TJPCov super-sample term of the counts
        halo_bias: Tinker10            # used by the SSC term only
        mean_log_mass_sigma: 0.25      # dex scatter of log10 M in a bin; needed for 'mass'

The likelihood statistics come from firecrown's ``BinnedClusterFactory``
(``firecrown.likelihood.clusters``), so augur needs a firecrown that has it.
"""

import logging

import numpy as np
import sacc
from firecrown.likelihood.two_point import TwoPoint
from firecrown.parameters import ParamsMap

from augur.utils.config_io import parse_array

logger = logging.getLogger(__name__)

CLUSTER_DATA_TYPES = {
    'counts': sacc.standard_types.cluster_counts,
    'mass': sacc.standard_types.cluster_mean_log_mass,
}
CLUSTER_TRACER_TYPES = ('survey', 'bin_z', 'bin_richness', 'bin_radius')


def _cluster_settings(cc_cfg):
    """Return the survey name and the observables of a cluster_counts section."""
    observables = list(cc_cfg.get('observables', ['counts']))
    unknown = set(observables) - set(CLUSTER_DATA_TYPES)
    if unknown:
        raise ValueError(f"Unknown cluster_counts observables {sorted(unknown)}; "
                         f"expected a subset of {list(CLUSTER_DATA_TYPES)}.")
    if 'mass' in observables and 'counts' not in observables:
        raise ValueError("cluster_counts: 'mass' needs 'counts' too; the mean-mass "
                         "variance is set by the expected number of clusters.")
    return cc_cfg.get('survey_name', 'clusters'), observables


def make_cluster_factory(config):
    """Build the firecrown BinnedClusterFactory of the cluster_counts section.

    Returns None when the config has no cluster_counts section.
    """
    cc_cfg = config.get('cluster_counts')
    if cc_cfg is None:
        return None
    try:
        from firecrown.likelihood.factories import BinnedClusterFactory
    except ImportError as exc:
        raise ImportError(
            "cluster_counts needs a firecrown with firecrown.likelihood.clusters "
            "(BinnedClusterFactory)."
        ) from exc
    survey_name, observables = _cluster_settings(cc_cfg)
    if 'model' not in cc_cfg:
        raise ValueError("cluster_counts needs a 'model' (a firecrown ClusterModelFactory).")
    return BinnedClusterFactory.model_validate({'abundance_factories': [{
        'survey_name': survey_name,
        'observables': observables,
        'model': cc_cfg['model'],
    }]})


def add_cluster_counts(config, S):
    """Add the cluster tracers and placeholder data points to a SACC object.

    Writes a ``survey`` tracer (area in deg^2), ``bin_z_i`` and ``bin_rich_j``
    tracers, and one point per (z, richness) bin and observable, on the
    tracers ``(survey, bin_z_i, bin_rich_j)``, as firecrown and TJPCov read them.

    Returns
    -------
    stats : list
        The firecrown cluster statistics, not yet read from *S*.
    tp_filters : list
        Always empty: scale cuts apply to two-point data only.
    """
    cc_cfg = config['cluster_counts']
    survey_name, observables = _cluster_settings(cc_cfg)
    z_edges = parse_array(cc_cfg['z_edges'])
    rich_edges = parse_array(cc_cfg['richness_log_edges'])
    for name, edges in (('z_edges', z_edges), ('richness_log_edges', rich_edges)):
        if len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError(f"cluster_counts.{name} must be increasing with >= 2 edges.")
    area = float(cc_cfg['survey_area'])
    if area <= 0:
        raise ValueError("cluster_counts.survey_area must be positive (deg^2).")

    S.add_tracer('survey', survey_name, area)
    z_labels = [f'bin_z_{i}' for i in range(len(z_edges) - 1)]
    rich_labels = [f'bin_rich_{j}' for j in range(len(rich_edges) - 1)]
    for label, lo, hi in zip(z_labels, z_edges[:-1], z_edges[1:]):
        S.add_tracer('bin_z', label, float(lo), float(hi))
    for label, lo, hi in zip(rich_labels, rich_edges[:-1], rich_edges[1:]):
        S.add_tracer('bin_richness', label, float(lo), float(hi))
    for obs in observables:
        for z_label in z_labels:
            for rich_label in rich_labels:
                S.add_data_point(CLUSTER_DATA_TYPES[obs],
                                 (survey_name, z_label, rich_label), 0.0)

    return create_cluster_statistics(config), []


def create_cluster_statistics(config):
    """Return new, unread firecrown statistics of the cluster_counts section."""
    factory = make_cluster_factory(config)
    return [] if factory is None else [f.create() for f in factory.abundance_factories]


def cluster_statistics(lk):
    """Return the statistics of a likelihood that are not two-point."""
    return [g.statistic for g in lk.statistics if not isinstance(g.statistic, TwoPoint)]


def add_cluster_parameter_defaults(lk, sys_params):
    """Return *sys_params* with crow defaults for the cluster parameters it lacks.

    Firecrown raises on missing sampler parameters, and the cluster ones
    (``mass_distribution_mu0`` ...) are rarely all listed under ``systematics``.
    """
    out = dict(sys_params.items())
    for st in cluster_statistics(lk):
        for name, value in st.required_parameters().get_default_values().items():
            if name not in out:
                logger.info("Cluster parameter %s not in systematics; using %g", name, value)
                out[name] = value
    return ParamsMap(out) if isinstance(sys_params, ParamsMap) else out
