"""Binned cluster counts and mean log mass, alone and joint with 3x2pt."""

import copy
import os

import numpy as np
import pytest
import sacc

pytest.importorskip("firecrown.likelihood.clusters")

from augur.analyze import Analyze  # noqa: E402
from augur.generate import generate  # noqa: E402
from augur.generate_utils.cluster_counts import (  # noqa: E402
    add_cluster_counts, cluster_statistics,
)
from augur.utils import cov_utils  # noqa: E402
from augur.utils.config_io import parse_config  # noqa: E402

base_path = os.path.dirname(__file__)

CLUSTERS = {
    'survey_area': 4000.0,
    'z_edges': [0.2, 0.4, 0.6],
    'richness_log_edges': [1.3, 1.5, 1.8],
    'observables': ['counts', 'mass'],
    'model': {
        'mass_distribution': {'type': 'MurataUnbinnedFactory',
                              'pivot_log_mass': 14.3, 'pivot_redshift': 0.5},
        'recipe': {'type': 'GridBinnedClusterRecipeFactory',
                   'mass_interval': [13.0, 16.0], 'true_z_interval': [0.1, 1.2]},
    },
    'covariance': {'ssc': False, 'mean_log_mass_sigma': 0.25},
}


def _config(tmp_path, joint, ssc=False):
    config = parse_config(f'{base_path}/test.yaml')
    if not joint:
        config = {k: config[k] for k in ('general', 'cosmo', 'fisher')}
    config['cluster_counts'] = copy.deepcopy(CLUSTERS)
    config['cluster_counts']['covariance']['ssc'] = ssc
    config['cov_options'] = {'cov_type': 'gaus_internal', 'fsky': 0.1}
    config['fiducial_sacc_path'] = str(tmp_path / 'fid.sacc')
    return config


def test_add_cluster_counts_layout(tmp_path):
    S = sacc.Sacc()
    stats, filters = add_cluster_counts(_config(tmp_path, joint=False), S)
    assert len(stats) == 1 and filters == []
    assert S.tracers['clusters'].sky_area == 4000.0
    assert {t.tracer_type for t in S.tracers.values()} == {'survey', 'bin_z', 'bin_richness'}
    for data_type in ('cluster_counts', 'cluster_mean_log_mass'):
        combs = S.get_tracer_combinations(data_type=data_type)
        assert len(combs) == 4 and combs[0] == ('clusters', 'bin_z_0', 'bin_rich_0')


@pytest.mark.parametrize('cc_update, match', [
    ({'observables': ['mass']}, "needs 'counts'"),
    ({'observables': ['shear']}, "Unknown"),
    ({'z_edges': [0.6, 0.2]}, "increasing"),
])
def test_cluster_counts_config_errors(tmp_path, cc_update, match):
    config = _config(tmp_path, joint=False)
    config['cluster_counts'].update(cc_update)
    with pytest.raises(ValueError, match=match):
        add_cluster_counts(config, sacc.Sacc())


def test_joint_block_diagonal(tmp_path):
    lk, S, _, sys_params = generate(_config(tmp_path, joint=True), return_all_outputs=True)
    i_n = S.indices(data_type='cluster_counts')
    i_m = S.indices(data_type='cluster_mean_log_mass')
    i_cl = np.concatenate([i_n, i_m])
    i_2pt = np.setdiff1d(np.arange(len(S.mean)), i_cl)
    cov = S.covariance.covmat
    counts = S.mean[i_n]

    assert len(lk.get_data_vector()) == len(S.mean)
    assert np.all(counts > 0) and np.all((S.mean[i_m] > 13) & (S.mean[i_m] < 16))
    assert not np.any(cov[np.ix_(i_2pt, i_cl)])
    np.testing.assert_allclose(cov[np.ix_(i_n, i_n)], np.diag(counts))
    np.testing.assert_allclose(np.diag(cov)[i_m], 0.25**2 / counts)
    assert sys_params['mass_distribution_mu0'] == 3.0


@pytest.mark.parametrize('completeness', [False, True])
def test_clusters_only_ssc(tmp_path, completeness):
    config = _config(tmp_path, joint=False, ssc=True)
    if completeness:
        config['cluster_counts']['model']['completeness'] = {'type': 'CompletenessAguena16Factory'}
    _, S, _, _ = generate(config, return_all_outputs=True)
    i_n = S.indices(data_type='cluster_counts')
    block = S.covariance.covmat[np.ix_(i_n, i_n)]
    ssc = block - np.diag(S.mean[i_n])
    np.testing.assert_allclose(block, block.T)
    assert np.all(np.diag(ssc) > 0) and np.linalg.eigvalsh(block).min() > 0


def test_clusters_only_fisher(tmp_path):
    config = _config(tmp_path, joint=False)
    config['fisher'] = {'var_pars': ['Omega_c', 'sigma8', 'mass_distribution_mu0'],
                        'step': 1e-2, 'output': str(tmp_path / 'fisher.dat'),
                        'fid_output': str(tmp_path / 'pars.dat')}
    fisher = Analyze(config).get_fisher_matrix()
    assert fisher.shape == (3, 3)
    assert np.all(np.isfinite(fisher)) and np.linalg.eigvalsh(fisher).min() > 0


def test_cluster_ssc_rescaled_to_crow_counts(tmp_path, monkeypatch):
    """SSC_ij N_i N_j / (G_i G_j), as in CLPipe's CLPCovariance.replace_crow_counts."""
    lk, S, tools, _ = generate(_config(tmp_path, joint=False), return_all_outputs=True)
    (st,) = cluster_statistics(lk)
    tjpcov_counts = np.array([10.0, 20.0, 30.0, 40.0])
    monkeypatch.setattr(cov_utils.TJPCovClusterGaussian, 'get_covariance',
                        lambda self: np.diag(tjpcov_counts))
    ssc = [cov_utils.get_cluster_ssc(S, st.cluster_recipe, tools.ccl_cosmo, f * tjpcov_counts)
           for f in (1.0, 2.0)]
    np.testing.assert_allclose(ssc[1], 4 * ssc[0])
    assert np.all(np.diag(ssc[0]) > 0)
