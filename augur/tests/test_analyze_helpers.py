import numpy as np
import pytest

from augur.analyze import Analyze


class DummyLikelihood:
    def get_data_vector(self):
        return np.array([])

    @property
    def inv_cov(self):
        return np.eye(0)


class DummyTools:
    def __init__(self, pars_fid):
        class Cosmo:
            def __init__(self, d):
                self.__dict__['_params_init_kwargs'] = d

            def to_dict(self):
                return self.__dict__['_params_init_kwargs']
        self._cosmo = Cosmo(pars_fid)
        self.ccl_factory = None

    def get_ccl_cosmology(self):
        return self._cosmo


def make_analyze(var_pars, pars_fid, extra_fisher_cfg=None, req_params=None):
    fisher_cfg = {'var_pars': var_pars, 'output': 'out', 'fid_output': 'fid'}
    if extra_fisher_cfg:
        fisher_cfg.update(extra_fisher_cfg)
    config = {'fisher': fisher_cfg}
    lk = DummyLikelihood()
    tools = DummyTools(pars_fid)
    if req_params is None:
        req_params = {}
    return Analyze(config, likelihood=lk, tools=tools, req_params=req_params)


def test_get_Om_basic():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7}
    a = make_analyze(['Omega_c', 'Omega_b', 'h'], pars)
    assert pytest.approx(a.get_Om()) == 0.25


def test_get_Om_with_mnu():
    mnu = 0.06
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': mnu}
    a = make_analyze(['Omega_c', 'Omega_b', 'm_nu', 'h'], pars)
    expected = pars['Omega_c'] + pars['Omega_b'] + pars['m_nu'] / pars['h'] / pars['h'] / 93.14
    assert pytest.approx(a.get_Om(), rel=1e-6) == expected


def test_get_Om_requires_Omega_c():
    pars = {'sigma8': 0.8}
    a = make_analyze(['sigma8'], pars)
    with pytest.raises(ValueError):
        a.get_Om()


def test_get_S8_basic():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'sigma8': 0.8}
    a = make_analyze(['Omega_c', 'sigma8'], pars)
    # Use Analyze.get_S8 which internally calls get_Om; ensure Omega_b present in pars_fid
    pars_full = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'sigma8': 0.8}
    a = make_analyze(['Omega_c', 'Omega_b', 'sigma8', 'h'], pars_full)
    expected = np.sqrt(a.get_Om() / 0.3) * pars_full['sigma8']
    assert pytest.approx(a.get_S8(), rel=1e-6) == expected


def test_get_S8_requires_sigma8():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7}
    a = make_analyze(['Omega_c', 'Omega_b', 'h'], pars)
    with pytest.raises(ValueError):
        a.get_S8()


def test_Jacobian_transform_entries():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.06, 'sigma8': 0.8}
    var_pars = ['Omega_c', 'Omega_b', 'm_nu', 'h', 'sigma8']
    extra_cfg = {'transform_Omega_m': True, 'transform_S8': True}
    a = make_analyze(var_pars, pars, extra_fisher_cfg=extra_cfg)
    J = a.Jacobian_transform()
    ind_c = var_pars.index('Omega_c')
    ind_b = var_pars.index('Omega_b')
    ind_nu = var_pars.index('m_nu')
    ind_sigma8 = var_pars.index('sigma8')

    assert J[ind_c][ind_b] == -1.0
    expected_nu = -1.0 / (pars['h'] * pars['h'] * 93.14)
    assert pytest.approx(J[ind_c][ind_nu], rel=1e-8) == expected_nu
    # Check S8 diagonal scaling
    expected_s8_diag = 1.0 / np.sqrt(a.get_Om() / 0.3)
    assert pytest.approx(J[ind_sigma8][ind_sigma8], rel=1e-8) == expected_s8_diag


def test_add_gaussian_priors_preserves_width_order():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    var_pars = ['Omega_c', 'h']
    extra_cfg = {'gaussian_priors': {'h': 0.2, 'Omega_c': 0.1}}
    a = make_analyze(var_pars, pars, extra_fisher_cfg=extra_cfg)
    a.Fij = np.zeros((len(var_pars), len(var_pars)))

    gprior_only = a.add_gaussian_priors(save_txt=False)

    assert gprior_only.shape == (2, 2)
    ind_c = var_pars.index('Omega_c')
    ind_h = var_pars.index('h')
    assert gprior_only[ind_c, ind_c] == pytest.approx(1.0 / 0.1**2)
    assert gprior_only[ind_h, ind_h] == pytest.approx(1.0 / 0.2**2)


def test_unpack_gaussian_priors_nonexistent_parameter_raises():
    # A prior on a parameter not in var_pars should raise at init time
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {'gaussian_priors': {'Omega_b': 0.05}}
    with pytest.raises(ValueError, match='not in var_pars'):
        make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)


def test_unpack_gaussian_priors_transform_Omega_m_allowed():
    # A prior on Omega_m is valid when transform_Omega_m is active,
    # even though Omega_m is not literally in var_pars
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7}
    extra_cfg = {
        'transform_Omega_m': True,
        'gaussian_priors': {'Omega_m': 0.01},
    }
    a = make_analyze(['Omega_c', 'Omega_b', 'h'], pars, extra_fisher_cfg=extra_cfg)
    assert 'Omega_m' in a.gprior_pars


def test_unpack_gaussian_priors_transform_S8_allowed():
    # A prior on S8 is valid when transform_S8 is active
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'sigma8': 0.8}
    extra_cfg = {
        'transform_S8': True,
        'gaussian_priors': {'S8': 0.02},
    }
    a = make_analyze(['Omega_c', 'h', 'sigma8'], pars, extra_fisher_cfg=extra_cfg)
    assert 'S8' in a.gprior_pars


def test_unpack_gaussian_priors_Omega_m_without_transform_raises():
    # Omega_m prior without transform_Omega_m active should raise
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7}
    extra_cfg = {'gaussian_priors': {'Omega_m': 0.01}}
    with pytest.raises(ValueError, match='not in var_pars'):
        make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)


# Tests for _unpack_transformations
def test_unpack_transformations_transform_Omega_m_true():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7}
    extra_cfg = {'transform_Omega_m': True}
    a = make_analyze(['Omega_c', 'Omega_b', 'h'], pars, extra_fisher_cfg=extra_cfg)
    assert a.transform_Omega_m is True
    assert a.transform_S8 is False


def test_unpack_transformations_transform_S8_true():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'sigma8': 0.8}
    extra_cfg = {'transform_S8': True}
    a = make_analyze(['Omega_c', 'Omega_b', 'h', 'sigma8'], pars, extra_fisher_cfg=extra_cfg)
    assert a.transform_S8 is True
    assert a.transform_Omega_m is False


def test_unpack_transformations_both_true():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'sigma8': 0.8}
    extra_cfg = {'transform_Omega_m': True, 'transform_S8': True}
    a = make_analyze(['Omega_c', 'Omega_b', 'h', 'sigma8'], pars, extra_fisher_cfg=extra_cfg)
    assert a.transform_Omega_m is True
    assert a.transform_S8 is True


def test_unpack_transformations_non_boolean_warns():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {'transform_Omega_m': 'yes'}  # Not a boolean
    with pytest.warns(UserWarning):
        a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    assert a.transform_Omega_m is False


def test_unpack_transformations_defaults_to_false():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    a = make_analyze(['Omega_c', 'h'], pars)
    assert a.transform_Omega_m is False
    assert a.transform_S8 is False


# Tests for _unpack_gaussian_priors
def test_unpack_gaussian_priors_single_prior():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    var_pars = ['Omega_c', 'h']
    extra_cfg = {'gaussian_priors': {'Omega_c': 0.05}}
    a = make_analyze(var_pars, pars, extra_fisher_cfg=extra_cfg)
    # _unpack_gaussian_priors is called during __init__
    assert a.gprior_pars == ['Omega_c']
    assert 0.05 in a.gpriors
    assert len(a.gpriors) == 1


def test_unpack_gaussian_priors_multiple_priors():
    pars = {'Omega_c': 0.2, 'h': 0.7, 'sigma8': 0.8}
    var_pars = ['Omega_c', 'h', 'sigma8']
    extra_cfg = {'gaussian_priors': {'Omega_c': 0.05, 'sigma8': 0.1}}
    a = make_analyze(var_pars, pars, extra_fisher_cfg=extra_cfg)
    # _unpack_gaussian_priors is called during __init__
    assert set(a.gprior_pars) == {'Omega_c', 'sigma8'}
    assert set(a.gpriors) == {0.05, 0.1}
    assert len(a.gpriors) == 2


def test_unpack_gaussian_priors_none_by_default():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    a = make_analyze(['Omega_c', 'h'], pars)
    assert a.gprior_pars is None
    assert len(a.gpriors) == 0


def test_unpack_gaussian_priors_extracts_all_priors():
    """Test that _unpack_gaussian_priors correctly extracts all gaussian prior widths"""
    pars = {'Omega_c': 0.2, 'h': 0.7, 'm_nu': 0.06}
    var_pars = ['Omega_c', 'h', 'm_nu']
    extra_cfg = {'gaussian_priors': {'Omega_c': 0.02, 'h': 0.05, 'm_nu': 0.01}}
    a = make_analyze(var_pars, pars, extra_fisher_cfg=extra_cfg)
    assert len(a.gprior_pars) == 3
    assert set(a.gprior_pars) == {'Omega_c', 'h', 'm_nu'}
    assert len(a.gpriors) == 3


# Tests for _unpack_derivative_method
def test_unpack_derivative_method_default():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    a = make_analyze(['Omega_c', 'h'], pars)
    assert a.derivative_method == 'numdifftools'
    assert a.step_size == 0.01
    assert a.derivative_args == {}


def test_unpack_derivative_method_numdifftools():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {'derivative_method': 'numdifftools', 'step': 0.005}
    a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    assert a.derivative_method == 'numdifftools'
    assert a.step_size == 0.005


def test_unpack_derivative_method_with_args():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {
        'derivative_method': 'numdifftools',
        'derivative_args': {'order': 4, 'full_output': True}
    }
    a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    assert a.derivative_method == 'numdifftools'
    assert a.derivative_args == {'order': 4, 'full_output': True}


def test_unpack_derivative_method_derivkit():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {'derivative_method': 'derivkit', 'step': 0.02}
    a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    assert a.derivative_method == 'derivkit'
    assert a.step_size == 0.02


# Tests for _unpack_norm_step
def test_unpack_norm_step_true_with_bounds():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {
        'var_pars': ['Omega_c', 'h'],
        'parameters': {'Omega_c': [0.1, 0.2, 0.3], 'h': [0.6, 0.7, 0.8]}
    }
    a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    # Now enable norm_step after initialization to trigger _unpack_norm_step behavior
    # Reinitialize with norm_step=True
    config = {'fisher': extra_cfg}
    lk = DummyLikelihood()
    tools = DummyTools(pars)
    req_params = {}
    a = Analyze(config, likelihood=lk, tools=tools, req_params=req_params, norm_step=True)
    assert a.norm_step is True
    expected_norm = np.array([0.3 - 0.1, 0.8 - 0.6])
    assert np.allclose(a.norm, expected_norm)


def test_unpack_norm_step_true_without_bounds_warns():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {'var_pars': ['Omega_c', 'h']}
    config = {'fisher': extra_cfg}
    lk = DummyLikelihood()
    tools = DummyTools(pars)
    req_params = {}
    with pytest.warns(UserWarning):
        a = Analyze(config, likelihood=lk, tools=tools, req_params=req_params, norm_step=True)
    assert a.norm_step is False


# Tests for _unpack_parameters_and_var_pars with 'parameters' option
def test_unpack_parameters_with_bounds():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {
        'parameters': {
            'Omega_c': [0.1, 0.2, 0.3],
            'h': [0.6, 0.7, 0.8]
        }
    }
    a = make_analyze([], pars, extra_fisher_cfg=extra_cfg)
    assert list(a.var_pars) == ['Omega_c', 'h']
    assert np.allclose(a.x, [0.2, 0.7])
    assert a.par_bounds.shape == (2, 2)
    assert np.allclose(a.par_bounds[0], [0.1, 0.3])
    assert np.allclose(a.par_bounds[1], [0.6, 0.8])


def test_unpack_parameters_scalar_values():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {
        'parameters': {
            'Omega_c': 0.2,
            'h': 0.7
        }
    }
    a = make_analyze([], pars, extra_fisher_cfg=extra_cfg)
    assert list(a.var_pars) == ['Omega_c', 'h']
    assert np.allclose(a.x, [0.2, 0.7])
    assert len(a.par_bounds) == 0


def test_unpack_var_pars_from_fiducial():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7}
    extra_cfg = {'var_pars': ['Omega_c', 'h']}
    a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    assert list(a.var_pars) == ['Omega_c', 'h']
    assert np.allclose(a.x, [0.2, 0.7])


def test_unpack_var_pars_from_req_params():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {'var_pars': ['Omega_c', 'test_param', 'h']}
    config = {'fisher': extra_cfg}
    lk = DummyLikelihood()
    tools = DummyTools(pars)
    req_params = {'test_param': 0.5}
    a = Analyze(config, likelihood=lk, tools=tools, req_params=req_params)
    assert list(a.var_pars) == ['Omega_c', 'test_param', 'h']
    assert np.allclose(a.x, [0.2, 0.5, 0.7])


def test_unpack_var_pars_unknown_parameter_raises():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {'var_pars': ['Omega_c', 'unknown_param']}
    config = {'fisher': extra_cfg}
    lk = DummyLikelihood()
    tools = DummyTools(pars)
    req_params = {}
    with pytest.raises(ValueError):
        Analyze(config, likelihood=lk, tools=tools, req_params=req_params)


def test_unpack_parameters_and_var_pars_precedence_warns():
    pars = {'Omega_c': 0.2, 'h': 0.7}
    extra_cfg = {
        'parameters': {'Omega_c': 0.2},
        'var_pars': ['h']
    }
    with pytest.warns(UserWarning):
        a = make_analyze(['h'], pars, extra_fisher_cfg=extra_cfg)
    # When both parameters and var_pars are present, parameters takes precedence
    # and a warning is issued indicating var_pars is ignored
    assert list(a.var_pars) == ['Omega_c']


# Tests for _validate_amplitude_in_var_pars
def test_validate_amplitude_sigma8_active_and_varied():
    # sigma8 is the active amplitude (non-None), can be varied
    pars = {'Omega_c': 0.2, 'h': 0.7, 'sigma8': 0.8, 'A_s': None}
    a = make_analyze(['Omega_c', 'sigma8'], pars)  # Should not raise
    assert 'sigma8' in list(a.var_pars)


def test_validate_amplitude_As_active_and_varied():
    # A_s is the active amplitude (non-None), can be varied
    pars = {'Omega_c': 0.2, 'h': 0.7, 'sigma8': None, 'A_s': 2.1e-9}
    a = make_analyze(['Omega_c', 'A_s'], pars)  # Should not raise
    assert 'A_s' in list(a.var_pars)


def test_validate_amplitude_both_in_var_pars_raises():
    # Both sigma8 and A_s in var_pars is always wrong
    pars = {'Omega_c': 0.2, 'h': 0.7, 'sigma8': 0.8, 'A_s': 2.1e-9}
    with pytest.raises(ValueError, match='mutually exclusive'):
        make_analyze(['Omega_c', 'sigma8', 'A_s'], pars)


def test_validate_amplitude_sigma8_varied_but_inactive_raises():
    # sigma8 is in var_pars but fiducial was built with A_s (sigma8 is None)
    pars = {'Omega_c': 0.2, 'h': 0.7, 'sigma8': None, 'A_s': 2.1e-9}
    with pytest.raises(ValueError, match='sigma8.*A_s|A_s.*sigma8'):
        make_analyze(['Omega_c', 'sigma8'], pars)


def test_validate_amplitude_As_varied_but_inactive_raises():
    # A_s is in var_pars but fiducial was built with sigma8 (A_s is None)
    pars = {'Omega_c': 0.2, 'h': 0.7, 'sigma8': 0.8, 'A_s': None}
    with pytest.raises(ValueError, match='A_s.*sigma8|sigma8.*A_s'):
        make_analyze(['Omega_c', 'A_s'], pars)


def test_validate_amplitude_in_var_pars_via_parameters_key_both_raises():
    # Same check applies when using the 'parameters' config key
    pars = {'Omega_c': 0.2, 'h': 0.7, 'sigma8': 0.8, 'A_s': None}
    extra_cfg = {'parameters': {'sigma8': 0.8, 'A_s': 2.1e-9, 'Omega_c': 0.2}}
    with pytest.raises(ValueError, match='mutually exclusive'):
        make_analyze([], pars, extra_fisher_cfg=extra_cfg)


def test_validate_amplitude_in_var_pars_via_parameters_key_As_inactive_raises():
    # Using 'parameters' path: A_s is None in fiducial, cannot vary it
    pars = {'Omega_c': 0.2, 'h': 0.7, 'sigma8': 0.8, 'A_s': None}
    extra_cfg = {'parameters': {'A_s': 2.1e-9, 'Omega_c': 0.2}}
    with pytest.raises(ValueError, match='A_s'):
        make_analyze([], pars, extra_fisher_cfg=extra_cfg)


# Tests for _unpack_mg_parameters
def test_unpack_mg_parameters_musigma_dict():
    pars = {
        'Omega_c': 0.2,
        'h': 0.7,
        'mg_parametrization': {
            'mu_Sigma': {
                'mu_0': 0.1,
                'sigma_0': 0.2,
                'c1_mg': 1.1,
                'c2_mg': 1.2,
                'lambda_mg': 0.5
            }
        }
    }
    extra_cfg = {'var_pars': ['Omega_c', 'h']}
    with pytest.warns(UserWarning):  # MG is experimental
        a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    assert a.pars_fid['mg_musigma_mu'] == 0.1
    assert a.pars_fid['mg_musigma_sigma'] == 0.2
    assert a.pars_fid['mg_musigma_c1'] == 1.1
    assert a.pars_fid['mg_musigma_c2'] == 1.2
    assert a.pars_fid['mg_musigma_lambda0'] == 0.5


def test_unpack_mg_parameters_musigma_defaults():
    pars = {
        'Omega_c': 0.2,
        'h': 0.7,
        'mg_parametrization': {
            'mu_Sigma': {}  # Empty dict should use defaults
        }
    }
    extra_cfg = {'var_pars': ['Omega_c', 'h']}
    with pytest.warns(UserWarning):
        a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    assert a.pars_fid['mg_musigma_mu'] == 0.0
    assert a.pars_fid['mg_musigma_sigma'] == 0.0
    assert a.pars_fid['mg_musigma_c1'] == 1.0
    assert a.pars_fid['mg_musigma_c2'] == 1.0
    assert a.pars_fid['mg_musigma_lambda0'] == 0.0


def test_unpack_mg_parameters_none():
    pars = {'Omega_c': 0.2, 'h': 0.7, 'mg_parametrization': None}
    extra_cfg = {'var_pars': ['Omega_c', 'h']}
    a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    # When mg_parametrization is None, _unpack_mg_parameters doesn't do anything
    # So mg_parametrization should still be in pars_fid as None
    assert a.pars_fid['mg_parametrization'] is None


# Tests for _unpack_baryonic_parameters
def test_unpack_baryonic_parameters_warns():
    pars = {'Omega_c': 0.2, 'h': 0.7, 'baryonic_effects': {'baryon_model': 'some_model'}}
    extra_cfg = {'var_pars': ['Omega_c', 'h']}
    with pytest.warns(UserWarning):
        a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    # baryonic_effects should be removed from pars_fid
    assert 'baryonic_effects' not in a.pars_fid


def test_unpack_baryonic_parameters_none():
    pars = {'Omega_c': 0.2, 'h': 0.7, 'baryonic_effects': None}
    extra_cfg = {'var_pars': ['Omega_c', 'h']}
    # When baryonic_effects is None, no warning is issued, it's just removed
    a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg=extra_cfg)
    # baryonic_effects should be removed from pars_fid when it's None
    assert 'baryonic_effects' not in a.pars_fid


def test_f_handles_list_input_for_x():
    pars = {'Omega_c': 0.25, 'A_s': 1e-9, 'mass_split': 0.0}
    sys_fid = {}
    a = make_analyze(['Omega_c'], pars)

    captured = {}

    def fake_compute_new_theory_vector(sys_pars, pars_dict):
        captured['pars'] = pars_dict
        return np.array([0.0])
    a.compute_new_theory_vector = fake_compute_new_theory_vector

    labels = ['Omega_c']
    x = [0.3]  # List input
    a.f(x, labels, pars, sys_fid)

    assert captured['pars']['Omega_c'] == 0.3


def test_f_with_parameters_only_in_pars_fid():
    pars = {'Omega_c': 0.25, 'sigma8': 0.8, 'A_s': 1e-9, 'mass_split': 0.0}
    sys_fid = {}
    a = make_analyze(['Omega_c', 'sigma8'], pars)

    captured = {}

    def fake_compute_new_theory_vector(sys_pars, pars_dict):
        captured['sys_pars'] = sys_pars
        captured['pars'] = pars_dict
        return np.array([0.0])
    a.compute_new_theory_vector = fake_compute_new_theory_vector

    labels = ['Omega_c', 'sigma8']
    x = np.array([0.3, 0.9])
    a.f(x, labels, pars, sys_fid)

    assert captured['sys_pars'] == {}
    assert captured['pars']['Omega_c'] == 0.3
    assert captured['pars']['sigma8'] == 0.9


def test_f_with_parameters_only_in_sys_fid():
    pars = {'A_s': 1e-9, 'mass_split': 0.0}
    sys_fid = {'sys1': 0.1, 'sys2': 0.2}
    a = make_analyze(['sys1', 'sys2'], pars, req_params=sys_fid)

    captured = {}

    def fake_compute_new_theory_vector(sys_pars, pars_dict):
        captured['sys_pars'] = sys_pars
        captured['pars'] = pars_dict
        return np.array([0.0])
    a.compute_new_theory_vector = fake_compute_new_theory_vector

    labels = ['sys1', 'sys2']
    x = np.array([0.15, 0.25])
    a.f(x, labels, pars, sys_fid)

    assert captured['sys_pars']['sys1'] == 0.15
    assert captured['sys_pars']['sys2'] == 0.25
    assert captured['pars'] == pars  # pars unchanged


def test_f_with_mixed_parameters():
    pars = {'Omega_c': 0.25, 'A_s': 1e-9, 'mass_split': 0.0}
    sys_fid = {'sys1': 0.1}
    a = make_analyze(['Omega_c', 'sys1'], pars, req_params=sys_fid)

    captured = {}

    def fake_compute_new_theory_vector(sys_pars, pars_dict):
        captured['sys_pars'] = sys_pars
        captured['pars'] = pars_dict
        return np.array([0.0])
    a.compute_new_theory_vector = fake_compute_new_theory_vector

    labels = ['Omega_c', 'sys1']
    x = np.array([0.3, 0.15])
    a.f(x, labels, pars, sys_fid)

    assert captured['sys_pars']['sys1'] == 0.15
    assert captured['pars']['Omega_c'] == 0.3


def test_f_raises_for_unrecognized_parameter():
    pars = {'Omega_c': 0.25, 'A_s': 1e-9, 'mass_split': 0.0}
    sys_fid = {}
    a = make_analyze(['Omega_c'], pars)

    with pytest.raises(ValueError, match='Parameter name unknown_param not recognized!'):
        a.f(np.array([0.3]), ['unknown_param'], pars, sys_fid)


def test_f_2d_input_with_wrong_shape_raises():
    pars = {'Omega_c': 0.25, 'A_s': 1e-9, 'mass_split': 0.0}
    sys_fid = {}
    a = make_analyze(['Omega_c'], pars)

    with pytest.raises(ValueError,
                       match='The labels should have the same length as the parameters!'):
        x = np.array([[0.3, 0.2], [0.4, 0.3]])  # Wrong shape for 1 label
        a.f(x, ['Omega_c'], pars, sys_fid)


def test_f_2d_input_returns_correct_shape():
    pars = {'Omega_c': 0.25, 'A_s': 1e-9, 'mass_split': 0.0}
    sys_fid = {}
    a = make_analyze(['Omega_c'], pars)

    captured = []

    def fake_compute_new_theory_vector(sys_pars, pars_dict):
        captured.append((sys_pars, pars_dict))
        return np.array([1.0])
    a.compute_new_theory_vector = fake_compute_new_theory_vector

    labels = ['Omega_c']
    x = np.array([[0.3]])  # 1x1 for 1 parameter
    result = a.f(x, labels, pars, sys_fid)

    assert result.shape == (1, 1)
    assert len(captured) == 1
