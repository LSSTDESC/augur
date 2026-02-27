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


def make_analyze(var_pars, pars_fid, extra_fisher_cfg=None):
    fisher_cfg = {'var_pars': var_pars, 'output': 'out', 'fid_output': 'fid'}
    if extra_fisher_cfg:
        fisher_cfg.update(extra_fisher_cfg)
    config = {'fisher': fisher_cfg}
    lk = DummyLikelihood()
    tools = DummyTools(pars_fid)
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
