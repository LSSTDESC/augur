import importlib.util
import os
import warnings
import numpy as np
import pyccl as ccl
import pytest

from augur.utils.config_io import write_fisher_to_file
from augur.analyze import (Analyze, _sum_mnu, _dsum_dpar, _neutrino_mass_floor,
                           _derivative_probe_drop)

# derivkit is optional (not in environment.yml); tests that need it are skipped without it.
requires_derivkit = pytest.mark.skipif(importlib.util.find_spec('derivkit') is None,
                                       reason='derivkit not installed')


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


def test_sum_mnu_every_convention():
    # CCL's conventions come in two shapes: sum/normal/inverted/equal/single store
    # Sum(m_nu) itself, 'list' stores one mass per species. Both must reduce to the
    # same float without anyone having to look at mass_split.
    assert _sum_mnu(0.06) == pytest.approx(0.06)
    assert _sum_mnu([0.05, 0.01, 0.0]) == pytest.approx(0.06)
    assert _sum_mnu(np.array([0.05, 0.01, 0.0])) == pytest.approx(0.06)
    assert _sum_mnu(0.0) == 0.0
    assert _sum_mnu([]) == 0.0
    assert isinstance(_sum_mnu(np.float64(0.06)), float)


def test_get_Om_list_mnu():
    # Regression for analyze.py:329. Under mass_split 'list' the fiducial cosmology
    # carries one mass per species, and `m_nu > 0.0` raised
    # `TypeError: '>' not supported between instances of 'list' and 'float'`
    # before the masses could be summed. The masses are held fixed here rather than
    # varied: varying m_nu under 'list' is a separate, invalid configuration.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': [0.05, 0.01, 0.0]}
    a = make_analyze(['Omega_c', 'Omega_b', 'h'], pars)
    expected = 0.25 + 0.06 / 0.7 / 0.7 / 93.14
    assert pytest.approx(a.get_Om(), rel=1e-6) == expected


def test_get_Om_list_mnu_matches_scalar():
    # The same total mass, written either way, has to give the same Omega_m.
    scalar = make_analyze(['Omega_c', 'Omega_b', 'h'],
                          {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.06})
    listed = make_analyze(['Omega_c', 'Omega_b', 'h'],
                          {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7,
                           'm_nu': [0.02, 0.02, 0.02]})
    assert pytest.approx(listed.get_Om(), rel=1e-12) == scalar.get_Om()


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


def test_dsum_dpar_scalar_mnu():
    # Every scalar split stores the total mass itself, so dSum/dm_nu is 1 by definition.
    assert _dsum_dpar('m_nu') == 1.0


def test_dsum_dpar_rejects_non_neutrino_parameter():
    # A lightest-mass parametrization would register its own parameter here; until one
    # exists, anything else is a caller error rather than a silent zero derivative.
    for par in ['m_nu_lightest', 'Omega_c', 'h']:
        with pytest.raises(ValueError, match='total neutrino mass'):
            _dsum_dpar(par)


def test_Jacobian_h_term_survives_fixed_mnu():
    # Regression: Omega_m depends on h through Omega_nu = Sum(m_nu)/(93.14 h^2) whether
    # or not the masses are varied. The h term used to be nested inside the m_nu guard,
    # so holding the masses fixed dropped it silently.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.06, 'sigma8': 0.8}
    var_pars = ['Omega_c', 'Omega_b', 'h', 'sigma8']
    a = make_analyze(var_pars, pars, extra_fisher_cfg={'transform_Omega_m': True})
    J = a.Jacobian_transform()
    ind_c, ind_h = var_pars.index('Omega_c'), var_pars.index('h')
    expected = 2.0 * 0.06 / (0.7 ** 3 * 93.14)
    assert pytest.approx(J[ind_c][ind_h], rel=1e-8) == expected


def test_Jacobian_h_term_survives_fixed_mnu_S8_row():
    # The same hole in the sigma8 row, reachable when only S8 is transformed.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.06, 'sigma8': 0.8}
    var_pars = ['Omega_c', 'Omega_b', 'h', 'sigma8']
    a = make_analyze(var_pars, pars, extra_fisher_cfg={'transform_S8': True})
    J = a.Jacobian_transform()
    ind_8, ind_h = var_pars.index('sigma8'), var_pars.index('h')
    expected = 0.8 * 0.06 / (a.get_Om() * 0.7 ** 3 * 93.14)
    assert pytest.approx(J[ind_8][ind_h], rel=1e-8) == expected


def test_Jacobian_neutrino_terms_absent_when_massless():
    # No neutrino mass, no neutrino coupling: h must not pick up a spurious entry.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.0, 'sigma8': 0.8}
    var_pars = ['Omega_c', 'Omega_b', 'h', 'sigma8']
    a = make_analyze(var_pars, pars, extra_fisher_cfg={'transform_Omega_m': True})
    J = a.Jacobian_transform()
    ind_c, ind_h = var_pars.index('Omega_c'), var_pars.index('h')
    assert J[ind_c][ind_h] == 0.0


def test_Jacobian_list_mnu_matches_scalar():
    # mass_split 'list' cannot vary its masses, so it always takes the fixed-mass path
    # above. The same total mass, written either way, must give the same Jacobian.
    var_pars = ['Omega_c', 'Omega_b', 'h', 'sigma8']
    common = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'sigma8': 0.8}
    cfg = {'transform_Omega_m': True}
    scalar = make_analyze(var_pars, {**common, 'm_nu': 0.06}, extra_fisher_cfg=cfg)
    listed = make_analyze(var_pars, {**common, 'm_nu': [0.02, 0.02, 0.02]},
                          extra_fisher_cfg=cfg)
    np.testing.assert_allclose(listed.Jacobian_transform(),
                               scalar.Jacobian_transform(), rtol=1e-12)


def test_neutrino_mass_floor_matches_ccl_constants():
    # The floor is where the lightest mass reaches zero, built from CCL's own
    # oscillation constants so it tracks them if they are ever updated.
    c = ccl.physical_constants
    nh = np.sqrt(c.DELTAM12_sq) + np.sqrt(c.DELTAM13_sq_pos)
    d13 = abs(c.DELTAM13_sq_neg)
    ih = np.sqrt(d13) + np.sqrt(d13 + c.DELTAM12_sq)
    assert pytest.approx(_neutrino_mass_floor('normal'), rel=1e-12) == nh
    assert pytest.approx(_neutrino_mass_floor('inverted'), rel=1e-12) == ih
    # The degenerate splits have no hierarchy, but masses still cannot be negative.
    for split in ['equal', 'single', 'sum']:
        assert _neutrino_mass_floor(split) == 0.0
    # 'list' has no scalar to step, and an unspecified split is unknown.
    for split in ['list', None]:
        assert _neutrino_mass_floor(split) is None


def test_ccl_is_silently_flat_below_zero_mass():
    # Why the degenerate floor is 0 and not None: CCL accepts a negative total under
    # 'equal', clamps Omega_nu to zero and raises nothing, so the theory below zero is
    # the massless one. A central difference straddling 0 then halves the derivative.
    # Reported upstream as https://github.com/LSSTDESC/CCL/issues/1324; revisit this
    # test once CCL warns or raises.
    def pk(m_nu):
        c = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, sigma8=0.8,
                          m_nu=m_nu, mass_split='equal',
                          transfer_function='eisenstein_hu')
        return ccl.linear_matter_power(c, 0.5, 1.0)
    assert pk(-0.006) == pk(0.0)
    assert pk(0.006) != pk(0.0)


def test_neutrino_mass_floor_is_where_ccl_stops_being_physical():
    # CCL raises at the right place for the normal hierarchy, but for inverted it
    # raises 1.5 meV too low and quietly returns a negative lightest mass in between.
    # The guard must not delegate to it.
    for split in ['normal', 'inverted']:
        floor = _neutrino_mass_floor(split)
        # At the floor the lightest species is massless; just above it, all positive.
        assert np.min(ccl.nu_masses(m_nu=floor*(1+1e-9), mass_split=split)) >= 0.0

    # Normal: CCL refuses below the floor, which is the correct behaviour.
    with pytest.raises(ValueError):
        ccl.nu_masses(m_nu=_neutrino_mass_floor('normal')*(1-1e-6), mass_split='normal')

    # Inverted: CCL only refuses ~1.5 meV lower, and in between returns a negative
    # lightest mass with no error at all. That window is why the floor is enforced here.
    ih = _neutrino_mass_floor('inverted')
    assert np.min(ccl.nu_masses(m_nu=ih*(1-1e-6), mass_split='inverted')) < 0.0
    assert np.min(ccl.nu_masses(m_nu=0.0985, mass_split='inverted')) < 0.0
    with pytest.raises(ValueError):
        ccl.nu_masses(m_nu=0.0977, mass_split='inverted')


def test_derivative_probe_drop_per_method():
    # five_pt_stencil samples x0 - 2h; numdifftools' step can be widened through
    # derivative_args, so it takes the same margin.
    assert _derivative_probe_drop('5pt_stencil', 0.006, {}, 0.12) == pytest.approx(0.012)
    assert _derivative_probe_drop('numdifftools', 0.006, {}, 0.12) == pytest.approx(0.012)
    pytest.importorskip('derivkit')
    # derivkit ignores `step` entirely: its half-width is max(frac*|x0|, base_abs).
    assert _derivative_probe_drop('derivkit', 0.006, {}, 0.12) == pytest.approx(0.0012)
    assert _derivative_probe_drop('derivkit', 0.006, {}, 0.06) == pytest.approx(0.001)
    assert _derivative_probe_drop('derivkit', 0.006, {'spacing': '5%'}, 0.12) \
        == pytest.approx(0.006)


@requires_derivkit
def test_derivative_probe_drop_reads_per_parameter_base_abs():
    # With base_abs given per parameter, the guard must use the floor the m_nu column
    # is actually built with, not the default and not another parameter's value.
    args = {'spacing': '1%', 'base_abs': {'m_nu': 5e-3, 'default': 1e-12}}
    assert _derivative_probe_drop('derivkit', 0.006, args, 0.06, par='m_nu') \
        == pytest.approx(5e-3)
    args = {'spacing': '1%', 'base_abs': {'A_s': 1e-12, 'default': 1e-3}}
    assert _derivative_probe_drop('derivkit', 0.006, args, 0.06, par='m_nu') \
        == pytest.approx(1e-3)


@requires_derivkit
def test_per_parameter_base_abs_below_floor_raises():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.004,
            'mass_split': 'equal'}
    cfg = {'derivative_method': 'derivkit',
           'derivative_args': {'spacing': '1%',
                               'base_abs': {'m_nu': 5e-3, 'default': 1e-12}}}
    with pytest.raises(ValueError, match='admits no total neutrino mass'):
        make_analyze(['Omega_c', 'm_nu'], pars, extra_fisher_cfg=cfg)


def test_varying_list_valued_mnu_raises_clearly():
    # Regression: this used to die in the pivot-vector cast with a numpy message
    # naming neither m_nu nor the split.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': [0.05, 0.01, 0.0],
            'mass_split': 'list'}
    with pytest.raises(ValueError, match='m_nu'):
        make_analyze(['Omega_c', 'm_nu'], pars)


def test_step_below_hierarchy_floor_raises():
    # Sum = 0.06 under the normal hierarchy: a 0.006 step reaches 0.048, well under
    # the 0.0592 floor, and CCL would abort the run partway through the derivatives.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.06,
            'mass_split': 'normal'}
    with pytest.raises(ValueError, match='admits no total neutrino mass'):
        make_analyze(['Omega_c', 'm_nu'], pars, extra_fisher_cfg={'step': 0.006})


def test_step_above_hierarchy_floor_is_accepted():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.12,
            'mass_split': 'normal'}
    a = make_analyze(['Omega_c', 'm_nu'], pars, extra_fisher_cfg={'step': 0.006})
    assert a.var_pars == ['Omega_c', 'm_nu']


@requires_derivkit
def test_derivkit_step_below_floor_raises():
    # The case a `Sum - 2*step` rule misses: derivkit never reads `step`, and its
    # 1e-3 default floor on the half-width reaches 0.059 from a 0.06 fiducial.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.06,
            'mass_split': 'normal'}
    cfg = {'derivative_method': 'derivkit', 'step': 1e-8}
    with pytest.raises(ValueError, match='admits no total neutrino mass'):
        make_analyze(['Omega_c', 'm_nu'], pars, extra_fisher_cfg=cfg)


@requires_derivkit
def test_inverted_floor_guards_ccl_silent_window():
    # Sum = 0.0995 with derivkit reaches 0.0985: CCL does not raise there, it returns
    # a negative lightest mass. Enforcing the true floor is the whole point.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.0995,
            'mass_split': 'inverted'}
    cfg = {'derivative_method': 'derivkit', 'step': 1e-8}
    with pytest.raises(ValueError, match='admits no total neutrino mass'):
        make_analyze(['Omega_c', 'm_nu'], pars, extra_fisher_cfg=cfg)


def test_degenerate_split_step_below_zero_raises():
    # Regression: Sum = 0.01 under 'equal' with a 0.006 step reaches -0.002, where CCL
    # silently returns the massless theory. This used to pass unchecked.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.01,
            'mass_split': 'equal'}
    with pytest.raises(ValueError, match='admits no total neutrino mass'):
        make_analyze(['Omega_c', 'm_nu'], pars, extra_fisher_cfg={'step': 0.006})


def test_degenerate_split_small_fiducial_accepted():
    # No hierarchy, so a fiducial far below the normal floor is fine while every
    # sample stays non-negative: 0.02 - 2*0.006 = 0.008.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.02,
            'mass_split': 'equal'}
    a = make_analyze(['Omega_c', 'm_nu'], pars, extra_fisher_cfg={'step': 0.006})
    assert a.var_pars == ['Omega_c', 'm_nu']


def test_no_floor_check_when_mnu_is_not_varied():
    # Nothing steps the masses, so the floor cannot be crossed.
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.06,
            'mass_split': 'normal'}
    a = make_analyze(['Omega_c', 'h'], pars, extra_fisher_cfg={'step': 0.006})
    assert 'm_nu' not in a.var_pars


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
    pars = {'Omega_c': 0.25, 'A_s': 1e-9, 'mass_split': 0.0, 'h': 0.7}
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

    labels = ['Omega_c', 'h']
    captured = []
    a = make_analyze(labels, pars)
    a.compute_new_theory_vector = fake_compute_new_theory_vector
    x = np.array([[0.3, 0.7],
                  [0.3, 0.7]])  # 2x2 for 2 parameters
    result = a.f(x, labels, pars, sys_fid)

    assert result.shape == (2, 1)
    assert len(captured) == 2


@pytest.mark.parametrize('m_nu, split, var_pars', [
    (-0.01, 'equal', ['Omega_c']),             # scalar, masses fixed
    (-0.01, 'equal', ['Omega_c', 'm_nu']),     # scalar, varied
    ([0.02, 0.02, -0.01], 'list', ['Omega_c']),  # CCL drops the negative species
])
def test_negative_fiducial_mnu_raises(m_nu, split, var_pars):
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': m_nu, 'mass_split': split}
    with pytest.raises(ValueError, match='negative neutrino masses'):
        make_analyze(var_pars, pars)


def test_zero_mass_in_list_fiducial_warns():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': [0.05, 0.01, 0.0],
            'mass_split': 'list'}
    with pytest.warns(UserWarning, match='zero mass'):
        make_analyze(['Omega_c'], pars)


def test_zero_scalar_mnu_does_not_warn():
    pars = {'Omega_c': 0.2, 'Omega_b': 0.05, 'h': 0.7, 'm_nu': 0.0, 'mass_split': 'equal'}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        make_analyze(['Omega_c'], pars)


def write_augur_fisher(tmp_path, stem, fisher, names, values):
    """Write a Fisher matrix and parameter file the way get_fisher_matrix does."""
    fisher_path = str(tmp_path / f'{stem}.dat')
    fid_path = str(tmp_path / f'{stem}_pars.dat')
    write_fisher_to_file(fisher, names, dict(zip(names, values)), fisher_path, fid_path)
    return fisher_path, fid_path


def random_fisher(n, seed):
    m = np.random.default_rng(seed).normal(size=(3 * n, n))
    return m.T @ m


def make_external_analyze(**extra_cfg):
    # Run parameters: Omega_c, h (cosmology) and lens0_bias (nuisance), in that order.
    return make_analyze(['Omega_c', 'h', 'lens0_bias'], {'Omega_c': 0.25, 'h': 0.7},
                        extra_fisher_cfg=extra_cfg, req_params={'lens0_bias': 1.5})


def test_external_fisher_is_embedded_by_name(tmp_path):
    a = make_external_analyze()
    a.Fij = random_fisher(3, 0)
    F_ext = random_fisher(2, 1)
    paths = write_augur_fisher(tmp_path, 'ext', F_ext, ['hubble', 'Omega_c'], [0.7, 0.25])
    a.add_external_fisher(*paths, rename={'hubble': 'h'})

    expected = a.Fij.copy()
    expected[np.ix_([1, 0], [1, 0])] += F_ext
    assert np.allclose(a.get_combined_fisher(save_txt=False), expected)
    assert list(a.external_fishers) == [os.path.realpath(paths[0])]


def test_marginalized_external_matches_joint_inversion(tmp_path):
    # Integrating out a parameter only the external has must give the same covariance
    # as inverting the joint matrix over all four parameters.
    a = make_external_analyze()
    a.Fij = random_fisher(3, 2)
    F_ext = random_fisher(3, 3)
    paths = write_augur_fisher(tmp_path, 'ext', F_ext, ['Omega_c', 'h', 'ext_bias'],
                               [0.25, 0.7, 2.0])
    a.add_external_fisher(*paths, marginalize=['ext_bias'])

    joint = np.zeros((4, 4))
    joint[:3, :3] += a.Fij
    joint[np.ix_([0, 1, 3], [0, 1, 3])] += F_ext
    assert np.allclose(np.linalg.inv(a.get_combined_fisher(save_txt=False)),
                       np.linalg.inv(joint)[:3, :3])


def test_marginalizing_unconstrained_parameter_raises(tmp_path):
    a = make_external_analyze()
    paths = write_augur_fisher(tmp_path, 'ext', np.diag([3.0, 2.0, 0.0]),
                               ['Omega_c', 'h', 'ext_bias'], [0.25, 0.7, 2.0])
    with pytest.raises(ValueError, match='barely constrains'):
        a.add_external_fisher(*paths, marginalize=['ext_bias'])


def test_external_parameter_missing_from_run_raises(tmp_path):
    a = make_external_analyze()
    paths = write_augur_fisher(tmp_path, 'ext', random_fisher(2, 4), ['Omega_c', 'w0'],
                               [0.25, -1.0])
    with pytest.raises(ValueError, match='not parameters of this run'):
        a.add_external_fisher(*paths)


def test_external_must_use_the_transformed_basis(tmp_path):
    a = make_analyze(['Omega_c', 'h'], {'Omega_c': 0.25, 'h': 0.7},
                     extra_fisher_cfg={'transform_Omega_m': True})
    old = write_augur_fisher(tmp_path, 'old', random_fisher(2, 5), ['Omega_c', 'h'],
                             [0.25, 0.7])
    with pytest.raises(ValueError, match=r"reports \['Omega_m'\] in place of \['Omega_c'\]"):
        a.add_external_fisher(*old)
    new = write_augur_fisher(tmp_path, 'new', random_fisher(2, 5), ['Omega_m', 'h'],
                             [0.25, 0.7])
    a.add_external_fisher(*new)


def test_external_fiducial_mismatch_warns_and_adds(tmp_path):
    a = make_external_analyze()
    paths = write_augur_fisher(tmp_path, 'ext', random_fisher(2, 6), ['Omega_c', 'h'],
                               [0.26, 0.7])
    with pytest.warns(UserWarning, match='fiducial values differ'):
        a.add_external_fisher(*paths)
    assert len(a.external_fishers) == 1


def test_same_external_cannot_be_added_twice(tmp_path):
    a = make_external_analyze()
    F_ext = random_fisher(3, 7)
    paths = write_augur_fisher(tmp_path, 'ext', F_ext, ['Omega_c', 'h', 'ext_bias'],
                               [0.25, 0.7, 2.0])
    a.add_external_fisher(*paths, marginalize=['ext_bias'])
    with pytest.raises(ValueError, match='already added'):
        a.add_external_fisher(*paths, marginalize=['ext_bias'])
    # A copy elsewhere, with its nuisance parameter renamed, matches on the shared entries.
    copy = write_augur_fisher(tmp_path, 'copy', F_ext, ['Omega_c', 'h', 'other_bias'],
                              [0.25, 0.7, 2.0])
    with pytest.raises(ValueError, match='already added'):
        a.add_external_fisher(*copy, marginalize=['other_bias'])
    # A different analysis of the same parameters is accepted.
    other = write_augur_fisher(tmp_path, 'other', random_fisher(2, 8), ['Omega_c', 'h'],
                               [0.25, 0.7])
    a.add_external_fisher(*other)
    assert len(a.external_fishers) == 2


def test_external_matching_this_run_raises(tmp_path):
    a = make_external_analyze()
    a.Fij = random_fisher(3, 9)
    paths = write_augur_fisher(tmp_path, 'ext', a.Fij[:2, :2], ['Omega_c', 'h'], [0.25, 0.7])
    a.add_external_fisher(*paths)
    with pytest.raises(ValueError, match='matches this run'):
        a.get_combined_fisher(save_txt=False)


def test_own_output_and_prior_files_are_rejected(tmp_path):
    paths = write_augur_fisher(tmp_path, 'fisher', random_fisher(3, 10),
                               ['Omega_c', 'h', 'lens0_bias'], [0.25, 0.7, 1.5])
    a = make_external_analyze(output=paths[0])
    with pytest.raises(ValueError, match='output of this run'):
        a.add_external_fisher(*paths)
    with pytest.raises(ValueError, match='includes priors'):
        a.add_external_fisher(paths[0] + '.with_priors', paths[1])


def test_combined_fisher_excludes_gaussian_priors(tmp_path):
    a = make_external_analyze(gaussian_priors={'h': 0.1})
    a.Fij = random_fisher(3, 11)
    a.add_gaussian_priors(save_txt=False)
    F_ext = random_fisher(2, 12)
    paths = write_augur_fisher(tmp_path, 'ext', F_ext, ['Omega_c', 'h'], [0.25, 0.7])
    a.add_external_fisher(*paths)

    expected = a.Fij.copy()
    expected[:2, :2] += F_ext
    assert np.allclose(a.get_combined_fisher(save_txt=False), expected)


def test_shared_nuisance_parameter_warns(tmp_path):
    a = make_external_analyze()
    paths = write_augur_fisher(tmp_path, 'ext', random_fisher(2, 13), ['h', 'lens0_bias'],
                               [0.7, 1.5])
    with pytest.warns(UserWarning, match='lens0_bias'):
        a.add_external_fisher(*paths)


def test_external_fishers_from_config(tmp_path):
    paths = write_augur_fisher(tmp_path, 'ext', random_fisher(3, 14),
                               ['Omega_c', 'h', 'ext_bias'], [0.25, 0.7, 2.0])
    spec = {'fisher': paths[0], 'fiducials': paths[1], 'marginalize': ['ext_bias']}
    a = make_external_analyze(external_fishers=[spec])
    assert list(a.external_fishers) == [os.path.realpath(paths[0])]


def test_fisher_pars_use_transformed_names():
    pars = {'Omega_c': 0.25, 'sigma8': 0.8, 'h': 0.7}
    a = make_analyze(['Omega_c', 'sigma8', 'h'], pars,
                     extra_fisher_cfg={'transform_Omega_m': True, 'transform_S8': True})
    assert a.fisher_pars == ['Omega_m', 'S8', 'h']
    assert list(a.var_pars) == ['Omega_c', 'sigma8', 'h']  # still the varied parameters
    assert a._fisher_fid_values() == pytest.approx([0.25, a.get_S8(), 0.7])


def test_fisher_pars_without_transforms_are_var_pars():
    a = make_external_analyze()
    assert a.fisher_pars == ['Omega_c', 'h', 'lens0_bias']
    assert a._fisher_fid_values() == pytest.approx([0.25, 0.7, 1.5])
