from copy import deepcopy
import numpy as np
import pyccl as ccl
import pytest
from pyccl.neutrinos import NeutrinoMassSplits
from firecrown.parameters import ParamsMap
from firecrown.updatable import get_default_params_map

from augur.utils import firecrown_interface as fci


def test_create_cM_relation_none():
    cfg = {}
    cfg_copy = deepcopy(cfg)
    assert fci._create_cM_relation(cfg_copy) is None


def test_create_cM_relation_invalid_type():
    cfg = {'cM_relation': {'type': 'not_a_string'}}
    with pytest.raises(ValueError):
        fci._create_cM_relation(cfg)


def test_create_pt_calculator_none():
    cfg = {}
    assert fci._create_pt_calculator(cfg, cosmo=None) is None


def test_create_pt_calculator_unknown_type():
    cfg = {'pt_calculator': {'type': 'unknown_type'}}
    with pytest.raises(ValueError):
        fci._create_pt_calculator(cfg, cosmo=None)


def test_create_pt_calculator_typeerror(monkeypatch):
    # Monkeypatch registry to point to a class that raises TypeError on init
    class BadCalc:
        def __init__(self, **kwargs):
            raise TypeError('bad init')

    monkeypatch.setitem(fci.PT_CALCULATOR_REGISTRY, 'bad_calc', BadCalc)
    cfg = {'pt_calculator': {'type': 'bad_calc', 'foo': 1}}
    with pytest.raises(ValueError):
        fci._create_pt_calculator(cfg, cosmo=None)


def test_create_hm_calculator_none():
    cfg = {}
    assert fci._create_hm_calculator(cfg, cosmo=None) is None


def test_create_ccl_factory_missing_amplitude():
    cfg = {'cosmo': {'Omega_c': 0.25}}
    with pytest.raises(ValueError):
        fci._create_ccl_factory(cfg)


def test_create_ccl_factory_camb_extra_requires_halofit(monkeypatch):
    # Prevent CAMBExtraParams from doing heavy work
    monkeypatch.setattr(fci, 'CAMBExtraParams', lambda **kwargs: object())
    cfg = {'cosmo': {'transfer_function': 'boltzmann_camb', 'A_s': 1e-9,
                     'extra_parameters': {'camb': {'some_param': 1}}}}
    with pytest.raises(ValueError):
        fci._create_ccl_factory(cfg)


def test_load_likelihood_from_yaml_errors():
    # Missing Firecrown_Factory
    cfg = {}
    with pytest.raises(ValueError):
        fci.load_likelihood_from_yaml(cfg, ccl_factory=None, S=None)

    # Multiple keys
    cfg = {'Firecrown_Factory': {'A': {}}, 'Other': {}}
    with pytest.raises(NameError):
        fci.load_likelihood_from_yaml(cfg, ccl_factory=None, S=None)

    # Invalid factory name
    cfg = {'Firecrown_Factory': {'NoSuchFactory': {}}}
    with pytest.raises(NameError):
        fci.load_likelihood_from_yaml(cfg, ccl_factory=None, S=None)


def test_create_twopoint_filter_unknown():
    with pytest.raises(ValueError):
        fci.create_twopoint_filter('unknown_combo', 'a', 'b', 1.0, 10.0)


# --- neutrino mass split plumbing -------------------------------------------------------
#
# CCLFactory rebuilds its cosmology from its own fields on every prepare(); mass_split is a
# frozen field there, so it has to be passed at construction or the factory runs 'normal'
# whatever the config says. eisenstein_hu + sigma8 keeps these tests CAMB-free.

def _nu_cfg(**cosmo_overrides):
    cosmo = {'Omega_c': 0.25, 'Omega_b': 0.05, 'h': 0.67, 'n_s': 0.96, 'sigma8': 0.8,
             'Omega_k': 0.0, 'w0': -1.0, 'wa': 0.0, 'Neff': 3.044, 'T_CMB': 2.7255,
             'transfer_function': 'eisenstein_hu', 'matter_power_spectrum': 'linear'}
    cosmo.update(cosmo_overrides)
    return {'cosmo': cosmo}


def _species_masses(cosmo_dict):
    return np.sort(ccl.nu_masses(m_nu=cosmo_dict['m_nu'], mass_split=cosmo_dict['mass_split']))


def test_mass_split_plumbed():
    """The cosmology the factory builds carries the split the config asked for."""
    tools, cosmo = fci.create_modeling_tools(_nu_cfg(m_nu=0.06, mass_split='equal'))
    assert tools.ccl_factory.mass_split is NeutrinoMassSplits.EQUAL

    # The default map carries m_nu = 0 (CosmologyVanillaLCDM), for which every split is
    # the same; write the config's value in so the comparison below can actually fail.
    defaults = get_default_params_map(tools)
    values = {k: defaults.get_from_full_name(k) for k in defaults.keys()}
    pmap = ParamsMap({**values, 'm_nu': 0.06})
    tools.update(pmap)
    tools.prepare()
    built = tools.get_ccl_cosmology().to_dict()
    asked = cosmo.to_dict()

    assert built['mass_split'] == asked['mass_split'] == 'equal'
    np.testing.assert_allclose(_species_masses(built), _species_masses(asked), rtol=1e-12)


def test_mass_split_list_registers_species():
    tools, _ = fci.create_modeling_tools(_nu_cfg(m_nu=[0.02, 0.02, 0.02], mass_split='list'))
    factory = tools.ccl_factory
    assert factory.mass_split is NeutrinoMassSplits.LIST
    assert factory.num_neutrino_masses == 3
    assert {'m_nu', 'm_nu_2', 'm_nu_3'} <= set(get_default_params_map(tools).keys())


def test_mass_split_reaches_mu_sigma_branch():
    """The early return for modified gravity builds its own factory; it must get the split too.
    Field-level only: creating the cosmology on this branch needs isitgr."""
    mg = {'mu_Sigma': {'mu_0': 0.0, 'sigma_0': 0.0, 'c1_mg': 1.0, 'c2_mg': 1.0, 'lambda_mg': 0.0}}
    factory, _ = fci._create_ccl_factory(_nu_cfg(m_nu=0.06, mass_split='single',
                                                 mg_parametrization=mg))
    assert factory.creation_mode is fci.CCLCreationMode.MU_SIGMA_ISITGR
    assert factory.mass_split is NeutrinoMassSplits.SINGLE


def test_mass_split_sum_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        fci._create_ccl_factory(_nu_cfg(m_nu=0.06, mass_split='sum'))
