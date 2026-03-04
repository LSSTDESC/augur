import copy
import pytest

from augur.utils import firecrown_interface as fci


def test_create_cM_relation_none():
    cfg = {}
    cfg_copy = copy.deepcopy(cfg)
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
