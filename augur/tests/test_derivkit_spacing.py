import numpy as np
import pytest

import derivkit.calculus_kit
import derivkit.derivative_kit

from augur.analyze import _resolve_base_abs
from augur.tests.test_analyze_helpers import make_analyze


def test_resolve_base_abs_mapping():
    got = _resolve_base_abs({'A_s': 1e-12, 'default': 1e-3}, ['Omega_c', 'A_s', 'wa'])
    assert got == [1e-3, 1e-12, 1e-3]


def test_resolve_base_abs_requires_default():
    with pytest.raises(ValueError, match='default'):
        _resolve_base_abs({'A_s': 1e-12}, ['Omega_c', 'A_s'])


def test_resolve_base_abs_rejects_unknown_parameter():
    # A typo would otherwise fall back to the default without a word.
    with pytest.raises(ValueError, match='A_S'):
        _resolve_base_abs({'A_S': 1e-12, 'default': 1e-3}, ['Omega_c', 'A_s'])


def test_resolve_base_abs_rejects_non_positive():
    with pytest.raises(ValueError, match='positive'):
        _resolve_base_abs({'default': 0.0}, ['Omega_c'])


class _RecordingKit:
    calls = []

    def __init__(self, function, x0):
        self.function, self.x0 = function, x0

    def differentiate(self, **kwargs):
        _RecordingKit.calls.append((self.x0, kwargs, self.function))
        # A marker column, so the stacking order can be checked.
        return np.full(3, float(len(_RecordingKit.calls)))


def test_mapping_routes_base_abs_to_each_parameter(monkeypatch):
    _RecordingKit.calls = []
    monkeypatch.setattr(derivkit.derivative_kit, 'DerivativeKit', _RecordingKit)
    pars = {'Omega_c': 0.25, 'A_s': 2.1e-9, 'wa': 0.0}
    var_pars = ['Omega_c', 'A_s', 'wa']
    cfg = {'derivative_method': 'derivkit',
           'derivative_args': {'method': 'adaptive', 'n_workers': 1, 'spacing': '1%',
                               'base_abs': {'A_s': 1e-12, 'default': 1e-3}}}
    a = make_analyze(var_pars, pars, extra_fisher_cfg=cfg)
    seen = []
    a.f = lambda y, *args, **kwargs: (seen.append(np.array(y, dtype=float)), np.zeros(3))[1]

    d = a.get_derivatives()

    assert [call[1]['base_abs'] for call in _RecordingKit.calls] == [1e-3, 1e-12, 1e-3]
    assert [call[0] for call in _RecordingKit.calls] == pytest.approx([0.25, 2.1e-9, 0.0])
    for _, kwargs, _ in _RecordingKit.calls:
        assert kwargs['spacing'] == '1%' and kwargs['order'] == 1
        assert kwargs['method'] == 'adaptive'
    # One row per parameter, in var_pars order.
    np.testing.assert_array_equal(d[:, 0], [1.0, 2.0, 3.0])
    # Each one-dimensional function moves only its own parameter.
    _RecordingKit.calls[2][2](0.5)
    np.testing.assert_allclose(seen[-1], [0.25, 2.1e-9, 0.5])
    # The config is left as it was, so a second call sees the same settings.
    assert a.derivative_args['base_abs'] == {'A_s': 1e-12, 'default': 1e-3}
    assert a.derivative_args['method'] == 'adaptive'


def test_scalar_base_abs_keeps_the_calculus_kit_path(monkeypatch):
    used = {}

    class _Calc:
        def __init__(self, function, x0):
            used['x0'] = np.array(x0)

        def jacobian(self, **kwargs):
            used['kwargs'] = kwargs
            return np.zeros((3, 2))

    def _no_kit(*args, **kwargs):
        raise AssertionError('per-parameter path taken for a scalar base_abs')

    monkeypatch.setattr(derivkit.calculus_kit, 'CalculusKit', _Calc)
    monkeypatch.setattr(derivkit.derivative_kit, 'DerivativeKit', _no_kit)
    cfg = {'derivative_method': 'derivkit',
           'derivative_args': {'method': 'adaptive', 'n_workers': 1, 'base_abs': 1e-3}}
    a = make_analyze(['Omega_c', 'h'], {'Omega_c': 0.25, 'h': 0.7}, extra_fisher_cfg=cfg)
    a.f = lambda y, *args, **kwargs: np.zeros(3)

    d = a.get_derivatives()

    assert used['kwargs']['base_abs'] == 1e-3
    assert d.shape == (2, 3)


def test_uniform_mapping_matches_scalar_with_real_derivkit():
    def f(y, *args, **kwargs):
        y = np.asarray(y, dtype=float)
        return np.array([np.sin(y[0]) * y[1], y[0] ** 2 + np.exp(y[1]), y[1] ** 3])

    out = {}
    for label, base_abs in [('scalar', 1e-3), ('mapping', {'default': 1e-3})]:
        cfg = {'derivative_method': 'derivkit',
               'derivative_args': {'method': 'adaptive', 'n_workers': 1, 'n_points': 9,
                                   'spacing': '1%', 'base_abs': base_abs, 'ridge': 1e-8}}
        a = make_analyze(['Omega_c', 'h'], {'Omega_c': 0.25, 'h': 0.7}, extra_fisher_cfg=cfg)
        a.f = f
        out[label] = np.array(a.get_derivatives(), dtype=float)

    np.testing.assert_allclose(out['mapping'], out['scalar'], rtol=1e-10, atol=1e-12)
    truth = np.array([[np.cos(0.25) * 0.7, 2 * 0.25, 0.0],
                      [np.sin(0.25), np.exp(0.7), 3 * 0.7 ** 2]])
    np.testing.assert_allclose(out['mapping'], truth, rtol=1e-4, atol=1e-8)
