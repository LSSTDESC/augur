"""Per-parameter derivative steps and `norm_step` from the config (augur#151)."""
import warnings

import numpy as np
import pytest

from augur.analyze import Analyze
from augur.utils.diff_utils import five_pt_stencil
from augur.tests.test_analyze_helpers import DummyTools

AMP = 2.1e-9
BOUNDS = {'Omega_c': [0.06, 0.27, 0.46], 'A_s': [1.e-9, AMP, 3.e-9],
          'wa': [-2.0, 0.0, 2.0], 'dz': [-0.05, 0.0, 0.05]}
X0 = np.array([0.27, AMP, 0.0, 0.0])
WIDTH = np.array([v[-1] - v[0] for v in BOUNDS.values()])
# Derivative method and args, and how far out each samples in steps.
METHODS = {'5pt_stencil': ('5pt_stencil', {}, 2.0), 'numdifftools': ('numdifftools', {}, 1.0),
           'adaptive': ('derivkit', {}, None)}


def toy(p):
    """A theory vector with very different parameter scales and a known Jacobian."""
    oc, a_s, wa, dz = p
    r = a_s / AMP
    return np.array([oc**2 * r, np.exp(0.4 * wa) * r, (1 + wa)**3 * oc, np.exp(5 * dz) * r])


def toy_jacobian(p):
    oc, a_s, wa, dz = p
    r = a_s / AMP
    return np.array([[2 * oc * r, 0, (1 + wa)**3, 0],
                     [oc**2 / AMP, np.exp(0.4 * wa) / AMP, 0, np.exp(5 * dz) / AMP],
                     [0, 0.4 * np.exp(0.4 * wa) * r, 3 * (1 + wa)**2 * oc, 0],
                     [0, 0, 0, 5 * np.exp(5 * dz) * r]])


class Toy(Analyze):
    """Analyze whose theory vector is `toy`, recording every point it is evaluated at."""

    def compute_new_theory_vector(self, _sys_pars, _pars):
        p = np.array([_pars['Omega_c'], _pars['A_s'], _pars['wa'], _sys_pars['dz']])
        self.samples.append(p)
        return toy(p)


class ToyLikelihood:
    def get_data_vector(self):
        return np.zeros(4)


def build(method='5pt_stencil', keyword=None, bounds=BOUNDS, **fisher):
    """Toy Analyze; `fisher` goes into the config, `keyword` is its norm_step argument."""
    name, args, _ = METHODS[method]
    if 'derivkit' in name:
        pytest.importorskip('derivkit')
    cfg = dict({'output': 'out', 'fid_output': 'fid', 'derivative_method': name,
                'derivative_args': dict(args), 'parameters': dict(bounds)}, **fisher)
    a = Toy({'fisher': cfg}, likelihood=ToyLikelihood(), norm_step=keyword,
            tools=DummyTools({'Omega_c': 0.27, 'A_s': AMP, 'wa': 0.0, 'h': 0.67}),
            req_params={'dz': 0.0})
    a.samples = []
    return a


def reach(a):
    return np.abs(np.array(a.samples) - X0).max(axis=0)


def rel_error(a):
    D, J = a.get_derivatives(force=True), toy_jacobian(X0)
    return np.max(np.abs(D - J).max(axis=1) / np.abs(J).max(axis=1))


@pytest.mark.parametrize('method, step, expected', [
    ('5pt_stencil', None, [0.01] * 4),
    ('5pt_stencil', '1e-3', [1e-3] * 4),          # YAML reads 1e-3 as a string
    ('5pt_stencil', {'A_s': '2e-11', 'wa': 0.05}, [0.01, 2e-11, 0.05, 0.01]),
    ('numdifftools', {'default': 1e-3, 'A_s': 2e-11}, [1e-3, 2e-11, 1e-3, 1e-3]),
    ('adaptive', 1e-3, None),                     # derivkit samples as derivative_args says
])
def test_step_forms(method, step, expected):
    a = build(method, **({} if step is None else {'step': step}))
    assert a._steps() == expected


@pytest.mark.parametrize('step, match', [({'Omega_cc': 1e-3}, 'Omega_cc'),
                                         ({'wa': 0.0}, 'positive'),
                                         ({'default': -1e-3}, 'positive')])
def test_bad_steps_raise(step, match):
    with pytest.raises(ValueError, match=match):
        build(step=step)._steps()


@pytest.mark.parametrize('method', ['5pt_stencil', 'numdifftools'])
def test_each_parameter_moves_by_its_own_step(method):
    a = build(method, step={'default': 1e-3, 'A_s': 2e-11, 'wa': 0.05})
    assert rel_error(a) < (1e-3 if method == 'numdifftools' else 1e-6)
    offsets = np.array(a.samples) - X0
    assert np.all(np.count_nonzero(offsets, axis=1) <= 1)
    np.testing.assert_allclose(reach(a), METHODS[method][2] * np.array(a._steps()), rtol=0.01)


def test_derivkit_per_parameter_steps_are_not_supported():
    with pytest.raises(ValueError, match='not supported for derivkit'):
        build('adaptive', step={'A_s': 2e-11})
    with pytest.raises(ValueError, match='not supported for derivkit'):
        build().get_derivatives(method='derivkit', step={'A_s': 2e-11})
    # A single step is ignored, as before.
    np.testing.assert_array_equal(build('adaptive', step=1e-3).get_derivatives(),
                                  build('adaptive').get_derivatives())


@pytest.mark.parametrize('method', ['5pt_stencil', 'numdifftools'])
def test_norm_step_steps_are_fractions_of_the_bounds(method):
    a = build(method, norm_step=True, step={'wa': 0.005})
    assert rel_error(a) < 1e-3
    np.testing.assert_allclose(reach(a), METHODS[method][2] * WIDTH * [0.01, 0.01, 0.005, 0.01],
                               rtol=0.01)


def test_norm_step_config_keyword_and_bounds():
    assert not build().norm_step and build(norm_step=True).norm_step
    assert not build(keyword=False, norm_step=True).norm_step   # the keyword wins
    with pytest.raises(ValueError, match='every varied parameter'):
        build(norm_step=True, bounds=dict(BOUNDS, wa=0.0))


def test_norm_step_is_not_applied_to_derivkit():
    expected = build('adaptive').get_derivatives()
    with pytest.warns(UserWarning, match='not applied to derivkit'):
        a = build('adaptive', norm_step=True)
        np.testing.assert_array_equal(a.get_derivatives(), expected)


def test_default_step_through_zero_warns():
    with pytest.warns(UserWarning, match='A_s'):
        build(step={'wa': 0.05})._steps()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        build(step={'A_s': 2e-11, 'wa': 0.05})._steps()


def test_five_pt_stencil_per_component_step():
    def g(x):
        return np.stack([np.sin(x[..., 0]) * x[..., 1], x[..., 1]**5], axis=-1)

    x0, h = np.array([0.3, 1.2]), np.array([1e-2, 1e-3])
    np.testing.assert_array_equal(five_pt_stencil(g, x0, h=1e-3),
                                  five_pt_stencil(g, x0, h=[1e-3] * 2))
    for j in range(2):
        np.testing.assert_array_equal(five_pt_stencil(g, x0, h=h)[j],
                                      five_pt_stencil(g, x0, h=h[j])[j])
    with pytest.raises(ValueError, match='positive'):
        five_pt_stencil(g, x0, h=[1e-3, 0.0])
