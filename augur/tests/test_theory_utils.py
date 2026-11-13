import numpy as np
import pytest

from augur.utils.theory_utils import compute_new_theory_vector


class _Recorder:
    """Stands in for both ModelingTools and the likelihood: records the ParamsMap it is given."""

    def __init__(self):
        self.pmap = None

    def reset(self):
        pass

    def update(self, pmap):
        self.pmap = pmap

    def prepare(self):
        pass

    def compute_theory_vector(self, tools):
        return np.zeros(3)


def test_compute_new_theory_vector_passes_sampled_params():
    lk, tools = _Recorder(), _Recorder()
    pars = {'Omega_c': 0.25, 'sigma8': 0.8, 'A_s': None, 'm_nu': 0.06, 'mass_split': 'equal',
            'transfer_function': 'eisenstein_hu', 'matter_power_spectrum': 'linear'}
    out = compute_new_theory_vector(lk, tools, {'lens0_bias': 1.5}, pars)

    assert out.shape == (3,)
    keys = set(tools.pmap.keys())
    # the sampled floats reach the ParamsMap, and both objects see the same map
    assert {'Omega_c', 'sigma8', 'm_nu', 'lens0_bias'} <= keys
    assert lk.pmap is tools.pmap


@pytest.mark.parametrize('A_s, kept, dropped', [(None, 'sigma8', 'A_s'),
                                                (2.1e-9, 'A_s', 'sigma8')])
def test_compute_new_theory_vector_keeps_one_amplitude(A_s, kept, dropped):
    """Only the amplitude parameter the factory was built with reaches the ParamsMap."""
    lk, tools = _Recorder(), _Recorder()
    pars = {'Omega_c': 0.25, 'sigma8': 0.8, 'A_s': A_s}
    compute_new_theory_vector(lk, tools, {}, pars)

    keys = set(tools.pmap.keys())
    assert kept in keys
    assert dropped not in keys
