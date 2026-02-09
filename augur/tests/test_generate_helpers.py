import pytest

from augur.generate import _get_tracers, _get_scale_cuts


def test_get_tracers_galaxy_density():
    assert _get_tracers('galaxy_density_cl', (0, 1)) == ('lens0', 'lens1')


def test_get_tracers_shear_ee():
    assert _get_tracers('galaxy_shear_cl_ee', (2, 3)) == ('src2', 'src3')


def test_get_tracers_shear_density():
    assert _get_tracers('galaxy_shearDensity_cl_e', (1, 4)) == ('lens1', 'src4')


def test_get_tracers_unrecognized():
    res = _get_tracers('unknown_stat', (0, 1))
    # Function returns a NotImplementedError instance rather than raising.
    assert isinstance(res, NotImplementedError)


def test_get_scale_cuts_scalar_lmax():
    stat_cfg = {'lmax': 500, 'tracer_combs': [[0, 0], [1, 1]]}
    lmax, kmax = _get_scale_cuts(stat_cfg, [0, 0])
    assert lmax == 500
    assert kmax is None


def test_get_scale_cuts_list_lmax():
    stat_cfg = {'lmax': [100, 200], 'tracer_combs': [[0, 0], [1, 1]]}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, (1, 1))
    lmax, _ = _get_scale_cuts(stat_cfg, [1, 1])
    assert lmax == 200


def test_get_scale_cuts_list_length_mismatch():
    stat_cfg = {'lmax': [100], 'tracer_combs': [[0, 0], [1, 1]]}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, (1, 1))


def test_get_scale_cuts_both_kmax_lmax():
    stat_cfg = {'lmax': 100, 'kmax': 0.2}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, [0, 0])


def test_get_scale_cuts_invalid_kmax_type():
    stat_cfg = {'kmax': 'not_a_number'}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, [0, 0])
