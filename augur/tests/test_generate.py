import pytest
import os
from ..generate import generate, get_noise_power, firecrown_sanitize
from math import pi


def test_generate(example_yaml):
    generate(example_yaml)
    assert os.path.isfile(example_yaml['generate']['two_point']['sacc_file'])


def test_get_noise_power():
    src = 'test'
    config = {'sources': {src: {'number_density': 10}}}  # per arcmin
    nbar = 10 * (180 * 60 / pi)**2  # per steradian
    config['sources'][src]['kind'] = 'NumberCountsSource'
    print(config)
    res = get_noise_power(config, src)
    assert res == pytest.approx(1/nbar)
    config['sources'][src]['kind'] = 'WLSource'
    config['sources'][src]['ellipticity_error'] = 4
    res = get_noise_power(config, src)
    assert res == pytest.approx(4*4/nbar)
    try:
        config['sources'][src]['kind'] = 'Bad'
        get_noise_power(config, src)
    except NotImplementedError:
        pass


def test_firecrown_sanitize(example_yaml):
    # we are going to test just the basic functionality
    assert 'augur' in example_yaml['generate']
    assert 'augur' not in firecrown_sanitize(example_yaml['generate'])
