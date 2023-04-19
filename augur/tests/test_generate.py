import os
from ..generate import generate


def test_generate(example_yaml):
    generate(example_yaml)
    assert os.path.isfile(example_yaml['generate']['two_point']['sacc_file'])
