import os
from ..generate import generate


def test_generate():
    generate('./examples/srd_y1_3x2.yaml')
    assert os.path.isfile(example_yaml['generate']['two_point']['sacc_file'])
