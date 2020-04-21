import pytest
import os
from ..analyze import analyze
from ..generate import generate

def test_analyze(example_yaml):
    generate(example_yaml)
    analyze(example_yaml)
    assert os.path.isfile(example_yaml['analyze']['cosmosis']['output'])
