import os
import numpy as np
from ..analyze import analyze
from ..generate import generate


def test_analyze(example_yaml):
    generate(example_yaml)
    analyze(example_yaml)
    out_fname = example_yaml['analyze']['cosmosis']['output_dir']+"/chain.txt"
    assert os.path.isfile(out_fname)
    n_pars = len(example_yaml['analyze']['cosmosis']['parameters'])
    assert np.loadtxt(out_fname).shape == (n_pars, n_pars)
