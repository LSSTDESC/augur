#
# Data Analysis Module
#

import firecrown
import pathlib
from .generate import firecrown_sanitize


def analyze(config):
    """ Analyzes the data, i.e. a thin wrapper to firecrown

    Parameters:
    ----------
    config : dict
        The yaml parsed dictional of the input yaml file

    """

    ana_config = config['analyze']
    config, data = firecrown.parse(firecrown_sanitize(ana_config))
    firecrown.run_cosmosis(config, data, pathlib.Path(config['cosmosis']['output_dir']))
