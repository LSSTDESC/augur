import numpy as np


def parse_config(config):
    """
    Utility to parse configuration file
    """
    if isinstance(config, str):
        from augur.parser import parse
        config = parse(config)
    elif isinstance(config, dict):
        pass
    else:
        raise ValueError('config must be a dictionary or path to a config file')
    return config


def read_fisher_from_file(base):
    '''
    Helper function to add external Fisher evaluations to the computed Fisher matrix in Augur.
    Requires the path have two files in the Augur format: fiducial and Fisher.

    This function does not check the compatibility of the fiducial systematic parameters
     or cosmology with the current Augur run; that is the user's responsibility.
    :param base: base path to the files (without _fiducials.dat or _fisher.dat)
    :return: fisher matrix and fiducial vector as numpy arrays
    '''
    try:
        fiducials = np.loadtxt(f"{base}_fiducials.dat")
        fisher = np.loadtxt(f"{base}_fisher.dat")
        # TODO: If we wind up changing the format of the Analyze object, we may want
        # to do some additional processing here to ensure compatibility/ease of use.
        return fisher, fiducials
    except Exception as e:
        raise RuntimeError(f"Could not read files at {base}. Exception: {e}")
