"""
This is similar to what was built in the Blinding likelihood
"""
import importlib 
import os
import types

from firecrown.likelihood.likelihood import NamedParameters
from firecrown.likelihood.likelihood import load_likelihood
from firecrown.likelihood.likelihood import load_likelihood_from_module_type

def load_module_from_path(path):
    """
    Load a module from a given path.

    Parameters
    ----------
    path : str
        Path to the module to load.

    Returns
    -------
    module
        Module loaded from the given path.
    """
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _test_likelihood(likelihood, like_type):
    """
    Tests if the likelihood has the required methods.

    Parameters
    ----------
    likelihood : str or module
        path to the likelihood or a module containing the likelihood
        must contain both `build_likelihood` and `compute_theory_vector` methods
    like_type : str
        Type of likelihood. Can be either 'str' or 'module'.
    """
    if like_type == "str":
        likelihood = load_module_from_path(likelihood)
    else:
        likelihood = likelihood

    # checks if the module has the attribute `build_likelihood`
    if not hasattr(likelihood, "build_likelihood"):
        raise AttributeError("The likelihood module must have a `build_likelihood` method.")

def _load_likelihood(likelihood, sacc_data):
    """
    This is a healper frunction from DESC/Blinding to load the firecrown likelihoods

    Parameters
    ----------
    likelihood : str or module
        path to the likelihood or a module containing the likelihood
        must contain both `build_likelihood` and `compute_theory_vector` methods
    sacc_data : str
        Path to the sacc data file.

    Returns
    -------
    likelihood
        The likelihood object
    tools
        The Modelling Tools object
    """
    build_parameters = NamedParameters({'sacc_data': sacc_data})

    if type(likelihood) == str:
        if not os.path.isfile(likelihood):
            raise FileNotFoundError(f"File {likelihood} not found.")

        # test the likelihood
        _test_likelihood(likelihood, "str")

        #load the likelihood
        likelihood, tools = load_likelihood(likelihood, build_parameters)

        # check if the likehood has a `compute_theory_vector` method
        if not hasattr(likelihood, "compute_theory_vector"):
            raise AttributeError("The likelihood must have a `compute_theory_vector` method.")
        return likelihood, tools
    elif isinstance(likelihood, types.ModuleType):
        # test the likelihood
        _test_likelihood(likelihood, "module")

        # tries to load the likelihood from module
        likelihood, tools = load_likelihood_from_module_type(likelihood, build_parameters)
        if not hasattr(likelihood, "compute_theory_vector"):
            raise AttributeError("The likelihood must have a `compute_theory_vector` method.")
        return likelihood, tools
    else:
        raise TypeError("Likelihood must be either a string or a module.")


def load_firecrown_likelihood(likelihood, cosmo, sacc_data, syst_dict=None):
    # if likelihood is a string, load it from firecrown
    pass