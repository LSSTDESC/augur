"""Cosmology initialisation from an Augur config dict."""

import pyccl as ccl
import warnings


def initialize_cosmology(config):
    """
    Build a :class:`pyccl.Cosmology` from the ``cosmo`` (and optional
    ``ccl_accuracy``) sections of an Augur config dictionary.

    Parameters
    ----------
    config : dict
        Full Augur configuration dictionary (must contain a ``cosmo`` key).

    Returns
    -------
    cosmo : pyccl.Cosmology
        Initialised CCL cosmology object.
    cosmo_cfg : dict
        The (potentially mutated) cosmology sub-dictionary — useful because
        modified-gravity entries are replaced in-place with CCL objects.
    """
    cosmo_cfg = config['cosmo']

    if cosmo_cfg.get("transfer_function") is None:
        cosmo_cfg['transfer_function'] = 'boltzmann_camb'

    if cosmo_cfg.get('extra_parameters') is None:
        cosmo_cfg['extra_parameters'] = dict()

    # ---- CCL accuracy overrides ----
    if 'ccl_accuracy' in config.keys():
        if 'spline_params' in config['ccl_accuracy'].keys():
            for key in config['ccl_accuracy']['spline_params'].keys():
                try:
                    type_here = type(ccl.spline_params[key])
                    value = config['ccl_accuracy']['spline_params'][key]
                    ccl.spline_params[key] = type_here(value)
                except KeyError:
                    print(f'The selected spline keyword `{key}` is not recognized.')
                except ValueError:
                    print(f'The selected value `{value}` could not be casted to `{type_here}`.')
        if 'gsl_params' in config['ccl_accuracy'].keys():
            for key in config['ccl_accuracy']['gsl_params'].keys():
                try:
                    type_here = type(ccl.gsl_params[key])
                    value = config['ccl_accuracy']['gsl_params'][key]
                    ccl.gsl_params[key] = type_here(value)
                except KeyError:
                    print(f'The selected GSL keyword `{key}` is not recognized.')
                except ValueError:
                    print(f'The selected value `{value}` could not be casted to `{type_here}`.')

    # ---- Modified gravity ----
    if cosmo_cfg.get('mg_parametrization', None) is not None:
        mg_cfg = cosmo_cfg['mg_parametrization']
        if mg_cfg.get('mu_Sigma', None) is not None:
            mu_sig = mg_cfg['mu_Sigma']
            required_keys = ['mu_0', 'sigma_0', 'c1_mg', 'c2_mg', 'lambda_mg']
            for key in required_keys:
                if key not in mu_sig.keys():
                    raise ValueError(
                        f'Missing required key `{key}` in '
                        '`mu_Sigma` modified gravity parametrization.'
                    )
                else:
                    try:
                        mu_sig[key] = float(mu_sig[key])
                    except ValueError:
                        print(
                            f'The selected value `{mu_sig[key]}` '
                            f'for `{key}` could not be casted to `float`.'
                        )
            cosmo_cfg['mg_parametrization'] = ccl.modified_gravity.mu_Sigma.MuSigmaMG(**mu_sig)

    # ---- Build cosmology object ----
    try:
        cosmo = ccl.Cosmology(**cosmo_cfg)
    except (KeyError, TypeError, ValueError) as e:
        print('Error in cosmology configuration. Check the config file.')
        raise e

    return cosmo, cosmo_cfg
