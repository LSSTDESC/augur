from firecrown.parameters import ParamsMap
import warnings


def compute_new_theory_vector(lk, tools, _sys_pars, _pars, return_all=False):
    """
    Utility function to update the likelihood and modeling tool objects to use a new
    set of parameters and compute a new theory prediction

    Parameters:
    -----------
    lk : firecrown.likelihood.Likelihood,
        Likelihood object to use.
    tools: firecrown.ModelingTools,
        ModelingTools object to use.
    _sys_pars : dict,
        Dictionary containing the "systematic" modeling parameters.
    _pars : dict,
        Dictionary containing the cosmological parameters
    return_all : bool,
        If `False` it will just return the predicted data vector. Else,
        it will return the internal ModelingTools and Likelihood objects.

    Returns:
    --------
    f_out : ndarray,
        Predicted data vector for the given input parameters _sys_pars, _pars.
    lk : firecrown.likelihood.Likelihood,
        Modified likelihood object centered on _pars, and _sys_pars.
    tools: firecrown.ModelingTools,
        Modified tools object with fiducial values on _pars, _sys_pars.
    """
    lk.reset()
    tools.reset()
    dict_all = {**_sys_pars, **_pars}
    extra_dict = {}
    # Support either amplitude convention being present in the incoming
    # dictionary; older/newer cosmology dicts may omit one of these keys.
    if 'A_s' not in dict_all:
        dict_all['A_s'] = None
    if 'sigma8' not in dict_all:
        dict_all['sigma8'] = None
    # else:
     # raise ValueError("Input parameter dictionary must contain either A_s or sigma8.")

    # Ensure required CCLFactory sampler parameters are present even when
    # callers provide a partial cosmology dictionary (common in use_sacc paths).
    ccl_factory = getattr(tools, 'ccl_factory', None)
    if ccl_factory is not None:
        for key in [
            'Omega_c', 'Omega_b', 'h', 'n_s', 'Omega_k',
            'Neff', 'm_nu', 'w0', 'wa', 'T_CMB',
        ]:
            if key not in dict_all and hasattr(ccl_factory, key):
                value = getattr(ccl_factory, key)
                # ccl_factory stores many of these as updatable parameter objects;
                # only pass through plain scalar values.
                if isinstance(value, (int, float)):
                    dict_all[key] = value

    # Hard fallback defaults for any still-missing core cosmology keys.
    default_cosmo = {
        'Neff': 3.044,
        'T_CMB': 2.7255,
        'm_nu': 0.0,
        'Omega_k': 0.0,
        'w0': -1.0,
        'wa': 0.0,
    }
    for key, value in default_cosmo.items():
        dict_all.setdefault(key, value)

    if dict_all['A_s'] is None:
        extra_dict['amplitude_parameter'] = 'sigma8'
        dict_all.pop('A_s')
    else:
        extra_dict['amplitude_parameter'] = 'as'
        dict_all.pop('sigma8')

    extra_dict['mass_split'] = dict_all.get('mass_split', 'equal')
    dict_all.pop('mass_split', None)

    hm = dict_all.pop('extra_parameters', None)
    if hm is not None and 'camb' in hm.keys():
        hm = hm.pop('camb')
        if hm is not None:
            halofit_version = hm.get('halofit_version')
            if halofit_version in ('mead2020_feedback',
                                   'mead',
                                   'mead2015',
                                   'mead2016'):
                dict_all['HMCode_logT_AGN'] = hm.get('HMCode_logT_AGN', 7.8)
                dict_all['HMCode_eta_baryon'] = hm.get('HMCode_eta_baryon', 0.603)
                dict_all['HMCode_A_baryon'] = hm.get('HMCode_A_baryon', 3.13)

    # Handle modified gravity parameters
    mg = dict_all.pop('mg_parametrization', None)
    if mg is not None:
        # mg may be a MuSigmaMG object (from cosmo.to_dict()) or a raw
        # config dict (from YAML).  Normalise to scalar attributes.
        from pyccl.modified_gravity import MuSigmaMG
        if isinstance(mg, MuSigmaMG):
            dict_all['mg_musigma_mu'] = float(mg.mu_0)
            dict_all['mg_musigma_sigma'] = float(mg.sigma_0)
            dict_all['mg_musigma_c1'] = float(mg.c1_mg)
            dict_all['mg_musigma_c2'] = float(mg.c2_mg)
            dict_all['mg_musigma_lambda0'] = float(mg.lambda_mg)
        elif isinstance(mg, dict):
            musigma = mg.get('mu_Sigma', None)
            if musigma is not None:
                dict_all['mg_musigma_mu'] = float(musigma.get('mu_0', 0.0))
                dict_all['mg_musigma_sigma'] = float(musigma.get('sigma_0', 0.0))
                dict_all['mg_musigma_c1'] = float(musigma.get('c1_mg', 1.0))
                dict_all['mg_musigma_c2'] = float(musigma.get('c2_mg', 1.0))
                dict_all['mg_musigma_lambda0'] = float(musigma.get('lambda_mg', 0.0))

    be = dict_all.pop('baryonic_effects', None)
    if be is not None:
        warnings.warn("Baryonic effects parameters specified but not currently \
                       implemented. Ignoring these parameters.")

    _ = dict_all.pop('transfer_function', None)
    _ = dict_all.pop('matter_power_spectrum', None)

    keys = list(dict_all.keys())

    # Remove None values
    for key in keys:
        if (dict_all[key] is None) or (dict_all[key] == 'None'):
            dict_all.pop(key)

    pmap = ParamsMap(dict_all)
    tools.update(pmap)
    tools.prepare()
    lk.update(pmap)
    f_out = lk.compute_theory_vector(tools)
    if return_all:
        return f_out, lk, tools
    else:
        return f_out
