from firecrown.parameters import ParamsMap


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
    cf : firecrown.CCLFactory,
        If passed, CCLFactory object to use.
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
    if dict_all['A_s'] is None:
        extra_dict['amplitude_parameter'] = 'sigma8'
        dict_all.pop('A_s')
    else:
        extra_dict['amplitude_parameter'] = 'as'
        dict_all.pop('sigma8')

    extra_dict['mass_split'] = dict_all['mass_split']
    dict_all.pop('mass_split')

    hm = dict_all.pop('extra_parameters')
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
    print(lk.compute_loglike(tools))
    if return_all:
        return f_out, lk, tools
    else:
        return f_out
