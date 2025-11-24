from packaging.version import Version
import pyccl as ccl
import firecrown
from firecrown.parameters import ParamsMap



def compute_new_theory_vector(lk, tools, _sys_pars, _pars, cf=None, return_all=False):
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
    from firecrown.ccl_factory import CCLFactory, CCLCreationMode, CAMBExtraParams, CCLPureModeTransferFunction
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
    camb_baryon = False
    if 'extra_parameters' in dict_all.keys():
        if 'camb' in dict_all['extra_parameters'].keys():
            extra_dict['camb_extra_params'] = CAMBExtraParams(**dict_all['extra_parameters']['camb'])
            if 'kmin' in dict_all['extra_parameters']['camb'].keys():
                extra_dict['camb_extra_params'].pop('kmin')
            if 'halofit_version' in dict_all['extra_parameters']['camb'].keys():
                halofit_version = dict_all['extra_parameters']['camb']['halofit_version']
                # Use membership test rather than chained `or` which evaluates
                # to the first truthy string; the original check was wrong and
                # only compared to the first value.
                if halofit_version in ('mead2020_feedback', 'mead', 'mead2015', 'mead2016'):
                    camb_baryon = True
        dict_all.pop('extra_parameters')
    keys = list(dict_all.keys())

    # Remove None values
    for key in keys:
        if (dict_all[key] is None) or (dict_all[key] == 'None'):
            dict_all.pop(key)
    if cf is None:
        if camb_baryon:
            cf = CCLFactory(**extra_dict, require_nonlinear_pk=True,
                            use_camb_hm_sampling=True,
                            creation_mode=CCLCreationMode.PURE_CCL_MODE,
                            pure_ccl_transfer_function=CCLPureModeTransferFunction.BOLTZMANN_CAMB
                            )
        else:
            cf = CCLFactory(**extra_dict, require_nonlinear_pk=True,
                            creation_mode=CCLCreationMode.PURE_CCL_MODE,
                            pure_ccl_transfer_function=CCLPureModeTransferFunction.EISENSTEIN_HU
                            )
        if tools.pt_calculator is not None:
            ptc = tools.get_pt_calculator()
            tools = firecrown.modeling_tools.ModelingTools(pt_calculator=ptc, ccl_factory=cf)
        else:
            tools = firecrown.modeling_tools.ModelingTools(ccl_factory=cf)
        tools.reset()
    #print(dict_all)
    pmap = ParamsMap(dict_all)
    #cf.update(pmap)
    tools.update(pmap)
    tools.prepare()
    lk.update(pmap)
    f_out = lk.compute_theory_vector(tools)
    print(lk.compute_loglike(tools))
    if return_all:
        return f_out, lk, tools
    else:
        return f_out
