import numpy as np
import pyccl as ccl
from augur.utils.diff_utils import five_pt_stencil
from augur import generate
from augur.utils.config_io import parse_config
from firecrown.parameters import ParamsMap


class Analyze(object):
    def __init__(self, config, likelihood=None, tools=None, req_params=None):
        """
        Run numerical derivatives of a likelihood to obtain a Fisher matrix estimate.

        Parameters:
        -----------
        config: str or dict
            Input config file or dictionary
        likelihood: firecrown.likelihood
            Input likelihood object that will be used to compute the derivatives.
            If None, we will call generate.
        tools: firecrown.modeling_tools.ModelingTools
            Modeling tools needed to reevaluate the likelihood. Required if a likelihood
            object is passed.
        req_params: dict
            Dictionary containing the required parameters for the likelihood. Required if a
            likelihood object is passed.

        Returns:
        --------
        fisher: np.ndarray
        Output Fisher matrix
        """
        # Load the likelihood if no likelihood is passed along
        if likelihood is None:
            likelihood, S, tools, req_params= generate(config, return_all_outputs=True)
        else:
            if (tools is None) or (req_params is None):
                raise ValueError('If a likelihood is passed tools and req_params are required! \
                                Please, remove the likelihood or add tools and req_params.')
        
        self.lk = likelihood  # Just to save some typing 
        self.tools = tools
        self.req_params = req_params
        
        _config = parse_config(config)  # Load full config
        # Get the fiducial cosmological parameters
        self.pars_fid = tools.get_ccl_cosmology().__dict__['_params_init_kwargs']

        # Load the relevant section of the configuration file
        self.config = _config['fisher']

        # Initialize pivot point
        self.x = []
        self.var_pars = None
        self.derivatives = None
        self.Fij = None
        # Load the parameters to vary
        # We will allow 2 options -- one where we pass something
        # a la cosmosis with parameters and minimum, central, and max
        # we can also allow priors
        if set(['parameters', 'var_pars']).issubset(set(self.config.keys())):
            raise Warning('Both `parameters` and `var_pars` found in Fisher. \
                        Ignoring `parameters and using fiducial cosmology \
                        as pivot.`')
        if 'parameters' in self.config.keys():
            self.var_pars = list(self.config['parameters'].keys())
            for var in self.var_pars:
                _val = self.config['parameters'][var]
                if isinstance(_val, list):
                    self.x.append(_val[1])
                else:
                    self.x.append(_val)
        # The other option is to pass just the parameter names and evaluate around
        # the fiducial values
        elif 'var_pars' in self.config.keys():
            var_pars = self.config['var_pars']
            for var in var_pars:
                if var in self.pars_fid.keys():
                    x.append(self.pars_fid[var])
                elif var in self.req_params.keys():
                    x.append(self.req_params[var])
                else:
                    raise ValueError(f'The requested parameter {var} is not \
                                    in the list of parameters in the likelihood.')
        # Cast to numpy array (this will be done later anyway)
        self.x = np.array(self.x)
        
    def f(self, x, labels, pars_fid, sys_fid):
        """
        Auxiliary Function that returns a theory vector evaluated at x.
        Labels are the name of the parameters x (with the same length and order)

        Parameters:
        -----------

        x : float, list or np.ndarray
            Point at which to compute the theory vector
        labels : list
            Names of parameters to vary
        pars_fid : dict
            Dictionary containing the fiducial ccl cosmological parameters
        sys_fid: dict
            Dictionary containing the fiducial `systematic` (required) parameters
            for the likelihood
        Returns:
        --------
        f_out : np.ndarray
                Theory vector computed at x.
        """
        
        if len(labels) != len(x):
            raise ValueError('The labels should have the same length as the parameters!')
        else:
            if isinstance(x, list):
                x = np.array(x)
            if x.ndim == 1:
                _pars = pars_fid.copy()
                _sys_pars = sys_fid.copy()
                for i in range(len(labels)):
                    if labels[i] in pars_fid.keys():
                        _pars.update({labels[i]: x[i]})
                    elif labels[i] in sys_fid.keys():
                        _sys_pars.update({labels[i]: x[i]})
                    else:
                        raise ValueError(f'Parameter name {labels[i]} not recognized!') 
                self.tools.reset()
                self.lk.reset()
                cosmo = ccl.Cosmology(**_pars)
                self.lk.update(ParamsMap(_sys_pars))
                self.tools.update(ParamsMap(_sys_pars))
                self.tools.prepare(cosmo)
                f_out = self.lk.compute_theory_vector(self.tools)
            elif x.ndim == 2:
                f_out = []
                for i in range(len(labels)):
                    _pars = pars_fid.copy()
                    _sys_pars = sys_fid.copy()
                    xi = x[i]
                    for j in range(len(labels)):
                        if labels[j] in pars_fid.keys():
                            _pars.update({labels[j]: xi[j]})
                        elif labels[j] in sys_fid.keys():
                            _sys_pars.update({labels[j]: xi[j]})
                        else:
                            raise ValueError(f'Parameter name {labels[j]} not recognized')
                    self.tools.reset()
                    self.lk.reset()
                    self.lk.update(ParamsMap(_sys_pars))
                    self.tools.update(ParamsMap(_sys_pars))
                    cosmo = ccl.Cosmology(**_pars)
                    self.tools.prepare(cosmo)
                    f_out.append(self.lk.compute_theory_vector(self.tools))
            return np.array(f_out)
        
    def get_derivatives(self, force=False):
        # Compute the derivatives with respect to the parameters in var_pars at x
        if (self.derivatives is None) or (force):
            self.derivatives = five_pt_stencil(lambda y: self.f(y, self.var_pars, self.pars_fid, self.req_params),
                                               self.x, h=float(self.config['step']))
            return self.derivatives
        else:
            return self.derivatives
    
    def get_fisher_matrix(self):
        # Compute Fisher matrix assuming Gaussian likelihood (around self.x)
        if self.derivatives is None:
            self.get_derivatives()
        if self.Fij is None:
            self.Fij = np.einsum('il, lm, jm', self.derivatives, self.lk.inv_cov, self.derivatives)
            return self.Fij
        else:
            return self.Fij

    def compute_fisher_bias(self):
        # Compute Fisher bias following the generalized Amara formalism
        # More details in Bianca's thesis and the note here: 
        # https://github.com/LSSTDESC/augur/blob/note_bianca/note/main.tex
        import os
        