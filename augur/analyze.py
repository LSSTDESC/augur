import numpy as np
import pyccl as ccl
from augur.utils.diff_utils import five_pt_stencil
from augur import generate
from augur.utils.config_io import parse_config
from firecrown.parameters import ParamsMap
from astropy.table import Table
import warnings
from packaging.version import Version
import firecrown


class Analyze(object):
    def __init__(self, config, likelihood=None, tools=None, req_params=None,
                 norm_step=True):
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
        norm_step : bool
            If `True` it internally normalizes the step size using the sampling bounds in order
            to determine the samples.

        Returns:
        --------
        fisher: np.ndarray
        Output Fisher matrix
        """

        config = parse_config(config)  # Load full config

        # Load the likelihood if no likelihood is passed along
        if likelihood is None:
            likelihood, S, tools, req_params = generate(config, return_all_outputs=True)
        else:
            if (tools is None) or (req_params is None):
                raise ValueError('If a likelihood is passed tools and req_params are required! \
                                Please, remove the likelihood or add tools and req_params.')
            else:
                # Load the ccl accuracy parameters if `generate` is not run
                if 'ccl_accuracy' in config.keys():
                    # Pass along spline control parameters
                    if 'spline_params' in config['ccl_accuracy'].keys():
                        for key in config['ccl_accuracy']['spline_params'].keys():
                            try:
                                type_here = type(ccl.spline_params[key])
                                value = config['ccl_accuracy']['spline_params'][key]
                                ccl.spline_params[key] = type_here(value)
                            except KeyError:
                                print(f'The selected spline keyword `{key}` is not recognized.')
                            except ValueError:
                                print(f'The selected value `{value}` could not be casted to \
                                      `{type_here}`.')
                    # Pass along GSL control parameters
                    if 'gsl_params' in config['ccl_accuracy'].keys():
                        for key in config['ccl_accuracy']['gsl_params'].keys():
                            try:
                                type_here = type(ccl.gsl_params[key])
                                value = config['ccl_accuracy']['gsl_params'][key]
                                ccl.gsl_params[key] = type_here(value)
                            except KeyError:
                                print(f'The selected GSL keyword `{key}` is not recognized.')
                            except ValueError:
                                print(f'The selected value `{value}` could not be casted to \
                                      `{type_here}`.')

        self.lk = likelihood  # Just to save some typing
        self.tools = tools
        self.req_params = req_params
        self.data_fid = self.lk.get_data_vector()
        self.norm_step = norm_step
        # Get the fiducial cosmological parameters
        self.pars_fid = tools.get_ccl_cosmology().__dict__['_params_init_kwargs']
        # CCL Factory placeholder (for newer firecrown)
        self.cf = None

        # Load the relevant section of the configuration file
        self.config = config['fisher']

        # Initialize pivot point
        self.x = []
        self.var_pars = None
        self.derivatives = None
        self.Fij = None
        self.bi = None
        self.biased_cls = None
        self.par_bounds = []
        self.norm = None
        # Load the parameters to vary
        # We will allow 2 options -- one where we pass something
        # a la cosmosis with parameters and minimum, central, and max
        # we can also allow priors
        if set(['parameters', 'var_pars']).issubset(set(self.config.keys())):
            warnings.warn('Both `parameters` and `var_pars` found in Fisher. \
                        Ignoring `parameters and using fiducial cosmology \
                        as pivot.`')

        if 'parameters' in self.config.keys():
            self.var_pars = list(self.config['parameters'].keys())
            for var in self.var_pars:
                _val = self.config['parameters'][var]
                if isinstance(_val, list):
                    self.x.append(_val[1])
                    self.par_bounds.append([_val[0], _val[-1]])
                else:
                    self.x.append(_val)

        # The other option is to pass just the parameter names and evaluate around
        # the fiducial values
        elif 'var_pars' in self.config.keys():
            self.var_pars = self.config['var_pars']
            for var in self.var_pars:
                if var in self.pars_fid.keys():
                    self.x.append(self.pars_fid[var])
                elif var in self.req_params.keys():
                    self.x.append(self.req_params[var])
                else:
                    raise ValueError(f'The requested parameter {var} is not \
                                    in the list of parameters in the likelihood.')
        # Cast to numpy array (this will be done later anyway)
        self.x = np.array(self.x)
        self.par_bounds = np.array(self.par_bounds)
        if (len(self.par_bounds) < 1) & (self.norm_step):
            self.norm_step = False
            warnings.warn('Parameter bounds not provided -- the step will not be normalized')
        # Normalize the pivot point given the sampling region
        if self.norm_step:
            self.norm = self.par_bounds[:, 1] - self.par_bounds[:, 0]
            self.x = (self.x - self.par_bounds[:, 0]) * 1/self.norm

    def f(self, x, labels, pars_fid, sys_fid, donorm=False):
        """
        Auxiliary Function that returns a theory vector evaluated at x.
        Labels are the name of the parameters x (with the same length and order)

        Parameters:
        -----------

        x : float, list or np.ndarray
            Point at which to compute the theory vector.
        labels : list
            Names of parameters to vary.
        pars_fid : dict
            Dictionary containing the fiducial ccl cosmological parameters
        sys_fid: dict
            Dictionary containing the fiducial `systematic` (required) parameters
            for the likelihood.
        norm: bool
            If `True` it normalizes the input parameters vector (useful for derivatives).
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
            # If we normalize the sampling we need to undo the normalization
            if donorm:
                x = self.norm * x + self.par_bounds[:, 0]

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

                f_out = self.compute_new_theory_vector(_sys_pars, _pars)

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
                    f_out.append(self.compute_new_theory_vector(_sys_pars, _pars))
            return np.array(f_out)

    def get_derivatives(self, force=False, method='5pt_stencil', step=None):
        """
        Auxiliary function to compute numerical derivatives of the helper function `f`

        Parameters:
        -----------
        force : bool
            If `True` force recalculation of the derivatives.
        method : str
            Method to compute derivatives, currently only `5pt_stencil` or `numdifftools`
            are allowed.
        step : float
            Step size for numerical differentiation
        """

        if step is None:
            step = float(self.config['step'])
        # Compute the derivatives with respect to the parameters in var_pars at x
        if (self.derivatives is None) or (force):
            if '5pt_stencil' in method:
                self.derivatives = five_pt_stencil(lambda y: self.f(y, self.var_pars, self.pars_fid,
                                                   self.req_params, donorm=self.norm_step),
                                                   self.x, h=step)
            elif 'numdifftools' in method:
                import numdifftools as nd
                if 'numdifftools_kwargs' in self.config.keys():
                    ndkwargs = self.config['numdifftools_kwargs']
                else:
                    ndkwargs = {}
                jacobian_calc = nd.Jacobian(lambda y: self.f(y, self.var_pars, self.pars_fid,
                                                             self.req_params,
                                                             donorm=self.norm_step),
                                            step=step,
                                            **ndkwargs)
                self.derivatives = jacobian_calc(self.x).T
            else:
                raise ValueError(f'Selected method: `{method}` is not available. \
                                 Please select 5pt_stencil or numdifftools.')
            if self.norm is not None:
                self.derivatives /= self.norm[:, None]
            return self.derivatives
        else:
            return self.derivatives

    def get_fisher_matrix(self, method='5pt_stencil', save_txt=True):
        # Compute Fisher matrix assuming Gaussian likelihood (around self.x)
        if self.derivatives is None:
            self.get_derivatives(method=method)
        if self.Fij is None:
            self.Fij = np.einsum('il, lm, jm', self.derivatives, self.lk.inv_cov, self.derivatives)
            if save_txt:
                np.savetxt(self.config['output'], self.Fij)
                tab_out = Table(self.x.T, names=self.var_pars)
                tab_out.write(self.config['fid_output'], format='ascii', overwrite=True)
            return self.Fij
        else:
            return self.Fij

    def get_fisher_bias(self):
        # Compute Fisher bias following the generalized Amara formalism
        # More details in Bianca's thesis and the note here:
        # https://github.com/LSSTDESC/augur/blob/note_bianca/note/main.tex

        # Allowing to provide externally calculated "systematics"
        # They should have the same ells as the original data-vector
        # and the same length
        import os

        if self.derivatives is None:
            self.get_derivatives()

        if self.Fij is None:
            self.get_fisher_matrix()

        if self.bi is not None:
            return self.bi

        else:
            _calculate_biased_cls = True

            # Try to read the biased data vector
            if 'biased_dv' in self.config['fisher_bias']:
                _sys_path = self.config['fisher_bias']['biased_dv']
                if (len(_sys_path) < 1) or (os.path.exists(_sys_path) is False):
                    _calculate_biased_cls = True
                else:
                    import astropy.table
                    if ('.dat' in _sys_path) or ('.txt' in _sys_path):
                        _format = 'ascii'
                    elif ('.fits' in _sys_path):
                        _format = 'fits'
                    else:
                        _format = None
                    biased_cls = astropy.table.Table.read(_sys_path, format=_format)
                    if len(biased_cls) != len(self.lk.get_data_vector()):
                        raise ValueError('The length of the provided Cls should be equal \
                                        to the length of the data-vector')
                    _calculate_biased_cls = False
                    self.biased_cls = biased_cls['dv_sys'] - self.data_fid

            # If there's no biased data vector, calculate it
            if _calculate_biased_cls:
                _x_here = []
                _labels_here = []
                if 'bias_params' in self.config['fisher_bias'].keys():
                    _pars_here = self.pars_fid.copy()
                    _sys_here = self.req_params.copy()
                    for key, value in self.config['fisher_bias']['bias_params'].items():
                        if key in _pars_here.keys():
                            _pars_here[key] = value
                            _x_here.append(value)
                            _labels_here.append(key)
                        elif key in _sys_here.keys():
                            _sys_here[key] = value
                            _x_here.append(value)
                            _labels_here.append(key)
                        else:
                            raise ValueError(f'The requested parameter `{key}` is not recognized. \
                                            Please make sure that it is part of your model.')
                else:
                    raise ValueError('bias_params is required if no biased_dv file is passed')

                self.biased_cls = self.f(_x_here, _labels_here, _pars_here, _sys_here,
                                         donorm=False) - self.data_fid

            Bj = np.einsum('l, lm, jm', self.biased_cls, self.lk.inv_cov, self.derivatives)
            bi = np.einsum('ij, j', np.linalg.inv(self.Fij), Bj)
            self.bi = bi
            return self.bi

    def compute_new_theory_vector(self, _sys_pars, _pars):
        """
        Utility function to update the likelihood and modeling tool objects to use a new
        set of parameters and compute a new theory prediction

        Parameters:
        -----------
        _sys_pars : dict,
            Dictionary containing the "systematic" modeling parameters.
        _pars : dict,
            Dictionary containing the cosmological parameters

        Returns:
        --------
        f_out : ndarray,
            Predicted data vector for the given input parameters _sys_pars, _pars.
        """
        self.lk.reset()
        self.tools.reset()
        if Version(firecrown.__version__) < Version('1.8.0a'):
            pmap = ParamsMap(_sys_pars)
            cosmo = ccl.Cosmology(**_pars)
            self.lk.update(pmap)
            self.tools.update(pmap)
            self.tools.prepare(cosmo)
            f_out = self.lk.compute_theory_vector(self.tools)
            return f_out
        else:
            from firecrown.ccl_factory import CCLFactory
            dict_all = {**_sys_pars, **_pars}
            extra_dict = {}
            if dict_all['A_s'] is None:
                extra_dict['amplitude_parameter'] = 'sigma8'
                dict_all.pop('A_s')
            else:
                extra_dict['amplitude_parameter'] = 'A_s'
                dict_all.pop('sigma8')

            extra_dict['mass_split'] = dict_all['mass_split']
            dict_all.pop('mass_split')
            if 'extra_parameters' in dict_all.keys():
                if 'camb' in dict_all['extra_parameters'].keys():
                    extra_dict['camb_extra_params'] = dict_all['extra_parameters']['camb']
                    if 'kmin' in dict_all['extra_parameters']['camb'].keys():
                        extra_dict['camb_extra_params'].pop('kmin')
                dict_all.pop('extra_parameters')
            keys = list(dict_all.keys())

            # Remove None values
            for key in keys:
                if (dict_all[key] is None) or (dict_all[key] == 'None'):
                    dict_all.pop(key)
            if self.cf is None:
                for key in extra_dict.keys():
                    print(extra_dict[key], type(extra_dict[key]))
                self.cf = CCLFactory(**extra_dict)
                self.tools = firecrown.modeling_tools.ModelingTools(ccl_factory=self.cf)
                self.tools.reset()
            pmap = ParamsMap(dict_all)
            self.cf.update(pmap)
            self.tools.update(pmap)
            self.tools.prepare()
            self.lk.update(pmap)
            f_out = self.lk.compute_theory_vector(self.tools)

            return f_out
