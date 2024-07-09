import numpy as np
import pyccl as ccl
from augur.utils.diff_utils import five_pt_stencil
from augur import generate
from augur.utils.config_io import parse_config
from firecrown.parameters import ParamsMap
from astropy.table import Table


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

        _config = parse_config(config)  # Load full config

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

        # Get the fiducial cosmological parameters
        self.pars_fid = tools.get_ccl_cosmology().__dict__['_params_init_kwargs']

        # Load the relevant section of the configuration file
        self.config = _config['fisher']

        # Initialize pivot point
        self.x = []
        self.var_pars = None
        self.derivatives = None
        self.Fij = None
        self.bi = None
        self.biased_cls = None
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
        if hasattr(x, "__len__"):
            if len(labels) != len(x):
                raise ValueError('The labels should have the same length as the parameters!')
        else:
            if isinstance(labels, list):
                raise ValueError('x is a scalar and labels has more than one entry')

        if isinstance(x, list):
            x = np.array(x)
        # Scalar variable
        if isinstance(x, (float, int)):
            _pars = pars_fid.copy()
            _sys_pars = sys_fid.copy()
            if labels in pars_fid.keys():
                _pars.update({labels: x})
            elif labels in sys_fid.keys():
                _sys_pars.update({labels: x})
            self.tools.reset()
            self.lk.reset()
            pmap = ParamsMap(_sys_pars)
            cosmo = ccl.Cosmology(**_pars)
            self.lk.update(pmap)
            self.tools.update(pmap)
            self.tools.prepare(cosmo)
            f_out = self.lk.compute_theory_vector(self.tools)
        # 1D
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
            pmap = ParamsMap(_sys_pars)
            cosmo = ccl.Cosmology(**_pars)
            self.lk.update(pmap)
            self.tools.update(pmap)
            self.tools.prepare(cosmo)
            f_out = self.lk.compute_theory_vector(self.tools)
        # 2D
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
                pmap = ParamsMap(_sys_pars)
                cosmo = ccl.Cosmology(**_pars)
                self.lk.update(pmap)
                self.tools.update(pmap)
                self.tools.prepare(cosmo)
                f_out.append(self.lk.compute_theory_vector(self.tools))
        return np.array(f_out)

    def get_derivatives(self, force=False, method='stem'):
        # Compute the derivatives with respect to the parameters in var_pars at x
        if (self.derivatives is None) or (force):
            if '5pt_stencil' in method:
                self.derivatives = five_pt_stencil(lambda y: self.f(y, self.var_pars, self.pars_fid,
                                                   self.req_params),
                                                   self.x, h=float(self.config['step']))
            elif 'numdifftools' in method:
                import numdifftools as nd
                if 'numdifftools_kwargs' in self.config.keys():
                    ndkwargs = self.config['numdifftools_kwargs']
                else:
                    ndkwargs = {}
                self.derivatives = nd.Gradient(lambda y: self.f(y, self.var_pars, self.pars_fid,
                                               self.req_params),
                                               step=float(self.config['step']),
                                               **ndkwargs)(self.x).T
            elif 'stem' in method:
                from derivative_calculator import DerivativeCalculator
                _aux_ders = np.zeros((len(self.x), len(self.data_fid)))
                for i in range(len(self.x)):
                    _aux = np.zeros((len(self.data_fid)))
                    for j in range(len(self.data_fid)):
                        calc = DerivativeCalculator(lambda y: self.f(y, self.var_pars[i],
                                                                     self.pars_fid,
                                                                     self.req_params)[j],
                                                    self.x[i],
                                                    dx=float(self.config['step']))
                        _aux[j] = calc.stem_method()
                    _aux_ders[i] = _aux
                self.derivatives = np.array(_aux_ders)
            else:
                raise ValueError(f'Selected method: `{method}` is not available. \
                                 Please select 5pt_stencil or numdifftools.')
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

                self.biased_cls = self.f(_x_here, _labels_here, _pars_here, _sys_here) \
                    - self.data_fid

            Bj = np.einsum('l, lm, jm', self.biased_cls, self.lk.inv_cov, self.derivatives)
            bi = np.einsum('ij, j', np.linalg.inv(self.Fij), Bj)
            self.bi = bi
            return self.bi
