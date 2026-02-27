import numpy as np
import pyccl as ccl
from augur.utils.diff_utils import five_pt_stencil
from augur import generate
from augur.utils.config_io import parse_config
from augur.utils.theory_utils import compute_new_theory_vector
from astropy.table import Table
import warnings
import pandas as pd

mnu_norm = 93.14  # eV


class Analyze(object):
    def __init__(self, config, likelihood=None, tools=None, req_params=None,
                 norm_step=False):
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
        fisher: np.ndarray,
        Output Fisher matrix
        """

        # config needs to specify likelihood yaml.
        # alternatively, can pass likelihood and tools objects at input parameters.
        # choose objects to take precedence.
        # Load the likelihood if no likelihood is passed along

        config = parse_config(config)  # Load full config
        if likelihood is None:
            likelihood, S, tools, req_params = generate(config, return_all_outputs=True)
        config = parse_config(config)  # Load full config

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
        self.pars_fid = tools.get_ccl_cosmology().to_dict()
        # need to potentially extract modified gravity parameters here and remove superfluous parameters
        if 'mg_parametrization' in self.pars_fid.keys():
            mg = self.pars_fid.pop('mg_parametrization')
            # mg is a MuSigmaMG object (from cosmo.to_dict()), not a raw dict
            from pyccl.modified_gravity import MuSigmaMG
            if isinstance(mg, MuSigmaMG):
                self.pars_fid['mg_musigma_mu'] = float(mg.mu_0)
                self.pars_fid['mg_musigma_sigma'] = float(mg.sigma_0)
                self.pars_fid['mg_musigma_c1'] = float(mg.c1_mg)
                self.pars_fid['mg_musigma_c2'] = float(mg.c2_mg)
                self.pars_fid['mg_musigma_lambda0'] = float(mg.lambda_mg)
            elif isinstance(mg, dict):
                musigma = mg.get('mu_Sigma', None)
                if musigma is not None:
                    self.pars_fid['mg_musigma_mu'] = float(musigma.get('mu_0', 0.0))
                    self.pars_fid['mg_musigma_sigma'] = float(musigma.get('sigma_0', 0.0))
                    self.pars_fid['mg_musigma_c1'] = float(musigma.get('c1_mg', 1.0))
                    self.pars_fid['mg_musigma_c2'] = float(musigma.get('c2_mg', 1.0))
                    self.pars_fid['mg_musigma_lambda0'] = float(musigma.get('lambda_mg', 0.0))
        if 'baryonic_effects' in self.pars_fid.keys():
            warnings.warn("Baryonic effects parameters specified but not currently implemented. Ignoring these parameters.")
            self.pars_fid.pop('baryonic_effects')

        self.cf = tools.ccl_factory

        # Load the relevant section of the configuration file
        self.config = config['fisher']

        # Initialize pivot point
        self.x = []
        self.var_pars = None
        self.derivatives = None
        self.Fij = None
        self.Fij_with_gprior = None
        self.Fij_df = None
        self.Fij_with_gprior_df = None
        self.biased_cls, self.bi = None, None
        self.par_bounds = []
        self.norm = None
        self.gpriors = []
        self.gprior_pars = None
        self.transform_Omega_m, self.Om = False, None
        self.transform_S8, self.S8 = False, None
        self.J = None
        self.derivative_method = None
        self.derivative_args = {}
        self.step_size = None
        if 'transform_S8' in self.config.keys():
            if type(self.config['transform_S8']) is not bool:
                warnings.warn('transform_S8 not a boolean, therefore \
                                not transforming Fisher to S8')
            else:
                self.transform_S8 = self.config['transform_S8']
        if 'transform_Omega_m' in self.config.keys():
            if type(self.config['transform_Omega_m']) is not bool:
                warnings.warn('transform_Omega_m not a boolean, therefore \
                                not transforming Fisher to Omega_m')
            else:
                self.transform_Omega_m = self.config['transform_Omega_m']
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
        self.x = np.array(self.x).astype(np.float64)
        self.par_bounds = np.array(self.par_bounds)
        if (len(self.par_bounds) < 1) & (self.norm_step):
            self.norm_step = False
            warnings.warn('Parameter bounds not provided -- the step will not be normalized')
        # Normalize the pivot point given the sampling region
        if self.norm_step:
            self.norm = np.array(self.par_bounds[:, 1]).astype(np.float64) - \
                np.array(self.par_bounds[:, 0]).astype(np.float64)

        # reads in associated gaussian prior width of parameters
        if 'gaussian_priors' in self.config.keys():
            self.gprior_pars = list(self.config['gaussian_priors'].keys())
            for var in self.gprior_pars:
                _val = self.config['gaussian_priors'][var]
                self.gpriors.append(_val)
        # derivative method
        self.derivative_method = self.config.get('derivative_method', '5pt_stencil')
        self.derivative_args = self.config.get('derivative_args', {})
        # step size
        self.step_size = float(self.config.get('step', 0.01))

    def get_Om(self):
        """
        Function that calculates the fiducial value of Omega_m from the input fiducial cosmology.

        Returns:
        --------
        Om : float
             Fiducial value of Om evaluated about pars_fid.
        """
        if self.Om is None:
            Om = 0.0
            if 'Omega_c' not in self.var_pars:
                raise ValueError('Require Omega_c to be specified \
                                 when transforming the Fisher matrix to Omega_m')
            if 'Omega_c' in self.pars_fid.keys():
                Om += self.pars_fid['Omega_c']
            if 'Omega_b' in self.pars_fid.keys():
                Om += self.pars_fid['Omega_b']
            if 'm_nu' in self.pars_fid.keys():
                m_nu = self.pars_fid['m_nu']
                if m_nu > 0.0:
                    if 'h' not in self.pars_fid.keys():
                        raise ValueError('Require h to be specified \
                                         when transforming the Fisher matrix to Omega_m with m_nu')
                    h = self.pars_fid['h']
                    Om += m_nu/h/h/mnu_norm
            self.Om = Om
        return self.Om

    def get_S8(self):
        """
        Function that calculates the fiducial value of S8 from the input fiducial cosmology.

        Returns:
        --------
        S8 : float
             Fiducial value of S8 evaluated about pars_fid.
        """
        if self.S8 is None:
            S8 = 0.0
            Om = self.get_Om()
            if 'sigma8' in self.pars_fid.keys():
                sigma_8 = self.pars_fid['sigma8']
                S8 = np.sqrt(Om/0.3) * sigma_8
            else:
                raise ValueError('Require sigma8 to be specified \
                                 when transforming the Fisher matrix to S8')
            self.S8 = S8
        return self.S8

    def Jacobian_transform(self):
        """
        Function that returns the Jacobian to transform the basis from:
        - Omega_c -> Omega_m
        - sigma8 -> S8
        and will replace its value in the Fisher matrix.
        If neither transform is specified, this function will return the Identity matrix
        to keep all subsequent operations consistent.

        Returns:
        --------
        J : np.ndarray
            Jacobian transformation matrix evaluated about pars_fid.
        """
        if self.J is None:
            J = np.identity(len(self.x))
            if self.transform_Omega_m:
                Om = self.get_Om()
                ind_c = None
                if 'Omega_c' in self.var_pars:
                    ind_c = np.where(np.array(self.var_pars) == 'Omega_c')[0][0]
                if ind_c is None:
                    raise ValueError('Require Omega_c to be specified \
                                     when transforming the Fisher matrix to Omega_m')
                J[ind_c][ind_c] = 1.0

                if 'Omega_b' in self.var_pars:
                    ind_b = np.where(np.array(self.var_pars) == 'Omega_b')[0][0]
                    J[ind_c][ind_b] = -1.0

                if 'm_nu' in self.var_pars:
                    mnu = self.pars_fid['m_nu']
                    h = self.pars_fid['h']
                    ind_nu = np.where(np.array(self.var_pars) == 'm_nu')[0][0]
                    J[ind_c][ind_nu] = -1.0/(h*h*mnu_norm)
                    if 'h' in self.var_pars:
                        ind_h = np.where(np.array(self.var_pars) == 'h')[0][0]
                        J[ind_c][ind_h] = 2.0 * mnu / (h*h*h*mnu_norm)

                print('Replaced Omega_c with Omega_m in Jacobian')

            if self.transform_S8:
                Om = self.get_Om()
                ind_sigma8 = None
                if 'sigma8' in self.var_pars:
                    ind_sigma8 = np.where(np.array(self.var_pars) == 'sigma8')[0][0]
                if ind_sigma8 is None:
                    raise ValueError('Require sigma8 to be specified \
                                     when transforming the Fisher matrix to S8')
                J[ind_sigma8][ind_sigma8] = 1.0/(np.sqrt(Om/0.3))
                sigma_8 = self.pars_fid['sigma8']

                # have not yet changed to Omega_m basis in dictionaries,
                # consistent in either scenario
                if 'Omega_c' in self.var_pars:
                    ind_c = np.where(np.array(self.var_pars) == 'Omega_c')[0][0]
                    J[ind_sigma8][ind_c] = -0.5 * sigma_8 / Om

                if not self.transform_Omega_m:
                    if 'Omega_b' in self.var_pars:
                        ind_b = np.where(np.array(self.var_pars) == 'Omega_b')[0][0]
                        J[ind_sigma8][ind_b] = -0.5 * sigma_8 / Om
                    if 'm_nu' in self.var_pars:
                        mnu = self.pars_fid['m_nu']
                        h = self.pars_fid['h']
                        ind_nu = np.where(np.array(self.var_pars) == 'm_nu')[0][0]
                        J[ind_sigma8][ind_nu] = -0.5 * sigma_8 / Om * (1.0/(h*h*mnu_norm))
                        if 'h' in self.var_pars:
                            ind_h = np.where(np.array(self.var_pars) == 'h')[0][0]
                            J[ind_sigma8][ind_h] = sigma_8 * mnu / (Om * h**3 * mnu_norm)
                print("Replaced sigma8 with S8 in Jacobian")
            self.J = J

        return self.J

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
                x = np.array(x).astype(np.float64)
            # If we normalize the sampling we need to undo the normalization
            if donorm:
                x = self.norm * x + np.array(self.par_bounds[:, 0]).astype(np.float64)

            if x.ndim == 1:
                _pars = pars_fid.copy()
                _sys_pars = sys_fid.copy()
                for i in range(len(labels)):
                    if labels[i] in pars_fid.keys():
                        _pars.update({labels[i]: x[i]})
                    elif labels[i] in sys_fid.keys():
                        _sys_pars[labels[i]] = x[i]
                    elif 'extra_parameters' in pars_fid.keys():
                        if 'camb' in pars_fid['extra_parameters'].keys():
                            if labels[i] in pars_fid['extra_parameters']['camb'].keys():
                                _pars['extra_parameters']['camb'].update({labels[i]: x[i]})
                                _sys_pars[labels[i]] = x[i]
                    else:
                        raise ValueError(f'Parameter name {labels[i]} not recognized!')

                f_out = self.compute_new_theory_vector(_sys_pars, _pars)

            elif x.ndim == 2:
                f_out = []
                for i in range(len(labels)):
                    _pars = pars_fid.copy()
                    # sys_fid is a ParamsMap object
                    _sys_pars = sys_fid.copy()
                    xi = x[i]
                    for j in range(len(labels)):
                        if labels[j] in pars_fid.keys():
                            _pars.update({labels[j]: xi[j]})
                        elif labels[j] in sys_fid.keys():
                            _sys_pars[labels[j]] = xi[j]
                        else:
                            raise ValueError(f'Parameter name {labels[j]} not recognized')
                    f_out.append(self.compute_new_theory_vector(_sys_pars, _pars))
            return np.array(f_out)

    def get_derivatives(self, force=False, method=None, step=None, **kwargs):
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
            step = self.step_size
        if method is None:
            method = self.derivative_method
        # Compute the derivatives with respect to the parameters in var_pars at x
        if (self.derivatives is None) or (force):
            if self.norm_step and 'derivkit' not in method:
                x_here = (self.x - np.array(self.par_bounds[:, 0]).astype(np.float64)) \
                    * 1/self.norm
            elif self.norm_step and 'derivkit' in method:
                warnings.warn('Using derivkit with norm_step=True not recommended.\
                               Forcing norm_step to False and continuing computation.')
                self.norm_step = False
                x_here = self.x
            else:
                x_here = self.x

            if '5pt_stencil' in method:
                self.derivatives = five_pt_stencil(lambda y: self.f(y, self.var_pars, self.pars_fid,
                                                   self.req_params, donorm=self.norm_step),
                                                   x_here, h=step)
            elif 'numdifftools' in method:
                import numdifftools as nd
                if kwargs != {}:
                    print('Overwriting config-specified numdifftools kwargs')
                    kwargs = kwargs
                else:
                    kwargs = self.derivative_args

                jacobian_calc = nd.Jacobian(lambda y: self.f(y, self.var_pars, self.pars_fid,
                                                             self.req_params,
                                                             donorm=self.norm_step),
                                            step=step,
                                            **kwargs)
                self.derivatives = jacobian_calc(x_here).T
            elif 'derivkit' in method:
                from derivkit.calculus_kit import CalculusKit
                if kwargs != {}:
                    print('Overwriting config-specified derivkit kwargs')
                    kwargs = kwargs
                elif self.derivative_args != {}:
                    kwargs = self.derivative_args
                else:
                    print('Using default Augur derivkit kwargs')
                    kwargs = {'method': 'adaptive',
                              'n_workers': 1,
                              'n_points': 27,
                              'spacing': '1%',
                              'base_abs': 1.e-3,
                              'ridge': 1.e-8
                              }
                    method_here = 'adaptive'

                method_here = kwargs.pop('method', 'adaptive')
                n_workers = kwargs.pop('n_workers', 1)
                calc_kit = CalculusKit(function=lambda y: self.f(y, self.var_pars,
                                       self.pars_fid,
                                       self.req_params,
                                       donorm=self.norm_step),
                                       x0=x_here)
                self.derivatives = calc_kit.jacobian(method=method_here,
                                                     n_workers=n_workers,
                                                     **kwargs).T
            else:
                raise ValueError(f'Selected method: `{method}` is not available. \
                                 Please select 5pt_stencil, numdifftools, or derivkit.')
            if (self.norm is not None) and (self.norm_step):
                self.derivatives /= self.norm[:, None]
            return self.derivatives
        else:
            return self.derivatives

    def add_gaussian_priors(self, save_txt=True):
        """
        Auxiliary function to add user-specified Gaussian priors to parameters.
        Called immediately after calculating the Fisher matrix if gaussian_priors
        header is specified.

        Parameters:
        -----------
        save_txt : bool
            Save files of the Fisher + Gaussian prior matrix and Gaussian prior-only
        """

        # 1) transformed parameters are specified (Omega_m or S8)
        # if they are, we check and make sure Omega_c/sigma8 priors not speficied
        # then we J transform, and apply 1/sigma^2 to the digaonal
        # 2) otherwise, just apply the J to the prior-only matrix and sum to Fij_with_prior
        if self.Fij_with_gprior is None:

            indices = []
            ind_sigma8 = None
            ind_c = None
            ind_m = None
            ind_S8 = None
            # TODO: check logic here to make sure transformed parameters handled correctly
            for gvar in self.gprior_pars:
                if gvar in self.var_pars:
                    indices.append(np.where(np.array(self.var_pars) == gvar)[0][0])
                elif gvar == 'Omega_m' and self.transform_Omega_m:
                    ind_c = np.where(np.array(self.var_pars) == 'Omega_c')[0][0]
                    ind_m = np.where(np.array(self.gprior_pars) == 'Omega_m')[0][0]
                    if 'Omega_c' in self.gprior_pars:
                        raise ValueError('Cannot set priors for both Omega_c and Omega_m')
                elif gvar == 'S8' and self.transform_S8:
                    ind_sigma8 = np.where(np.array(self.var_pars) == 'sigma8')[0][0]
                    ind_S8 = np.where(np.array(self.gprior_pars) == 'S8')[0][0]
                    if 'sigma8' in self.gprior_pars:
                        raise ValueError('Cannot set priors for both sigma8 and S8')
                else:
                    warnings.warn(f'The requested prior `{gvar}` is not recognized. \
                                       Please make sure that it is part of your model.')

            self.Fij_with_gprior = np.copy(self.Fij)
            gprior_only = np.zeros((len(self.Fij), len(self.Fij)))
            for i in range(len(indices)):
                j = indices[i]
                gprior_only[j][j] += 1.0/self.gpriors[i]**2

            J = self.Jacobian_transform()
            gprior_only = J.T @ gprior_only @ J
            if ind_sigma8 is not None and ind_S8 is not None:
                gprior_only[ind_sigma8][ind_sigma8] = 1.0/self.gpriors[ind_S8]**2
            if ind_c is not None and ind_m is not None:
                gprior_only[ind_c][ind_c] = 1.0/self.gpriors[ind_m]**2

            self.Fij_with_gprior += gprior_only

            if save_txt:
                np.savetxt(self.config['output']+".priors_only", gprior_only)
                np.savetxt(self.config['output']+".with_priors", self.Fij_with_gprior)

        # Build a pandas DataFrame indexed by varied parameters (rows and columns)
        try:
            self.Fij_with_gprior_df = pd.DataFrame(self.Fij_with_gprior,
                                                   index=self.var_pars,
                                                   columns=self.var_pars
                                                   )
        except Exception:
            # Fallback in case var_pars is None or sizes mismatch
            self.Fij_with_gprior_df = pd.DataFrame(self.Fij_with_gprior)
        return self.Fij_with_gprior

    def add_external_fisher(self, external_fisher, method=None, save_txt=True):
        if self.Fij is None:
            self.get_fisher_matrix(method=method, save_txt=save_txt)

        # external fisher is a path to an augur-like setup or a pandas datafram
        # if it is a dataframe, then we read it in and add it directly
        # if not, we need a helper to read in the text files into a dataframe object to then sum.
        raise NotImplementedError("External fisher addition not yet implemented.")
        # F_ext, fid_ext = read_fisher_from_file(external_fisher)

    def get_fisher_matrix(self, method=None, save_txt=True, **kwargs):
        # Compute Fisher matrix assuming Gaussian likelihood (around self.x)
        if self.derivatives is None:
            self.get_derivatives(method=method, **kwargs)
        if self.Fij is None:
            self.Fij = np.einsum('il, lm, jm', self.derivatives, self.lk.inv_cov, self.derivatives)

            J = self.Jacobian_transform()
            self.Fij = J.T @ self.Fij @ J

            # Build a pandas DataFrame indexed by varied parameters (rows and columns)
            try:
                self.Fij_df = pd.DataFrame(self.Fij, index=self.var_pars, columns=self.var_pars)
            except Exception:
                # Fallback in case var_pars is None or sizes mismatch
                self.Fij_df = pd.DataFrame(self.Fij)

            save_vals = np.copy(self.x.T)
            save_names = np.copy(self.var_pars)

            if self.transform_S8:
                # swap sigma8 with S8
                ind_sigma8 = np.where(np.array(self.var_pars) == 'sigma8')[0][0]
                save_vals[ind_sigma8] = self.get_S8()
                save_names[ind_sigma8] = 'S8'
            if self.transform_Omega_m:
                # swap Omega_c with Omega_m
                ind_c = np.where(np.array(self.var_pars) == 'Omega_c')[0][0]
                save_vals[ind_c] = self.get_Om()
                save_names[ind_c] = 'Omega_m'

            if save_txt:
                np.savetxt(self.config['output'], self.Fij)
                tab_out = Table(save_vals, names=save_names)
                tab_out.write(self.config['fid_output'], format='ascii', overwrite=True)
                fid = self.f(self.x, self.var_pars, self.pars_fid, self.req_params)
                np.savetxt(self.config['output']+".theory_vector", fid)
                np.savetxt(self.config['output']+".derivatives", self.derivatives)
        if self.gprior_pars is not None:
            print('adding priors')
            self.add_gaussian_priors(save_txt=save_txt)

        return self.Fij

    def get_fisher_bias(self, force=False, method=None, save_txt=True, use_fid=False):
        # Compute Fisher bias following the generalized Amara formalism
        # More details in Bianca's thesis and the note here:
        # https://github.com/LSSTDESC/augur/blob/note_bianca/note/main.tex

        # Allowing to provide externally calculated "systematics"
        # They should have the same ells as the original data-vector
        # and the same length
        import os

        if self.derivatives is None:
            self.get_derivatives(force=force, method=method)

        if self.Fij is None:
            self.get_fisher_matrix(method=method, save_txt=save_txt)

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
                    if use_fid:
                        self.biased_cls = biased_cls['dv_sys'] - self.data_fid
                    else:
                        self.biased_cls = biased_cls['dv_sys'] - self.f(self.x,
                                                                        self.var_pars,
                                                                        self.pars_fid,
                                                                        self.req_params,
                                                                        donorm=False)
            # If there's no biased data vector, calculate it
            if _calculate_biased_cls:
                _x_here = []
                _labels_here = []
                if self.transform_S8 or self.transform_Omega_m:
                    raise ValueError("Fisher Biasing involving derived parameters is ill-defined.")
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
                if use_fid:
                    self.biased_cls = self.f(_x_here, _labels_here, _pars_here, _sys_here,
                                             donorm=False) - self.data_fid
                else:
                    self.biased_cls = self.f(_x_here, _labels_here, _pars_here, _sys_here,
                                             donorm=False) - self.f(self.x, self.var_pars,
                                                                    self.pars_fid,
                                                                    self.req_params,
                                                                    donorm=False)
            Bj = np.einsum('l, lm, jm', self.biased_cls, self.lk.inv_cov, self.derivatives)
            bi = np.einsum('ij, j', np.linalg.inv(self.Fij), Bj)
            self.bi = bi

            J = self.Jacobian_transform()
            self.bi = J.T @ self.bi
            save_names = np.copy(self.var_pars)

            if self.transform_S8:
                # swap sigma8 with S8
                ind_sigma8 = np.where(np.array(self.var_pars) == 'sigma8')[0][0]
                save_names[ind_sigma8] = 'S8'
            if self.transform_Omega_m:
                # swap Omega_c with Omega_m
                ind_c = np.where(np.array(self.var_pars) == 'Omega_c')[0][0]
                save_names[ind_c] = 'Omega_m'
            if save_txt:
                tab_out = Table(self.bi, names=save_names)
                tab_out.write(self.config['fid_output']+".biased_params",
                              format='ascii', overwrite=True)
                np.savetxt(self.config['output']+".theory_vector_biased", self.biased_cls)
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
        f_out = compute_new_theory_vector(self.lk, self.tools, _sys_pars, _pars)

        return f_out
