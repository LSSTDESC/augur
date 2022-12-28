"""Data Generation module

This code generates synthetic dataset by cobbling
together a suitable configuration file for firecrown
and then convincing it to generate data.

"""

import numpy as np
import pyccl as ccl
import sacc
import firecrown

def setup_WLSource(sacc_tracer, ellipticity_error, number_density, Nz_type, systematics=None, **Nz_kwargs):
    dict_out = dict()
    dict_out['kind'] = 'WLSource'
    dict_out['sacc_tracer'] = sacc_tracer
    dict_out['Nz_type'] = Nz_type
    dict_out['ellipticity_error'] = ellipticity_error
    dict_out['number_density'] = number_density
    for key in Nz_kwargs.keys():
        dict_out[key] = Nz_kwargs[key]
    dict_out['mult_bias'] = 0.
    dict_out['ia_bias'] = 0.
    dict_out['alphaz'] = 0.
    dict_out['alphag'] = 0.
    dict_out['z_piv'] = 0.
    return dict_out

def setup_NumberCountSource(sacc_tracer, number_density, bias, Nz_type, **Nz_kwargs):
    dict_out = dict()
    dict_out['kind'] = 'NumberCountsSource'
    dict_out['sacc_tracer'] = sacc_tracer
    dict_out['Nz_type'] = Nz_type
    dict_out['number_density'] = number_density
    dict_out['bias'] = bias
    for key in Nz_kwargs.keys():
        dict_out[key] = Nz_kwargs[key]
    return dict_out

cosmo = ccl.Cosmology(Omega_b=config['Omega_b'], 
                      Omega_c=config['Omega_c'], 
                      n_s=config['n_s'], sigma8=config['sigma8'],
                      h=config['h0'])

# ndens = [2.40, 3.82, 4.29, 4.05, 3.44]
ndens = config['ndens']  # List with the number density (in units of arcmin^-2)

S = sacc.Sacc()  # Placeholder sacc file



