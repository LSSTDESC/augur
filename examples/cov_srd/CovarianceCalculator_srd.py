from tjpcov.covariance_calculator import CovarianceCalculator

# In order to use the config file in tests, you need your working directory to be the root folder
import os
os.chdir('../')

config_yml = "/nethome/chisa002/TJPCov/tests/data/srd/conf_covariance_gaussian_fsky_fourier_srd.yaml"
cc = CovarianceCalculator(config_yml)
cov = cc.get_covariance()
s = cc.create_sacc_cov(output="srd_3x2pt_Y1_covariance.fits")

exit()
