#!/usr/bin/env python3
"""Run CMB lensing-only Fourier forecast with TJPCov covariance."""

from augur.generate import generate
from augur.analyze import Analyze


def main():
    config = 'examples/srd_y1_cmb_lensing_tjpcov.yml'

    # Build fiducial SACC + likelihood + TJPCov covariance.
    like, S, tools, sys_params  = generate(config, write_sacc=True)

    # Run Fisher forecast and write outputs defined in the YAML.
    ao = Analyze(config, sys_params=sys_params, tools=tools, likelihood=like)
    fisher = ao.get_fisher_matrix()
    print('Fisher matrix shape:', fisher.shape)


if __name__ == '__main__':
    main()
