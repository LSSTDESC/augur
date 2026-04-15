#!/usr/bin/env python3
"""Run joint 3x2pt + CMB lensing Fourier forecast with TJPCov covariance."""

from augur.generate import generate
from augur.analyze import Analyze
import sacc

def main():
    config = 'examples/srd_y1_3x2pt_cmb_lensing_tjpcov.yml'
    sacc_path = './output/srd_y1_3x2pt_cmb_lensing_tjpcov.sacc'
    S = sacc.Sacc.load_fits(sacc_path)

    # Build fiducial SACC + likelihood + TJPCov covariance.
    like, S, tools, sys_params = generate(
        config,
        write_sacc=False,
        use_sacc=S,
        sacc_path=sacc_path,
        return_all_outputs=True,
    )

    # Run Fisher forecast and write outputs defined in the YAML.
    ao = Analyze(config, req_params=sys_params, tools=tools, likelihood=like)
    fisher = ao.get_fisher_matrix()
    print('Fisher matrix shape:', fisher.shape)


if __name__ == '__main__':
    main()
