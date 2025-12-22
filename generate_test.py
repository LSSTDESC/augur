from augur.generate import generate
from augur.analyze import Analyze
from augur.postprocess import postprocess


import numpy as np
# lk = generate('./examples/srd_y10_3x2.yml', return_all_outputs=False)
# lk = generate('./examples/srd_y1_3x2.yml', return_all_outputs=False)

# ao = Analyze('./examples/srd_y10_3x2.yml', lk)
# fisher = ao.get_fisher_matrix(method='5pt_stencil')

# print(ao.Fij)
# lk = generate('./examples/config_test.yml', return_all_outputs=False)

cf = './examples/srd_y1_3x2.yml'
lk, _,tools, req_params = generate(cf, return_all_outputs=True)
print(req_params.items())
ao = Analyze(cf, lk, tools=tools, req_params=req_params, norm_step=False)
fisher = ao.get_fisher_matrix(method='5pt_stencil')

postprocess(cf)