from augur.generate import generate
from augur.analyze import Analyze
from augur.postprocess import postprocess


cf = './augur/tests/test.yaml'
lk, _, tools, req_params = generate(cf, return_all_outputs=True)
print(req_params.items())
ao = Analyze(cf, lk, tools=tools, req_params=req_params, norm_step=False)
fisher = ao.get_fisher_matrix(method='5pt_stencil')

# postprocess(cf)
