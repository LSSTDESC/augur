from augur.generate import generate
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory


def build_likelihood(_):
    like, _, _, _ = generate('./config_test.yml', return_all_outputs=True)
    tools = ModelingTools(ccl_factory=CCLFactory(require_nonlinear_pk=True))
    return like, tools
