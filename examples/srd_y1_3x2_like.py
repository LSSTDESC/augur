from augur.generate import generate


def build_likelihood(_):
    like, S, tools, sys_params, ccl_factory = generate('./config_test.yml', return_all_outputs=True)
    like.reset()
    tools.reset()
    return like, tools
