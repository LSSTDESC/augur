from augur.generate import generate


def build_likelihood(_):
    like, tools, sys_params = generate('./config_test.yml', return_all_outputs=True)
    like.reset()
    tools.reset()
    return like
