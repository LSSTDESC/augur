from augur.generate import generate


def build_likelihood(_):
    like, S, tools, sys_params = generate('./config_test.yml', return_all_outputs=True)
    return like
