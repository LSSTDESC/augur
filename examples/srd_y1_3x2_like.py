from augur.generate import generate


def build_likelihood(_):
    like, S, tools = generate('./config_test.yml', return_all_outputs=True, force_read=True)
    return like
