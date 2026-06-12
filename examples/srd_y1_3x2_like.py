from augur.generate import generate
import sacc

def build_likelihood(_):
    # S = sacc.Sacc.load_fits('./test_sacc.sacc')
    like, tools, sys_params = generate('./config_test.yml', return_all_outputs=True)#, use_sacc = S, sacc_path='./test_sacc.sacc')
    like.reset()
    tools.reset()
    return like