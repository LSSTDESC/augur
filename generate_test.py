from augur.generate import generate
from augur.analyze import Analyze

import numpy as np
lk = generate('./examples/srd_y1_3x2.yml', return_all_outputs=False)
#ao = Analyze('./examples/srd_y10_3x2.yml', lk)
#fisher = ao.get_fisher_matrix(method='5pt_stencil')

#print(ao.Fij)
#lk = generate('./examples/config_test.yml', return_all_outputs=False)
